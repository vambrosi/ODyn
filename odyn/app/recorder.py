"""
The behavior sync recorder: the DAQ acquisition the rig used to run from a
separate script, driven from the app so the file lands where the database
expects it, named for the experiment it belongs to.

`Recording` is the engine and has no Qt in it: it owns the H5 file (none
when `path` is `None`: a test that only shows the traces) and the DAQ task
(or a simulated source, for machines without one) and is stepped by whoever
calls `tick()`. `RecorderPane` is the widget around it.

The file layout is the one `odyn.sync` reads, unchanged from the rig script:
root attributes `mouse`, `experiment`, `start_time`, `rate_hz`, and at Stop
`stop_time`, `n_samples`, `duration_s`; datasets `encoder` (N, 3),
`respiration` (N, 2), `2pFrameSync`, `cameraFrameSync`, `odorPulse` (N,) and
`filter_changes` (M, 3). What the app adds are attributes naming the writer
version and, when known, the session and experiment folder.

USB-6009 channel map (single ended, as built):
    ai0  T2   CamStrobe    Mako Out3 exposure strobe
    ai1  T5   EncoderA     encoder phase 1
    ai2  T8   EncoderB     encoder phase 2
    ai4  T3   FrameClock   2P frame clock (high for frame, low for flyback)
    ai6  T9   Resp         AD8495 gain-stage output (thermocouple)
    ai7  T12  OdorPulse    olfactometer, high for the odor delivery
"""

from __future__ import annotations

import time

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np

from ..utils import logger

WRITER_VERSION = "odyn.app.recorder 1"

DEVICE = "Dev1"
PHYS_CHANS = "Dev1/ai0,Dev1/ai1,Dev1/ai2,Dev1/ai4,Dev1/ai6,Dev1/ai7"
RAW_NAMES = ["CamStrobe", "EncoderA", "EncoderB", "FrameClock", "Resp", "OdorPulse"]
TERMINALS = ["T2", "T5", "T8", "T3", "T9", "T12"]
RATE = 5000  # samples/s per channel

# indices into the 6 raw channels
CAM, ENC_A, ENC_B, FRAME, RESP, ODOR = 0, 1, 2, 3, 4, 5

# encoder -> velocity
WHEEL_DIAM_CM = 15.24  # 6 inch wheel
PULSES_PER_REV = 600
DECODE_X = 1  # 1x decode (count A rising edges)
ENCODER_SIGN = +1  # flip to -1 if forward running reads negative
ENC_THRESH = 2.5  # V, midpoint for 0-5 V encoder with pull-ups
VEL_SMOOTH_S = 0.1  # s, velocity smoothing window
DIST_PER_COUNT = np.pi * WHEEL_DIAM_CM / (PULSES_PER_REV * DECODE_X)  # cm/count

# respiration filter defaults (tunable in the GUI)
RESP_HP_DEFAULT = 0.5  # Hz, 0 disables the high-pass
RESP_LP_DEFAULT = 15.0  # Hz, 0 disables the low-pass
RESP_ORDER = 2


def build_resp_sos(hp: float, lp: float, order: int = RESP_ORDER, fs: float = RATE):
    """Butterworth bandpass / highpass / lowpass from cutoffs in Hz. None = pass-through."""
    from scipy.signal import butter

    nyq = fs / 2.0
    hp = max(0.0, hp)
    lp = max(0.0, lp)

    if hp > 0 and lp > 0:
        if hp >= lp:
            raise ValueError("HP must be below LP")
        return butter(order, [hp / nyq, lp / nyq], btype="band", output="sos")

    if lp > 0:
        return butter(order, lp / nyq, btype="low", output="sos")

    if hp > 0:
        return butter(order, hp / nyq, btype="high", output="sos")

    return None


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


class Source(Protocol):
    """Where samples come from: the DAQ, or a stand-in."""

    def start(self) -> None: ...

    def read(self) -> np.ndarray:
        """Every sample since the last read, as `(k, 6)` float32."""
        ...

    def stop(self) -> None: ...


class DaqSource:
    """The National Instruments board, through `nidaqmx`."""

    def __init__(self):
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration

        self._nidaqmx = nidaqmx
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            PHYS_CHANS,
            terminal_config=TerminalConfiguration.RSE,
            min_val=-10.0,
            max_val=10.0,
        )
        self.task.timing.cfg_samp_clk_timing(
            RATE, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=2 * RATE
        )

    def start(self) -> None:
        self.task.start()

    def read(self) -> np.ndarray:
        from nidaqmx.constants import READ_ALL_AVAILABLE

        data = self.task.read(number_of_samples_per_channel=READ_ALL_AVAILABLE)
        arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[:, None]

        return arr.T

    def stop(self) -> None:
        try:
            self.task.stop()
            self.task.close()
        except Exception:
            pass


class SimulatedSource:
    """
    A stand-in for the board: a 19 Hz frame clock in 20 s bursts, a 4 s odor
    pulse 5 s into each, a camera strobe, a breathing-like respiration trace
    and a slowly turning wheel. For trying the app without the rig.
    """

    def __init__(self, rate: int = RATE):
        self.rate = rate
        self._started = 0.0
        self._sent = 0

    def start(self) -> None:
        self._started = time.monotonic()
        self._sent = 0

    def read(self) -> np.ndarray:
        elapsed = time.monotonic() - self._started
        total = int(elapsed * self.rate)
        k = total - self._sent

        if k <= 0:
            return np.empty((0, 6), np.float32)

        t = (self._sent + np.arange(k)) / self.rate
        self._sent = total

        cycle = t % 30.0  # a 20 s acquisition every 30 s
        frame = ((cycle < 20.0) & ((t * 19.0) % 1.0 < 0.5)) * 5.0
        odor = ((cycle >= 5.0) & (cycle < 9.0)) * 5.0
        camera = ((t * 19.0) % 1.0 < 0.5) * 4.6
        resp = 2.0 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
        counts = (t * 20.0) % 1.0  # 20 encoder counts a second
        enc_a = (counts < 0.5) * 5.0
        enc_b = (((counts + 0.25) % 1.0) < 0.5) * 5.0

        return np.column_stack([camera, enc_a, enc_b, frame, resp, odor]).astype(
            np.float32
        )

    def stop(self) -> None:
        pass


def available_source() -> Source:
    """The board if `nidaqmx` and a device are there, otherwise the stand-in."""
    try:
        return DaqSource()

    except Exception as error:  # no nidaqmx, or no device
        logger.warning(f"No DAQ available ({error}); the recorder is simulated.")

        return SimulatedSource()


# --------------------------------------------------------------------------- #
# The recording
# --------------------------------------------------------------------------- #


class Recording:
    """
    One sync file being written.

    ```python
    recording = Recording(path, mouse="m442", experiment="e1", source=SimulatedSource())
    recording.start()
    ...
    chunk = recording.tick()      # call it every ~50 ms
    ...
    recording.stop()
    ```

    `tick` reads what the source has, derives velocity and filtered
    respiration, appends everything to the file and returns the chunk for
    display as a dict of arrays.
    """

    def __init__(
        self,
        path: None | str | Path,
        *,
        mouse: str,
        experiment: str,
        source: Source,
        hp_hz: float = RESP_HP_DEFAULT,
        lp_hz: float = RESP_LP_DEFAULT,
        extra_attributes: None | dict[str, Any] = None,
    ):
        self.path = None if path is None else Path(path)
        self.mouse = mouse
        self.experiment = experiment
        self.source = source
        self.extra_attributes = extra_attributes or {}

        self.hp_hz, self.lp_hz = hp_hz, lp_hz
        self.sos = build_resp_sos(hp_hz, lp_hz)
        self.zi = None

        self.h5: None | h5py.File = None
        self.total = 0
        self.filt_log: list[tuple[int, float, float]] = []
        self.recording = False

        self.W = max(1, int(VEL_SMOOTH_S * RATE))
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self.last_A = False
        self.last_pos = 0.0
        self.pos_tail = np.zeros(self.W, np.float64)

    def start(self) -> None:
        """Open the file (none in a test: the traces show, nothing is kept)."""
        self.filt_log = [(0, self.hp_hz, self.lp_hz)]
        self.total = 0
        self.zi = None
        self._reset_stream_state()

        if self.path is None:
            self.source.start()
            self.recording = True
            return

        start_dt = datetime.now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.path, "w")

        fa = self.h5.attrs
        fa["mouse"] = self.mouse
        fa["experiment"] = self.experiment
        fa["start_time"] = start_dt.isoformat()
        fa["rate_hz"] = RATE
        fa["writer_version"] = WRITER_VERSION

        for key, value in self.extra_attributes.items():
            fa[key] = value

        self.enc_dset = self.h5.create_dataset(
            "encoder",
            shape=(0, 3),
            maxshape=(None, 3),
            chunks=(RATE, 3),
            dtype="float32",
            compression="gzip",
        )
        ea = self.enc_dset.attrs
        ea["column_names"] = ["EncoderA", "EncoderB", "Velocity"]
        ea["units"] = ["V", "V", "cm/s"]
        ea["terminals"] = ["T5", "T8"]
        ea["wheel_diam_cm"] = WHEEL_DIAM_CM
        ea["pulses_per_rev"] = PULSES_PER_REV
        ea["decode_x"] = DECODE_X
        ea["encoder_sign"] = ENCODER_SIGN
        ea["encoder_thresh_v"] = ENC_THRESH
        ea["vel_smooth_s"] = VEL_SMOOTH_S
        ea["dist_per_count_cm"] = DIST_PER_COUNT

        self.resp_dset = self.h5.create_dataset(
            "respiration",
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(RATE, 2),
            dtype="float32",
            compression="gzip",
        )
        rfa = self.resp_dset.attrs
        rfa["column_names"] = ["RespRaw", "RespFiltered"]
        rfa["units"] = ["V", "V"]
        rfa["terminal"] = "T9"
        rfa["resp_filter"] = "butterworth"
        rfa["resp_order"] = RESP_ORDER
        rfa["resp_hp_hz"] = self.hp_hz
        rfa["resp_lp_hz"] = self.lp_hz

        self.frame_dset = self._channel(
            "2pFrameSync", "T3", "2P frame clock; high during frame, low during flyback"
        )
        self.cam_dset = self._channel(
            "cameraFrameSync", "T2", "Mako Out3 exposure strobe"
        )
        self.odor_dset = self._channel(
            "odorPulse", "T12", "Olfactometer pulse; high during odor delivery"
        )

        self.source.start()
        self.recording = True

    def _channel(self, name: str, terminal: str, description: str) -> h5py.Dataset:
        assert self.h5 is not None
        dset = self.h5.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(RATE,),
            dtype="float32",
            compression="gzip",
        )
        dset.attrs["units"] = "V"
        dset.attrs["terminal"] = terminal
        dset.attrs["description"] = description

        return dset

    def apply_filter(self, hp_hz: float, lp_hz: float) -> None:
        """Change the respiration display filter; logged into the file."""
        self.sos = build_resp_sos(hp_hz, lp_hz)
        self.zi = None
        self.hp_hz, self.lp_hz = hp_hz, lp_hz

        if self.recording and self.h5 is not None:
            self.resp_dset.attrs["resp_hp_hz"] = hp_hz
            self.resp_dset.attrs["resp_lp_hz"] = lp_hz
            self.filt_log.append((self.total, hp_hz, lp_hz))

    def tick(self) -> None | dict[str, np.ndarray]:
        """Read, derive, append, and return the chunk for display."""
        if not self.recording:
            return None

        chunk = self.source.read()
        k = chunk.shape[0]

        if k == 0:
            return None

        # velocity: 1x quadrature on A rising edges, direction from B
        A = chunk[:, ENC_A] > ENC_THRESH
        B = chunk[:, ENC_B] > ENC_THRESH
        A_ext = np.concatenate(([self.last_A], A))
        rise = (~A_ext[:-1]) & (A_ext[1:])
        step = np.where(B, -1.0, 1.0) * ENCODER_SIGN * DIST_PER_COUNT
        inc = np.where(rise, step, 0.0)
        pos = self.last_pos + np.cumsum(inc)
        ext = np.concatenate([self.pos_tail, pos])
        vel = ((ext[self.W :] - ext[: -self.W]) / (self.W / RATE)).astype(np.float32)
        self.last_pos = pos[-1]
        self.last_A = bool(A[-1])
        self.pos_tail = ext[-self.W :]

        # respiration filter (causal, stateful)
        resp_raw = chunk[:, RESP]

        if self.sos is None:
            respf = resp_raw.copy()
        else:
            from scipy.signal import sosfilt, sosfilt_zi

            if self.zi is None:
                self.zi = sosfilt_zi(self.sos) * float(resp_raw[0])

            respf, self.zi = sosfilt(self.sos, resp_raw, zi=self.zi)

        respf = respf.astype(np.float32)
        n = self.total

        if self.h5 is not None:
            for dset, columns in (
                (
                    self.enc_dset,
                    np.column_stack([chunk[:, ENC_A], chunk[:, ENC_B], vel]),
                ),
                (self.resp_dset, np.column_stack([resp_raw, respf])),
                (self.frame_dset, chunk[:, FRAME]),
                (self.cam_dset, chunk[:, CAM]),
                (self.odor_dset, chunk[:, ODOR]),
            ):
                dset.resize(n + k, axis=0)
                dset[n : n + k] = columns

            self.h5.flush()

        self.total += k

        return {
            "t": (n + np.arange(k)) / RATE,
            "vel": vel,
            "respf": respf,
            "odor": chunk[:, ODOR],
            "frame": chunk[:, FRAME],
            "cam": chunk[:, CAM],
        }

    @property
    def duration_s(self) -> float:
        return self.total / RATE

    def stop(self) -> None:
        if not self.recording:
            return

        self.recording = False
        self.source.stop()

        if self.h5 is None:
            return

        fc = np.array(self.filt_log, dtype=np.float64)
        ds = self.h5.create_dataset("filter_changes", data=fc)
        ds.attrs["column_names"] = ["sample_index", "hp_hz", "lp_hz"]
        ds.attrs["description"] = (
            "One row per filter event. Row 0 is the initial filter at "
            "recording start; subsequent rows record the sample index when "
            "'Apply filter' was clicked and the new HP/LP cutoffs. "
            "RespFiltered reflects whichever filter was active at each sample."
        )
        self.h5.attrs["n_samples"] = self.total
        self.h5.attrs["duration_s"] = self.total / RATE
        self.h5.attrs["stop_time"] = datetime.now().isoformat()
        self.h5.flush()
        self.h5.close()
        self.h5 = None


def sync_file_name(mouse: str, experiment: str, started: None | datetime = None) -> str:
    """`mouse_experiment_YYYY-MM-DD_HH-MM-SS.h5`, as the rig always named them."""
    started = started or datetime.now()

    return f"{mouse}_{experiment}_{started:%Y-%m-%d_%H-%M-%S}.h5"


def minmax_decimate(t: np.ndarray, y: np.ndarray, target: int):
    """Thin to ~target points keeping per-bin min and max so edges survive."""
    n = y.size

    if n <= 2 * target:
        return t, y

    idx = np.linspace(0, n, target + 1).astype(int)
    starts = idx[:-1]
    lo = np.minimum.reduceat(y, starts)
    hi = np.maximum.reduceat(y, starts)
    tt = np.repeat(t[starts], 2)
    yy = np.empty(tt.size, dtype=y.dtype)
    yy[0::2] = lo
    yy[1::2] = hi

    return tt, yy
