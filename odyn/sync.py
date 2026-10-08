"""
Reading and decoding the behavior sync H5 file.

The rig's DAQ records, at one sample rate and on one clock, the 2P frame clock,
the olfactometer's odor valve, the camera strobe, the respiration probe and the
wheel encoder. This module turns the frame clock and the valve into
per-acquisition numbers (`SyncBlock`s) without touching the database;
`Database.process_sync` matches them to acquisitions and trials and stores them.

The file layout is the writer's (`record_behavior.py` on the rig): root
attributes `mouse`, `experiment`, `rate_hz`, `start_time`, and after Stop
`stop_time`, `n_samples`, `duration_s`; datasets `2pFrameSync`,
`cameraFrameSync`, `odorPulse`, `respiration` and `encoder`, all N samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from .utils import Object, jsonable

# Logic level of the TTL channels, halfway between 0 and 5 V. The writer uses
# the same value for the encoder.
LOGIC_THRESHOLD_V = 2.5

# Two frame-clock edges closer than this are one frame bouncing, not two
# frames: frames arrive at 12-48 Hz, so 5 ms is far below a real period.
MIN_FRAME_INTERVAL_S = 0.005

# Frames further apart than this belong to different acquisitions. Frames are
# tens of milliseconds apart and acquisitions ten seconds, so any value in
# between works; two seconds also keeps z-stack slices (~2.8 s apart) as
# separate bursts rather than one long one.
BURST_GAP_S = 2.0

# A burst shorter than this fraction of the usual one is a fragment (an
# acquisition that was interrupted, or a stray trigger), not an acquisition.
FRAGMENT_FRACTION = 0.5

FRAME_CHANNEL = "2pFrameSync"
ODOR_CHANNEL = "odorPulse"
CAMERA_CHANNEL = "cameraFrameSync"


@dataclass
class SyncHeader:
    """What the file says about itself, as `streams` stores it."""

    path: Path
    rate_hz: float
    start_time: None | datetime
    stop_time: None | datetime
    n_samples: int
    attributes: Object

    @property
    def interrupted(self) -> bool:
        """
        Whether the recording was stopped by a crash rather than by Stop.

        The writer streams samples as they come and writes `stop_time`,
        `n_samples` and `duration_s` only when Stop is pressed, so a file with
        the data but without them was not closed properly.
        """
        return self.stop_time is None


@dataclass
class SyncBlock:
    """One burst of the frame clock, decoded."""

    sync_block: int  # index of the burst in the file, fragments included
    start_sample: int  # sample of the first frame's onset
    frame_count: int
    frame_period_s: float
    odor_pulses: int  # valve pulses starting inside the burst
    odor_on_frame: None | int = None  # frame during which the valve opened
    odor_off_frame: None | int = None  # first frame with the valve closed
    odor_on_residual_s: None | float = None  # onset minus that frame's onset
    odor_on_sample: None | int = None
    odor_off_sample: None | int = None
    camera_frames: None | int = None  # camera strobes inside the burst


@dataclass
class DecodedSync:
    """Everything `Database.process_sync` needs from one file."""

    header: SyncHeader
    blocks: list[SyncBlock]
    fragments: list[SyncBlock] = field(default_factory=list)

    def time_at(self, sample: int) -> None | datetime:
        """Wall clock of a sample, on the olfactometer computer's clock."""
        if self.header.start_time is None:
            return None

        return self.header.start_time + timedelta(seconds=sample / self.header.rate_hz)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_header(path: str | Path) -> SyncHeader:
    """
    The file's attributes, without loading any channel.

    Every attribute of the file and of its datasets is kept in `attributes`,
    since the writer's constants (channel map, wheel geometry, filter
    settings) are recorded nowhere else.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        attributes: Object = {"file": _attrs(f)}

        for name, dataset in f.items():
            attributes[name] = {"shape": list(dataset.shape), **_attrs(dataset)}

        rate_hz = float(f.attrs["rate_hz"])
        start = f.attrs.get("start_time")
        stop = f.attrs.get("stop_time")

        # `n_samples` is only written at Stop; the datasets know regardless
        n_samples = int(f.attrs.get("n_samples", f[FRAME_CHANNEL].shape[0]))

    return SyncHeader(
        path=path,
        rate_hz=rate_hz,
        start_time=None if start is None else datetime.fromisoformat(str(start)),
        stop_time=None if stop is None else datetime.fromisoformat(str(stop)),
        n_samples=n_samples,
        attributes=attributes,
    )


def read_channel(path: str | Path, name: str) -> np.ndarray:
    """One channel as a 1-D float32 array (the first column of a 2-D one)."""
    with h5py.File(path, "r") as f:
        data = f[name][:]

    return np.asarray(data if data.ndim == 1 else data[:, 0], dtype=np.float32)


def _attrs(node) -> Object:
    return {key: jsonable(_plain(value)) for key, value in node.attrs.items()}


def _plain(value):
    # h5py hands back numpy scalars, arrays and bytes; JSON wants Python
    if isinstance(value, bytes):
        return value.decode()

    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]

    if hasattr(value, "item"):
        return value.item()

    return value


# --------------------------------------------------------------------------- #
# Edges and bursts
# --------------------------------------------------------------------------- #


def rising_edges(
    signal: np.ndarray, threshold: float = LOGIC_THRESHOLD_V
) -> np.ndarray:
    """Sample indices where `signal` goes above `threshold` (the first high sample)."""
    high = signal > threshold

    return np.flatnonzero(~high[:-1] & high[1:]) + 1


def pulses(signal: np.ndarray, threshold: float = LOGIC_THRESHOLD_V) -> np.ndarray:
    """
    Each pulse as `(on_sample, off_sample)`, both the first sample of their level.

    A pulse already high when the file starts, or still high when it ends, has
    only one edge and is left out rather than paired with the wrong one.
    """
    high = signal > threshold
    change = np.diff(high.astype(np.int8))

    ons = np.flatnonzero(change == 1) + 1
    offs = np.flatnonzero(change == -1) + 1

    if offs.size and ons.size and offs[0] < ons[0]:
        offs = offs[1:]

    n = min(ons.size, offs.size)

    return np.column_stack([ons[:n], offs[:n]])


def frame_onsets(
    frame_signal: np.ndarray,
    rate_hz: float,
    min_interval_s: float = MIN_FRAME_INTERVAL_S,
) -> np.ndarray:
    """Sample index of every frame's onset, with bounces removed."""
    edges = rising_edges(frame_signal)

    if edges.size == 0:
        return edges

    min_gap = int(min_interval_s * rate_hz)
    keep = [int(edges[0])]

    for edge in edges[1:]:
        if edge - keep[-1] >= min_gap:
            keep.append(int(edge))

    return np.asarray(keep, dtype=np.int64)


def split_bursts(
    onsets: np.ndarray, rate_hz: float, gap_s: float = BURST_GAP_S
) -> list[np.ndarray]:
    """Frame onsets split at every gap longer than `gap_s`."""
    if onsets.size == 0:
        return []

    gaps = np.diff(onsets) / rate_hz
    breaks = np.flatnonzero(gaps > gap_s) + 1

    return [burst for burst in np.split(onsets, breaks)]


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def decode(path: str | Path) -> DecodedSync:
    """
    Decode a sync file into per-burst numbers.

    Reads the frame clock, the valve and the camera strobe once. Bursts much
    shorter than the usual one are set aside as `fragments`; everything else
    is a `SyncBlock` in file order.
    """
    header = read_header(path)
    rate = header.rate_hz

    frames = frame_onsets(read_channel(path, FRAME_CHANNEL), rate)
    valve = pulses(read_channel(path, ODOR_CHANNEL))
    camera = rising_edges(read_channel(path, CAMERA_CHANNEL))

    bursts = split_bursts(frames, rate)

    if not bursts:
        return DecodedSync(header, [])

    # The usual burst is the acquisition length; one acquisition per burst is
    # the norm, so the mode is the right reference even when a few are short.
    sizes = np.array([b.size for b in bursts])
    usual = int(np.bincount(sizes).argmax())
    threshold = FRAGMENT_FRACTION * usual

    blocks: list[SyncBlock] = []
    fragments: list[SyncBlock] = []

    for index, burst in enumerate(bursts):
        block = _decode_burst(index, burst, rate, valve, camera)

        if burst.size < threshold:
            fragments.append(block)
        else:
            blocks.append(block)

    return DecodedSync(header, blocks, fragments)


def _decode_burst(
    index: int,
    burst: np.ndarray,
    rate: float,
    valve: np.ndarray,
    camera: np.ndarray,
) -> SyncBlock:
    period = float(np.median(np.diff(burst))) / rate if burst.size > 1 else 0.0

    # The burst spans its frames plus the last frame's own duration
    start = int(burst[0])
    end = int(burst[-1]) + max(int(round(period * rate)), 1)

    block = SyncBlock(
        sync_block=index,
        start_sample=start,
        frame_count=int(burst.size),
        frame_period_s=period,
        odor_pulses=0,
        camera_frames=int(np.count_nonzero((camera >= start) & (camera < end))),
    )

    inside = (
        valve[(valve[:, 0] >= start) & (valve[:, 0] < end)] if valve.size else valve
    )
    block.odor_pulses = int(len(inside))

    if block.odor_pulses == 0:
        return block

    # The first pulse is the one; any extra is reported through `odor_pulses`
    on, off = (int(v) for v in inside[0])

    # The frame during which the valve opened: the last onset at or before it.
    # `searchsorted(..., 'right') - 1` is that frame, and cannot be -1 since
    # the pulse starts inside the burst.
    on_frame = int(np.searchsorted(burst, on, side="right") - 1)
    block.odor_on_frame = on_frame
    block.odor_on_residual_s = float(on - burst[on_frame]) / rate
    block.odor_on_sample = on
    block.odor_off_sample = off

    # The first frame that began with the valve already closed. A valve still
    # open when the burst ends gets `frame_count`, one past the last frame.
    block.odor_off_frame = int(np.searchsorted(burst, off, side="left"))

    return block


def summary(decoded: DecodedSync) -> Object:
    """A JSON-ready account of a decode, for `call_output`."""
    blocks = decoded.blocks

    return {
        "file": decoded.header.path.name,
        "rate_hz": decoded.header.rate_hz,
        "interrupted": decoded.header.interrupted,
        "blocks": len(blocks),
        "fragments": [f.frame_count for f in decoded.fragments],
        "frames_per_block": sorted({b.frame_count for b in blocks}),
        "blocks_with_odor": sum(b.odor_pulses > 0 for b in blocks),
        "blocks_without_camera": sum(not b.camera_frames for b in blocks),
    }


__all__ = [
    "DecodedSync",
    "SyncBlock",
    "SyncHeader",
    "decode",
    "read_header",
    "read_channel",
    "summary",
]
