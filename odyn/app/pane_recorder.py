"""
The recorder tab: the live traces and the Start/Stop around a `Recording`.

The file is written on this computer, `~/.odyn/sync/<session folder>/
<experiment>/<mouse>_<experiment>_<started>.h5`, so a share that drops never
stops a recording, and copied at Stop (in the background) to where `submit`
looks for it: `<main folder>/<session folder>/<experiment>/sync/`, with the
experiment's `raw/` and `processed/` folders made beside it so the ScanImage
side has somewhere to save. A copy that fails is retried by Submit (see
`recordings`).

The draft notes every recording (`Draft.add_recording`) and from then on the
session's mouse and date are fixed, since the file is named for them.
Recording an experiment again asks first, and replaces the earlier file.

"Test" runs the source and shows the traces without writing anything: it is
what the rig runs to check the wiring.

The source is the DAQ board. Without one (no `nidaqmx`, no driver, the board
unplugged) nothing starts. The app records from the stand-in only when started
with `--simulate`.
"""

from __future__ import annotations

import threading

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils import logger
from .context import Context
from .draft import Draft
from .recorder import (
    RATE,
    RESP_HP_DEFAULT,
    RESP_LP_DEFAULT,
    DaqSource,
    Recording,
    SimulatedSource,
    Source,
    minmax_decimate,
    sync_file_name,
)
from .recordings import (
    copy_to_share,
    hide_on_share,
    local_path,
    make_experiment_folders,
    share_path,
)

TICK_MS = 50
REDRAW_MS = 200
WINDOW_S = 10.0
POINTS_PER_TRACE = 1500

TEST_MOUSE = "test"

# The file label, wide enough for a session path on one line
TARGET_WIDTH_PX = 640

TRACES = (
    ("vel", "velocity (cm/s)"),
    ("respf", "respiration (V)"),
    ("frame", "frame clock (V)"),
    ("odor", "odor pulse (V)"),
    ("cam", "camera strobe (V)"),
)


def recording_path(
    local_root: Path,
    draft: Draft,
    experiment: str,
    started: None | datetime = None,
) -> tuple[Path, str, str]:
    """
    Where the next file goes `(on this computer, on the share relative to
    the main folder, the mouse name it is stamped with)`.
    """
    mouse = f"m{draft.mouse_id}"
    name = sync_file_name(mouse, experiment, started or datetime.now())

    return (
        local_path(local_root, draft.folder, experiment, name),
        share_path(draft.folder, experiment, name),
        mouse,
    )


class Copier(QObject):
    """
    Copies a finished recording to the share on a thread of its own, so a
    slow share never freezes the window. `done(draft path, experiment, file,
    error)` comes back on the GUI thread, `error` empty when it worked.
    """

    done = Signal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running: set[str] = set()  # files being copied

    def copy(self, draft: Draft, experiment: str, record: dict, main_folder: Path):
        file = str(record["file"])
        self.running.add(file)

        def run():
            try:
                copy_to_share(
                    Path(record["local"]), Path(main_folder) / record["share"]
                )
                error = ""
            except OSError as failure:
                error = str(failure) or type(failure).__name__

            self.done.emit(str(draft.path), experiment, file, error)

        threading.Thread(target=run, name=f"copy {file}", daemon=True).start()


class TraceBuffer:
    """The last `WINDOW_S` seconds of each trace, thinned for drawing."""

    def __init__(self):
        self.chunks: deque[dict[str, np.ndarray]] = deque()
        self.latest = 0.0

    def clear(self) -> None:
        self.chunks.clear()
        self.latest = 0.0

    def add(self, chunk: dict[str, np.ndarray]) -> None:
        self.chunks.append(chunk)
        self.latest = float(chunk["t"][-1])

        while self.chunks and self.chunks[0]["t"][-1] < self.latest - WINDOW_S:
            self.chunks.popleft()

    def trace(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.chunks:
            return np.empty(0), np.empty(0)

        t = np.concatenate([c["t"] for c in self.chunks])
        y = np.concatenate([c[key] for c in self.chunks])

        return minmax_decimate(t, y, POINTS_PER_TRACE)


class RecorderPane(QWidget):
    """The widget: draft-aware naming, filter controls, live plot."""

    def __init__(self, context: Context, source: None | Source = None, parent=None):
        super().__init__(parent)
        self.context = context
        self.source: None | Source = source
        self.recording: None | Recording = None
        self.buffer = TraceBuffer()

        layout = QVBoxLayout(self)

        where = QGroupBox("Where the file goes")
        where_form = QFormLayout(where)
        self.session_label = QLabel("no session selected in the right bar")
        self.experiment = QLabel("")
        self.target = QLabel("")
        self.target.setMinimumWidth(TARGET_WIDTH_PX)
        where_form.addRow("Session", self.session_label)
        where_form.addRow("Experiment", self.experiment)
        where_form.addRow("File", self.target)
        layout.addWidget(where)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start recording")
        self.start_button.clicked.connect(self.start)
        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self.test)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.status = QLabel("idle")
        controls.addWidget(self.start_button)
        controls.addWidget(self.test_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.status, stretch=1)

        controls.addWidget(QLabel("Resp HP (Hz)"))
        self.hp = QDoubleSpinBox()
        self.hp.setRange(0.0, 100.0)
        self.hp.setDecimals(2)
        self.hp.setValue(RESP_HP_DEFAULT)
        controls.addWidget(self.hp)
        controls.addWidget(QLabel("LP (Hz)"))
        self.lp = QDoubleSpinBox()
        self.lp.setRange(0.0, RATE / 2)
        self.lp.setDecimals(1)
        self.lp.setValue(RESP_LP_DEFAULT)
        controls.addWidget(self.lp)
        apply = QPushButton("Apply filter")
        apply.clicked.connect(self._apply_filter)
        controls.addWidget(apply)
        layout.addLayout(controls)

        self.canvas = None
        self.lines = {}
        self._make_plot(layout)

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(TICK_MS)
        self.tick_timer.timeout.connect(self._tick)
        self.redraw_timer = QTimer(self)
        self.redraw_timer.setInterval(REDRAW_MS)
        self.redraw_timer.timeout.connect(self._redraw)

        self.copier = Copier(self)
        self.copier.done.connect(self._copied)
        self._writing: None | tuple[Draft, str, dict] = None  # draft, exp, record

        context.selection_changed.connect(self.follow)
        context.draft_changed.connect(self.follow)
        self.follow()

    # ----------------------------------------------------------------- #
    # The plot
    # ----------------------------------------------------------------- #

    def _make_plot(self, layout: QVBoxLayout) -> None:
        try:
            import matplotlib

            matplotlib.use("QtAgg", force=False)
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

        except Exception as error:  # pragma: no cover - depends on the machine
            logger.warning(f"No live plot ({error})")
            layout.addWidget(QLabel(f"Live plot unavailable: {error}"))
            return

        figure = Figure(figsize=(8, 6), tight_layout=True)
        self.axes = figure.subplots(len(TRACES), 1, sharex=True)
        self.canvas = FigureCanvasQTAgg(figure)

        for axis, (key, label) in zip(self.axes, TRACES):
            (line,) = axis.plot([], [], linewidth=0.8)
            axis.set_ylabel(label, fontsize=8)
            axis.tick_params(labelsize=8)
            self.lines[key] = line

        self.axes[-1].set_xlabel("time (s)", fontsize=8)
        layout.addWidget(self.canvas, stretch=1)

    def _redraw(self) -> None:
        if self.canvas is None or not self.buffer.chunks:
            return

        latest = self.buffer.latest

        for key, line in self.lines.items():
            t, y = self.buffer.trace(key)
            line.set_data(t, y)
            axis = line.axes
            axis.set_xlim(max(0.0, latest - WINDOW_S), max(WINDOW_S, latest))

            if y.size:
                lo, hi = float(y.min()), float(y.max())
                pad = 0.1 * (hi - lo) or 0.5
                axis.set_ylim(lo - pad, hi + pad)

        self.canvas.draw_idle()

    # ----------------------------------------------------------------- #
    # Naming
    # ----------------------------------------------------------------- #

    @property
    def draft(self) -> None | Draft:
        return self.context.draft

    @property
    def picked(self) -> None | str:
        """The experiment selected in the right bar: what Start records."""
        name = self.context.selected_experiment
        draft = self.draft

        return name if draft is not None and draft.experiment(name or "") else None

    def follow(self) -> None:
        """Show the selected session and experiment (frozen while recording)."""
        if self.recording is not None:
            return  # the names are frozen while a file is being written

        draft = self.draft

        if draft is None:
            self.session_label.setText("no session selected in the right bar")
        elif draft.mouse_id is None:
            self.session_label.setText(f"{draft.label}: pick its mouse first")
        else:
            self.session_label.setText(f"{draft.label} in {draft.folder}")

        self.experiment.setText(self.picked or "")
        self._refresh_target()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.follow()
        super().showEvent(event)

    def _refresh_target(self) -> None:
        if self.recording is not None:
            return

        name = self.picked

        if not name:
            self.target.setText("select an experiment in the right bar")
            return

        if self.draft is None or self.draft.mouse_id is None:
            self.target.setText("pick a session with a mouse (or just test)")
            return

        _, share, _ = recording_path(self.context.recordings, self.draft, name)
        self.target.setText(share)

    # ----------------------------------------------------------------- #
    # Start / stop
    # ----------------------------------------------------------------- #

    def test(self) -> None:
        """Run the source and show the traces; nothing is written."""
        self._begin(None, TEST_MOUSE, self.picked or "test")

    def start(self) -> None:
        if self.recording is not None:
            return

        name = self.picked
        draft = self.draft

        if draft is None or draft.mouse_id is None:
            self.context.report(
                "Recorder",
                "Select a session with a mouse in the right bar (or just test).",
            )
            return

        if name is None:
            self.context.report(
                "Recorder",
                "Select the experiment to record in the right bar (add one first "
                "if the session has none).",
            )
            return

        earlier = Draft.latest(draft.experiment(name))

        if earlier is not None and not self.context.confirm(
            "Recorder",
            f"{name} already has a recording ({earlier['file']}).\n\n"
            "Do you want to record a new one?\n"
            "(Only the new one will be used.)",
            "Record again",
        ):
            return

        started = datetime.now()
        local, share, mouse = recording_path(
            self.context.recordings, draft, name, started
        )

        # Where ScanImage saves, if the share is there; made again at the copy
        try:
            make_experiment_folders(self.context.main_folder / draft.folder / name)
        except OSError as error:
            logger.warning(f"Could not make the experiment folders yet: {error}")

        started_ok = self._begin(
            local,
            mouse,
            name,
            {
                "writer_started": started.isoformat(),
                "session_folder": draft.folder,
                "experiment_folder": name,
            },
        )

        if not started_ok:
            return

        record = {
            "file": local.name,
            "local": str(local),
            "share": share,
            "started": started.isoformat(sep=" ", timespec="seconds"),
        }
        draft.add_recording(name, record)
        self._writing = (draft, name, record)
        self.context.recording_now = str(local)

        if earlier is not None and earlier.get("copied"):
            self._hide(draft, name, earlier)

        self.context.announce_draft()

    def _begin(
        self,
        path: None | Path,
        mouse: str,
        name: str,
        attributes: None | dict = None,
    ) -> bool:
        """Start recording (into `path`, or nowhere for a test); whether it did."""
        if self.recording is not None:
            return False

        if self.source is None:
            try:
                self.source = DaqSource()

            except Exception as error:  # no nidaqmx, no driver, or no board
                logger.exception("No DAQ")
                self.context.report(
                    "Recorder",
                    f"Cannot reach the DAQ board: {error}\n\nCheck that it is "
                    "plugged in and that NI-DAQmx and `nidaqmx` are installed.",
                )
                return False

        self.recording = Recording(
            path,
            mouse=mouse,
            experiment=name,
            source=self.source,
            hp_hz=self.hp.value(),
            lp_hz=self.lp.value(),
            extra_attributes=attributes,
        )

        try:
            self.recording.start()

        except Exception as error:
            self.recording = None
            self.context.report("Recorder", f"Could not start:\n\n{error}")
            return False

        self.buffer.clear()
        self.target.setText("(test: nothing is written)" if path is None else str(path))
        self.start_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.tick_timer.start()
        self.redraw_timer.start()
        simulated = (
            " (simulated source)" if isinstance(self.source, SimulatedSource) else ""
        )
        self.context.status.emit(
            f"{'Testing' if path is None else 'Recording ' + path.name}{simulated}"
        )

        return True

    def stop(self, *, final_read: bool = True) -> None:
        """
        Stop and close the file. `final_read` first takes what the source
        still holds. A read that failed (which stops the recording) skips it.
        """
        if self.recording is None:
            return

        self.tick_timer.stop()
        self.redraw_timer.stop()

        if final_read:
            self._tick()

            if self.recording is None:
                return  # that read failed, and stopped everything already

        recording, self.recording = self.recording, None

        try:
            recording.stop()

        except Exception as error:
            self.context.report("Recorder", f"Trouble closing the file:\n\n{error}")

        what = "test" if recording.path is None else recording.path.name
        self.status.setText(f"stopped after {recording.duration_s:.1f} s: {what}")
        self.start_button.setEnabled(True)
        self.test_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.follow()  # the selection may have moved meanwhile

        if self._writing is not None:
            draft, name, record = self._writing
            self._writing = None
            self.context.recording_now = None
            draft = self.context.live_draft(draft.path) or draft
            draft.update_recording(
                name,
                record["file"],
                stopped=datetime.now().isoformat(sep=" ", timespec="seconds"),
            )
            self.context.status.emit(f"Saved {record['file']}; copying it to the share")
            self.copier.copy(draft, name, record, self.context.main_folder)

    def _copied(self, draft_path: str, name: str, file: str, error: str) -> None:
        """A copy to the share ended: note it in the draft, or say it did not."""
        self.copier.running.discard(file)
        draft = self.context.live_draft(Path(draft_path))

        if error:
            message = (
                f"Could not copy {file} to the network share ({error}).\n"
                f"'Submit' will try to copy it again."
            )
            self.status.setText(message)
            self.context.status.emit(message)
            return

        if draft is None:
            return  # submitted or gone meanwhile: Submit found the copy itself

        draft.update_recording(
            name, file, copied=datetime.now().isoformat(sep=" ", timespec="seconds")
        )
        record = next(
            (r for r in draft.recordings(name) if r.get("file") == file), None
        )

        if record is not None and record.get("replaced"):
            self._hide(draft, name, record)  # re-recorded while it copied

        self.context.status.emit(f"Copied {file} to the share")

    def _hide(self, draft: Draft, name: str, record: dict) -> None:
        """A replaced recording's copy, out of the decoder's sight (or at Submit)."""
        try:
            hide_on_share(self.context.main_folder / record["share"])
            draft.update_recording(name, record["file"], hidden=True)

        except OSError as error:
            logger.warning(f"Could not hide {record['file']} yet: {error}")

    def _tick(self) -> None:
        if self.recording is None:
            return

        try:
            chunk = self.recording.tick()

        except Exception as error:
            logger.exception("Recording tick failed")
            self.stop(final_read=False)
            self.context.report("Recorder", f"Recording stopped:\n\n{error}")
            return

        if chunk is not None:
            self.buffer.add(chunk)
            self.status.setText(f"recording {self.recording.duration_s:.1f} s")

    def _apply_filter(self) -> None:
        if self.recording is not None:
            self.recording.apply_filter(self.hp.value(), self.lp.value())

    def close_down(self) -> None:
        """Stop a recording that is still running when the window closes."""
        self.stop()
