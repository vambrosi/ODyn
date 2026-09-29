"""
The recorder tab: the live traces and the Start/Stop around a `Recording`.

The file goes where `submit` will look for it: `<main folder>/<session
folder>/<experiment>/sync/<mouse>_<experiment>_<started>.h5`, with the
experiment's `raw/` and `processed/` folders made beside it so the ScanImage
side has somewhere to save. Recording into an experiment the draft does not
list yet adds it to the draft, so the day's paperwork stays in step with what
was actually recorded.

"Test" runs the source and shows the traces without writing anything: it is
what the rig runs to check the wiring.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..database import SYNC_FOLDER
from ..utils import logger
from .context import Context
from .draft import Draft
from .recorder import (
    RATE,
    RESP_HP_DEFAULT,
    RESP_LP_DEFAULT,
    Recording,
    SimulatedSource,
    Source,
    available_source,
    minmax_decimate,
    sync_file_name,
)

TICK_MS = 50
REDRAW_MS = 200
WINDOW_S = 10.0
POINTS_PER_TRACE = 1500

TEST_MOUSE = "test"

# The file label, wide enough for a session path on one line
TARGET_WIDTH_PX = 640

# Folders every experiment gets, so ScanImage and the pipeline find their place
EXPERIMENT_FOLDERS = ("raw", SYNC_FOLDER, "processed")

TRACES = (
    ("vel", "velocity (cm/s)"),
    ("respf", "respiration (V)"),
    ("frame", "frame clock (V)"),
    ("odor", "odor pulse (V)"),
    ("cam", "camera strobe (V)"),
)


def recording_path(
    main_folder: Path,
    draft: Draft,
    experiment: str,
    started: None | datetime = None,
) -> tuple[Path, str]:
    """Where the next file goes and the mouse name it is stamped with."""
    started = started or datetime.now()
    mouse = f"m{draft.mouse_id}"
    folder = Path(main_folder) / draft.folder / experiment

    return folder / SYNC_FOLDER / sync_file_name(mouse, experiment, started), mouse


def make_experiment_folders(experiment_folder: Path) -> None:
    for name in EXPERIMENT_FOLDERS:
        (experiment_folder / name).mkdir(parents=True, exist_ok=True)


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
        self.experiment = QComboBox()
        self.experiment.currentTextChanged.connect(self._experiment_edited)
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

        context.selection_changed.connect(self.follow)
        context.draft_changed.connect(self.follow)
        self._following = False
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

    def follow(self) -> None:
        """
        Show the selected session and its experiment opened last: the draft's
        experiments to pick from, that one picked. A recording goes into an
        experiment the session lists, never into a new name.
        """
        if self.recording is not None:
            return  # the names are frozen while a file is being written

        draft = self.draft
        self._following = True
        self.experiment.blockSignals(True)
        self.experiment.clear()

        if draft is None:
            self.session_label.setText("no session selected in the right bar")
        elif draft.mouse_id is None:
            self.session_label.setText(f"{draft.label}: pick its mouse first")
        else:
            self.session_label.setText(f"{draft.label} in {draft.folder}")

        names = [] if draft is None else [str(e["name"]) for e in draft.experiments]
        self.experiment.addItems(names)
        picked = self.context.experiment
        self.experiment.setCurrentIndex(names.index(picked) if picked in names else -1)
        self.experiment.blockSignals(False)
        self._following = False
        self._refresh_target()

    def _experiment_edited(self, name: str) -> None:
        """A listed experiment picked here is picked everywhere."""
        self._refresh_target()

        if self._following or self.draft is None:
            return

        if self.draft.experiment(name.strip()) is not None:
            self.context.select_experiment(name.strip())

    def _refresh_target(self) -> None:
        if self.recording is not None:
            return

        name = self.experiment.currentText().strip()

        if not name:
            self.target.setText("add an experiment to the session first")
            return

        if self.draft is None or self.draft.mouse_id is None:
            self.target.setText("pick a session with a mouse (or just test)")
            return

        path, _ = recording_path(self.context.main_folder, self.draft, name)
        self.target.setText(str(path.relative_to(self.context.main_folder)))

    # ----------------------------------------------------------------- #
    # Start / stop
    # ----------------------------------------------------------------- #

    def test(self) -> None:
        """Run the source and show the traces; nothing is written."""
        self._begin(None, TEST_MOUSE, self.experiment.currentText().strip() or "test")

    def start(self) -> None:
        if self.recording is not None:
            return

        name = self.experiment.currentText().strip()

        draft = self.draft

        if draft is None or draft.mouse_id is None:
            self.context.report(
                "Recorder",
                "Select a session with a mouse in the right bar (or just test).",
            )
            return

        if not name or draft.experiment(name) is None:
            self.context.report("Recorder", "Add an experiment to the session first.")
            return

        started = datetime.now()
        path, mouse = recording_path(self.context.main_folder, draft, name, started)
        make_experiment_folders(path.parent.parent)

        self._begin(
            path,
            mouse,
            name,
            {
                "writer_started": started.isoformat(),
                "session_folder": draft.folder,
                "experiment_folder": name,
            },
        )

    def _begin(
        self,
        path: None | Path,
        mouse: str,
        name: str,
        attributes: None | dict = None,
    ) -> None:
        if self.recording is not None:
            return

        if self.source is None:
            self.source = available_source()

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
            self.context.report("Recorder", f"Could not start: {error}")
            return

        self.buffer.clear()
        self.target.setText("(test: nothing is written)" if path is None else str(path))
        self.start_button.setEnabled(False)
        self.test_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.experiment.setEnabled(False)
        self.tick_timer.start()
        self.redraw_timer.start()
        simulated = (
            " (simulated source)" if isinstance(self.source, SimulatedSource) else ""
        )
        self.context.status.emit(
            f"{'Testing' if path is None else 'Recording ' + path.name}{simulated}"
        )

    def stop(self) -> None:
        if self.recording is None:
            return

        self.tick_timer.stop()
        self.redraw_timer.stop()
        self._tick()
        recording, self.recording = self.recording, None

        try:
            recording.stop()

        except Exception as error:
            self.context.report("Recorder", f"Trouble closing the file: {error}")

        what = "test" if recording.path is None else recording.path.name
        self.status.setText(f"stopped after {recording.duration_s:.1f} s: {what}")
        self.start_button.setEnabled(True)
        self.test_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.experiment.setEnabled(True)
        self._refresh_target()

        if recording.path is not None:
            self.context.status.emit(f"Saved {recording.path}")

    def _tick(self) -> None:
        if self.recording is None:
            return

        try:
            chunk = self.recording.tick()

        except Exception as error:
            logger.exception("Recording tick failed")
            self.context.report("Recorder", f"Recording stopped: {error}")
            self.stop()
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
