"""
The window an experiment's motion correction is checked in, opened from the
QC page, with a tab per step. For now one: **Tests**, the raw video and
every finished run's test video side by side, playing in step (`player`),
each with its numbers. A click selects a tile; **Show selected** keeps the
selected ones; **Show report** opens the selected run's plot of shifts;
**Pick candidate** has every acquisition corrected with the selected run's
settings (see `cluster.pick_candidate`).

Everything is opened from the experiment's local copy (`cache`): played from
the share, a video is slow.
"""

from __future__ import annotations

import math

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..groups import RAW_PREVIEW_FOLDER
from . import style
from .player import Video, VideoGrid

# Where the raw video is in experiment folders from before `raw/`: the
# sweep's baseline
LEGACY_RAW_FOLDER = "rigid"

VIDEO_SUFFIXES = (".avi", ".mp4")


@dataclass
class Run:
    """A finished test of the experiment: its name, local folder and numbers."""

    name: str
    folder: Path
    corr_with_ref: None | float = None
    flow_px: None | float = None
    jitter_px: None | float = None
    at_limit: None | float = None


def _videos(folder: Path, kind: str) -> list[Path]:
    """The videos of `kind` ("raw", "test") in `folder`, not raw | test ones."""
    if not folder.is_dir():
        return []

    return [
        path
        for path in sorted(folder.glob(f"*_{kind}.*"))
        if path.suffix in VIDEO_SUFFIXES and not path.stem.endswith("_raw_test")
    ]


def raw_video(folder: Path) -> None | Path:
    """The experiment's raw video, in its local `folder`, or `None`."""
    for place in (RAW_PREVIEW_FOLDER, LEGACY_RAW_FOLDER):
        found = _videos(folder / place, "raw")

        if found:
            return found[0]

    return None


def test_video(folder: Path) -> None | Path:
    """
    A run's test video, in its local `folder`, or `None`: the test alone, or
    raw | test in folders from before the raw video was apart.
    """
    found = _videos(folder, "test")

    if found:
        return found[0]

    both = [
        path
        for path in sorted(folder.glob("*_raw_test.*"))
        if path.suffix in VIDEO_SUFFIXES
    ]

    return both[0] if both else None


def _number(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def numbers_line(run: Run) -> str:
    """A run's numbers for its tile, `corr 0.912  flow 0.52 px  jitter 0.10 px`."""
    parts = []

    if _number(run.corr_with_ref):
        parts.append(f"corr {run.corr_with_ref:.3f}")

    if _number(run.flow_px):
        parts.append(f"flow {run.flow_px:.2f} px")

    if _number(run.jitter_px):
        parts.append(f"jitter {run.jitter_px:.2f} px")

    if _number(run.at_limit) and run.at_limit > 0:
        parts.append(f"at limit {run.at_limit:.1%}")

    return "  ".join(parts)


class McorWindow(QWidget):
    """
    The motion correction of one experiment, from its local `folder`, and
    its finished `runs` (see the module description). `status` gets the
    window messages; `pick` gets the name of the picked run (none: no
    picking); `closed` fires as the window closes.
    """

    closed = Signal()

    def __init__(
        self,
        title: str,
        folder: Path,
        runs: list[Run],
        status: Callable[[str], None],
        pick: None | Callable[[str], None] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Motion correction: {title}")
        self.setStyleSheet(style.window_stylesheet())  # as the main window
        self.folder = folder
        self.status = status
        self.pick = pick
        self.runs: dict[str, Run] = {}  # each tile's run, by the tile's name

        self.state = QLabel("")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.state)
        layout.addWidget(self.tabs)

        tests = QWidget()
        tests_layout = QVBoxLayout(tests)
        tests_layout.setContentsMargins(0, 0, 0, 0)
        self.grid = VideoGrid()
        self.grid.selection_changed.connect(self._enable)
        tests_layout.addWidget(self.grid, stretch=1)

        row = QHBoxLayout()
        self.missing = QLabel("")
        self.show_report = QPushButton("Show report")
        self.show_report.setToolTip("Open shifts report for this run")
        self.show_report.clicked.connect(self._show_report)
        self.pick_button = QPushButton("Pick candidate")
        self.pick_button.setToolTip(
            "Motion correct every acquisition with this run's settings"
        )
        self.pick_button.clicked.connect(self._pick)
        self.pick_button.setVisible(pick is not None)
        row.addWidget(self.missing, stretch=1)
        row.addWidget(self.show_report)
        row.addWidget(self.pick_button)
        tests_layout.addLayout(row)
        index = self.tabs.addTab(tests, "Tests")
        self.tabs.setTabToolTip(index, "The raw video and each test run, side by side")

        full = QShortcut(QKeySequence(QKeySequence.StandardKey.FullScreen), self)
        full.activated.connect(self._toggle_full_screen)

        self.set_runs(runs)
        self.resize(1200, 800)

    def set_runs(self, runs: list[Run]) -> None:
        """Show the raw video and `runs`' test videos, those downloaded."""
        videos = []
        self.runs = {}
        raw = raw_video(self.folder)

        if raw is not None:
            videos.append(Video("raw", raw))

        absent = []

        for run in runs:
            path = test_video(run.folder)

            if path is None:
                absent.append(run.name)
                continue

            videos.append(Video(run.name, path, numbers_line(run)))
            self.runs[run.name] = run

        self.grid.set_videos(videos)
        problems = ([] if raw is not None else ["raw"]) + absent
        self.missing.setText(
            f"Not downloaded yet: {', '.join(problems)}" if problems else ""
        )
        self._enable()

    def _picked_run(self) -> None | Run:
        picked = self.grid.selected

        if len(picked) != 1:
            return None

        return self.runs.get(picked[0].name)

    def set_status(self, text: str) -> None:
        """Show the experiment status (see `pane_qc.my_experiments`)."""
        self.state.setText(f"Status: {text}")

    def _enable(self, *_) -> None:
        run = self._picked_run()
        self.show_report.setEnabled(
            run is not None and (run.folder / "shifts.png").is_file()
        )
        self.pick_button.setEnabled(run is not None)

    def _pick(self) -> None:
        run = self._picked_run()

        if run is not None and self.pick is not None:
            self.pick(run.name)

    def _show_report(self) -> None:
        run = self._picked_run()

        if run is None:
            return

        path = run.folder / "shifts.png"

        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.status(f"Could not open {path}")

    def _toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.grid.pause()
        self.grid.stop_reading()
        self.closed.emit()
        super().closeEvent(event)
