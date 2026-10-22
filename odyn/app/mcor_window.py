"""
The window an experiment's motion correction is checked in, opened from the
QC page. As the main window: the steps on a bar at the left (Full run,
Tests, Final), a page each, and the tools of the page on a bar at the right,
each with its shortcut (in its tooltip).

**Tests** shows the raw video and every finished run's test video side by
side, playing in step (`player`), each with its numbers. A click selects a
tile, Shift adds one; a right click opens the tools for it. The tools: Fit,
Mirrored pointer, Focused only, Clear focus, Star, Set aside (into a strip
under the grid, in order), Bring back, the report of the focused run (its
plot of shifts), and Pick candidate, which has every acquisition corrected
with the focused run settings, recording the runs set aside (see
`cluster.pick_candidate`). Space plays, the arrows step (Shift: 10 frames),
`[` and `]` change the speed.

Everything is opened from the experiment's local copy (`cache`): played from
the share, a video is slow.
"""

from __future__ import annotations

import math

import numpy as np

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDesktopServices,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..groups import RAW_PREVIEW_FOLDER, frame_map_path, read_frame_map
from . import style
from .player import Video, VideoGrid
from .plots import RAW_COLOR, Series, ShiftPlots, read_summary, run_color
from .symbols import icon

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
    flow_worst_px: None | float = None  # the largest 90th percentile
    flow_worst_acq: None | int = None  # and its acquisition
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
    """
    A run's numbers for its tile, `corr 0.912  flow 0.52 / 2.10 px  jitter
    0.10 px`: the flow typical (median) and worst (largest 90th percentile).
    """
    parts = []

    if _number(run.corr_with_ref):
        parts.append(f"corr {run.corr_with_ref:.3f}")

    if _number(run.flow_px) and _number(run.flow_worst_px):
        parts.append(f"flow {run.flow_px:.2f} / {run.flow_worst_px:.2f} px")
    elif _number(run.flow_px):
        parts.append(f"flow {run.flow_px:.2f} px")

    if _number(run.jitter_px):
        parts.append(f"jitter {run.jitter_px:.2f} px")

    if _number(run.at_limit) and run.at_limit > 0:
        parts.append(f"at limit {run.at_limit:.1%}")

    return "  ".join(parts)


def details_line(run: Run) -> str:
    """A run's numbers in words, for the line under the grid."""
    parts = []

    if _number(run.corr_with_ref):
        parts.append(f"correlation with the reference {run.corr_with_ref:.3f}")

    if _number(run.flow_px):
        parts.append(f"residual flow {run.flow_px:.2f} px typical")

    if _number(run.flow_worst_px):
        where = (
            f" (acquisition {int(run.flow_worst_acq)})"
            if _number(run.flow_worst_acq)
            else ""
        )
        parts.append(f"{run.flow_worst_px:.2f} px worst{where}")

    if _number(run.jitter_px):
        parts.append(f"patch jitter {run.jitter_px:.2f} px")

    if _number(run.at_limit) and run.at_limit > 0:
        parts.append(f"{run.at_limit:.1%} of frames at the shift limit")

    return f"{run.name}: {', '.join(parts)}" if parts else run.name


class McorWindow(QWidget):
    """
    The motion correction of one experiment, from its local `folder`, and
    its finished `runs` (see the module description). `status` gets the
    window messages; `pick` gets the name of the picked run and those set
    aside, first to last (none: no picking); `closed` fires as the window
    closes.
    """

    closed = Signal()

    def __init__(
        self,
        title: str,
        folder: Path,
        runs: list[Run],
        status: Callable[[str], None],
        pick: None | Callable[[str, list[str]], None] = None,
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
        self.pages = QStackedWidget()
        middle = QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        self.left = self._bar("Steps", middle)
        middle.addWidget(self.pages, stretch=1)
        self.right = self._bar("Tools", middle)

        layout = QVBoxLayout(self)
        layout.addWidget(self.state)
        layout.addLayout(middle, stretch=1)

        # The steps, in the order of the work; only Tests has a page so far
        self.steps = QActionGroup(self)
        self.full_run_action = self._action(
            self.left, "full-run", "Full run", "Ctrl+1", checkable=True
        )
        self.tests_action = self._action(
            self.left, "tests", "Tests", "Ctrl+2", checkable=True
        )
        self.final_action = self._action(
            self.left, "final", "Final", "Ctrl+3", checkable=True
        )

        for action in (self.full_run_action, self.tests_action, self.final_action):
            self.steps.addAction(action)

        self.full_run_action.setEnabled(False)
        self.final_action.setEnabled(False)
        self.tests_action.setChecked(True)

        tests = QWidget()
        tests_layout = QVBoxLayout(tests)
        tests_layout.setContentsMargins(8, 0, 8, 0)
        self.grid = VideoGrid()
        self.grid.focus_changed.connect(self._enable)
        self.grid.menu_requested.connect(self._menu)
        self.plots = ShiftPlots()
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(self.grid)
        self.split.addWidget(self.plots)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 1)
        self.split.setChildrenCollapsible(False)
        self.split.setSizes([600, 220])
        tests_layout.addWidget(self.split, stretch=1)
        self.grid.frame_changed.connect(self._mark_frame)
        self.grid.hovered.connect(self._show_details)
        self.hovered: None | Video = None
        self.frame_x: None | np.ndarray = None  # each video frame on the plots
        # The numbers of the run under the pointer, or else the focused one:
        # one line, so the grid keeps its room
        self.details = QLabel("")
        self.details.setFixedHeight(self.details.fontMetrics().height() + 4)
        self.details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        tests_layout.addWidget(self.details)
        self.missing = QLabel("")
        tests_layout.addWidget(self.missing)
        self.pages.addWidget(tests)

        # The tools
        self.fit_action = self._action(self.right, "fit", "Fit (zoom out)", "0")
        self.fit_action.triggered.connect(self.grid.fit)
        self.pointer_action = self._action(
            self.right, "pointer", "Mirrored pointer", "H", checkable=True
        )
        self.pointer_action.setChecked(self.grid.view.mirror)
        self.pointer_action.toggled.connect(self.grid.set_mirror)
        self.only_action = self._action(
            self.right, "focused-only", "Focused only", "F", checkable=True
        )
        self.only_action.toggled.connect(self.grid.show_focused_only)
        self.clear_action = self._action(
            self.right, "clear-focus", "Clear focus", "Esc"
        )
        self.clear_action.triggered.connect(self.grid.clear_focus)
        self.right.addSeparator()
        self.star_action = self._action(self.right, "star", "Star", "S")
        self.star_action.triggered.connect(self.grid.star_focused)
        self.aside_action = self._action(self.right, "set-aside", "Set aside", "X")
        self.aside_action.triggered.connect(self.grid.set_aside_focused)
        self.back_action = self._action(
            self.right, "undo", "Bring back the last set aside", "Ctrl+Z"
        )
        self.back_action.triggered.connect(lambda: self.grid.bring_back())
        self.right.addSeparator()
        self.plots_action = self._action(
            self.right, "plots", "Plots of the focused runs", "P", checkable=True
        )
        self.plots_action.setChecked(True)
        self.plots_action.toggled.connect(self.plots.setVisible)
        self.report_action = self._action(
            self.right, "report", "Open the report of the focused run", "R"
        )
        self.report_action.triggered.connect(self._show_report)
        self._foot(self.right)
        self.pick_action = self._action(
            self.right,
            "pick",
            "Pick candidate: motion correct every acquisition with the "
            "focused run settings",
            "Ctrl+Return",
        )
        self.pick_action.triggered.connect(self._pick)
        self.pick_action.setVisible(pick is not None)

        # The transport, from the keyboard
        for key, do in (
            ("Space", self.grid.toggle),
            ("Right", lambda: self.grid.step_by(1)),
            ("Left", lambda: self.grid.step_by(-1)),
            ("Shift+Right", lambda: self.grid.step_by(10)),
            ("Shift+Left", lambda: self.grid.step_by(-10)),
            ("]", lambda: self.grid.change_speed(1)),
            ("[", lambda: self.grid.change_speed(-1)),
        ):
            QShortcut(QKeySequence(key), self).activated.connect(do)

        full = QShortcut(QKeySequence(QKeySequence.StandardKey.FullScreen), self)
        full.activated.connect(self._toggle_full_screen)

        self.set_runs(runs)
        self.resize(1200, 800)

    # ----------------------------------------------------------------------- #
    # The bars
    # ----------------------------------------------------------------------- #

    def _bar(self, name: str, row: QHBoxLayout) -> QToolBar:
        """A vertical bar of icons, as the main window has them."""
        bar = QToolBar(name)
        bar.setOrientation(Qt.Orientation.Vertical)
        bar.setMovable(False)
        bar.setIconSize(QSize(style.BAR_ICON_PX, style.BAR_ICON_PX))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(*style.BAR_MARGINS)
        holder_layout.addWidget(bar)
        row.addWidget(holder)

        return bar

    def _foot(self, bar: QToolBar) -> None:
        """Push what follows to the bottom of the bar."""
        filler = QWidget()
        filler.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        bar.addWidget(filler)

    def _action(
        self, bar: QToolBar, symbol: str, title: str, key: str, checkable=False
    ) -> QAction:
        """A bar button with its shortcut, named in its tooltip."""
        action = QAction(icon(symbol), "", self)
        sequence = QKeySequence(key)
        action.setShortcut(sequence)
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        native = sequence.toString(QKeySequence.SequenceFormat.NativeText)
        action.setToolTip(f"{title} ({native})")
        action.setCheckable(checkable)
        bar.addAction(action)
        button = bar.widgetForAction(action)

        if isinstance(button, QToolButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        return action

    # ----------------------------------------------------------------------- #
    # Videos
    # ----------------------------------------------------------------------- #

    def set_runs(self, runs: list[Run]) -> None:
        """Show the raw video and `runs`' test videos, those downloaded."""
        videos = []
        self.runs = {}
        raw = raw_video(self.folder)

        if raw is not None:
            # Gray, as its correlation on the plots
            videos.append(Video("raw", raw, candidate=False, color=RAW_COLOR))

        absent = []

        for run in runs:
            path = test_video(run.folder)

            if path is None:
                absent.append(run.name)
                continue

            color = run_color(len(self.runs))
            videos.append(Video(run.name, path, numbers_line(run), color=color))
            self.runs[run.name] = run

        self.grid.set_videos(videos)
        problems = ([] if raw is not None else ["raw"]) + absent
        notes = [f"Not downloaded yet: {', '.join(problems)}"] if problems else []
        notes += self._load_plots(videos, raw)
        self.missing.setText("\n".join(notes))
        self._enable()

    def _load_plots(self, videos: list[Video], raw: None | Path) -> list[str]:
        """
        Read the runs' per-frame numbers for the plots, and where each video
        frame falls on them (from the raw video's frame map). Returns what
        could not be plotted, in words.
        """
        series, missing = [], []

        for video in videos:
            if video.name not in self.runs:
                continue

            data = read_summary(self.runs[video.name].folder)

            if data is None:
                missing.append(video.name)
            else:
                series.append(Series(video.name, video.color, data))

        self.plots.set_series(series)
        self.frame_x = None

        if series and raw is not None:
            frames = read_frame_map(frame_map_path(raw))
            data = series[0].data
            starts = dict(
                zip(
                    data["acq_ids"].tolist(),
                    np.concatenate([[0], np.cumsum(data["frame_counts"])[:-1]]),
                )
            )

            if frames is not None and set(frames["acq_id"]) <= set(starts):
                self.frame_x = np.array(
                    [
                        starts[acq_id] + within
                        for acq_id, within in zip(
                            frames["acq_id"], frames["frame_in_acq"]
                        )
                    ],
                    float,
                )

        self._mark_frame(self.grid.index)

        if missing:
            return [f"No per-frame numbers to plot (older tests): {', '.join(missing)}"]

        return []

    def _mark_frame(self, index: int) -> None:
        """Mark the frame the videos show on the plots."""
        x = self.frame_x
        self.plots.set_marker(
            float(x[index]) if x is not None and index < len(x) else None
        )

    def _picked_run(self) -> None | Run:
        """The focused run, when one alone is."""
        picked = self.grid.focused

        if len(picked) != 1:
            return None

        return self.runs.get(picked[0].name)

    def set_status(self, text: str) -> None:
        """Show the experiment status (see `pane_qc.my_experiments`)."""
        self.state.setText(f"Status: {text}")

    def _show_details(self, video: None | Video = None) -> None:
        """The numbers of the run under the pointer, else the one focused."""
        self.hovered = video
        focused = self.grid.focused
        shown = video or (focused[0] if len(focused) == 1 else None)
        run = None if shown is None else self.runs.get(shown.name)
        self.details.setText("" if run is None else details_line(run))

    def _enable(self, *_) -> None:
        self.plots.show_runs([video.name for video in self.grid.focused])
        self._show_details(self.hovered)
        run = self._picked_run()
        focused = self.grid.focused
        only = self.grid.only is not None
        self.report_action.setEnabled(
            run is not None and (run.folder / "shifts.png").is_file()
        )
        self.pick_action.setEnabled(run is not None)
        self.clear_action.setEnabled(bool(focused))
        self.star_action.setEnabled(bool(focused))
        self.aside_action.setEnabled(any(video.candidate for video in focused))
        self.back_action.setEnabled(bool(self.grid.aside))
        self.only_action.setEnabled(bool(focused) or only)
        self.only_action.blockSignals(True)
        self.only_action.setChecked(only)
        self.only_action.blockSignals(False)

    def _menu(self, tile, where) -> None:
        """The tools for a tile, at `where` (a right click)."""
        menu = QMenu(self)

        if tile in self.grid.aside:
            menu.addAction("Bring back", lambda: self.grid.bring_back(tile))
        else:
            for action, text in (
                (self.star_action, "Unstar" if tile.starred else "Star"),
                (self.aside_action, "Set aside"),
                (self.report_action, "Open the report"),
                (self.pick_action, "Pick candidate"),
            ):
                if action.isVisible():
                    item = menu.addAction(text, action.trigger)
                    item.setEnabled(action.isEnabled())
                    item.setShortcut(action.shortcut())

        menu.popup(where)

    def _pick(self) -> None:
        run = self._picked_run()

        if run is not None and self.pick is not None:
            self.pick(run.name, [video.name for video in self.grid.set_aside])

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
