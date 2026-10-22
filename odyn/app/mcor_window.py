"""
The window an experiment's motion correction is checked in, opened from the
QC page. As the main window: the steps on a bar at the left (Raw check,
Tests, Final check), a page each, and the tools of the page on a bar at the
right, each with its shortcut (in its tooltip).

The full run (`full/`, written by the whole correction) has every
acquisition raw and as corrected and saved, at half size, read from the
files as they play (too long to hold), with the acquisition shown and the
plots over every frame. Two steps show it, on the same page:

**Raw check** shows the raw video alone. E excludes the acquisition shown;
Finished raw check, at the foot of the bar, records the list
(`Group.check_raw`) and goes on to the tests. The window opens on it until
the raw check is done.

**Final check** shows raw | corrected, once the picked correction is saved
(a pick of the settings the full run already used goes straight there).
The same list of exclusions, still open to change, goes with Approve
(`Group.approve_mcor_files`).

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
import pandas as pd

from collections.abc import Callable, Sequence
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

from ..groups import (
    FULL_FOLDER,
    RAW_PREVIEW_FOLDER,
    frame_map_path,
    read_frame_map,
)
from . import style
from .player import Video, VideoGrid
from .plots import RAW_COLOR, Series, ShiftPlots, read_summary, run_color
from .symbols import icon

# The pages, in the order of the steps on the left bar
# The steps of the work, and the pages they show: the raw check and the final
# check are the same page, the first with the raw video alone, the last with
# raw | corrected
RAW_STEP, TESTS_STEP, FINAL_STEP = 0, 1, 2
FULL_PAGE, TESTS_PAGE = 0, 1

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


def frame_positions(
    frames: None | pd.DataFrame, data: dict[str, np.ndarray]
) -> None | np.ndarray:
    """
    Where each video frame (`frames`, its frame map) falls on the plots of
    `data` (a `shifts_summary.npz`), or `None` when they do not match.
    """
    if frames is None:
        return None

    starts = dict(
        zip(
            data["acq_ids"].tolist(),
            np.concatenate([[0], np.cumsum(data["frame_counts"])[:-1]]),
        )
    )

    if not set(frames["acq_id"]) <= set(starts):
        return None

    return np.array(
        [
            starts[acq_id] + within
            for acq_id, within in zip(frames["acq_id"], frames["frame_in_acq"])
        ],
        float,
    )


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
        check_raw: None | Callable[[list[int]], None] = None,
        excluded: Sequence[int] = (),
        approve: None | Callable[[list[int]], None] = None,
        final_ready: bool = False,
        raw_checked: bool = False,
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
        self.check_raw = check_raw
        self.approve = approve
        self.final_ready = final_ready  # the picked correction is saved
        self.raw_checked = raw_checked
        self.step = TESTS_STEP
        self.full_videos: list[Video] = []  # raw, corrected
        self.full_shown: None | int = None  # the step the full grid is set for
        self.excluded = [int(acq_id) for acq_id in excluded]  # in order
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

        # The steps, in the order of the work
        self.steps = QActionGroup(self)
        self.raw_action = self._action(
            self.left, "full-run", "Raw check", "Ctrl+1", checkable=True
        )
        self.tests_action = self._action(
            self.left, "tests", "Tests", "Ctrl+2", checkable=True
        )
        self.final_action = self._action(
            self.left, "final", "Final check", "Ctrl+3", checkable=True
        )

        for action in (self.raw_action, self.tests_action, self.final_action):
            self.steps.addAction(action)

        self.tests_action.setChecked(True)
        self.raw_action.setEnabled(False)
        self.final_action.setEnabled(False)
        self.raw_action.triggered.connect(lambda: self.show_step(RAW_STEP))
        self.tests_action.triggered.connect(lambda: self.show_step(TESTS_STEP))
        self.final_action.triggered.connect(lambda: self.show_step(FINAL_STEP))
        self._build_full_page()

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
        self.pages.setCurrentIndex(TESTS_PAGE)

        # The tools
        self.fit_action = self._action(self.right, "fit", "Fit (zoom out)", "0")
        self.fit_action.triggered.connect(lambda: self.current_grid().fit())
        self.pointer_action = self._action(
            self.right, "pointer", "Mirrored pointer", "H", checkable=True
        )
        self.pointer_action.setChecked(self.grid.view.mirror)
        self.pointer_action.toggled.connect(self.grid.set_mirror)
        self.pointer_action.toggled.connect(self.full_grid.set_mirror)
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
        self.back_action.triggered.connect(self._undo)
        self.exclude_action = self._action(
            self.right, "exclude", "Exclude the acquisition shown (again: keep it)", "E"
        )
        self.exclude_action.triggered.connect(self._toggle_exclusion)
        self.right.addSeparator()
        self.plots_action = self._action(
            self.right, "plots", "Plots of the focused runs", "P", checkable=True
        )
        self.plots_action.setChecked(True)
        self.plots_action.toggled.connect(self.plots.setVisible)
        self.plots_action.toggled.connect(self.full_plots.setVisible)
        self.report_action = self._action(
            self.right, "report", "Open the report of the focused run", "R"
        )
        self.report_action.triggered.connect(self._show_report)
        self._foot(self.right)
        self.pick_action = self._action(
            self.right,
            "pick",
            "Pick candidate",
            "Ctrl+Return",
        )
        self.pick_action.triggered.connect(self._pick)
        self.pick_action.setVisible(pick is not None)
        self.check_action = self._action(
            self.right,
            "check-raw",
            "Finished raw check: record the acquisitions excluded",
            "Ctrl+Return",
        )
        self.check_action.triggered.connect(self._check_raw)
        self.approve_action = self._action(
            self.right,
            "approve",
            "Approve: the corrected files are used from now on, the excluded "
            "acquisitions left out",
            "Ctrl+Return",
        )
        self.approve_action.triggered.connect(self._approve)

        # The transport, from the keyboard
        for key, do in (
            ("Space", lambda: self.current_grid().toggle()),
            ("Right", lambda: self.current_grid().step_by(1)),
            ("Left", lambda: self.current_grid().step_by(-1)),
            ("Shift+Right", lambda: self.current_grid().step_by(10)),
            ("Shift+Left", lambda: self.current_grid().step_by(-10)),
            ("]", lambda: self.current_grid().change_speed(1)),
            ("[", lambda: self.current_grid().change_speed(-1)),
        ):
            QShortcut(QKeySequence(key), self).activated.connect(do)

        full = QShortcut(QKeySequence(QKeySequence.StandardKey.FullScreen), self)
        full.activated.connect(self._toggle_full_screen)

        self.set_runs(runs)
        self.set_full()
        self.resize(1200, 800)

    def _approve(self) -> None:
        if self.approve is not None:
            self.approve(sorted(self.excluded))

    # ----------------------------------------------------------------------- #
    # The full run
    # ----------------------------------------------------------------------- #

    def _build_full_page(self) -> None:
        """Raw and corrected, every acquisition, with their plots."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 0, 8, 0)
        self.full_info = QLabel("")
        layout.addWidget(self.full_info)
        self.full_grid = VideoGrid()
        self.full_grid.frame_changed.connect(self._full_frame)
        self.full_plots = ShiftPlots()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.full_grid)
        split.addWidget(self.full_plots)
        split.setChildrenCollapsible(False)
        split.setSizes([600, 220])
        layout.addWidget(split, stretch=1)
        self.excluded_label = QLabel("")
        layout.addWidget(self.excluded_label)
        self.full_missing = QLabel("")
        layout.addWidget(self.full_missing)
        self.pages.insertWidget(FULL_PAGE, page)
        self.full_frames: None | pd.DataFrame = None  # the frame map
        self.full_x: None | np.ndarray = None
        self.full_data: None | dict[str, np.ndarray] = None

    def set_full(self) -> None:
        """
        Read the full run from `full/` in the local folder, when there, and
        open on the step the work is at: the raw check until it is done, the
        final check once the picked correction is saved, else the tests.
        """
        folder = self.folder / FULL_FOLDER
        raw = next(iter(_videos(folder, "raw")), None)
        corrected = next(iter(_videos(folder, "mcor")), None)
        there = raw is not None and corrected is not None
        self.raw_action.setEnabled(there)
        self.final_action.setEnabled(there and self.final_ready)

        if not there:
            return

        self.full_videos = [
            Video("raw", raw, candidate=False, color=RAW_COLOR, stream=True),
            Video("corrected", corrected, color=run_color(0), stream=True),
        ]
        self.full_shown = None
        self.full_frames = read_frame_map(frame_map_path(raw))
        data = read_summary(folder)
        self.full_data = data
        self.full_x = None

        if data is None:
            self.full_plots.set_series([])
            self.full_missing.setText(
                "No per-frame numbers to plot (written while correcting)"
            )
        else:
            self.full_plots.set_series([Series("corrected", run_color(0), data)])
            self.full_plots.show_runs(["corrected"])
            self.full_missing.setText("")
            self.full_x = frame_positions(self.full_frames, data)

        if self.final_ready:
            self.show_step(FINAL_STEP)
        elif not self.raw_checked:
            self.show_step(RAW_STEP)
        else:
            self.show_step(self.step)

        self._show_exclusions()

    def _shown_acquisition(self) -> None | int:
        """The acquisition of the full run frame shown."""
        frames, index = self.full_frames, self.full_grid.index

        if frames is None or index >= len(frames):
            return None

        return int(frames["acq_id"].iloc[index])

    def _full_frame(self, index: int) -> None:
        """Where the full run is: its acquisition, and the marker."""
        frames = self.full_frames
        where = ""
        acq_id = self._shown_acquisition()

        if acq_id is not None:
            order = list(dict.fromkeys(frames["acq_id"].tolist()))
            where = f"Acquisition {acq_id} ({order.index(acq_id) + 1} of {len(order)})"

            if acq_id in self.excluded:
                where += ", excluded"

        self.full_info.setText(
            "   ".join(part for part in (where, "Videos at half resolution") if part)
        )
        x = self.full_x
        self.full_plots.set_marker(
            float(x[index]) if x is not None and index < len(x) else None
        )

    def _toggle_exclusion(self) -> None:
        """Exclude the acquisition shown, or keep it again."""
        acq_id = self._shown_acquisition()

        if acq_id is None:
            return

        if acq_id in self.excluded:
            self.excluded.remove(acq_id)
        else:
            self.excluded.append(acq_id)

        self._show_exclusions()

    def _undo(self) -> None:
        """Cmd+Z: keep the last excluded (raw or final check), or bring back."""
        if self.step != TESTS_STEP:
            if self.excluded:
                self.excluded.pop()
                self._show_exclusions()
        else:
            self.grid.bring_back()

    def _show_exclusions(self) -> None:
        """The acquisitions excluded: in words, and shaded on the plots."""
        listed = ", ".join(str(acq_id) for acq_id in sorted(self.excluded))
        self.excluded_label.setText(f"Excluded: {listed}" if listed else "")
        data = self.full_data
        spans = []

        if data is not None:
            starts = np.concatenate([[0], np.cumsum(data["frame_counts"])])

            for k, acq_id in enumerate(data["acq_ids"].tolist()):
                if acq_id in self.excluded:
                    spans.append((float(starts[k]), float(starts[k + 1] - 1)))

        self.full_plots.set_excluded(spans)
        self._full_frame(self.full_grid.index)

        self._enable()

    def _check_raw(self) -> None:
        if self.check_raw is not None:
            self.check_raw(sorted(self.excluded))

    def current_grid(self) -> VideoGrid:
        """The grid of the page shown."""
        return self.full_grid if self.pages.currentIndex() == FULL_PAGE else self.grid

    def show_step(self, step: int) -> None:
        """
        Show `step`: the raw check (the raw video alone), the tests, or the
        final check (raw | corrected), the last two on the same page.
        """
        for grid in (self.grid, self.full_grid):
            grid.pause()

        if step != TESTS_STEP and not self.full_videos:
            step = TESTS_STEP

        if step != TESTS_STEP and step != self.full_shown:
            self.full_shown = step
            videos = self.full_videos[:1] if step == RAW_STEP else self.full_videos
            self.full_grid.set_videos(videos, columns=len(videos))

        self.step = step
        self.pages.setCurrentIndex(TESTS_PAGE if step == TESTS_STEP else FULL_PAGE)
        {
            RAW_STEP: self.raw_action,
            TESTS_STEP: self.tests_action,
            FINAL_STEP: self.final_action,
        }[step].setChecked(True)
        self._enable()

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
        action.setToolTip(self._tip(title, key))
        action.setCheckable(checkable)
        bar.addAction(action)
        button = bar.widgetForAction(action)

        if isinstance(button, QToolButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        return action

    @staticmethod
    def _tip(title: str, key: str) -> str:
        """A tooltip naming its shortcut, as this system writes it."""
        native = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)

        return f"{title} ({native})"

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
            self.frame_x = frame_positions(frames, series[0].data)

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

    def set_status(self, text: str, final_ready: None | bool = None) -> None:
        """
        Show the experiment status (see `pane_qc.my_experiments`), and
        whether its picked correction is saved (Approve then, not the raw
        check), when given.
        """
        self.state.setText(f"Status: {text}")

        if final_ready is not None and final_ready != self.final_ready:
            self.final_ready = final_ready
            self.final_action.setEnabled(final_ready and bool(self.full_videos))
            self._enable()

    def raw_check_done(self) -> None:
        """The raw check is recorded: on to the tests."""
        self.raw_checked = True
        self.show_step(TESTS_STEP)

    def final_check(self) -> None:
        """The picked correction is saved: on to the final check."""
        self.final_ready = True
        self.final_action.setEnabled(bool(self.full_videos))
        self.show_step(FINAL_STEP)

    def _show_details(self, video: None | Video = None) -> None:
        """The numbers of the run under the pointer, else the one focused."""
        self.hovered = video
        focused = self.grid.focused
        shown = video or (focused[0] if len(focused) == 1 else None)
        run = None if shown is None else self.runs.get(shown.name)
        self.details.setText("" if run is None else details_line(run))

    def _enable(self, *_) -> None:
        tests = self.step == TESTS_STEP
        full = not tests  # the raw or the final check

        for action in (
            self.only_action,
            self.clear_action,
            self.star_action,
            self.aside_action,
            self.back_action,
            self.report_action,
            self.pick_action,
        ):
            action.setEnabled(tests)

        # The foot of the bar: each page's own last step
        self.pick_action.setVisible(tests and self.pick is not None)
        # Each check's own last step: the raw check, or Approve (with the
        # same exclusions)
        checking = self.step == RAW_STEP and self.check_raw is not None
        approving = self.step == FINAL_STEP and self.approve is not None
        self.check_action.setVisible(checking)
        self.check_action.setEnabled(checking)
        self.approve_action.setVisible(approving)
        self.approve_action.setEnabled(approving)
        self.exclude_action.setEnabled(full and self.full_frames is not None)

        if not tests:
            self.back_action.setEnabled(full and bool(self.excluded))
            self.back_action.setToolTip(self._tip("Keep the last excluded", "Ctrl+Z"))
            return

        self.back_action.setToolTip(
            self._tip("Bring back the last set aside", "Ctrl+Z")
        )

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
        for grid in (self.grid, self.full_grid):
            grid.pause()
            grid.stop_reading()
            grid.close_streams()

        self.closed.emit()
        super().closeEvent(event)
