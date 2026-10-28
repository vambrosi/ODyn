"""`odyn.app.player`: videos side by side, playing in step."""

import time

import numpy as np
import pytest

from PySide6.QtCore import QEventLoop, QPointF, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication

from odyn.app.player import Video, VideoGrid, read_frames


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def a_video(path, frames=6, height=40, width=60):
    """
    Frame i is a flat gray of 20 * (i + 1) (round 255), so frames can be told
    apart.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height), isColor=False
    )

    for i in range(frames):
        writer.write(np.full((height, width), 20 * (i + 1) % 256, np.uint8))

    writer.release()

    return path


def loaded(grid, timeout_s=10):
    deadline = time.monotonic() + timeout_s

    while grid.reader is not None and grid.reader.isRunning():
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

        if time.monotonic() > deadline:
            raise TimeoutError("the videos were not read in time")

    QApplication.processEvents()


def test_a_video_is_read_whole_and_gray(tmp_path):
    frames = read_frames(a_video(tmp_path / "a.avi", frames=4))

    assert frames.shape == (4, 40, 60) and frames.dtype == np.uint8
    assert abs(int(frames[2].mean()) - 60) <= 3

    with pytest.raises(OSError):
        read_frames(tmp_path / "missing.avi")


def test_the_videos_play_in_step(app, tmp_path):
    grid = VideoGrid()
    grid.set_videos(
        [
            Video("raw", a_video(tmp_path / "raw.avi", frames=6)),
            Video("rigid", a_video(tmp_path / "rigid.avi", frames=6), "corr 0.9"),
            Video("short", a_video(tmp_path / "short.avi", frames=4)),
        ]
    )
    loaded(grid)

    # Together, up to the shortest
    assert grid.frame_count == 4 and grid.columns.value() == 2
    grid.seek(2)
    assert [tile.index for tile in grid.tiles] == [2, 2, 2]
    grid.seek(10)
    assert grid.index == 3 and grid.position.text() == "4 / 4"

    # Playing goes round
    grid.played_from = grid.index
    grid.speed.setCurrentIndex(0)
    assert grid.frame_at(100) == 0  # 10 frames/s: one frame later
    grid.play()
    assert grid.timer.isActive() and grid.play_button.text() == "Pause"
    grid.toggle()
    assert not grid.timer.isActive()
    grid.close()


def test_zoom_and_pan_follow_on_every_tile(app, tmp_path):
    grid = VideoGrid()
    grid.set_videos([Video(n, a_video(tmp_path / f"{n}.avi")) for n in "ab"])
    loaded(grid)
    first, second = grid.tiles

    for tile in (first, second):
        tile.resize(186, 144)  # inside: 180 x 120, the frame's shape (60 x 40)

    first.zoomed.emit(4.0, QPointF(0.25, 0.25))
    assert second.view.zoom == 4.0
    assert first.source_rect() == second.source_rect()
    assert first.source_rect().width() == pytest.approx(15)

    # Zoomed in, a tile of another shape is filled: the part shown takes its
    # shape, as far as the frame goes
    for tile in (first, second):
        tile.resize(366, 144)  # inside: 360 x 120, three times wider than high

    shown = first.source_rect()
    assert (shown.width(), shown.height()) == (pytest.approx(30), pytest.approx(10))
    assert first.target_rect().width() == pytest.approx(360)

    first.panned.emit(QPointF(-1, -1))  # as far as it goes: the corner
    assert second.source_rect().topLeft().x() == pytest.approx(0)

    grid.fit()
    assert second.source_rect().width() == pytest.approx(60)  # whole, at 1
    grid.close()


def test_the_pointer_shows_on_every_other_tile(app, tmp_path):
    grid = VideoGrid()
    grid.set_videos([Video(n, a_video(tmp_path / f"{n}.avi")) for n in "ab"])
    loaded(grid)
    first, second = grid.tiles

    for tile in (first, second):
        tile.resize(186, 144)

    first.hovered.emit(QPointF(0.5, 0.25))
    assert grid.view.pointer == QPointF(0.5, 0.25)
    assert grid.view.pointer_tile is first
    at = second.tile_point(grid.view.pointer)
    target = second.target_rect()
    assert at.x() == pytest.approx(target.x() + 90)
    assert at.y() == pytest.approx(target.y() + 30)

    # Leaving a tile after entering the next keeps the next one's
    second.hovered.emit(QPointF(0.1, 0.1))
    first.hovered.emit(None)
    assert grid.view.pointer_tile is second
    second.hovered.emit(None)
    assert grid.view.pointer is None

    # Off, and kept off for the next videos
    grid.set_mirror(False)
    grid.set_videos([Video("c", a_video(tmp_path / "c.avi"))])
    assert grid.view.mirror is False
    grid.close()


def test_playing_keeps_time_whatever_the_ticks(app, tmp_path):
    from odyn.app.player import SPEEDS

    grid = VideoGrid()
    assert SPEEDS == (10, 20, 30, 60, 120, 240)

    # A tick per frame, up to the screen's refresh
    grid.speed.setCurrentIndex(SPEEDS.index(30))
    assert grid.timer.interval() == 33
    grid.speed.setCurrentIndex(SPEEDS.index(240))
    assert grid.timer.interval() == 16

    # The frame shown is the one due by the clock: a late tick skips ahead
    grid.set_videos([Video("a", a_video(tmp_path / "a.avi", frames=60))])
    loaded(grid)
    grid.seek(5)
    grid.play()
    assert grid.played_from == 5
    assert grid.frame_at(100) == 5 + 24  # 240 frames/s, 0.1 s
    assert grid.frame_at(1000) == (5 + 240) % 60

    # The slider moved while playing: on from there
    grid.slider.setValue(40)
    assert grid.played_from == 40
    grid.close()


def test_tiles_are_focused_starred_and_set_aside(app, tmp_path):
    grid = VideoGrid()
    grid.set_videos(
        [Video("raw", a_video(tmp_path / "raw.avi"), candidate=False)]
        + [Video(n, a_video(tmp_path / f"{n}.avi")) for n in "abc"]
    )
    loaded(grid)
    grid.show()
    raw, a, b, c = grid.tiles

    def names(videos):
        return [video.name for video in videos]

    # A click focuses one; Shift adds or removes one
    a.clicked.emit(False)
    c.clicked.emit(True)
    assert names(grid.focused) == ["a", "c"]
    b.clicked.emit(False)
    assert names(grid.focused) == ["b"]
    c.clicked.emit(True)
    c.clicked.emit(True)
    assert names(grid.focused) == ["b"]

    # Focused only keeps those focused then, whatever the focus after
    a.clicked.emit(True)
    grid.show_focused_only(True)
    assert names(grid.visible()) == ["a", "b"]
    grid.clear_focus()
    assert grid.focused == [] and names(grid.visible()) == ["a", "b"]
    grid.show_focused_only(False)
    assert names(grid.visible()) == ["raw", "a", "b", "c"]

    # A star stays; starring all the starred unstars them
    a.clicked.emit(False)
    grid.star_focused()
    grid.clear_focus()
    assert names(grid.starred) == ["a"]
    a.clicked.emit(False)
    b.clicked.emit(True)
    grid.star_focused()
    assert names(grid.starred) == ["a", "b"]
    grid.star_focused()
    assert grid.starred == []

    # Select all: every tile shown, for every run on the plots
    grid.focus_all()
    assert names(grid.focused) == ["raw", "a", "b", "c"]
    grid.clear_focus()

    # Hidden, in order, out of the grid and into the strip; the raw too
    c.clicked.emit(False)
    grid.set_aside_focused()
    a.clicked.emit(False)
    raw.clicked.emit(True)
    grid.set_aside_focused()
    assert names(grid.set_aside) == ["c", "raw", "a"]  # at once: grid order
    assert (c.aside_rank, a.aside_rank) == (1, 3)
    assert grid.layout_grid.indexOf(c) == -1 and grid.layout_strip.indexOf(c) == 0
    assert grid.strip.isVisible()
    assert grid.focused == []

    # Select all leaves the hidden ones hidden
    grid.focus_all()
    assert names(grid.focused) == ["b"]
    grid.clear_focus()

    # Brought back: the last one, or one clicked in the strip
    grid.bring_back()
    grid.bring_back()
    assert names(grid.set_aside) == ["c"] and a.aside_rank is None
    c.clicked.emit(False)
    assert grid.set_aside == [] and not grid.strip.isVisible()
    grid.close()


def test_a_runs_tile_shows_its_numbers_and_its_older_videos(tmp_path):
    from odyn.app.mcor_window import Run, numbers_line, raw_video, test_video

    run = Run("s128", tmp_path, corr_with_ref=0.91234, flow_px=0.5, at_limit=0.0)
    assert numbers_line(run) == "corr 0.912  flow 0.50 px"
    assert numbers_line(Run("x", tmp_path, jitter_px=float("nan"))) == ""
    assert numbers_line(Run("x", tmp_path, at_limit=0.012)) == "at limit 1.2%"

    # The flow typical and worst, and in words under the grid
    from odyn.app.mcor_window import details_line

    run = Run("s128", tmp_path, 0.91, 0.4, flow_worst_px=2.1, flow_worst_acq=17)
    assert numbers_line(run) == "corr 0.910  flow 0.40 / 2.10 px"
    assert details_line(run) == (
        "s128: correlation with the reference 0.910, residual flow 0.40 px "
        "typical, 2.10 px worst (acquisition 17)"
    )
    assert details_line(Run("raw", tmp_path)) == "raw"

    # Folders from before the raw video was apart: raw | test, and the raw
    # video in the sweep's baseline
    (tmp_path / "rigid").mkdir()
    (tmp_path / "rigid" / "e1_raw.avi").write_bytes(b"")
    (tmp_path / "e1_raw_test.avi").write_bytes(b"")
    assert raw_video(tmp_path) == tmp_path / "rigid" / "e1_raw.avi"
    assert test_video(tmp_path) == tmp_path / "e1_raw_test.avi"

    # Now: the raw video in `raw/`, and the test alone first
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "e1_raw.mp4").write_bytes(b"")
    (tmp_path / "e1_test.avi").write_bytes(b"")
    assert raw_video(tmp_path) == tmp_path / "raw" / "e1_raw.mp4"
    assert test_video(tmp_path) == tmp_path / "e1_test.avi"
    assert test_video(tmp_path / "missing") is None


def test_a_narrow_tile_leaves_its_numbers_out(app, tmp_path):
    from odyn.app.player import Video, VideoTile, View

    tile = VideoTile(
        Video("s128_o64_d6", tmp_path / "x.avi", "corr 0.912  flow 0.52 px"), View()
    )
    tile.resize(600, 300)
    assert tile.numbers_fit()
    tile.resize(150, 150)
    assert not tile.numbers_fit()
    assert tile.toolTip() == ""  # nothing pops up over the videos
    assert not VideoTile(Video("raw", tmp_path / "x.avi"), View()).numbers_fit()


def test_the_qc_window_looks_as_the_main_window(app, tmp_path):
    from odyn.app import style
    from odyn.app.mcor_window import McorWindow

    window = McorWindow("e1", tmp_path, [], print)
    assert window.styleSheet() == style.window_stylesheet()

    # The steps at the left; without a full run, only the tests
    steps = [a for a in window.left.actions() if not a.isSeparator()]
    assert [a.toolTip() for a in steps] == [
        f"{name} ({QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)})"
        for name, key in (
            ("Raw check", "Ctrl+1"),
            ("Tests", "Ctrl+2"),
            ("Final check", "Ctrl+3"),
        )
    ]
    assert [a.isEnabled() for a in steps] == [False, True, False]
    assert window.tests_action.isChecked()

    # Every tool says its key
    tools = [a for a in window.right.actions() if a.toolTip()]
    assert all(a.toolTip().endswith(")") for a in tools)
    assert window.fit_action.shortcut() == QKeySequence("0")
    window.close()


def test_the_qc_window_is_driven_from_the_keyboard(app, tmp_path):
    from PySide6.QtTest import QTest

    from odyn.app.mcor_window import McorWindow, Run

    a_video(tmp_path / "raw" / "e1_raw.avi", frames=30)
    a_video(tmp_path / "e1_test.avi", frames=30)
    window = McorWindow("e1", tmp_path, [Run("test", tmp_path)], print)
    window.show()
    window.activateWindow()
    grid = window.grid
    loaded(grid)

    QTest.keyClick(window, Qt.Key.Key_Right)
    assert grid.index == 1
    QTest.keyClick(window, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert grid.index == 11
    QTest.keyClick(window, Qt.Key.Key_Left)
    assert grid.index == 10
    assert grid.speed.currentText() == "240 frames/s"  # the fastest
    QTest.keyClick(window, Qt.Key.Key_BracketLeft)
    assert grid.speed.currentText() == "120 frames/s"
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert grid.timer.isActive()
    QTest.keyClick(window, Qt.Key.Key_Space)
    assert not grid.timer.isActive()

    # Show selected and Clear selection follow the selection; Show all
    # the view; none stays pressed
    actions = (window.only_action, window.all_action, window.clear_action)
    assert not any(action.isCheckable() for action in actions)
    assert not any(action.isEnabled() for action in actions)
    grid.tiles[1].clicked.emit(False)
    assert window.report_action.isEnabled() is False  # no plot in this folder
    QTest.keyClick(window, Qt.Key.Key_F)
    assert [v.name for v in grid.visible()] == ["test"]
    assert window.all_action.isEnabled()
    QTest.keyClick(window, Qt.Key.Key_Escape)  # the view stays
    assert grid.focused == [] and not window.only_action.isEnabled()
    assert [v.name for v in grid.visible()] == ["test"]
    QTest.keyClick(window, Qt.Key.Key_F, Qt.KeyboardModifier.ShiftModifier)
    assert len(grid.visible()) == 2 and not window.all_action.isEnabled()

    # A double click shows its tile alone; Show selected again narrows
    grid.tiles[0].double_clicked.emit()
    assert [v.name for v in grid.focused] == ["raw"]
    assert [v.name for v in grid.visible()] == ["raw"]
    window.all_action.trigger()
    window.select_action.trigger()
    window.only_action.trigger()
    grid.tiles[1].clicked.emit(False)
    window.only_action.trigger()
    assert [v.name for v in grid.visible()] == ["test"]
    window.all_action.trigger()
    window.clear_action.trigger()

    # Star, hide and undo; the raw video hides too
    grid.tiles[1].clicked.emit(False)
    QTest.keyClick(window, Qt.Key.Key_S)
    assert [v.name for v in grid.starred] == ["test"]
    grid.tiles[0].clicked.emit(False)
    assert window.aside_action.isEnabled()
    grid.tiles[1].clicked.emit(False)
    QTest.keyClick(window, Qt.Key.Key_X)
    assert [v.name for v in grid.set_aside] == ["test"]
    assert window.back_action.toolTip().startswith("Undo last hide")
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert grid.set_aside == []

    # Select all
    window.select_action.trigger()
    assert len(grid.focused) == 2

    # The mirrored pointer starts off
    assert grid.view.mirror is False
    QTest.keyClick(window, Qt.Key.Key_H)
    assert grid.view.mirror is True
    window.close()


def test_the_plots_show_the_focused_runs_on_shared_axes(app, tmp_path):
    from odyn.app.mcor_window import McorWindow, Run
    from odyn.app.plots import RAW_COLOR
    from odyn.groups import frame_map_path, write_frame_map
    from odyn.mcor_report import write_shifts_summary

    # Two acquisitions of 100 frames; the videos keep 3 of each
    raw = a_video(tmp_path / "raw" / "e1_raw.avi")
    write_frame_map(frame_map_path(raw), [4, 4, 4, 9, 9, 9], [0, 50, 99] * 2)
    runs = []

    for name, size in (("rigid", 1.0), ("s128", 3.0)):
        folder = tmp_path / name
        a_video(folder / "e1_test.avi")
        shifts = np.stack([np.linspace(0, size, 200)] * 2, axis=1)
        patches = None if name == "rigid" else (np.ones((200, 4)),) * 2
        write_shifts_summary(
            folder / "shifts_summary.npz",
            shifts,
            patches,
            [0.3] * 200,
            [0.8] * 200,
            [4, 9],
            [100, 100],
        )
        runs.append(Run(name, folder))

    a_video(tmp_path / "old" / "e1_test.avi")  # from before: no numbers
    runs.append(Run("old", tmp_path / "old"))
    window = McorWindow("e1", tmp_path, runs, print)
    plots = window.plots
    assert [s.name for s in plots.series] == ["rigid", "s128"]
    assert "older tests): old" in window.missing.text()

    # Shared ranges, over every run loaded
    assert plots.ranges["dy"][1] > 3.0 and plots.ranges["dy"][0] < 0.0
    assert plots.edges.tolist() == [0, 100]
    assert plots.ranges["corr"][0] == 0  # the correlation axis starts at 0

    # Only the focused runs drawn; the marker follows the video frame
    assert plots.shown == []
    window.grid.tiles[2].clicked.emit(False)
    assert plots.shown == ["s128"]
    loaded(window.grid)
    window.grid.seek(4)  # acquisition 9, its frame 50
    assert plots.marker == 150
    assert window.grid.tiles[1].video.color is not None
    assert window.grid.tiles[0].video.color == RAW_COLOR  # as its gray line

    # One line of numbers: the run under the pointer, else the focused one
    assert window.details.text() == "s128"
    window.grid.tiles[1].hovered.emit(QPointF(0.5, 0.5))
    assert window.details.text() == "rigid"
    window.grid.tiles[1].hovered.emit(None)
    assert window.details.text() == "s128"
    window.grid.tiles[1].clicked.emit(True)
    assert window.details.text() == ""  # two focused, none hovered

    # Drawn (the picture is kept once drawn whole), marker and all
    plots.resize(400, 200)
    image = plots.grab().toImage()
    assert plots._picture is not None
    marker = plots._x(plots._rows()[0], plots.marker)
    assert image.pixelColor(int(marker), 30).alpha() > 0

    window.plots_action.trigger()
    assert not plots.isVisibleTo(window)
    window.close()


def test_the_full_run_page_streams_every_acquisition(app, tmp_path):
    from odyn.app.mcor_window import FULL_PAGE, TESTS_PAGE, McorWindow
    from odyn.app.player import FrameStream
    from odyn.groups import frame_map_path, write_frame_map
    from odyn.mcor_report import write_shifts_summary

    full = tmp_path / "full"
    raw = a_video(full / "G_e1_raw.avi", frames=6)
    a_video(full / "G_e1_mcor.avi", frames=6)
    write_frame_map(frame_map_path(raw), [4, 4, 4, 9, 9, 9], [0, 50, 99] * 2)
    write_shifts_summary(
        full / "shifts_summary.npz",
        np.zeros((200, 2)),
        None,
        [0.3] * 200,
        [0.8] * 200,
        [4, 9],
        [100, 100],
    )

    # Opened on the raw check, as it comes first: the raw video alone
    window = McorWindow("e1", tmp_path, [], print)
    assert window.raw_action.isEnabled() and window.raw_action.isChecked()
    assert not window.final_action.isEnabled()
    assert window.pages.currentIndex() == FULL_PAGE
    grid = window.current_grid()
    assert grid is window.full_grid
    assert [tile.video.name for tile in grid.tiles] == ["raw"]
    loaded(grid)

    # Read from the file as shown, not held
    assert isinstance(grid.tiles[0].video.frames, FrameStream)
    assert grid.frame_count == 6
    grid.seek(4)
    assert window.full_info.text().startswith("Acquisition 9 (2 of 2)")
    assert "half resolution" in window.full_info.text()
    assert window.full_plots.marker == 150
    assert window.full_plots.shown == ["corrected"]

    # The same selection tools as the tests; the tests' own greyed
    assert not window.report_action.isEnabled()
    assert not window.pick_action.isVisible()
    grid.tiles[0].double_clicked.emit()
    assert window.star_action.isEnabled() and window.aside_action.isEnabled()
    assert window.all_action.isEnabled()
    window.all_action.trigger()

    # A click on the plots goes to the frame shown nearest there
    plots = window.full_plots
    plots.resize(400, 200)
    rect = plots._rows()[0]
    assert plots.frame_at(rect.left()) == 0
    assert plots.frame_at(rect.left() - 10) is None
    plots.frame_clicked.emit(160.0)  # acquisition 9, past frame 50 of it
    assert grid.index == 4
    plots.frame_clicked.emit(0.0)
    assert grid.index == 0

    window.tests_action.trigger()
    assert window.pages.currentIndex() == TESTS_PAGE
    assert window.current_grid() is window.grid
    window.close()


def test_without_a_full_run_the_window_opens_on_the_tests(app, tmp_path):
    from odyn.app.mcor_window import TESTS_PAGE, McorWindow

    window = McorWindow("e1", tmp_path, [], print)
    assert not window.raw_action.isEnabled()
    assert window.pages.currentIndex() == TESTS_PAGE
    window.close()


def test_acquisitions_are_excluded_on_the_full_run(app, tmp_path):
    from PySide6.QtTest import QTest

    from odyn.app.mcor_window import McorWindow
    from odyn.groups import frame_map_path, write_frame_map
    from odyn.mcor_report import write_shifts_summary

    full = tmp_path / "full"
    raw = a_video(full / "G_e1_raw.avi", frames=9)
    a_video(full / "G_e1_mcor.avi", frames=9)
    write_frame_map(frame_map_path(raw), [4] * 3 + [9] * 3 + [12] * 3, [0, 1, 2] * 3)
    write_shifts_summary(
        full / "shifts_summary.npz",
        np.zeros((30, 2)),
        None,
        [],
        [],
        [4, 9, 12],
        [10] * 3,
    )
    checked = []

    # Opened with the exclusions recorded before
    window = McorWindow(
        "e1", tmp_path, [], print, check_raw=checked.append, excluded=[12]
    )
    window.show()
    window.activateWindow()
    loaded(window.full_grid)
    assert window.excluded_label.text() == "Excluded: 12"
    assert window.full_plots.excluded == [(20.0, 29.0)]
    assert window.check_action.isVisible() and not window.pick_action.isVisible()

    # E excludes the acquisition shown, again keeps it; Cmd+Z undoes
    window.full_grid.seek(4)  # acquisition 9
    QTest.keyClick(window, Qt.Key.Key_E)
    assert window.excluded == [12, 9]
    assert "Acquisition 9 (2 of 3), excluded" in window.full_info.text()
    QTest.keyClick(window, Qt.Key.Key_E)
    assert window.excluded == [12]
    QTest.keyClick(window, Qt.Key.Key_E)
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.excluded == [12]
    QTest.keyClick(window, Qt.Key.Key_E)

    # Finished: what is excluded goes to the record, sorted; once recorded,
    # on to the tests
    window.check_action.trigger()
    assert checked == [[9, 12]]
    window.raw_check_done()
    assert window.tests_action.isChecked() and window.current_grid() is window.grid
    window.close()


def test_the_final_check_shows_raw_and_corrected_and_approves(app, tmp_path):
    from odyn.app.mcor_window import McorWindow
    from odyn.groups import frame_map_path, write_frame_map

    full = tmp_path / "full"
    raw = a_video(full / "G_e1_raw.avi", frames=6)
    a_video(full / "G_e1_mcor.avi", frames=6)
    write_frame_map(frame_map_path(raw), [4] * 3 + [9] * 3, [0, 1, 2] * 2)
    checked, approved = [], []
    window = McorWindow(
        "e1",
        tmp_path,
        [],
        print,
        check_raw=checked.append,
        excluded=[9],
        approve=approved.append,
        raw_checked=True,
    )

    # Raw check done, nothing saved yet: the tests, no final check
    assert window.tests_action.isChecked()
    assert window.raw_action.isEnabled() and not window.final_action.isEnabled()

    # Back to the raw check: its own button, not Approve
    window.raw_action.trigger()
    assert window.check_action.isVisible() and not window.approve_action.isVisible()

    # The picked correction saved: raw | corrected, Approve, same exclusions
    window.final_check()
    assert window.final_action.isEnabled() and window.final_action.isChecked()
    tiles = window.full_grid.tiles
    assert [tile.video.name for tile in tiles] == ["raw", "corrected"]
    assert window.approve_action.isVisible() and not window.check_action.isVisible()
    assert window.exclude_action.isEnabled()

    # The raw video hides here too; Cmd+Z undoes the last of hide or exclude
    window.full_grid.tiles[0].clicked.emit(False)
    window.aside_action.trigger()
    assert [v.name for v in window.full_grid.set_aside] == ["raw"]
    window.full_grid.seek(0)  # acquisition 4
    window.exclude_action.trigger()
    assert window.back_action.toolTip().startswith("Undo last exclusion")
    window.back_action.trigger()
    assert window.excluded == [9]
    assert window.back_action.toolTip().startswith("Undo last hide")
    window.back_action.trigger()
    assert window.full_grid.set_aside == [] and not window.back_action.isEnabled()

    window.approve_action.trigger()
    assert approved == [[9]] and checked == []
    window.close()

    # Opened once saved: on the final check
    window = McorWindow("e1", tmp_path, [], print, final_ready=True)
    assert window.final_action.isChecked()
    assert len(window.full_grid.tiles) == 2
    window.close()


def test_the_saved_files_open_from_the_window(app, tmp_path, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    from odyn.app.mcor_window import McorWindow
    from odyn.app.pane_qc import saved_folder

    folder = saved_folder(tmp_path, "20260304/m1/e1, 20260304/m1/e2")
    assert folder == tmp_path / "20260304" / "m1" / "e1" / "processed" / "mcor"

    # Not saved yet: greyed
    window = McorWindow("e1", tmp_path, [], print, saved_folder=folder)
    assert not window.saved_action.isEnabled()
    window.close()

    folder.mkdir(parents=True)
    opened = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: opened.append(url) or True
    )
    window = McorWindow("e1", tmp_path, [], print, saved_folder=folder)
    assert window.saved_action.isEnabled()
    assert window.saved_action.shortcut().toString() == "O"
    window.saved_action.trigger()
    assert [url.toLocalFile() for url in opened] == [str(folder)]
    window.close()
