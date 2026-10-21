"""`odyn.app.player`: videos side by side, playing in step."""

import time

import numpy as np
import pytest

from PySide6.QtCore import QEventLoop, QPointF
from PySide6.QtWidgets import QApplication

from odyn.app.player import Video, VideoGrid, read_frames


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def a_video(path, frames=6, height=40, width=60):
    """Frame i is a flat gray of 20 * (i + 1), so frames can be told apart."""
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height), isColor=False
    )

    for i in range(frames):
        writer.write(np.full((height, width), 20 * (i + 1), np.uint8))

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
    grid._step()
    assert grid.index == 0
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

    first.zoomed.emit(4.0, QPointF(0.25, 0.25))
    assert second.view.zoom == 4.0
    assert first.source_rect() == second.source_rect()
    assert first.source_rect().width() == pytest.approx(15)

    first.panned.emit(QPointF(-1, -1))  # as far as it goes: the corner
    assert second.source_rect().topLeft().x() == pytest.approx(0)

    grid.fit()
    assert second.source_rect().width() == pytest.approx(60)
    grid.close()


def test_selected_tiles_can_be_shown_alone(app, tmp_path):
    grid = VideoGrid()
    grid.set_videos([Video(n, a_video(tmp_path / f"{n}.avi")) for n in "abc"])
    loaded(grid)
    grid.show()

    assert not grid.show_selected.isEnabled()
    grid.tiles[0].clicked.emit()
    grid.tiles[2].clicked.emit()
    assert [v.name for v in grid.selected] == ["a", "c"]

    grid.show_selected.click()
    assert [v.name for v in grid.visible()] == ["a", "c"]
    assert grid.show_all.isEnabled()

    grid.tiles[2].clicked.emit()  # let one go
    grid.show_all.click()
    assert [v.name for v in grid.visible()] == ["a", "b", "c"]
    assert [v.name for v in grid.selected] == ["a"]
    assert grid.show_selected.text() == "Show selected (1)"
    assert grid.position.text().endswith(f"/ {grid.frame_count}")  # not the 1

    # All let go at once, and every tile shown again
    grid.tiles[1].clicked.emit()
    grid.show_selected.click()
    assert grid.deselect_all.isEnabled()
    grid.deselect_all.click()
    assert grid.selected == [] and len(grid.visible()) == 3
    assert not grid.deselect_all.isEnabled() and not grid.show_all.isEnabled()
    assert grid.show_selected.text() == "Show selected"
    grid.close()


def test_a_runs_tile_shows_its_numbers_and_its_older_videos(tmp_path):
    from odyn.app.mcor_window import Run, numbers_line, raw_video, test_video

    run = Run("s128", tmp_path, corr_with_ref=0.91234, flow_px=0.5, at_limit=0.0)
    assert numbers_line(run) == "corr 0.912  flow 0.50 px"
    assert numbers_line(Run("x", tmp_path, jitter_px=float("nan"))) == ""
    assert numbers_line(Run("x", tmp_path, at_limit=0.012)) == "at limit 1.2%"

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
    assert "corr 0.912" in tile.toolTip()  # still there, under the pointer
    assert not VideoTile(Video("raw", tmp_path / "x.avi"), View()).numbers_fit()


def test_the_qc_window_looks_as_the_main_window(app, tmp_path):
    from odyn.app import style
    from odyn.app.mcor_window import McorWindow

    window = McorWindow("e1", tmp_path, [], print)
    assert window.styleSheet() == style.window_stylesheet()
    window.close()
