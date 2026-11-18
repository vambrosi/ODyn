"""
`odyn.app.curation_window`, offscreen: the ROIs of a group curated on
candidates drawn by hand (`test_segmentation.drawn_candidates`), with clicks,
sliders and keys, undo, the card of the selected cell, the shortcuts card,
and the draft kept on this computer. Opening it from the QC page is tested in
`test_app_gui.py`.
"""

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Needs Qt and the system libraries it loads: skipped where either is missing
try:
    from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
    from PySide6.QtGui import QKeySequence, QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel, QToolBar

except ImportError as error:  # a missing library, not a missing module
    pytest.skip(f"Qt cannot load here: {error}", allow_module_level=True)

from odyn.app.curation_window import (  # noqa: E402
    COLORS,
    LEGEND,
    CurationWindow,
    counts_text,
    outline,
    size_at,
    size_position,
)
from odyn.segmentation import Edits  # noqa: E402
from test_segmentation import drawn_candidates, drawn_image  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def files(folder, found=None):
    """
    Candidates (drawn ones by default) and a mean image like them, with a
    spot where no basin is (`drawn_image`).
    """
    found = found or drawn_candidates()
    found.write(folder / "cellpose_group_1.h5")
    tifffile.imwrite(folder / "mean_group_1.tif", drawn_image(found))

    return folder / "mean_group_1.tif", folder / "cellpose_group_1.h5"


def opened(app, folder, mean=None, candidates=None, submit=None) -> CurationWindow:
    """The window on the files in `folder`, once they are read."""
    if mean is None:
        mean, candidates = files(folder)

    window = CurationWindow(
        "20260304/m1/e1", 1, mean, candidates, folder / "draft.json", submit=submit
    )
    window.show()
    window.loader.wait()
    app.processEvents()  # what it read comes back through the event loop
    app.processEvents()

    return window


def press(window, key: str) -> None:
    """Press `key` as a person would: what it does is what its shortcut does."""
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window, 2000)
    QTest.keySequence(window, QKeySequence(key))


def area_applied(window) -> None:
    """
    Wait until the area range applies, once its steps pause: its timer, not
    a fixed time (other windows of earlier tests can keep the loop busy).
    """
    deadline = time.monotonic() + 5

    while window.ranging.isActive() and time.monotonic() < deadline:
        QTest.qWait(20)

    assert not window.ranging.isActive()


def on_canvas(window, basin: int) -> QRectF:
    """Where the pixels of the cell of `basin` are on the canvas."""
    rows, columns = np.divmod(window.curation.cells[basin].pixels, 44)
    canvas = window.canvas
    corner = canvas.canvas_point(rows.min(), columns.min())

    return QRectF(corner, canvas.canvas_point(rows.max() + 1, columns.max() + 1))


def kinds(window) -> list[str]:
    """What the outlines drawn are, by their colors (see `COLORS`)."""
    kinds = [kind.partition("-")[0] for _, _, kind in window.canvas.shapes]
    assert set(kinds) <= set(COLORS)

    return kinds


def test_an_outline_runs_along_the_edges_of_the_pixels(app):
    # A square of 2 x 2 pixels at rows 1-2, columns 3-4, in an image 10 wide
    path, box = outline(np.array([13, 14, 23, 24]), 10)
    assert box == QRectF(3, 1, 2, 2)
    assert path.boundingRect() == QRectF(3, 1, 2, 2)
    assert path.elementCount() == 8  # four sides, a line each

    # An L: two more sides, and runs join along an edge
    path, box = outline(np.array([13, 23, 24]), 10)
    assert box == QRectF(3, 1, 2, 2) and path.elementCount() == 12


def test_the_size_slider_is_logarithmic_from_one_pixel_to_the_basin():
    assert size_at(0, 500) == 1 and size_at(1000, 500) == 500
    assert size_at(500, 400) == 20  # halfway: the geometric middle
    assert all(size_at(size_position(n, 400), 400) == n for n in (1, 2, 20, 400))


def test_the_counts_leave_out_what_is_none():
    def cells(kept=(), excluded=()):
        """Stand-ins: so many cells of each kind kept, or of each reason not."""
        made = []

        for kind, count in kept:
            made += [SimpleNamespace(kind=kind, excluded_by=None, kept=True)] * count

        for reason, count in excluded:
            made += [
                SimpleNamespace(kind="cellpose", excluded_by=reason, kept=False)
            ] * count

        return made

    everything = cells(
        kept=[("cellpose", 536), ("resized", 2), ("added", 1), ("image", 1)],
        excluded=[("rejected", 1), ("area", 11), ("cellpose", 184)],
    )
    assert counts_text(everything) == (
        "540 cells kept (2 resized, 1 added from basin, 1 added from image) and "
        "196 excluded (1 rejected, 11 excluded by area, 184 excluded by Cellpose)"
    )
    assert counts_text(cells([("cellpose", 2)], [("rejected", 1)])) == (
        "2 cells kept and 1 excluded (1 rejected)"
    )
    assert counts_text(cells([("cellpose", 1)])) == "1 cell kept"
    assert counts_text([]) == "0 cells kept"


def test_the_window_starts_from_the_cells_cellpose_keeps(app, tmp_path):
    window = opened(app, tmp_path)

    # The dim spot, which fails the flow check, is there but hidden
    assert [cell.basin for cell in window.curation.kept()] == [1, 2]
    assert window.curation.cells[3].dropped
    assert window.counts.text() == (
        "2 cells kept and 1 excluded (1 excluded by Cellpose)"
    )
    assert kinds(window) == ["cellpose", "cellpose"]
    assert window.threshold_line.text() == "0.00 (default)"
    assert (window.canvas.image.width(), window.canvas.image.height()) == (44, 30)
    assert not window.card.isVisible()  # nothing selected
    assert window.show_rejected.toolTip() == "Show rejected cells (R)"

    # A legend for every kind of outline, above the image
    legend = [label.text() for label in window.findChildren(QLabel)]
    assert all(text in legend for text in LEGEND.values())

    # The bar: undo and redo, what is shown, the image, and the shortcuts;
    # Submit at its foot (with nothing to hand the ROIs to, hidden)
    groups = [[]]
    bars = window.findChildren(QToolBar)
    bar = next(bar for bar in bars if bar.windowTitle() == "ROIs")

    for action in bar.actions():
        if action.isSeparator():
            groups.append([])
        elif action.text():  # not the room above Submit
            groups[-1].append(action.text())

    assert groups == [
        ["Undo", "Redo"],
        [
            "Show outlines",
            "Show rejected cells",
            "Show cells excluded by area",
            "Show cells excluded by Cellpose",
        ],
        ["Show whole image", "Brightness and gamma"],
        ["Shortcuts", "Submit the ROIs"],
    ]
    window.close()


def test_a_click_on_a_cell_excluded_by_cellpose_adds_it(app, tmp_path):
    window = opened(app, tmp_path)
    dim = window.curation.cells[3].pixels.copy()

    # Hidden: a click on it adds it as it is, and selects it
    window.click(24, 38)
    assert window.selected == 3 and window.curation.cells[3].kind == "added"
    np.testing.assert_array_equal(window.curation.cells[3].pixels, dim)
    assert window.line.text().startswith("Added the candidate under the point")

    # Shown: dashed, and the same by the mouse
    press(window, "Ctrl+Z")
    press(window, "C")
    assert kinds(window) == ["cellpose", "cellpose", "dropped"]
    at = window.canvas.canvas_point(24.5, 38.5).toPoint()
    QTest.mouseClick(
        window.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier(0), at
    )
    assert window.selected == 3 and window.curation.cells[3].kind == "added"
    assert kinds(window) == ["cellpose", "cellpose", "selected"]
    window.close()


def test_a_click_selects_a_cell_and_one_elsewhere_adds_the_candidate(app, tmp_path):
    window = opened(app, tmp_path)
    canvas = window.canvas

    # A click on cell 1 (the mouse itself, at its center pixel) selects it
    at = canvas.canvas_point(10.5, 10.5).toPoint()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier(0), at)
    assert window.selected == 1
    assert window.cell_line.text() == "Cell 1, threshold 0.00"
    assert window.cell_state.text() == ""  # of Cellpose, at the threshold
    assert kinds(window) == ["cellpose", "selected"]
    size = len(window.curation.cells[1].pixels)
    assert window.size_line.text() == f"{size} px, {size:.1f} µm²"  # 1 um/px

    # Again: let go
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier(0), at)
    assert window.selected is None

    # On the dim spot, which Cellpose dropped: added, and selected
    window.click(24, 41)
    assert window.selected == 3 and window.curation.cells[3].kind == "added"
    assert window.line.text().startswith("Added the candidate under the point")
    assert window.cell_state.text() == "added"

    # By the spot, where no basin is: a cell grown from the image
    window.click(4, 40)
    assert window.selected == 4 and window.curation.from_image(4)
    assert window.line.text().startswith("No Cellpose basin here: grew a cell")
    assert window.cell_line.text() == "Cell 4, from the image"
    assert window.cell_state.text() == ""
    assert not window.use_basin.isChecked() and not window.use_basin.isEnabled()
    assert not window.follow.isEnabled() and window.reject.text() == "Delete"
    assert kinds(window)[-1] == "selected" and "image" not in kinds(window)
    press(window, "Esc")
    assert kinds(window)[-1] == "image"
    window.close()


def test_b_switches_a_cell_between_its_basin_and_the_image(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(10, 10)
    press(window, "[")  # a size of its own
    before = len(window.curation.cells[1].pixels)
    assert window.use_basin.isChecked() and window.use_basin.isEnabled()

    press(window, "B")
    assert window.curation.from_image(1) and not window.use_basin.isChecked()
    assert window.cell_line.text() == "Cell 1, from the image"
    assert window.counts.text().startswith("2 cells kept (1 added from image)")
    assert window.reject.text() == "Reject"  # its Cellpose cell stays

    # Back with the button, as it was, and undone: from the image again
    window.use_basin.trigger()
    assert not window.curation.from_image(1) and window.use_basin.isChecked()
    assert len(window.curation.cells[1].pixels) == before
    assert window.cell_state.text() == "resized"
    press(window, "Ctrl+Z")
    assert window.curation.from_image(1) and not window.use_basin.isChecked()
    window.close()


def test_a_cell_grown_where_no_basin_is_is_deleted_not_rejected(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(4, 40)
    assert window.selected == 4 and window.reject.text() == "Delete"

    press(window, "Delete")
    assert 4 not in window.curation.cells and window.selected is None
    assert window.counts.text() == (
        "2 cells kept and 1 excluded (1 excluded by Cellpose)"
    )

    press(window, "Ctrl+Z")
    assert window.curation.from_image(4)
    window.close()


def test_a_change_that_takes_the_selected_cell_away_lets_it_go(app, tmp_path):
    window = opened(app, tmp_path)

    # Grown where no basin is (numbered above the basins), then undone
    window.click(4, 40)
    assert window.selected == 4
    press(window, "Ctrl+Z")
    assert window.selected is None and not window.card.isVisible()
    assert window.counts.text().startswith("2 cells kept")

    # Redone: back, but no longer selected
    press(window, "Ctrl+Shift+Z")
    assert window.curation.from_image(4) and window.selected is None

    # The dim spot, excluded by Cellpose: added, then undone, it is excluded
    # again and not selected
    window.click(24, 41)
    assert window.selected == 3
    press(window, "Ctrl+Z")
    assert window.selected is None and window.curation.cells[3].dropped
    window.close()


def test_the_selected_cell_is_resized_by_its_slider_and_keys(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(10, 10)
    start = len(window.curation.cells[1].pixels)

    height = window.card.height()
    press(window, "]")
    assert len(window.curation.cells[1].pixels) == start + 1
    assert window.curation.cells[1].kind == "resized"
    assert kinds(window) == ["cellpose", "selected"]
    assert window.cell_state.text() == "resized"
    assert window.card.height() == height  # its second line was there, empty

    window.size.setValue(window.size.value() + 100)  # larger, toward the basin
    larger = len(window.curation.cells[1].pixels)
    assert larger > start + 1
    assert window.size_line.text().startswith(f"{larger} px")

    # Outside the area range too
    window.min_area.setValue(larger + 1)
    area_applied(window)
    assert window.cell_state.text() == "resized and excluded by area"
    window.min_area.setValue(0)
    area_applied(window)

    # T: the threshold of all again
    press(window, "T")
    assert len(window.curation.cells[1].pixels) == start
    assert window.curation.cells[1].kind == "cellpose"
    assert window.cell_state.text() == ""
    window.close()


def test_the_threshold_slider_makes_every_cell_larger_to_the_right(app, tmp_path):
    window = opened(app, tmp_path)
    before = {b: len(c.pixels) for b, c in window.curation.cells.items()}

    window.threshold.setValue(window.threshold.value() + 20)  # 1.0 lower
    app.processEvents()
    assert window.threshold_line.text() == "-1.00"
    assert all(len(window.curation.cells[b].pixels) > n for b, n in before.items())

    # Undone, and done again
    press(window, "Ctrl+Z")
    assert window.threshold_line.text() == "0.00 (default)"
    assert {b: len(c.pixels) for b, c in window.curation.cells.items()} == before
    press(window, "Ctrl+Shift+Z")
    assert window.threshold_line.text() == "-1.00"
    window.close()


def test_the_cells_not_kept_hide_until_shown(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(10, 10)

    press(window, "Delete")
    assert window.curation.cells[1].rejected and window.reject.text() == "Restore"
    assert window.cell_state.text() == "rejected"
    assert kinds(window) == ["cellpose", "selected"]  # dashed while selected

    press(window, "Esc")
    assert kinds(window) == ["cellpose"]
    press(window, "R")
    assert window.show_rejected.isChecked()
    assert kinds(window) == ["rejected", "cellpose"]
    assert window.counts.text() == (
        "1 cell kept and 2 excluded (1 rejected, 1 excluded by Cellpose)"
    )

    # The area range: the other cell outside it, hidden until shown
    window.min_area.setValue(len(window.curation.cells[2].pixels) + 1)
    area_applied(window)
    assert window.curation.cells[2].excluded_by == "area"
    assert kinds(window) == ["rejected"]
    press(window, "E")
    assert kinds(window) == ["rejected", "excluded"]

    # The dim spot, excluded by Cellpose (and small): hidden until shown
    press(window, "C")
    assert window.show_dropped.isChecked()
    assert kinds(window) == ["rejected", "excluded", "dropped"]
    window.close()


def test_the_area_range_applies_as_its_arrows_step_it(app, tmp_path):
    window = opened(app, tmp_path)
    areas = {b: len(c.pixels) for b, c in window.curation.cells.items()}
    assert areas[1] > areas[2] + 5  # 1 um/px

    # Up a step from just under cell 2, in the box: cell 2 outside the range
    # once the steps pause, without leaving the box
    window.min_area.setValue(areas[2] - 1)
    area_applied(window)
    assert not window.curation.cells[2].outside
    window.min_area.setFocus()
    QTest.keyClick(window.min_area, Qt.Key.Key_Up)  # as its up arrow
    assert window.ranging.isActive() and not window.curation.cells[2].outside
    area_applied(window)
    assert window.min_area.hasFocus()
    assert window.curation.cells[2].outside and window.curation.cells[1].kept
    assert "1 excluded by area" in window.counts.text()

    # One step back undoes it
    press(window, "Ctrl+Z")
    assert not window.curation.cells[2].outside
    assert window.min_area.value() == areas[2] - 1
    window.close()


def test_the_card_of_the_selected_cell_leaves_it_in_view(app, tmp_path):
    window = opened(app, tmp_path)
    canvas, card = window.canvas, window.card
    middle = QPointF(canvas.width() / 2, canvas.height() / 2)

    # Cell 1, top left in the image: the card in the bottom right corner
    window.click(10, 10)
    assert card.isVisible() and canvas.rect().contains(card.geometry())
    assert card.geometry().topLeft().x() > middle.x()
    assert card.geometry().topLeft().y() > middle.y()
    assert not QRectF(card.geometry()).intersects(on_canvas(window, 1))

    # The dim spot, bottom right: the card goes to a corner clear of it
    window.click(24, 41)
    assert window.selected == 3 and canvas.rect().contains(card.geometry())
    assert not QRectF(card.geometry()).intersects(on_canvas(window, 3))

    # The wheel over the card resizes nothing (and zooms nothing)
    size, view = window.size.value(), canvas.source_rect()
    at = QPointF(window.size.rect().center())
    wheel = QWheelEvent(
        at,
        QPointF(window.size.mapToGlobal(at.toPoint())),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(window.size, wheel)
    assert window.size.value() == size and canvas.source_rect() == view

    # Nothing selected, no card
    press(window, "Esc")
    assert not card.isVisible()
    window.close()


def test_the_shortcuts_card_shows_in_the_middle_until_put_away(app, tmp_path):
    window = opened(app, tmp_path)
    canvas, card = window.canvas, window.help
    window.click(10, 10)

    press(window, "?")
    assert card.isVisible() and window.help_action.isChecked()
    middle = card.geometry().center() - canvas.rect().center()
    assert abs(middle.x()) <= 1 and abs(middle.y()) <= 1
    keys = [label.text() for label in card.findChildren(QLabel)]
    assert "Shortcuts" in keys and "Decrease cell area by 1 px" in keys

    # Esc puts it away first, then lets the cell go
    press(window, "Esc")
    assert not card.isVisible() and window.selected == 1
    press(window, "Esc")
    assert window.selected is None

    # A click on it puts it away too, and is not a click on the image
    window.help_action.trigger()
    assert card.isVisible()
    QTest.mouseClick(card, Qt.MouseButton.LeftButton)
    assert not card.isVisible() and not window.help_action.isChecked()
    assert window.selected is None and window.curation.edits == Edits()
    window.close()


def test_the_draft_is_kept_here_and_picked_up_again(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(24, 41)
    window.close()  # kept as it closes, without waiting

    again = opened(app, tmp_path, *files(tmp_path))
    assert again.curation.cells[3].kind == "added"
    assert again.line.text() == "Picked up your draft from where you left it."
    again.close()

    # Candidates made again: the draft is for the old ones
    found = drawn_candidates()
    found.peak[0] -= 1
    other = opened(app, tmp_path, *files(tmp_path, found))
    assert other.curation.cells[3].dropped
    assert other.line.text().startswith("Your draft was for other candidates")
    other.close()


def test_without_candidates_the_window_says_how_to_make_them(app, tmp_path):
    mean, candidates = tmp_path / "mean.tif", tmp_path / "cells.h5"
    window = opened(app, tmp_path, mean, candidates)
    assert window.curation is None
    assert "cellpose --group 1" in window.canvas.message
    window.close()

    files(tmp_path)
    window = opened(app, tmp_path, mean, tmp_path / "cellpose_group_1.h5")
    assert "mean-images --group 1" in window.canvas.message
    window.close()


def test_submit_hands_the_rois_as_they_stand_to_the_page(app, tmp_path):
    handed = []
    window = opened(app, tmp_path, submit=lambda *given: handed.append(given))
    assert window.submit_action.isVisible() and window.submit_action.isEnabled()

    window.click(24, 41)  # adds the cell Cellpose drops
    press(window, "Ctrl+Return")

    [(curated, counts)] = handed
    assert list(curated.table["basin"]) == [1, 2, 3]
    assert curated.edits == window.curation.edits
    assert counts == window.counts.text() == "3 cells kept (1 added from basin)"
    assert (tmp_path / "draft.json").is_file()  # kept first
    window.close()

    # With nothing kept, nothing to submit; with no page to take them, no
    # Submit at all
    window = opened(app, tmp_path, submit=lambda *given: handed.append(given))
    window.curation.edits = Edits(rejected={1, 2})
    window.changed()
    assert not window.submit_action.isEnabled()
    window.close()
    window = opened(app, tmp_path)
    assert not window.submit_action.isVisible()
    window.close()


def test_with_no_draft_the_window_starts_from_the_rois_submitted_last(app, tmp_path):
    import dataclasses
    import os

    from datetime import datetime

    from odyn.app.curation_window import DRAFT_VERSION, write_draft
    from odyn.segmentation import Curation, curated_name

    mean, candidates = files(tmp_path)
    found = drawn_candidates()

    def submitted(user, when, edits, candidates=None):
        curated = Curation(found, edits).curated()

        if candidates is not None:
            curated = dataclasses.replace(curated, candidates=candidates)

        name = curated_name(1, user, datetime.fromisoformat(when))
        about = {"user": user, "submitted_at": when, "edits_version": DRAFT_VERSION}
        curated.write(tmp_path / name, **about)

    submitted("bob", "2026-11-16T10:00:00", Edits(rejected={1}))
    submitted("alice", "2026-11-17T14:02:00", Edits(rejected={2}))
    submitted("carol", "2026-11-18T09:00:00", Edits(), candidates="other ones")

    # The last one on these candidates (carol's are on others)
    window = opened(app, tmp_path, mean, candidates)
    assert window.curation.edits.rejected == {2}
    assert window.line.text() == (
        "Started from the ROIs submitted by alice on 2026-11-17 at 14:02."
    )
    window.close()

    # A draft kept here wins, with a word on the ROIs submitted since
    draft = tmp_path / "draft.json"
    write_draft(draft, found, Edits(threshold=-1.0))
    before = datetime(2026, 11, 1).timestamp()
    os.utime(draft, (before, before))
    window = opened(app, tmp_path, mean, candidates)
    assert window.curation.edits.threshold == -1.0
    assert window.line.text() == (
        "Picked up your draft from where you left it. Other ROIs were "
        "submitted since, by alice on 2026-11-17 at 14:02."
    )
    window.close()
