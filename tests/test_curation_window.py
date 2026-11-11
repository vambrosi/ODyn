"""
`odyn.app.curation_window`, offscreen: the ROIs of a group curated on
candidates drawn by hand (`test_segmentation.drawn_candidates`), with clicks,
sliders and keys, undo, and the draft kept on this computer. Opening it from
the QC page is tested in `test_app_gui.py`.
"""

import os

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Needs Qt and the system libraries it loads: skipped where either is missing
try:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

except ImportError as error:  # a missing library, not a missing module
    pytest.skip(f"Qt cannot load here: {error}", allow_module_level=True)

from odyn.app.curation_window import (  # noqa: E402
    COLORS,
    CurationWindow,
    outline,
    size_at,
    size_position,
)
from test_segmentation import drawn_candidates  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def files(folder, found=None):
    """Candidates (drawn ones by default) and a mean image like them."""
    found = found or drawn_candidates()
    found.write(folder / "cellpose_group_1.h5")
    tifffile.imwrite(folder / "mean_group_1.tif", (found.prob + 6).astype(np.float32))

    return folder / "mean_group_1.tif", folder / "cellpose_group_1.h5"


def opened(app, folder, mean=None, candidates=None) -> CurationWindow:
    """The window on the files in `folder`, once they are read."""
    if mean is None:
        mean, candidates = files(folder)

    window = CurationWindow(
        "20260304/m1/e1", 1, mean, candidates, folder / "draft.json"
    )
    window.show()
    window.loader.wait()
    app.processEvents()  # what it read comes back through the event loop
    app.processEvents()

    return window


def shortcut(window, key: str) -> QShortcut:
    [found] = [
        s for s in window.findChildren(QShortcut) if s.key() == QKeySequence(key)
    ]

    return found


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


def test_the_window_starts_from_the_cells_cellpose_keeps(app, tmp_path):
    window = opened(app, tmp_path)

    assert sorted(window.curation.cells) == [1, 2]
    assert window.counts.text().startswith("2 cells kept (0 added, 0 resized)")
    assert kinds(window) == ["cellpose", "cellpose"]
    assert window.threshold_line.text() == "Threshold 0.00 (the Cellpose default)"
    assert (window.canvas.image.width(), window.canvas.image.height()) == (44, 30)
    window.close()


def test_a_click_selects_a_cell_and_one_elsewhere_adds_the_candidate(app, tmp_path):
    window = opened(app, tmp_path)
    canvas = window.canvas

    # A click on cell 1 (the mouse itself, at its center pixel) selects it
    at = canvas.canvas_point(10.5, 10.5).toPoint()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier(0), at)
    assert window.selected == 1
    assert window.cell_line.text() == "Cell 1, cellpose: threshold 0.00"
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

    # Outside every basin: nothing, and why
    window.click(0, 43)
    assert window.selected is None
    assert window.line.text().startswith("No Cellpose candidate covers this point")
    window.close()


def test_the_selected_cell_is_resized_by_its_slider_and_keys(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(10, 10)
    start = len(window.curation.cells[1].pixels)

    shortcut(window, "]").activated.emit()
    assert len(window.curation.cells[1].pixels) == start + 1
    assert window.curation.cells[1].kind == "resized"
    assert kinds(window) == ["cellpose", "selected"]

    window.size.setValue(window.size.value() + 100)  # larger, toward the basin
    larger = len(window.curation.cells[1].pixels)
    assert larger > start + 1
    assert window.size_line.text().startswith(f"{larger} px")

    # T: the threshold of all again
    shortcut(window, "T").activated.emit()
    assert len(window.curation.cells[1].pixels) == start
    assert window.curation.cells[1].kind == "cellpose"
    window.close()


def test_the_threshold_slider_makes_every_cell_larger_to_the_right(app, tmp_path):
    window = opened(app, tmp_path)
    before = {b: len(c.pixels) for b, c in window.curation.cells.items()}

    window.threshold.setValue(window.threshold.value() + 20)  # 1.0 lower
    app.processEvents()
    assert window.threshold_line.text() == "Threshold -1.00"
    assert all(len(window.curation.cells[b].pixels) > n for b, n in before.items())

    # Undone, and done again
    shortcut(window, "Ctrl+Z").activated.emit()
    assert window.threshold_line.text() == "Threshold 0.00 (the Cellpose default)"
    assert {b: len(c.pixels) for b, c in window.curation.cells.items()} == before
    shortcut(window, "Ctrl+Shift+Z").activated.emit()
    assert window.threshold_line.text() == "Threshold -1.00"
    window.close()


def test_rejected_and_excluded_cells_hide_until_shown(app, tmp_path):
    window = opened(app, tmp_path)
    window.click(10, 10)

    shortcut(window, "Delete").activated.emit()
    assert window.curation.cells[1].rejected and window.reject.text() == "Restore"
    assert kinds(window) == ["cellpose", "selected"]  # dashed while selected

    shortcut(window, "Esc").activated.emit()
    assert kinds(window) == ["cellpose"]
    window.show_rejected.setChecked(True)
    assert kinds(window) == ["rejected", "cellpose"]
    assert window.counts.text() == (
        "1 cell kept (0 added, 0 resized), 1 rejected, 0 outside the area range"
    )

    # The area range: the other cell outside it, hidden until shown
    window.min_area.setValue(len(window.curation.cells[2].pixels) + 1)
    window._area_changed()
    assert window.curation.cells[2].excluded
    assert kinds(window) == ["rejected"]
    window.show_excluded.setChecked(True)
    assert kinds(window) == ["rejected", "excluded"]
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
    assert 3 not in other.curation.cells
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
