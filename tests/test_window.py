"""
The entry window: what the left bar selects, and where an edit ends up.

Qt is optional, so these are skipped where it is not installed. What they check
is the wiring rather than the look -- that choosing a section swaps the form,
that the note follows the selection, and above all that typing anywhere reaches
the draft on disk, since that is what makes closing the app safe.
"""

import os

import pytest

pytest.importorskip("PySide6")

# Before any widget exists, so a test run never opens a window on someone's
# screen. `setdefault`, so it can still be watched with QT_QPA_PLATFORM=cocoa.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from odyn import Database  # noqa: E402
from odyn.app import window as W  # noqa: E402
from odyn.app.draft import Draft  # noqa: E402

PANEL = [{"vial_position": 1, "odor_id": 1}]


@pytest.fixture(scope="session")
def qt():
    """
    The one `QApplication` a process may have. Nothing is shown, so the widgets
    need no other tending: Python collects them with the test.
    """
    return QApplication.instance() or QApplication([])


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "main", project="test")
    database.add_panel(panel_name="print_v3", vials=PANEL)

    return database


@pytest.fixture
def drafts(tmp_path, monkeypatch):
    """Drafts go somewhere disposable, never the real `~/.odyn/drafts`."""
    folder = tmp_path / "drafts"
    monkeypatch.setattr(W, "drafts_folder", lambda: folder)

    return folder


@pytest.fixture
def window(qt, db, drafts):
    return W.EntryWindow(db, [])


# --------------------------------------------------------------------------- #
# The frame
# --------------------------------------------------------------------------- #


def test_it_opens_with_no_sessions(window):
    """No dialog on the way in: the window is there and a session is added to it."""
    assert window.draft is None
    assert "add one" in window.status.text()


def test_the_bar_and_the_dock_are_where_they_belong(window):
    """Qt's own areas, so collapsing and resizing are not ours to write."""
    assert window.toolBarArea(window.bar) == Qt.ToolBarArea.LeftToolBarArea
    assert window.dockWidgetArea(window.notes) == Qt.DockWidgetArea.RightDockWidgetArea


def test_the_form_draws_one_box_not_two(window):
    """The scroll area and the form inside it would otherwise both draw one."""
    window._add_session()

    assert window.stack.widget(0).frameShape() == QScrollArea.Shape.NoFrame


# --------------------------------------------------------------------------- #
# Choosing what to edit
# --------------------------------------------------------------------------- #


def test_adding_a_session_selects_it(window):
    window._add_session()

    assert window.draft is not None
    assert window.picker.currentText() == window.draft.label


def test_a_new_session_has_no_mouse_yet(window):
    window._add_session()

    assert window.draft.mouse_id is None
    assert "No mouse number yet." in window.status.text()


def test_the_bar_switches_what_is_shown(window):
    window._add_session()

    window._select("panel")
    assert isinstance(window.stack.widget(0), W.PanelPane)

    window._select("session")
    assert isinstance(window.stack.widget(0), W.FormPane)


def test_two_sessions_are_edited_independently(window):
    window._add_session()
    first = window.draft
    first.update_session(goal="first mouse")

    window._add_session()
    second = window.draft

    assert first.path != second.path
    assert second.session.get("goal") is None

    window.picker.setCurrentIndex(0)
    assert window.draft.session["goal"] == "first mouse"


# --------------------------------------------------------------------------- #
# Where an edit ends up
# --------------------------------------------------------------------------- #


def test_the_note_is_in_the_dock_not_on_the_form(window):
    """It is written all day, so it stays visible while the rest is filled in."""
    window._add_session()

    form = window.stack.widget(0).widget().layout()
    labels = [
        form.itemAt(row, form.ItemRole.LabelRole).widget().text()
        for row in range(form.rowCount())
    ]

    assert "Note" not in labels
    assert "Goal" in labels


def test_typing_a_note_reaches_the_draft_on_disk(window):
    window._add_session()

    window.notes.editor.setPlainText("genteal drops before starting")

    assert Draft.load(window.draft.path).session["note"] == (
        "genteal drops before starting"
    )


def test_the_note_follows_the_selected_session(window):
    """Switching must not write one session's note onto another."""
    window._add_session()
    window.notes.editor.setPlainText("about the first")

    window._add_session()
    assert window.notes.editor.toPlainText() == ""

    window.picker.setCurrentIndex(0)
    assert window.notes.editor.toPlainText() == "about the first"

    # The second was never given one, despite having been shown.
    assert Draft.load(window.drafts[1].path).session.get("note") is None


def test_a_form_field_reaches_the_draft(window):
    window._add_session()

    form = window.stack.widget(0).widget().layout()
    row = form.itemAt(0, form.ItemRole.FieldRole).widget()

    row.editor.setText("10x pre/post ket/xyl")
    row._changed("10x pre/post ket/xyl")

    assert Draft.load(window.draft.path).session["goal"] == "10x pre/post ket/xyl"


def test_a_value_the_field_cannot_read_is_shown_in_place(window):
    """And is not written, so a typo never reaches the draft as a string."""
    window._add_session()

    fields = W.form_fields(window.db, "session")
    weight = next(f for f in fields if f.key == "mouse_weight_g")

    written = []
    row = W.FieldRow(weight, None, lambda field, value: written.append(value))
    row._changed("forgot to take")

    # `isHidden` rather than `isVisible`: nothing here is on screen, and a
    # child of an unshown parent is never "visible" whatever it was told.
    assert not row.error.isHidden()
    assert "number" in row.error.text()
    assert written == []
