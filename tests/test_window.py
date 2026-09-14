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

from PySide6.QtCore import QSize, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from odyn.app.icons import icon, label_icon  # noqa: E402

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


def form_of(window):
    """
    The form layout inside whatever pane is showing.

    A `FormPane` holds a centring layout, and the column with the rows in it
    sits in the middle of that.
    """
    centred = window.stack.widget(0).widget().layout()

    return centred.itemAt(1).widget().layout()


def labels(window) -> list[str]:
    """The prompts down the left of whatever form is showing."""
    form = form_of(window)

    return [
        form.itemAt(row, form.ItemRole.LabelRole).widget().text()
        for row in range(form.rowCount())
    ]


def editor(window, label: str):
    """The widget beside one prompt, found by name rather than by position."""
    form = form_of(window)

    for row in range(form.rowCount()):
        if form.itemAt(row, form.ItemRole.LabelRole).widget().text() == label:
            return form.itemAt(row, form.ItemRole.FieldRole).widget()

    raise AssertionError(f"no {label!r} row on this form")


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
    assert window.session_actions[0].isChecked()


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

    window._choose(0)
    assert window.draft.session["goal"] == "first mouse"


# --------------------------------------------------------------------------- #
# Where an edit ends up
# --------------------------------------------------------------------------- #


def test_the_note_is_in_the_dock_not_on_the_form(window):
    """It is written all day, so it stays visible while the rest is filled in."""
    window._add_session()

    assert "Note" not in labels(window)
    assert "Goal" in labels(window)


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

    window._choose(0)
    assert window.notes.editor.toPlainText() == "about the first"

    # The second was never given one, despite having been shown.
    assert Draft.load(window.drafts[1].path).session.get("note") is None


def test_a_form_field_reaches_the_draft(window):
    window._add_session()

    row = editor(window, "Goal")

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


# --------------------------------------------------------------------------- #
# The bars
# --------------------------------------------------------------------------- #


def test_the_bar_has_a_section_per_kind_and_one_per_session(window):
    """Separated by the toolbar's own separator, not by drawn lines."""
    window._add_session()

    tips = [action.toolTip() for action in window.bar.actions()]

    assert tips[:2] == ["Session", "Odors"]
    assert tips[-1] == "Add session"
    assert any(action.isSeparator() for action in window.bar.actions())


def test_a_session_is_badged_by_position_until_it_has_a_mouse(window):
    """So two unnamed sessions can still be told apart in the bar."""
    window._add_session()
    window._add_session()

    assert [draft.mouse_id for draft in window.drafts] == [None, None]
    assert all(not action.icon().isNull() for action in window.session_actions)


def test_naming_the_mouse_changes_what_the_bar_says(window):
    window._add_session()
    window._identify(mouse_id=442)

    assert window.session_actions[0].toolTip() == window.draft.label
    assert "m442" in window.session_actions[0].toolTip()


def test_the_mouse_id_is_on_the_form_and_reaches_the_draft(window):
    """Typed at the top of the session, which is what the badge then shows."""
    window._add_session()

    assert labels(window)[:2] == ["Mouse ID", "Date"]

    box = editor(window, "Mouse ID")
    box.setText("m442")
    box.editingFinished.emit()

    assert Draft.load(window.draft.path).mouse_id == 442
    assert "m442" in window.session_actions[0].toolTip()


def test_tabbing_out_of_the_mouse_id_does_not_crash(qt, window):
    """
    Qt emits `editingFinished` *during* the focus change, so rebuilding the
    form from it destroys both the box that emitted and the box about to take
    focus, and Qt then walks into freed memory. Driven through real focus
    moves, because emitting the signal by hand does not reproduce it.
    """
    window._add_session()
    window.show()

    mouse, day = editor(window, "Mouse ID"), editor(window, "Date")

    mouse.setFocus(Qt.FocusReason.TabFocusReason)
    qt.processEvents()
    mouse.setText("442")

    day.setFocus(Qt.FocusReason.TabFocusReason)
    qt.processEvents()

    assert window.draft.mouse_id == 442
    assert "m442" in window.session_actions[0].toolTip()

    day.setText("2026-07-08")
    mouse.setFocus(Qt.FocusReason.TabFocusReason)
    qt.processEvents()

    assert window.draft.date == "2026-07-08"


def test_adding_a_session_from_the_bar_does_not_crash(qt, window):
    """
    The `+` is a toolbar action. Rebuilding the bar would delete it while Qt is
    still delivering its `triggered`, so sessions are inserted before it rather
    than the bar being cleared.
    """
    window.show()

    window.add_action.trigger()
    qt.processEvents()
    window.add_action.trigger()
    qt.processEvents()

    assert len(window.drafts) == 2
    assert len(window.session_actions) == 2

    # And the `+` is still the last thing in the bar, after both of them.
    assert window.bar.actions()[-1] is window.add_action


def test_a_mouse_id_that_is_not_a_number_is_ignored(window):
    """Rather than stored: the badge and the folder lookup both need a number."""
    window._add_session()

    box = editor(window, "Mouse ID")
    box.setText("the grey one")
    box.editingFinished.emit()

    assert window.draft.mouse_id is None


def test_choosing_a_session_switches_everything_to_it(window):
    window._add_session()
    window._identify(mouse_id=442)
    window._add_session()
    window._identify(mouse_id=357)

    window._choose(0)

    assert window.draft.mouse_id == 442
    assert [action.isChecked() for action in window.session_actions] == [True, False]


# --------------------------------------------------------------------------- #
# The notes dock can be brought back
# --------------------------------------------------------------------------- #


def test_the_right_bar_reopens_a_closed_notes_dock(window):
    """
    Without it the dock's own close button would hide the notes permanently,
    since nothing else puts them back.
    """
    # Shown, because a widget's visibility is only meaningful once its window is.
    window.show()

    assert window.toolBarArea(window.panels) == Qt.ToolBarArea.RightToolBarArea
    assert [action.toolTip() for action in window.panels.actions()] == ["Notes"]

    window.notes.close()
    assert not window.notes.isVisible()
    assert not window.notes_button.isChecked()

    window.notes_button.trigger()
    assert window.notes.isVisible()


# --------------------------------------------------------------------------- #
# How it looks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("section", ["session", "panel"])
def test_every_pane_sits_in_a_column_of_the_same_width(qt, window, section):
    """
    A line the width of the window is tiring to read, so each pane keeps its
    width and the space goes to the margins.

    Width, not just the cap: a maximum on its own leaves the column at the
    width its contents ask for, which is far narrower than this.
    """
    window._add_session()
    window._select(section)
    window.show()

    centred = window.stack.widget(0).widget().layout()
    column = centred.itemAt(1).widget()

    assert column.maximumWidth() == W.COLUMN_WIDTH
    assert centred.count() == 3  # stretch, column, stretch
    assert window.stack.widget(0).frameShape() == QScrollArea.Shape.NoFrame

    window.resize(1900, 900)
    qt.processEvents()

    assert column.width() == W.COLUMN_WIDTH


def test_the_column_gives_way_on_a_narrow_window(qt, window):
    """It is a cap, not a demand: a small window is not made to scroll."""
    window._add_session()
    window.show()
    window.resize(700, 900)
    qt.processEvents()

    column = window.stack.widget(0).widget().layout().itemAt(1).widget()

    assert column.width() < W.COLUMN_WIDTH


def test_entries_are_flat_and_rounded(window):
    """One stylesheet on the window, rather than a setting per widget."""
    assert "border-radius" in window.styleSheet()
    assert "padding" in window.styleSheet()


def test_icons_are_drawn_rather_than_looked_up(qt):
    """
    Nothing names a font family, which is what makes Qt search its aliases --
    slowly, and with a warning, when the name is not a real family.
    """
    for name in ("session", "odors", "notes", "experiment", "add"):
        assert not icon(name).pixmap(26, 26).isNull()

    for text in ("1", "442", "1234"):
        assert not label_icon(text).pixmap(26, 26).isNull()


def test_a_session_icon_is_its_number(window):
    """Drawn as the icon, not squeezed under a picture where it cannot be read."""
    window._add_session()

    assert window.session_actions[0].icon().cacheKey() == label_icon("1").cacheKey()

    window._identify(mouse_id=442)

    assert window.session_actions[0].icon().cacheKey() == label_icon("442").cacheKey()


# --------------------------------------------------------------------------- #
# The notes banner
# --------------------------------------------------------------------------- #


def test_the_banner_says_what_the_notes_are_about(window):
    """Otherwise the box quietly changing content is easy to miss."""
    assert window.notes.windowTitle() == "Notes"

    window._add_session()
    assert window.notes.windowTitle() == f"Session notes — {window.draft.label}"


def test_the_banner_follows_a_rename_without_losing_the_note(window):
    """A session is usually named while someone is part way through typing."""
    window._add_session()
    window.notes.editor.setPlainText("half-typed note")

    window._identify(mouse_id=442)

    assert "m442" in window.notes.windowTitle()
    assert window.notes.editor.toPlainText() == "half-typed note"


def test_the_banner_follows_the_selection(window):
    window._add_session()
    window._identify(mouse_id=442)
    window._add_session()

    assert "no mouse yet" in window.notes.windowTitle()

    window._choose(0)
    assert "m442" in window.notes.windowTitle()


def test_the_window_is_called_odyn(window):
    assert window.windowTitle() == "ODyn"


def test_the_bars_are_flat(window):
    """The native style paints a gradient; this sits with the rest instead."""
    assert "QToolBar {" in window.styleSheet()
    assert "background: palette(window)" in window.styleSheet()


def test_hover_and_chosen_look_the_same(window):
    """Both mean "this one"; a separate hover appearance is only noise."""
    style = window.styleSheet()

    assert "QToolBar QToolButton:hover," in style
    assert "QToolBar QToolButton:checked," in style


@pytest.mark.parametrize(
    "named", ["session", "odors", "notes", "experiment", "add", None]
)
def test_every_icon_is_quiet_at_rest_and_bright_when_chosen(qt, named):
    """
    Two shades in one icon, which Qt picks between by state. A stylesheet
    cannot recolour a drawing, so the drawing carries both -- and an icon
    loaded from a file gets the same treatment as a drawn one, or the sections
    would stay dim while the sessions lit up.
    """
    from PySide6.QtGui import QIcon

    made = icon(named) if named else label_icon("4")

    rest = made.pixmap(QSize(64, 64), QIcon.Mode.Normal, QIcon.State.Off).toImage()
    lit = made.pixmap(QSize(64, 64), QIcon.Mode.Normal, QIcon.State.On).toImage()
    hovered = made.pixmap(QSize(64, 64), QIcon.Mode.Active, QIcon.State.Off).toImage()

    assert rest != lit
    assert hovered == lit


def test_icon_shades_follow_the_theme(qt):
    """
    Taken from the palette at both ends, so the bright shade is white on a dark
    theme and black on a light one.
    """
    from PySide6.QtGui import QColor, QIcon, QPalette

    def brightest(palette):
        qt.setPalette(palette)
        image = (
            label_icon("4")
            .pixmap(QSize(64, 64), QIcon.Mode.Normal, QIcon.State.On)
            .toImage()
        )

        return max(
            QColor(image.pixel(x, y)).lightness()
            for x in range(64)
            for y in range(64)
            if QColor.fromRgba(image.pixel(x, y)).alpha() > 200
        )

    was = qt.palette()

    dark = QPalette()
    dark.setColor(QPalette.ColorRole.WindowText, QColor("white"))
    dark.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))

    light = QPalette()
    light.setColor(QPalette.ColorRole.WindowText, QColor("black"))
    light.setColor(QPalette.ColorRole.Window, QColor("white"))

    try:
        assert brightest(dark) > 200
        assert brightest(light) < 60
    finally:
        qt.setPalette(was)


def test_a_number_is_drawn_the_same_size_as_a_picture(qt):
    """
    Everything in the bar comes from one drawing path, so a session's number
    and a section's glyph are the same height. They were not while some came
    from files and some were drawn, and the bar read as two sizes.
    """
    from PySide6.QtGui import QColor, QIcon

    def height(made):
        image = made.pixmap(
            QSize(W.icons.WIDTH, W.icons.HEIGHT), QIcon.Mode.Normal, QIcon.State.Off
        ).toImage()
        rows = [
            y
            for x in range(image.width())
            for y in range(image.height())
            if QColor.fromRgba(image.pixel(x, y)).alpha() > 60
        ]

        return max(rows) - min(rows) + 1

    glyphs = [height(icon(n)) for n in ("session", "odors", "experiment", "add")]

    # Within a quarter of each other: a glyph is drawn to its own em box, so
    # they are close rather than identical.
    assert max(glyphs) - min(glyphs) <= 0.25 * max(glyphs)
    assert abs(height(label_icon("1")) - max(glyphs)) <= 0.25 * max(glyphs)


def test_a_long_number_still_fits_inside_the_icon(qt):
    """A three-digit mouse is the usual case, so it must not run off the edge."""
    from PySide6.QtGui import QColor, QIcon

    for text in ("1", "442", "1234"):
        image = (
            label_icon(text)
            .pixmap(
                QSize(W.icons.WIDTH, W.icons.HEIGHT), QIcon.Mode.Normal, QIcon.State.Off
            )
            .toImage()
        )
        columns = [
            x
            for x in range(image.width())
            for y in range(image.height())
            if QColor.fromRgba(image.pixel(x, y)).alpha() > 60
        ]

        assert min(columns) > 0, f"{text} touches the left edge"
        assert max(columns) < image.width() - 1, f"{text} touches the right edge"


def test_the_bar_slots_keep_the_drawing_proportions(window):
    """Or the glyphs would be squashed to fit a square slot."""
    slot = window.bar.iconSize()

    assert slot.height() == W.ICON_HEIGHT
    assert abs(slot.width() / slot.height() - W.icons.WIDTH / W.icons.HEIGHT) < 0.05


# --------------------------------------------------------------------------- #
# Closing and coming back
# --------------------------------------------------------------------------- #


def test_a_started_session_is_still_there_after_closing(window, drafts):
    """
    What makes the app safe to quit. A session is usually left with only the
    animal's number on it for a while, and that must survive the day ending.
    """
    window._add_session()
    window._identify(mouse_id=442)
    path = window.draft.path

    window.close()

    assert path.exists()
    assert [draft.mouse_id for draft in Draft.pending(drafts)] == [442]


def test_a_session_nobody_typed_into_is_thrown_away(window, drafts):
    """The other half: a stray click must not leave a session to sort out."""
    window._add_session()

    window.close()

    assert Draft.pending(drafts) == []


def test_closing_keeps_the_sessions_that_were_typed_into(window, drafts):
    """Only the untouched ones go, whatever order they were started in."""
    window._add_session()
    window._add_session()
    window._identify(mouse_id=357)
    window._add_session()

    window.close()

    assert [draft.mouse_id for draft in Draft.pending(drafts)] == [357]
