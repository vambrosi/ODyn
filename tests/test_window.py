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
from PySide6.QtGui import QColor, QIcon  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QScrollArea,
)

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
    """
    The prompts down the left of whatever form is showing.

    A row spanning both columns has no prompt, and reads as `''` rather than
    being left out, so a label's position still says which row it is on.
    """
    form = form_of(window)
    found = []

    for row in range(form.rowCount()):
        prompt = form.itemAt(row, form.ItemRole.LabelRole)
        found.append("" if prompt is None else prompt.widget().text())

    return found


def _ink(made: QIcon) -> tuple[int, int]:
    """
    How much of the canvas an icon actually covers, as `(width, height)`.

    What is drawn, not what it was told to draw into: the sizes the eye
    compares in the bar are these.
    """
    image = made.pixmap(
        QSize(W.icons.WIDTH, W.icons.HEIGHT), QIcon.Mode.Normal, QIcon.State.Off
    ).toImage()

    covered = [
        (x, y)
        for x in range(image.width())
        for y in range(image.height())
        if QColor.fromRgba(image.pixel(x, y)).alpha() > 60
    ]

    assert covered, "nothing was drawn at all"

    across = [x for x, _ in covered]
    down = [y for _, y in covered]

    return max(across) - min(across) + 1, max(down) - min(down) + 1


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
# How a field is drawn
# --------------------------------------------------------------------------- #


def prompt_of(window, label: str):
    """The label widget beside one field, which carries its colour."""
    form = form_of(window)

    for row in range(form.rowCount()):
        found = form.itemAt(row, form.ItemRole.LabelRole)

        if found is not None and found.widget().text() == label:
            return found.widget()

    raise AssertionError(f"no {label!r} row on this form")


def test_a_column_and_an_annotation_are_asked_for_the_same_way(window):
    """
    The animal is a column and the weight is an annotation, and the form is
    built from one list, so neither is a widget written out by hand.
    """
    window._add_session()

    for label in ("Mouse ID", "Mouse weight (g)"):
        assert isinstance(editor(window, label), W.FieldRow)


def test_a_required_prompt_is_red_and_the_rest_are_not(window):
    window._add_session()

    assert W.REQUIRED_COLOR in prompt_of(window, "Mouse ID").styleSheet()
    assert W.REQUIRED_COLOR not in prompt_of(window, "Goal").styleSheet()


def test_the_window_says_what_the_red_means(window):
    """A colour nothing explains is decoration."""
    found = window.findChildren(QLabel)

    said = [
        child
        for child in found
        if child.text() == W.REQUIRED_NOTE and W.REQUIRED_COLOR in child.styleSheet()
    ]

    assert len(said) == 1


def test_what_is_still_missing_is_not_also_listed_at_the_bottom(window):
    """The red prompts say it, beside the box, and update as they are filled."""
    window._add_session()

    assert "Still to fill in" not in window.status.text()


def test_a_yes_or_no_is_a_tick_that_starts_unticked(window):
    """No third state: what a tick asks about did not happen unless it is set."""
    window._add_session()

    box = editor(window, "Flag").editor

    assert isinstance(box, QCheckBox)
    assert not box.isChecked()


def test_ticking_reaches_the_draft_and_unticking_says_no(window):
    window._add_session()

    box = editor(window, "Flag").editor

    box.setChecked(True)
    assert Draft.load(window.draft.path).session["flag"] is True

    box.setChecked(False)
    assert Draft.load(window.draft.path).session["flag"] is False


def test_a_field_with_a_list_of_answers_is_a_dropdown(window):
    """Whether the list came from a column or from the registry."""
    window._add_session()
    window._add_experiment()

    assert isinstance(editor(window, "Objective").editor, QComboBox)
    assert isinstance(editor(window, "Depth class").editor, QComboBox)


def test_the_mouse_can_be_written_with_or_without_its_m(window):
    """Which is how the lab writes it, and how the badge shows it back."""
    window._add_session()
    editor(window, "Mouse ID")._changed("m442")

    assert window.draft.mouse_id == 442

    window._add_session()
    editor(window, "Mouse ID")._changed("357")

    assert window.draft.mouse_id == 357


def test_the_date_is_checked_before_it_is_written(window):
    window._add_session()

    row = editor(window, "Date")
    row._changed("08/07/2026")

    assert not row.error.isHidden()
    assert window.draft.date != "08/07/2026"


# --------------------------------------------------------------------------- #
# The bars
# --------------------------------------------------------------------------- #


def test_the_bar_puts_the_sessions_above_the_views_of_one(window):
    """
    Which session comes first and on its own. Everything below it is a view of
    that session and only one can be selected at a time, so those read as one
    group with nothing drawn through the middle of them.
    """
    window._add_session()
    window._add_session()
    window._add_experiment()

    tips = [
        "|" if action.isSeparator() else action.toolTip()
        for action in window.bar.actions()
    ]

    assert tips[2] == "Add session", "the sessions come first, then their `+`"
    assert tips[3] == "|"
    assert tips[4:] == [
        "Session",
        "Odors",
        "e1 — right-click to remove",
        "Add experiment",
    ]
    assert tips.count("|") == 1, "no line between the views and the experiments"


def test_the_two_adds_do_not_look_alike(window):
    """Which thing a `+` adds should not need a hover to find out."""
    assert (
        window.add_action.icon().cacheKey()
        != window.add_experiment_action.icon().cacheKey()
    )


def test_a_session_is_badged_by_position_until_it_has_a_mouse(window):
    """So two unnamed sessions can still be told apart in the bar."""
    window._add_session()
    window._add_session()

    assert [draft.mouse_id for draft in window.drafts] == [None, None]
    assert all(not action.icon().isNull() for action in window.session_actions)


def test_naming_the_mouse_changes_what_the_bar_says(window):
    window._add_session()
    window._identify(mouse_id=442)

    assert window.session_actions[0].toolTip().startswith(window.draft.label)
    assert "m442" in window.session_actions[0].toolTip()


def test_the_mouse_id_is_on_the_form_and_reaches_the_draft(window):
    """Typed at the top of the session, which is what the badge then shows."""
    window._add_session()

    assert labels(window)[:2] == ["Mouse ID", "Date"]

    row = editor(window, "Mouse ID")
    row.editor.setText("m442")
    row._changed("m442")

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

    mouse = editor(window, "Mouse ID").editor
    day = editor(window, "Date").editor

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

    # And the `+` still closes the sessions, after both of them.
    bar = window.bar.actions()

    assert bar.index(window.add_action) == bar.index(window.session_actions[-1]) + 1


def test_a_mouse_id_that_is_not_a_number_is_ignored(window):
    """Rather than stored: the badge and the folder lookup both need a number."""
    window._add_session()

    row = editor(window, "Mouse ID")
    row._changed("the grey one")

    assert window.draft.mouse_id is None
    assert not row.error.isHidden(), "and is said so in place"


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
    for name in W.icons.GLYPHS:
        assert not icon(name).pixmap(26, 26).isNull()

    for text in ("1", "m442", "e1", "m1234"):
        assert not label_icon(text).pixmap(26, 26).isNull()


def test_a_session_icon_is_its_number(window):
    """Drawn as the icon, not squeezed under a picture where it cannot be read."""
    window._add_session()

    assert window.session_actions[0].icon().cacheKey() == label_icon("1").cacheKey()

    window._identify(mouse_id=442)

    assert window.session_actions[0].icon().cacheKey() == label_icon("m442").cacheKey()


# --------------------------------------------------------------------------- #
# The notes banner
# --------------------------------------------------------------------------- #


def test_the_banner_says_what_the_notes_are_about(window):
    """Otherwise the box quietly changing content is easy to miss."""
    assert window.notes.windowTitle() == "Notes"

    window._add_session()
    assert window.notes.windowTitle() == f"{window.draft.label} — Session notes"


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


def test_labels_of_the_same_length_are_drawn_the_same_size(qt):
    """
    `e1` and `e2` beside each other at different sizes reads as a mistake. The
    font is fixed-width and sized by how many characters there are rather than
    by measuring the characters themselves, so it cannot happen.
    """
    sizes = {
        text: W.icons._label_font(text).pixelSize() for text in ("e1", "e2", "e9", "m7")
    }

    assert len(set(sizes.values())) == 1, sizes

    # What is left over is the shapes of the characters themselves, which is
    # ordinary typography rather than two sizes in the bar.
    drawn = [_ink(label_icon(text))[1] for text in sizes]

    assert max(drawn) - min(drawn) <= 2

    # And a longer label is set smaller, which is what pays for the above.
    assert W.icons._label_font("m442").pixelSize() < sizes["e1"]


def test_a_badged_icon_sits_in_the_same_band_as_a_plain_one(qt):
    """Or the two `+` would stand out from the row they are in."""
    plain = _ink(icon("experiment"))[1]
    badged = _ink(icon("add-experiment"))[1]

    assert badged <= plain * 1.15


def test_nothing_is_drawn_to_the_very_edge(qt):
    """The last row of pixels is where Qt's scaling loses it."""
    for name in W.icons.GLYPHS:
        made = icon(name).pixmap(
            QSize(W.icons.WIDTH, W.icons.HEIGHT), QIcon.Mode.Normal, QIcon.State.Off
        )
        image = made.toImage()
        edges = [
            (x, y)
            for x in range(image.width())
            for y in range(image.height())
            if (x in (0, image.width() - 1) or y in (0, image.height() - 1))
            and QColor.fromRgba(image.pixel(x, y)).alpha() > 60
        ]

        assert edges == [], f"{name} reaches the canvas edge at {edges[:3]}"


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
# Experiments
# --------------------------------------------------------------------------- #


def test_adding_an_experiment_selects_it_and_puts_it_in_the_bar(window):
    window._add_session()
    window._add_experiment()

    assert window.experiment == "e1"
    assert list(window.experiment_actions) == ["e1"]
    assert window.experiment_actions["e1"].isChecked()
    assert Draft.load(window.draft.path).experiment("e1") == {"name": "e1"}


def test_experiments_are_numbered_as_they_are_added(window):
    window._add_session()
    window._add_experiment()
    window._add_experiment()

    assert list(window.experiment_actions) == ["e1", "e2"]


def test_a_removed_number_is_used_again(window):
    """The recordings on disk are e1, e2, e3 with no gaps, so these must match."""
    window._add_session()
    window._add_experiment()
    window._add_experiment()

    window._remove_experiment("e1")
    window._add_experiment()

    assert list(window.experiment_actions) == ["e1", "e2"]


def test_an_experiment_has_its_own_form(window):
    window._add_session()
    window._add_experiment()

    shown = labels(window)

    assert shown[0] == "Objective"
    assert "Mouse ID" not in shown, "that is the session's, not the experiment's"
    assert "Note" not in shown, "the dock owns it"


def test_a_field_on_the_experiment_form_reaches_that_experiment(window):
    window._add_session()
    window._add_experiment()
    window._add_experiment()

    row = editor(window, "FOV depth (um)")
    row._changed("-55")

    stored = Draft.load(window.draft.path)

    assert stored.experiment("e2")["fov_depth_um"] == -55
    assert "fov_depth_um" not in stored.experiment("e1")


def test_the_objective_is_chosen_from_a_list(window):
    """A typo here rescales everything measured in microns."""
    window._add_session()
    window._add_experiment()

    box = editor(window, "Objective").editor
    box.setCurrentText("10")

    assert Draft.load(window.draft.path).experiment("e1")["objective"] == 10


def test_the_note_follows_the_selected_experiment(window):
    """Each experiment has its own, and neither is the session's."""
    window._add_session()
    window._add_experiment()
    window.notes.editor.setPlainText("z-stack after this one")

    window._select("session")
    assert window.notes.editor.toPlainText() == ""

    window._select("e1")
    assert window.notes.editor.toPlainText() == "z-stack after this one"

    stored = Draft.load(window.draft.path)

    assert stored.experiment("e1")["note"] == "z-stack after this one"
    assert stored.session.get("note") is None


def test_the_banner_says_which_experiment_the_note_is_about(window):
    window._add_session()
    window._identify(mouse_id=442)
    window._add_experiment()

    assert window.notes.windowTitle() == "m442 — e1 notes"


def test_experiments_belong_to_their_own_session(window):
    """Switching sessions must not offer one session's experiments on another."""
    window._add_session()
    window._add_experiment()
    window._add_experiment()

    window._add_session()
    assert list(window.experiment_actions) == []
    assert window.experiment is None, "and what was shown falls back to the session"

    window._choose(0)
    assert list(window.experiment_actions) == ["e1", "e2"]


def test_removing_an_experiment_goes_back_to_the_session(window):
    """There is nothing to show once it is gone."""
    window._add_session()
    window._add_experiment()

    window._remove_experiment("e1")

    assert window.draft.experiments == []
    assert list(window.experiment_actions) == []
    assert window.showing == "session"
    assert window.section_actions["session"].isChecked()


def test_the_remove_button_is_on_the_experiment_form(window):
    """The `+` is easy to press twice, and a spare experiment blocks the submit."""
    window._add_session()
    window._add_experiment()

    form = form_of(window)
    last = form.rowCount() - 1
    row = form.itemAt(last, form.ItemRole.FieldRole).widget()
    button = window.remove_button

    assert button.text() == "Remove e1"
    assert button.width() == W.BUTTON_WIDTH, "not as wide as the column"

    # Pushed right: the stretch before it is what takes up the rest of the row.
    assert row.layout().itemAt(0).spacerItem() is not None
    assert row.layout().itemAt(1).widget() is button

    button.click()

    assert window.draft.experiments == []


def test_adding_an_experiment_from_the_bar_does_not_crash(qt, window):
    """
    Its icons are rebuilt when the session changes, which the session ones are
    not. Safe only because the `+` doing the rebuilding is never one of them.
    """
    window.show()
    window._add_session()

    window.add_experiment_action.trigger()
    qt.processEvents()
    window.add_experiment_action.trigger()
    qt.processEvents()

    window.experiment_actions["e1"].trigger()
    qt.processEvents()

    assert window.experiment == "e1"
    assert [entry["name"] for entry in window.draft.experiments] == ["e1", "e2"]


def test_an_experiment_cannot_be_added_without_a_session(window):
    """The `+` is always there, and there is nothing for it to add to."""
    window._add_experiment()

    assert window.draft is None
    assert list(window.experiment_actions) == []


# --------------------------------------------------------------------------- #
# Removing from the bar
# --------------------------------------------------------------------------- #


def offer_over(window, action):
    """
    What the bar's right-click menu would offer over one of its icons.

    The position is taken from the button Qt made for the action, since
    `actionAt` works in laid-out coordinates -- so the bar has to have been
    laid out, which happens on the event loop rather than on `show`. Never
    shows the menu: that blocks until someone answers it.
    """
    QApplication.processEvents()
    button = window.bar.widgetForAction(action)
    offer = window._removal_at(button.geometry().center())

    return None if offer is None else offer[0]


@pytest.fixture
def agreeing(monkeypatch):
    """
    Answer yes to the confirmation, without a dialog nothing can click.

    Removing a session that was typed into asks first, and `QMessageBox` waits
    for an answer, so a test that does not stand in for the person hangs.
    """
    monkeypatch.setattr(
        W.QMessageBox,
        "question",
        staticmethod(lambda *_, **__: W.QMessageBox.StandardButton.Yes),
    )


def test_right_clicking_a_session_offers_to_remove_it(window, drafts, agreeing):
    window.show()
    window._add_session()
    window._identify(mouse_id=442)

    assert offer_over(window, window.session_actions[0]) == "m442"

    window._remove_session(window.drafts[0])

    assert window.drafts == []
    assert Draft.pending(drafts) == []
    assert window.session_actions == []


def test_right_clicking_an_experiment_offers_to_remove_it(window):
    window.show()
    window._add_session()
    window._add_experiment()

    assert offer_over(window, window.experiment_actions["e1"]) == "e1"


def test_nothing_else_in_the_bar_offers_to_be_removed(window):
    """A right-click on a section or on a `+` gets no menu at all."""
    window.show()
    window._add_session()

    for action in (
        window.add_action,
        window.add_experiment_action,
        window.section_actions["session"],
        window.section_actions["panel"],
    ):
        assert offer_over(window, action) is None, action.toolTip()


def test_removing_a_session_takes_its_draft_file_with_it(window, drafts):
    """Left behind, it would come back the next time the app opens."""
    window._add_session()
    path = window.draft.path

    window._remove_session(window.draft)

    assert not path.exists()


def test_a_session_that_was_typed_into_asks_before_going(window, monkeypatch):
    """It is the only copy of what someone spent the day typing."""
    window._add_session()
    window._identify(mouse_id=442)
    path = window.draft.path

    asked = []
    monkeypatch.setattr(
        W.QMessageBox,
        "question",
        staticmethod(
            lambda *args, **_: asked.append(args[-1]) or W.QMessageBox.StandardButton.No
        ),
    )

    window._remove_session(window.drafts[0])

    assert len(window.drafts) == 1, "answering no keeps it"
    assert path.exists()
    assert "cannot be undone" in asked[0]


def test_an_untouched_session_goes_without_a_question(window, monkeypatch):
    """Nothing would be lost, so asking is only in the way."""
    window._add_session()

    monkeypatch.setattr(
        W.QMessageBox,
        "question",
        staticmethod(lambda *_, **__: pytest.fail("should not have asked")),
    )

    window._remove_session(window.drafts[0])

    assert window.drafts == []


def test_removing_a_session_leaves_the_others_alone(window, agreeing):
    window._add_session()
    window._identify(mouse_id=442)
    window._add_session()
    window._identify(mouse_id=357)

    window._remove_session(window.drafts[0])

    assert [draft.mouse_id for draft in window.drafts] == [357]
    assert len(window.session_actions) == 1
    assert window.draft.mouse_id == 357


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
