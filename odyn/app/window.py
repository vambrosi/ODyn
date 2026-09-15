"""
The entry window.

Needs Qt, an optional dependency -- `pip install -e .[gui]`. Launch it with
`python -m odyn.app MAIN_FOLDER`.

The window holds no state of its own. Every edit goes straight into the `Draft`
and therefore straight to disk, so closing the app mid-session loses nothing.
What each form asks for comes from `fields.py`, which builds it from the
annotation registry.

**LAYOUT**
Three of Qt's own `QMainWindow` parts, rather than nested layouts: a toolbar
down the left edge to choose what is being edited, a dock on the right for the
note, and a stack in the middle holding the forms. Qt then handles collapsing,
resizing and remembering the arrangement between runs.

The note is not on any form. It belongs to whatever the left bar has selected,
and lives in the dock so it stays readable while the rest is filled in.

The bar reads in three groups: what to edit about the session, the sessions
themselves, and the selected session's experiments. Each of the last two ends
in a `+` that adds one. `showing` says which of them is selected -- `'session'`,
`'panel'`, or an experiment's name.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .draft import Draft, drafts_folder
from .fields import BOOLEAN, LONG_TEXT, Field, form_fields
from . import icons
from .icons import icon, label_icon
from .submit import check_all, session_path, submit_all

# A blank dropdown entry, so "not recorded" stays different from a real answer.
UNSET = "—"

# Held in the dock rather than on a form, so it is visible while anything else
# is being filled in.
NOTE = "note"

# How large the bar icons are drawn on screen, in the same proportion they are
# drawn in, so nothing is scaled unevenly.
ICON_HEIGHT = 39
ICON_WIDTH = round(ICON_HEIGHT * icons.WIDTH / icons.HEIGHT)

# A line of text much wider than this is tiring to read: the eye loses the
# start of the next line. The form sits in a column of this width, centred,
# rather than stretching with the window.
COLUMN_WIDTH = 800
FIELD_WIDTH = 260

# Buttons on a form take their own width rather than the column's: one as wide
# as the page reads as a banner, and is a large target for something that
# cannot be undone.
BUTTON_WIDTH = 160

# What marks a field an answer is expected for, and the note that says so.
# Chosen to read against both a light and a dark background, which the palette's
# own colours do not: `link-visited` is near-black on one of them.
REQUIRED_COLOR = "#d0636b"
REQUIRED_NOTE = "Required fields"

# Flat, rounded, padded. Qt's native entry is a sunken bevel, which reads as a
# hole in the page next to the rest of this.
STYLE = """
QLineEdit, QPlainTextEdit, QComboBox {
    border: 1px solid palette(mid);
    border-radius: 6px;
    padding: 5px 8px;
    background: palette(base);
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus {
    border: 1px solid palette(highlight);
}
QComboBox::drop-down { border: none; width: 18px; }

/* Flat and generously padded, to sit with the inputs above rather than as a
   raised native button beside them. */
QPushButton {
    border: 1px solid palette(mid);
    border-radius: 6px;
    padding: 9px 16px;
    background: palette(window);
}
QPushButton:hover { background: palette(midlight); }
QPushButton:pressed { background: palette(mid); }

/* The bars are painted with a gradient by the native style. Flat, so they sit
   with the rest of the window rather than standing off it. */
QToolBar {
    background: palette(window);
    border: none;
    spacing: 2px;
    padding: 4px;
}

/* Hover and chosen look the same on purpose: both mean "this one", and a
   second appearance for hovering only adds noise. The fill sits inside the
   drawn square; the square and its label brighten with it, which the icon
   does for itself. */
QToolBar QToolButton {
    border: none;
    border-radius: 8px;
    padding: 3px;
}
QToolBar QToolButton:hover,
QToolBar QToolButton:checked,
QToolBar QToolButton:pressed {
    background: palette(midlight);
}
"""


class FieldRow(QWidget):
    """One registry field, drawn to suit its type, saving as it is edited."""

    def __init__(self, field: Field, value, on_change):
        super().__init__()

        self.field = field
        self.on_change = on_change
        self.editor = self._editor(value)

        self.error = QLabel()
        self.error.setStyleSheet("color: palette(link-visited);")
        self.error.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor)
        layout.addWidget(self.error)

        if field.description:
            self.editor.setToolTip(field.description)

    def _editor(self, value) -> QWidget:
        """A tick for a yes/no, a dropdown for a closed answer, a line otherwise."""
        if self.field.value_type == BOOLEAN:
            # No third "not recorded" state: unticked means no. Everything a
            # tick is asked about here is something that did not happen unless
            # someone says it did.
            box = QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda on: self._changed("yes" if on else "no"))

            return box

        if self.field.options:
            box = QComboBox()
            box.addItem(UNSET)
            box.addItems(list(self.field.options))
            box.setCurrentText(_as_text(value) or UNSET)
            box.currentTextChanged.connect(self._changed)
        elif self.field.value_type == LONG_TEXT:
            box = QPlainTextEdit(_as_text(value))
            box.setMinimumHeight(90)

            # A text area has no `editingFinished`, and prose is typed over a
            # whole session, so it saves as it is written.
            box.textChanged.connect(lambda: self._changed(box.toPlainText()))
        else:
            box = QLineEdit(_as_text(value))

            # On finishing rather than per keystroke: a half-typed number is
            # not a parse error worth showing anyone.
            box.editingFinished.connect(lambda: self._changed(box.text()))

        box.setMinimumWidth(FIELD_WIDTH)

        return box

    def _changed(self, text: str) -> None:
        if text == UNSET:
            text = ""

        try:
            self.on_change(self.field, self.field.parse(text))
            self.error.hide()
        except ValueError as wrong:
            self.error.setText(str(wrong))
            self.error.show()


class Pane(QScrollArea):
    """
    A scrolling column of fields, the same width whatever is in it.

    Centred rather than stretched: a line as wide as the window is tiring to
    read, so the form keeps its width and the space goes to the margins.
    """

    def __init__(self, form: QFormLayout):
        super().__init__()

        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        column = QWidget()
        column.setLayout(form)
        column.setMaximumWidth(COLUMN_WIDTH)

        # A maximum is only a cap: left to itself the column would sit at the
        # width its contents ask for, which is far narrower. Expanding makes it
        # grow into the cap and stop there.
        column.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # The column takes what it can and the margins share the rest, so it
        # reaches its cap on a wide window instead of stopping at the width its
        # contents happen to ask for.
        centred = QHBoxLayout()
        centred.addStretch(1)
        centred.addWidget(column, 100)
        centred.addStretch(1)

        holder = QWidget()
        holder.setLayout(centred)

        self.setWidget(holder)
        self.setWidgetResizable(True)

        # Otherwise the scroll area draws a border and the column inside draws
        # another, which reads as a box inside a box.
        self.setFrameShape(QScrollArea.Shape.NoFrame)


class FormPane(Pane):
    """
    A form built from registry fields.

    `values` is what to show, `on_change(field, value)` is called as each is
    edited. The note is left out: the dock owns it.

    `trailing` rows go below the fields, as `(label, widget)`: what belongs on
    the form but is not a value, such as a button to undo adding the thing
    being edited.
    """

    def __init__(self, fields: list[Field], values: dict, on_change, trailing=()):
        form = QFormLayout()

        for field in fields:
            if field.key == NOTE:
                continue

            form.addRow(
                _prompt(field), FieldRow(field, values.get(field.key), on_change)
            )

        for label, widget in trailing:
            # No prompt means the whole width, which is what a button wants:
            # indented under a blank label it reads as an answer to nothing.
            form.addRow(widget) if not label else form.addRow(label, widget)

        super().__init__(form)


class PanelPane(Pane):
    """Which rack of vials, and when each was mixed."""

    def __init__(self, draft: Draft, db):
        self.draft = draft
        names = sorted(db.panels["panel_name"]) if len(db.panels) else []

        self.panel = QComboBox()
        self.panel.addItems([UNSET] + names)
        self.panel.setCurrentText(draft.panel.get("panel_name") or UNSET)
        self.panel.currentTextChanged.connect(
            lambda text: draft.set_panel(panel_name=None if text == UNSET else text)
        )

        self.made_on = QLineEdit(draft.panel.get("made_on") or "")
        self.made_on.setPlaceholderText("YYYY-MM-DD")
        self.made_on.editingFinished.connect(self._set_made_on)

        self.vials = QPlainTextEdit(_vial_text(draft.panel.get("vial_dates", {})))
        self.vials.setPlaceholderText(
            "One per line, for vials mixed on a different day.\n"
            "'vial #: date', numbered as in the odor log:\n"
            "4: 2026-07-21"
        )
        self.vials.textChanged.connect(self._set_vials)

        for box in (self.panel, self.made_on):
            box.setMinimumWidth(FIELD_WIDTH)

        form = QFormLayout()
        form.addRow("Panel", self.panel)
        form.addRow("Mixed on", self.made_on)
        form.addRow("Vials mixed separately", self.vials)

        super().__init__(form)

    def _set_made_on(self) -> None:
        self.draft.set_panel(made_on=self.made_on.text().strip() or None)

    def _set_vials(self) -> None:
        dates = {}

        for line in self.vials.toPlainText().splitlines():
            position, _, date = line.partition(":")

            if position.strip().isdigit() and date.strip():
                dates[int(position.strip())] = date.strip()

        self.draft.set_panel(vial_dates=dates)


class NotesDock(QDockWidget):
    """
    The note for whatever the left bar has selected.

    One dock reused as the selection moves, rather than a note box on every
    form: it is the field people write in all day, so it stays open.
    """

    def __init__(self):
        super().__init__("Notes")
        self.setObjectName("notes")  # so saveState can remember it

        self.target: None | tuple[Draft, None | str] = None
        self.loading = False

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "Anything worth knowing, including what went wrong."
        )
        self.editor.textChanged.connect(self._changed)

        self.setWidget(self.editor)

    def show_note(self, draft: None | Draft, experiment: None | str = None) -> None:
        """
        Point the dock at a session's note, or at one experiment's.

        The banner says which, because the box itself changing content is easy
        to miss when several sessions are open.
        """
        self.target = None if draft is None else (draft, experiment)

        held = ""

        if draft is None:
            self.setWindowTitle("Notes")
        else:
            where = (
                draft.session
                if experiment is None
                else (draft.experiment(experiment) or {})
            )
            held = where.get(NOTE) or ""

            self.retitle()

        # Setting the text fires `textChanged`, which would write the note we
        # just loaded back onto whatever is now selected.
        self.loading = True
        self.editor.setPlainText(held)
        self.loading = False

        self.editor.setEnabled(draft is not None)

    def retitle(self) -> None:
        """
        Say again what the note is about, without reloading it.

        Renaming a session changes the banner while someone may be part way
        through typing into the box, so the text is left exactly as it is.
        """
        if self.target is None:
            self.setWindowTitle("Notes")
            return

        draft, experiment = self.target
        about = "Session" if experiment is None else experiment

        self.setWindowTitle(f"{draft.label} — {about} notes")

    def _changed(self) -> None:
        if self.loading or self.target is None:
            return

        draft, experiment = self.target
        text = self.editor.toPlainText() or None

        if experiment is None:
            draft.update_session(**{NOTE: text})
        else:
            draft.set_experiment(experiment, **{NOTE: text})


class EntryWindow(QMainWindow):
    """A day's sessions: what is being edited is chosen from the left bar."""

    def __init__(self, db, drafts: list[Draft]):
        super().__init__()

        self.db = db
        self.drafts = list(drafts)
        self.showing = "session"
        self.selected = 0

        self.setWindowTitle("ODyn")
        self.resize(1000, 700)
        self.setStyleSheet(STYLE)

        self.bar = QToolBar("Sections")
        self.bar.setObjectName("sections")
        self.bar.setMovable(False)
        self.bar.setIconSize(QSize(ICON_WIDTH, ICON_HEIGHT))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.bar)

        # Removing is on a right-click rather than on a button in the corner of
        # the icon: at this size that button would be a few pixels across, and
        # hitting it by accident would throw away a session someone typed.
        self.bar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.bar.customContextMenuRequested.connect(self._bar_menu)

        self.notes = NotesDock()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.notes)

        # Its own bar on the other edge, so a closed dock can be brought back.
        # Without it, closing the notes would hide them for good.
        self.panels = QToolBar("Panels")
        self.panels.setObjectName("panels")
        self.panels.setMovable(False)
        self.panels.setIconSize(QSize(ICON_WIDTH, ICON_HEIGHT))
        self.addToolBar(Qt.ToolBarArea.RightToolBarArea, self.panels)

        self.notes_button = self.panels.addAction(icon("notes"), "Notes")
        self.notes_button.setToolTip("Notes")
        self.notes_button.setCheckable(True)
        self.notes_button.setChecked(True)
        # Read from the action rather than taking the signal's argument:
        # `triggered` carries the checked state only sometimes, and a bare
        # `setVisible` would then be called with nothing.
        self.notes_button.triggered.connect(
            lambda: self.notes.setVisible(self.notes_button.isChecked())
        )

        # So closing the dock by its own X un-checks the button that reopens it.
        self.notes.visibilityChanged.connect(self.notes_button.setChecked)

        self.stack = QStackedWidget()

        self.status = QLabel()
        self.status.setWordWrap(True)

        # Says what the red prompts on the form mean, in the same red.
        legend = QLabel(REQUIRED_NOTE)
        legend.setStyleSheet(f"color: {REQUIRED_COLOR};")

        submit = QPushButton("Check and submit the day")
        submit.clicked.connect(self._submit_day)

        bottom = QHBoxLayout()
        bottom.addWidget(legend)
        bottom.addWidget(self.status, stretch=1)
        bottom.addWidget(submit)

        layout = QVBoxLayout()
        layout.addWidget(self.stack, stretch=1)
        layout.addLayout(bottom)

        holder = QWidget()
        holder.setLayout(layout)
        self.setCentralWidget(holder)

        self._build_bar()
        self._restore_layout()

    # ------------------------------------------------------------------ #
    # The left bar
    # ------------------------------------------------------------------ #

    def _build_bar(self) -> None:
        """
        The parts of the bar that never change: what to edit, and the `+`.

        Session icons are inserted between them as sessions are added, rather
        than by rebuilding: clearing a toolbar destroys its actions, and this
        runs from an action's own `triggered`.
        """
        self.section_actions = {}
        self.session_actions = []
        self.experiment_actions = {}

        # Which session comes first and on its own, because it says what
        # everything below is about. The rest are the views of it, and exactly
        # one of them is selected at a time, so they read as one group with no
        # line drawn through the middle of it.
        self.add_action = self.bar.addAction(icon("add-session"), "Add session")
        self.add_action.setToolTip("Add session")
        self.add_action.triggered.connect(self._add_session)

        self.bar.addSeparator()

        for name, label, picture in (
            ("session", "Session", "session"),
            ("panel", "Odors", "odors"),
        ):
            action = self.bar.addAction(icon(picture), label)
            action.setToolTip(label)
            action.setCheckable(True)
            action.setChecked(name == self.showing)
            action.triggered.connect(lambda _, which=name: self._select(which))
            self.section_actions[name] = action

        self.add_experiment_action = self.bar.addAction(
            icon("add-experiment"), "Add experiment"
        )
        self.add_experiment_action.setToolTip("Add experiment")
        self.add_experiment_action.triggered.connect(self._add_experiment)

        for draft in self.drafts:
            self._add_session_action(draft)

        self._rebuild_experiments()
        self._show()

    def _add_session_action(self, draft: Draft) -> None:
        """One session icon, inserted just before the `+` that adds them."""
        action = QAction(label_icon("?"), draft.label, self)
        action.setCheckable(True)
        action.triggered.connect(lambda _, which=draft: self._choose(which))

        self.bar.insertAction(self.add_action, action)
        self.session_actions.append(action)

        self._update_badges()

    def _rebuild_experiments(self) -> None:
        """
        The selected session's experiments, as icons before their own `+`.

        Rebuilt rather than updated in place, because a different session has a
        different number of them. Safe where `_update_badges` is not: nothing
        here is ever the action whose `triggered` is running -- the `+` and the
        session icons are not touched -- and each is released with
        `deleteLater`, so Qt finishes delivering to it first.
        """
        for action in self.experiment_actions.values():
            self.bar.removeAction(action)
            action.deleteLater()

        self.experiment_actions = {}
        draft = self.draft

        if draft is None:
            return

        for entry in draft.experiments:
            name = entry.get("name")

            if name is None:
                continue

            action = QAction(label_icon(name), name, self)
            action.setToolTip(f"{name} — right-click to remove")
            action.setCheckable(True)
            action.setChecked(name == self.showing)
            action.triggered.connect(lambda _, which=name: self._select(which))

            self.bar.insertAction(self.add_experiment_action, action)
            self.experiment_actions[name] = action

    def _bar_menu(self, where) -> None:
        """Offer to remove whichever session or experiment was right-clicked."""
        offer = self._removal_at(where)

        if offer is None:
            return

        what, remove = offer

        menu = QMenu(self)
        menu.addAction(f"Remove {what}").triggered.connect(remove)
        menu.exec(self.bar.mapToGlobal(where))

    def _removal_at(self, where):
        """
        What a right-click there offers, as `(what, remove)`, or `None`.

        Only a session or an experiment can be removed, so a click anywhere
        else in the bar gets no menu rather than an empty one. Kept apart from
        showing the menu, which blocks until someone answers it.
        """
        clicked = self.bar.actionAt(where)

        if clicked in self.session_actions:
            draft = self.drafts[self.session_actions.index(clicked)]

            return draft.label, lambda: self._remove_session(draft)

        for name, action in self.experiment_actions.items():
            if action is clicked:
                return name, lambda: self._remove_experiment(name)

        return None

    def _select(self, which: str) -> None:
        """
        Choose what is being edited for the current session.

        `which` is a section -- `'session'` or `'panel'` -- or an experiment's
        name, which cannot be mistaken for one: experiments are `e1`, `e2`.
        """
        self.showing = which
        self._recheck()
        self._show()

    def _recheck(self) -> None:
        """Mark the one section or experiment being edited, and unmark the rest."""
        for name, action in {**self.section_actions, **self.experiment_actions}.items():
            action.setChecked(name == self.showing)

    def _choose(self, which: int | Draft) -> None:
        """Choose which session the forms and the note are about."""
        self.selected = which if isinstance(which, int) else self.drafts.index(which)

        # Sessions do not share experiments, so one selected on the session
        # being left has nothing to show on the one arrived at.
        if self.experiment is not None and self.experiment not in self._experiments():
            self.showing = "session"

        self._update_badges()
        self._rebuild_experiments()
        self._recheck()
        self._show()

    def _update_badges(self) -> None:
        """
        Redraw what each session icon says and which is checked.

        In place, never by rebuilding: this runs from a field's
        `editingFinished`, which Qt emits *during* a focus change, so
        destroying widgets here takes the box that emitted it and the box about
        to receive focus with it.
        """
        for index, (draft, action) in enumerate(zip(self.drafts, self.session_actions)):
            # The position until someone types a number, so the icons can be
            # told apart from the moment they appear. Drawn as the icon itself
            # rather than as a badge over one: a number squeezed under a
            # picture is unreadable at the size a toolbar gives it.
            # An `m` in front once there is a number, as the folders and the
            # lab both write it. Without one it is a bare position, which is
            # the point: it says the session has not been named yet.
            badge = f"m{draft.mouse_id}" if draft.mouse_id is not None else index + 1

            action.setIcon(label_icon(str(badge)))
            action.setToolTip(f"{draft.label} — right-click to remove")
            action.setText(draft.label)
            action.setChecked(index == self.selected)

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #

    @property
    def draft(self) -> None | Draft:
        """The session being edited, or `None` before any has been added."""
        at = self.selected

        return self.drafts[at] if 0 <= at < len(self.drafts) else None

    @property
    def experiment(self) -> None | str:
        """The experiment being edited, or `None` while a section is shown."""
        return None if self.showing in self.section_actions else self.showing

    def _experiments(self) -> list[str]:
        """The selected session's experiment names, in the order they are held."""
        draft = self.draft

        if draft is None:
            return []

        return [entry["name"] for entry in draft.experiments if entry.get("name")]

    def _add_session(self) -> None:
        draft = Draft.start(drafts_folder())

        self.drafts.append(draft)
        self.selected = len(self.drafts) - 1

        # A new session has none, and an experiment of the one before it is not
        # one of this session's.
        if self.experiment is not None:
            self.showing = "session"

        self._add_session_action(draft)
        self._rebuild_experiments()
        self._recheck()
        self._show()

    def _remove_session(self, draft: Draft) -> None:
        """
        Throw a session away, asking first if anything was typed into it.

        The draft file goes with it: it is the only copy, so leaving it behind
        would bring the session back the next time the app opens.
        """
        if not draft.is_untouched:
            answer = QMessageBox.question(
                self,
                "Remove this session?",
                f"{draft.label} has been filled in. Removing it discards what "
                f"was typed, and cannot be undone.",
            )

            if answer != QMessageBox.StandardButton.Yes:
                return

        draft.discard()
        self._drop_sessions({draft.path})

    def _drop_sessions(self, gone: set) -> None:
        """Take submitted sessions out of the bar, keeping the rest in place."""
        for draft, action in list(zip(self.drafts, self.session_actions)):
            if draft.path in gone:
                self.bar.removeAction(action)
                self.session_actions.remove(action)
                action.deleteLater()

        self.drafts = [draft for draft in self.drafts if draft.path not in gone]
        self.selected = min(self.selected, max(0, len(self.drafts) - 1))

        if self.experiment is not None and self.experiment not in self._experiments():
            self.showing = "session"

        self._update_badges()
        self._rebuild_experiments()
        self._recheck()
        self._show()

    # ------------------------------------------------------------------ #
    # Experiments
    # ------------------------------------------------------------------ #

    def _add_experiment(self) -> None:
        """Start the next experiment of the selected session, and show it."""
        draft = self.draft

        if draft is None:
            return

        name = _next_experiment(self._experiments())
        draft.set_experiment(name)

        self.showing = name
        self._rebuild_experiments()
        self._recheck()
        self._show()

    def _remove_experiment(self, name: str) -> None:
        """
        Undo adding one.

        Worth having on the form rather than leaving to a rebuild of the draft
        file: an experiment nobody recorded blocks the whole day's submit, and
        the `+` is easy to press twice.
        """
        draft = self.draft

        if draft is None:
            return

        draft.remove_experiment(name)

        self.showing = "session"
        self._rebuild_experiments()
        self._recheck()
        self._show()

    def _identify(self, **named) -> None:
        """Set the animal or the day, and redraw the badge that shows it."""
        draft = self.draft

        if draft is None:
            return

        draft.identify(**named)
        self._update_badges()
        self.notes.retitle()
        self._refresh_status()

    def _show(self) -> None:
        """Put the selected session's chosen form in the stack."""
        while self.stack.count():
            old = self.stack.widget(0)
            self.stack.removeWidget(old)

            # Not dropped on the floor: Qt may still be delivering an event to
            # something inside it, and `deleteLater` waits for that to finish.
            old.deleteLater()

        draft = self.draft

        if draft is not None:
            self.stack.addWidget(self._pane(draft))

        self.notes.show_note(draft, self.experiment)
        self._refresh_status()

    def _pane(self, draft: Draft) -> Pane:
        """The form for whatever the left bar has selected."""
        if self.showing == "panel":
            return PanelPane(draft, self.db)

        name = self.experiment

        if name is None:
            return FormPane(
                form_fields(self.db, "session"),
                self._session_values(draft),
                lambda field, value: self._session_change(draft, field, value),
            )

        return FormPane(
            form_fields(self.db, "experiment"),
            draft.experiment(name) or {},
            lambda field, value: draft.set_experiment(name, **{field.key: value}),
            trailing=self._remove_row(name),
        )

    def _session_values(self, draft: Draft) -> dict:
        """
        What to show on the session form.

        The animal and the day are not annotations -- they say which session
        this is -- so they are held on the draft itself and put back here under
        the keys their columns have.
        """
        return {
            **draft.session,
            "mouse_id": draft.mouse_id,
            "session_date": draft.date,
        }

    def _session_change(self, draft: Draft, field: Field, value) -> None:
        """Send an edited session field wherever that field is kept."""
        if field.key == "mouse_id":
            self._identify(mouse_id=value)
        elif field.key == "session_date":
            self._identify(date=value)
        else:
            draft.update_session(**{field.key: value})

    def _remove_row(self, name: str) -> list[tuple[str, QWidget]]:
        """
        A way back out of a `+` pressed by mistake.

        Pushed to the right of the column, away from the fields: it is the last
        thing on the form and the only one that undoes something, so it should
        not read as another prompt to answer.
        """
        self.remove_button = QPushButton(f"Remove {name}")
        self.remove_button.setFixedWidth(BUTTON_WIDTH)
        self.remove_button.clicked.connect(lambda: self._remove_experiment(name))

        beside = QHBoxLayout()
        beside.setContentsMargins(0, 0, 0, 0)
        beside.addStretch(1)
        beside.addWidget(self.remove_button)

        holder = QWidget()
        holder.setLayout(beside)

        return [("", holder)]

    def _refresh_status(self) -> None:
        draft = self.draft

        if draft is None:
            self.status.setText("No sessions yet — add one to start.")
            return

        # What is still missing is said by the red prompts on the form, which
        # are right beside the box to fill in and update as they are filled.
        lines = []

        if draft.mouse_id is None:
            lines.append("No mouse number yet.")

        elif session_path(self.db.main_folder, draft) is None:
            lines.append("No recordings found yet.")

        self.status.setText("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Submitting
    # ------------------------------------------------------------------ #

    def _submit_day(self) -> None:
        checked = check_all(self.drafts, self.db)
        ready = [result for result in checked if not result.blocked]

        if not ready:
            QMessageBox.warning(self, "Nothing to submit", _report(checked))
            return

        confirm = QMessageBox.question(
            self,
            "Submit the day",
            f"{len(ready)} of {len(checked)} sessions are ready.\n\n"
            f"{_report(checked)}\n\n"
            f"Submitting reads the metadata of every recording, which takes a "
            f"few minutes. Go ahead?",
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return

        results = submit_all([result.draft for result in ready], self.db)

        self._drop_sessions({r.draft.path for r in results if r.submitted})

        QMessageBox.information(self, "Submitted", _report(results))

    # ------------------------------------------------------------------ #
    # Between runs
    # ------------------------------------------------------------------ #

    def _restore_layout(self) -> None:
        """Panel sizes and dock positions, as they were left."""
        saved = QSettings("ODyn", "entry").value("layout")

        if saved is not None:
            self.restoreState(saved)

    def closeEvent(self, event) -> None:
        QSettings("ODyn", "entry").setValue("layout", self.saveState())

        for draft in self.drafts:
            if draft.is_untouched:
                draft.discard()

        event.accept()


def _report(results) -> str:
    """One line per session, saying what happened or what is in the way."""
    lines = []

    for result in results:
        if result.error:
            lines.append(f"✗ {result.draft.label}: {result.error}")
        elif result.submitted:
            written = result.written
            lines.append(
                f"✔ {result.draft.label}: {len(written.experiments)} experiments, "
                f"{written.annotations} annotations"
            )
        elif result.blocked:
            # One line each, rather than joined: they are read down the page,
            # and several joined onto one line wrap into a paragraph.
            lines += [
                f"✗ {problem.where}: {problem.what}"
                for problem in result.problems
                if problem.blocking
            ]
        else:
            lines.append(f"• {result.draft.label}: ready")

    return "\n".join(lines)


def _prompt(field: Field) -> QLabel:
    """
    The prompt beside a field, in red where an answer is expected.

    Colour rather than a marker, and the same colour as the note at the bottom
    of the window that says what it means. A mid red, so it stands out against
    a light and a dark background alike.
    """
    label = QLabel(field.prompt)

    if field.required:
        label.setStyleSheet(f"color: {REQUIRED_COLOR};")

    return label


def _as_text(value) -> str:
    """A stored value as the box should show it."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return "yes" if value else "no"

    return str(value)


def _next_experiment(used: list[str]) -> str:
    """
    The next free `e1`, `e2`, ... for a session.

    The lowest free number rather than one past the highest, so removing an
    experiment and adding another does not leave a gap the recordings will not
    have.
    """
    number = 1

    while f"e{number}" in used:
        number += 1

    return f"e{number}"


def _vial_text(dates: dict) -> str:
    """`{'4': '2026-07-21'}` as the lines someone edits."""
    return "\n".join(f"{position}: {date}" for position, date in sorted(dates.items()))


def run(main_folder: Path | str, *, project: None | str = None) -> int:
    """Open the app on a main folder, with any unfinished sessions restored."""
    from ..database import Database

    application = QApplication.instance() or QApplication([])
    db = Database(main_folder, project=project) if project else Database(main_folder)

    folder = drafts_folder()
    broken = Draft.unreadable(folder)

    if broken:
        QMessageBox.warning(
            None,
            "Some drafts could not be read",
            "These are still on disk and can be recovered by hand:\n\n"
            + "\n".join(f"• {path.name}: {why}" for path, why in broken),
        )

    window = EntryWindow(db, Draft.pending(folder))
    window.show()

    return application.exec()
