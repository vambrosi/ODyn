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
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .draft import Draft, drafts_folder
from .fields import BOOLEAN, ENUM, LONG_TEXT, Field, form_fields, missing_required
from .icons import icon, label_icon
from .submit import check_all, session_folders, submit_all

# A blank dropdown entry, so "not recorded" stays different from a real answer.
UNSET = "—"

# Held in the dock rather than on a form, so it is visible while anything else
# is being filled in.
NOTE = "note"

# A line of text much wider than this is tiring to read: the eye loses the
# start of the next line. The form sits in a column of this width, centred,
# rather than stretching with the window.
COLUMN_WIDTH = 620
FIELD_WIDTH = 260

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
        """A dropdown for a closed answer, a box for prose, a line otherwise."""
        if self.field.value_type in (BOOLEAN, ENUM):
            box = QComboBox()
            box.addItem(UNSET)
            box.addItems(
                ["yes", "no"]
                if self.field.value_type == BOOLEAN
                else list(self.field.options)
            )
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


class FormPane(QScrollArea):
    """
    A scrolling form built from registry fields.

    `values` is what to show, `on_change(field, value)` is called as each is
    edited. The note is left out: the dock owns it.
    """

    def __init__(self, fields: list[Field], values: dict, on_change, extra=()):
        super().__init__()

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        for label, widget in extra:
            form.addRow(label, widget)

        for field in fields:
            if field.key == NOTE:
                continue

            form.addRow(field.prompt, FieldRow(field, values.get(field.key), on_change))

        column = QWidget()
        column.setLayout(form)
        column.setMaximumWidth(COLUMN_WIDTH)

        # Centred, so the form keeps its width as the window grows and the
        # space goes to the margins instead of to the lines.
        centred = QHBoxLayout()
        centred.addStretch()
        centred.addWidget(column)
        centred.addStretch()

        holder = QWidget()
        holder.setLayout(centred)

        self.setWidget(holder)
        self.setWidgetResizable(True)

        # Otherwise the scroll area draws a border and the form inside draws
        # another, which reads as a box inside a box.
        self.setFrameShape(QScrollArea.Shape.NoFrame)


class PanelPane(QWidget):
    """Which rack of vials, and when each was mixed."""

    def __init__(self, draft: Draft, db):
        super().__init__()

        self.draft = draft
        names = sorted(db.panels["panel_name"]) if len(db.panels) else []

        self.panel = QComboBox()
        self.panel.addItems([UNSET] + names)
        self.panel.setCurrentText(draft.panel.get("panel_name") or UNSET)
        self.panel.currentTextChanged.connect(
            lambda text: draft.set_panel(panel_name=None if text == UNSET else text)
        )

        self.made_on = QLineEdit(draft.panel.get("made_on") or "")
        self.made_on.setPlaceholderText("YYYY-MM-DD, the day the rack was mixed")
        self.made_on.editingFinished.connect(self._set_made_on)

        self.vials = QPlainTextEdit(_vial_text(draft.panel.get("vial_dates", {})))
        self.vials.setPlaceholderText(
            "One per line, for vials mixed apart from the rest:\n4: 2026-07-21"
        )
        self.vials.textChanged.connect(self._set_vials)

        for box in (self.panel, self.made_on):
            box.setMinimumWidth(FIELD_WIDTH)

        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Panel", self.panel)
        form.addRow("Mixed on", self.made_on)
        form.addRow("Vials mixed separately", self.vials)

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
        about = "Session" if experiment is None else experiment.upper()

        self.setWindowTitle(f"{about} notes — {draft.label}")

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

        self.setWindowTitle("ODyn — session entry")
        self.resize(1000, 700)
        self.setStyleSheet(STYLE)

        self.bar = QToolBar("Sections")
        self.bar.setObjectName("sections")
        self.bar.setMovable(False)
        self.bar.setIconSize(QSize(26, 26))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.bar)

        self.notes = NotesDock()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.notes)

        # Its own bar on the other edge, so a closed dock can be brought back.
        # Without it, closing the notes would hide them for good.
        self.panels = QToolBar("Panels")
        self.panels.setObjectName("panels")
        self.panels.setMovable(False)
        self.panels.setIconSize(QSize(26, 26))
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

        submit = QPushButton("Check and submit the day")
        submit.clicked.connect(self._submit_day)

        bottom = QHBoxLayout()
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

        self.bar.addSeparator()

        self.add_action = self.bar.addAction(icon("add"), "Add session")
        self.add_action.setToolTip("Add session")
        self.add_action.triggered.connect(self._add_session)

        for draft in self.drafts:
            self._add_session_action(draft)

        self._show()

    def _add_session_action(self, draft: Draft) -> None:
        """One session icon, inserted just before the `+` that adds them."""
        action = QAction(label_icon("?"), draft.label, self)
        action.setCheckable(True)
        action.triggered.connect(lambda _, which=draft: self._choose(which))

        self.bar.insertAction(self.add_action, action)
        self.session_actions.append(action)

        self._update_badges()

    def _select(self, which: str) -> None:
        """Choose what is being edited for the current session."""
        self.showing = which

        for name, action in self.section_actions.items():
            action.setChecked(name == which)

        self._show()

    def _choose(self, which: int | Draft) -> None:
        """Choose which session the forms and the note are about."""
        self.selected = which if isinstance(which, int) else self.drafts.index(which)

        self._update_badges()
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
            badge = draft.mouse_id if draft.mouse_id is not None else index + 1

            action.setIcon(label_icon(str(badge)))
            action.setToolTip(draft.label)
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

    def _add_session(self) -> None:
        draft = Draft.start(drafts_folder())

        self.drafts.append(draft)
        self.selected = len(self.drafts) - 1
        self._add_session_action(draft)
        self._show()

    def _drop_sessions(self, gone: set) -> None:
        """Take submitted sessions out of the bar, keeping the rest in place."""
        for draft, action in list(zip(self.drafts, self.session_actions)):
            if draft.path in gone:
                self.bar.removeAction(action)
                self.session_actions.remove(action)
                action.deleteLater()

        self.drafts = [draft for draft in self.drafts if draft.path not in gone]
        self.selected = min(self.selected, max(0, len(self.drafts) - 1))

        self._update_badges()
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
            self.stack.addWidget(
                PanelPane(draft, self.db)
                if self.showing == "panel"
                else FormPane(
                    form_fields(self.db, "session"),
                    draft.session,
                    lambda field, value: draft.update_session(**{field.key: value}),
                    extra=self._identity_rows(draft),
                )
            )

        self.notes.show_note(draft)
        self._refresh_status()

    def _identity_rows(self, draft: Draft) -> list[tuple[str, QWidget]]:
        """
        Which animal and which day, above the rest of the session form.

        Not annotations: they say which session this is, and the mouse is what
        the left bar puts on the badge.
        """
        mouse = QLineEdit("" if draft.mouse_id is None else str(draft.mouse_id))
        mouse.setPlaceholderText("the number, as in 442")
        mouse.setMinimumWidth(FIELD_WIDTH)
        mouse.editingFinished.connect(lambda: self._set_mouse(mouse))

        day = QLineEdit(draft.date)
        day.setMinimumWidth(FIELD_WIDTH)
        day.editingFinished.connect(lambda: self._set_date(day))

        return [("Mouse ID", mouse), ("Date", day)]

    def _set_mouse(self, box: QLineEdit) -> None:
        text = box.text().strip().lstrip("mM")

        if text.isdigit() and int(text) > 0:
            self._identify(mouse_id=int(text))

    def _set_date(self, box: QLineEdit) -> None:
        try:
            self._identify(date=box.text())
        except ValueError:
            box.setText("" if self.draft is None else self.draft.date)

    def _refresh_status(self) -> None:
        draft = self.draft

        if draft is None:
            self.status.setText("No sessions yet — add one to start.")
            return

        lines = []
        wanted = missing_required(form_fields(self.db, "session"), draft.session)

        if wanted:
            lines.append("Still to fill in: " + ", ".join(f.label for f in wanted))

        if draft.mouse_id is None:
            lines.append("No mouse number yet.")
        elif not session_folders(self.db.main_folder, draft):
            lines.append(
                "No recordings found yet — they are usually copied over at the "
                "end of the day."
            )

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
            if draft.is_empty:
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
            reasons = "; ".join(
                problem.what for problem in result.problems if problem.blocking
            )
            lines.append(f"✗ {result.draft.label}: {reasons}")
        else:
            lines.append(f"• {result.draft.label}: ready")

    return "\n".join(lines)


def _as_text(value) -> str:
    """A stored value as the box should show it."""
    if value is None:
        return ""

    if isinstance(value, bool):
        return "yes" if value else "no"

    return str(value)


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
