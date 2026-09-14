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

from PySide6.QtCore import QSettings, Qt
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
from .submit import check_all, session_folders, submit_all

# A blank dropdown entry, so "not recorded" stays different from a real answer.
UNSET = "—"

# Held in the dock rather than on a form, so it is visible while anything else
# is being filled in.
NOTE = "note"

# Wide enough that a goal or a drug name is readable without resizing.
FIELD_WIDTH = 260


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

        holder = QWidget()
        holder.setLayout(form)

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
        """Point the dock at a session's note, or at one experiment's."""
        self.target = None if draft is None else (draft, experiment)

        held = ""

        if draft is not None:
            where = (
                draft.session
                if experiment is None
                else (draft.experiment(experiment) or {})
            )
            held = where.get(NOTE) or ""

        # Setting the text fires `textChanged`, which would write the note we
        # just loaded back onto whatever is now selected.
        self.loading = True
        self.editor.setPlainText(held)
        self.loading = False

        self.editor.setEnabled(draft is not None)

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

        self.setWindowTitle("ODyn — session entry")
        self.resize(1000, 700)

        self.bar = QToolBar("Sections")
        self.bar.setObjectName("sections")
        self.bar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.bar)

        self.notes = NotesDock()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.notes)

        self.stack = QStackedWidget()

        # Which session, until the mouse icons replace it.
        self.picker = QComboBox()
        self.picker.currentIndexChanged.connect(lambda _: self._show())

        new_session = QPushButton("Add session")
        new_session.clicked.connect(self._add_session)

        self.status = QLabel()
        self.status.setWordWrap(True)

        submit = QPushButton("Check and submit the day")
        submit.clicked.connect(self._submit_day)

        top = QHBoxLayout()
        top.addWidget(QLabel("Session"))
        top.addWidget(self.picker, stretch=1)
        top.addWidget(new_session)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status, stretch=1)
        bottom.addWidget(submit)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.stack, stretch=1)
        layout.addLayout(bottom)

        holder = QWidget()
        holder.setLayout(layout)
        self.setCentralWidget(holder)

        self._build_bar()
        self._refresh_picker()
        self._restore_layout()

    # ------------------------------------------------------------------ #
    # The left bar
    # ------------------------------------------------------------------ #

    def _build_bar(self) -> None:
        """Sections of the bar, separated the way a toolbar separates them."""
        for name, label in (("session", "Session"), ("panel", "Odors")):
            action = self.bar.addAction(label)
            action.setCheckable(True)
            action.setChecked(name == self.showing)
            action.triggered.connect(lambda _, which=name: self._select(which))

        self.bar.addSeparator()

    def _select(self, which: str) -> None:
        self.showing = which

        for action in self.bar.actions():
            if action.text():
                action.setChecked(action.text().lower().startswith(which[:4]))

        self._show()

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #

    @property
    def draft(self) -> None | Draft:
        """The session being edited, or `None` before any has been added."""
        index = self.picker.currentIndex()

        return self.drafts[index] if 0 <= index < len(self.drafts) else None

    def _refresh_picker(self) -> None:
        chosen = self.picker.currentIndex()

        self.picker.blockSignals(True)
        self.picker.clear()
        self.picker.addItems([draft.label for draft in self.drafts])
        self.picker.setCurrentIndex(max(0, min(chosen, len(self.drafts) - 1)))
        self.picker.blockSignals(False)

        self._show()

    def _add_session(self) -> None:
        self.drafts.append(Draft.start(drafts_folder()))
        self._refresh_picker()
        self.picker.setCurrentIndex(len(self.drafts) - 1)

    def _show(self) -> None:
        """Put the selected session's chosen form in the stack."""
        while self.stack.count():
            self.stack.removeWidget(self.stack.widget(0))

        draft = self.draft

        if draft is not None:
            self.stack.addWidget(
                PanelPane(draft, self.db)
                if self.showing == "panel"
                else FormPane(
                    form_fields(self.db, "session"),
                    draft.session,
                    lambda field, value: draft.update_session(**{field.key: value}),
                )
            )

        self.notes.show_note(draft)
        self._refresh_status()

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
        done = {r.draft.path for r in results if r.submitted}
        self.drafts = [draft for draft in self.drafts if draft.path not in done]

        QMessageBox.information(self, "Submitted", _report(results))
        self._refresh_picker()

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
