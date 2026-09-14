"""
The entry window: one session being filled in, saved as it is typed.

Needs Qt, which is an optional dependency -- `pip install -e .[gui]`. Launch it
with `python -m odyn.app MAIN_FOLDER`.

The window holds no state of its own. Every edit goes straight into the `Draft`
and therefore straight to disk, so closing the app mid-session loses nothing
and the next launch offers the day back. What goes on the form comes from
`fields.py`, which builds it from the annotation registry.

Submitting is `submit.submit`: it reports what is wrong before writing, and
writes nothing until there is nothing blocking.
"""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .draft import Draft, drafts_folder
from .fields import (
    BOOLEAN,
    ENUM,
    LONG_TEXT,
    Field,
    form_fields,
    missing_required,
)
from .submit import check_all, session_folders, submit_all

# A blank dropdown entry, so "not recorded" stays different from a real answer.
UNSET = "—"


class FieldRow(QWidget):
    """
    One registry field on the form, drawn to suit its type.

    Emits nothing: it calls `on_change` with the parsed value whenever the box
    is done being edited, and shows the parse error in place when there is one.
    """

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
        """
        A dropdown for a closed set of answers, a box for a block of prose,
        a line for everything else.
        """
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

            return box

        if self.field.value_type == LONG_TEXT:
            box = QPlainTextEdit(_as_text(value))
            box.setMinimumHeight(120)

            # No `editingFinished` on a text area, and a note is typed over the
            # whole session, so it is saved as it is written.
            box.textChanged.connect(lambda: self._changed(box.toPlainText()))

            return box

        box = QLineEdit(_as_text(value))

        # On finishing rather than on every keystroke: a half-typed number is
        # not a parse error worth showing anyone.
        box.editingFinished.connect(lambda: self._changed(box.text()))

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


class SessionTab(QWidget):
    """The day: goal, weight, headplate, the note, whether it is flagged."""

    def __init__(self, draft: Draft, fields: list[Field]):
        super().__init__()

        self.draft = draft
        self.fields = fields

        form = QFormLayout()

        for field in fields:
            form.addRow(
                field.prompt, FieldRow(field, draft.session.get(field.key), self._set)
            )

        holder = QWidget()
        holder.setLayout(form)

        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    def _set(self, field: Field, value) -> None:
        self.draft.update_session(**{field.key: value})


class ExperimentTab(QWidget):
    """One experiment: which objective, how deep, where, what for."""

    def __init__(self, draft: Draft, name: str, fields: list[Field]):
        super().__init__()

        self.draft = draft
        self.name = name

        entry = draft.experiment(name) or {}
        form = QFormLayout()

        objective = QComboBox()
        objective.addItems([UNSET, "10", "20"])
        objective.setCurrentText(_as_text(entry.get("objective")) or UNSET)
        objective.currentTextChanged.connect(self._set_objective)

        # A column rather than an annotation: the micron-per-pixel scale
        # depends on it, and the TIFFs do not record it.
        form.addRow("Objective", objective)

        for field in fields:
            form.addRow(field.prompt, FieldRow(field, entry.get(field.key), self._set))

        holder = QWidget()
        holder.setLayout(form)

        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    def _set(self, field: Field, value) -> None:
        self.draft.set_experiment(self.name, **{field.key: value})

    def _set_objective(self, text: str) -> None:
        self.draft.set_experiment(
            self.name, objective=None if text == UNSET else int(text)
        )


class PanelTab(QWidget):
    """Which rack of vials, and when each was mixed."""

    def __init__(self, draft: Draft, db):
        super().__init__()

        self.draft = draft
        self.db = db

        names = sorted(db.panels["panel_name"]) if len(db.panels) else []

        self.panel = QComboBox()
        self.panel.addItems([UNSET] + names)
        self.panel.setCurrentText(draft.panel.get("panel_name") or UNSET)
        self.panel.currentTextChanged.connect(self._set_panel)

        self.made_on = QLineEdit(draft.panel.get("made_on") or "")
        self.made_on.setPlaceholderText("YYYY-MM-DD, the day the rack was mixed")
        self.made_on.editingFinished.connect(self._set_made_on)

        self.vials = QPlainTextEdit(_vial_text(draft.panel.get("vial_dates", {})))
        self.vials.setPlaceholderText(
            "One per line, for vials mixed apart from the rest:\n4: 2026-07-21"
        )
        self.vials.textChanged.connect(self._set_vials)

        form = QFormLayout(self)
        form.addRow("Panel", self.panel)
        form.addRow("Mixed on", self.made_on)
        form.addRow("Vials mixed separately", self.vials)

    def _set_panel(self, text: str) -> None:
        self.draft.set_panel(panel_name=None if text == UNSET else text)

    def _set_made_on(self) -> None:
        self.draft.set_panel(made_on=self.made_on.text().strip() or None)

    def _set_vials(self) -> None:
        dates = {}

        for line in self.vials.toPlainText().splitlines():
            if ":" not in line:
                continue

            position, _, date = line.partition(":")

            if position.strip().isdigit() and date.strip():
                dates[int(position.strip())] = date.strip()

        self.draft.set_panel(vial_dates=dates)


class NewSessionDialog(QDialog):
    """Which animal, which day. Everything else follows from those two."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("New session")

        self.mouse = QSpinBox()
        self.mouse.setRange(1, 999_999)
        self.mouse.setPrefix("m")

        self.date = QLineEdit(Date.today().isoformat())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout(self)
        form.addRow("Mouse", self.mouse)
        form.addRow("Date", self.date)
        form.addRow(buttons)


class EntryWindow(QMainWindow):
    """
    A day's sessions, open for as long as the day lasts.

    Several mice are run in sequence and their recordings are all copied off at
    the end, so the window holds every draft of the day and a picker chooses
    which one the tabs are showing.
    """

    def __init__(self, db, drafts: list[Draft]):
        super().__init__()

        self.db = db
        self.drafts = list(drafts)

        self.setWindowTitle("ODyn — session entry")
        self.resize(760, 680)

        self.picker = QComboBox()
        self.picker.currentIndexChanged.connect(self._switch_session)

        new_session = QPushButton("Add session")
        new_session.clicked.connect(self._add_session)

        top = QHBoxLayout()
        top.addWidget(QLabel("Session"))
        top.addWidget(self.picker, stretch=1)
        top.addWidget(new_session)

        self.tabs = QTabWidget()

        add = QPushButton("Add experiment")
        add.clicked.connect(self._add_experiment)

        self.submit_button = QPushButton("Check and submit the day")
        self.submit_button.clicked.connect(self._submit_day)

        self.status = QLabel()
        self.status.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addStretch()
        buttons.addWidget(self.submit_button)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

        holder = QWidget()
        holder.setLayout(layout)

        self.setCentralWidget(holder)
        self._refresh_picker()

    @property
    def draft(self) -> Draft:
        """The session the tabs are showing."""
        return self.drafts[max(0, self.picker.currentIndex())]

    def _refresh_picker(self) -> None:
        """Rebuild the session list, keeping whichever one was selected."""
        chosen = self.picker.currentIndex()

        self.picker.blockSignals(True)
        self.picker.clear()
        self.picker.addItems([draft.label for draft in self.drafts])
        self.picker.setCurrentIndex(max(0, min(chosen, len(self.drafts) - 1)))
        self.picker.blockSignals(False)

        self._switch_session()

    def _switch_session(self) -> None:
        self._build_tabs()
        self._refresh_status()

    def _add_session(self) -> None:
        dialog = NewSessionDialog(self)

        while dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                draft = Draft.start(
                    self.draft.path.parent,
                    mouse_id=dialog.mouse.value(),
                    date=dialog.date.text(),
                )
            except ValueError as wrong:
                QMessageBox.warning(dialog, "Try again", str(wrong))
                continue

            if any(other.path == draft.path for other in self.drafts):
                QMessageBox.information(
                    self, "Already open", "That session is already in the list."
                )
                return

            # Saved immediately so it survives a crash before anything is typed.
            self.drafts.append(draft.save())
            self._refresh_picker()
            self.picker.setCurrentIndex(len(self.drafts) - 1)

            return

    def _build_tabs(self) -> None:
        self.tabs.clear()
        self.tabs.addTab(
            SessionTab(self.draft, form_fields(self.db, "session")), "Session"
        )

        experiment_fields = form_fields(self.db, "experiment")

        for entry in self.draft.experiments:
            name = entry["name"]
            self.tabs.addTab(ExperimentTab(self.draft, name, experiment_fields), name)

        self.tabs.addTab(PanelTab(self.draft, self.db), "Odors")

    def _add_experiment(self) -> None:
        """Named by position: the folders are `e1`, `e2` and so on."""
        taken = {entry["name"] for entry in self.draft.experiments}
        number = 1

        while f"e{number}" in taken:
            number += 1

        self.draft.set_experiment(f"e{number}")
        self._build_tabs()
        self.tabs.setCurrentIndex(self.tabs.count() - 2)
        self._refresh_status()

    def _refresh_status(self) -> None:
        """What is still missing, shown before anyone presses submit."""
        lines = []

        wanted = missing_required(form_fields(self.db, "session"), self.draft.session)

        if wanted:
            lines.append("Still to fill in: " + ", ".join(f.label for f in wanted))

        if not session_folders(self.db.main_folder, self.draft):
            lines.append(
                "No recordings found for this session yet — they are usually "
                "copied over at the end of the day."
            )

        self.status.setText("\n".join(lines))

    def _submit_day(self) -> None:
        """
        Check every session, show what is wrong, then submit what is ready.

        Reading the TIFF metadata of a day's recordings takes a while, so the
        check is shown and confirmed first rather than run into.
        """
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

        # Submitted drafts were archived, so only the rest stay in the window.
        self.drafts = [
            draft
            for draft in self.drafts
            if not any(r.draft.path == draft.path and r.submitted for r in results)
        ]

        QMessageBox.information(self, "Submitted", _report(results))

        if not self.drafts:
            self.close()
            return

        self._refresh_picker()

    def closeEvent(self, event) -> None:
        """Nothing to save on the way out: every edit was saved as it happened."""
        for draft in self.drafts:
            if draft.is_empty:
                draft.discard()

        event.accept()


def _report(results) -> str:
    """One line per session, saying what happened or what is in the way."""
    lines = []

    for result in results:
        who = result.draft.label

        if result.error:
            lines.append(f"✗ {who}: {result.error}")
        elif result.submitted:
            written = result.written
            lines.append(
                f"✔ {who}: {len(written.experiments)} experiments, "
                f"{written.annotations} annotations"
            )
        elif result.blocked:
            reasons = "; ".join(
                problem.what for problem in result.problems if problem.blocking
            )
            lines.append(f"✗ {who}: {reasons}")
        else:
            lines.append(f"• {who}: ready")

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
    """
    Open the app on a main folder, offering any unfinished draft first.

    Returns Qt's exit code.
    """
    from ..database import Database

    application = QApplication.instance() or QApplication([])

    db = Database(main_folder, project=project) if project else Database(main_folder)

    folder = drafts_folder()
    unfinished = Draft.pending(folder)
    broken = Draft.unreadable(folder)

    if broken:
        QMessageBox.warning(
            None,
            "Some drafts could not be read",
            "These are still on disk and can be recovered by hand:\n\n"
            + "\n".join(f"• {path.name}: {why}" for path, why in broken),
        )

    if not unfinished:
        first = _ask_for_session(folder)

        if first is None:
            return 0

        unfinished = [first.save()]

    window = EntryWindow(db, unfinished)
    window.show()

    return application.exec()


def _ask_for_session(folder: Path) -> None | Draft:
    """Ask which animal and day, and open that draft."""
    dialog = NewSessionDialog()

    while dialog.exec() == QDialog.DialogCode.Accepted:
        try:
            return Draft.start(
                folder, mouse_id=dialog.mouse.value(), date=dialog.date.text()
            )
        except ValueError as wrong:
            QMessageBox.warning(dialog, "Try again", str(wrong))

    return None
