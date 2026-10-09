"""
The day's work: one draft per session, filled in as the part runs and
submitted at the end.

Two pages show the selected draft (`Context.draft`): the **session page**
(mouse, date, folder, the session's annotations, the odor panel) and the
**part page**, one part of the session: a loop (`e1`: its settings,
annotations, conditions and trial block calculator), or its grabs or z-stacks
(the objective, and each TIFF found in the folder with its header values and
notes). `Sessions` owns the drafts and the actions that both the pages and the
window's right bar call; what Check and Submit find goes out through
`Sessions.reported`.
"""

from __future__ import annotations

import re

import pandas as pd

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..utils import CHECK, CROSS
from .context import Context
from .draft import FILE_SETTINGS, FOLDER_OF, PART_SETTINGS, Draft, Removals
from .fields import Field, annotation_fields, form_fields
from .forms import Form, FormRows, GrabTimingForm, PanelPicker, TimingForm, fit_row
from .models import FrameModel, fit_columns
from .pane_registry import current_vials
from .recordings import make_part_folders
from .style import COMBO_EXTRA_PX, PAGE_MAX_PX, required_label, text_style
from .submit import Problem, check, list_tiffs, report, running, submit

TIMING_KEYS = (
    "block_baseline_s",
    "block_odor_s",
    "block_post_odor_delay_s",
    "block_iti_s",
    "block_post_odor_acq_s",
    "block_n_odors",
    "block_n_trials",
)


def centered(inner: QWidget) -> QScrollArea:
    """`inner` in a scroll area, no wider than `PAGE_MAX_PX` and centered."""
    inner.setMaximumWidth(PAGE_MAX_PX)
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    # The page grows first; the stretches only take what is past its maximum
    row.addStretch(0)
    row.addWidget(inner, 1)
    row.addStretch(0)

    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(holder)

    return area


def session_label(draft: Draft, position: int) -> str:
    """How the right bar names a session: its mouse, or `s<n>` before one."""
    return f"s{position + 1}" if draft.mouse_id is None else f"m{draft.mouse_id}"


def counted(count: int, word: str) -> str:
    """`1 loop`, `3 loops`."""
    return f"{count} {word}{'' if count == 1 else 's'}"


# The highest acquisition (TIFF) number a condition can name
MAX_ACQUISITION = 99999


def _column_widths(choices: list[str]) -> tuple[int, int, int]:
    """
    Fixed widths for a condition row's columns, so the header's names sit
    over them: a number box fits five digits, the dropdown its longest
    choice, the last column a remove button.
    """
    metrics = QLabel().fontMetrics()
    number = metrics.horizontalAdvance("99999") + COMBO_EXTRA_PX
    widest = max([metrics.horizontalAdvance(c) for c in choices] or [0])
    condition = max(widest, metrics.horizontalAdvance("Condition")) + COMBO_EXTRA_PX

    return number, condition, QPushButton("remove").sizeHint().width()


def _condition_columns(widgets, widths: tuple[int, int, int]) -> QHBoxLayout:
    """A condition row's columns (or their header's names), in one layout."""
    number, condition, last = widths
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)

    for widget, width in zip(widgets, (number, number, condition, last)):
        widget.setFixedWidth(width)
        row.addWidget(widget)

    row.addStretch(1)

    return row


class ConditionHeader(QWidget):
    """The names over the condition rows' columns."""

    def __init__(self, choices: list[str], parent=None):
        super().__init__(parent)
        names = [QLabel(text) for text in ("Starts at", "Ends at", "Condition")]

        for name in names:
            name.setStyleSheet(text_style("computed"))

        self.setLayout(_condition_columns([*names, QWidget()], _column_widths(choices)))


class ConditionRow(QWidget):
    """
    One range of acquisitions, by the number ScanImage ends their TIFF
    names with (both ends included), and the condition the animal was in.
    """

    changed = Signal()

    def __init__(self, choices: list[str], parent=None):
        super().__init__(parent)
        self.start = QSpinBox()
        self.end = QSpinBox()

        for box in (self.start, self.end):
            box.setRange(1, MAX_ACQUISITION)
            box.valueChanged.connect(lambda _: self.changed.emit())

        self.condition = QComboBox()
        self.condition.addItems(choices)
        self.condition.currentTextChanged.connect(lambda _: self.changed.emit())
        self.remove = QPushButton("remove")
        widgets = (self.start, self.end, self.condition, self.remove)
        self.setLayout(_condition_columns(widgets, _column_widths(choices)))

        height = max(w.sizeHint().height() for w in (self.start, self.condition))

        for widget in (self.start, self.end, self.condition):
            widget.setFixedHeight(height)

        fit_row(widgets, height)

    def set_value(self, value: dict) -> None:
        for widget in (self.start, self.end, self.condition):
            widget.blockSignals(True)

        self.start.setValue(int(value.get("start", 1)))
        self.end.setValue(int(value.get("end", 1)))
        condition = str(value.get("condition", ""))

        if condition and self.condition.findText(condition) < 0:
            self.condition.addItem(condition)  # no longer offered: still shown

        self.condition.setCurrentText(condition)

        for widget in (self.start, self.end, self.condition):
            widget.blockSignals(False)

    def value(self) -> dict:
        return {
            "start": self.start.value(),
            "end": self.end.value(),
            "condition": self.condition.currentText(),
        }


class ConditionList(FormRows):
    """
    A loop's conditions: which acquisitions were recorded at baseline
    and which under anesthesia, saved to the draft as they are typed.
    """

    def __init__(self, form, choices: list[str], saved, on_change):
        super().__init__(form, "Conditions", "none noted", ConditionHeader(choices))
        self.choices = choices
        self.on_change = on_change

        for value in saved:
            self.add(value, announce=False)

    def add(self, value: None | dict = None, announce: bool = True) -> None:
        """A row: `value`, or the range after the last one, next condition."""
        if value is None:
            last = self.rows[-1].value() if self.rows else None
            start = 1 if last is None else min(last["end"] + 1, MAX_ACQUISITION)
            following = (
                0
                if last is None or last["condition"] not in self.choices
                else (self.choices.index(last["condition"]) + 1) % len(self.choices)
            )
            value = {"start": start, "end": start, "condition": self.choices[following]}

        row = ConditionRow(self.choices)
        row.set_value(value)
        row.changed.connect(self._changed)
        row.remove.clicked.connect(lambda: self.remove(row))
        self.insert(row)

        if announce:
            self._changed()

    def remove(self, row: ConditionRow) -> None:
        self.take(row)
        self._changed()

    def values(self) -> list[dict]:
        return [row.value() for row in self.rows]

    def _changed(self) -> None:
        self.on_change(self.values())


class LoopForm(QWidget):
    """
    One loop of a session: its settings, annotations and conditions, and the
    trial block calculator.
    """

    def __init__(
        self,
        draft: Draft,
        name: str,
        fields: list[Field],
        conditions: tuple[str, ...] = (),
        parent=None,
    ):
        super().__init__(parent)
        self.draft = draft
        self.name = name
        self.conditions: None | ConditionList = None

        entry = draft.part(name) or {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # The objective is the form's first row; the block keys are drawn by
        # the calculator instead of as plain rows
        plain = [f for f in fields if f.key not in TIMING_KEYS]
        self.form = Form(plain)
        self.form.set_values(
            {k: v for k, v in entry.items() if k not in PART_SETTINGS}
            | {"objective": entry.get("objective")}
        )
        self.form.changed.connect(lambda key, value: self._set(**{key: value}))

        left = QVBoxLayout()
        left.addWidget(self.form)

        # Conditions are per acquisition, which only loops have one of each
        if entry.get("kind", "loop") == "loop" and conditions:
            self.conditions = ConditionList(
                self.form.layout(),
                list(conditions),
                entry.get("conditions", []),
                lambda values: self._set(conditions=values or None),
            )
            add = QPushButton("Add condition")
            add.clicked.connect(lambda: self.conditions.add())
            left.addWidget(add)

        self.timing = TimingForm()
        self.timing.set_values(entry)
        self.timing.changed.connect(lambda: self._set(**self.timing.values()))

        row = QHBoxLayout()
        row.addLayout(left, 1)
        row.addWidget(self.timing, 0, Qt.AlignmentFlag.AlignTop)
        left.addStretch(1)
        layout.addLayout(row)

        self.submitted = QLabel()
        self.submitted.setStyleSheet("color: #2a7;")
        layout.addWidget(self.submitted)
        self.refresh_status()

    def refresh_status(self) -> None:
        entry = self.draft.part(self.name) or {}
        loop_id = entry.get("submitted_id")
        self.submitted.setText(
            "" if loop_id is None else f"Already in the database as loop_id {loop_id}"
        )

    def _set(self, **fields) -> None:
        self.draft.set_part(self.name, **fields)


# The rows a TIFF table shows before it scrolls
TIFF_ROWS = 8


class FilesForm(QWidget):
    """
    A session's grabs or z-stacks: the objective they were recorded with, and
    each TIFF found in the folder on the share, with what its header says and
    what is noted about it (a z-stack also names the loop it was associated
    with). Notes are kept per file name, so a file noted before it reached the
    share shows up once it does.
    """

    def __init__(
        self,
        context: Context,
        draft: Draft,
        name: str,
        fields: list[Field],
        parent=None,
    ):
        super().__init__(parent)
        self.context = context
        self.draft = draft
        self.name = name
        self.kind = (draft.part(name) or {}).get("kind", "grab")
        self.file: None | str = None
        self.file_form: None | Form = None
        self.loop: None | QComboBox = None

        objective = [f for f in fields if f.key == "objective"]
        self.file_fields = [f for f in fields if f.key != "objective"]

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout()
        outer.addLayout(layout, 1)

        # Grabs may run with a sync recording, whose camera needs a frame count
        self.timing: None | GrabTimingForm = None

        if self.kind == "grab":
            self.timing = GrabTimingForm()
            outer.addWidget(self.timing, 0, Qt.AlignmentFlag.AlignTop)

        self.form = Form(objective)
        self.form.set_values({"objective": (draft.part(name) or {}).get("objective")})
        self.form.changed.connect(
            lambda key, value: self.draft.set_part(self.name, **{key: value})
        )
        layout.addWidget(self.form)

        top = QHBoxLayout()
        self.found = QLabel()
        self.found.setStyleSheet(text_style("computed"))
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        top.addWidget(self.found, 1)
        top.addWidget(refresh)
        layout.addLayout(top)

        self.table = QTableView()
        self.table.setModel(FrameModel())
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        fit_columns(self.table)
        rows = self.table.verticalHeader().defaultSectionSize()
        self.table.setFixedHeight(
            rows * TIFF_ROWS + self.table.horizontalHeader().sizeHint().height() + 2
        )
        self.table.selectionModel().currentRowChanged.connect(
            lambda current, _: self._pick(current.row())
        )
        layout.addWidget(self.table)

        self.notes = QGroupBox("Pick a file above to write notes")
        self.notes_layout = QVBoxLayout(self.notes)
        layout.addWidget(self.notes)
        self.refresh()

    @property
    def model(self) -> FrameModel:
        model = self.table.model()
        assert isinstance(model, FrameModel)

        return model

    def refresh(self) -> None:
        """Read the folder again (on the database thread: it is on the share)."""
        draft = self.draft

        if draft.mouse_id is None:
            return

        folder = f"{draft.folder}/{self.name}"
        self.found.setText(f"Reading {folder}/raw...")
        self.context.run(
            lambda db: list_tiffs(db.main_folder, folder, self.kind),
            self._listed,
            "list TIFFs",
            owner=self,
        )

    def _listed(self, frame) -> None:
        noted = set(self.draft.files(self.name))
        missing = sorted(noted - set(frame["file"]))

        if missing:
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame({"file": missing, "started": "not on the server"}),
                ],
                ignore_index=True,
            )

        frame["notes"] = ["yes" if f in noted else "" for f in frame["file"]]
        count = len(frame) - len(missing)
        self.found.setText(
            f"{count} TIFF{'' if count == 1 else 's'} in "
            f"{self.draft.folder}/{self.name}/raw"
        )
        keep = self.file
        files = list(frame["file"])
        self.model.set_frame(frame.set_index("file"))  # its first column

        if keep in files:
            self.table.selectRow(files.index(keep))

    def _pick(self, row: int) -> None:
        frame = self.model.frame

        if row < 0 or row >= len(frame):
            return

        self.file = str(frame.index[row])
        self._show_notes()

    def _show_notes(self) -> None:
        if self.file_form is not None:
            self.file_form.setParent(None)
            self.file_form.deleteLater()

        self.file_form = self.loop = None
        file = self.file

        if file is None:
            return

        noted = self.draft.files(self.name).get(file, {})
        self.notes.setTitle(file)

        self.file_form = Form(self.file_fields)
        self.file_form.set_values(
            {k: v for k, v in noted.items() if k not in FILE_SETTINGS}
        )
        self.file_form.changed.connect(lambda key, value: self._note(**{key: value}))

        if self.kind == "zstack":
            # The form's first row, so its label lines up with the others
            loops = [
                str(e["name"]) for e in self.draft.parts if e.get("kind") == "loop"
            ]
            self.loop = QComboBox()
            self.loop.addItems(["", *loops])
            self.loop.setCurrentText(str(noted.get("loop", "")))
            self.loop.currentTextChanged.connect(
                lambda text: self._note(loop=text or None)
            )
            layout = self.file_form.layout()
            assert isinstance(layout, QFormLayout)
            layout.insertRow(0, "Associated with the loop", self.loop)

        self.notes_layout.addWidget(self.file_form)

    def _note(self, **fields) -> None:
        if self.file is None:
            return

        self.draft.set_file(self.name, self.file, **fields)
        notes = self.model.frame

        if "notes" in notes and self.file in notes.index:
            notes.loc[self.file, "notes"] = "yes"
            self.model.layoutChanged.emit()

    def refresh_status(self) -> None:
        """Nothing to say after a submit that kept the draft: the table says it."""


class SessionPage(QWidget):
    """The selected draft's session: who, when, what was noted, which vials."""

    def __init__(self, sessions: Sessions, parent=None):
        super().__init__(parent)
        self.sessions = sessions
        self.context = sessions.context
        self.draft: None | Draft = None
        self.form: None | Form = None

        inner = QWidget()
        layout = QVBoxLayout(inner)

        # The session's identity, as one row
        self.identity = QGroupBox("Session")
        row = QHBoxLayout(self.identity)
        self.mouse = QComboBox()
        self.mouse.setEditable(True)
        self.mouse.setMinimumWidth(120)
        self.mouse.currentTextChanged.connect(self._mouse_changed)
        self.date = QLineEdit()
        self.date.setPlaceholderText("YYYY-MM-DD")
        self.date.setMaximumWidth(130)
        self.date.editingFinished.connect(self._date_changed)
        self.folder = QLabel()
        self.folder.setStyleSheet(text_style("computed"))

        for label, widget in (
            (required_label("Mouse"), self.mouse),
            (required_label("Date"), self.date),
            (QLabel("Folder"), self.folder),
        ):
            row.addWidget(label)
            row.addWidget(widget)
            row.addSpacing(18)

        row.addStretch(1)
        layout.addWidget(self.identity)

        self.info = QGroupBox("Info")
        self.info_layout = QVBoxLayout(self.info)
        layout.addWidget(self.info)

        vials = QGroupBox("Odor panel")
        vials_layout = QVBoxLayout(vials)
        self.vial_list = PanelPicker()
        self.vial_list.changed.connect(self._vials_changed)
        vials_layout.addWidget(self.vial_list)
        layout.addWidget(vials)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(centered(inner))
        self.body = inner
        self.show_draft(None)

    def show_draft(self, draft: None | Draft) -> None:
        """Fill the page from `draft`; with None, an empty, disabled page."""
        self.draft = draft
        self.body.setEnabled(draft is not None)

        self.mouse.blockSignals(True)
        self.mouse.clear()
        self.mouse.addItems(["", *[f"m{m}" for m in self.sessions.mice]])
        self.mouse.setCurrentText(
            "" if draft is None or draft.mouse_id is None else f"m{draft.mouse_id}"
        )
        self.mouse.blockSignals(False)
        self.date.setText("" if draft is None else draft.date)
        self._refresh_identity()

        if self.form is not None:
            self.form.setParent(None)
            self.form.deleteLater()
            self.form = None

        if draft is not None:
            self.form = Form(self.sessions.session_fields)
            self.form.set_values(draft.session)
            self.form.changed.connect(
                lambda key, value: draft.update_session(**{key: value})
            )
            self.info_layout.addWidget(self.form)

        self.vial_list.blockSignals(True)
        self.vial_list.set_panels(self.sessions.panels, self.sessions.vials)
        self.vial_list.show_vials([] if draft is None else draft.vial_ids)
        self.vial_list.blockSignals(False)

    def _refresh_identity(self) -> None:
        draft = self.draft

        # Folders and recordings are named and placed for the mouse and the day
        locked = draft is not None and draft.fixed
        why = (
            "Cannot be changed after folder or recording was created" if locked else ""
        )

        for widget in (self.mouse, self.date):
            widget.setEnabled(not locked)
            widget.setToolTip(why)

        if draft is None:
            self.identity.setTitle("No session selected")
            self.folder.setText("")
        elif draft.mouse_id is None:
            self.identity.setTitle(f"Session {draft.where}")
            self.folder.setText("pick a mouse first")
        else:
            self.identity.setTitle(f"Session {draft.where}")
            self.folder.setText(draft.folder)

    def _mouse_changed(self, text: str) -> None:
        if self.draft is None:
            return

        text = text.strip().lstrip("mM")

        try:
            mouse_id = int(text) if text else None
        except ValueError:
            return

        if mouse_id is not None:
            try:
                self.draft.identify(mouse_id=mouse_id)
            except ValueError as error:
                self.context.report("Mouse", str(error))
                return

            self._refresh_identity()
            self.context.announce_draft()

    def _date_changed(self) -> None:
        if self.draft is None or self.date.text() == self.draft.date:
            return

        try:
            self.draft.identify(date=self.date.text())
        except ValueError as error:
            self.context.report("Date", str(error))
            return

        self._refresh_identity()
        self.context.announce_draft()

    def _vials_changed(self) -> None:
        if self.draft is not None:
            self.draft.set_vials(self.vial_list.checked())


class NoSessionPage(QWidget):
    """What the middle shows when no session is selected."""

    MESSAGE = "No session selected\nPick one or add one using the right bar buttons"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.message = QLabel(self.MESSAGE)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        font = self.message.font()
        font.setPointSizeF(font.pointSizeF() * 1.3)
        self.message.setFont(font)
        self.message.setStyleSheet(text_style("computed"))
        layout = QVBoxLayout(self)
        layout.addWidget(self.message)


class PartPage(QWidget):
    """One part of the selected draft: a loop, or its grabs or z-stacks."""

    TITLES = {"loop": "Loop", "grab": "Grabs", "zstack": "Z-stacks"}

    def __init__(self, sessions: Sessions, parent=None):
        super().__init__(parent)
        self.sessions = sessions
        self.draft: None | Draft = None
        self.name: None | str = None
        self.shown: None | LoopForm | FilesForm = None

        inner = QWidget()
        layout = QVBoxLayout(inner)
        self.box = QGroupBox("Loop")
        self.box_layout = QVBoxLayout(self.box)
        layout.addWidget(self.box)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(centered(inner))

    def title(self, draft: Draft, name: str) -> str:
        """`Loop 20260929/m442/e1`, `Grabs 20260929/m442/grabs`..."""
        kind = (draft.part(name) or {}).get("kind", "loop")

        return f"{self.TITLES.get(kind, 'Experiment')} {draft.where}/{name}"

    def show_part(self, draft: None | Draft, name: None | str) -> None:
        """Fill the page from part `name` of `draft`."""
        if self.shown is not None:
            self.shown.setParent(None)
            self.shown.deleteLater()
            self.shown = None

        self.draft, self.name = draft, name

        if draft is None or name is None or draft.part(name) is None:
            self.box.setTitle("Loop")
            return

        self.box.setTitle(self.title(draft, name))
        kind = (draft.part(name) or {}).get("kind", "loop")

        if kind == "loop":
            self.shown = LoopForm(
                draft,
                name,
                self.sessions.loop_fields,
                self.sessions.condition_choices,
            )
        else:
            self.shown = FilesForm(
                self.sessions.context,
                draft,
                name,
                self.sessions.file_fields.get(kind, []),
            )

        self.box_layout.addWidget(self.shown)


class Sessions(QObject):
    """
    The drafts in progress and what can be done to them.

    `drafts` is in the order the sessions were started. The pages and the
    window's right bar call `new`, `remove`, `add_loop`, `add_grabs`,
    `add_zstacks`, `remove_part`, `undo`, `check` and `submit`; each acts on
    the selected draft (`Context.draft`). Removing only marks the draft or the
    part removed and pushes it on `removals`, the undo stack.

    - `listed` fires when drafts come or go (the right bar rebuilds);
    - `removals_changed` fires when the undo stack may have changed;
    - `reported(text)` carries what Check or Submit found;
    - `part_added(name)` fires once a new part is selected.
    """

    listed = Signal()
    removals_changed = Signal()
    reported = Signal(str)
    part_added = Signal(str)

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.drafts: list[Draft] = []
        self.removals = Removals(context.drafts)
        self.session_fields: list[Field] = []
        self.loop_fields: list[Field] = []
        self.file_fields: dict[str, list[Field]] = {}
        self.condition_choices: tuple[str, ...] = ()
        self.mice: list[int] = []
        self.panels: list[tuple[int, str]] = []
        self.vials = None

        self.session_page = SessionPage(self)
        self.part_page = PartPage(self)
        self.no_session_page = NoSessionPage()

        context.changed.connect(self.reload_choices)
        context.selection_changed.connect(self._follow_selection)
        context.draft_changed.connect(self._follow_draft)
        self.reload_choices()
        self.reload_drafts()

    @property
    def current(self) -> None | Draft:
        """The selected draft, when it is one of these."""
        draft = self.context.draft

        return draft if any(draft is d for d in self.drafts) else None

    def position(self, draft: Draft) -> int:
        return next(i for i, d in enumerate(self.drafts) if d is draft)

    def label(self, draft: Draft) -> str:
        return session_label(draft, self.position(draft))

    # ----------------------------------------------------------------- #
    # Loading
    # ----------------------------------------------------------------- #

    def reload_choices(self) -> None:
        """What the forms offer: the registry's fields, the mice, the vials."""

        def load(db):
            vials = db.vials.join(db.panels["panel_name"], on="panel_id")
            vials = vials.join(db.odors["odor_name"], on="odor_id")
            panels = [(int(i), str(n)) for i, n in db.panels["panel_name"].items()]

            conditions = next(
                (
                    f.options
                    for f in annotation_fields(db, "acquisition")
                    if f.key == "condition"
                ),
                (),
            )

            return (
                form_fields(db, "session"),
                (form_fields(db, "loop"), conditions),
                {kind: form_fields(db, kind) for kind in FOLDER_OF},
                [int(m) for m in db.mice.index],
                (panels, current_vials(vials)),
            )

        self.context.run(load, self._choices_loaded, "session choices")

    def _choices_loaded(self, loaded) -> None:
        session_fields, (loop_fields, conditions), files, mice, (panels, vials) = loaded
        self.file_fields = files
        self.condition_choices = tuple(conditions)
        self.session_fields = [
            f for f in session_fields if f.key not in ("mouse_id", "session_date")
        ]
        self.loop_fields = loop_fields
        self.mice = mice
        self.panels, self.vials = panels, vials
        self.refresh_current()

    def reload_drafts(self) -> None:
        """Read the drafts from disk; select the first unless one still is."""
        self.drafts = sorted(
            Draft.pending(self.context.drafts), key=lambda d: d.path.name
        )
        kept = next(
            (
                d
                for d in self.drafts
                if self.context.draft is not None and d.path == self.context.draft.path
            ),
            None,
        )
        self.listed.emit()
        self.context.select(kept or (self.drafts[0] if self.drafts else None))
        self.refresh_current()

    def refresh_current(self) -> None:
        """Redraw both pages from the selected draft."""
        self.session_page.show_draft(self.current)
        self.part_page.show_part(self.current, self.context.part)

    def _follow_selection(self) -> None:
        draft = self.current

        if self.session_page.draft is not draft:
            self.session_page.show_draft(draft)

        page = self.part_page

        if page.draft is not draft or page.name != self.context.part:
            page.show_part(draft, self.context.part)

    def _follow_draft(self) -> None:
        """The selected draft changed (mouse, date, parts, recordings)."""
        draft = self.current
        page = self.part_page

        if self.session_page.draft is draft:
            self.session_page._refresh_identity()

        if draft is not None and page.name != self.context.part:
            page.show_part(draft, self.context.part)
        elif draft is not None and page.shown is not None and page.name is not None:
            page.box.setTitle(page.title(draft, page.name))

    # ----------------------------------------------------------------- #
    # Actions
    # ----------------------------------------------------------------- #

    def new(self) -> Draft:
        """Start a session, listed last, and select it."""
        draft = Draft.start(self.context.drafts, user=self.context.user)
        self.drafts.append(draft)
        self.listed.emit()
        self.context.select(draft)

        return draft

    def remove(self) -> None:
        """
        Remove the selected draft (can be undone). Nothing is selected
        afterwards, so a second click cannot remove another session.
        """
        draft = self.current

        if draft is None:
            return

        draft.remove()
        self.removals.push(draft)
        self.drafts.pop(self.position(draft))
        self.listed.emit()
        self.context.select(None)
        self.refresh_current()
        self.removals_changed.emit()

    def undo(self) -> None | tuple[Draft, None | str]:
        """
        Bring back the latest removed session or part and select it;
        returns `(draft, part or None)`, None if nothing was left.
        """
        restored = self.removals.undo(self.drafts)
        self.removals_changed.emit()

        if restored is None:
            return None

        draft, part = restored

        if part is None:
            self.drafts.append(draft)
            self.drafts.sort(key=lambda d: d.path.name)
            self.listed.emit()
            self.context.select(draft)
        else:
            self.context.select(draft, part)
            self.context.announce_draft()

        self.refresh_current()

        return restored

    def next_undo(self) -> None | str:
        """What undo would bring back, as the bottom bar says it; None if nothing."""
        found = self.removals.next()

        if found is None:
            return None

        draft, part = found

        if part is None:
            return f"Undo remove (session {draft.where})"

        return f"Undo remove ({draft.describe(part)})"

    def add_loop(self) -> None:
        """
        Add the next loop (`e<n>`) to the selected draft in one click, make its
        folders on the share, and select it. The number is one past every
        loop of the draft (removed ones too) and every `eN` folder already in
        the session's folder on the share.
        """
        draft = self._identified("Loop")

        if draft is None:
            return

        self._add_part(draft, draft.next_loop(self._loops_on_share(draft)), "loop")

    def add_grabs(self) -> None:
        """
        Add the session's grabs (once; removed ones come back as they were),
        make their folders, and select them.
        """
        self._add_folder("grab")

    def add_zstacks(self) -> None:
        """
        Add the session's z-stacks (once; removed ones come back as they were),
        make their folders, and select them.
        """
        self._add_folder("zstack")

    def _add_folder(self, kind: str) -> None:
        what = "Grabs" if kind == "grab" else "Z-stacks"
        draft = self._identified(what)

        if draft is None:
            return

        name = FOLDER_OF[kind]

        if draft.part(name) is not None:
            self.context.select(draft, name)
            self.part_added.emit(name)
            return

        # Back as it was, notes included; its entry in the undo list no longer
        # applies, so Undo goes on to the one before it
        if draft.is_removed(name):
            draft.restore_part(name)
            self.context.announce_draft()
            self.context.select(draft, name)
            self.removals_changed.emit()
            self.part_added.emit(name)
            return

        self._add_part(draft, name, kind)

    def _identified(self, what: str) -> None | Draft:
        """The selected draft, once it has a mouse (its folders are named for it)."""
        draft = self.current

        if draft is None:
            self.context.report(what, "Add a session first")
            return None

        if draft.mouse_id is None:
            self.context.report(
                what,
                "Pick a mouse for the session first\n(it is needed for the folder name)",
            )
            return None

        return draft

    def _loops_on_share(self, draft: Draft) -> tuple[int, ...]:
        """The numbers of the `eN` folders already in the session's folder."""
        folder = self.context.main_folder / draft.folder

        try:
            return tuple(
                int(path.name[1:])
                for path in folder.iterdir()
                if path.is_dir() and re.fullmatch(r"e\d+", path.name)
            )

        except OSError:
            return ()

    def _add_part(self, draft: Draft, name: str, kind: str) -> None:
        draft.set_part(name)

        try:
            make_part_folders(self.context.main_folder / draft.folder / name, kind)
            draft.set_part(name, folder_made=True)

        except OSError as error:
            self.context.report(
                PartPage.TITLES[kind],
                f"Could not create {draft.folder}/{name} on the server:\n\n{error}"
                "\n\nCreate it manually before copying the TIFFs!",
            )

        self.context.announce_draft()
        self.context.select(draft, name)
        self.part_added.emit(name)

    def remove_part(self, name: None | str = None) -> None:
        """Remove part `name` (the selected one if None) from the draft."""
        draft = self.current
        name = name or self.context.part

        if draft is None or name is None or draft.part(name) is None:
            return

        draft.remove_part(name)
        self.removals.push(draft, name)
        self.context.announce_draft()
        self.removals_changed.emit()

    # ----------------------------------------------------------------- #
    # Check and submit
    # ----------------------------------------------------------------- #

    def check(self) -> None:
        draft = self.current

        if draft is None:
            return

        self.reported.emit(running(draft.label, "checking"))
        self.context.run(
            lambda db: check(db, draft, self.context.olfactometer),
            lambda problems: self._report(
                draft,
                problems,
                (
                    f"fix errors to submit {CROSS}"
                    if any(p.blocking for p in problems)
                    else f"ready to submit {CHECK}"
                ),
            ),
            "check",
        )

    def submit(self) -> None:
        draft = self.current

        if draft is None:
            return

        self.reported.emit(
            running(draft.label, "submitting, reading the recordings can take a while")
        )
        busy = {self.context.recording_now} - {None}

        self.context.run(
            lambda db: submit(
                db, draft, busy=busy, olfactometer=self.context.olfactometer
            ),
            lambda result: self._submitted(draft, result),
            "submit",
            writes=True,
        )

    def _report(self, draft: Draft, problems: list[Problem], result: str) -> None:
        parts = [str(p["name"]) for p in draft.parts]
        self.reported.emit(report(draft.label, problems, parts, result))

    def _submitted(self, draft: Draft, result) -> None:
        problems = list(result.problems)
        written = result.written

        if written is None:
            outcome = f"fix errors to submit {CROSS}"

        else:
            # The session is in the database even if a sync file did not match
            problems += [
                Problem(name, f"could not process the sync file: {message}", False)
                for name, message in written.process_failed.items()
            ]
            outcome = (
                f"submitted as session {written.session_id} "
                f"({counted(len(written.loops), 'loop')}, "
                f"{counted(len(written.grabs), 'grab')}, "
                f"{counted(len(written.zstacks), 'z-stack')}, "
                f"{counted(written.annotations, 'annotation')}, sync processed for "
                f"{', '.join(written.processed) or 'none'}) {CHECK}"
            )

        if result.error:
            outcome = f"stopped: {result.error} (submit again after fixing it) {CROSS}"

        self._report(draft, problems, outcome)

        if result.submitted:
            self.reload_drafts()
        elif self.part_page.shown is not None:
            self.part_page.shown.refresh_status()
