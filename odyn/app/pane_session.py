"""
The day's work: one draft per session, filled in as the part runs and
submitted at the end.

Two pages show the selected draft (`Context.draft`): the **session page**
(mouse, date, folder, the session's annotations, the odor panel) and the
**part page**, one part of the session: an experiment (`e1`, a loop or a grab:
its settings, annotations and conditions, and the calculator for its type), or
its z-stacks (the objective, and each TIFF found in the folder with its header
values and notes). `Sessions` owns the drafts and the actions that both the
pages and the window's right bar call; what Check and Submit find goes out
through `Sessions.reported`.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..utils import CHECK, CROSS
from .context import Context
from .draft import (
    EXP_TYPES,
    FOLDER_OF,
    PART_SETTINGS,
    STACK_SETTINGS,
    Draft,
    Removals,
)
from .fields import Field, annotation_fields, form_fields
from .forms import Form, FormRows, GrabTimingForm, PanelPicker, TimingForm, fit_row
from .pane_registry import current_vials
from .recordings import make_part_folders
from .style import COMBO_EXTRA_PX, PAGE_MAX_PX, required_label, text_style
from .submit import Problem, check, report, running, submit

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


class ExperimentForm(QWidget):
    """
    One experiment of a session: its type (a loop or a grab), settings,
    annotations and conditions, beside the calculator for its type (a loop's
    trial block, or how long a grab runs). `retyped` fires when the type
    changes, since the page then shows the other set.
    """

    retyped = Signal()

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
        self.condition: None | QComboBox = None

        entry = draft.part(name) or {}
        loop = entry.get("exp_type", "loop") == "loop"

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

        # The type goes first, so its label lines up with the others
        self.type = QComboBox()
        self.type.addItems(list(EXP_TYPES))
        self.type.setCurrentText(entry.get("exp_type", "loop"))
        self.type.currentTextChanged.connect(self._retype)
        rows = self.form.layout()
        assert isinstance(rows, QFormLayout)
        rows.insertRow(0, required_label("Type"), self.type)

        left = QVBoxLayout()
        left.addWidget(self.form)

        # Conditions are per acquisition: a loop's by ranges of their numbers,
        # a grab's one acquisition by a single choice
        saved = entry.get("conditions", [])

        if conditions and loop:
            self.conditions = ConditionList(
                rows,
                list(conditions),
                saved,
                lambda values: self._set(conditions=values or None),
            )
            add = QPushButton("Add condition")
            add.clicked.connect(lambda: self.conditions.add())
            left.addWidget(add)

        elif conditions:
            self.condition = QComboBox()
            self.condition.addItems(["", *conditions])
            self.condition.setCurrentText(str(saved[0]["condition"]) if saved else "")
            self.condition.currentTextChanged.connect(self._set_condition)
            rows.addRow("Condition", self.condition)

        if loop:
            self.timing: TimingForm | GrabTimingForm = TimingForm()
            self.timing.set_values(entry)
            self.timing.changed.connect(lambda: self._set(**self.timing.values()))
        else:
            self.timing = GrabTimingForm()

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
        exp_id = entry.get("submitted_id")
        self.submitted.setText(
            "" if exp_id is None else f"Already in the database as exp_id {exp_id}"
        )

    def _set(self, **fields) -> None:
        self.draft.set_part(self.name, **fields)

    def _retype(self, exp_type: str) -> None:
        self._set(exp_type=exp_type)
        self.retyped.emit()

    def _set_condition(self, condition: str) -> None:
        """A grab's condition, as a range over its one acquisition."""
        value = [{"start": 1, "end": 1, "condition": condition}] if condition else None
        self._set(conditions=value)


class ZStacksForm(QWidget):
    """
    A session's z-stacks: the objective they were recorded with, then one box
    per z-stack taken, with what is noted about it, the experiment it was
    taken for among it. Nothing here waits for the TIFFs: notes can be written
    while a z-stack runs, and the cluster gives the first z-stack the TIFF
    numbered lowest in `zstacks/raw`, and so on (`submit.match_stacks`).
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
        self.stack_fields = [f for f in fields if f.key != "objective"]
        self.boxes: list[QGroupBox] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        objective = [f for f in fields if f.key == "objective"]
        self.form = Form(objective)
        self.form.set_values({"objective": (draft.part(name) or {}).get("objective")})
        self.form.changed.connect(
            lambda key, value: self.draft.set_part(self.name, **{key: value})
        )
        layout.addWidget(self.form)

        hint = QLabel(
            "    The entries below are matched in order with the TIFFs in "
            f"{draft.folder}/{name}/raw\n"
            "    The TIFFs are ordered according to the numbers at the end of file their names"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(text_style("computed"))
        layout.addWidget(hint)

        self.stacks_layout = QVBoxLayout()
        layout.addLayout(self.stacks_layout)

        row = QHBoxLayout()
        self.add = QPushButton("Add z-stack")
        self.add.clicked.connect(self._add)
        row.addWidget(self.add)
        row.addStretch(1)
        layout.addLayout(row)

        self._build()

    def _build(self) -> None:
        """One box per z-stack, in order."""
        for box in self.boxes:
            box.setParent(None)
            box.deleteLater()

        self.boxes = []

        for index, noted in enumerate(self.draft.stacks(self.name)):
            box = self._box(index, noted)
            self.boxes.append(box)
            self.stacks_layout.addWidget(box)

    def _box(self, index: int, noted: dict) -> QGroupBox:
        title = f"Z-stack {index + 1}"

        if noted.get("file"):
            title += f" ({noted['file']})"

        box = QGroupBox(title)
        box_layout = QVBoxLayout(box)

        form = Form(self.stack_fields)
        form.set_values({k: v for k, v in noted.items() if k not in STACK_SETTINGS})
        form.changed.connect(
            lambda key, value: self.draft.set_stack(self.name, index, **{key: value})
        )

        # The form's first row, so its label lines up with the others
        experiments = [
            str(e["name"]) for e in self.draft.parts if e.get("kind") == "experiment"
        ]
        experiment = QComboBox()
        experiment.addItems(["", *experiments])
        experiment.setCurrentText(str(noted.get("experiment", "")))
        experiment.currentTextChanged.connect(
            lambda text: self.draft.set_stack(self.name, index, experiment=text or None)
        )
        form_layout = form.layout()
        assert isinstance(form_layout, QFormLayout)
        form_layout.insertRow(0, "Taken for the experiment", experiment)
        box_layout.addWidget(form)

        # One in the database stays (its notes went with it)
        if noted.get("submitted_id") is None:
            row = QHBoxLayout()
            row.addStretch(1)
            remove = QPushButton("Remove")
            remove.setStatusTip(f"Take Z-stack {index + 1} and its notes out")
            remove.clicked.connect(lambda: self._remove(index))
            row.addWidget(remove)
            box_layout.addLayout(row)

        return box

    def _add(self) -> None:
        self.draft.add_stack(self.name)
        self._build()

    def _remove(self, index: int) -> None:
        if not self.context.confirm(
            "Remove z-stack",
            f"Remove Z-stack {index + 1} and its notes? The ones after it move up",
            "Remove",
        ):
            return

        self.draft.remove_stack(self.name, index)
        self._build()

    def refresh_status(self) -> None:
        """After a submit that kept the draft: the z-stacks now in the database."""
        self._build()


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
    """One part of the selected draft: an experiment, or its z-stacks."""

    TITLES = {"loop": "Loop", "grab": "Grab", "zstack": "Z-stacks"}

    def __init__(self, sessions: Sessions, parent=None):
        super().__init__(parent)
        self.sessions = sessions
        self.draft: None | Draft = None
        self.name: None | str = None
        self.shown: None | ExperimentForm | ZStacksForm = None

        inner = QWidget()
        layout = QVBoxLayout(inner)
        self.box = QGroupBox("Experiment")
        self.box_layout = QVBoxLayout(self.box)
        layout.addWidget(self.box)

        layout.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(centered(inner))

    def title(self, draft: Draft, name: str) -> str:
        """`Loop 20260929/m442/e1`, `Z-stacks 20260929/m442/zstacks`..."""
        entry = draft.part(name) or {}
        kind = entry.get("kind", "experiment")
        word = entry.get("exp_type", "loop") if kind == "experiment" else kind

        return f"{self.TITLES.get(word, 'Experiment')} {draft.where}/{name}"

    def show_part(self, draft: None | Draft, name: None | str) -> None:
        """Fill the page from part `name` of `draft`."""
        if self.shown is not None:
            self.shown.setParent(None)
            self.shown.deleteLater()
            self.shown = None

        self.draft, self.name = draft, name

        if draft is None or name is None or draft.part(name) is None:
            self.box.setTitle("Experiment")
            return

        self.box.setTitle(self.title(draft, name))
        kind = (draft.part(name) or {}).get("kind", "experiment")

        if kind == "experiment":
            self.shown = ExperimentForm(
                draft,
                name,
                self.sessions.exp_fields,
                self.sessions.condition_choices,
            )
            self.shown.retyped.connect(lambda: self.show_part(draft, name))
        else:
            self.shown = ZStacksForm(
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
    window's right bar call `new`, `remove`, `add_experiment`, `add_zstacks`,
    `remove_part`, `undo`, `check` and `submit`; each acts on
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
        self.exp_fields: list[Field] = []
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
                (form_fields(db, "experiment"), conditions),
                {kind: form_fields(db, kind) for kind in FOLDER_OF},
                [int(m) for m in db.mice.index],
                (panels, current_vials(vials)),
            )

        self.context.run(load, self._choices_loaded, "session choices", one_lock=True)

    def _choices_loaded(self, loaded) -> None:
        session_fields, (exp_fields, conditions), files, mice, (panels, vials) = loaded
        self.file_fields = files
        self.condition_choices = tuple(conditions)
        self.session_fields = [
            f for f in session_fields if f.key not in ("mouse_id", "session_date")
        ]
        self.exp_fields = exp_fields
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

    def add_experiment(self) -> None:
        """
        Add the next experiment (`e<n>`, a loop until its type is changed) to
        the selected draft in one click, make its folders on the share, and
        select it. The number is one past every experiment of the draft
        (removed ones too) and every `eN` folder already in the session's
        folder on the share.
        """
        draft = self._identified("Experiment")

        if draft is None:
            return

        name = draft.next_experiment(self._experiments_on_share(draft))
        self._add_part(draft, name, "experiment")

    def add_zstacks(self) -> None:
        """
        Add the session's z-stacks (once; removed ones come back as they were),
        make their folders, and select them.
        """
        self._add_folder("zstack")

    def _add_folder(self, kind: str) -> None:
        draft = self._identified(PartPage.TITLES[kind])

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

    def _experiments_on_share(self, draft: Draft) -> tuple[int, ...]:
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

        # The z-stacks start with the first one listed (Add z-stack for more)
        if kind == "zstack" and not draft.stacks(name):
            draft.add_stack(name)

        try:
            make_part_folders(self.context.main_folder / draft.folder / name, kind)
            draft.set_part(name, folder_made=True)

        except OSError as error:
            self.context.report(
                "Experiment" if kind == "experiment" else PartPage.TITLES[kind],
                f"Could not create {draft.folder}/{name} on the server:\n\n{error}"
                "\n\nCreate it manually before copying the TIFFs",
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

        self.reported.emit(running(draft.label, "submitting"))
        busy = {self.context.recording_now} - {None}

        self.context.run(
            lambda db: submit(
                db, draft, busy=busy, olfactometer=self.context.olfactometer
            ),
            lambda result: self._submitted(draft, result),
            "submit",
            writes=True,
        )

    def _report(
        self,
        draft: Draft,
        problems: list[Problem],
        result: str,
        to_do: list[tuple[str, str]] = (),
    ) -> None:
        parts = [str(p["name"]) for p in draft.parts]
        self.reported.emit(report(draft.label, problems, parts, result, to_do))

    def _submitted(self, draft: Draft, result) -> None:
        if result.error:
            outcome = f"stopped: {result.error} (submit again after fixing it) {CROSS}"

        elif result.queued is None:
            outcome = f"fix errors to submit {CROSS}"

        else:
            outcome = f"sent to the cluster (the QC page shows its progress) {CHECK}"

        # The cluster adds the session once every part's TIFFs are on the
        # server, so say where each goes
        to_do = (
            [
                (name, f"copy TIFF files to {draft.folder}/{name}/raw")
                for name in (str(p["name"]) for p in draft.parts)
            ]
            if result.submitted
            else []
        )
        self._report(draft, list(result.problems), outcome, to_do)

        if result.submitted:
            self.reload_drafts()
        elif self.part_page.shown is not None:
            self.part_page.shown.refresh_status()
