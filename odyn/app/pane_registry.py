"""
The registries: users, mice, odors, panels and vials.

Each is a table of what the database holds beside a form to add to it. These
write straight to the database (through the worker), since a mouse or a vial
is a fact on its own and does not wait for the end of the day.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..database import WILD_TYPE
from .context import Context
from .fields import COLUMN_FIELDS, ODOR_FIELDS
from .forms import Form, FormRows, RecipeForm, fit_row
from .style import COMBO_EXTRA_PX, FORM_PX, text_style
from .symbols import icon
from .models import FrameModel, fit_columns
from .panel import UNLIMITED_PX


def _table() -> QTableView:
    view = QTableView()
    view.setModel(FrameModel())
    view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    view.setAlternatingRowColors(True)
    fit_columns(view)

    return view


class TableAndForm(QWidget):
    """
    A registry page: its table on the left, its form on the right.

    Like the bottom panel: the arrow at the form's edge hides the form down
    to that edge (the table takes the rest) and brings it back, and dragging
    the line between table and form widens the form. Every page shares one
    form width, so switching pages does not move it, and the form opens again
    whenever its page is shown, so it is never forgotten behind the arrow.
    """

    # Shared by every page; dragging any page's form changes it for all
    form_px = FORM_PX

    def __init__(self, table: QWidget, form: QWidget, parent=None):
        super().__init__(parent)
        self.table = table

        self.toggle = QToolButton()
        self.toggle.setObjectName("panelToggle")
        self.toggle.setIconSize(QSize(20, 20))
        self.toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle.clicked.connect(lambda: self.set_open(not self.is_open()))

        strip = QVBoxLayout()
        strip.setContentsMargins(2, 0, 2, 0)
        strip.addWidget(self.toggle)
        strip.addStretch(1)

        # The form scrolls rather than squeezing when the window is short
        self.form = QScrollArea()
        self.form.setWidget(form)
        self.form.setWidgetResizable(True)
        self.form.setFrameShape(QScrollArea.Shape.NoFrame)
        self.form.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form.setMinimumWidth(FORM_PX)

        # The arrow's strip stays when the form is hidden
        self.side = QWidget()
        side_row = QHBoxLayout(self.side)
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(0)
        side_row.addLayout(strip)
        side_row.addWidget(self.form, 1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setObjectName("tableAndForm")
        self.splitter.setHandleWidth(5)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(table)
        self.splitter.addWidget(self.side)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.splitterMoved.connect(self._moved)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self.set_open(True)

    def is_open(self) -> bool:
        return not self.form.isHidden()

    def set_open(self, opened: bool) -> None:
        self.form.setHidden(not opened)
        self.toggle.setIcon(icon("form-close" if opened else "form-open"))
        tip = "Hide the form" if opened else "Show the form"
        self.toggle.setToolTip(tip)
        self.toggle.setStatusTip(tip)

        # Closed, the side is only as wide as its arrow, and cannot be dragged
        strip_px = self.side.minimumSizeHint().width()
        self.side.setMaximumWidth(UNLIMITED_PX if opened else strip_px)

        if opened:
            self._apply_width()
        elif total := sum(self.splitter.sizes()):
            self.splitter.setSizes([total - strip_px, strip_px])

    def showEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self.set_open(True)
        super().showEvent(event)

    def _apply_width(self) -> None:
        total = sum(self.splitter.sizes())

        if total:
            least = self.side.minimumSizeHint().width()
            own = max(min(TableAndForm.form_px, total - 100), least)
            self.splitter.setSizes([total - own, own])

    def _moved(self, *_) -> None:
        if self.is_open():
            TableAndForm.form_px = self.splitter.sizes()[1]


def _model(view: QTableView) -> FrameModel:
    model = view.model()
    assert isinstance(model, FrameModel)

    return model


class UsersPane(QWidget):
    """
    Who is using the app: one button per person, like a login screen.

    Users are added with `python -m odyn.admin <folder> add-user <login>
    "<Name>"`; the login is what the workstations know the person by, the
    name is what shows here.
    """

    picked = Signal(str)

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.buttons: dict[str, QPushButton] = {}

        outer = QVBoxLayout(self)
        outer.addStretch(2)

        title = QLabel("Users")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        outer.addWidget(title)

        self.hint = QLabel("")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet(text_style("computed"))
        outer.addWidget(self.hint)
        outer.addSpacing(16)

        row = QHBoxLayout()
        row.addStretch(1)
        self.names = QVBoxLayout()
        self.names.setSpacing(8)
        row.addLayout(self.names)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(3)

        context.changed.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        self.context.run(lambda db: db.users, self._show, "users")

    def _show(self, users) -> None:
        while self.names.count():
            item = self.names.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.buttons = {}
        active = users[~users["retired"].astype(bool)].sort_values("display_name")
        current = self.context.user

        for _, row in active.iterrows():
            user_name = str(row["user_name"])
            label = str(row["display_name"])
            button = QPushButton(label)
            button.setMinimumSize(320, 48)
            button.setStyleSheet("font-size: 16px;")
            button.setCheckable(True)
            button.setChecked(user_name == current)
            button.clicked.connect(lambda _=False, name=user_name: self._pick(name))
            self.buttons[user_name] = button
            self.names.addWidget(button)

        if active.empty:
            self.hint.setText(
                "No users yet: add one with  python -m odyn.admin <folder> "
                'add-user <login> "<Name>"'
            )
        elif current in self.buttons:
            self.hint.setText(f"Signed in as {self.buttons[current].text()}")
        else:
            self.hint.setText("Click on your name")

    def _pick(self, user_name: str) -> None:
        for name, button in self.buttons.items():
            button.setChecked(name == user_name)

        self.hint.setText(f"Signed in as {self.buttons[user_name].text()}")

        if user_name != self.context.user:
            self.context.set_user(user_name)

        self.picked.emit(user_name)


# Line's genotype options (as shown for the user). "na" = not genotyped.
GENOTYPE_CHOICES = ("na", "het", "hom", "wt")


class PickRow(QWidget):
    """
    One registered name picked from a dropdown (a mouse's line or virus, a
    mix's component), with a genotype dropdown for a line, and a remove button.
    """

    def __init__(self, names: list[str], genotyped: bool, parent=None):
        super().__init__(parent)
        self.name = QComboBox()
        self.name.addItems(names)

        # The open list is as wide as the longest name, whatever the box's width
        self.name.view().setMinimumWidth(self.name.view().sizeHintForColumn(0) + 24)
        self.genotype: None | QComboBox = None
        self.remove = QPushButton("remove")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.name, 1)

        if genotyped:
            self.genotype = QComboBox()
            self.genotype.addItems(GENOTYPE_CHOICES)
            self.genotype.setToolTip("na: not genotyped")

            # Fits its longest choice, and is the same on every row
            metrics = self.genotype.fontMetrics()
            widest = max(metrics.horizontalAdvance(c) for c in GENOTYPE_CHOICES)
            self.genotype.setFixedWidth(widest + COMBO_EXTRA_PX)
            self.name.currentTextChanged.connect(self._name_picked)
            row.addWidget(self.genotype)
            self._name_picked(self.name.currentText())

        row.addWidget(self.remove)
        fit_row((self.name, self.genotype, self.remove), self.name.sizeHint().height())

    def _name_picked(self, name: str) -> None:
        """Wild type has nothing to genotype: its choice is fixed to 'na'."""
        wild = name == WILD_TYPE

        if wild:
            self.genotype.setCurrentText("na")

        self.genotype.setEnabled(not wild)

    def set_value(self, name: str, genotype: str = "na") -> None:
        if self.name.findText(name) < 0:
            self.name.addItem(name)  # a name no longer listed still shows

        self.name.setCurrentText(name)

        if self.genotype is not None:
            self.genotype.setCurrentText(genotype)

    def value(self) -> str | tuple[str, str]:
        """The name, or `(name, genotype)` for a line."""
        if self.genotype is None:
            return self.name.currentText()

        return self.name.currentText(), self.genotype.currentText()


class PickList(FormRows):
    """
    The rows of one list of registered names: a mouse's lines or viruses, a
    mix's components (see `FormRows`).

    The rows speak for a mouse only once they were loaded for it or edited
    (`speaks_for`): otherwise saving leaves what is stored alone, so typing a
    number to fix a mouse's sex does not wipe its lines.
    """

    def __init__(
        self,
        context: Context,
        what: str,
        genotyped: bool,
        form: QFormLayout,
        title: str,
        none_text: str = "none",
    ):
        super().__init__(form, title, none_text)
        self.context = context
        self.what = what
        self.genotyped = genotyped
        self.names: list[str] = []
        self.loaded_for: None | int = None
        self.edited = False
        self.locked = False

    def add(self, name: None | str = None, genotype: str = "na") -> None:
        """A row: blank for a new entry, or showing `name` (and its genotype)."""
        if name is None and not self.names:
            self.context.report(
                self.what.capitalize(),
                f"No {self.what} are listed yet: they are added to the database.",
            )
            return

        row = PickRow(self.names, self.genotyped)
        row.setEnabled(not self.locked)

        if name is not None:
            row.set_value(name, genotype)

        row.remove.clicked.connect(lambda: self.remove(row, edited=True))
        self.insert(row)
        self.edited = self.edited or name is None

    def remove(self, row: PickRow, edited: bool = False) -> None:
        self.edited = self.edited or edited
        self.take(row)

    def load(self, owner: None | int, entries) -> None:
        """Show what `owner` has: names, or `(name, genotype)` pairs."""
        for row in list(self.rows):
            self.remove(row)

        for entry in entries:
            self.add(*entry) if isinstance(entry, tuple) else self.add(entry)

        self.loaded_for, self.edited = owner, False

    def speaks_for(self, owner: int) -> bool:
        return self.edited or self.loaded_for == owner

    def set_locked(self, locked: bool) -> None:
        """Shown but not editable (what a stored odor is made of)."""
        self.locked = locked

        for row in self.rows:
            row.setEnabled(not locked)

    def values(self) -> list:
        return [row.value() for row in self.rows]

    def repeated(self) -> bool:
        names = [row.name.currentText() for row in self.rows]

        return len(set(names)) < len(names)


def _exists_note(what: str) -> str:
    return f"{what} ID already exists, saving will overwrite data."


def _stored(frame, key: int) -> dict:
    """One stored row as `{column: value}`, a missing value as `None`."""
    return {
        column: None if value is None or value != value else value  # NaN
        for column, value in frame.loc[key].items()
    }

def _watch_id(form: Form, key: str, what: str, stored) -> None:
    """
    Say "already exists" under the ID while it is typed, so a stored entry
    is noticed before the field is left (which is when it loads: loading on
    every keystroke would load 4 on the way to typing 45).
    """
    row = form.rows[key]
    note = _exists_note(what)

    def typed(text: str) -> None:
        try:
            value = row.field.parse(text)
        except ValueError:
            return  # said when the field is left

        if value is not None and stored(value):
            row.show_note(note, "warning")
        elif not row.note.isHidden():
            row.note.setVisible(False)

    row.editor.textChanged.connect(typed)


def _buttons(actions) -> tuple[QHBoxLayout, list[QPushButton]]:
    """A row of buttons, `(text, action)` each, sharing its width."""
    row, buttons = QHBoxLayout(), []

    for text, action in actions:
        button = QPushButton(text)
        button.clicked.connect(action)
        row.addWidget(button)
        buttons.append(button)

    return row, buttons


class MicePane(QWidget):
    """
    The mice, and a form to register one or change what is stored about it.

    Typing the number of a stored mouse (or clicking it in the table) loads
    it into the form, with a note that saving overwrites it; saving over a
    stored mouse asks first.
    """

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.mice = None  # the table's frame, indexed by mouse_id
        self.carried: dict[int, dict[str, str]] = {}
        self.injected: dict[int, list[str]] = {}

        self.table = _table()
        holder = QWidget()
        side = QVBoxLayout(holder)
        side.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Register or update a mouse")
        box_layout = QVBoxLayout(box)
        self.form = Form(list(COLUMN_FIELDS["mouse"]))
        box_layout.addWidget(self.form)

        # The lists' rows are the form's own, below its fields
        form = self.form.layout()
        self.line_list = PickList(context, "lines", True, form, "Lines")
        self.virus_list = PickList(context, "viruses", False, form, "Viruses")

        buttons, _ = _buttons(
            (
                ("Add line", lambda: self.line_list.add()),
                ("Add virus", lambda: self.virus_list.add()),
                ("Clear", self.clear),
                ("Save mouse", self._save),
            )
        )
        box_layout.addLayout(buttons)

        side.addWidget(box)
        side.addStretch(1)
        self.split = TableAndForm(self.table, holder)
        QHBoxLayout(self).addWidget(self.split)

        self.table.clicked.connect(self._pick)
        self.form.changed.connect(self._typed)
        _watch_id(self.form, "mouse_id", "Mouse", self.stored)
        context.changed.connect(self.reload)
        self.reload()

    def clear(self) -> None:
        """Empty the form, for a new mouse."""
        self.form.clear()
        self.line_list.load(None, [])
        self.virus_list.load(None, [])
        self.table.clearSelection()

    def reload(self) -> None:
        def load(db):
            mice = db.mice.copy()
            carried = {
                int(mouse_id): dict(zip(rows["line_name"], rows["genotype"]))
                for mouse_id, rows in db.mouse_lines.groupby(level=0)
            }
            injected = {
                int(mouse_id): list(rows["virus_name"])
                for mouse_id, rows in db.mouse_viruses.groupby(level=0)
            }
            mice["lines"] = [
                ", ".join(
                    f"{line} ({genotype})"
                    for line, genotype in carried.get(int(m), {}).items()
                )
                or None
                for m in mice.index
            ]
            mice["viruses"] = [
                ", ".join(injected.get(int(m), [])) or None for m in mice.index
            ]

            return (
                mice,
                carried,
                injected,
                [str(name) for name in db.lines.index],
                [str(name) for name in db.viruses.index],
            )

        self.context.run(load, self._loaded, "mice")

    def _loaded(self, loaded) -> None:
        self.mice, self.carried, self.injected, lines, viruses = loaded
        self.line_list.names, self.virus_list.names = lines, viruses
        _model(self.table).set_frame(self.mice)

    def stored(self, mouse_id) -> bool:
        return self.mice is not None and mouse_id in self.mice.index

    def _typed(self, key: str, value) -> None:
        """A stored mouse's number was typed: show what is stored."""
        if key == "mouse_id" and self.stored(value):
            self.load(value)

    def _pick(self, index) -> None:
        row = _model(self.table).row_values(index.row())
        self.load(int(row["mouse_id"]))

    def load(self, mouse_id: int) -> None:
        """Fill the form with what is stored about `mouse_id`."""
        row = _stored(self.mice, mouse_id)
        self.form.set_values(
            {"mouse_id": mouse_id, "sex": row["mouse_sex"], "dob": row["mouse_dob"]}
        )
        self.line_list.load(mouse_id, list(self.carried.get(mouse_id, {}).items()))
        self.virus_list.load(mouse_id, self.injected.get(mouse_id, []))
        self.form.rows["mouse_id"].show_note(_exists_note("Mouse"), "warning")

    def _save(self) -> None:
        self.form.commit()

        if problems := self.form.problems():
            self.context.report("Mouse", " ".join(problems))
            return

        values = self.form.values()

        if "mouse_id" not in values:
            self.context.report("Mouse", "A mouse needs its number.")
            return

        for listed in (self.line_list, self.virus_list):
            if listed.repeated():
                self.context.report(
                    "Mouse", f"One of the {listed.what} is listed twice."
                )
                return

        mouse_id = values["mouse_id"]
        stored = self.stored(mouse_id)

        if stored and not self.context.confirm(
            "Mouse", _exists_note("Mouse"), "Overwrite"
        ):
            return

        # The rows are all the mouse's lines (none left means none) when they
        # were loaded for it, edited, or the mouse is new; else (its number
        # typed and saved at once) what is stored stays. Same for the viruses.
        def listed(pick: PickList):
            return pick.values() if pick.speaks_for(mouse_id) or not stored else None

        lines, viruses = listed(self.line_list), listed(self.virus_list)

        self.context.run(
            lambda db: db.set_mouse(
                mouse_id=mouse_id,
                sex=values.get("sex"),
                dob=values.get("dob"),
                lines=None if lines is None else dict(lines),
                viruses=viruses,
            ),
            None,
            "set_mouse",
            writes=True,
        )


class OdorsPane(QWidget):
    """
    The odor registry, and a form to add an odor or fill in its facts.

    Like the Mice page: a stored odor's number (or a click on it) loads it,
    and saving over it asks first. Its name and what it is made of never
    change (the trials and vials refer to them), so they are shown locked;
    only the vapor pressure and its source can be overwritten.
    """

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.odors = None  # the table's frame, indexed by odor_id
        self.added: None | int = None  # loaded once the table has it
        self.made_of: dict[int, list[int]] = {}
        self.chemicals: dict[str, int] = {}  # "4 - ethyl butyrate": 4

        self.table = _table()
        holder = QWidget()
        side = QVBoxLayout(holder)
        side.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Register or update an odor")
        box_layout = QVBoxLayout(box)
        self.form = Form(list(ODOR_FIELDS))
        box_layout.addWidget(self.form)

        # A mix is what it is made of: two or more chemicals, fixed once added
        self.composition = PickList(
            context,
            "chemicals",
            False,
            self.form.layout(),
            "Composition",
            none_text="Not a mix of odors",
        )
        buttons, (self.add_component, *_) = _buttons(
            (
                ("Add component", lambda: self.composition.add()),
                ("Clear", self.clear),
                ("Save odor", self._save),
            )
        )
        box_layout.addLayout(buttons)

        side.addWidget(box)
        side.addStretch(1)
        self.split = TableAndForm(self.table, holder)
        QHBoxLayout(self).addWidget(self.split)

        self.table.clicked.connect(self._pick)
        self.form.changed.connect(self._typed)
        _watch_id(self.form, "odor_id", "Odor", self.stored)
        context.changed.connect(self.reload)
        self.reload()

    def clear(self) -> None:
        """Empty the form and unlock it, for a new odor."""
        self.form.clear()
        self.composition.load(None, [])
        self._lock(False)
        self.table.clearSelection()

    def reload(self) -> None:
        def load(db):
            odors = db.odors.copy()
            names = odors["odor_name"]
            components = db.odor_components["component_id"]
            parts = components.map(lambda i: f"{i} {names.get(i, '?')}")
            odors["components"] = (
                parts.groupby(level=0)
                .agg(" + ".join)
                .reindex(odors.index)
                .where(odors["mix"].astype(bool), "")
            )
            made_of = {
                int(odor_id): [int(c) for c in rows]
                for odor_id, rows in components.groupby(level=0)
            }
            chemicals = {
                f"{i} - {n}": int(i)
                for i, n in names[~odors["mix"].astype(bool)].items()
                if i != 0  # the solvent is not mixed in
            }

            return odors, made_of, chemicals

        self.context.run(load, self._loaded, "odors")

    def _loaded(self, loaded) -> None:
        self.odors, self.made_of, self.chemicals = loaded
        self.composition.names = list(self.chemicals)
        _model(self.table).set_frame(self.odors)

        if self.added is not None and self.stored(self.added):
            self.load(self.added)
            self.added = None

    def stored(self, odor_id) -> bool:
        return self.odors is not None and odor_id in self.odors.index

    def _typed(self, key: str, value) -> None:
        """A stored odor's number loads it; a new one unlocks the form."""
        if key != "odor_id":
            return

        if self.stored(value):
            self.load(value)

        elif self.composition.locked:
            # What was shown belongs to another odor, whose name is taken
            self.form.rows["odor_name"].set_value(None)
            self.composition.load(None, [])
            self._lock(False)

    def _pick(self, index) -> None:
        row = _model(self.table).row_values(index.row())
        self.load(int(row["odor_id"]))

    def load(self, odor_id: int) -> None:
        """Fill the form with what is stored about `odor_id`, name locked."""
        row = _stored(self.odors, odor_id)
        self.form.set_values({"odor_id": odor_id, **row})
        self.composition.load(odor_id, [self._label(i) for i in self._parts(odor_id)])
        self._lock(True)
        self.form.rows["odor_id"].show_note(_exists_note("Odor"), "warning")

    def _lock(self, locked: bool) -> None:
        # Shown like a value read from elsewhere, not like a placeholder
        name = self.form.rows["odor_name"].editor
        name.setEnabled(not locked)
        name.setStyleSheet(text_style("computed") if locked else "")
        name.setToolTip(
            "A stored odor's name never changes"
            if locked
            else self.form.rows["odor_name"].field.description
        )
        self.composition.set_locked(locked)
        self.add_component.setEnabled(not locked)

    def _parts(self, odor_id: int) -> list[int]:
        return self.made_of.get(odor_id, [])

    def _label(self, odor_id: int) -> str:
        """A chemical as its dropdown shows it."""
        for label, number in self.chemicals.items():
            if number == odor_id:
                return label

        return f"{odor_id} - ?"

    def _save(self) -> None:
        self.form.commit()

        if problems := self.form.problems():
            self.context.report("Odor", " ".join(problems))
            return

        values = self.form.values()

        if "odor_id" not in values or "odor_name" not in values:
            self.context.report("Odor", "An odor needs its number and a name.")
            return

        if self.composition.repeated():
            self.context.report("Odor", "A component is listed twice.")
            return

        odor_id = values["odor_id"]
        components = [self.chemicals.get(label) for label in self.composition.values()]
        facts = {
            "vapor_pressure_mmhg": values.get("vapor_pressure_mmhg"),
            "vapor_pressure_source": values.get("vapor_pressure_source"),
        }

        if not self.stored(odor_id):
            self.context.run(
                lambda db: db.add_odor(
                    odor_id=odor_id,
                    odor_name=values["odor_name"],
                    components=components or None,
                    **facts,
                ),
                lambda _: setattr(self, "added", odor_id),
                "add_odor",
                writes=True,
            )
            return

        stored_name = str(self.odors.loc[odor_id, "odor_name"])

        if values["odor_name"] != stored_name or sorted(components) != sorted(
            self._parts(odor_id)
        ):
            self.context.report(
                "Odor", "Cannot change the name and composition of an odor."
            )
            return

        if not self.context.confirm("Odor", _exists_note("Odor"), "Overwrite"):
            return

        self.context.run(
            lambda db: db.update_odor(odor_id=odor_id, **facts),
            None,
            "update_odor",
            writes=True,
        )


def current_vials(vials):
    """
    The vial that is in each position of each panel now: the one made last
    (the highest `vial_id` among those made the same day), ordered by panel
    and position, so remade racks do not pile up in the table.
    """
    if vials.empty:
        return vials

    ordered = vials.sort_values(
        ["panel_name", "vial_position", "made_on"], na_position="first"
    ).reset_index()
    latest = ordered.groupby(["panel_id", "vial_position"], sort=False).tail(1)
    latest = latest.sort_values(["panel_name", "vial_position"])

    return latest.set_index("vial_id")


class PanelsPane(QWidget):
    """Panels and their vials, with the recipe calculator for new ones."""

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.odors: list[tuple[int, str]] = []
        self.chemicals: dict[int, tuple[str, None | float]] = {}
        self.made_of: dict[int, list[int]] = {}
        self._panels = None
        self._vial_odors = None
        self._all_vials = None

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(
            QLabel(
                "Vials in the olfactometer: the latest made for each panel and position"
            )
        )
        self.table = _table()
        left_layout.addWidget(self.table, 1)

        new_panel = QHBoxLayout()
        self.panel_name = QLineEdit()
        self.panel_name.setPlaceholderText("new panel name, e.g. print_v3")
        add_panel = QPushButton("Add panel")
        add_panel.clicked.connect(self._add_panel)
        new_panel.addWidget(self.panel_name, 1)
        new_panel.addWidget(add_panel)
        left_layout.addLayout(new_panel)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        head = QFormLayout()
        self.panel = QComboBox()
        self.made_on = QLineEdit()
        self.made_on.setPlaceholderText("YYYY-MM-DD, blank for today")
        head.addRow("Panel", self.panel)
        head.addRow("Made on", self.made_on)
        right_layout.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.recipes_widget = QWidget()
        self.recipes = QVBoxLayout(self.recipes_widget)
        self.recipes.addStretch(1)
        scroll.setWidget(self.recipes_widget)
        right_layout.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        add_vial = QPushButton("Add a vial to the list")
        add_vial.clicked.connect(self._add_recipe)
        copy_last = QPushButton("Copy the panel's last rack")
        copy_last.clicked.connect(self._copy_last)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        save = QPushButton("Save these vials")
        save.clicked.connect(self._save)
        buttons.addWidget(add_vial)
        buttons.addWidget(copy_last)
        buttons.addStretch(1)
        buttons.addWidget(clear)
        buttons.addWidget(save)
        right_layout.addLayout(buttons)

        self.split = TableAndForm(left, right)
        QHBoxLayout(self).addWidget(self.split)

        self.forms: list[RecipeForm] = []

        context.changed.connect(self.reload)
        self.reload()

    def reload(self) -> None:
        def load(db):
            vials = db.vials.join(db.panels["panel_name"], on="panel_id")
            vials = vials.join(db.odors["odor_name"], on="odor_id")
            columns = [
                "panel_name",
                "vial_position",
                "odor_name",
                "made_on",
                "vial_sccm",
                "total_sccm",
                "total_volume_ml",
                "solvent_volume_ml",
            ]
            active = db.odors[~db.odors["retired"].astype(bool)]
            odors = [(int(i), str(n)) for i, n in active["odor_name"].items()]
            chemicals = {
                int(i): (
                    str(row["odor_name"]),
                    (
                        None
                        if row["vapor_pressure_mmhg"] != row["vapor_pressure_mmhg"]
                        else float(row["vapor_pressure_mmhg"])
                    ),
                )
                for i, row in db.odors[~db.odors["mix"].astype(bool)].iterrows()
            }
            made_of: dict[int, list[int]] = {}

            for odor_id, component_id in db.odor_components["component_id"].items():
                made_of.setdefault(int(odor_id), []).append(int(component_id))

            return (
                current_vials(vials)[columns],
                db.panels,
                odors,
                (chemicals, made_of),
                db.vial_odors,
                db.vials,
            )

        self.context.run(load, self._show, "vials")

    def _show(self, loaded) -> None:
        vials, panels, odors, (chemicals, made_of), vial_odors, all_vials = loaded
        _model(self.table).set_frame(vials)
        self.odors, self.chemicals, self.made_of = odors, chemicals, made_of
        self._panels, self._vial_odors, self._all_vials = panels, vial_odors, all_vials

        current = self.panel.currentText()
        self.panel.blockSignals(True)
        self.panel.clear()
        self.panel.addItems([str(name) for name in panels["panel_name"]])
        if current:
            self.panel.setCurrentText(current)
        self.panel.blockSignals(False)

    def _add_panel(self) -> None:
        name = self.panel_name.text().strip()

        if not name:
            return

        self.context.run(
            lambda db: db.add_panel(panel_name=name),
            lambda _: self.panel_name.clear(),
            "add_panel",
            writes=True,
        )

    def _add_recipe(self) -> RecipeForm:
        form = RecipeForm(self.odors, self.chemicals, self.made_of)
        form.position.setValue(len(self.forms) + 1)
        self.forms.append(form)
        self.recipes.insertWidget(self.recipes.count() - 1, form)

        return form

    def _copy_last(self) -> None:
        """Fill the list with the panel's most recent rack, ready to re-date."""
        panel = self.panel.currentText()
        panel_ids = self._panels.index[self._panels["panel_name"] == panel]

        if not len(panel_ids) or not len(self._all_vials):
            return

        mine = self._all_vials[self._all_vials["panel_id"] == panel_ids[0]]

        # The newest vial per position: rows are inserted in order, so the
        # highest id in a position is the last one made
        latest = (
            mine.reset_index()
            .sort_values("vial_id")
            .drop_duplicates("vial_position", keep="last")
        )

        self._clear_forms()

        for _, row in latest.iterrows():
            form = self._add_recipe()
            form.position.setValue(int(row["vial_position"]))
            form.delivered.setCurrentIndex(
                max(form.delivered.findData(int(row["odor_id"])), 0)
            )
            form.vial_sccm.setValue(float(row["vial_sccm"]))
            form.total_sccm.setValue(float(row["total_sccm"]))
            form.total_volume.setValue(float(row["total_volume_ml"]))

            if row["vial_id"] not in self._vial_odors.index:
                continue

            pipetted = self._vial_odors.loc[[row["vial_id"]]]
            form.set_targets(
                {
                    int(c["odor_id"]): float(c["target_ppm"])
                    for _, c in pipetted.iterrows()
                }
            )

    def clear(self) -> None:
        """Empty the list of vials and the date, for another batch."""
        self._clear_forms()
        self.made_on.clear()

    def _clear_forms(self) -> None:
        for form in self.forms:
            form.setParent(None)
            form.deleteLater()

        self.forms = []

    def _save(self) -> None:
        panel = self.panel.currentText()

        if not panel:
            self.context.report("Vials", "Pick a panel, or add one first.")
            return

        if not self.forms:
            self.context.report("Vials", "Nothing to save: add a vial to the list.")
            return

        vials = [form.vial() for form in self.forms]
        made_on = self.made_on.text().strip() or None

        self.context.run(
            lambda db: db.add_vials(panel_name=panel, vials=vials, made_on=made_on),
            lambda _: self._clear_forms(),
            "add_vials",
            writes=True,
        )
