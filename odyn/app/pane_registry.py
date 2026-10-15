"""
The registries: users, mice, odors, panels and vials.

Each is a table of what the database holds beside a form to add to it. These
write straight to the database (through the worker), since a mouse or a vial
is a fact on its own and does not wait for the end of the day.
"""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLayout,
    QLabel,
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
from .fields import COLUMN_FIELDS, ODOR_FIELDS, PANEL_FIELDS
from .forms import Form, FormRows, RecipeForm, fit_row
from .style import BUTTON_SPACING_PX, COMBO_EXTRA_PX, FORM_PX, text_style
from .symbols import icon
from .models import FrameModel, fit_columns, sortable
from .panel import UNLIMITED_PX


def _table() -> QTableView:
    view = QTableView()
    view.setObjectName("registryTable")  # no frame: it meets the bars' lines
    view.setModel(FrameModel())
    view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    view.setAlternatingRowColors(True)
    fit_columns(view)
    sortable(view)

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

    def __init__(
        self,
        table: QWidget,
        form: QWidget,
        buttons: None | QLayout = None,
        parent=None,
    ):
        """`buttons` stay at the foot of the form, outside its scrolling."""
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
        self.form.setObjectName("registryForm")
        self.form.setWidget(form)
        self.form.setWidgetResizable(True)
        self.form.setFrameShape(QScrollArea.Shape.NoFrame)
        self.form.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form.setMinimumWidth(FORM_PX)

        # What hides with the form: it, and its buttons under it
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.form, 1)

        if buttons is not None:
            buttons.setContentsMargins(*FORM_MARGINS_PX)
            body.addLayout(buttons)

        # The arrow's strip stays when the form is hidden
        self.side = QWidget()
        side_row = QHBoxLayout(self.side)
        side_row.setContentsMargins(0, 0, 0, 0)
        side_row.setSpacing(0)
        side_row.addLayout(strip)
        side_row.addWidget(self.body, 1)

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
        return not self.body.isHidden()

    def set_open(self, opened: bool) -> None:
        self.body.setHidden(not opened)
        self.toggle.setIcon(icon("form-close" if opened else "form-open"))
        tip = "Hide form" if opened else "Show form"
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


# Room between a registry form (and its buttons) and the edges around it
FORM_MARGINS_PX = (4, 8, 12, 8)


def _titled(title: str) -> tuple[QWidget, QVBoxLayout]:
    """A registry form's body: its title in bold over what goes in it."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(*FORM_MARGINS_PX)
    heading = QLabel(title)
    heading.setStyleSheet("font-weight: bold;")
    layout.addWidget(heading)

    return box, layout


def _fill(page: QWidget, split: TableAndForm) -> None:
    """The page is its table and form, edge to edge."""
    layout = QHBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(split)


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

        context.follow_changes(self, self.reload)
        self.reload()

    def reload(self) -> None:
        self.context.run(lambda db: db.users, self._show, "users", one_lock=True)

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
            self.hint.setText("No users yet (add one via python -m odyn.admin)")
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
                f"No {self.what} registered in the database",
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
    return f"{what} ID already exists (saving will overwrite data)"


def _stored(frame, key: int) -> dict:
    """One stored row as `{column: value}`, a missing value as `None`."""
    return {
        column: None if value is None or value != value else value  # NaN
        for column, value in frame.loc[key].items()
    }


def _watch_id(form: Form, key: str, note: str, stored) -> None:
    """
    Show `note` ("already exists") under the ID while it is typed, so a
    stored entry is noticed before the field is left (which is when it loads:
    loading on every keystroke would load 4 on the way to typing 45).
    """
    row = form.rows[key]

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
    row.setSpacing(BUTTON_SPACING_PX)

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
        box, box_layout = _titled("Add or update a mouse")
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
        side.addWidget(box)
        side.addStretch(1)
        self.split = TableAndForm(self.table, holder, buttons)
        _fill(self, self.split)

        self.table.clicked.connect(self._pick)
        self.form.changed.connect(self._typed)
        _watch_id(self.form, "mouse_id", _exists_note("Mouse"), self.stored)
        context.follow_changes(self, self.reload)
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

        self.context.run(load, self._loaded, "mice", one_lock=True)

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
            self.context.report("Mouse", "\n".join(problems))
            return

        values = self.form.values()

        if "mouse_id" not in values:
            self.context.report("Mouse", "Mouse ID is required")
            return

        for listed in (self.line_list, self.virus_list):
            if listed.repeated():
                self.context.report("Mouse", f"Some {listed.what} are listed twice")
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
        self.context.status.emit(f"Saving m{mouse_id}")

        self.context.run(
            lambda db: db.set_mouse(
                mouse_id=mouse_id,
                sex=values.get("sex"),
                dob=values.get("dob"),
                lines=None if lines is None else dict(lines),
                viruses=viruses,
            ),
            lambda _: self.context.status.emit(f"Saved m{mouse_id}"),
            "set_mouse",
            writes=True,
            one_lock=True,
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
        box, box_layout = _titled("Add or update an odor")
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
        side.addWidget(box)
        side.addStretch(1)
        self.split = TableAndForm(self.table, holder, buttons)
        _fill(self, self.split)

        self.table.clicked.connect(self._pick)
        self.form.changed.connect(self._typed)
        _watch_id(self.form, "odor_id", _exists_note("Odor"), self.stored)
        context.follow_changes(self, self.reload)
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

        self.context.run(load, self._loaded, "odors", one_lock=True)

    def _saved(self, odor_id: int) -> None:
        self.added = odor_id  # loaded once the table has it
        self.context.status.emit(f"Saved odor {odor_id}")

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
        name = self.form.rows["odor_name"].editor
        name.setEnabled(not locked)
        name.setToolTip(
            "Odor names cannot be changed"
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
            self.context.report("Odor", "\n".join(problems))
            return

        values = self.form.values()

        if "odor_id" not in values or "odor_name" not in values:
            self.context.report("Odor", "Odor ID and name are required")
            return

        if self.composition.repeated():
            self.context.report("Odor", "Some components are listed twice")
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
                lambda _: self._saved(odor_id),
                "add_odor",
                writes=True,
                one_lock=True,
            )
            return

        stored_name = str(self.odors.loc[odor_id, "odor_name"])

        if values["odor_name"] != stored_name or sorted(components) != sorted(
            self._parts(odor_id)
        ):
            self.context.report(
                "Odor", "Cannot change the name and composition of an odor"
            )
            return

        if not self.context.confirm("Odor", _exists_note("Odor"), "Overwrite"):
            return

        self.context.run(
            lambda db: db.update_odor(odor_id=odor_id, **facts),
            lambda _: self.context.status.emit(f"Saved odor {odor_id}"),
            "update_odor",
            writes=True,
            one_lock=True,
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


PANEL_NOTE = "Panel already exists (saving adds a new batch of vials)"


class PanelsPane(QWidget):
    """
    The odor panels: the vials in the olfactometer now, and a form to add a
    panel or record a new batch of its vials.

    Like the Mice and Odors pages: a stored panel's name (or a click on one
    of its vials) loads it. Its positions and odors never change, so they are
    shown locked, one box per vial, prefilled with the last recipe and
    checked; unchecking a box leaves that vial out of the batch (it was not
    remade). A new name sets up a new panel, whose vials are added freely.
    """

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.odors: list[tuple[int, str]] = []
        self.chemicals: dict[int, tuple[str, None | float]] = {}
        self.made_of: dict[int, list[int]] = {}
        self._panels = None
        self._positions = None
        self._vials = None
        self._vial_odors = None
        self.added: None | str = None  # loaded once the table has it
        self.cards: list[RecipeForm] = []

        self.table = _table()
        holder = QWidget()
        side = QVBoxLayout(holder)
        side.setContentsMargins(0, 0, 0, 0)
        box, box_layout = _titled("Add a panel or a batch of vials")
        self.form = Form(list(PANEL_FIELDS))
        box_layout.addWidget(self.form)

        self.vials_title = QLabel("Vials")
        self.vials_title.setStyleSheet("font-weight: bold;")
        box_layout.addWidget(self.vials_title)
        self.cards_box = QVBoxLayout()
        box_layout.addLayout(self.cards_box)

        buttons, (self.add_vial, *_) = _buttons(
            (
                ("Add vial", lambda: self.add_card()),
                ("Clear", self.clear),
                ("Save panel", self._save),
            )
        )
        side.addWidget(box)
        side.addStretch(1)
        self.split = TableAndForm(self.table, holder, buttons)
        _fill(self, self.split)

        self.table.clicked.connect(self._pick)
        self.form.changed.connect(self._typed)
        _watch_id(self.form, "panel_name", PANEL_NOTE, self.stored)
        context.follow_changes(self, self.reload)
        self.reload()

    def reload(self) -> None:
        def load(db):
            vials = db.vials.join(db.panels["panel_name"], on="panel_id")
            vials = vials.join(db.odors["odor_name"], on="odor_id")
            # What is typed into the olfactometer first; the ids that only
            # the database uses (vial_id, panel_id) are left out
            columns = [
                "vial_position",
                "odor_id",
                "odor_name",
                "panel_name",
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
                (
                    current_vials(vials)[columns].reset_index(drop=True)
                    if not vials.empty
                    else vials
                ),
                (db.panels, db.panel_positions, db.vials, db.vial_odors),
                odors,
                (chemicals, made_of),
            )

        self.context.run(load, self._loaded, "vials", one_lock=True)

    def _saved(self, name: str) -> None:
        self.added = name  # loaded once the table has it
        self.context.status.emit(f"Saved {name}")

    def _loaded(self, loaded) -> None:
        shown, stored, odors, (chemicals, made_of) = loaded
        self._panels, self._positions, self._vials, self._vial_odors = stored
        self.odors, self.chemicals, self.made_of = odors, chemicals, made_of
        _model(self.table).set_frame(shown)

        if self.added is not None and self.stored(self.added):
            self.load(self.added)
            self.added = None

    # ----------------------------------------------------------------- #
    # Loading a panel
    # ----------------------------------------------------------------- #

    def stored(self, name) -> bool:
        return self._panel_id(name) is not None

    def _panel_id(self, name) -> None | int:
        if self._panels is None or name is None:
            return None

        # Whatever the case it is typed in: "16 odors" is "16 Odors"
        names = self._panels["panel_name"].str.casefold()
        found = self._panels.index[names == str(name).strip().casefold()]

        return int(found[0]) if len(found) else None

    @property
    def locked(self) -> bool:
        return not self.add_vial.isEnabled()

    def _typed(self, key: str, value) -> None:
        """A stored panel's name loads it; a new one unlocks the form."""
        if key != "panel_name":
            return

        if self.stored(value):
            self.load(value)

        elif self.locked:
            # The vials shown belong to another panel
            self._clear_cards()
            self._lock(False)

    def _pick(self, index) -> None:
        self.load(str(_model(self.table).row_values(index.row())["panel_name"]))

    def load(self, name: str) -> None:
        """
        Fill the form with a stored panel: one locked, checked box per
        position, holding the last recipe made for it.
        """
        panel_id = self._panel_id(name)
        description = self._panels.loc[panel_id, "description"]
        self.form.set_values(
            {
                "panel_name": self._panels.loc[panel_id, "panel_name"],
                "description": None if description != description else description,
            }
        )
        self._clear_cards()
        positions = self._positions[self._positions.index == panel_id]
        mine = self._vials[self._vials["panel_id"] == panel_id]

        for _, row in positions.sort_values("vial_position").iterrows():
            position = int(row["vial_position"])
            card = self.add_card()
            card.fix(position, int(row["odor_id"]))
            made = mine[mine["vial_position"] == position]

            if made.empty:
                continue  # nothing made for it yet: the defaults stand

            # The newest vial in the position is the one made last
            vial_id = int(made.index.max())
            targets = {}

            if vial_id in self._vial_odors.index:
                pipetted = self._vial_odors.loc[[vial_id]]
                targets = {
                    int(c["odor_id"]): float(c["target_ppm"])
                    for _, c in pipetted.iterrows()
                }

            card.set_recipe(made.loc[vial_id], targets)

        self._lock(True)
        self.form.rows["panel_name"].show_note(PANEL_NOTE, "warning")

    def _lock(self, locked: bool) -> None:
        """A stored panel's description and set of vials are its own."""
        description = self.form.rows["description"].editor
        description.setEnabled(not locked)
        self.add_vial.setEnabled(not locked)

    # ----------------------------------------------------------------- #
    # The vials
    # ----------------------------------------------------------------- #

    def add_card(self) -> RecipeForm:
        """A vial box, in the next free position."""
        card = RecipeForm(self.odors, self.chemicals, self.made_of)
        taken = [c.position.value() for c in self.cards]
        card.position.setValue(max(taken, default=0) + 1)
        card.remove.clicked.connect(lambda: self._remove_card(card))
        self.cards.append(card)
        self.cards_box.addWidget(card)

        return card

    def _remove_card(self, card: RecipeForm) -> None:
        self.cards.remove(card)
        card.hide()
        card.deleteLater()

    def _clear_cards(self) -> None:
        for card in list(self.cards):
            self._remove_card(card)

    def clear(self) -> None:
        """Empty the form and unlock it, for a new panel."""
        self.form.clear()
        self._clear_cards()
        self._lock(False)
        self.table.clearSelection()

    # ----------------------------------------------------------------- #
    # Saving
    # ----------------------------------------------------------------- #

    def _save(self) -> None:
        self.form.commit()

        if problems := self.form.problems():
            self.context.report("Panel", "\n".join(problems))
            return

        values = self.form.values()
        name = values.get("panel_name")
        made_on = values.get("made_on")

        if not name:
            self.context.report("Panel", "Panel name is required")
            return

        if not self.stored(name):
            self._save_new(name, values.get("description"), made_on)
            return

        name = str(self._panels.loc[self._panel_id(name), "panel_name"])

        if not self.locked:
            # Typed and saved at once: the panel was never loaded
            self.context.report(
                "Panel",
                f"Panel '{name}' already exists\n(pick it from the list to add a batch)",
            )
            return

        batch = [card.vial() for card in self.cards if card.included]

        if not batch:
            self.context.report("Panel", "Check at least one vial")
            return

        made = (
            "made today"
            if made_on in (None, date.today().isoformat())
            else f"made on {made_on}"
        )

        if not self.context.confirm(
            "Panel",
            f"This adds a new batch to the {name} panel:\n{len(batch)} of its "
            f"{len(self.cards)} vials were included ({made})",
            "Save batch",
        ):
            return

        self.context.run(
            lambda db: db.add_vials(panel_name=name, vials=batch, made_on=made_on),
            lambda _: self._saved(name),
            "add_vials",
            writes=True,
            one_lock=True,
        )

    def _save_new(self, name: str, description, made_on) -> None:
        """A new panel: its positions, then its first batch of vials."""
        vials = [card.vial() for card in self.cards]
        positions = [v["vial_position"] for v in vials]

        if not vials:
            self.context.report("Panel", "Add the panel's vials first")
            return

        if len(set(positions)) < len(positions):
            self.context.report("Panel", "Two vials are in the same position")
            return

        def add(db):
            db.add_panel(
                panel_name=name,
                positions={v["vial_position"]: v["odor_id"] for v in vials},
                description=description,
            )
            db.add_vials(panel_name=name, vials=vials, made_on=made_on)

        self.context.run(
            add,
            lambda _: self._saved(name),
            "add_panel",
            writes=True,
            one_lock=True,
        )
