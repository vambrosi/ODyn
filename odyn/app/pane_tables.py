"""
The tables, as they are: pick one from the list at the top, filter it, click
a row to see it whole in the side panel (and its annotations, when it is
something that can be annotated), as the registry pages show theirs.

Nothing here writes. The one thing it offers beyond looking is annotating the
selected row, for the once-a-day and after-the-fact facts (a mouse's weight
was typed wrong, an experiment was bad) that do not belong to a draft.
"""

from __future__ import annotations

import numbers

from typing import Any

import pandas as pd

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..utils import ANNOTATION_TARGETS
from .context import Context
from .fields import Field, annotation_fields
from .forms import Form
from .models import FrameModel, fit_columns, show, sortable
from .pane_registry import FORM_MARGINS_PX, TableAndForm
from .style import ROW_SPACING_PX

# The tables offered, in the order they are listed. `annotations_for:<target>`
# entries are the pivoted views, one column per key.
TABLES: tuple[str, ...] = (
    "mice",
    "sessions",
    "experiments",
    "zstacks",
    "acquisitions",
    "streams",
    "sync_blocks",
    "programs",
    "trials",
    "events",
    "odors",
    "odor_components",
    "panels",
    "vials",
    "vial_odors",
    "experiment_vials",
    "lines",
    "mouse_lines",
    "viruses",
    "mouse_viruses",
    "users",
    "annotation_keys",
    "annotations",
    "annotations_for:mouse",
    "annotations_for:session",
    "annotations_for:experiment",
    "annotations_for:zstack",
    "annotations_for:acquisition",
    "annotations_for:program",
    "mcor_files",
    "method_calls",
    "outputs",
    "missing_annotations",
)


def call_tables(db) -> list[str]:
    """
    One `method_calls:<Class.method>` entry per method ever called, the
    JSON inputs and outputs spread into `input_*`/`output_*` columns.
    """
    calls = db.calls

    if calls.empty:
        return []

    return [f"method_calls:{name}" for name in sorted(calls["method_name"].unique())]


# Which annotation target a table's rows are, if any
TARGET_OF = {
    "mice": "mouse",
    "sessions": "session",
    "experiments": "experiment",
    "zstacks": "zstack",
    "acquisitions": "acquisition",
    "programs": "program",
}


def read_table(db, name: str) -> pd.DataFrame:
    """The DataFrame behind a `TABLES` entry, on the database thread."""
    if name.startswith("annotations_for:"):
        return db.annotations_for(name.split(":", 1)[1])

    if name.startswith("method_calls:"):
        return db.latest_calls(name.split(":", 1)[1])

    if name == "missing_annotations":
        return db.missing_annotations()

    return getattr(db, name)


def filter_frame(frame: pd.DataFrame, text: str) -> pd.DataFrame:
    """Rows where any cell, or the index, contains `text` (case-insensitive)."""
    text = text.strip().lower()

    if not text or frame.empty:
        return frame

    shown = frame.reset_index()
    cells = shown.map(show) if hasattr(shown, "map") else shown.applymap(show)
    keep = cells.apply(
        lambda column: column.str.lower().str.contains(text, regex=False)
    )

    return frame[keep.any(axis=1).to_numpy()]


def _plain(value: Any) -> Any:
    """A DataFrame cell as the Python value the form's field reads."""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "item"):
        return value.item()

    return value


# A value longer than this, or over several lines, gets a box that wraps
LONG_VALUE = 60

# The most lines a long value's box shows before it scrolls
LONG_LINES = 12


def locked_value(text: str) -> QWidget:
    """
    A value that is shown, not typed: read-only, looking like a locked field,
    and its text can still be copied. A long one wraps over a few lines.
    """
    if "\n" not in text and len(text) <= LONG_VALUE:
        line = QLineEdit(text)
        line.setObjectName("locked")
        line.setReadOnly(True)
        line.setCursorPosition(0)

        return line

    box = QPlainTextEdit(text)
    box.setObjectName("locked")
    box.setReadOnly(True)
    box.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
    lines = sum(1 + len(part) // LONG_VALUE for part in text.split("\n"))
    spacing = box.fontMetrics().lineSpacing()
    margin = int(2 * box.document().documentMargin())
    box.setFixedHeight(min(lines, LONG_LINES) * spacing + margin + 16)

    return box


class DetailPane(QWidget):
    """
    One row as a form of locked fields, and its annotations if it has any,
    for the side panel of the Tables page. `buttons` go at the panel's foot.
    """

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.target_type: None | str = None
        self.target_id: None | int = None
        self.form: None | Form = None
        self.values: dict[str, QWidget] = {}  # each column's shown value

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*FORM_MARGINS_PX)
        self.title = QLabel("Pick a row in the table")
        self.title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.title)

        values = QWidget()
        self.rows = QFormLayout(values)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.rows.setVerticalSpacing(ROW_SPACING_PX)
        layout.addWidget(values)

        self.annotate_box = QWidget()
        annotate = QVBoxLayout(self.annotate_box)
        annotate.setContentsMargins(0, 12, 0, 0)
        self.annotate_title = QLabel("Annotations")
        self.annotate_title.setStyleSheet("font-weight: bold;")
        annotate.addWidget(self.annotate_title)
        self.form_holder = QVBoxLayout()
        annotate.addLayout(self.form_holder)
        layout.addWidget(self.annotate_box)
        layout.addStretch(1)
        self.annotate_box.hide()

        self.save = QPushButton("Save annotations")
        self.save.clicked.connect(self._save)
        self.save.hide()
        self.buttons = QHBoxLayout()
        self.buttons.addStretch(1)
        self.buttons.addWidget(self.save)

        self._fields: dict[str, list[Field]] = {}
        self._loaded: dict[str, Any] = {}

    def set_fields(self, fields: dict[str, list[Field]]) -> None:
        self._fields = fields

    def show_row(self, table: str, values: dict[str, Any], index_value: Any) -> None:
        """Show one row; `values` includes the index column."""
        self.title.setText(f"{table}: {show(index_value)}")

        while self.rows.rowCount():
            self.rows.removeRow(0)

        self.values = {}

        for key, value in values.items():
            shown = locked_value(show(value, pretty=True))
            self.values[str(key)] = shown
            self.rows.addRow(QLabel(str(key)), shown)

        target = TARGET_OF.get(table)

        if (
            target is None
            or target not in self._fields
            or not isinstance(index_value, numbers.Number)
        ):
            self.annotate_box.hide()
            self.save.hide()
            self.target_type = self.target_id = None
            return

        self.target_type = target
        self.target_id = int(index_value)
        self.annotate_title.setText(f"Annotations of {target} {self.target_id}")
        self._loaded = {}
        self._rebuild_form()
        self.annotate_box.show()
        self.save.show()
        self._load_current()

    def _rebuild_form(self) -> None:
        if self.form is not None:
            self.form.setParent(None)
            self.form.deleteLater()

        assert self.target_type is not None
        self.form = Form(self._fields[self.target_type])
        self.form_holder.addWidget(self.form)

    def _load_current(self) -> None:
        target_type, target_id = self.target_type, self.target_id

        def load(db):
            current = db.annotations_for(target_type)

            if target_id not in current.index:
                return {}

            row = current.loc[target_id]

            return {k: _plain(v) for k, v in row.items() if not pd.isna(v)}

        def loaded(values):
            if self.form is not None and self.target_id == target_id:
                self._loaded = values
                self.form.set_values(values)

        self.context.run(load, loaded, "load annotations", one_lock=True)

    def _save(self) -> None:
        if self.form is None or self.target_type is None or self.target_id is None:
            return

        target_type, target_id = self.target_type, self.target_id
        typed = self.form.values()

        # Only what changed is written: a blank that was blank stays unwritten,
        # a blank that held a value clears it
        values = {k: v for k, v in typed.items() if self._loaded.get(k) != v}
        values.update({k: None for k in self._loaded if k not in typed})

        if not values:
            self.context.status.emit("Nothing changed")
            return

        def write(db):
            db.add_annotations(
                target_type=target_type, target_id=target_id, values=values
            )

            return len(values)

        self.context.run(
            write,
            lambda n: self.context.status.emit(f"Saved {n} annotations"),
            "save annotations",
            writes=True,
            one_lock=True,
        )


class TablesPane(QWidget):
    """
    The table browser, laid out like the registry pages: the table on the
    left under its picker and filter, the picked row in the side panel.
    """

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.table_name: None | str = None
        self.frame: pd.DataFrame = pd.DataFrame()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(8, 8, 8, 8)
        top.setSpacing(8)
        self.picker = QComboBox()
        self.picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.picker.addItems(TABLES)
        self.picker.currentTextChanged.connect(self._pick)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter rows")
        self.filter.textChanged.connect(self._apply_filter)
        self.count = QLabel("")
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        top.addWidget(self.picker)
        top.addWidget(self.filter, stretch=1)
        top.addWidget(self.count)
        top.addWidget(self.reload_button)
        left_layout.addLayout(top)

        self.view = QTableView()
        self.view.setObjectName("registryTable")
        self.view.setModel(FrameModel())
        self.view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.view.setAlternatingRowColors(True)
        self.view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.view.clicked.connect(self._row_clicked)
        fit_columns(self.view)
        sortable(self.view)
        left_layout.addWidget(self.view)

        self.detail = DetailPane(context)
        self.split = TableAndForm(left, self.detail, self.detail.buttons)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.split)

        context.follow_changes(self, self.reload)
        self._pick(self.picker.currentText())

    def reload(self) -> None:
        """Re-read the shown table (and the annotation fields, once)."""
        self.context.run(
            call_tables, self._set_call_tables, "list calls", one_lock=True
        )

        if not self.detail._fields:

            def fields(db):
                return {
                    target: annotation_fields(db, target)
                    for target in ANNOTATION_TARGETS
                }

            self.context.run(
                fields, self.detail.set_fields, "load fields", one_lock=True
            )

        if self.table_name is None:
            return

        name = self.table_name

        def load(db):
            return read_table(db, name)

        def loaded(frame):
            if self.table_name == name:
                self.frame = frame
                self._apply_filter()

        self.context.run(load, loaded, f"read {name}", one_lock=True)

    def _set_call_tables(self, names: list[str]) -> None:
        """Keep the per-method entries after the fixed ones, without losing the pick."""
        picked = self.picker.currentText()
        wanted = [*TABLES, *names]
        shown = [self.picker.itemText(i) for i in range(self.picker.count())]

        if shown == wanted:
            return

        self.picker.blockSignals(True)
        self.picker.clear()
        self.picker.addItems(wanted)

        if picked in wanted:
            self.picker.setCurrentText(picked)

        self.picker.blockSignals(False)

    def _pick(self, name: str) -> None:
        self.table_name = name or None
        self.frame = pd.DataFrame()
        self._apply_filter()
        self.reload()

    def _apply_filter(self) -> None:
        shown = filter_frame(self.frame, self.filter.text())
        model = self.view.model()
        assert isinstance(model, FrameModel)
        model.set_frame(shown)
        self.count.setText(f"{len(shown)} of {len(self.frame)} rows")

    def _row_clicked(self, index) -> None:
        if not index.isValid() or self.table_name is None:
            return

        model = self.view.model()
        assert isinstance(model, FrameModel)
        row = index.row()
        self.detail.show_row(
            self.table_name, model.row_values(row), model.index_value(row)
        )
