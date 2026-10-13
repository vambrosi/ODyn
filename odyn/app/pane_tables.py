"""
The tables, as they are: pick one on the left, filter it, click a row to see
it whole (and its annotations, when it is something that can be annotated).

Nothing here writes. The one thing it offers beyond looking is annotating the
selected row, for the once-a-day and after-the-fact facts (a mouse's weight
was typed wrong, a loop was bad) that do not belong to a draft.
"""

from __future__ import annotations

import numbers

from typing import Any

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..utils import ANNOTATION_TARGETS
from .context import Context
from .fields import Field, annotation_fields
from .forms import Form
from .models import FrameModel, fit_columns, show

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
    calls = db.method_calls

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


class DetailPane(QWidget):
    """One row as key/value lines, and its annotations if it has any."""

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.target_type: None | str = None
        self.target_id: None | int = None
        self.form: None | Form = None

        layout = QVBoxLayout(self)
        self.title = QLabel("Pick a row")
        self.title.setStyleSheet("font-weight: bold")
        layout.addWidget(self.title)

        # One column of values that wrap, so a note or a JSON reads whole
        self.row_view = QTableView()
        self.row_view.setModel(FrameModel())
        self.row_view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.row_view.setWordWrap(True)
        self.row_view.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self.row_view.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.row_view.horizontalHeader().sectionResized.connect(
            lambda *_: self.row_view.resizeRowsToContents()
        )
        layout.addWidget(self.row_view, stretch=2)

        self.annotate_box = QWidget()
        annotate = QVBoxLayout(self.annotate_box)
        annotate.setContentsMargins(0, 0, 0, 0)
        self.annotate_title = QLabel("Annotations")
        annotate.addWidget(self.annotate_title)
        self.form_holder = QVBoxLayout()
        annotate.addLayout(self.form_holder)
        buttons = QHBoxLayout()
        self.save = QPushButton("Save annotations")
        self.save.clicked.connect(self._save)
        buttons.addStretch()
        buttons.addWidget(self.save)
        annotate.addLayout(buttons)
        layout.addWidget(self.annotate_box, stretch=3)
        self.annotate_box.hide()

        self._fields: dict[str, list[Field]] = {}
        self._loaded: dict[str, Any] = {}

    def set_fields(self, fields: dict[str, list[Field]]) -> None:
        self._fields = fields

    def show_row(self, table: str, values: dict[str, Any], index_value: Any) -> None:
        """Show one row; `values` includes the index column."""
        self.title.setText(f"{table}: {show(index_value)}")

        frame = pd.DataFrame(
            {"value": [show(v, pretty=True) for v in values.values()]},
            index=pd.Index(list(values.keys()), name="column"),
        )
        model = self.row_view.model()
        assert isinstance(model, FrameModel)
        model.set_frame(frame)

        # The column names never wrap: the name column is as wide as the
        # widest of them, the value column takes the rest
        metrics = self.row_view.fontMetrics()
        widest = max(metrics.horizontalAdvance(str(k)) for k in values) + 24
        self.row_view.setColumnWidth(0, widest)
        self.row_view.resizeRowsToContents()

        target = TARGET_OF.get(table)

        if (
            target is None
            or target not in self._fields
            or not isinstance(index_value, numbers.Number)
        ):
            self.annotate_box.hide()
            self.target_type = self.target_id = None
            return

        self.target_type = target
        self.target_id = int(index_value)
        self.annotate_title.setText(f"Annotations of {target} {self.target_id}")
        self._loaded = {}
        self._rebuild_form()
        self.annotate_box.show()
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

        self.context.run(load, loaded, "load annotations")

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
        )


class TablesPane(QWidget):
    """The table browser."""

    def __init__(self, context: Context, parent=None):
        super().__init__(parent)
        self.context = context
        self.table_name: None | str = None
        self.frame: pd.DataFrame = pd.DataFrame()

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        layout.addWidget(splitter)

        self.list = QListWidget()
        self.list.addItems(TABLES)
        self.list.currentTextChanged.connect(self._pick)
        splitter.addWidget(self.list)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        top = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter rows")
        self.filter.textChanged.connect(self._apply_filter)
        self.count = QLabel("")
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        top.addWidget(self.filter, stretch=1)
        top.addWidget(self.count)
        top.addWidget(self.reload_button)
        middle_layout.addLayout(top)

        self.view = QTableView()
        self.view.setModel(FrameModel())
        self.view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.view.setAlternatingRowColors(True)
        self.view.setSortingEnabled(False)
        self.view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.view.clicked.connect(self._row_clicked)
        fit_columns(self.view)
        middle_layout.addWidget(self.view)
        splitter.addWidget(middle)

        self.detail = DetailPane(context)
        splitter.addWidget(self.detail)
        # Fixed start widths: the list fits its longest name, the table is the
        # only pane that grows with the window
        splitter.setSizes([330, 760, 400])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        context.changed.connect(self.reload)
        self.list.setCurrentRow(0)

    def reload(self) -> None:
        """Re-read the shown table (and the annotation fields, once)."""
        self.context.run(call_tables, self._set_call_tables, "list calls")

        if not self.detail._fields:

            def fields(db):
                return {
                    target: annotation_fields(db, target)
                    for target in ANNOTATION_TARGETS
                }

            self.context.run(fields, self.detail.set_fields, "load fields")

        if self.table_name is None:
            return

        name = self.table_name

        def load(db):
            return read_table(db, name)

        def loaded(frame):
            if self.table_name == name:
                self.frame = frame
                self._apply_filter()

        self.context.run(load, loaded, f"read {name}")

    def _set_call_tables(self, names: list[str]) -> None:
        """Keep the per-method entries after the fixed ones, without losing the pick."""
        current = self.list.currentItem()
        picked = current.text() if current else None
        wanted = [*TABLES, *names]
        shown = [self.list.item(i).text() for i in range(self.list.count())]

        if shown == wanted:
            return

        self.list.blockSignals(True)
        self.list.clear()
        self.list.addItems(wanted)

        if picked in wanted:
            self.list.setCurrentRow(wanted.index(picked))

        self.list.blockSignals(False)

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
