"""
A pandas DataFrame shown as a Qt table.

`FrameModel` is read-only: the tables are what the database holds, and
changing them goes through the `Database` methods, never through a cell.
"""

from __future__ import annotations

import json
import math

from typing import Any

import pandas as pd

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class FrameModel(QAbstractTableModel):
    """A DataFrame as a table, with the index as its first column."""

    def __init__(self, frame: None | pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._columns: list[str] = []
        self._rows: list[list[Any]] = []

        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame
        shown = frame.reset_index()
        self._columns = [str(c) for c in shown.columns]
        self._rows = shown.values.tolist()
        self.endResetModel()

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section]

        return str(section + 1)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        value = self._rows[index.row()][index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            return show(value)

        if role == Qt.ItemDataRole.ToolTipRole:
            text = show(value)
            return text if len(text) > 40 else None

        if role == Qt.ItemDataRole.TextAlignmentRole:
            # Top-left, so a wrapped cell starts beside its neighbors
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        return None

    def row_values(self, row: int) -> dict[str, Any]:
        """One row as `{column: value}`, index included."""
        return dict(zip(self._columns, self._rows[row]))

    def index_value(self, row: int) -> Any:
        """The DataFrame index of a table row."""
        return self._frame.index[row]


# A column is sized to its contents up to this; JSON, logs and notes would
# otherwise push everything else off the screen
MAX_COLUMN_PX = 360

# Room around a column's name in its header
HEADER_PADDING_PX = 24


def fit_columns(view, max_px: int = MAX_COLUMN_PX) -> None:
    """
    Keep a view's columns as wide as their contents, as a double-click on
    every header would, capped at `max_px` so a long text column stays a
    column, and never narrower than their name: sized once a frame is set,
    and again on every reset.
    """

    def fit():
        view.resizeColumnsToContents()
        header = view.horizontalHeader()
        model = view.model()

        for column in range(header.count()):
            # The name read from the header's own font: some styles (macOS)
            # leave it out of the size they report for the section
            name = str(model.headerData(column, Qt.Orientation.Horizontal) or "")
            named = header.fontMetrics().horizontalAdvance(name) + HEADER_PADDING_PX
            width = min(header.sectionSize(column), max_px)
            header.resizeSection(column, max(width, named))

    view.model().modelReset.connect(fit)
    fit()


def show(value: Any, pretty: bool = False) -> str:
    """
    A cell's value as text; nothing for NULL, so tables read cleanly.

    `pretty` lays a JSON object out over lines, for a view that wraps.
    """
    if value is None:
        return ""

    if pretty and isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)

    if isinstance(value, float):
        if math.isnan(value):
            return ""

        return (
            str(int(value))
            if value.is_integer() and abs(value) < 1e15
            else f"{value:.6g}"
        )

    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""

        return (
            value.strftime("%Y-%m-%d %H:%M:%S")
            if value.time() != value.min.time()
            else value.strftime("%Y-%m-%d")
        )

    try:
        if pd.isna(value):
            return ""

    except (TypeError, ValueError):
        pass

    return str(value)
