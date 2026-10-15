"""
A pandas DataFrame shown as a Qt table.

`FrameModel` is read-only: the tables are what the database holds, and
changing them goes through the `Database` methods, never through a cell.
A click on a column's header sorts by it (`sortable`); `source_row` turns a
row of the view back into the row of the frame it shows.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import numbers

from typing import Any

import pandas as pd

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class FrameModel(QAbstractTableModel):
    """
    A DataFrame as a table, with the index as its first column, unless it is
    only the rows' order (unnamed 0, 1, 2...), which the row numbers show.
    """

    def __init__(self, frame: None | pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._columns: list[str] = []
        self._rows: list[list[Any]] = []
        self._unsorted: list[list[Any]] = []
        self._order: list[int] = []  # the frame's row shown in each view row
        self._sorted_by: None | tuple[str, Qt.SortOrder] = None

        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame: pd.DataFrame) -> None:
        """Show `frame`, still sorted by the column picked, if it has it."""
        self.beginResetModel()
        self._frame = frame
        order_only = frame.index.name is None and isinstance(frame.index, pd.RangeIndex)
        shown = frame.reset_index(drop=order_only)
        self._columns = [str(c) for c in shown.columns]
        self._unsorted = shown.values.tolist()
        self._arrange()
        self.endResetModel()

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        """
        Sort by `column` (Qt calls this on a header click); a negative one is
        the frame's own order. Blanks go last either way.
        """
        self.layoutAboutToBeChanged.emit()
        self._sorted_by = (
            (self._columns[column], order) if 0 <= column < len(self._columns) else None
        )
        self._arrange()
        self.layoutChanged.emit()

    def _arrange(self) -> None:
        order = list(range(len(self._unsorted)))

        if self._sorted_by is not None and self._sorted_by[0] in self._columns:
            name, direction = self._sorted_by
            column = self._columns.index(name)
            values = [row[column] for row in self._unsorted]
            filled = [i for i in order if not _blank(values[i])]
            filled.sort(
                key=lambda i: _sort_key(values[i]),
                reverse=direction == Qt.SortOrder.DescendingOrder,
            )
            order = filled + [i for i in order if _blank(values[i])]

        self._order = order
        self._rows = [self._unsorted[i] for i in order]

    def source_row(self, row: int) -> int:
        """The position in `frame` of what view row `row` shows."""
        return self._order[row]

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
        return self._frame.index[self._order[row]]


def sortable(view) -> None:
    """
    Let a click on a column's header sort `view` (a `FrameModel`'s), keeping
    the frame's own order until one is clicked.
    """
    view.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    view.setSortingEnabled(True)


def _blank(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # a list or a dict in a cell
        return False


def _sort_key(value: Any) -> tuple:
    """Numbers by value, then dates, then text (ignoring case)."""
    if isinstance(value, bool):
        return (0, int(value), "")
    if isinstance(value, numbers.Number):
        return (0, float(value), "")
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return (1, pd.Timestamp(value).value, "")

    return (2, 0, show(value).lower())


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

        # A date (midnight) without its time; `Timestamp.min` is not midnight
        return (
            value.strftime("%Y-%m-%d")
            if value == value.normalize()
            else value.strftime("%Y-%m-%d %H:%M:%S")
        )

    try:
        if pd.isna(value):
            return ""

    except (TypeError, ValueError):
        pass

    return str(value)
