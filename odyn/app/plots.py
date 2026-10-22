"""
The shifts of the test runs over their frames, under the grid of videos.

Three rows share the frame axis: dy, dx and each frame's correlation with the
reference. Every run's line goes on the same axes, so the runs compare at a
glance; the ranges cover every run loaded (not only those shown), so they
stay put as the runs shown change. For piecewise rigid, the 10th to 90th
percentile of the patches is a band around the median. The raw frames'
correlation is a gray line under the others. Light lines mark where an
acquisition starts, and a marker follows the frame the videos show.

The data are the runs' `shifts_summary.npz` (see `mcor_report`).

Drawn with QPainter, as each pixel column's lowest and highest value, so a
test of tens of thousands of frames draws at once; kept as a picture, so
only the marker is drawn as the videos play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..mcor_report import SHIFTS_SUMMARY

# The rows: the key in the data, its label, and whether 0 gets a line (the
# shifts) or is the bottom of the axis (the correlation, never below 0 here)
ROWS = (
    ("dy", "dy (px)", "line"),
    ("dx", "dx (px)", "line"),
    ("corr", "corr", "floor"),
)

# Room for the labels at the left, and between the rows, in px
LEFT_PX = 64
GAP_PX = 6
ROW_MIN_PX = 40

RAW_COLOR = QColor(150, 150, 150)
EXCLUDED_COLOR = QColor(235, 87, 87, 50)  # an acquisition left out

# Distinct colors for the runs, in order (as a tile shows its run's)
PALETTE = (
    "#4e9bff",
    "#ff7a45",
    "#3ecf8e",
    "#d65bd6",
    "#f2c94c",
    "#56ccf2",
    "#eb5757",
    "#9b8cff",
    "#6fcf97",
    "#f2994a",
)


def run_color(index: int) -> QColor:
    """The color of the run at `index` among the candidates."""
    return QColor(PALETTE[index % len(PALETTE)])


@dataclass
class Series:
    """One run's per-frame numbers, and the color it is drawn in."""

    name: str
    color: QColor
    data: dict[str, np.ndarray] = field(repr=False)


def read_summary(folder: Path) -> None | dict[str, np.ndarray]:
    """The `shifts_summary.npz` of a run folder, or `None` without one."""
    try:
        with np.load(Path(folder) / SHIFTS_SUMMARY) as content:
            return {key: np.asarray(content[key]) for key in content.files}
    except (OSError, ValueError, KeyError):
        return None


class ShiftPlots(QWidget):
    """The plots (see the module description)."""

    def __init__(self, rows=ROWS, parent=None):
        super().__init__(parent)
        self.rows = tuple(rows)  # see `ROWS`
        self.series: list[Series] = []
        self.shown: list[str] = []  # the names of the series drawn
        self.raw: None | np.ndarray = None  # the raw frames' correlation
        self.edges: np.ndarray = np.zeros(0, int)  # where each acquisition starts
        self.frames = 0
        self.marker: None | float = None
        self.ranges: dict[str, tuple[float, float]] = {}
        self._picture: None | QPixmap = None  # the plots, without the marker
        self.excluded: list[tuple[float, float]] = []  # frame spans, shaded
        count = len(self.rows)
        self.setMinimumHeight(count * ROW_MIN_PX + (count - 1) * GAP_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_series(self, series: list[Series]) -> None:
        """Every run that can be shown; the ranges cover them all."""
        self.series = series
        self.frames = max(
            (
                len(s.data[key])
                for s in series
                for key, _, _ in self.rows
                if key in s.data
            ),
            default=0,
        )
        first = series[0].data if series else {}
        counts = first.get("frame_counts")
        self.edges = (
            np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
            if counts is not None
            else np.zeros(0, int)
        )
        self.raw = next(
            (s.data["raw_corr"] for s in series if "raw_corr" in s.data), None
        )
        self.ranges = {}

        for key, _, zero in self.rows:
            values = []

            for s in series:
                for name in (key, f"{key}_p10", f"{key}_p90"):
                    if name in s.data:
                        values.append(s.data[name])

            if key == "corr" and self.raw is not None:
                values.append(self.raw)

            if not values:
                continue

            joined = np.concatenate([np.ravel(v) for v in values])
            joined = joined[np.isfinite(joined)]

            if not len(joined):
                continue

            low, high = float(joined.min()), float(joined.max())

            if zero:
                low, high = min(low, 0.0), max(high, 0.0)

            pad = (high - low) * 0.05 or 1.0
            low, high = low - pad, high + pad

            if zero == "floor":  # a correlation: from 0, and never above 1
                low, high = 0.0, min(high, 1.0)

            self.ranges[key] = (low, high)

        self._redraw()

    def show_runs(self, names: list[str]) -> None:
        if list(names) != self.shown:
            self.shown = list(names)
            self._redraw()

    def set_excluded(self, spans: list[tuple[float, float]]) -> None:
        """Shade these spans of frames (acquisitions left out)."""
        self.excluded = list(spans)
        self._redraw()

    def set_marker(self, frame: None | float) -> None:
        """Mark `frame` (an index into the plotted frames), or nothing."""
        self.marker = frame
        self.update()

    # ----------------------------------------------------------------------- #
    # Drawing
    # ----------------------------------------------------------------------- #

    def _rows(self) -> list[QRectF]:
        # A pixel inside the widget all round, so no frame line is cut off
        count = len(self.rows)
        height = (self.height() - (count - 1) * GAP_PX - 2) / count
        width = self.width() - LEFT_PX - 8

        return [
            QRectF(LEFT_PX, 1 + i * (height + GAP_PX), width, height)
            for i in range(count)
        ]

    def _x(self, rect: QRectF, frame: float) -> float:
        return rect.left() + frame / max(1, self.frames - 1) * rect.width()

    def _y(self, rect: QRectF, key: str, value: float) -> float:
        low, high = self.ranges[key]

        return rect.bottom() - (value - low) / (high - low) * rect.height()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        """
        The cached plots, and the marker over them (all that changes as the
        videos play).
        """
        ratio = self.devicePixelRatioF()
        size = QSize(round(self.width() * ratio), round(self.height() * ratio))

        if self._picture is None or self._picture.size() != size:
            picture = QPixmap(size)
            picture.setDevicePixelRatio(ratio)
            picture.fill(Qt.GlobalColor.transparent)
            self._draw_plots(QPainter(picture))
            self._picture = picture  # only once drawn whole

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._picture)
        text = self.palette().color(self.foregroundRole())

        if self.marker is not None and self.frames:
            painter.setPen(QPen(text, 2))

            for rect in self._rows():
                x = self._x(rect, self.marker)
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

        painter.end()

    def _redraw(self) -> None:
        """Draw the plots again (new runs, size or colors), not just the marker."""
        self._picture = None
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self._picture = None
        super().resizeEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        self._picture = None  # a theme change, say
        super().changeEvent(event)

    def _draw_plots(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        text = self.palette().color(self.foregroundRole())
        line = QColor(text)
        line.setAlpha(40)
        drawn = [s for s in self.series if s.name in self.shown]

        for rect, (key, label, zero) in zip(self._rows(), self.rows):
            painter.setPen(QPen(line, 1))
            painter.drawRect(rect)
            painter.setPen(text)
            painter.drawText(
                QRectF(0, rect.top(), LEFT_PX - 8, rect.height()),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            if key not in self.ranges or not self.frames:
                continue

            low, high = self.ranges[key]

            for value, align in (
                (high, Qt.AlignmentFlag.AlignTop),
                (low, Qt.AlignmentFlag.AlignBottom),
            ):
                painter.drawText(
                    QRectF(0, rect.top(), LEFT_PX - 8, rect.height()),
                    Qt.AlignmentFlag.AlignRight | align,
                    f"{value:.2f}" if key == "corr" else f"{value:.1f}",
                )

            painter.setPen(QPen(line, 1))

            for start in self.edges[1:]:
                x = self._x(rect, start)
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))

            if zero == "line":
                y = self._y(rect, key, 0.0)
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

            painter.save()
            painter.setClipRect(rect)

            for start, end in self.excluded:
                left, right = self._x(rect, start), self._x(rect, end)
                painter.fillRect(
                    QRectF(left, rect.top(), max(1.0, right - left), rect.height()),
                    EXCLUDED_COLOR,
                )

            if key == "corr" and self.raw is not None:
                self._draw_line(painter, rect, key, self.raw, RAW_COLOR)

            for s in drawn:
                if f"{key}_p10" in s.data and f"{key}_p90" in s.data:
                    band = QColor(s.color)
                    band.setAlpha(60)
                    self._draw_band(
                        painter,
                        rect,
                        key,
                        s.data[f"{key}_p10"],
                        s.data[f"{key}_p90"],
                        band,
                    )

                if key in s.data:
                    self._draw_line(painter, rect, key, s.data[key], s.color)

            painter.restore()

        painter.end()

    def _columns(self, rect: QRectF, values: np.ndarray):
        """Each pixel column's lowest and highest value, with its x."""
        columns = max(1, int(rect.width()))
        values = np.asarray(values, float)
        bins = np.linspace(0, len(values), min(columns, len(values)) + 1).astype(int)
        low = np.minimum.reduceat(values, bins[:-1])
        high = np.maximum.reduceat(values, bins[:-1])
        xs = [self._x(rect, (a + b - 1) / 2) for a, b in zip(bins[:-1], bins[1:])]

        return xs, low, high

    def _draw_line(self, painter, rect, key, values, color) -> None:
        xs, low, high = self._columns(rect, values)
        painter.setPen(QPen(color, 1.2))
        path = QPainterPath()

        for i, (x, a, b) in enumerate(zip(xs, low, high)):
            if not (np.isfinite(a) and np.isfinite(b)):
                continue

            top, bottom = self._y(rect, key, b), self._y(rect, key, a)

            if i == 0 or path.elementCount() == 0:
                path.moveTo(x, bottom)
            else:
                path.lineTo(x, bottom)

            path.lineTo(x, top)

        painter.drawPath(path)

    def _draw_band(self, painter, rect, key, lows, highs, color) -> None:
        xs, low, _ = self._columns(rect, lows)
        _, _, high = self._columns(rect, highs)
        points = [QPointF(x, self._y(rect, key, b)) for x, b in zip(xs, high)]
        points += [
            QPointF(x, self._y(rect, key, a)) for x, a in zip(reversed(xs), low[::-1])
        ]
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF(points))
        painter.setBrush(Qt.BrushStyle.NoBrush)
