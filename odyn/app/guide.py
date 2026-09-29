"""
Guide to what the side bar buttons do, shown when no session is selected.

It shows a transparent overlay with arrows pointing to each button. Each
arrow has a text at its base explaining the button's function. Position of
the arrows are read every time the window is painted, so it is shown in the
correct position if the window is resized.

A label that would overlap another label, or that would overlap the page's
message, is left out (a small window drops some rather than piling them up).
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QPainterPath, QPalette, QPen, QPolygonF
from PySide6.QtWidgets import QApplication, QToolBar, QToolButton, QWidget

from .style import RADIUS

# How far a label sits from its button, and the arrow head's size
ARROW_PX = 56
HEAD_PX = 10

# The gap left between an arrow's tip and its button
TIP_GAP_PX = 12
PADDING_PX = (8, 3)


@dataclass
class Pointer:
    """One label and its arrow, in the layer's frame."""

    text: str
    label: QRect
    start: QPointF
    end: QPointF


class Guide(QWidget):
    """
    The overlay layer.

    `targets()` gives `(bar, actions, texts)` per bar with:
    - the actions for each of the bar's buttons;
    - the texts for each arrow (or their button status tip if not given).

    `keep_clear` is a widget no label may cover.
    It covers its parent and follows its size.
    """

    def __init__(self, parent: QWidget, targets, keep_clear: None | QWidget = None):
        super().__init__(parent)
        self.targets = targets
        self.keep_clear = keep_clear
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.parent() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())

        return False

    def set_active(self, active: bool) -> None:
        self.setVisible(active)

        if active:
            self.raise_()
            self.update()

    # ----------------------------------------------------------------- #
    # Where everything goes
    # ----------------------------------------------------------------- #

    def pointers(self) -> list[Pointer]:
        """Every label that fits, with its arrow, left bar then right bar."""
        metrics = self.fontMetrics()
        placed: list[Pointer] = []
        clear = self._clear_rect()

        for bar, actions, texts in self.targets():
            for action in actions:
                button = bar.widgetForAction(action)

                if not isinstance(button, QToolButton) or not button.isVisible():
                    continue

                text = texts.get(action) or action.statusTip()
                pointer = self._point_at(bar, button, text, metrics)

                if clear is not None and pointer.label.intersects(clear):
                    continue

                if not any(pointer.label.intersects(p.label) for p in placed):
                    placed.append(pointer)

        return placed

    def _clear_rect(self) -> None | QRect:
        """Where `keep_clear`'s text is, in the layer's frame."""
        widget = self.keep_clear

        if widget is None or not widget.isVisible():
            return None

        size = widget.fontMetrics().size(0, widget.text())
        center = widget.mapTo(self.parent(), widget.rect().center())

        return QRect(0, 0, size.width(), size.height()).translated(
            center.x() - size.width() // 2, center.y() - size.height() // 2
        )

    def _point_at(self, bar: QToolBar, button: QToolButton, text: str, metrics):
        corner = button.mapTo(self.parent(), button.rect().topLeft())
        box = QRect(corner, button.size())

        # A bar left of the middle points right, one right of it points left
        left_bar = box.center().x() < self.width() / 2
        width = metrics.horizontalAdvance(text) + 2 * PADDING_PX[0]
        height = metrics.height() + 2 * PADDING_PX[1]
        y = box.center().y()

        if left_bar:
            tip = QPointF(box.right() + TIP_GAP_PX, y)
            label = QRect(int(tip.x()) + ARROW_PX, y - height // 2, width, height)
            start = QPointF(label.left(), y)

        else:
            tip = QPointF(box.left() - TIP_GAP_PX, y)
            label = QRect(
                int(tip.x()) - ARROW_PX - width, y - height // 2, width, height
            )
            start = QPointF(label.right(), y)

        return Pointer(text, label, start, tip)

    # ----------------------------------------------------------------- #
    # Drawing
    # ----------------------------------------------------------------- #

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt's name)
        palette = QApplication.palette()
        ink = palette.color(QPalette.ColorRole.Highlight)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for pointer in self.pointers():
            painter.setPen(QPen(ink, 1.5))
            painter.drawLine(pointer.start, pointer.end)
            self._head(painter, pointer, ink)

            # The label boxed like the bottom bar's: readable over anything
            painter.setBrush(palette.color(QPalette.ColorRole.Window))
            painter.setPen(QPen(ink, 1))
            path = QPainterPath()
            path.addRoundedRect(QRectF(pointer.label), RADIUS, RADIUS)
            painter.drawPath(path)
            painter.setPen(palette.color(QPalette.ColorRole.WindowText))
            painter.drawText(pointer.label, Qt.AlignmentFlag.AlignCenter, pointer.text)

        painter.end()

    @staticmethod
    def _head(painter: QPainter, pointer: Pointer, ink) -> None:
        direction = 1 if pointer.end.x() < pointer.start.x() else -1
        tip = pointer.end
        back = tip.x() + direction * HEAD_PX

        painter.setBrush(ink)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(
            QPolygonF(
                [
                    tip,
                    QPointF(back, tip.y() - HEAD_PX / 2),
                    QPointF(back, tip.y() + HEAD_PX / 2),
                ]
            )
        )
