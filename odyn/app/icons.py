"""
The window's icons.

Placeholders are drawn rather than loaded: a letter or a number in a rounded
square, painted with the application's own font. Nothing looks a font family up
by name, which is what makes this free of the alias search Qt does -- and slow
to do -- when an SVG names a family the system has not got.

Real artwork goes in `icons/` as `<name>.svg` and is used in preference, so
replacing a placeholder is dropping a file in beside them. SVG rather than one
sprite sheet: Qt rasterizes it per screen, so the same file stays sharp on a
laptop and on an external monitor.

**USAGE**
```python
icon("session")        # session.svg if it is there, else a drawn "s"
label_icon("442")      # a drawn number, for a session in the left bar
```
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap

FOLDER = Path(__file__).parent / "icons"

# Drawn at this size and scaled by Qt, so it only has to be big enough not to
# look soft on a high-density screen.
SIZE = 64

INK = "#a0a4a8"

# What each section is drawn as until someone supplies artwork for it.
PLACEHOLDERS = {
    "session": "s",
    "odors": "o",
    "notes": "n",
    "experiment": "e",
    "add": "+",
}


@lru_cache(maxsize=None)
def icon(name: str) -> QIcon:
    """The named icon: its own SVG if there is one, else a drawn placeholder."""
    path = FOLDER / f"{name}.svg"

    if path.exists():
        return QIcon(str(path))

    return label_icon(PLACEHOLDERS.get(name, name[:1]))


@lru_cache(maxsize=None)
def label_icon(text: str) -> QIcon:
    """
    A short label in a rounded square, drawn in the application's own font.

    Used for the section icons and for a session's number, so every slot in the
    bar is the same size and reads the same way.
    """
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    edge = SIZE * 0.08
    box = QRectF(edge, edge, SIZE - 2 * edge, SIZE - 2 * edge)

    painter.setPen(QPen(QColor(INK), SIZE * 0.06))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(box, SIZE * 0.2, SIZE * 0.2)

    # Whatever the app is already using, so no family is looked up by name and
    # a long label still fits inside the square.
    font = QFont(painter.font())
    font.setPixelSize(int(SIZE * (0.5 if len(text) < 3 else 0.34)))
    font.setBold(True)
    painter.setFont(font)

    painter.setPen(QColor(INK))
    painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return QIcon(pixmap)
