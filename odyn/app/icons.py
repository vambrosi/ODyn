"""
The window's icons, drawn rather than loaded.

Every icon is a glyph from an icon font or a short label in the application's
own font, painted into the same square at the same size. One drawing path, so a
lettered session number and a pictured section cannot end up looking like
different sizes -- which is what happened while some icons came from files and
others were drawn.

**Every icon is a shape, not a picture.** Only its coverage is kept and the
colour is this module's: quiet at rest, the full text colour when hovered or
chosen. That is what lets one drawing follow a dark or a light theme.

The glyphs come from Material Design Icons, carried by `qtawesome`. Changing
one is changing a name in `GLYPHS`; the full set is at pictogrammers.com/library
/mdi. Without `qtawesome` installed the sections fall back to letters, so the
app still runs.

**USAGE**
```python
icon("odors")       # the scent glyph
label_icon("442")   # a session's number
```
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import QApplication

try:
    import qtawesome
except ImportError:  # pragma: no cover - only on an install without the extra
    qtawesome = None

# Drawn at this size and scaled by Qt, so it only has to be big enough not to
# look soft on a high-density screen. Wider than tall because a mouse number is
# three digits: in a square, fitting those across leaves them half the height of
# a picture, and the bar reads as two different sizes.
WIDTH, HEIGHT = 48, 48

# How far the resting colour sits from the text colour, towards the background.
DIMMED = 0.45

# How much of the square a glyph fills, and the box a label is fitted into.
# A glyph is drawn to its own em box and a label is not, so a label is measured
# and shrunk to match rather than given a fixed size -- otherwise a digit comes
# out half the height of a picture, and three digits run off the edge.
GLYPH_SCALE = 0.70
LABEL_HEIGHT = 0.70
LABEL_WIDTH = 0.70

# Which picture each section gets, and the letter to fall back to.
GLYPHS = {
    "session": ("fa6s.clipboard", "s"),
    "odors": ("fa6s.vials", "o"),
    "notes": ("fa6s.note-sticky", "n"),
    "experiment": ("fa6s.microscope", "e"),
    "add": ("fa6s.plus", "+"),
}


def icon(name: str) -> QIcon:
    """The named section's icon."""
    glyph, fallback = GLYPHS.get(name, ("", name[:1]))

    return _shaded(glyph if qtawesome else "", fallback, *_shades())


def label_icon(text: str) -> QIcon:
    """A short label, for a session's number in the left bar."""
    return _shaded("", text, *_shades())


def _shades() -> tuple[str, str]:
    """The resting and the active colour, both from the palette."""
    palette = QApplication.palette()
    bright = palette.color(QPalette.ColorRole.WindowText)
    behind = palette.color(QPalette.ColorRole.Window)

    return bright.name(), _towards(bright, behind, DIMMED).name()


def _towards(colour: QColor, other: QColor, amount: float) -> QColor:
    """`colour` moved `amount` of the way to `other`."""
    return QColor(
        round(colour.red() + (other.red() - colour.red()) * amount),
        round(colour.green() + (other.green() - colour.green()) * amount),
        round(colour.blue() + (other.blue() - colour.blue()) * amount),
    )


@lru_cache(maxsize=None)
def _shaded(glyph: str, label: str, bright: str, dim: str) -> QIcon:
    """
    The two shades of one shape, as an icon Qt picks from by state.

    Keyed on the colours too, so a change of theme is a different entry rather
    than a stale one.
    """
    shape = _draw(glyph, label)
    made = QIcon()

    for shade, states in (
        (dim, ((QIcon.Mode.Normal, QIcon.State.Off),)),
        (
            bright,
            (
                (QIcon.Mode.Normal, QIcon.State.On),
                (QIcon.Mode.Active, QIcon.State.Off),
                (QIcon.Mode.Active, QIcon.State.On),
                (QIcon.Mode.Selected, QIcon.State.Off),
                (QIcon.Mode.Selected, QIcon.State.On),
            ),
        ),
    ):
        tinted = _tint(shape, shade)

        for mode, state in states:
            made.addPixmap(tinted, mode, state)

    return made


def _draw(glyph: str, label: str) -> QPixmap:
    """
    The glyph if there is one, else the label, as a shape for `_tint`.

    Drawn in black: only where it covers is kept, and the colour comes later.
    """
    pixmap = QPixmap(WIDTH, HEIGHT)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    text, font = _glyph_font(glyph) if glyph else (label, _label_font(label))

    painter.setFont(font)
    painter.setPen(QColor("black"))
    painter.drawText(QRectF(0, 0, WIDTH, HEIGHT), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return pixmap


def _glyph_font(glyph: str) -> tuple[str, QFont]:
    """The character and the icon font it is drawn in."""
    font = qtawesome.font(glyph.split(".")[0], int(HEIGHT * GLYPH_SCALE))

    return qtawesome.charmap(glyph), font


def _label_font(text: str) -> QFont:
    """
    The application's own font, sized so `text` fills the same box as a glyph.

    Measured rather than fixed, so one digit is as tall as a picture and four
    still fit. Nothing is looked up by name, which keeps this clear of the
    alias search Qt does -- slowly -- for a family the system has not got.
    """
    tall, wide = HEIGHT * LABEL_HEIGHT, WIDTH * LABEL_WIDTH

    font = QFont(QApplication.font())
    font.setBold(True)

    # Set close together for a long number, which buys back some of the height
    # that fitting three or four digits across the square otherwise costs.
    if len(text) > 2:
        font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 88)

    for pixels in range(int(tall * 1.4), 7, -1):
        font.setPixelSize(pixels)
        drawn = QFontMetrics(font).tightBoundingRect(text)

        if drawn.width() <= wide and drawn.height() <= tall:
            break

    return font


def _tint(shape: QPixmap, ink: str) -> QPixmap:
    """The same shape in one flat colour, keeping only where it covers."""
    tinted = QPixmap(shape.size())
    tinted.fill(Qt.GlobalColor.transparent)

    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, shape)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QColor(ink))
    painter.end()

    return tinted
