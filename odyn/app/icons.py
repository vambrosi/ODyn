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
    QFontDatabase,
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
# A glyph is drawn to its own em box and a label is not, so a label is fitted
# to the box rather than given a fixed size -- otherwise a digit comes out half
# the height of a picture, and four characters run off the edge.
GLYPH_SCALE = 0.70
LABEL_HEIGHT = 0.70
LABEL_WIDTH = 0.92

# Labels are set narrow, which is what lets `m442` be read at all on a square
# icon: four characters across a square leave them a third of the height of a
# picture, and every character shaved off buys that back.
LABEL_STRETCH = 95

# Nothing is drawn right to the edge: a glyph's ink can reach past the box it
# is centred in, and the last row of pixels is where Qt's scaling loses it.
INSET = 0.94

# A badged glyph moves up and left to clear its corner, and the badge sits in
# the corner with a gap punched around it so the two never touch. The pair is
# then shrunk as a whole, so a badged icon covers the same band as a plain one
# rather than standing a quarter taller than everything beside it.
BADGED_SCALE = 1.0
BADGE_SCALE = 0.35
BADGE_GAP = 1.3
BADGED_TOGETHER = 0.90

# Which picture each section gets, the badge in its corner, and the characters
# to fall back to. The badge is what tells the two `+` apart: which thing is
# being added should not need a hover to find out.
GLYPHS = {
    "session": ("fa6s.clipboard", "", "s"),
    "odors": ("fa6s.vials", "", "o"),
    "notes": ("fa6s.note-sticky", "", "n"),
    "experiment": ("fa6s.microscope", "", "e"),
    "add-session": ("fa6s.clipboard", "fa6s.circle-plus", "+s"),
    "add-experiment": ("fa6s.microscope", "fa6s.circle-plus", "+e"),
}


def icon(name: str) -> QIcon:
    """The named section's icon."""
    glyph, badge, fallback = GLYPHS.get(name, ("", "", name[:1]))

    if not qtawesome:
        glyph = badge = ""

    return _shaded(glyph, badge, fallback, *_shades())


def label_icon(text: str) -> QIcon:
    """A short label, for a session's mouse or an experiment's name."""
    return _shaded("", "", text, *_shades())


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
def _shaded(glyph: str, badge: str, label: str, bright: str, dim: str) -> QIcon:
    """
    The two shades of one shape, as an icon Qt picks from by state.

    Keyed on the colours too, so a change of theme is a different entry rather
    than a stale one.
    """
    shape = _draw(glyph, badge, label)
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


def _draw(glyph: str, badge: str, label: str) -> QPixmap:
    """
    The glyph if there is one, else the label, as a shape for `_tint`.

    Drawn in black: only where it covers is kept, and the colour comes later.
    """
    pixmap = QPixmap(WIDTH, HEIGHT)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setPen(QColor("black"))

    if not glyph:
        painter.setFont(_label_font(label))
        painter.drawText(_box(1), Qt.AlignmentFlag.AlignCenter, label)
    elif not badge:
        painter.setFont(_glyph_font(GLYPH_SCALE))
        painter.drawText(_box(1), Qt.AlignmentFlag.AlignCenter, _character(glyph))
    else:
        _draw_badged(painter, glyph, badge)

    painter.end()

    return pixmap


def _draw_badged(painter: QPainter, glyph: str, badge: str) -> None:
    """The glyph in the top left, the badge in the bottom right corner."""
    painter.translate(WIDTH / 2, HEIGHT / 2)
    painter.scale(BADGED_TOGETHER, BADGED_TOGETHER)
    painter.translate(-WIDTH / 2, -HEIGHT / 2)

    painter.setFont(_glyph_font(GLYPH_SCALE * BADGED_SCALE))
    painter.drawText(
        _box(BADGED_SCALE, at=(0.0, 0.0)),
        Qt.AlignmentFlag.AlignCenter,
        _character(glyph),
    )

    corner = _box(BADGE_SCALE, at=(1.0, 1.0))

    # A gap punched out of whatever is underneath, so the badge reads as a
    # badge rather than as part of the picture it sits on.
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    painter.setBrush(QColor("black"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(corner.center(), *([corner.width() * BADGE_GAP / 2] * 2))

    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setPen(QColor("black"))
    painter.setFont(_glyph_font(BADGE_SCALE))
    painter.drawText(corner, Qt.AlignmentFlag.AlignCenter, _character(badge))


def _box(scale: float, at: tuple[float, float] = (0.5, 0.5)) -> QRectF:
    """
    A square `scale` of the drawable area, placed by `at` from (0, 0) to (1, 1).

    `at` is where in the leftover room the box sits, so `(0, 0)` is the top
    left corner, `(1, 1)` the bottom right and `(0.5, 0.5)` the middle.
    """
    margin = WIDTH * (1 - INSET) / 2
    room = WIDTH * INSET
    side = room * scale

    return QRectF(
        margin + (room - side) * at[0], margin + (room - side) * at[1], side, side
    )


def _character(glyph: str) -> str:
    """The one character an icon font draws a picture for."""
    return qtawesome.charmap(glyph)


def _glyph_font(scale: float) -> QFont:
    """The icon font, at `scale` of the canvas."""
    return qtawesome.font("fa6s", int(HEIGHT * scale))


def _label_font(text: str) -> QFont:
    """The font a label of this length is drawn in."""
    return _label_size(len(text))


@lru_cache(maxsize=None)
def _label_size(count: int) -> QFont:
    """
    A fixed-width font, as large as `count` characters fit the box at.

    Fixed-width and sized by the count rather than by the text, so every label
    of the same length comes out identically: measuring the text itself makes
    `e1` and `e2` different sizes, which reads as a mistake. Nothing is looked
    up by name, which keeps this clear of the alias search Qt does -- slowly --
    for a family the system has not got.
    """
    tall, wide = HEIGHT * LABEL_HEIGHT, WIDTH * LABEL_WIDTH

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setBold(True)
    font.setStretch(LABEL_STRETCH)

    # In case the system font is not actually fixed-width: `0` is then only a
    # stand-in for the widest character, so ask for the real thing as well.
    font.setStyleHint(QFont.StyleHint.Monospace)

    for pixels in range(int(tall * 1.6), 5, -1):
        font.setPixelSize(pixels)
        metrics = QFontMetrics(font)

        # `capHeight` is the font's own, not this text's, so a label with no
        # tall letters in it is not blown up to fill the box on its own.
        if (
            metrics.horizontalAdvance("0" * count) <= wide
            and (metrics.capHeight() or metrics.ascent()) <= tall
        ):
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
