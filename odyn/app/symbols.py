"""
The window's icons, drawn rather than loaded.

Every icon is a glyph from the Material Symbols font or a short label in the
application's own font, painted into the same square at the same size. One
drawing path, so a lettered label and a pictured section cannot end up
looking like different sizes.

**Every icon is a shape, not a picture.** Only its coverage is kept and the
color is this module's: quiet at rest, the full text color when hovered, and
the highlighted-text color, filled in, when chosen. That is what lets one
drawing follow a dark or a light theme.

The glyphs come from Material Symbols Rounded (`assets/material-symbols`), a
variable font: a glyph is named by its icon name (the font turns `science`
into a flask), and its `FILL` axis draws the same glyph outlined or filled.
Changing an icon is changing a name in `GLYPHS`; the full set is at
fonts.google.com/icons. Without the font the sections fall back to letters, so
the app still runs.

**USAGE**
```python
icon("odors")       # the flask glyph
label_icon("m442")  # a session, by its mouse
```
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

FONT_FILE = (
    Path(__file__).parent / "assets/material-symbols" / "MaterialSymbolsRounded.ttf"
)

# Drawn at this size and scaled by Qt, so it only has to be big enough not to
# look soft on a high-density screen.
WIDTH, HEIGHT = 48, 48

# How far the resting color sits from the text color, towards the background.
DIMMED = 0.45

# How much of the square a glyph's em box fills (the font leaves room around
# each picture inside its em box, so this is larger than a label's box), and
# the box a label is fitted into.
GLYPH_SCALE = 0.92
LABEL_HEIGHT = 0.70
LABEL_WIDTH = 0.92

# The font's weight axis (100-700); 400 matches the text beside it.
GLYPH_WEIGHT = 400

# Labels are set narrow, which is what lets `m442` be read on a square icon.
LABEL_STRETCH = 95

# Nothing is drawn right to the edge: a glyph's ink can reach past the box it
# is centered in, and the last row of pixels is where Qt's scaling loses it.
INSET = 0.94

# A badged glyph moves up and left to clear its corner, and the badge sits in
# the corner with a gap punched around it so the two never touch. The pair is
# then shrunk as a whole, so a badged icon covers the same band as a plain one.
BADGED_SCALE = 1.0
BADGE_SCALE = 0.45
BADGE_GAP = 1.1
BADGED_TOGETHER = 0.90

# Which picture each section gets, the badge in its corner, and the characters
# to fall back to. The badge is what tells two `+` apart.
GLYPHS = {
    "users": ("person", "", "u"),
    "mice": ("pest_control_rodent", "", "m"),
    "odors": ("labs", "", "o"),
    "panels": ("lab_panel", "", "p"),
    "tables": ("table", "", "t"),
    "qc": ("fact_check", "", "q"),
    "add-session": ("calendar_today", "add_circle", "+s"),
    "add-experiment": ("experiment", "add_circle", "+e"),
    "undo": ("undo", "", "<"),
    "remove": ("delete", "", "x"),
    "check-session": ("checklist", "", "c"),
    "submit-session": ("upload", "", "u"),
    "recorder": ("radio_button_checked", "", "r"),
    "panel-open": ("keyboard_arrow_up", "", "^"),
    "panel-close": ("keyboard_arrow_down", "", "v"),
    "form-open": ("right_panel_open", "", "<"),
    "form-close": ("right_panel_close", "", ">"),
}


def icon(name: str) -> QIcon:
    """The named section's icon."""
    glyph, badge, fallback = GLYPHS.get(name, ("", "", name[:1]))

    if not _family():
        glyph = badge = ""

    return _shaded(glyph, badge, fallback, *_shades())


def label_icon(text: str) -> QIcon:
    """A short label, for a session's mouse or an experiment's name."""
    return _shaded("", "", text, *_shades())


@lru_cache(maxsize=None)
def _family() -> str:
    """The icon font's family, loaded once; empty if it cannot be."""
    if not FONT_FILE.exists():
        return ""

    families = QFontDatabase.applicationFontFamilies(
        QFontDatabase.addApplicationFont(str(FONT_FILE))
    )

    return families[0] if families else ""


def _shades() -> tuple[str, str, str]:
    """The resting, the hovered and the chosen color, all from the palette."""
    palette = QApplication.palette()
    bright = palette.color(QPalette.ColorRole.WindowText)
    behind = palette.color(QPalette.ColorRole.Window)
    chosen = palette.color(QPalette.ColorRole.HighlightedText)

    return _towards(bright, behind, DIMMED).name(), bright.name(), chosen.name()


def _towards(color: QColor, other: QColor, amount: float) -> QColor:
    """`color` moved `amount` of the way to `other`."""
    return QColor(
        round(color.red() + (other.red() - color.red()) * amount),
        round(color.green() + (other.green() - color.green()) * amount),
        round(color.blue() + (other.blue() - color.blue()) * amount),
    )


@lru_cache(maxsize=None)
def _shaded(
    glyph: str, badge: str, label: str, dim: str, bright: str, chosen: str
) -> QIcon:
    """
    The shades of one shape, as an icon Qt picks from by state.

    Keyed on the colors too, so a change of theme is a different entry rather
    than a stale one.
    """
    outlined = _draw(glyph, badge, label, filled=False)
    filled = _draw(glyph, badge, label, filled=True)
    made = QIcon()

    for shape, shade, states in (
        (outlined, dim, ((QIcon.Mode.Normal, QIcon.State.Off),)),
        (outlined, bright, ((QIcon.Mode.Active, QIcon.State.Off),)),
        (
            filled,
            chosen,
            (
                (QIcon.Mode.Normal, QIcon.State.On),
                (QIcon.Mode.Active, QIcon.State.On),
                (QIcon.Mode.Selected, QIcon.State.On),
            ),
        ),
    ):
        tinted = _tint(shape, shade)

        for mode, state in states:
            made.addPixmap(tinted, mode, state)

    return made


def _draw(glyph: str, badge: str, label: str, filled: bool) -> QPixmap:
    """
    The glyph if there is one, else the label, as a shape for `_tint`.

    Drawn in black: only where it covers is kept, and the color comes later.
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
        painter.setFont(_glyph_font(GLYPH_SCALE, filled))
        painter.drawText(_box(1), Qt.AlignmentFlag.AlignCenter, glyph)
    else:
        _draw_badged(painter, glyph, badge, filled)

    painter.end()

    return pixmap


def _draw_badged(painter: QPainter, glyph: str, badge: str, filled: bool) -> None:
    """The glyph in the top left, the badge in the bottom right corner."""
    painter.translate(WIDTH / 2, HEIGHT / 2)
    painter.scale(BADGED_TOGETHER, BADGED_TOGETHER)
    painter.translate(-WIDTH / 2, -HEIGHT / 2)

    painter.setFont(_glyph_font(GLYPH_SCALE * BADGED_SCALE, filled))
    painter.drawText(
        _box(BADGED_SCALE, at=(0.0, 0.0)), Qt.AlignmentFlag.AlignCenter, glyph
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
    painter.setFont(_glyph_font(BADGE_SCALE, True))
    painter.drawText(corner, Qt.AlignmentFlag.AlignCenter, badge)


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


def _glyph_font(scale: float, filled: bool) -> QFont:
    """The icon font at `scale` of the canvas, outlined or filled."""
    font = QFont(_family())
    font.setPixelSize(int(HEIGHT * INSET * scale))
    # Variable axes need Qt 6.7; before that every glyph is drawn outlined
    if hasattr(font, "setVariableAxis"):
        font.setVariableAxis(QFont.Tag("FILL"), 1.0 if filled else 0.0)
        font.setVariableAxis(QFont.Tag("wght"), GLYPH_WEIGHT)

    return font


def _label_font(text: str) -> QFont:
    """The font a label of this length is drawn in."""
    return _label_size(len(text))


@lru_cache(maxsize=None)
def _label_size(count: int) -> QFont:
    """
    A fixed-width font, as large as `count` characters fit the box at.

    Fixed-width and sized by the count rather than by the text, so every label
    of the same length comes out identically: measuring the text itself makes
    `e1` and `e2` different sizes, which reads as a mistake.
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
    """The same shape in one flat color, keeping only where it covers."""
    tinted = QPixmap(shape.size())
    tinted.fill(Qt.GlobalColor.transparent)

    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, shape)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QColor(ink))
    painter.end()

    return tinted
