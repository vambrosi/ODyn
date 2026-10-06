"""
The one stylesheet the app uses: flat, rounded controls with some room
around their text, and a highlight when the mouse is over a button.

Colors come from the palette (`palette(...)`), so the look follows the
system's light or dark theme instead of fighting it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QLabel

RADIUS = 6

# Text sizes, in points: the bottom panel (its tab buttons and its output) and
# the bottom bar's texts' distance from the window's sides
PANEL_FONT_PT = 13
STATUS_INSET_PX = 14

# The icons in the side bars, and the room around the buttons: the foot is
# larger so the last button does not sit on the bottom bar's line
BAR_ICON_PX = 28
BAR_MARGINS = (6, 8, 6, 14)

# How wide a registry page's form (Mice, Odors, Panels) opens, and the least
# it can be dragged to: wide enough for each page's widest row
FORM_PX = 520

# The gap between a form's rows, set rather than left to the platform's style
# so rows built from several widgets (a mouse's lines) are the same size
ROW_SPACING_PX = 8

# The session and experiment pages stop growing at this width and are
# centered in what is left
PAGE_MAX_PX = 1000

# What a dropdown needs besides its text: the padding on both sides, the
# drop-down arrow's width, the border, and a little air (see STYLESHEET)
COMBO_EXTRA_PX = 2 * 8 + 24 + 2 + 12

# Styling a control's frame makes Qt drop the native arrows, so they are
# drawn from these (a mid gray that reads on light and dark backgrounds)
ICONS = Path(__file__).parent / "icons"
DOWN = (ICONS / "arrow-down.svg").as_posix()
UP = (ICONS / "arrow-up.svg").as_posix()

STYLESHEET = f"""
QPushButton {{
    background: palette(button);
    color: palette(button-text);
    border: 1px solid palette(mid);
    border-radius: {RADIUS}px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background: palette(midlight);
    border-color: palette(highlight);
}}
QPushButton:pressed {{
    background: palette(mid);
}}
QPushButton:checked {{
    background: palette(highlight);
    color: palette(highlighted-text);
    border-color: palette(highlight);
}}
QPushButton:disabled {{
    color: palette(mid);
    border-color: palette(midlight);
}}

QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: palette(base);
    color: palette(text);
    border: 1px solid palette(mid);
    border-radius: {RADIUS}px;
    padding: 5px 8px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: palette(highlight);
}}
QLineEdit#shown {{
    background: transparent;
    border-color: transparent;
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: palette(mid);
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: url("{DOWN}");
    width: 10px;
    height: 6px;
}}
QComboBox QAbstractItemView {{
    border: 1px solid palette(mid);
    border-radius: {RADIUS}px;
    selection-background-color: palette(highlight);
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: padding;
    border: none;
    width: 18px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{UP}");
    width: 10px;
    height: 6px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{DOWN}");
    width: 10px;
    height: 6px;
}}

QListWidget, QTableView, QScrollArea, QGroupBox {{
    border: 1px solid palette(mid);
    border-radius: {RADIUS}px;
}}
QGroupBox {{
    margin-top: 12px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
"""


# Text colors that are not the palette's: what the app computed, what is
# wrong, what is worth a look. Two sets, since the dark theme needs paler ones
TEXT_COLORS = {
    "light": {"computed": "#555555", "error": "#b00020", "warning": "#8a6d00"},
    "dark": {"computed": "#b8b8b8", "error": "#ff8a80", "warning": "#ffd54f"},
}


def is_dark() -> bool:
    """Whether the application palette is a dark theme."""
    app = QApplication.instance()

    if app is None:
        return False

    return app.palette().window().color().lightness() < 128


def required_label(text: str, required: bool = True) -> QLabel:
    """A form label. (Required fields use the error color.)"""
    label = QLabel(text)

    if required:
        label.setStyleSheet(text_style("error"))
        label.setToolTip("Required")

    return label


def text_style(kind: str) -> str:
    """A `color:` stylesheet for `computed`, `error` or `warning` text."""
    return f"color: {TEXT_COLORS['dark' if is_dark() else 'light'][kind]};"


def apply(app: None | QApplication = None) -> None:
    """Put the stylesheet on the application (once is enough)."""
    app = app or QApplication.instance()

    if app is not None and app.styleSheet() != STYLESHEET:
        app.setStyleSheet(STYLESHEET)


def mix(color: QColor, other: QColor, amount: float) -> QColor:
    """`color` moved `amount` of the way to `other`."""
    return QColor(
        *(
            round(x + (y - x) * amount)
            for x, y in zip(color.getRgb()[:3], other.getRgb()[:3])
        )
    )


def window_colors() -> dict[str, str]:
    """
    The window's own colors, from the palette so light and dark both work.

    `line` is every division (beside the bars, above the bottom panel, above
    the bottom bar); `soft` is the background of the bottom panel's open tab;
    `dim` the text of the others.
    """
    palette = QApplication.palette()
    window = palette.color(QPalette.ColorRole.Window)
    text = palette.color(QPalette.ColorRole.WindowText)

    return {
        "line": window.darker(165 if is_dark() else 118).name(),
        "soft": mix(window, text, 0.12).name(),
        "dim": mix(text, window, 0.35).name(),
    }


def window_stylesheet() -> str:
    """
    What the main window adds to `STYLESHEET`: square frames and plain
    division lines everywhere; only buttons and input fields are rounded.
    """
    c = window_colors()

    return f"""
QScrollArea {{
    border: none;
}}
QGroupBox, QListWidget, QTableView {{
    border-radius: 0px;
}}
QFrame#line {{
    background: {c['line']};
    border: none;
}}
QStatusBar {{
    background: palette(window);
    border-top: 1px solid {c['line']};
}}
QStatusBar::item {{
    border: none;
}}
QSplitter#middle::handle:vertical {{
    /* one line at the top of a 5 px handle, so it is easy to grab; the
       rest is the bottom panel's color */
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {c['line']}, stop:0.2 {c['line']},
        stop:0.21 palette(base), stop:1 palette(base));
}}
QSplitter#tableAndForm::handle:horizontal {{
    /* one line in the middle of a 5 px handle, so it is easy to grab */
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 palette(window), stop:0.4 palette(window),
        stop:0.41 {c['line']}, stop:0.6 {c['line']},
        stop:0.61 palette(window), stop:1 palette(window));
}}
QToolBar {{
    background: palette(window);
    border: none;
    spacing: 4px;
}}
QToolBar::separator {{
    background: {c['line']};
    height: 1px;
    margin: 6px 4px;
}}
QToolBar QToolButton {{
    border: 1px solid transparent;
    border-radius: {RADIUS}px;
    padding: 4px;
}}
QToolBar QToolButton:hover {{
    background: palette(midlight);
    border-color: palette(highlight);
}}
QToolBar QToolButton:checked {{
    background: palette(highlight);
    border-color: palette(highlight);
}}
QLabel#tip {{
    border: 1px solid palette(highlight);
    border-radius: {RADIUS}px;
    padding: 2px 8px;
    margin: 6px 0px 6px {STATUS_INSET_PX - 2}px;
}}
QLabel#selection {{
    background: palette(highlight);
    color: palette(highlighted-text);
    border: 1px solid palette(highlight);
    border-radius: {RADIUS}px;
    padding: 2px 8px;
    margin: 6px 0px;
}}
QWidget#bottomPanel {{
    background: palette(base);
}}
QPushButton#panelTab {{
    font-size: {PANEL_FONT_PT}pt;
    background: transparent;
    border: none;
    border-radius: {RADIUS}px;
    padding: 3px 10px;
    color: {c['dim']};
}}
QPushButton#panelTab:hover {{
    color: palette(window-text);
}}
QPushButton#panelTab:checked {{
    background: {c['soft']};
    color: palette(window-text);
}}
QToolButton#panelToggle {{
    border: none;
    border-radius: {RADIUS}px;
    padding: 2px;
}}
QToolButton#panelToggle:hover {{
    background: {c['soft']};
}}
QPlainTextEdit#panelText {{
    background: palette(base);
    color: palette(window-text);
    border: none;
    padding: 0px 6px;
}}
"""
