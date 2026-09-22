"""
The one stylesheet the app uses: flat, rounded controls with some room
around their text, and a highlight when the mouse is over a button.

Colors come from the palette (`palette(...)`), so the look follows the
system's light or dark theme instead of fighting it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

RADIUS = 6

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


def text_style(kind: str) -> str:
    """A `color:` stylesheet for `computed`, `error` or `warning` text."""
    return f"color: {TEXT_COLORS['dark' if is_dark() else 'light'][kind]};"


def apply(app: None | QApplication = None) -> None:
    """Put the stylesheet on the application (once is enough)."""
    app = app or QApplication.instance()

    if app is not None and app.styleSheet() != STYLESHEET:
        app.setStyleSheet(STYLESHEET)
