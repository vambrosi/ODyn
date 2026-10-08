"""
The panel under the pages: tabs of text, like an editor's terminal panel.

**Check** holds what Check and Submit found; **Log** is the `odyn` log, so
what the database does (on its own thread) can be followed as it happens.
Collapsed, only the tab strip shows.
"""

from __future__ import annotations

import html
import logging
import re

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .style import PANEL_FONT_PT, text_style
from .symbols import icon

# Qt's "no maximum" for a widget's height
UNLIMITED_PX = 16777215


# Terminal color codes (as the `odyn` log writes them for a console), and the
# text colors that stand in for them in the Log tab
ANSI = re.compile(r"\033\[([0-9;]*)m")
ANSI_KINDS = {
    "31": "error",
    "32": "ok",
    "33": "warning",
    "34": "info",
    "37": "computed",
}

# A level name in the Log tab is colored as the console colors it
LEVEL_CODES = {
    logging.DEBUG: "\033[0;37m",
    logging.INFO: "\033[1;34m",
    logging.WARNING: "\033[1;33m",
    logging.ERROR: "\033[1;31m",
    logging.CRITICAL: "\033[1;31m",
}


def log_html(text: str) -> str:
    """
    `text` as HTML for the Log tab: its terminal color codes become colored
    (and bold) text, and its spaces and line breaks are kept.
    """
    out = ['<span style="white-space: pre-wrap;">']
    end = 0
    colored = False

    for match in ANSI.finditer(text):
        out.append(html.escape(text[end : match.start()], quote=False))
        end = match.end()

        if colored:
            out.append("</span>")
            colored = False

        codes = match.group(1).split(";")
        kind = next((ANSI_KINDS[c] for c in codes if c in ANSI_KINDS), None)

        if kind is not None:
            bold = " font-weight: bold;" if "1" in codes else ""
            out.append(f'<span style="{text_style(kind)}{bold}">')
            colored = True

    out.append(html.escape(text[end:], quote=False))

    if colored:
        out.append("</span>")

    out.append("</span>")

    return "".join(out).replace("\n", "<br>")


class _PanelFormatter(logging.Formatter):
    """`time LEVEL message`, the level in its console color."""

    def formatMessage(
        self, record: logging.LogRecord
    ) -> str:  # noqa: N802 (logging's name)
        code = LEVEL_CODES.get(record.levelno, "")
        reset = "\033[0m" if code else ""

        return f"{record.asctime} {code}{record.levelname}{reset} {record.message}"


class _LogSignal(QObject):
    line = Signal(str)


class PanelLogHandler(logging.Handler):
    """
    Log records as lines for the panel.

    Through a signal, because the database worker logs from its own thread
    and only the GUI thread may touch a widget: the signal queues the line.
    """

    def __init__(self):
        super().__init__()
        self.signal = _LogSignal()
        self.setFormatter(_PanelFormatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.signal.line.emit(self.format(record))
        except RuntimeError:  # the panel is gone (the window closed)
            pass


class BottomPanel(QWidget):
    """
    Tabs of text on the input fields' background: the tabs are flat buttons.

    Clicking a tab or the arrow opens the panel; clicking the open tab again
    or the arrow closes it to its tab strip. It lives in `splitter` under the
    pages, which take whatever height it leaves.

    **USAGE**
    ```python
    panel.show_text("Check", "RESULT  ready to submit")
    ```
    """

    NAMES = ("Check", "Log")
    OPEN_PX = 220

    def __init__(self, splitter: QSplitter, parent=None):
        super().__init__(parent)
        self.setObjectName("bottomPanel")
        # A plain QWidget paints a stylesheet background only with this
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.splitter = splitter
        self.open_px = self.OPEN_PX
        self.current = 0

        self.header = QWidget()
        header_row = QHBoxLayout(self.header)
        header_row.setContentsMargins(6, 6, 6, 4)
        header_row.setSpacing(2)
        self.buttons = QButtonGroup(self)
        self.body = QStackedWidget()
        self.texts: dict[str, QPlainTextEdit] = {}

        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(PANEL_FONT_PT)

        for index, name in enumerate(self.NAMES):
            tab = QPushButton(name)
            tab.setObjectName("panelTab")
            tab.setCheckable(True)
            tab.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            tab.clicked.connect(lambda _=False, i=index: self._tab_clicked(i))
            self.buttons.addButton(tab, index)
            header_row.addWidget(tab)

            text = QPlainTextEdit()
            text.setObjectName("panelText")
            text.setReadOnly(True)
            text.setFont(mono)
            self.texts[name] = text
            self.body.addWidget(text)

        self.buttons.button(0).setChecked(True)
        header_row.addStretch(1)
        self.toggle = QToolButton()
        self.toggle.setObjectName("panelToggle")
        self.toggle.setIconSize(QSize(20, 20))
        self.toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggle.clicked.connect(lambda: self.set_open(not self.is_open()))
        header_row.addWidget(self.toggle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.set_open(False)

    def is_open(self) -> bool:
        return not self.body.isHidden()

    def show_text(self, name: str, text: str) -> None:
        """
        Replace tab `name`'s text, and open the panel on it. Terminal color
        codes in `text` show as colors (see `log_html`).
        """
        self.texts[name].clear()
        self.texts[name].appendHtml(log_html(text))
        self._select(self.NAMES.index(name))
        self.set_open(True)

    def append(self, name: str, line: str) -> None:
        """Add a line to tab `name`, leaving the panel as it is."""
        self.texts[name].appendPlainText(line)

    def append_log(self, line: str) -> None:
        """
        Add a log line to the Log tab, its terminal color codes shown as
        colors (see `log_html`).
        """
        self.texts["Log"].appendHtml(log_html(line))

    def set_open(self, opened: bool) -> None:
        # Closing remembers the height, so opening again brings it back
        if not opened and self.is_open() and self.isVisible():
            self.open_px = max(self.height(), 80)

        self.body.setHidden(not opened)
        self.toggle.setIcon(icon("panel-close" if opened else "panel-open"))
        self.toggle.setToolTip("Close panel" if opened else "Open panel")
        header_px = self.header.sizeHint().height()
        self.setMaximumHeight(UNLIMITED_PX if opened else header_px)

        own = self.open_px if opened else header_px
        total = sum(self.splitter.sizes())

        if total:
            self.splitter.setSizes([max(total - own, 0), own])

    def _select(self, index: int) -> None:
        self.current = index
        self.buttons.button(index).setChecked(True)
        self.body.setCurrentIndex(index)

    def _tab_clicked(self, index: int) -> None:
        if self.is_open() and index == self.current:
            self.set_open(False)
        else:
            self._select(index)
            self.set_open(True)
