"""
The main window: two bars of buttons around the page being shown, a panel of
text under it, a bottom bar, and the worker thread that owns the database.

- **Left bar**: the registries and browsing (Users | Mice, Odors, Panels,
  Tables | QC), the Recorder at its foot.
- **Right bar**: the day's sessions, one block each (the session, then its
  experiments), and at its foot Add session, Add experiment, Check session
  and Submit session, which act on the session selected last.
- **Bottom panel**: what Check and Submit found, and the `odyn` log.
- **Bottom bar**: what the hovered button does on the left, the user on the
  right.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..utils import logger
from . import style
from .context import Context
from .draft import Draft, drafts_folder
from .pane_qc import QcPane
from .pane_recorder import RecorderPane
from .pane_registry import MicePane, OdorsPane, PanelsPane, UsersPane
from .pane_session import Sessions
from .pane_tables import TablesPane
from .panel import BottomPanel, PanelLogHandler
from .symbols import icon, label_icon
from .worker import DatabaseWorker

# How long a message from the app (not a hovered button) stays on the bar
STATUS_MS = 8000


def line(vertical: bool) -> QFrame:
    """A 1 px division in the window's line color (see `style`)."""
    frame = QFrame()
    frame.setObjectName("line")

    if vertical:
        frame.setFixedWidth(1)
    else:
        frame.setFixedHeight(1)

    return frame


class MainWindow(QMainWindow):
    """
    The app.

    Built only once the worker has opened the database (`opened`), so every
    pane's first load finds it there. `worker.start()` is the caller's job,
    which is what lets a test build the window around a worker on a
    temporary database.

    What the middle shows is `view`: `"page"` for a left-bar page, or
    `"session"` / `"experiment"` for the selected draft (`context.draft`) and
    experiment (`context.experiment`), which the right bar highlights.
    """

    def __init__(
        self,
        worker: DatabaseWorker,
        drafts: None | Path = None,
        recorder_source=None,
    ):
        super().__init__()
        style.apply()
        self.setStyleSheet(style.window_stylesheet())
        self.worker = worker
        self.context = Context(worker, worker.main_folder, drafts or drafts_folder())
        self.context.window = self
        self.setWindowTitle(
            f"ODyn: {worker.main_folder} ({worker.project or 'default'})"
        )
        self.resize(1300, 850)
        self.view = "page"

        # The middle: the pages over the bottom panel
        self.pages = QStackedWidget()
        middle = QSplitter(Qt.Orientation.Vertical)
        middle.setObjectName("middle")
        middle.setHandleWidth(5)
        middle.setChildrenCollapsible(False)
        middle.addWidget(self.pages)
        self.panel = BottomPanel(middle)
        middle.addWidget(self.panel)
        middle.setStretchFactor(0, 1)
        middle.setStretchFactor(1, 0)

        # One row, edge to edge: bar | line | middle | line | bar. The bars
        # carry their own margins, so the lines run the full height.
        central = QWidget()
        self.row = QHBoxLayout(central)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(0)
        self.setCentralWidget(central)

        self.group = QActionGroup(self)
        self.group.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)

        self.users = UsersPane(self.context)
        self.mice = MicePane(self.context)
        self.odors = OdorsPane(self.context)
        self.panels = PanelsPane(self.context)
        self.tables = TablesPane(self.context)
        self.qc = QcPane(self.context)
        self.recorder = RecorderPane(self.context, source=recorder_source)
        self.sessions = Sessions(self.context)
        self.pages.addWidget(self.sessions.session_page)
        self.pages.addWidget(self.sessions.experiment_page)

        # Left: registries and browsing, the recorder at the foot
        self.left = self._bar("Pages")
        self.row.addWidget(line(vertical=True))
        self.row.addWidget(middle, 1)
        self.row.addWidget(line(vertical=True))
        self.page_actions: dict[QWidget, QAction] = {}

        for entry in (
            ("users", "Users", self.users),
            None,
            ("mice", "Mice", self.mice),
            ("odors", "Odors", self.odors),
            ("panels", "Panels", self.panels),
            ("tables", "Tables", self.tables),
            None,
            ("qc", "QC", self.qc),
            "foot",
            ("recorder", "Recorder", self.recorder),
        ):
            if entry is None:
                self.left.addSeparator()
            elif entry == "foot":
                self._foot(self.left)
            else:
                key, title, pane = entry
                self._page_action(key, title, pane)

        # Right: the day's sessions (rebuilt as they change), the day's
        # actions at the foot
        self.right = self._bar("Today")
        self.today_foot = self._foot(self.right)
        self.today: dict[tuple[int, None | str], QAction] = {}
        self.today_extras: list[QAction] = []

        for key, tip, action in (
            ("add-session", "Add session", self._add_session),
            ("add-experiment", "Add experiment", self.sessions.add_experiment),
            ("check-session", "Check session", self.sessions.check),
            ("submit-session", "Submit session", self.sessions.submit),
        ):
            self._command(self.right, icon(key), tip).triggered.connect(
                lambda _=False, a=action: a()
            )

        # Bottom: what is hovered or said on the left, the user on the right,
        # both the same distance from their edge. The left text is a label
        # (see `event`) because Qt's own status message sits closer to the
        # edge than anything else can; the bar leaves 2 px before its first
        # item already.
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.tip = QLabel()
        self.tip.setContentsMargins(style.STATUS_INSET_PX - 2, 4, 0, 6)
        bar.addWidget(self.tip, 1)
        self.user_label = QLabel()
        self.user_label.setContentsMargins(0, 4, style.STATUS_INSET_PX, 6)
        bar.addPermanentWidget(self.user_label)
        self._tip_timer = QTimer(self)
        self._tip_timer.setSingleShot(True)
        self._tip_timer.timeout.connect(lambda: self.tip.setText(""))

        # The odyn log goes to the panel's Log tab
        self.log_handler = PanelLogHandler()
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.signal.line.connect(
            lambda text: self.panel.append("Log", text)
        )
        logger.addHandler(self.log_handler)

        self.context.status.connect(self.say)
        self.context.user_changed.connect(self._show_user)
        self.context.selection_changed.connect(self._follow)
        self.context.draft_changed.connect(self._rebuild_today)
        self.sessions.listed.connect(self._rebuild_today)
        self.sessions.reported.connect(lambda text: self.panel.show_text("Check", text))
        self.sessions.experiment_added.connect(
            lambda name: self.open_experiment(self.context.draft, name)
        )

        # Picking a name is the first thing to do; the day's work comes next
        self.users.picked.connect(lambda _: self._after_login())

        self._show_user(self.context.user)
        self._rebuild_today()
        self.show_page(self.users)

    # ----------------------------------------------------------------- #
    # What the middle shows
    # ----------------------------------------------------------------- #

    def show_page(self, pane: QWidget) -> None:
        """Show a left-bar page."""
        self.view = "page"
        self.pages.setCurrentWidget(pane)
        self.page_actions[pane].setChecked(True)

    def open_session(self, draft: None | Draft) -> None:
        """Select `draft` and show its session page."""
        keep = self.context.experiment if draft is self.context.draft else None
        self.context.select(draft, keep)
        self.view = "session"
        self.pages.setCurrentWidget(self.sessions.session_page)
        self._highlight()

    def open_experiment(self, draft: None | Draft, name: str) -> None:
        """Select experiment `name` of `draft` and show its page."""
        self.context.select(draft, name)
        self.view = "experiment"
        self.pages.setCurrentWidget(self.sessions.experiment_page)
        self._highlight()

    def _add_session(self) -> None:
        self.open_session(self.sessions.new())

    def _after_login(self) -> None:
        # With no session yet, the empty page says where to add one
        self.open_session(self.sessions.current)

    def _follow(self) -> None:
        """
        The selection moved (the recorder picked another experiment, one was
        removed): an experiment page follows it, or falls back to its session.
        """
        if self.view == "experiment" and self.context.experiment is None:
            self.view = "session"
            self.pages.setCurrentWidget(self.sessions.session_page)

        self._highlight()

    def _highlight(self) -> None:
        """Check the button of what the middle shows."""
        if self.view == "page":
            return

        key = (
            id(self.context.draft),
            self.context.experiment if self.view == "experiment" else None,
        )
        wanted = self.today.get(key)

        if wanted is not None:
            wanted.setChecked(True)
        elif self.group.checkedAction() is not None:
            self.group.checkedAction().setChecked(False)

    # ----------------------------------------------------------------- #
    # The right bar's sessions
    # ----------------------------------------------------------------- #

    def _rebuild_today(self) -> None:
        """One block per draft: the session, then its experiments."""
        self._follow()  # an experiment removed: its page goes first

        for action in [*self.today.values(), *self.today_extras]:
            self.right.removeAction(action)
            self.group.removeAction(action)
            action.deleteLater()

        self.today.clear()
        self.today_extras.clear()

        for position, draft in enumerate(self.sessions.drafts):
            if position:
                separator = QAction(self)
                separator.setSeparator(True)
                self.right.insertAction(self.today_foot, separator)
                self.today_extras.append(separator)

            session = self._today_action(
                self.sessions.label(draft),
                draft.label,
                lambda _=False, d=draft: self.open_session(d),
            )
            self.today[(id(draft), None)] = session

            for entry in draft.experiments:
                name = str(entry["name"])
                self.today[(id(draft), name)] = self._today_action(
                    name,
                    f"{draft.label} / {name}",
                    lambda _=False, d=draft, n=name: self.open_experiment(d, n),
                )

        self._highlight()

    def _today_action(self, label: str, tip: str, opened) -> QAction:
        action = QAction(label_icon(label), "", self)
        action.setToolTip(tip)
        action.setStatusTip(tip)
        action.setCheckable(True)
        action.triggered.connect(opened)
        self.group.addAction(action)
        self.right.insertAction(self.today_foot, action)
        self._shape(self.right, action)

        return action

    # ----------------------------------------------------------------- #
    # Building the bars
    # ----------------------------------------------------------------- #

    def _bar(self, name: str) -> QToolBar:
        """A vertical bar of icons in its own margins, added to the row."""
        bar = QToolBar(name)
        bar.setOrientation(Qt.Orientation.Vertical)
        bar.setMovable(False)
        bar.setIconSize(QSize(style.BAR_ICON_PX, style.BAR_ICON_PX))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        holder = QWidget()
        holder_layout = QVBoxLayout(holder)
        holder_layout.setContentsMargins(*style.BAR_MARGINS)
        holder_layout.addWidget(bar)
        self.row.addWidget(holder)

        return bar

    def _foot(self, bar: QToolBar) -> QAction:
        """Push what follows to the bottom of the bar, under a separator."""
        filler = QWidget()
        filler.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        foot = bar.addWidget(filler)
        bar.addSeparator()

        return foot

    def _page_action(self, key: str, title: str, pane: QWidget) -> None:
        self.pages.addWidget(pane)
        action = QAction(icon(key), "", self)
        action.setToolTip(title)
        action.setStatusTip(title)
        action.setCheckable(True)
        action.triggered.connect(lambda: self.show_page(pane))
        self.group.addAction(action)
        self.left.addAction(action)
        self._shape(self.left, action)
        self.page_actions[pane] = action

    def _command(self, bar: QToolBar, symbol: QIcon, tip: str) -> QAction:
        """A plain (not checkable) bar button."""
        action = QAction(symbol, "", self)
        action.setToolTip(tip)
        action.setStatusTip(tip)
        bar.addAction(action)
        self._shape(bar, action)

        return action

    def _shape(self, bar: QToolBar, action: QAction) -> None:
        button = bar.widgetForAction(action)

        if isinstance(button, QToolButton):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # ----------------------------------------------------------------- #
    # The bottom bar
    # ----------------------------------------------------------------- #

    def say(self, text: str) -> None:
        """A message from the app, on the bottom bar for a few seconds."""
        self.tip.setText(text)
        self._tip_timer.start(STATUS_MS)

    def event(self, event: QEvent) -> bool:
        """What a hovered button does goes to the bottom bar's left label."""
        if event.type() == QEvent.Type.StatusTip:
            self._tip_timer.stop()
            self.tip.setText(event.tip())
            return True

        return super().event(event)

    def _show_user(self, user: None | str) -> None:
        self.user_label.setText(f"user: {user}" if user else "no user picked")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt's name)
        if self.recorder.recording is not None:
            answer = QMessageBox.question(
                self,
                "Recording",
                "A recording is running. Stop it and close?",
            )

            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.recorder.close_down()

        logger.removeHandler(self.log_handler)
        self.worker.stop()
        event.accept()


def open_window(
    main_folder: str | Path,
    project: None | str = None,
    user: None | str = None,
    can_create: bool = False,
    drafts: None | Path = None,
    recorder_source=None,
) -> tuple[QApplication, DatabaseWorker, "list[MainWindow]"]:
    """
    Start the worker and, once the database is open, show the window.

    Returns the application, the worker and a one-element list the window is
    put in when it appears (empty until then), so a caller can wait on it.
    """
    app = QApplication.instance() or QApplication([])
    worker = DatabaseWorker(main_folder, project, user=user, can_create=can_create)
    windows: list[MainWindow] = []

    def opened():
        window = MainWindow(worker, drafts=drafts, recorder_source=recorder_source)
        windows.append(window)
        window.show()

    def failed(message: str):
        QMessageBox.critical(None, "ODyn", f"Could not open the database:\n{message}")
        app.quit()

    worker.opened.connect(opened)
    worker.open_failed.connect(failed)
    worker.start()

    return app, worker, windows
