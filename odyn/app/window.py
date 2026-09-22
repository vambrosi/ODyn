"""
The main window: one tab per job, a status bar, and the worker thread that
owns the database.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from .context import Context
from .draft import drafts_folder
from .pane_qc import QcPane
from .pane_recorder import RecorderPane
from .pane_registry import MicePane, OdorsPane, PanelsPane, UsersPane
from .pane_session import SessionPane
from .pane_tables import TablesPane
from . import style
from .worker import DatabaseWorker


class MainWindow(QMainWindow):
    """
    The app.

    Built only once the worker has opened the database (`opened`), so every
    pane's first load finds it there. `worker.start()` is the caller's job,
    which is what lets a test build the window around a worker on a
    temporary database.
    """

    def __init__(
        self,
        worker: DatabaseWorker,
        drafts: None | Path = None,
        recorder_source=None,
    ):
        super().__init__()
        style.apply()
        self.worker = worker
        self.context = Context(worker, worker.main_folder, drafts or drafts_folder())
        self.context.window = self
        self.setWindowTitle(
            f"ODyn: {worker.main_folder} ({worker.project or 'default'})"
        )
        self.resize(1300, 850)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.session = SessionPane(self.context)
        self.recorder = RecorderPane(self.context, source=recorder_source)
        self.odors = OdorsPane(self.context)
        self.panels = PanelsPane(self.context)
        self.mice = MicePane(self.context)
        self.users = UsersPane(self.context)
        self.tables = TablesPane(self.context)
        self.qc = QcPane(self.context)

        for pane, title in (
            (self.users, "Users"),
            (self.session, "Session"),
            (self.recorder, "Recorder"),
            (self.odors, "Odors"),
            (self.panels, "Panels"),
            (self.mice, "Mice"),
            (self.tables, "Tables"),
            (self.qc, "QC"),
        ):
            self.tabs.addTab(pane, title)

        # Picking a name is the first thing to do; the day's work comes next
        self.users.picked.connect(lambda _: self.tabs.setCurrentWidget(self.session))

        self.user_label = QLabel()
        self.statusBar().addPermanentWidget(self.user_label)
        self.context.status.connect(
            lambda text: self.statusBar().showMessage(text, 8000)
        )
        self.context.user_changed.connect(self._show_user)
        self._show_user(self.context.user)

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
