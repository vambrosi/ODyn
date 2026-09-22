"""
What every pane shares: the worker, who is using the app, and a way to say
the database changed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QMessageBox, QWidget

from .worker import DatabaseWorker, Job


class Context(QObject):
    """
    Shared by the panes.

    - `run(job, on_done, label)` hands work to the database thread and shows
      a message box if it fails;
    - `changed` fires after anything was written, so every pane reloads what
      it shows;
    - `user` is who is using the app right now, as picked in the Users tab;
    - `draft` and `experiment` are the session and experiment folder being
      worked on, the one state every tab shows and follows:
      `selection_changed` fires when either is picked, `draft_changed` when
      the picked draft's contents change (mouse, date, an experiment added
      or removed), so the Session and Recorder tabs never disagree.
    """

    changed = Signal()
    user_changed = Signal(str)
    status = Signal(str)
    selection_changed = Signal()
    draft_changed = Signal()

    def __init__(self, worker: DatabaseWorker, main_folder: Path, drafts: Path):
        super().__init__()
        self.worker = worker
        self.main_folder = Path(main_folder)
        self.drafts = Path(drafts)
        self._user = worker.user
        self.window: None | QWidget = None
        self.draft: Any = None  # a `Draft`, or None
        self.experiment: None | str = None
        self.last_report: None | tuple[str, str] = None

    @property
    def user(self) -> None | str:
        return self._user

    def select(self, draft: Any, experiment: None | str = None) -> None:
        """Make `draft` (and one of its experiments) what every tab works on."""
        if experiment is not None and (
            draft is None or draft.experiment(experiment) is None
        ):
            experiment = None

        if experiment is None and draft is not None and draft.experiments:
            experiment = str(draft.experiments[-1]["name"])

        if draft is self.draft and experiment == self.experiment:
            return

        self.draft, self.experiment = draft, experiment
        self.selection_changed.emit()

    def select_experiment(self, experiment: None | str) -> None:
        """Switch to another experiment of the same draft."""
        self.select(self.draft, experiment)

    def announce_draft(self) -> None:
        """The picked draft changed (mouse, date, experiments): tabs redraw."""
        if self.draft is not None and self.experiment is not None:
            if self.draft.experiment(self.experiment) is None:
                self.experiment = None

        if (
            self.experiment is None
            and self.draft is not None
            and self.draft.experiments
        ):
            self.experiment = str(self.draft.experiments[-1]["name"])

        self.draft_changed.emit()

    def set_user(self, user: str) -> None:
        self._user = user
        self.worker.set_user(user)
        self.user_changed.emit(user)

    def run(
        self,
        function: Callable[..., Any],
        on_done: None | Callable[[Any], None] = None,
        label: str = "",
        *,
        writes: bool = False,
    ) -> Job:
        """
        Run `function(db)` on the database thread.

        `on_done` gets the result on the GUI thread. A failure is shown in a
        message box. `writes=True` announces a change to every pane afterwards.
        """
        job = self.worker.run(function, label)

        def finished(result):
            if on_done is not None:
                on_done(result)

            if writes:
                self.changed.emit()

        job.done.connect(finished)
        job.failed.connect(lambda message: self.report(f"{job.label} failed", message))

        return job

    def report(self, title: str, message: str) -> None:
        """Say what went wrong: the status bar, and a box that does not block."""
        self.status.emit(f"{title}: {message}")
        box = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=self.window)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.open()
        self.last_report = (title, message)
