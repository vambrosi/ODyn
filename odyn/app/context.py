"""
What every pane shares: the worker, who is using the app, and a way to say
the database changed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget
from shiboken6 import isValid

from .olfactometer import OlfactometerFolders
from .recordings import LOCAL_FOLDER
from .worker import DatabaseWorker, Job


class Context(QObject):
    """
    Shared by the panes.

    - `run(job, on_done, label)` hands work to the database thread and shows
      a message box if it fails;
    - `changed` fires after anything was written, so every pane reloads what
      it shows (`follow_changes`: the one on screen now, the others when next
      shown);
    - while the database thread has work, the cursor shows it is busy;
    - `user` is who is using the app right now, as picked in the Users tab;
    - `draft` and `part` are the session and the part of it (an experiment
      or its z-stacks) being worked on, the one state every tab shows
      and follows:
      `selection_changed` fires when either is picked, `draft_changed` when
      the picked draft's contents change (mouse, date, a part added
      or removed), so the Session and Recorder tabs never disagree;
    - `cache_changed` fires when tests were downloaded here or deleted
      (`cache_base` is where, beside the drafts).
    """

    changed = Signal()
    user_changed = Signal(str)
    status = Signal(str)
    selection_changed = Signal()
    draft_changed = Signal()
    cache_changed = Signal()

    def __init__(self, worker: DatabaseWorker, main_folder: Path, drafts: Path):
        super().__init__()
        self.worker = worker
        self.main_folder = Path(main_folder)
        self.drafts = Path(drafts)

        # Where the olfactometer keeps its files here (settings beside the drafts)
        self.olfactometer = OlfactometerFolders.read(self.drafts.parent)
        self._user = worker.user
        self.window: None | QWidget = None
        self.draft: Any = None  # a `Draft`, or None
        self.part: None | str = None

        # What the right bar has selected: "session", "part" or None
        self.focus: None | str = None
        self.last_report: None | tuple[str, str] = None

        # The local file a recording is being written into, if one is
        self.recording_now: None | str = None

        # Jobs given to the database thread and not back yet (the busy cursor)
        self.pending = 0

        # The draft object the window holds for a draft file (the window sets
        # this to look through its sessions): edits must go to that object,
        # since it is the one that saves
        self.live_draft: Callable[[Path], Any] = lambda path: (
            self.draft
            if self.draft is not None and Path(self.draft.path) == Path(path)
            else None
        )

    @property
    def recordings(self) -> Path:
        """Where sync recordings are written first: beside the drafts folder."""
        return self.drafts.parent / LOCAL_FOLDER

    @property
    def cache_base(self) -> Path:
        """Where downloaded tests go (`cache`): beside the drafts folder."""
        return self.drafts.parent

    @property
    def user(self) -> None | str:
        return self._user

    def select(self, draft: Any, part: None | str = None) -> None:
        """Make `draft` (and one of its parts) what every tab works on."""
        if part is not None and (draft is None or draft.part(part) is None):
            part = None

        if part is None and draft is not None and draft.parts:
            part = str(draft.parts[-1]["name"])

        if draft is self.draft and part == self.part:
            return

        self.draft, self.part = draft, part
        self.selection_changed.emit()

    @property
    def selected_part(self) -> None | str:
        """The part, when it (not only its session) is what is selected."""
        return self.part if self.focus == "part" else None

    def select_part(self, part: None | str) -> None:
        """Switch to another part of the same draft."""
        self.select(self.draft, part)

    def announce_draft(self) -> None:
        """The picked draft changed (mouse, date, parts): tabs redraw."""
        if self.draft is not None and self.part is not None:
            if self.draft.part(self.part) is None:
                self.part = None

        if self.part is None and self.draft is not None and self.draft.parts:
            self.part = str(self.draft.parts[-1]["name"])

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
        owner: None | QObject = None,
        one_lock: bool = False,
    ) -> Job:
        """
        Run `function(db)` on the database thread.

        `on_done` gets the result on the GUI thread. A failure is shown in a
        message box. `writes=True` announces a change to every pane afterwards.
        `owner` is the widget `on_done` fills in; if it was deleted in the
        meantime (a page rebuilt for a new selection), `on_done` is skipped.
        `one_lock` holds the database lock for the whole job, which saves
        round trips over the network for short jobs of several reads or
        writes; never for a job that also copies or reads files.
        """
        job = self.worker.run(function, label, one_lock)
        self._busy(+1)

        def finished(result):
            try:
                if on_done is not None and (owner is None or isValid(owner)):
                    on_done(result)

                # Before the cursor goes back: the reloads it starts keep it busy
                if writes:
                    self.changed.emit()

            finally:
                self._busy(-1)

        def failed(message):
            self._busy(-1)
            self.report(f"{job.label} failed", message)

        job.done.connect(finished)
        job.failed.connect(failed)

        return job

    def _busy(self, step: int) -> None:
        """Count the jobs out; the cursor says busy while there are any."""
        before, self.pending = self.pending, max(0, self.pending + step)

        if before == 0 and self.pending > 0:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        elif before > 0 and self.pending == 0:
            QApplication.restoreOverrideCursor()

    def follow_changes(self, widget: QWidget, reload: Callable[[], None]) -> None:
        """
        Call `reload` after a change, at once if `widget` is on screen, else
        the next time it is shown: each reload is a trip to the database (over
        the network), and the page on screen should not wait behind the
        others.
        """
        stale = _ReloadOnShow(widget, reload)
        widget.installEventFilter(stale)
        self.changed.connect(stale.changed)

    def report(self, title: str, message: str) -> None:
        """
        Say what went wrong: the status bar, and a box that does not block.

        The box shows `message` as written; the status bar gets it on one line.
        """
        self.status.emit(f"{title}: {' '.join(message.split())}")
        box = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=self.window)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.open()
        self.last_report = (title, message)

    def confirm(self, title: str, message: str, action: str) -> bool:
        """
        Ask before doing something that cannot be taken back: `action` names
        the button that goes ahead, the other cancels (and is the default).
        """
        box = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=self.window)
        go = box.addButton(action, QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.exec()

        return box.clickedButton() is go


class _ReloadOnShow(QObject):
    """Reloads a page after a change now, or when it is shown if hidden."""

    def __init__(self, widget: QWidget, reload: Callable[[], None]):
        super().__init__(widget)
        self.widget = widget
        self.reload = reload
        self.stale = False

    def changed(self) -> None:
        if self.widget.isVisible():
            self.reload()
        else:
            self.stale = True

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QEvent.Type.Show and self.stale:
            self.stale = False
            self.reload()

        return False
