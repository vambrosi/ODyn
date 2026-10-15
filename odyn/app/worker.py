"""
The one thread that talks to the database.

A `Database` holds a SQLite connection, which must be used from the thread
that made it, and every use of it takes the lock file and may wait on the
share. So the window never touches the database itself: it hands a function
to the worker, which runs it against its own `Database` and reports back
through signals on the job. The window stays responsive while a submit reads
a folder of TIFFs or processing chews through a sync file.

**USAGE**
```python
worker = DatabaseWorker(main_folder, project, user)
worker.start()

job = worker.run(lambda db: db.mice)
job.done.connect(show_table)
job.failed.connect(show_error)
```
"""

from __future__ import annotations

import gc
import queue
import traceback

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from ..database import Database
from ..utils import logger


class Job(QObject):
    """One unit of work handed to the worker, and how its result comes back."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        function: Callable[[Database], Any],
        label: str = "",
        one_lock: bool = False,
    ):
        super().__init__()
        self.function = function
        self.label = label or getattr(function, "__name__", "job")
        self.one_lock = one_lock


class DatabaseWorker(QThread):
    """
    Runs jobs against a `Database` it opens in its own thread, one at a time.

    Jobs run in the order they were given. `opened` fires once the database
    is ready (or `open_failed` if it could not be), so the window can wait
    for it before offering anything.
    """

    opened = Signal()
    open_failed = Signal(str)

    # Every result goes through this one signal, from the worker thread to the
    # thread this object lives in, carrying the job with it. Emitting from the
    # job itself would lose results: nothing but the queue holds a job, and a
    # Python-side connection dies with its sender before a queued delivery is
    # picked up by the other thread.
    _finished = Signal(object, object, object)  # job, result, error message

    def __init__(
        self,
        main_folder: str | Path,
        project: None | str,
        user: None | str = None,
        can_create: bool = False,
    ):
        super().__init__()
        self.main_folder = Path(main_folder)
        self.project = project
        self.user = user
        self.can_create = can_create
        self._jobs: queue.Queue[None | Job] = queue.Queue()
        self.db: None | Database = None
        self._finished.connect(self._deliver)

    def run(  # type: ignore[override]
        self, function=None, label: str = "", one_lock: bool = False
    ) -> Any:
        """
        Called with a function: queue a job and return it. With `one_lock`,
        the job holds the database lock from start to end (`Database.held`),
        for short jobs of several reads or writes.

        Called with nothing: the thread body (Qt calls it), which opens the
        database and works through the queue until `stop`.
        """
        if function is not None:
            job = Job(function, label, one_lock)
            self._jobs.put(job)

            return job

        try:
            self.db = Database(
                self.main_folder,
                project=self.project,
                can_create=self.can_create,
                user=self.user,
            )

        except Exception as error:
            logger.exception("Could not open the database")
            self.open_failed.emit(f"{type(error).__name__}: {error}")

            return

        self.opened.emit()

        while True:
            job = self._jobs.get()

            if job is None:
                break

            try:
                with self.db.held() if job.one_lock else nullcontext():
                    result = job.function(self.db)

            except Exception as error:
                logger.error(f"{job.label} failed:\n{traceback.format_exc()}")
                self._finished.emit(job, None, f"{type(error).__name__}: {error}")

            else:
                self._finished.emit(job, result, None)

        # SQLite closes a connection only in the thread that opened it, and the
        # database and its groups refer to each other, so a later collection
        # elsewhere would try; collect them here
        self.db = None
        gc.collect()

    def _deliver(self, job: Job, result: Any, error: None | str) -> None:
        """On the GUI thread: hand the job its result."""
        if error is None:
            job.done.emit(result)
        else:
            job.failed.emit(error)

    def stop(self) -> None:
        """Finish the queued jobs, then end the thread."""
        self._jobs.put(None)
        self.wait()

    def set_user(self, user: str) -> None:
        """Change who the recorded calls are attributed to, from now on."""
        self.user = user

        def rename(db: Database) -> None:
            # `user` is declared Final for readers; the picker is the one
            # place it legitimately changes
            object.__setattr__(db, "user", user)

        self.run(rename, "set_user")
