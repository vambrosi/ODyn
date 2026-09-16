"""
One lock per database file, shared by every process on every machine.

Workstations and cluster nodes do not see each other SQLite's locks, so either
side can corrupt the file by just reading while the other writes. (The reader
that sees a journal an no lock "recovers" the journal into the file.) Creating
a file that does not already exist is a single step, so both sides see holds.

The lock sits next to the database, e.g. `.odyn/odyn.db.lock`, and holds who
took it. It must be held for every use of the connection, reads included.
"""

from __future__ import annotations

import getpass
import json
import os
import random
import socket
import threading
import time
import uuid

from pathlib import Path

from .utils import logger

# Holds are meant to be short, so and old lock more likely than not
# belongs to processess that died holding.
STALE_AFTER_S = 120

# Say who is holding the database once a wait gets noticeable
REPORT_AFTER_S = 5


class DatabaseLock:
    """
    Re-entrant lock on one database file, across threads and processes.

    ```python
    lock = DatabaseLock(Path(".odyn/odyn.db"))
    with lock:
        ...  # every read and write of the database goes here
    ```

    Only the outermost `with` takes the file, so nested uses are free.
    """

    def __init__(self, db_path: Path):
        self.path = db_path.with_name(db_path.name + ".lock")

        # Threads in this process wait here; other processes wait on the file.
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._token: None | str = None

    def __enter__(self) -> bool:
        """Returns whether this is the outermost `with`."""
        self._thread_lock.acquire()

        if self._depth == 0:
            try:
                self._acquire_file()
            except BaseException:
                self._thread_lock.release()
                raise

        self._depth += 1
        return self._depth == 1

    def __exit__(self, *exc) -> None:
        try:
            self._depth -= 1

            if self._depth == 0:
                self._release_file()

        finally:
            self._thread_lock.release()

    # ----------------------------------------------------------------------- #

    def _acquire_file(self) -> None:
        self._token = uuid.uuid4().hex
        holder = json.dumps(
            {
                "token": self._token,
                "user": getpass.getuser(),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "since": time.time(),
            }
        )

        start = time.monotonic()
        reported = False

        while True:
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

            # Windows reports a file that is still being deleted as
            # PermissionError rather than FileExistsError.
            except (FileExistsError, PermissionError):
                other = self._read_holder()

                if other is not None and time.time() - other["since"] > STALE_AFTER_S:
                    self._take_over(other)
                    continue

                if not reported and time.monotonic() - start > REPORT_AFTER_S:
                    logger.info(f"Waiting for the database, {_describe(other)}...")
                    reported = True

                time.sleep(random.uniform(0.01, 0.05))
                continue

            try:
                os.write(handle, holder.encode())

            finally:
                os.close(handle)

            return

    def _release_file(self) -> None:
        # Only deletes this process file (in case the lock was taken over).
        # A holder without a token is a file this process could not read just
        # now (another process had it open), not evidence of a takeover.
        current = self._read_holder()

        if current is not None and current.get("token", self._token) != self._token:
            logger.error(
                f"The database lock was taken over while this process held it "
                f"({_describe(current)}). What was written may be inconsistent, "
                f"so run an integrity check."
            )
            return

        # On Windows a file cannot be deleted while another process has it open,
        # and waiters open it briefly to see who holds it. Those reads last
        # milliseconds, so retrying is enough.
        give_up = time.monotonic() + REPORT_AFTER_S

        while True:
            try:
                os.remove(self.path)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                if time.monotonic() > give_up:
                    raise
                time.sleep(0.01)

    def _read_holder(self) -> None | dict:
        """What the lock file says, or None if it is gone or half written."""
        try:
            holder = json.loads(self.path.read_text())

        except (OSError, ValueError):
            # Being created or deleted right now. Fall back to its age on disk,
            # so a file left empty by a crash still goes stale eventually.
            try:
                return {"since": self.path.stat().st_mtime}

            except OSError:
                return None

        return holder if isinstance(holder, dict) and "since" in holder else None

    def _take_over(self, stale: dict) -> None:
        # Renaming first rather than deleting: two processes can find the same
        # stale lock at once, and only one rename of it succeeds.
        moved = self.path.with_name(f"{self.path.name}.stale-{uuid.uuid4().hex}")

        try:
            os.rename(self.path, moved)

        except OSError:
            return  # someone else got there first

        # Between reading the stale holder and renaming, another process may
        # have taken it over and created a fresh lock. Put that one back.
        try:
            moved_holder = json.loads(moved.read_text())

        except (OSError, ValueError):
            moved_holder = {}

        if stale.get("token") is not None and moved_holder.get("token") != stale.get(
            "token"
        ):
            try:
                os.rename(moved, self.path)

            except OSError:
                pass

            return

        logger.warning(f"Took over a stale database lock ({_describe(stale)}).")

        try:
            os.remove(moved)

        except OSError:
            pass


def _describe(holder: None | dict) -> str:
    if not holder or "user" not in holder:
        return "held by another process"

    since = time.strftime("%H:%M:%S", time.localtime(holder["since"]))
    return f"held by {holder['user']} on {holder['host']} since {since}"
