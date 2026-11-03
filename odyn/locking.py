"""
One lock per shared file (the database, or any other file several programs
write), shared by every process on every machine.

Workstations and cluster nodes do not see each other SQLite's locks, so either
side can corrupt the file by just reading while the other writes. (The reader
that sees a journal an no lock "recovers" the journal into the file.) Creating
a file that does not already exist is a single step, so both sides see holds.

The lock sits next to its file, e.g. `.odyn/odyn.db.lock`, and holds who
took it. Its age is the lock file time against the server clock
(`server_now`), so machines whose clocks disagree still agree on which
lock is stale. The database lock must be held for every use of the connection,
reads included.

Get a lock with `file_lock(path)`: one object per file in each process, since
two would wait for each other (each in-process lock re-enters only itself).
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
from .shared import read_json, server_now

# Holds are meant to be short, so and old lock more likely than not
# belongs to processess that died holding.
STALE_AFTER_S = 120

# Say who is holding the database once a wait gets noticeable
REPORT_AFTER_S = 5


class FileLock:
    """
    Re-entrant lock on one file, across threads and processes.

    ```python
    lock = file_lock(Path(".odyn/odyn.db"))
    with lock:
        ...  # every read and write of the database goes here
    ```

    Only the outermost `with` takes the file, so nested uses are free.
    """

    def __init__(self, path: Path):
        self.path = path.with_name(path.name + ".lock")

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
        offset = None  # server clock minus this one, asked once per wait

        while True:
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

            # Windows reports a file that is still being deleted as
            # PermissionError rather than FileExistsError.
            except (FileExistsError, PermissionError):
                if offset is None:
                    offset = self._clock_offset()

                age = self._file_age(offset)

                if age is not None and age > STALE_AFTER_S:
                    other = self._read_holder()

                    if other is not None:
                        self._take_over(other)
                        continue

                if not reported and time.monotonic() - start > REPORT_AFTER_S:
                    other = self._read_holder()
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

    def age(self) -> None | float:
        """How long ago the lock was taken, in seconds, or `None` if it is free."""
        return self._file_age(self._clock_offset())

    def _clock_offset(self) -> float:
        """The server clock minus this machine's (see `server_now`)."""
        try:
            return server_now(self.path.parent) - time.time()
        except OSError:  # a folder this user cannot write: this clock, then
            return 0.0

    def _file_age(self, offset: float) -> None | float:
        """The lock file age by the server clock, or `None` if it is gone."""
        try:
            return time.time() + offset - self.path.stat().st_mtime
        except OSError:
            return None

    def _read_holder(self) -> None | dict:
        """What the lock file says, or None if it is gone or half written."""
        try:
            holder = read_json(self.path)

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
            moved_holder = read_json(moved)

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


# One lock object per file in this process (see `file_lock`)
_LOCKS: dict[str, FileLock] = {}
_LOCKS_LOCK = threading.Lock()


def file_lock(path: str | Path) -> FileLock:
    """The lock of the file at `path`, the same object each time in a process."""
    key = os.path.abspath(path)

    with _LOCKS_LOCK:
        if key not in _LOCKS:
            _LOCKS[key] = FileLock(Path(key))

        return _LOCKS[key]


def _describe(holder: None | dict) -> str:
    if not holder or "user" not in holder:
        return "held by another process"

    since = time.strftime("%H:%M:%S", time.localtime(holder["since"]))
    return f"held by {holder['user']} on {holder['host']} since {since}"
