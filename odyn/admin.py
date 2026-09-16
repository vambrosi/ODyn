"""
Maintenance of a database: manual edits, copies, migrations and the lock.

Everything here goes through the same lock file as `Database`, so it is safe to
run while other people and cluster jobs are using the database.

**COMMAND LINE**
```
python -m odyn.admin <main_folder> [--project NAME] status
python -m odyn.admin <main_folder> sql "UPDATE ...; DELETE ..."
python -m odyn.admin <main_folder> script edits.sql
python -m odyn.admin <main_folder> backup NAME
python -m odyn.admin <main_folder> snapshot
python -m odyn.admin <main_folder> unlock [--yes]
python -m odyn.admin <main_folder> migrate
```

**PYTHON**
```python
from odyn.admin import Admin

admin = Admin(main_folder)
admin.sql(statements="UPDATE experiments SET exp_name = 'x' WHERE exp_id = 10")
```
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import uuid

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .database import Database, _has_database
from .locking import DatabaseLock, _describe
from .migrate import SCHEMA_VERSION, migrate, open_calls
from .utils import (
    DB_TIMEOUT_S,
    CallFrame,
    CallRecorder,
    check_name,
    database_path,
    logger,
    record_call,
)

BACKUPS_FOLDER = "backups"
SNAPSHOTS_FOLDER = "snapshots"


class Admin(CallRecorder):
    """
    Maintenance tools for a database.

    **USAGE**
    ```python
    admin = Admin(main_folder)                     # or project="name"

    admin.status()                                 # version, lock, copies
    admin.sql(statements="UPDATE ...")             # back up, then apply
    admin.script("edits.sql")                      # the same, from a file
    admin.backup("before-cleanup")                 # backups/before-cleanup.db
    admin.snapshot()                               # snapshots/odyn.db for other tools
    admin.unlock(yes=True)                         # remove a stuck lock
    admin.migrate()                                # upgrade the schema
    ```
    """

    def __init__(self, main_folder: str | Path, project: None | str = None):
        self.main_folder = Path(main_folder).resolve()
        self.project = project
        self.path = database_path(self.main_folder, project)

        if not _has_database(self.path):
            raise FileNotFoundError(
                f"No database at '{self.path}'. Check the main folder and the "
                "project name for typos."
            )

        # Calls recorded from here belong to the database.
        self.group_id = 0
        self._call_stack: list[CallFrame] = []
        self._db: None | Database = None

    @property
    def db(self) -> Database:
        # Opened only when needed, so `status`, `unlock` and `migrate` still
        # work on a database this version of odyn cannot open.
        if self._db is None:
            self._db = Database(self.main_folder, project=self.project)

        return self._db

    # ----------------------------------------------------------------------- #
    # Manual edits
    # ----------------------------------------------------------------------- #

    @record_call
    def sql(self, *, statements: str) -> int:
        """
        Back up the database, then run `statements` as one transaction.

        Either every statement is applied or none is. The backup goes to
        `backups/before-sql-<time>-<call id>.db`, and the call is recorded.
        Do not put `BEGIN` or `COMMIT` in `statements`.

        **RETURNS**
        How many rows changed.

        **EXAMPLE**
        ```python
        admin.sql(statements="UPDATE experiments SET exp_name = 'x' WHERE exp_id = 10")
        ```
        """
        if not statements.strip():
            raise ValueError("No SQL statements given.")

        db = self.db
        name = f"before-sql-{time.strftime('%Y%m%d-%H%M%S')}-{self.current_call_id}"
        target = self.path.parent / BACKUPS_FOLDER / f"{name}.db"

        # The backup and the change happen in one hold of the lock, so the
        # backup is exactly the database the change was applied to.
        with db._locked() as con:
            temp = _claim_and_copy(con, target)
            before = con.total_changes

            try:
                # One script, so a failing statement undoes the ones before it.
                # The extra ';' closes a last statement written without one.
                con.executescript(f"BEGIN IMMEDIATE;\n{statements}\n;\nCOMMIT;")

            except BaseException:
                if con.in_transaction:
                    con.rollback()

                temp.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise

            rows = con.total_changes - before

        _finish_copy(temp, target)
        db._reset_caches()

        backup = target.relative_to(self.main_folder).as_posix()
        self.set_output({"rows": rows, "backup": backup})
        logger.info(f"Changed {rows} rows. Backup: '{target}'.")

        return rows

    def script(self, path: str | Path) -> int:
        """Run a file of SQL statements, exactly like `sql`."""
        return self.sql(statements=Path(path).read_text())

    # ----------------------------------------------------------------------- #
    # Copies
    # ----------------------------------------------------------------------- #

    def backup(self, name: str) -> Path:
        """
        Copy the database to `backups/<name>.db` beside it, checked afterwards.

        Never overwrites: if the name is taken, delete the old copy first or
        choose another name.
        """
        name = name.removesuffix(".db")
        check_name("Backup", name)

        target = self.path.parent / BACKUPS_FOLDER / f"{name}.db"

        with self._raw_connection() as con:
            temp = _claim_and_copy(con, target)

        _finish_copy(temp, target)
        logger.info(f"Backed up the database to '{target}'.")

        return target

    def snapshot(self) -> Path:
        """
        Refresh `snapshots/<database file>` beside the database.

        Point DB Browser, R or any other program at the snapshot, never at the
        database itself: those programs do not take odyn's lock. The snapshot
        is replaced each time, so close programs that have it open first.
        """
        target = self.path.parent / SNAPSHOTS_FOLDER / self.path.name

        with self._raw_connection() as con:
            temp = _copy(con, target)

        try:
            _check_copy(temp)
            os.replace(temp, target)

        except PermissionError as error:
            temp.unlink(missing_ok=True)
            raise PermissionError(
                f"Could not replace '{target}'. Close any program that has it "
                "open and try again."
            ) from error

        except BaseException:
            temp.unlink(missing_ok=True)
            raise

        logger.info(f"Snapshot written to '{target}'.")
        return target

    @contextmanager
    def _raw_connection(self) -> Generator[sqlite3.Connection]:
        # A plain connection under the lock, so copies work whatever the
        # schema version. Never nested inside `self.db._locked()`: two lock
        # objects on one file in one process would wait for each other.
        with DatabaseLock(self.path):
            con = sqlite3.connect(self.path, timeout=DB_TIMEOUT_S)

            try:
                yield con
            finally:
                con.close()

    # ----------------------------------------------------------------------- #
    # Schema and lock
    # ----------------------------------------------------------------------- #

    def status(self) -> dict:
        """Log and return the schema version, who holds the lock, and the copies."""
        holder = DatabaseLock(self.path)._read_holder()

        with self._raw_connection() as con:
            version = con.execute("PRAGMA user_version;").fetchone()[0]
            running = open_calls(con)

        backups = sorted((self.path.parent / BACKUPS_FOLDER).glob("*.db"))
        snapshot = self.path.parent / SNAPSHOTS_FOLDER / self.path.name

        status = {
            "path": str(self.path),
            "size_mb": round(self.path.stat().st_size / 2**20, 1),
            "schema_version": version,
            "code_schema_version": SCHEMA_VERSION,
            "lock": _describe(holder) if holder else "free",
            "open_calls": running,
            "backups": [path.name for path in backups],
            "snapshot": (
                time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(snapshot.stat().st_mtime)
                )
                if snapshot.exists()
                else None
            ),
        }

        logger.info(f"Database:  {status['path']} ({status['size_mb']} MB)")
        logger.info(f"Schema:    v{version} (this code expects v{SCHEMA_VERSION})")
        logger.info(f"Lock:      {status['lock']}")

        if running is None:
            logger.info("Calls:     this schema does not record when calls end")
            
        else:
            logger.info(f"Calls:     {len(running)} started recently and not ended")

            for call_id, name, group, since in running:
                logger.info(
                    f"             {call_id}: {name} (group {group}) since {since}"
                )

        logger.info(f"Backups:   {len(backups)}")
        logger.info(f"Snapshot:  {status['snapshot'] or 'none'}")

        return status

    def unlock(self, *, yes: bool = False) -> bool:
        """
        Delete the lock file, for a lock left by a process that died.

        Stale locks are taken over by themselves after a while, so this is only
        for not waiting. Deleting a lock that is still in use can corrupt the
        database, so it shows who holds it and needs `yes=True` to go ahead.

        **RETURNS**
        Whether a lock file was deleted.
        """
        lock = DatabaseLock(self.path)
        holder = lock._read_holder()

        if holder is None:
            logger.info("The database is not locked.")
            return False

        age_min = (time.time() - holder["since"]) / 60
        logger.warning(f"The lock is {_describe(holder)} ({age_min:.0f} min ago).")

        if not yes:
            logger.info("Nothing deleted. Pass 'yes=True' (or '--yes') to delete it.")
            return False

        lock.path.unlink(missing_ok=True)
        logger.warning("Deleted the lock file.")
        return True

    def migrate(self, *, force: bool = False) -> None:
        """
        Upgrade the database schema to the version this code expects.

        Refuses while recorded calls have not ended (see `status`); `force=True`
        goes ahead, for calls whose processes are known to be dead.
        """
        if self._db is not None:
            raise RuntimeError(
                "This Admin has the database open. Use a new Admin to migrate."
            )

        migrate(self.main_folder, self.project, force=force)


# --------------------------------------------------------------------------- #
# Copies
# --------------------------------------------------------------------------- #


def _copy(con: sqlite3.Connection, target: Path) -> Path:
    """Copy the database behind `con` to a temporary file beside `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    destination = sqlite3.connect(temp)

    try:
        con.backup(destination)
    except BaseException:
        destination.close()
        temp.unlink(missing_ok=True)
        raise

    destination.close()
    return temp


def _claim_and_copy(con: sqlite3.Connection, target: Path) -> Path:
    """Reserve `target` so nothing overwrites it, then copy beside it."""
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise FileExistsError(
            f"'{target}' already exists. Delete it or choose another name."
        ) from None

    try:
        return _copy(con, target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _finish_copy(temp: Path, target: Path) -> None:
    """Check a copy made by `_claim_and_copy` and move it onto its reservation."""
    try:
        _check_copy(temp)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise


def _check_copy(path: Path) -> None:
    con = sqlite3.connect(path)

    try:
        result = con.execute("PRAGMA quick_check;").fetchone()[0]
    finally:
        con.close()

    if result != "ok":
        raise RuntimeError(f"The copy at '{path}' failed its check: {result}")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main(argv: None | list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odyn.admin",
        description="Maintenance of an odyn database.",
    )
    parser.add_argument("main_folder")
    parser.add_argument("--project", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="schema version, lock holder and copies")

    commands.add_parser("sql", help="back up, then run SQL").add_argument("statements")
    commands.add_parser("script", help="the same, from a file").add_argument("file")

    commands.add_parser("backup", help="copy to backups/NAME.db").add_argument("name")
    commands.add_parser("snapshot", help="refresh the copy other programs read")
    commands.add_parser("unlock", help="delete a stuck lock").add_argument(
        "--yes", action="store_true"
    )
    commands.add_parser("migrate", help="upgrade the schema").add_argument(
        "--force", action="store_true", help="even if recorded calls have not ended"
    )

    args = parser.parse_args(argv)
    admin = Admin(args.main_folder, project=args.project)

    if args.command == "status":
        admin.status()

    elif args.command == "sql":
        admin.sql(statements=args.statements)

    elif args.command == "script":
        admin.script(args.file)

    elif args.command == "backup":
        admin.backup(args.name)

    elif args.command == "snapshot":
        admin.snapshot()

    elif args.command == "unlock":
        if args.yes:
            admin.unlock(yes=True)

        # Shows who holds it before asking
        elif not admin.unlock() and DatabaseLock(admin.path).path.exists():
            if input("Delete the lock file? [y/N] ").strip().lower() == "y":
                admin.unlock(yes=True)

    elif args.command == "migrate":
        admin.migrate(force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
