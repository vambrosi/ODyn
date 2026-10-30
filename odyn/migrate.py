"""
Database schema migrations. Run them through the admin tool:
    `python -m odyn.admin <main_folder> [--project <name>] migrate`

To inspect older migrations run:
    `git log -p odyn/latest.sql`

A migration:
- Backs up DB to `.odyn/backups/<time>-<project or main>-snapshot-v<OLD>.db`
- Runs `BEFORE_SQL`, when the migration needs Python too (files to write)
- Applies `latest.sql` migration to DB
- Updates `user_version`
- Runs VACUUM, when `VACUUM_AFTER` (to give back the room of dropped data)
"""

from __future__ import annotations

import os
import sqlite3

from collections.abc import Callable
from pathlib import Path

from .locking import DatabaseLock
from .utils import DB_TIMEOUT_S, backup_path, call_log_path, database_path, logger

# When adding a new migration you should:
# - Overwrite latest.sql with the latest migration;
# - Overwrite create.sql with compatible DB schema;
# - Bump the SCHEMA_VERSION to match, and set MIGRATES_FROM;
# - Set BEFORE_SQL and VACUUM_AFTER (None and False unless needed);
# - Copy the previous create.sql to tests/previous_schema.sql;
# - Run test_migration.py, then `python -m odyn.tools diagram`.

SCHEMA_VERSION = 7

# Which version `latest.sql` upgrades from, or None when a version is a rebuild
# rather than a migration: a new database is started from `create.sql` and the
# old one stays readable with the tagged older code. v5 and v6 are rebuilds.
MIGRATES_FROM: None | int = 6

# What the migration does outside SQL, before `latest.sql`: given the open
# connection, the main folder and the project. v7: the logs into their files.
BEFORE_SQL: None | Callable[[sqlite3.Connection, Path, None | str], None] = (
    None  # set below
)

# VACUUM after the migration: rewrites the whole file, to give back the room
# of what was dropped. Needs room for a copy of the database.
VACUUM_AFTER = True

# Calls running for longer than this limit are probably dead
RUNNING_FOR_AT_MOST = "-2 days"
LATEST_MIGRATION = Path(__file__).parent / "latest.sql"


def migrate(
    main_folder: str | Path, project: None | str = None, force: bool = False
) -> None:
    """
    Migrate DB from vMIGRATES_FROM up to vSCHEMA_VERSION.

    Refuses while recorded calls are still running, since they may write
    through the old schema when they finish. `force` goes ahead anyway, for
    calls known to be dead.
    """

    db_path = database_path(main_folder, project)
    if not db_path.exists():
        raise FileNotFoundError(f"No database at '{db_path}'.")

    # Held for the whole migration, including the checks before and after:
    # everything else that opens the database waits until it is done.
    with DatabaseLock(db_path):
        # Can wait longer than usual and manages transactions explicitly
        con = sqlite3.connect(db_path, timeout=DB_TIMEOUT_S * 4)
        con.isolation_level = None

        # Connection `con` "context manager"
        try:
            version = con.execute("PRAGMA user_version;").fetchone()[0]

            if version == SCHEMA_VERSION:
                logger.info(f"Database already at v{SCHEMA_VERSION}.")
                return

            if MIGRATES_FROM is None:
                raise RuntimeError(
                    f"Schema v{SCHEMA_VERSION} is a rebuild, not a migration, so "
                    f"there is no upgrade path from v{version}. Start a new "
                    "database (a project) and keep this one for the older code."
                )

            if version != MIGRATES_FROM:
                raise RuntimeError(f"Expected v{MIGRATES_FROM} but got v{version}")

            check_no_open_calls(con, force)

            check_integrity(con)

            # Backs up DB using SQLite online backup API
            backup = backup_path(main_folder, project, f"snapshot-v{version}")
            backup.parent.mkdir(parents=True, exist_ok=True)

            # Only another migration of this database, in the same second,
            # could have written it (and it would have to wait for the lock).
            if backup.exists():
                raise FileExistsError(f"'{backup}' already exists. Try again.")

            dest = sqlite3.connect(backup)

            try:
                con.backup(dest)
            finally:
                dest.close()

            logger.info(f"Backed up database to '{backup}'.")

            if BEFORE_SQL is not None:
                BEFORE_SQL(con, Path(main_folder), project)

            # Dropping tables can violate FOREIGN KEY contraints
            migration_script = LATEST_MIGRATION.read_text()
            con.execute("PRAGMA foreign_keys = OFF;")

            # Executes migration and version bump as a unit
            try:
                con.executescript(f"""
                    BEGIN EXCLUSIVE;
                    {migration_script}
                    PRAGMA user_version = {SCHEMA_VERSION};
                    COMMIT;
                """)

            except Exception:
                # BEGIN EXCLUSIVE may fail before a transaction exists (e.g. the DB
                # is locked); don't let a failed ROLLBACK mask the real error.
                try:
                    con.execute("ROLLBACK;")
                except sqlite3.OperationalError:
                    pass
                raise

            finally:
                con.execute("PRAGMA foreign_keys = ON;")

            logger.info("Running migration checks...")

            check_integrity(con)
            check_foreign_keys(con)

            logger.info(f"Migrated database to v{SCHEMA_VERSION}.")

            if VACUUM_AFTER:
                before = db_path.stat().st_size
                con.execute("VACUUM;")
                after = db_path.stat().st_size
                logger.info(
                    f"Vacuumed: {before / 2**20:.1f} MB -> {after / 2**20:.1f} MB."
                )

        finally:
            con.close()


def open_calls(con: sqlite3.Connection) -> None | list[tuple]:
    """
    Recorded calls that started recently and have not ended.

    Returns `(method_call_id, method_name, group_id, called_at)` rows, or `None`
    for a database too old to record when calls end.
    """
    columns = {row[1] for row in con.execute("PRAGMA table_info(method_calls);")}

    if "ended_at" not in columns:
        return None

    return con.execute(
        """
        SELECT method_call_id, method_name, group_id, called_at
            FROM method_calls
            WHERE ended_at IS NULL
              AND called_at >= datetime('now', 'localtime', ?)
            ORDER BY method_call_id;
        """,
        [RUNNING_FOR_AT_MOST],
    ).fetchall()


def check_no_open_calls(con: sqlite3.Connection, force: bool) -> None:
    running = open_calls(con)

    if running is None:
        logger.warning("This database does not record when calls end.")
        return

    if not running:
        return

    listed = "\n".join(
        f"  call {call_id}: {name} (group {group}) since {since}"
        for call_id, name, group, since in running
    )

    if not force:
        raise RuntimeError(
            f"{len(running)} recorded calls have not ended:\n{listed}\n"
            "Wait for them, or use force=True (or --force)."
        )

    logger.warning(f"Migrating despite {len(running)} calls that have not ended:")
    logger.warning(listed)


def check_integrity(con: sqlite3.Connection) -> None:
    result = con.execute("PRAGMA integrity_check;").fetchone()[0]

    if result != "ok":
        raise RuntimeError(f"Integrity check failed: {result}")


def check_foreign_keys(con: sqlite3.Connection) -> None:
    violations = con.execute("PRAGMA foreign_key_check;").fetchall()

    if violations:
        raise RuntimeError(f"Foreign key violations: {violations}")


def export_call_logs(
    con: sqlite3.Connection, main_folder: Path, project: None | str
) -> None:
    """
    v6 -> v7: each call's `call_log` into its file (see `call_log_path`), read
    back to check. A file already there (a migration tried before) must hold
    the same text, or this stops before anything is dropped.
    """
    rows = con.execute("""
        SELECT method_call_id, call_log
            FROM method_calls
            WHERE call_log != ''
            ORDER BY method_call_id;
    """)
    written = there = 0

    for call_id, text in rows:
        path = call_log_path(main_folder, project, call_id)

        if path.exists():
            if _read(path) != text:
                raise RuntimeError(
                    f"'{path}' is there but holds another log than call "
                    f"{call_id}'s. Move it away and migrate again."
                )

            there += 1
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.tmp")

        # newline="": the same bytes from every OS (the share is read by all)
        with open(temp, "w", encoding="utf-8", newline="") as file:
            file.write(text)

        os.replace(temp, path)

        if _read(path) != text:
            raise RuntimeError(f"'{path}' does not read back as call {call_id}'s log.")

        written += 1

    logger.info(f"Wrote {written} call logs to files ({there} already there).")


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as file:
        return file.read()


BEFORE_SQL = export_call_logs
