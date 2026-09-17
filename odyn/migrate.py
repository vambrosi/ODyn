"""
Database schema migrations. Run them through the admin tool:
    `python -m odyn.admin <main_folder> [--project <name>] migrate`

To inspect older migrations run:
    `git log -p odyn/latest.sql`

A migration:
- Backs up DB to `.odyn/backups/<time>-<project or main>-snapshot-v<OLD>.db`
- Applies `latest.sql` migration to DB
- Updates `user_version`
"""

from __future__ import annotations

import sqlite3

from pathlib import Path

from .locking import DatabaseLock
from .utils import DB_TIMEOUT_S, backup_path, database_path, logger

# When adding a new migration you should:
# - Overwrite latest.sql with the latest migration;
# - Overwrite create.sql with compatible DB schema;
# - Bump the SCHEMA_VERSION to match;
# - Run test_migration.py, then `python -m odyn.tools diagram`.

SCHEMA_VERSION = 4

# Calls running for longer than this limit are probably dead
RUNNING_FOR_AT_MOST = "-2 days"
LATEST_MIGRATION = Path(__file__).parent / "latest.sql"


def migrate(
    main_folder: str | Path, project: None | str = None, force: bool = False
) -> None:
    """
    Migrate DB from v(SCHEMA_VERSION-1) up to vSCHEMA_VERSION.

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

            if version != SCHEMA_VERSION - 1:
                raise RuntimeError(f"Expected v{SCHEMA_VERSION - 1} but got v{version}")

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
