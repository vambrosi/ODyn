"""
Database schema migrations. Run them through the admin tool:
    `python -m odyn.admin <main_folder> [--project <name>] migrate`

To inspect older migrations run:
    `git log -p odyn/latest.sql`

A migration:
- Backs up DB to `backups/snapshot_v<OLD>.db` beside it
- Applies `latest.sql` migration to DB
- Updates `user_version`
"""

from __future__ import annotations

import sqlite3

from pathlib import Path

from .locking import DatabaseLock
from .utils import DB_TIMEOUT_S, database_path, logger

# When adding a new migration you should:
# - Overwrite latest.sql with the latest migration;
# - Overwrite create.sql with compatible DB schema;
# - Bump the SCHEMA_VERSION to match;
# - Run test_migration.py, then `python -m odyn.tools diagram`.

SCHEMA_VERSION = 2
LATEST_MIGRATION = Path(__file__).parent / "latest.sql"


def migrate(main_folder: str | Path, project: None | str = None) -> None:
    """Migrate DB from v(SCHEMA_VERSION-1) up to vSCHEMA_VERSION."""

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

            check_integrity(con)

            # Backs up DB using SQLite online backup API
            backups = db_path.parent / "backups"
            backups.mkdir(exist_ok=True)

            backup_path = backups / f"snapshot_v{version}.db"
            dest = sqlite3.connect(backup_path)

            try:
                con.backup(dest)
            finally:
                dest.close()

            logger.info(f"Backed up database to '{backup_path}'.")

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


def check_integrity(con: sqlite3.Connection) -> None:
    result = con.execute("PRAGMA integrity_check;").fetchone()[0]

    if result != "ok":
        raise RuntimeError(f"Integrity check failed: {result}")


def check_foreign_keys(con: sqlite3.Connection) -> None:
    violations = con.execute("PRAGMA foreign_key_check;").fetchall()

    if violations:
        raise RuntimeError(f"Foreign key violations: {violations}")
