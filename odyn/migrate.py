"""
Database schema migrations. Run this script using:
    `python -m odyn.migrate db_main_folder`

To only regenerate the schema diagram (needs GraphViz `dot`):
    `python -m odyn.migrate --diagram`

To inspect older migrations run:
    `git log -p odyn/latest.sql`

This script:
- Backs up DB to `.odyn/backups/snapshot_v<OLD>.db`
- Applies `latest.sql` migration to DB
- Updates `user_version`
- Regenerates the schema diagram (`schema.svg`) from `create.sql`
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys

from pathlib import Path

from .utils import DB_TIMEOUT_S, ODYN_FOLDER, logger

# When adding a new migration you should:
# - Overwrite latest.sql with the latest migration;
# - Overwrite create.sql with compatible DB schema;
# - Bump the SCHEMA_VERSION to match;
# - Run this script and test_migration.py.

SCHEMA_VERSION = 2
LATEST_MIGRATION = Path(__file__).parent / "latest.sql"

CREATE_SCRIPT = Path(__file__).parent / "create.sql"
DIAGRAM_SCRIPT = Path(__file__).parent / "diagram.sql"
SCHEMA_DIAGRAM = Path(__file__).parent / "schema.svg"


def migrate(main_folder: str | Path, *, diagram: bool = True) -> None:
    """
    Migrate DB from v(SCHEMA_VERSION-1) up to vSCHEMA_VERSION.

    `diagram` regenerates `schema.svg`. It is a tracked file, so pass `False`
    from anything that runs repeatedly (the tests), or a stray graphviz version
    rewrites all of it as a side effect.
    """

    db_path = Path(main_folder) / ODYN_FOLDER / "odyn.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No database at '{db_path}'.")

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

    # Keep the diagram in sync with the schema.
    if diagram:
        generate_diagram()


def generate_diagram() -> None:
    """
    Regenerate `schema.svg` from `create.sql` (needs GraphViz `dot`).

    Builds a in-memory DB from the current schema, renders it to GraphViz
    DOT via `diagram.sql`, and pipes that through `dot`. Skips
    with a warning if `dot` is not installed.
    """

    con = sqlite3.connect(":memory:")

    try:
        con.executescript(CREATE_SCRIPT.read_text())
        rows = con.execute(DIAGRAM_SCRIPT.read_text()).fetchall()
    finally:
        con.close()

    # SQLite3 CLI joins result rows with newlines.
    # We do the same here so the DOT statements stay on separate lines.
    dot = "\n".join((row[0] or "") for row in rows)

    try:
        result = subprocess.run(
            ["dot", "-Tsvg"], input=dot, capture_output=True, text=True, check=True
        )

    except FileNotFoundError:
        logger.warning("GraphViz 'dot' not found; skipping schema.svg regeneration.")
        return

    except subprocess.CalledProcessError as error:
        logger.warning(f"Could not generate schema diagram: {error.stderr.strip()}")
        return

    SCHEMA_DIAGRAM.write_text(result.stdout)
    logger.info(f"Regenerated schema diagram at '{SCHEMA_DIAGRAM}'.")


def check_integrity(con: sqlite3.Connection) -> None:
    result = con.execute("PRAGMA integrity_check;").fetchone()[0]

    if result != "ok":
        raise RuntimeError(f"Integrity check failed: {result}")


def check_foreign_keys(con: sqlite3.Connection) -> None:
    violations = con.execute("PRAGMA foreign_key_check;").fetchall()

    if violations:
        raise RuntimeError(f"Foreign key violations: {violations}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("USAGE: python -m odyn.migrate <main_folder> | --diagram")

    if sys.argv[1] == "--diagram":
        generate_diagram()
    else:
        migrate(sys.argv[1])
