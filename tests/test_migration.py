"""
Guards against create.sql / migration drift: a fresh DB built from create.sql
must have the same schema as an old DB walked forward by the migration runner.
"""

import re
import sqlite3

from pathlib import Path

from odyn.migrate import SCHEMA_VERSION, migrate

ODYN_FOLDER = ".odyn"
PACKAGE = Path(__file__).resolve().parents[1] / "odyn"

CREATE_SQL = PACKAGE / "create.sql"
PREVIOUS_SCHEMA = Path(__file__).parent / "previous_schema.sql"


def normalize(sql: str | None) -> str:
    """
    Compare schemas ignoring whitespace, IF NOT EXISTS, and
    identifier quoting (ALTER TABLE RENAME adds double quotes).
    """
    if sql is None:
        return ""

    sql = re.sub(r"\bIF NOT EXISTS\b", "", sql, flags=re.IGNORECASE)
    sql = sql.replace('"', "")

    return re.sub(r"\s+", " ", sql).strip().lower()


def get_schema(db_path: Path) -> list[tuple]:
    con = sqlite3.connect(db_path)

    try:
        rows = con.execute("""
            SELECT type, name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name;
        """).fetchall()

    finally:
        con.close()

    return [(t, n, normalize(sql)) for (t, n, sql) in rows]


def build(db_path: Path, script: Path, version: int) -> None:
    con = sqlite3.connect(db_path)

    try:
        con.executescript(script.read_text())
        con.execute(f"PRAGMA user_version = {version};")
        con.commit()

    finally:
        con.close()


def test_migration_matches_fresh_schema(tmp_path):
    fresh = tmp_path / "fresh.db"
    build(fresh, CREATE_SQL, SCHEMA_VERSION)

    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    old = main_folder / ODYN_FOLDER / "odyn.db"
    build(old, PREVIOUS_SCHEMA, SCHEMA_VERSION - 1)

    migrate(main_folder)

    assert get_schema(old) == get_schema(fresh)
    con = sqlite3.connect(old)

    assert con.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    con.close()


def test_migration_preserves_rows_and_backfills(tmp_path):
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)

    old = main_folder / ODYN_FOLDER / "odyn.db"
    build(old, PREVIOUS_SCHEMA, SCHEMA_VERSION - 1)

    con = sqlite3.connect(old)
    con.execute("INSERT INTO groups (group_id) VALUES (0);")
    con.execute("""
        INSERT INTO method_calls
            ( method_call_id
            , group_id
            , method_name
            , parameters
            , git_commit
            , call_output
            ) VALUES ( 1, 0
                     , 'Group.run_motion_correction'
                     , '{"pw_rigid": true}', 'h', NULL
                     );
    """)
    con.commit()
    con.close()

    migrate(main_folder)

    con = sqlite3.connect(old)

    row = con.execute("""
        SELECT call_output, parameters_used
            FROM method_calls WHERE method_call_id = 1;
    """).fetchone()

    con.close()
    assert row == (None, "{}")  # NULL output preserved, resolved backfilled
