"""
Guards against create.sql / migration drift: a fresh DB built from create.sql
must have the same schema as an old DB walked forward by the migration runner.
"""

import re
import sqlite3

from pathlib import Path

import pytest

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

    # ADD COLUMN splices its column in as "..., new_col ..., FOREIGN KEY"
    sql = re.sub(r"\s*,\s*", ", ", sql)

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


# A minimal FK-valid chain. `migrate` runs PRAGMA foreign_key_check afterwards,
# so orphan rows would fail the migration rather than the assertion.
SEED = r"""
INSERT INTO groups (group_id) VALUES (0);

INSERT INTO method_calls
    ( method_call_id
    , group_id
    , method_name
    , parameter_inputs
    , git_commit
    , parameters_used
    ) VALUES (1, 0, 'Group.run_motion_correction', '{}', 'h', '{}');
"""


def migrated_db(tmp_path):
    """An old DB with one recorded call, walked forward one version."""
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)

    old = main_folder / ODYN_FOLDER / "odyn.db"
    build(old, PREVIOUS_SCHEMA, SCHEMA_VERSION - 1)

    con = sqlite3.connect(old)

    try:
        con.executescript(SEED)
        con.commit()
    finally:
        con.close()

    migrate(main_folder)

    return old


def test_migration_keeps_calls_and_leaves_their_end_unknown(tmp_path):
    """When earlier calls ended was never recorded, so it is not made up."""
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        row = con.execute("""
            SELECT method_name, parameters_used, ended_at
                FROM method_calls WHERE method_call_id = 1;
        """).fetchone()
    finally:
        con.close()

    assert row == ("Group.run_motion_correction", "{}", None)


def test_ended_at_must_be_a_datetime(tmp_path):
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE method_calls SET ended_at = 'yesterday';")

        con.execute("UPDATE method_calls SET ended_at = '2026-09-16 10:00:00';")
    finally:
        con.close()


def test_migration_waits_for_calls_that_have_not_ended(tmp_path, monkeypatch):
    """
    A current database stands in for the next version's old one, with a
    migration that changes nothing, so only the open-call check is tested.
    """
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    db_path = main_folder / ODYN_FOLDER / "odyn.db"
    build(db_path, CREATE_SQL, SCHEMA_VERSION - 1)

    no_op = tmp_path / "no_op.sql"
    no_op.write_text("SELECT 1;")
    monkeypatch.setattr("odyn.migrate.LATEST_MIGRATION", no_op)

    con = sqlite3.connect(db_path)
    con.executescript(SEED)  # a call with no ended_at, started just now
    con.commit()
    con.close()

    with pytest.raises(RuntimeError, match="1 recorded calls have not ended"):
        migrate(main_folder)

    migrate(main_folder, force=True)

    con = sqlite3.connect(db_path)
    assert con.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    con.close()


def test_old_calls_do_not_block_a_migration(tmp_path, monkeypatch):
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    db_path = main_folder / ODYN_FOLDER / "odyn.db"
    build(db_path, CREATE_SQL, SCHEMA_VERSION - 1)

    no_op = tmp_path / "no_op.sql"
    no_op.write_text("SELECT 1;")
    monkeypatch.setattr("odyn.migrate.LATEST_MIGRATION", no_op)

    con = sqlite3.connect(db_path)
    con.executescript(SEED)
    con.execute("UPDATE method_calls SET called_at = '2020-01-01 00:00:00';")
    con.commit()
    con.close()

    migrate(main_folder)
