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

    (backup,) = (main_folder / ODYN_FOLDER / "backups").glob("*.db")
    assert backup.name.endswith(f"-main-snapshot-v{SCHEMA_VERSION - 1}.db")

    con = sqlite3.connect(old)

    assert con.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    con.close()


# Minimal FK-valid chains. `migrate` runs PRAGMA foreign_key_check afterwards,
# so orphan rows would fail the migration rather than the assertion.
SEED = r"""
INSERT INTO groups (group_id) VALUES (0);

INSERT INTO method_calls
    ( method_call_id, group_id, method_name
    , parameter_inputs, git_commit, parameters_used
    , ended_at
    ) VALUES
        ( 1, 0, 'Group.run_motion_correction'
        , '{"is_test": true}', 'abc123', '{}'
        , '2026-09-16 10:00:00'
        ),

        ( 2, 0, 'Database.add_experiment'
        , '{}', 'unknown-hash', '{}'
        , '2026-09-16 10:00:00'
        ),

        ( 3, 0, 'Group.outcome_count'
        , '{}', 'abc123', '{}'
        , NULL
        );

UPDATE method_calls
    SET called_at = '2020-01-01 00:00:00'
    WHERE ended_at IS NULL;

INSERT INTO outputs (method_call_id, file_path, removed)
    VALUES (1, 'a.png', FALSE);
"""

# The same call in the current schema, for the tests that stand a current
# database in for the next version's old one.
CURRENT_SEED = r"""
INSERT INTO groups (group_id) VALUES (0);

INSERT INTO method_calls
    ( method_call_id, group_id, user
    , method_name, module, code
    , parameter_inputs, parameters_used
    ) VALUES ( 1, 0, 'someone'
             , 'Group.run_motion_correction', 'odyn.groups', '{}'
             , '{}', '{}'
             );
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


def test_migration_keeps_calls_and_moves_the_commit_into_code(tmp_path):
    """What was never recorded is left unknown rather than made up."""
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        rows = con.execute("""
            SELECT method_call_id, user, module, code
                 , environment, consumed_calls
                 , parameter_inputs, ended_at
                FROM method_calls ORDER BY method_call_id;
        """).fetchall()

        outputs = con.execute("SELECT method_call_id FROM outputs;").fetchall()
    finally:
        con.close()

    assert rows == [
        (
            1,
            "unknown",
            "odyn.groups",
            '{"odyn":{"commit":"abc123","dirty":null}}',
            None,
            None,
            '{"is_test": true}',
            "2026-09-16 10:00:00",
        ),
        (
            2,
            "unknown",
            "odyn.database",
            '{"odyn":{"commit":null,"dirty":null}}',
            None,
            None,
            "{}",
            "2026-09-16 10:00:00",
        ),
        (
            3,
            "unknown",
            "odyn.groups",
            '{"odyn":{"commit":"abc123","dirty":null}}',
            None,
            None,
            "{}",
            None,
        ),
    ]
    assert outputs == [(1,)]


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
    con.executescript(CURRENT_SEED)  # a call with no ended_at, started just now
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
    con.executescript(CURRENT_SEED)
    con.execute("UPDATE method_calls SET called_at = '2020-01-01 00:00:00';")
    con.commit()
    con.close()

    migrate(main_folder)
