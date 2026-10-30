"""
Guards against create.sql / migration drift: a fresh DB built from create.sql
must have the same schema as an old DB walked forward by the migration runner.

When a version is a rebuild rather than a migration (`MIGRATES_FROM is None`)
there is nothing to walk forward, so those tests skip and only the refusal is
checked. The open-call tests stand a current database in for the previous
version, so they run either way.
"""

import re
import sqlite3

from pathlib import Path

import pytest

from odyn.migrate import MIGRATES_FROM, SCHEMA_VERSION, migrate

ODYN_FOLDER = ".odyn"
PACKAGE = Path(__file__).resolve().parents[1] / "odyn"

CREATE_SQL = PACKAGE / "create.sql"
PREVIOUS_SCHEMA = Path(__file__).parent / "previous_schema.sql"

# The walk-forward tests only mean something when `latest.sql` migrates. A
# rebuild version skips them rather than deleting them, so the drift check is
# back the moment the next migration lands.
needs_migration = pytest.mark.skipif(
    MIGRATES_FROM is None,
    reason=f"v{SCHEMA_VERSION} is a rebuild; there is no migration to check",
)


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


@needs_migration
def test_migration_matches_fresh_schema(tmp_path):
    fresh = tmp_path / "fresh.db"
    build(fresh, CREATE_SQL, SCHEMA_VERSION)

    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    old = main_folder / ODYN_FOLDER / "odyn.db"
    build(old, PREVIOUS_SCHEMA, MIGRATES_FROM)

    migrate(main_folder)

    assert get_schema(old) == get_schema(fresh)

    (backup,) = (main_folder / ODYN_FOLDER / "backups").glob("*.db")
    assert backup.name.endswith(f"-main-snapshot-v{MIGRATES_FROM}.db")

    con = sqlite3.connect(old)

    assert con.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    con.close()


# Minimal FK-valid chains, in the previous schema. `migrate` runs PRAGMA
# foreign_key_check afterwards, so orphan rows would fail the migration
# rather than the assertion. Call 2 logged nothing; call 3 was written on
# Windows (its line ends are kept as they are).
SEED = r"""
INSERT OR IGNORE INTO groups (group_id) VALUES (0);

INSERT INTO method_calls
    ( method_call_id, group_id, user, method_name, module, code
    , parameter_inputs, parameters_used, call_log, ended_at
    ) VALUES
        ( 1, 0, 'alice', 'Group.run_motion_correction', 'odyn.groups', '{}'
        , '{}', '{}', 'line one' || char(10) || 'line two' || char(10)
        , '2026-09-16 10:00:00'
        ),

        ( 2, 0, 'alice', 'Database.add_experiment', 'odyn.database', '{}'
        , '{}', '{}', '', '2026-09-16 10:00:00'
        ),

        ( 1234, 0, 'bob', 'Database.set_mouse', 'odyn.database', '{}'
        , '{}', '{}', 'on Windows' || char(13) || char(10)
        , '2026-09-16 10:00:00'
        );

INSERT INTO outputs (method_call_id, file_path, removed)
    VALUES (1, 'a.png', FALSE);
"""

# The same call in the current schema, for the tests that stand a current
# database in for the next version's old one.
CURRENT_SEED = r"""
INSERT OR IGNORE INTO groups (group_id) VALUES (0);

INSERT INTO method_calls
    ( method_call_id, group_id, user
    , method_name, module, code
    , parameter_inputs, parameters_used
    ) VALUES ( 1, 0, 'someone'
             , 'Group.run_motion_correction', 'odyn.groups', '{}'
             , '{}', '{}'
             );
"""


@pytest.fixture
def stand_in_migration(tmp_path, monkeypatch):
    """
    A migration that changes nothing, from the version below the current one,
    so a current database can stand in for the previous version's old one.
    """
    no_op = tmp_path / "no_op.sql"
    no_op.write_text("SELECT 1;")
    monkeypatch.setattr("odyn.migrate.LATEST_MIGRATION", no_op)
    monkeypatch.setattr("odyn.migrate.MIGRATES_FROM", SCHEMA_VERSION - 1)
    monkeypatch.setattr("odyn.migrate.BEFORE_SQL", None)


def test_a_rebuild_refuses_to_migrate(tmp_path, monkeypatch):
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    db_path = main_folder / ODYN_FOLDER / "odyn.db"
    build(db_path, CREATE_SQL, SCHEMA_VERSION - 1)

    monkeypatch.setattr("odyn.migrate.MIGRATES_FROM", None)

    with pytest.raises(RuntimeError, match="rebuild, not a migration"):
        migrate(main_folder)

    con = sqlite3.connect(db_path)
    assert con.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION - 1
    con.close()


def migrated_db(tmp_path):
    """An old DB with a few recorded calls, walked forward one version."""
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True, exist_ok=True)

    old = main_folder / ODYN_FOLDER / "odyn.db"
    build(old, PREVIOUS_SCHEMA, MIGRATES_FROM)

    con = sqlite3.connect(old)

    try:
        con.executescript(SEED)
        con.commit()
    finally:
        con.close()

    migrate(main_folder)

    return old


def a_log(db_path, call_id):
    folder = db_path.parent / "logs" / "main" / f"{call_id // 1000:04d}"
    path = folder / f"{call_id}.log"

    return path.read_bytes() if path.exists() else None


@needs_migration
def test_migration_moves_each_log_into_its_file(tmp_path):
    db_path = migrated_db(tmp_path)
    con = sqlite3.connect(db_path)

    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(method_calls);")}
        calls = con.execute("SELECT method_call_id FROM method_calls;").fetchall()
        outputs = con.execute("SELECT method_call_id FROM outputs;").fetchall()
    finally:
        con.close()

    assert "call_log" not in columns
    assert calls == [(1,), (2,), (1234,)] and outputs == [(1,)]
    assert a_log(db_path, 1) == b"line one\nline two\n"
    assert a_log(db_path, 2) is None  # nothing logged, no file
    assert a_log(db_path, 1234) == b"on Windows\r\n"


@needs_migration
def test_a_migration_tried_again_finds_its_files(tmp_path):
    folder = tmp_path / "main" / ODYN_FOLDER / "logs" / "main" / "0000"
    folder.mkdir(parents=True)
    (folder / "1.log").write_bytes(b"line one\nline two\n")

    assert a_log(migrated_db(tmp_path), 1) == b"line one\nline two\n"


@needs_migration
def test_another_log_in_the_way_stops_the_migration(tmp_path):
    folder = tmp_path / "main" / ODYN_FOLDER / "logs" / "main" / "0000"
    folder.mkdir(parents=True)
    (folder / "1.log").write_bytes(b"someone else's")

    with pytest.raises(RuntimeError, match="holds another log than call 1's"):
        migrated_db(tmp_path)

    con = sqlite3.connect(tmp_path / "main" / ODYN_FOLDER / "odyn.db")

    try:
        assert con.execute("PRAGMA user_version;").fetchone()[0] == MIGRATES_FROM
        (kept,) = con.execute(
            "SELECT call_log FROM method_calls WHERE method_call_id = 1;"
        ).fetchone()
        assert kept == "line one\nline two\n"
    finally:
        con.close()


@needs_migration
def test_the_room_of_the_logs_is_given_back(tmp_path):
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    db_path = main_folder / ODYN_FOLDER / "odyn.db"
    build(db_path, PREVIOUS_SCHEMA, MIGRATES_FROM)

    con = sqlite3.connect(db_path)
    con.executescript(SEED)
    con.execute(
        "UPDATE method_calls SET call_log = ? WHERE method_call_id = 1;", ["x" * 2**20]
    )
    con.commit()
    con.close()
    before = db_path.stat().st_size

    migrate(main_folder)

    assert db_path.stat().st_size < before - 2**19
    assert len(a_log(db_path, 1)) == 2**20


@needs_migration
def test_ended_at_must_be_a_datetime(tmp_path):
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("UPDATE method_calls SET ended_at = 'yesterday';")

        con.execute("UPDATE method_calls SET ended_at = '2026-09-16 10:00:00';")
    finally:
        con.close()


def test_migration_waits_for_calls_that_have_not_ended(
    tmp_path, monkeypatch, stand_in_migration
):
    """
    A current database stands in for the next version's old one, with a
    migration that changes nothing, so only the open-call check is tested.
    """
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    db_path = main_folder / ODYN_FOLDER / "odyn.db"
    build(db_path, CREATE_SQL, SCHEMA_VERSION - 1)

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


def test_old_calls_do_not_block_a_migration(tmp_path, monkeypatch, stand_in_migration):
    main_folder = tmp_path / "main"
    (main_folder / ODYN_FOLDER).mkdir(parents=True)
    db_path = main_folder / ODYN_FOLDER / "odyn.db"
    build(db_path, CREATE_SQL, SCHEMA_VERSION - 1)

    con = sqlite3.connect(db_path)
    con.executescript(CURRENT_SEED)
    con.execute("UPDATE method_calls SET called_at = '2020-01-01 00:00:00';")
    con.commit()
    con.close()

    migrate(main_folder)
