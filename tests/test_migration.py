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

    migrate(main_folder, diagram=False)

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

INSERT INTO experiments
    ( exp_id, exp_name, exp_type
    , exp_start, mouse_id
    , height_px, width_px, height_um, width_um
    , frame_count, frame_rate
    , laser_power_920, laser_power_1040
    , loop_acq_interval_s
    ) VALUES
        ( 1, '20260317_m317_e1', 'loop'
        , '2026-03-17 10:00:00', 'm317'
        , 512, 512, 512.0, 512.0
        , 280, 14.0
        , 10, 0
        , 10.0
        );

INSERT INTO acquisitions
    (acq_id
    , exp_id
    , acq_start
    , raw_path
    ) VALUES
        ( 1
        , 1
        , '2026-03-17 10:00:01'
        , '20260317\m317\e1\raw\20260317_m317_e1_00001.tif'
        );

INSERT INTO programs
    ( program_id
    , exp_id
    , program_name
    , program_type
    , program_start
    , program_path
    ) VALUES
        ( 1
        , 1
        , 'program_1'
        , 'passive'
        , '2026-03-17 10:00:00'
        , '20260317\m317\e1\olfactometer\program_1_Events.csv'
        );

INSERT INTO mcor_files
    ( acq_id
    , mcor_path
    , last_updated_by
    ) VALUES
        ( 1
        , '20260317\m317\e1\processed\mcor\20260317_m317_e1_00001_mcor.tif'
        , 1
        );

INSERT INTO outputs
    ( method_call_id
    , file_path
    , removed
    ) VALUES
        ( 1
        , '20260317\m317\e1\movies\preview.avi'
        , FALSE
        );
"""


def migrated_db(tmp_path):
    """An old DB seeded with Windows-style paths, walked forward one version."""
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

    migrate(main_folder, diagram=False)

    return old


def test_migration_normalizes_stored_paths(tmp_path):
    """
    Paths are stored relative to main_folder so the DB works from any machine,
    but rows written on Windows kept backslashes and did not resolve elsewhere.
    """
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        stored = [
            con.execute("SELECT raw_path FROM acquisitions;").fetchone()[0],
            con.execute("SELECT program_path FROM programs;").fetchone()[0],
            con.execute("SELECT mcor_path FROM mcor_files;").fetchone()[0],
            con.execute("SELECT file_path FROM outputs;").fetchone()[0],
        ]
    finally:
        con.close()

    assert not any("\\" in path for path in stored), stored
    assert stored[0] == "20260317/m317/e1/raw/20260317_m317_e1_00001.tif"


def test_migration_backfills_mcor_source(tmp_path):
    """Everything already stored was produced by run_motion_correction."""
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        row = con.execute(
            "SELECT source, approved, last_updated_by FROM mcor_files;"
        ).fetchone()
    finally:
        con.close()

    assert row == ("caiman", 0, 1)


def test_mcor_source_is_required_and_checked(tmp_path):
    """
    No default on purpose: a default would silently mislabel a source the
    caller forgot to pass, which is the thing the column exists to prevent.
    """
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("""
                INSERT INTO mcor_files (acq_id, mcor_path, last_updated_by)
                    VALUES (2, 'a/b.tif', 1);
            """)

        with pytest.raises(sqlite3.IntegrityError):
            con.execute("""
                INSERT INTO mcor_files (acq_id, mcor_path, source, last_updated_by)
                    VALUES (2, 'a/b.tif', 'normcorre', 1);
            """)
    finally:
        con.close()


def test_migration_preserves_unrelated_rows(tmp_path):
    con = sqlite3.connect(migrated_db(tmp_path))

    try:
        row = con.execute("""
            SELECT method_name, call_output, parameters_used
                FROM method_calls WHERE method_call_id = 1;
        """).fetchone()
    finally:
        con.close()

    assert row == ("Group.run_motion_correction", None, "{}")
