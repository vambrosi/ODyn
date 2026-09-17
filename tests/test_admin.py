"""
`odyn.admin`: manual SQL, copies of the database, and the lock and schema tools.
"""

import json
import sqlite3
import threading
import time

import pytest

from odyn import Database
from odyn.admin import Admin, main
from odyn.locking import DatabaseLock
from odyn.migrate import SCHEMA_VERSION
from test_locking import watch


def odor_names(db):
    return set(db.from_query("SELECT odor_name FROM odors;")["odor_name"])


def backups(main_folder):
    return sorted((main_folder / ".odyn" / "backups").glob("*.db"))


@pytest.fixture
def same_second(monkeypatch):
    """Every backup name gets the same time, as if made in the same second."""
    monkeypatch.setattr("odyn.utils.time.strftime", lambda _: "20260916-120000")


@pytest.fixture
def admin(tmp_path):
    Database(tmp_path, can_create=True)
    return Admin(tmp_path)


# --------------------------------------------------------------------------- #
# sql and script
# --------------------------------------------------------------------------- #


def test_sql_backs_up_applies_and_is_recorded(admin):
    before = odor_names(admin.db)

    rows = admin.sql(statements="INSERT INTO odors (odor_name) VALUES ('new odor');")

    assert rows == 1
    assert odor_names(admin.db) == before | {"new odor"}

    (backup,) = backups(admin.main_folder)
    assert backup.name.endswith("-main-before-sql-1.db")
    old = sqlite3.connect(backup)
    assert "new odor" not in {
        row[0] for row in old.execute("SELECT odor_name FROM odors;")
    }
    old.close()

    call = admin.db.latest_calls("Admin.sql").iloc[0]
    assert call["rows"] == 1 and call["backup"].startswith(".odyn/backups/")


def test_several_statements_are_all_or_nothing(admin):
    with pytest.raises(sqlite3.IntegrityError):
        admin.sql(
            statements="INSERT INTO odors (odor_name) VALUES ('first'); "
            "INSERT INTO odors (odor_name) VALUES (NULL)"
        )

    assert "first" not in odor_names(admin.db)
    assert not list((admin.main_folder / ".odyn" / "backups").glob("*"))
    assert not admin.db._con.in_transaction


def test_schema_changes_apply(admin):
    admin.sql(statements="ALTER TABLE odors ADD COLUMN note TEXT")

    columns = admin.db.from_query("PRAGMA table_info(odors);")["name"]
    assert "note" in set(columns)


def test_script_runs_a_file(admin, tmp_path):
    script = tmp_path / "trigger.sql"
    script.write_text("""
        -- A comment; with a semicolon
        CREATE TABLE renames (old TEXT, new TEXT);
        CREATE TRIGGER remember AFTER UPDATE OF odor_name ON odors
        BEGIN
            INSERT INTO renames VALUES (old.odor_name, new.odor_name);
        END;
        UPDATE odors SET odor_name = 'semi;colon' WHERE odor_id = 1;
        """)

    assert admin.script(script) == 2  # the update and the row its trigger added
    assert "semi;colon" in odor_names(admin.db)


def test_edits_run_under_the_lock(admin):
    unlocked = watch(admin.db)

    admin.sql(statements="UPDATE odors SET odor_name = 'y' WHERE odor_id = 1;")

    assert unlocked == []


# --------------------------------------------------------------------------- #
# backup and snapshot
# --------------------------------------------------------------------------- #


def test_backup_writes_a_checked_copy(admin, same_second):
    path = admin.backup()

    assert path == backups(admin.main_folder)[0]
    assert path.name == "20260916-120000-main-manual-backup.db"
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
    con.close()
    assert not list(path.parent.glob(".*.tmp"))


def test_backup_never_overwrites(admin, same_second):
    path = admin.backup()
    written = path.stat().st_mtime_ns

    with pytest.raises(FileExistsError, match="Delete it or try again"):
        admin.backup()

    assert path.stat().st_mtime_ns == written


def test_backups_of_every_database_share_one_folder(admin, same_second):
    Database(admin.main_folder, project="p1", can_create=True)
    project = Admin(admin.main_folder, project="p1")

    project.backup()
    admin.backup()
    project.sql(statements="INSERT INTO odors (odor_name) VALUES ('x');")

    assert [path.name for path in backups(admin.main_folder)] == [
        "20260916-120000-main-manual-backup.db",
        "20260916-120000-p1-before-sql-1.db",
        "20260916-120000-p1-manual-backup.db",
    ]
    assert project.status()["backups"] == [
        "20260916-120000-p1-before-sql-1.db",
        "20260916-120000-p1-manual-backup.db",
    ]


def test_project_names_are_checked(admin):
    for bad in ("../elsewhere", ".hidden", "with space", ""):
        with pytest.raises(ValueError, match="Project names are letters"):
            Admin(admin.main_folder, project=bad)

    for reserved in ("main", "Main", "mAIN"):
        with pytest.raises(ValueError, match="reserved for the main database"):
            Database(admin.main_folder, project=reserved, can_create=True)

    Database(admin.main_folder, project="2026-09-16_before.cleanup", can_create=True)


def test_snapshot_is_replaced_each_time(admin):
    first = admin.snapshot()
    admin.sql(statements="INSERT INTO odors (odor_name) VALUES ('later');")
    second = admin.snapshot()

    assert first == second == admin.path.parent / "snapshots" / "odyn.db"
    con = sqlite3.connect(second)
    assert "later" in {row[0] for row in con.execute("SELECT odor_name FROM odors;")}
    con.close()


def test_copies_wait_for_the_lock(admin):
    other = DatabaseLock(admin.path)
    released = []

    def hold():
        with other:
            time.sleep(0.3)
            released.append(time.monotonic())

    thread = threading.Thread(target=hold)
    thread.start()
    time.sleep(0.05)

    admin.snapshot()
    copied_at = time.monotonic()

    thread.join(timeout=5)
    assert released and copied_at >= released[0]
    assert not other.path.exists()


# --------------------------------------------------------------------------- #
# status, unlock, migrate
# --------------------------------------------------------------------------- #


def test_status_reports_version_and_lock(admin):
    status = admin.status()

    assert status["schema_version"] == SCHEMA_VERSION
    assert status["lock"] == "free"


def test_calls_record_when_they_end(admin):
    admin.sql(statements="UPDATE odors SET odor_name = 'z' WHERE odor_id = 1;")

    with pytest.raises(sqlite3.IntegrityError):
        admin.sql(statements="INSERT INTO odors (odor_name) VALUES (NULL);")

    calls = admin.db.method_calls
    assert calls["ended_at"].notna().all() and len(calls) == 2


def test_status_lists_calls_that_have_not_ended(admin):
    with admin.db._locked() as con, con:
        con.execute("""
            INSERT INTO method_calls
                (group_id, method_name, parameter_inputs, git_commit, parameters_used)
                VALUES (0, 'Group.run_motion_correction', '{}', 'h', '{}');
        """)

    (call,) = admin.status()["open_calls"]
    assert call[1] == "Group.run_motion_correction"


def test_unlock_needs_yes(admin):
    lock = DatabaseLock(admin.path)
    lock.path.write_text(
        json.dumps({"token": "t", "user": "u", "host": "h", "since": time.time()})
    )

    assert admin.unlock() is False and lock.path.exists()
    assert admin.unlock(yes=True) is True and not lock.path.exists()
    assert admin.unlock(yes=True) is False


def test_status_and_unlock_work_on_an_old_schema(admin):
    """They must not need a Database, which refuses a schema it does not expect."""
    con = sqlite3.connect(admin.path)
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1};")
    con.close()

    fresh = Admin(admin.main_folder)
    assert fresh.status()["schema_version"] == SCHEMA_VERSION - 1
    assert fresh.unlock() is False


def test_migrate_needs_its_own_admin(admin, odyn_log):
    admin.db  # this Admin now holds a connection
    with pytest.raises(RuntimeError, match="new Admin"):
        admin.migrate()

    Admin(admin.main_folder).migrate()
    assert f"already at v{SCHEMA_VERSION}" in odyn_log.text


def test_a_missing_database_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        Admin(tmp_path / "typo")


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_command_line(admin, tmp_path):
    folder = str(admin.main_folder)

    main([folder, "sql", "UPDATE odors SET odor_name = 'cli' WHERE odor_id = 1;"])
    assert "cli" in odor_names(admin.db)

    script = tmp_path / "edits.sql"
    script.write_text("UPDATE odors SET odor_name = 'file' WHERE odor_id = 1;")
    main([folder, "script", str(script)])
    assert "file" in odor_names(admin.db)

    main([folder, "backup"])
    assert backups(admin.main_folder)[-1].name.endswith("-main-manual-backup.db")

    main([folder, "snapshot"])
    main([folder, "unlock", "--yes"])
    main([folder, "status"])
