"""
The database lock file, and that `Database` never touches the file without it.

A workstation and a cluster node do not see each other's SQLite locks on the
share, so a single read outside the lock can corrupt the database. The last
test is the one that matters most: it watches every statement SQLite runs.
"""

import json
import sqlite3
import threading
import time

import pytest

from odyn import Database
from odyn.locking import STALE_AFTER_S, DatabaseLock
from odyn.migrate import SCHEMA_VERSION
from odyn.utils import logger
from test_add_mcor_files import build, write_mcor


@pytest.fixture
def odyn_log(caplog):
    """`caplog` for odyn's logger, which does not pass records up to the root."""
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


# --------------------------------------------------------------------------- #
# The lock itself
# --------------------------------------------------------------------------- #


def test_file_exists_only_while_held(tmp_path):
    lock = DatabaseLock(tmp_path / "odyn.db")

    with lock:
        holder = json.loads(lock.path.read_text())
        assert holder["pid"] > 0 and holder["user"]

    assert not lock.path.exists()


def test_nested_uses_take_the_file_once(tmp_path):
    lock = DatabaseLock(tmp_path / "odyn.db")

    with lock as outer:
        token = json.loads(lock.path.read_text())["token"]

        with lock as inner:
            assert json.loads(lock.path.read_text())["token"] == token

        assert lock.path.exists(), "the inner release must not free the file"

    assert (outer, inner) == (True, False)
    assert not lock.path.exists()


def test_a_second_holder_waits_for_the_first(tmp_path):
    # Two instances stand in for two processes: they share only the file.
    first, second = DatabaseLock(tmp_path / "odyn.db"), DatabaseLock(
        tmp_path / "odyn.db"
    )
    events = []

    def other():
        with second:
            events.append("second")

    with first:
        thread = threading.Thread(target=other)
        thread.start()
        time.sleep(0.3)
        events.append("first done")

    thread.join(timeout=5)
    assert events == ["first done", "second"]


def test_a_stale_lock_is_taken_over(tmp_path, odyn_log):
    lock = DatabaseLock(tmp_path / "odyn.db")
    lock.path.write_text(
        json.dumps(
            {
                "token": "dead",
                "user": "someone",
                "host": "crashed-machine",
                "pid": 1,
                "since": time.time() - STALE_AFTER_S - 1,
            }
        )
    )

    with lock:
        assert json.loads(lock.path.read_text())["token"] != "dead"

    assert "Took over a stale database lock" in odyn_log.text
    assert not lock.path.exists()
    assert not list(tmp_path.glob("*.stale-*"))


def test_a_fresh_lock_is_not_taken_over(tmp_path):
    first, second = DatabaseLock(tmp_path / "odyn.db"), DatabaseLock(
        tmp_path / "odyn.db"
    )
    acquired = threading.Event()

    def other():
        with second:
            acquired.set()

    with first:
        thread = threading.Thread(target=other)
        thread.start()
        assert not acquired.wait(0.3)

    thread.join(timeout=5)
    assert acquired.is_set()


def test_a_holder_does_not_delete_a_lock_taken_over_from_it(tmp_path, odyn_log):
    lock = DatabaseLock(tmp_path / "odyn.db")

    with lock:
        # Someone decided this holder was dead and took the file.
        lock.path.write_text(json.dumps({"token": "new", "since": time.time()}))

    assert json.loads(lock.path.read_text())["token"] == "new"
    assert "taken over while this process held it" in odyn_log.text


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #


def test_a_missing_database_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError, match="can_create=True"):
        Database(tmp_path / "typo")

    with pytest.raises(FileNotFoundError):
        Database(tmp_path, project="typo")

    assert list(tmp_path.iterdir()) == [], "nothing may be left on disk"


def test_an_existing_database_opens_without_can_create(tmp_path):
    Database(tmp_path, can_create=True)
    Database(tmp_path)


def test_from_query_does_not_write(tmp_path):
    db = Database(tmp_path, can_create=True)

    with pytest.raises(Exception):
        db.from_query("INSERT INTO odors (odor_name) VALUES ('x');")

    assert "x" not in set(db.odors["odor_name"])


def test_an_uncommitted_write_is_not_left_open(tmp_path, odyn_log):
    db = Database(tmp_path, can_create=True)

    with db._locked() as con:
        con.execute("INSERT INTO odors (odor_name) VALUES ('x');")

    assert not db._con.in_transaction
    assert "x" not in set(db.odors["odor_name"])
    assert "never committed" in odyn_log.text


def test_reads_wait_for_another_holder(tmp_path):
    db = Database(tmp_path, can_create=True)
    other = DatabaseLock(db.path)
    released = []

    def hold():
        with other:
            time.sleep(0.3)
            released.append(time.monotonic())

    thread = threading.Thread(target=hold)
    thread.start()
    time.sleep(0.05)

    db.experiments
    read_at = time.monotonic()

    thread.join(timeout=5)
    assert released and read_at >= released[0]


def test_nothing_is_left_behind(tmp_path):
    db, group = build(tmp_path)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()
    db.from_query("SELECT * FROM experiments;")

    assert not DatabaseLock(db.path).path.exists()


def test_a_migration_while_open_stops_the_next_use(tmp_path):
    """A notebook left open must not keep writing through an old schema."""
    db = Database(tmp_path, can_create=True)
    db.experiments

    with DatabaseLock(db.path):
        con = sqlite3.connect(db.path)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1};")
        con.close()

    with pytest.raises(RuntimeError, match="out of date"):
        db.method_calls


def watch(db):
    """
    Statements SQLite runs while `db` does not hold its lock.

    SQLite calls the trace callback for each statement, so any use of the
    connection that skips `_locked()` lands in the returned list.
    """
    unlocked = []

    def trace(statement):
        if db._lock._depth == 0:
            unlocked.append(statement)

    db._con.set_trace_callback(trace)
    return unlocked


def test_ingestion_and_reads_run_under_the_lock(tmp_path):
    pytest.importorskip("skimage")  # generate_data warps frames with it
    from generate_data import generate

    generate(tmp_path, acquisitions=2)
    db = Database(tmp_path, can_create=True)
    unlocked = watch(db)

    db.add_experiment(rel_path="20260101/m001/e1")
    db.add_experiment(rel_path="20260101/m001/e1")  # already there

    for table in (
        "acquisitions",
        "acquisition_trials",
        "events",
        "experiments",
        "group_experiments",
        "mcor_files",
        "method_calls",
        "odors",
        "outputs",
        "programs",
        "trials",
    ):
        getattr(db, table)

    db.from_query("SELECT count(*) FROM experiments;")
    db.latest_calls("add_experiment")
    db.latest_output("add_experiment")

    group = db.add_group(exp_ids=[1])

    for table in (
        "acquisitions",
        "acquisition_trials",
        "events",
        "experiments",
        "mcor_files",
        "method_calls",
        "outputs",
        "programs",
        "trials",
    ):
        getattr(group, table)

    group.latest_calls("add_mcor_files")
    group.latest_output("add_mcor_files")

    assert unlocked == []


def test_mcor_writes_run_under_the_lock(tmp_path):
    db, group = build(tmp_path)
    unlocked = watch(db)

    write_mcor(tmp_path, 1)
    group.add_mcor_files()
    write_mcor(tmp_path, 2)
    group.add_mcor_files(overwrite=True)
    group.approve_mcor_files(exclude_acq_ids=[2])

    assert unlocked == []
