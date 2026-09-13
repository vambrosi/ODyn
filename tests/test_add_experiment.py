"""
`Database.add_experiment` reads a folder of ScanImage TIFFs into the schema.

This had no tests at all until v3, which is how a rename that broke three entry
points surfaced as one failure in an unrelated file. The point here is to pin
down *where each fact lands*, because v3 moved several of them: the mouse names
a session rather than an experiment, the rig settings became annotations, and
the sync-derived timing left `acquisitions` entirely.
"""

import shutil
import sqlite3

import pytest

from odyn import Database
from odyn.utils import DEFAULT_PROJECT

from generate_data import EXP, EXP_DATE, EXP_START, MOUSE, generate

REL_PATH = f"{EXP_DATE}/{MOUSE}/{EXP}"
ACQUISITIONS = 3

# `generate` refuses fewer frames than it takes to show a full response, so this
# is its own default rather than something smaller.
FRAMES = 64


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """One synthetic recording, generated once: writing TIFFs is the slow part."""
    folder = tmp_path_factory.mktemp("recording")

    # motion=0 keeps it cheap; none of these tests look at the pixels.
    generate(
        folder,
        acquisitions=ACQUISITIONS,
        frames=FRAMES,
        height=32,
        width=32,
        motion=0.0,
    )

    return folder


@pytest.fixture
def db(recording, tmp_path):
    """A database over this test's own copy of the recording."""
    # Copied rather than shared: `Database` writes '.odyn/odyn.db' inside the
    # folder, so tests that add or re-add would otherwise see each other's rows.
    folder = tmp_path / "main"
    shutil.copytree(recording, folder)

    database = Database(folder, project="test")
    database.add_experiment(rel_path=REL_PATH)

    return database


# --------------------------------------------------------------------------- #
# Which database this version may open
# --------------------------------------------------------------------------- #


def test_the_shared_database_is_unreachable(recording, tmp_path):
    """
    `project=None` used to mean the shared `.odyn/odyn.db`, which is v2 and
    stays with the tagged release. The failure this prevents is worse than an
    error: on a main folder with no database yet, the old behavior would have
    created a v3 file exactly where everyone expects the shared one.
    """
    folder = tmp_path / "main"
    shutil.copytree(recording, folder)

    with pytest.raises(ValueError, match="project"):
        Database(folder, project=None)

    assert not (folder / ".odyn" / "odyn.db").exists()


def test_leaving_the_project_out_uses_the_default(recording, tmp_path):
    """Omitting it is the common case, and must not reach the shared database."""
    folder = tmp_path / "main"
    shutil.copytree(recording, folder)

    database = Database(folder)

    assert database.project == DEFAULT_PROJECT
    assert database.path == folder / ".odyn" / "projects" / f"{DEFAULT_PROJECT}.db"
    assert not (folder / ".odyn" / "odyn.db").exists()


# --------------------------------------------------------------------------- #
# Where each fact lands
# --------------------------------------------------------------------------- #


def test_the_mouse_names_a_session_not_an_experiment(db):
    """`experiments.mouse_id` is gone: a mouse has sessions, sessions have experiments."""
    row = db.con.execute("SELECT * FROM sessions;").fetchone()

    assert row["mouse_id"] == MOUSE
    assert row["session_date"] == EXP_START.date().isoformat()
    # The session folder is the experiment's parent, not the experiment.
    assert row["session_path"] == f"{EXP_DATE}/{MOUSE}"

    columns = {c[1] for c in db.con.execute("PRAGMA table_info(experiments);")}
    assert "mouse_id" not in columns


def test_the_experiment_points_at_its_session(db):
    experiment = db.con.execute("SELECT * FROM experiments;").fetchone()
    session = db.con.execute("SELECT * FROM sessions;").fetchone()

    assert experiment["session_id"] == session["session_id"]
    assert experiment["exp_name"] == f"{EXP_DATE}_{MOUSE}_{EXP}"
    assert experiment["exp_type"] == "loop"


def test_a_second_experiment_reuses_the_session(db):
    """Same mouse, same day, one session -- that is what `_session_id` is for."""
    session_id = db.con.execute("SELECT session_id FROM sessions;").fetchone()[0]

    # A second experiment folder for the same mouse on the same day. Only its
    # metadata matters here, so it is inserted directly rather than generated.
    with db.con as con:
        cur = con.cursor()
        again = _session_id_of(cur, session_path=f"{EXP_DATE}/{MOUSE}")

    assert again == session_id
    assert db.con.execute("SELECT count(*) FROM sessions;").fetchone()[0] == 1


def _session_id_of(cur, *, session_path):
    from odyn.database import _session_id

    return _session_id(
        cur,
        mouse_id=MOUSE,
        session_date=EXP_START.date().isoformat(),
        session_path=session_path,
    )


def test_rig_settings_became_annotations(db):
    """Tier 3: nothing in odyn reads them, so they are not columns any more."""
    columns = {c[1] for c in db.con.execute("PRAGMA table_info(experiments);")}

    for gone in ("laser_power_920", "laser_power_1040", "loop_acq_interval_s"):
        assert gone not in columns

    stored = {
        row["key"]: row["value"]
        for row in db.con.execute(
            "SELECT key, value FROM annotations WHERE target_type = 'experiment';"
        )
    }

    # From generate_data's SI header: powers [10 0], loop interval 10 s.
    assert stored["laser_power_920"] == 10
    assert stored["laser_power_1040"] == 0
    assert stored["loop_acq_interval_s"] == 10.0


def test_annotations_are_attributed_to_the_call_that_wrote_them(db):
    """An annotation's author, time and commit all come from its method call."""
    row = db.con.execute(
        """
        SELECT a.key, m.method_name, m.user
            FROM annotations AS a
            JOIN method_calls AS m ON m.method_call_id = a.method_call_id
            LIMIT 1;
        """
    ).fetchone()

    assert row["method_name"] == "Database.add_experiment"
    assert row["user"]


def test_annotation_keys_must_be_registered(db):
    """The registry is a foreign key, so a typo cannot invent a key."""
    call = db.con.execute("SELECT method_call_id FROM method_calls;").fetchone()[0]
    experiment = db.con.execute("SELECT exp_id FROM experiments;").fetchone()[0]

    with pytest.raises(sqlite3.IntegrityError):
        with db.con as con:
            con.execute(
                """
                INSERT INTO annotations
                    (target_type, target_id, key, value, method_call_id)
                    VALUES ('experiment', ?, 'laser_powr_920', 10, ?);
                """,
                [experiment, call],
            )


# --------------------------------------------------------------------------- #
# Acquisitions
# --------------------------------------------------------------------------- #


def test_every_raw_tiff_becomes_an_acquisition(db):
    rows = db.con.execute(
        "SELECT * FROM acquisitions ORDER BY acq_start;"
    ).fetchall()

    assert len(rows) == ACQUISITIONS
    assert all(row["raw_path"].startswith(f"{REL_PATH}/raw/") for row in rows)
    assert all("\\" not in row["raw_path"] for row in rows)


def test_acquisitions_carry_no_sync_timing(db):
    """
    Odor timing moved to `acquisition_sync`, which stays empty until the sync
    file is copied over and decoded -- possibly days after ingestion.
    """
    columns = {c[1] for c in db.con.execute("PRAGMA table_info(acquisitions);")}

    for gone in ("odor_start", "odor_end", "h5_to_acq_ms"):
        assert gone not in columns

    assert db.con.execute("SELECT count(*) FROM acquisition_sync;").fetchone()[0] == 0


def test_the_experiment_gets_a_group_of_its_own(db):
    """One group per ingested experiment, so motion correction has a unit."""
    row = db.con.execute(
        """
        SELECT ge.group_id, ge.exp_id FROM group_experiments AS ge;
        """
    ).fetchone()

    assert row is not None
    assert row["group_id"] != 0  # 0 is the Database's own provenance row


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_the_call_is_recorded_with_its_code_and_user(db):
    row = db.con.execute(
        """
        SELECT * FROM method_calls
            WHERE method_name = 'Database.add_experiment'
            ORDER BY method_call_id DESC LIMIT 1;
        """
    ).fetchone()

    assert row["group_id"] == 0
    assert row["user"]
    assert row["module"] == "odyn.database"
    assert "odyn" in row["code"]
    assert "python" in row["environment"]


def test_nothing_is_inserted_twice(db):
    """The guard is `exp_start`, which is UNIQUE, so a re-run is a no-op."""
    db.add_experiment(rel_path=REL_PATH)

    counts = {
        table: db.con.execute(f"SELECT count(*) FROM {table};").fetchone()[0]
        for table in ("sessions", "experiments", "acquisitions")
    }

    assert counts == {"sessions": 1, "experiments": 1, "acquisitions": ACQUISITIONS}
