"""
Annotations: the facts about a session or experiment that only a person knows.

They are stored as entity-attribute-value rows, which is what lets a new key
appear without a migration. That storage is not what users should see, so most
of what is checked here is the *reading* side: a wide table of plain values,
with the append-only history collapsed to the current value of each key.
"""

import sqlite3

import pytest

from odyn import Database

from helpers import seed_rows


@pytest.fixture
def db(tmp_path):
    """A database with one session, experiment, acquisitions and group."""
    database = Database(tmp_path, project="test")
    seeded = seed_rows(database)

    # `add_annotation` needs a live call to attribute the write to, and these
    # tests call it directly rather than through ingestion.
    database.exp_id = seeded.exp_id
    database.session_id = seeded.session_id

    return database


def annotate(db, key, value, *, target_type="experiment", target_id=None):
    db.add_annotation(
        target_type=target_type,
        target_id=db.exp_id if target_id is None else target_id,
        key=key,
        value=value,
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def test_an_annotation_is_attributed_to_its_call(db):
    annotate(db, "fov_depth_um", 70.0)

    row = db.con.execute(
        """
        SELECT a.value, m.method_name, m.user
            FROM annotations AS a
            JOIN method_calls AS m ON m.method_call_id = a.method_call_id;
        """
    ).fetchone()

    assert row["value"] == 70.0
    assert row["method_name"] == "Database.add_annotation"
    assert row["user"]


def test_an_unregistered_key_is_refused(db):
    """A typo must not invent a key, or the registry stops meaning anything."""
    with pytest.raises(ValueError, match="No annotation key"):
        annotate(db, "fov_depht_um", 70.0)


def test_a_key_for_another_target_is_refused(db):
    """`fov_depth_um` describes an experiment; a session has no field of view."""
    with pytest.raises(ValueError, match="No annotation key"):
        annotate(db, "fov_depth_um", 70.0, target_type="session",
                 target_id=db.session_id)


def test_a_target_that_does_not_exist_is_refused(db):
    """
    `target_id` is polymorphic, so no foreign key covers it. Without this an
    annotation could name an experiment that was never there and nothing would
    ever notice.
    """
    with pytest.raises(ValueError, match="no experiment 9999"):
        annotate(db, "fov_depth_um", 70.0, target_id=9999)

    with pytest.raises(ValueError, match="Cannot annotate"):
        annotate(db, "fov_depth_um", 70.0, target_type="mouse", target_id=1)


def test_an_id_from_a_dataframe_index_works(db):
    """
    The obvious way to get an id is out of a DataFrame, which hands back a numpy
    integer. That is not an `int` to `isinstance`, and sqlite3 will not adapt
    it, so it has to be coerced rather than refused.

    Getting it as far as the database also needs `record_call` to serialize it;
    see `test_record_call.py`.
    """
    exp_id = db.experiments.index[0]

    assert not isinstance(exp_id, int), "test is pointless if this is a plain int"

    annotate(db, "fov_depth_um", 70.0, target_id=exp_id)

    assert db.annotations_for("experiment")["fov_depth_um"].iloc[0] == 70.0


def test_the_declared_type_is_enforced(db):
    with pytest.raises(ValueError, match="declared real"):
        annotate(db, "fov_depth_um", "seventy microns")


def test_an_enum_only_takes_its_options(db):
    annotate(db, "depth_class", "superficial")

    with pytest.raises(ValueError, match="expected one of"):
        annotate(db, "depth_class", "medium")


def test_numbers_stay_numbers(db):
    """
    The column is `ANY`, which is what keeps a number comparable as a number
    rather than sorting like text.
    """
    annotate(db, "fov_depth_um", 70)

    stored = db.con.execute(
        "SELECT typeof(value) FROM annotations WHERE key = 'fov_depth_um';"
    ).fetchone()[0]

    assert stored == "real"


def test_a_new_key_needs_no_migration(db):
    """The whole point of storing these as rows rather than columns."""
    db.add_annotation_key(
        applies_to="experiment",
        key="test_batch",
        label="Test batch tag",
        value_type="text",
        description="The tag used in this batch.",
    )

    annotate(db, "test_batch", "B-2026-04")

    assert db.annotations_for("experiment")["test_batch"].iloc[0] == "B-2026-04"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_corrections_keep_their_history_but_the_latest_wins(db):
    annotate(db, "fov_depth_um", 70.0)
    annotate(db, "fov_depth_um", 85.0)

    assert len(db.annotations) == 2, "the first value should still be there"
    assert db.annotations_for("experiment")["fov_depth_um"].iloc[0] == 85.0


def test_the_wide_table_leaves_out_multi_valued_keys(db):
    """
    Keeping every column scalar is what lets the result behave like a table --
    sortable, groupable, comparable. Notes and flags live in `db.annotations`.
    """
    annotate(db, "fov_depth_um", 70.0)
    annotate(db, "note", "focus drifted after the injection")
    annotate(db, "note", "second note")

    wide = db.annotations_for("experiment")

    assert "fov_depth_um" in wide.columns
    assert "note" not in wide.columns

    notes = db.annotations.query("key == 'note'")
    assert list(notes["value"]) == [
        "focus drifted after the injection",
        "second note",
    ]


def test_the_wide_table_is_numeric_where_the_registry_says_so(db):
    """Otherwise an ANY column arrives as object dtype and comparisons go wrong."""
    annotate(db, "fov_depth_um", 70.0)
    annotate(db, "depth_class", "superficial")

    wide = db.annotations_for("experiment")

    assert wide["fov_depth_um"].dtype.kind == "f"
    assert not wide.query("fov_depth_um > 100").shape[0]
    assert wide.query("fov_depth_um > 50").shape[0] == 1


def test_it_joins_onto_the_experiments_table(db):
    """The reason the index is the target's own id."""
    annotate(db, "fov_depth_um", 70.0)

    joined = db.experiments.join(db.annotations_for("experiment"))

    assert joined.loc[db.exp_id, "fov_depth_um"] == 70.0
    assert joined.loc[db.exp_id, "exp_type"] == "loop"


def test_an_unannotatable_target_says_so(db):
    with pytest.raises(ValueError, match="Nothing can be annotated"):
        db.annotations_for("mouse")


# --------------------------------------------------------------------------- #
# Nagging
# --------------------------------------------------------------------------- #


def test_required_keys_are_listed_until_they_are_filled(db):
    """This is the list a session GUI would prompt from."""
    missing = db.missing_annotations()
    wanted = set(zip(missing["target_type"], missing["key"]))

    assert ("experiment", "fov_depth_um") in wanted
    assert ("session", "mouse_weight_g") in wanted

    annotate(db, "fov_depth_um", 70.0)

    missing = db.missing_annotations()
    wanted = set(zip(missing["target_type"], missing["key"]))

    assert ("experiment", "fov_depth_um") not in wanted
    assert ("session", "mouse_weight_g") in wanted


def test_optional_keys_are_never_nagged_about(db):
    """A flag records something that happened; it is not a thing owed."""
    missing = db.missing_annotations()

    assert "flag" not in set(missing["key"])
    assert "note" not in set(missing["key"])


# --------------------------------------------------------------------------- #
# Retiring
# --------------------------------------------------------------------------- #


def test_a_retired_key_takes_no_new_values(db):
    annotate(db, "pmt_gain", 20.0)
    db.retire_annotation_key(applies_to="experiment", key="pmt_gain")

    with pytest.raises(ValueError, match="retired"):
        annotate(db, "pmt_gain", 25.0)


def test_retiring_keeps_what_was_already_written(db):
    """
    The difference between retiring a key and deleting one. Old values were
    true when they were written, so they stay readable.
    """
    annotate(db, "pmt_gain", 20.0)
    db.retire_annotation_key(applies_to="experiment", key="pmt_gain")

    assert db.annotations_for("experiment")["pmt_gain"].iloc[0] == 20.0
    assert len(db.annotations.query("key == 'pmt_gain'")) == 1


def test_retiring_a_required_key_stops_the_nagging(db):
    """Otherwise a key nobody may fill in is demanded forever."""
    wanted = lambda: set(zip(db.missing_annotations()["target_type"],
                             db.missing_annotations()["key"]))

    assert ("experiment", "fov_depth_um") in wanted()

    db.retire_annotation_key(applies_to="experiment", key="fov_depth_um")

    assert ("experiment", "fov_depth_um") not in wanted()


def test_a_key_retired_by_mistake_can_come_back(db):
    db.retire_annotation_key(applies_to="experiment", key="pmt_gain")
    db.retire_annotation_key(applies_to="experiment", key="pmt_gain", retired=False)

    annotate(db, "pmt_gain", 20.0)

    assert db.annotations_for("experiment")["pmt_gain"].iloc[0] == 20.0


def test_retiring_something_that_is_not_there_says_so(db):
    with pytest.raises(ValueError, match="No annotation key"):
        db.retire_annotation_key(applies_to="experiment", key="pmt_gian")
