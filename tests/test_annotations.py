"""
Annotations: the typed key registry, the append-only values, the latest-wins
pivot and the list of what is still missing.
"""

import pandas as pd
import pytest

from odyn import Database


@pytest.fixture
def db(tmp_path):
    db = Database(tmp_path, can_create=True, user="tester")
    db.set_mouse(mouse_id=1)
    db.add_session(mouse_id=1, session_date="2026-01-01")

    with db._writing() as cur:
        for exp_id in (1, 2):
            cur.execute(
                """
                INSERT INTO experiments
                    ( exp_id, session_id, exp_name, exp_type, exp_path, exp_start
                    , height_px, width_px, height_um, width_um, frame_count
                    , frame_rate, objective
                    ) VALUES (?, 1, ?, 'loop', 'p', ?, 1, 1, 1, 1, 1, 1, 20);
                """,
                [exp_id, f"e{exp_id}", f"2026-01-01 1{exp_id}:00:00"],
            )

    return db


def annotate(db, target_id, **values):
    db.add_annotations(target_type="experiment", target_id=target_id, values=values)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def test_values_are_coerced_to_their_declared_type(db):
    annotate(
        db,
        1,
        fov_depth_um="70",
        laser_power_920=40.0,
        flag="yes",
        depth_goal="Superficial",
        treatment="fine",
    )

    current = db.annotations_for("experiment").loc[1]
    assert current["fov_depth_um"] == 70.0
    assert current["laser_power_920"] == 40
    assert current["flag"] is True
    assert current["depth_goal"] == "superficial"
    assert current["treatment"] == "fine"


@pytest.mark.parametrize(
    "values, message",
    [
        ({"fov_depth_um": "deep"}, "declared real"),
        ({"laser_power_920": 3.7}, "not a whole number"),
        ({"flag": "maybe"}, "not yes or no"),
        ({"depth_goal": "medium"}, "expected one of"),
        ({"nope": 1}, "No annotation key"),
        ({"mouse_weight_g": 1}, "No annotation key"),  # a session key
    ],
)
def test_wrong_values_are_refused_before_anything_is_written(db, values, message):
    with pytest.raises(ValueError, match=message):
        annotate(db, 1, treatment="first", **values)

    assert len(db.annotations) == 0


def test_a_missing_target_is_refused(db):
    with pytest.raises(ValueError, match="no experiment 9"):
        annotate(db, 9, treatment="x")

    with pytest.raises(ValueError, match="Cannot annotate"):
        db.add_annotations(target_type="vial", target_id=1, values={"treatment": "x"})


def test_dates_are_validated_and_read_back_as_dates(db):
    db.add_annotation_key(
        applies_to="experiment",
        key="checked_on",
        label="Checked on",
        value_type="date",
        description="When the recording was looked at.",
    )
    annotate(db, 1, checked_on="2026-02-03")

    assert db.annotations_for("experiment").loc[1, "checked_on"] == pd.Timestamp(
        "2026-02-03"
    )

    with pytest.raises(ValueError, match="Not a date"):
        annotate(db, 1, checked_on="soon")


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #


def test_the_latest_value_wins_and_the_history_stays(db):
    annotate(db, 1, fov_depth_um=70)
    annotate(db, 1, fov_depth_um=75)

    assert db.annotations_for("experiment").loc[1, "fov_depth_um"] == 75
    assert list(db.annotations["value"]) == [70, 75]


def test_none_clears_a_key(db):
    db.update_annotation_key(applies_to="experiment", key="fov_depth_um", required=True)
    annotate(db, 1, fov_depth_um=70, treatment="x")
    annotate(db, 1, fov_depth_um=None)

    current = db.annotations_for("experiment")
    assert pd.isna(current.loc[1, "fov_depth_um"])
    assert current.loc[1, "treatment"] == "x"

    missing = db.missing_annotations()
    assert (1, "fov_depth_um") in set(zip(missing["target_id"], missing["key"]))


def test_the_pivot_has_one_row_per_target_and_typed_columns(db):
    annotate(db, 1, fov_depth_um=70, pmt_gain=20)
    annotate(db, 2, fov_depth_um=200)

    frame = db.annotations_for("experiment")
    assert frame.index.name == "exp_id"
    assert list(frame.index) == [1, 2]
    assert frame["fov_depth_um"].dtype.kind == "f"
    assert pd.isna(frame.loc[2, "pmt_gain"])

    deep = frame.query("fov_depth_um > 100")
    assert list(deep.index) == [2]


def test_nothing_annotated_is_an_empty_frame(db):
    frame = db.annotations_for("session")
    assert len(frame) == 0

    with pytest.raises(ValueError, match="Nothing can be annotated on a 'vial'"):
        db.annotations_for("vial")


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


def test_keys_are_registered_updated_and_retired(db):
    db.add_annotation_key(
        applies_to="experiment",
        key="cohort",
        label="Cohort",
        value_type="enum",
        allowed_values=["a", "b"],
        description="Which cohort.",
        source="user",
    )
    key = db.annotation_keys.loc[("experiment", "cohort")]
    assert key["allowed_values"] == ["a", "b"]  # parsed, on every pandas
    assert key["source"] == "user"

    db.update_annotation_key(
        applies_to="experiment", key="cohort", add_values=["c"], label="Cohort name"
    )
    key = db.annotation_keys.loc[("experiment", "cohort")]
    assert key["allowed_values"] == ["a", "b", "c"]
    assert key["label"] == "Cohort name"

    annotate(db, 1, cohort="c")

    db.retire_annotation_key(applies_to="experiment", key="cohort")

    with pytest.raises(ValueError, match="retired"):
        annotate(db, 2, cohort="a")

    # What was written stays readable
    assert db.annotations_for("experiment").loc[1, "cohort"] == "c"

    db.retire_annotation_key(applies_to="experiment", key="cohort", retired=False)
    annotate(db, 2, cohort="a")

    with pytest.raises(ValueError, match="not an enum"):
        db.update_annotation_key(
            applies_to="experiment", key="treatment", add_values=["x"]
        )

    with pytest.raises(ValueError, match="No annotation key"):
        db.retire_annotation_key(applies_to="experiment", key="nope")


def test_required_keys_are_nagged_until_filled(db):
    assert len(db.missing_annotations()) == 0  # nothing is required out of the box

    db.update_annotation_key(applies_to="experiment", key="fov_depth_um", required=True)
    db.update_annotation_key(applies_to="session", key="mouse_weight_g", required=True)
    missing = db.missing_annotations()

    assert set(zip(missing["target_type"], missing["target_id"], missing["key"])) == {
        ("experiment", 1, "fov_depth_um"),
        ("experiment", 2, "fov_depth_um"),
        ("session", 1, "mouse_weight_g"),
    }

    annotate(db, 1, fov_depth_um=70)
    db.add_annotations(
        target_type="session", target_id=1, values={"mouse_weight_g": 24}
    )

    missing = db.missing_annotations()
    assert list(zip(missing["target_id"], missing["key"])) == [(2, "fov_depth_um")]

    # A retired key stops being demanded
    db.retire_annotation_key(applies_to="experiment", key="fov_depth_um")
    assert len(db.missing_annotations()) == 0


def test_every_annotation_carries_its_call(db):
    annotate(db, 1, treatment="x", flag=True)

    calls = set(db.annotations["method_call_id"])
    assert len(calls) == 1

    call = db.method_calls.loc[calls.pop()]
    assert call["method_name"] == "Database.add_annotations"
    assert call["parameter_inputs"]["values"] == {"treatment": "x", "flag": True}
    assert call["user"] == "tester"
