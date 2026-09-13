"""
What `@record_call` stores about a call.

The awkward part is JSON. What reaches a method is whatever the caller had to
hand -- an id out of a DataFrame index, a `Path`, a threshold that came back
NaN -- and none of those are JSON. Two of them used to fail in ways that were
hard to read, so most of this file is about arguments rather than about results.
"""

import json
import math
import sqlite3

from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from odyn import Database
from odyn.utils import jsonable

from helpers import seed_rows


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path, project="test")
    database.seeded = seed_rows(database)

    return database


def recorded(db, method_name="Database.add_annotation"):
    return db.con.execute(
        """
        SELECT * FROM method_calls
            WHERE method_name = ? ORDER BY method_call_id DESC LIMIT 1;
        """,
        [method_name],
    ).fetchone()


# --------------------------------------------------------------------------- #
# jsonable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(7), 7),
        (np.float32(1.5), 1.5),
        (np.bool_(True), True),
        (np.array([1, 2]), [1, 2]),
        (pd.Series([1, 2]), [1, 2]),
        ({1, 2}, [1, 2]),
        ((1, 2), [1, 2]),
        (3, 3),
        ("x", "x"),
        (None, None),
    ],
)
def test_values_survive_as_themselves(value, expected):
    assert jsonable(value) == expected


def test_non_finite_floats_become_null(db):
    """
    These are the dangerous ones: `json.dumps` writes a bare `NaN` without
    complaining, which is not valid JSON, so the `json_valid` CHECK rejects it
    when the row is written -- for `parameters_used` that is the UPDATE at the
    *end* of the call, after all the work is done.
    """
    for value in (float("nan"), float("inf"), float("-inf")):
        assert jsonable(value) is None

    assert jsonable({"a": [1.0, float("nan")]}) == {"a": [1.0, None]}


def test_anything_unrecognized_becomes_text():
    """Losing an argument's exact form beats failing the call over it."""
    assert jsonable(object()).startswith("<object object")


def test_paths_are_readable(tmp_path):
    assert jsonable(tmp_path) == str(tmp_path)


@pytest.mark.parametrize(
    "value, expected",
    [
        (datetime(2026, 8, 11, 13, 36, 0, 482264), "2026-08-11 13:36:00.482264"),
        (pd.Timestamp("2026-08-11 13:36:00.482264"), "2026-08-11 13:36:00.482264"),
        (np.datetime64("2026-08-11T13:36:00.482264"), "2026-08-11 13:36:00.482264"),
        (date(2026, 8, 11), "2026-08-11"),
        (time(13, 36), "13:36:00"),
    ],
)
def test_times_are_written_the_way_the_columns_are(value, expected):
    """
    A space, not the ISO 'T'. SQLite's own `datetime('now')` writes a space, so
    every datetime column in the database reads that way, and a datetime landing
    in `parameters_used` should not be the one thing that reads differently.
    """
    assert jsonable(value) == expected


def test_a_date_or_time_does_not_take_a_separator():
    """
    `date` and `time` have no `sep`, so they fall back -- and `time` is the trap:
    called positionally it reads the argument as a *timespec* and raises
    ValueError rather than TypeError.
    """
    assert jsonable(date(2026, 8, 11)) == "2026-08-11"
    assert jsonable(time(13, 36)) == "13:36:00"


# --------------------------------------------------------------------------- #
# Through a real recorded call
# --------------------------------------------------------------------------- #


def test_an_id_from_a_dataframe_is_recorded(db):
    """
    `record_call` serializes kwargs before the method body runs, so coercing
    inside the method is too late: this used to raise `TypeError` from a line
    about JSON, before the call had done anything.
    """
    exp_id = db.experiments.index[0]
    assert not isinstance(exp_id, int), "pointless test if this is a plain int"

    db.add_annotation(
        target_type="experiment", target_id=exp_id, key="fov_depth_um", value=70.0
    )

    stored = json.loads(recorded(db)["parameter_inputs"])

    assert stored["target_id"] == 1
    assert isinstance(stored["target_id"], int)


def test_what_is_stored_is_always_valid_json(db):
    """The `json_valid` CHECK is the thing that would reject it."""
    db.add_annotation(
        target_type="experiment",
        target_id=db.seeded.exp_id,
        key="fov_depth_um",
        value=np.float64(70.0),
    )

    row = recorded(db)

    for column in ("parameter_inputs", "parameters_used", "code", "environment"):
        json.loads(row[column])  # raises if the column is not valid JSON


def test_a_bad_argument_does_not_lose_the_call(db):
    """
    A value nothing knows how to serialize is recorded as text rather than
    stopping a method that would otherwise have worked.
    """
    db.add_annotation(
        target_type="experiment",
        target_id=db.seeded.exp_id,
        key="fov_depth_raw",
        value="70 um",
    )

    assert len(db.annotations) == 1
