"""
What `@record_call` stores about a call's arguments and output.

What reaches a method is whatever the caller had to hand -- an id out of a
DataFrame index, a `Path`, a threshold that came back NaN -- and none of those
are JSON. `jsonable` turns them into something the `json_valid` CHECKs accept.
"""

import json

from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from odyn import Database, record_call
from odyn.utils import jsonable

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
        (pd.Index([1, 2]), [1, 2]),
        ({1, 2}, [1, 2]),
        ((1, 2), [1, 2]),
        ({1: "a"}, {"1": "a"}),
        (3, 3),
        ("x", "x"),
        (None, None),
    ],
)
def test_values_survive_as_themselves(value, expected):
    assert jsonable(value) == expected


def test_non_finite_floats_become_null():
    """`json.dumps` writes a bare `NaN`, which is not valid JSON."""
    for value in (float("nan"), float("inf"), float("-inf"), np.float64("nan")):
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
def test_times_are_written_like_the_database_writes_them(value, expected):
    """A space, not the ISO 'T', like SQLite's own `datetime('now')`."""
    assert jsonable(value) == expected


# --------------------------------------------------------------------------- #
# Through a recorded call
# --------------------------------------------------------------------------- #


@record_call
def example(db, *, ids=(), threshold=0.5, when=None, folder=None):
    db.set_output({"ids": ids, "threshold": threshold, "when": when})


def test_awkward_arguments_are_recorded(tmp_path):
    """
    `record_call` writes the arguments before the function body runs and the
    output after it returns, so both ends have to cope.
    """
    db = Database(tmp_path, can_create=True)

    example(
        db,
        ids=pd.Index([np.int64(1), np.int64(2)]),
        threshold=np.float64("nan"),
        when=datetime(2026, 9, 17, 9, 30),
        folder=tmp_path,
    )

    with db._locked() as con:
        row = con.execute("""
            SELECT parameter_inputs, parameters_used, call_output, call_flag
                FROM method_calls WHERE method_name = 'Database.example';
        """).fetchone()

    inputs, used, output = (json.loads(column) for column in row[:3])

    assert inputs == {
        "ids": [1, 2],
        "threshold": None,
        "when": "2026-09-17 09:30:00",
        "folder": str(tmp_path),
    }
    assert used == inputs
    assert output == {"ids": [1, 2], "threshold": None, "when": "2026-09-17 09:30:00"}
    assert row[3] == 0
