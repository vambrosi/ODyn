"""
What `@record_call` stores about a call's arguments and output.

What reaches a method is whatever the caller had to hand -- an id out of a
DataFrame index, a `Path`, a threshold that came back NaN -- and none of those
are JSON. `jsonable` turns them into something the `json_valid` CHECKs accept.
"""

import importlib
import json
import platform
import shutil
import subprocess

from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from odyn import Database, record_call
from odyn.utils import get_code, jsonable

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


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def git(folder, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=folder,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A git repository named `project` with one module committed."""
    if shutil.which("git") is None:
        pytest.skip("needs git")

    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "analysis.py").write_text("def run(group, *, x=1):\n    return x\n")

    git(folder, "init", "-q")
    git(folder, "add", ".")
    git(folder, "commit", "-q", "-m", "first")

    return folder


def recorded_calls(db):
    with db._locked() as con:
        return con.execute("""
            SELECT user, method_name, module, code, environment, consumed_calls
                FROM method_calls ORDER BY method_call_id;
        """).fetchall()


@record_call
def produce(db):
    db.set_output({"value": 1})


@record_call
def consume(db):
    db.latest_output("Database.produce")


def test_calls_record_who_where_and_with_what(tmp_path, monkeypatch):
    monkeypatch.setenv("ODYN_USER", "ana")
    db = Database(tmp_path, can_create=True)

    produce(db)

    user, name, module, code, environment, consumed = recorded_calls(db)[0]

    assert (user, name, module, consumed) == (
        "ana",
        "Database.produce",
        "test_record_call",
        None,
    )
    assert json.loads(environment)["python"] == platform.python_version()

    if "odyn" in json.loads(code):
        assert set(json.loads(code)["odyn"]) == {"commit", "dirty"}


def test_a_call_records_the_outputs_it_read(tmp_path):
    db = Database(tmp_path, can_create=True)

    produce(db)
    consume(db)
    db.latest_output("Database.produce")  # outside a call: nothing to record

    calls = recorded_calls(db)

    assert len(calls) == 2
    assert json.loads(calls[1][5]) == [1]


def test_code_names_the_repository_a_function_comes_from(repo, monkeypatch):
    monkeypatch.syspath_prepend(str(repo))
    analysis = importlib.import_module("analysis")

    (state,) = [
        info for name, info in get_code(analysis.run).items() if name == "project"
    ]
    assert state["dirty"] is False

    (repo / "notes.txt").write_text("untracked files are not code that ran")
    assert get_code(analysis.run)["project"]["dirty"] is False

    (repo / "analysis.py").write_text("def run(group, *, x=2):\n    return x\n")
    assert get_code(analysis.run)["project"]["dirty"] is True
