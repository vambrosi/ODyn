"""
The table readers: one spec per table, one cache, and frames that look the
same on every pandas.
"""

import pandas as pd
import pytest

from odyn import Database
from odyn import utils


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path, can_create=True, user="tester")


SPECS = [value for value in vars(utils).values() if isinstance(value, utils.TableSpec)]


def test_every_spec_reads_from_a_fresh_database(db):
    with db._locked() as con:
        for spec in SPECS:
            frame = utils._read_table(con, spec, spec.all_rows)
            assert frame.index.name == (
                spec.index if spec is not utils.ANNOTATION_KEYS else None
            )

            if spec.in_group is not None:
                utils._read_table(con, spec, spec.in_group, [1])


def test_null_text_reads_as_none_on_every_pandas(db):
    db.set_mouse(mouse_id=1)
    db.add_odor(odor_id=99, odor_name="x")
    db.add_line(line_name="Line-A")

    # Columns nobody filled: object of None, never str of NaN
    assert db.mice["mouse_sex"].dtype == object
    assert db.mice.loc[1, "mouse_sex"] is None
    assert db.lines.loc["Line-A", "jax_strain_number"] is None
    assert db.odors.loc[99, "vapor_pressure_source"] is None

    # Numbers keep NaN, so arithmetic and `.notna()` keep working
    assert db.odors["vapor_pressure_mmhg"].dtype.kind == "f"
    assert pd.isna(db.odors.loc[99, "vapor_pressure_mmhg"])


def test_json_columns_hold_python_objects(db):
    db.add_odor(odor_id=99, odor_name="x")

    call = db.method_calls.iloc[0]
    assert call["parameter_inputs"] == {"odor_id": 99, "odor_name": "x"}
    assert isinstance(call["code"], dict)
    assert call["consumed_calls"] is None

    keys = db.annotation_keys
    assert keys.loc[("loop", "depth_goal"), "allowed_values"] == [
        "superficial",
        "deep",
    ]
    assert keys.loc[("loop", "treatment"), "allowed_values"] is None


def test_the_cache_is_one_dict_that_writes_clear(db):
    db.mice
    db.odors
    assert set(db._tables) == {"mice", "odors"}

    db.set_mouse(mouse_id=1)
    assert db._tables == {}
    assert 1 in db.mice.index


def test_a_group_view_is_the_same_reader_restricted(db):
    db.set_mouse(mouse_id=1)
    db.add_session(mouse_id=1, session_date="2026-01-01")

    with db._writing() as cur:
        for loop_id in (1, 2):
            cur.execute(
                """
                INSERT INTO loops
                    ( loop_id, session_id, loop_number, loop_path, loop_start
                    , height_px, width_px, height_um, width_um, frames_per_slice
                    , frame_rate, objective
                    ) VALUES (?, 1, ?, ?, ?, 1, 1, 1, 1, 1, 1, 20);
                """,
                [loop_id, loop_id, f"p{loop_id}", f"2026-01-01 1{loop_id}:00:00"],
            )

    group = db.add_group([2])

    assert list(db.loops.index) == [1, 2]
    assert list(group.loops.index) == [2]
    assert list(group.sessions.index) == [1]
    assert list(group.loops.columns) == list(db.loops.columns)
