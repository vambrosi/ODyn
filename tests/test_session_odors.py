"""
`session_odors`: what was in each vial, and how it was made, for one session.

Odor identity is global and lives in `odors`. The vial it sat in, the dilution
and the date it was mixed are true of one session only, and they come off the
`odorLog` sheet of the session's log workbook -- the same six fields every time,
which is why this is a table and not annotations.
"""

import sqlite3

from datetime import datetime

import pytest

from odyn import Database

from helpers import seed_rows

PANEL = [
    {"odor_id": 1, "vial": 1, "goal_ppm": "0.3", "sccm": 200,
     "made_on": "2026-07-06"},
    {"odor_id": 17, "vial": 4, "goal_ppm": "44.2", "sccm": 200,
     "made_on": "2026-07-06"},
    {"odor_id": 0, "vial": 3, "goal_ppm": "na", "sccm": 200},
]


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path, project="test")
    database.seeded = seed_rows(database)

    return database


def test_a_panel_is_recorded_against_its_session(db):
    db.set_session_odors(session_id=db.seeded.session_id, odors=PANEL)

    stored = db.session_odors.loc[db.seeded.session_id]

    assert sorted(stored.index) == [0, 1, 17]
    assert stored.loc[1, "vial"] == 1
    assert stored.loc[1, "sccm"] == 200


def test_goal_ppm_keeps_what_was_written(db):
    """
    The log holds ranges and words as well as numbers. Reading '1 to 24' as a
    number would invent a precision nobody recorded.
    """
    db.set_session_odors(
        session_id=db.seeded.session_id,
        odors=[{"odor_id": 21, "goal_ppm": "1 to 24"}, {"odor_id": 0, "goal_ppm": "na"}],
    )

    stored = db.session_odors.loc[db.seeded.session_id, "goal_ppm"]

    assert set(stored) == {"1 to 24", "na"}


def test_a_mixing_date_is_stored_as_a_date(db):
    """The workbook holds a datetime at midnight; the time is not real."""
    db.set_session_odors(
        session_id=db.seeded.session_id,
        odors=[{"odor_id": 1, "made_on": datetime(2026, 7, 6, 0, 0)}],
    )

    raw = db.con.execute("SELECT made_on FROM session_odors;").fetchone()[0]

    assert raw == "2026-07-06"


def test_setting_a_panel_replaces_the_previous_one(db):
    """
    A panel is written down as one table and is only true as a whole. Adding
    row by row would leave a half-updated panel looking complete.
    """
    db.set_session_odors(session_id=db.seeded.session_id, odors=PANEL)
    db.set_session_odors(
        session_id=db.seeded.session_id, odors=[{"odor_id": 2, "vial": 1}]
    )

    stored = db.session_odors.loc[db.seeded.session_id]

    assert list(stored.index) == [2]


def test_the_same_odor_cannot_be_in_two_vials(db):
    with pytest.raises(ValueError, match="more than once"):
        db.set_session_odors(
            session_id=db.seeded.session_id,
            odors=[{"odor_id": 1, "vial": 1}, {"odor_id": 1, "vial": 2}],
        )


def test_an_unknown_odor_is_refused(db):
    """`odors` is the global list; a panel cannot invent one."""
    with pytest.raises(sqlite3.IntegrityError):
        db.set_session_odors(
            session_id=db.seeded.session_id, odors=[{"odor_id": 9999, "vial": 1}]
        )


def test_a_misspelt_field_is_refused(db):
    """Otherwise it would be dropped in silence and nobody would miss it."""
    with pytest.raises(ValueError, match="not things recorded about a vial"):
        db.set_session_odors(
            session_id=db.seeded.session_id, odors=[{"odor_id": 1, "vail": 1}]
        )


def test_an_entry_without_an_odor_is_refused(db):
    with pytest.raises(ValueError, match="needs an 'odor_id'"):
        db.set_session_odors(
            session_id=db.seeded.session_id, odors=[{"vial": 1, "sccm": 200}]
        )


def test_a_session_that_does_not_exist_is_refused(db):
    with pytest.raises(ValueError, match="no session 9999"):
        db.set_session_odors(session_id=9999, odors=PANEL)


def test_the_panel_is_recorded_as_provenance(db):
    """
    The panel as passed is what the call is answerable for. Nothing is written
    to `call_output`: the result is whatever was passed in, and no later call
    reads it back.
    """
    db.set_session_odors(session_id=db.seeded.session_id, odors=PANEL)

    call = db.latest_calls("set_session_odors").iloc[0]

    assert call["input_session_id"] == db.seeded.session_id
    assert [entry["odor_id"] for entry in call["input_odors"]] == [1, 17, 0]
    assert not call["call_flag"]
    assert not [name for name in call.index if name.startswith("output_")]
