"""
Schema-integrity checks for create.sql.

The composite foreign keys must keep a trial's program and acquisition tied to
the same loop (the "diamond"), and a consistent loop ->
program/acquisition -> trial -> event chain must insert cleanly.
"""

import sqlite3

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "odyn"
CREATE_SQL = PACKAGE / "create.sql"
DIAMOND_SQL = Path(__file__).parent / "diamond.sql"
INSERTION_SQL = Path(__file__).parent / "insertion.sql"


def _fresh_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.executescript(CREATE_SQL.read_text())
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def test_diamond_constraint_rejected():
    """A trial mixing a program and acquisition from different loops must
    violate the composite foreign keys."""
    con = _fresh_db()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.executescript(DIAMOND_SQL.read_text())
    finally:
        con.close()


def test_valid_chain_inserts():
    """A consistent loop -> program/acquisition -> trial -> event chain
    inserts without error."""
    con = _fresh_db()
    try:
        con.executescript(INSERTION_SQL.read_text())
    finally:
        con.close()


def _with_two_sessions() -> sqlite3.Connection:
    """The valid chain, plus a second session (2) of the same mouse."""
    con = _fresh_db()
    con.executescript(INSERTION_SQL.read_text())
    con.execute(
        "INSERT INTO sessions (session_id, mouse_id, session_date) "
        "VALUES (2, 1, '2025-02-02');"
    )

    return con


def test_a_zstack_is_only_for_a_loop_of_its_own_session():
    con = _with_two_sessions()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE zstacks SET session_id = 2 WHERE zstack_id = 1;")

    con.close()


def test_a_sync_block_is_an_acquisitions_or_a_grabs_not_both():
    con = _with_two_sessions()
    con.execute("UPDATE sync_blocks SET grab_id = NULL WHERE grab_id = 1;")

    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        con.execute("UPDATE sync_blocks SET grab_id = 1 WHERE acq_id = 1;")

    con.close()


def test_a_loops_stream_is_its_sessions():
    con = _with_two_sessions()

    with pytest.raises(sqlite3.IntegrityError):
        con.execute("UPDATE streams SET session_id = 2 WHERE loop_id = 1;")

    con.close()
