"""
Schema-integrity checks for create.sql.

The composite foreign keys must keep a trial's program and acquisition tied to
the same experiment (the "diamond"), and a consistent experiment ->
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
    """A trial mixing a program and acquisition from different experiments must
    violate the composite foreign keys."""
    con = _fresh_db()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            con.executescript(DIAMOND_SQL.read_text())
    finally:
        con.close()


def test_valid_chain_inserts():
    """A consistent experiment -> program/acquisition -> trial -> event chain
    inserts without error."""
    con = _fresh_db()
    try:
        con.executescript(INSERTION_SQL.read_text())
    finally:
        con.close()
