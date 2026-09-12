"""
Shared setup for the database tests.

`seed_rows` builds the minimal row skeleton a test needs before it can call
anything: a session, one experiment, its acquisitions, and a group holding it.
It exists so a schema change breaks in one place instead of in every test file
that happens to need a database. Rows are built as dicts through `_db_insert`,
the same helper ingestion uses, so a renamed column is one key here rather than
a literal string in several files.

Tests whose subject is the schema (`test_constraints.py`, `test_migration.py`)
do not use this: they keep their own literal SQL, because a helper that hides
the column names would hide what they are checking.
"""

from __future__ import annotations

from dataclasses import dataclass

from odyn.database import _db_insert

EXP_DIR = "20260101/m001/e1"
STEM = "20260101_m001_e1"

# Over 4 frames on purpose: tifffile folds a leading axis of 4 or less into a
# single multi-sample page, which is not what a real recording looks like.
FRAMES, HEIGHT, WIDTH = 6, 64, 80


@dataclass(frozen=True)
class Seeded:
    """The ids `seed_rows` created, so a test can name the ones it needs."""

    session_id: int
    exp_id: int
    acq_ids: list[int]
    group_id: int


def seed_rows(
    db,
    *,
    acquisitions: int = 3,
    mouse_id: str = "m001",
    session_date: str = "2026-01-01",
    session_path: str = "20260101/m001",
    exp_name: str = STEM,
    exp_dir: str = EXP_DIR,
    exp_start: str = "2026-01-01 10:00:00",
    height_px: int = HEIGHT,
    width_px: int = WIDTH,
    frame_count: int = FRAMES,
    frame_rate: float = 14.0,
    group_id: int = 1,
) -> Seeded:
    """One session, one experiment, its acquisitions, and a group holding it."""

    with db.con as con:
        cur = con.cursor()

        session_id = _db_insert(cur, "sessions", {
            "mouse_id": mouse_id,
            "session_date": session_date,
            "session_path": session_path,
        })

        exp_id = _db_insert(cur, "experiments", {
            "session_id": session_id,
            "exp_name": exp_name,
            "exp_type": "loop",
            "exp_start": exp_start,
            "height_px": height_px,
            "width_px": width_px,
            "height_um": float(height_px),
            "width_um": float(width_px),
            "frame_count": frame_count,
            "frame_rate": frame_rate,
        })

        acq_ids = [
            _db_insert(cur, "acquisitions", {
                "exp_id": exp_id,
                "acq_start": f"2026-01-01 10:0{index}:00",
                "raw_path": f"{exp_dir}/raw/{exp_name}_{index + 1:05d}.tif",
            })
            for index in range(acquisitions)
        ]

        _db_insert(cur, "groups", {"group_id": group_id})
        _db_insert(cur, "group_experiments",
                   {"group_id": group_id, "exp_id": exp_id})

    return Seeded(session_id, exp_id, acq_ids, group_id)


def add_group(db, *, group_id: int, exp_ids: list[int]):
    """A second group over experiments that already exist."""

    with db.con as con:
        cur = con.cursor()
        _db_insert(cur, "groups", {"group_id": group_id})

        for exp_id in exp_ids:
            _db_insert(cur, "group_experiments",
                       {"group_id": group_id, "exp_id": exp_id})

    return db.groups[group_id]
