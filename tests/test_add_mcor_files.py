"""
`Group.add_mcor_files` registers motion corrected files made outside the
database.

The risk it guards against is a wrong file being attached to an acquisition,
which would quietly corrupt every analysis downstream.
"""

import sqlite3

import numpy as np
import pytest
import tifffile

from odyn import Database
from odyn.groups import MCOR_LAYOUT, McorFlag, McorSource, _tiff_shape

EXP_DIR = "20260101/m001/e1"
STEM = "20260101_m001_e1"

# Over 4 frames on purpose: tifffile folds a leading axis of 4 or less into a
# single multi-sample page, which is not what a real recording looks like.
FRAMES, HEIGHT, WIDTH = 6, 64, 80


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def build(tmp_path, acquisitions=3):
    """A database with one experiment, its acquisitions, and a group."""
    db = Database(tmp_path)

    with db.con as con:
        con.execute(
            """
            INSERT INTO experiments
                ( exp_id, exp_name, exp_type, exp_start, mouse_id
                , height_px, width_px, height_um, width_um, frame_count, frame_rate
                , laser_power_920, laser_power_1040, loop_acq_interval_s
                ) VALUES ( 1, ?, 'loop', '2026-01-01 10:00:00', 'm001'
                         , ?, ?, ?, ?, ?
                         , 14.0, 10, 0, 10.0);
            """,
            [STEM, HEIGHT, WIDTH, float(HEIGHT), float(WIDTH), FRAMES],
        )

        for index in range(acquisitions):
            con.execute(
                """
                INSERT INTO acquisitions (acq_id, exp_id, acq_start, raw_path)
                    VALUES (?, 1, ?, ?);
                """,
                [
                    index + 1,
                    f"2026-01-01 10:0{index}:00",
                    f"{EXP_DIR}/raw/{STEM}_{index + 1:05d}.tif",
                ],
            )

        con.execute("INSERT INTO groups (group_id) VALUES (1);")
        con.execute("INSERT INTO group_experiments (group_id, exp_id) VALUES (1, 1);")

    return db, db.groups[1]


def write_mcor(
    tmp_path,
    index,
    *,
    source=McorSource.CAIMAN,
    frames=FRAMES,
    height=HEIGHT,
    width=WIDTH,
    dtype=np.float32,
    **kwargs,
):
    """Put a plausible motion corrected file where the group will look."""
    folder, suffix = MCOR_LAYOUT[source]
    path = tmp_path / EXP_DIR / folder / f"{STEM}_{index:05d}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(path, np.zeros((frames, height, width), dtype), **kwargs)

    return path


def stored(db):
    return {
        row[0]: (row[1], row[2])
        for row in db.con.execute("SELECT acq_id, mcor_path, source FROM mcor_files;")
    }


# --------------------------------------------------------------------------- #
# Reading TIFF headers
# --------------------------------------------------------------------------- #


def test_shape_and_count_from_a_big_file(tmp_path):
    """Frames here are far larger than the TIFF overhead, so size alone works."""
    path = write_mcor(tmp_path, 1, height=256, width=256, frames=6)

    assert _tiff_shape(path, 6) == ((256, 256), 6)


def test_count_is_right_even_when_the_estimate_is_not(tmp_path):
    """
    Asking for the wrong number forces the exact page walk, which is also what
    rescues files the size estimate cannot handle.
    """
    path = write_mcor(tmp_path, 1, height=256, width=256, frames=6)

    assert _tiff_shape(path, 999)[1] == 6


def test_tiny_frames_still_report_the_right_count(tmp_path):
    """With small frames the per-page overhead can exceed a whole frame."""
    path = write_mcor(tmp_path, 1, height=8, width=8, frames=6)

    assert _tiff_shape(path, 6) == ((8, 8), 6)


def test_size_comes_off_the_open_file(tmp_path):
    """
    `_tiff_shape` reads the size from tifffile's own handle so that checking a
    file costs one round trip instead of two. `filehandle.size` is not part of
    tifffile's documented API, and if it were renamed or changed meaning the
    frame count would come out wrong rather than raise, so pin it here.
    """
    path = write_mcor(tmp_path, 1)

    with tifffile.TiffFile(path) as tif:
        assert tif.filehandle.size == path.stat().st_size


def test_a_missing_file_raises_file_not_found(tmp_path):
    """
    Missing and unreadable are reported separately, and this is what tells
    them apart now that nothing lists the folder first.
    """
    with pytest.raises(FileNotFoundError):
        _tiff_shape(tmp_path / "not_here.tif", 6)


# --------------------------------------------------------------------------- #
# Adding files
# --------------------------------------------------------------------------- #


def test_files_are_found_beside_the_raw_files(tmp_path):
    db, group = build(tmp_path)
    for index in (1, 2, 3):
        write_mcor(tmp_path, index)

    group.add_mcor_files()
    rows = stored(db)

    assert set(rows) == {1, 2, 3}
    assert rows[1] == (f"{EXP_DIR}/processed/mcor/{STEM}_00001_mcor.tif", "caiman")


def test_paths_are_stored_posix_style(tmp_path):
    """The database is read from several machines, so separators must not vary."""
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1)

    group.add_mcor_files()

    assert "\\" not in stored(db)[1][0]


def test_looking_for_the_wrong_source_finds_nothing(tmp_path):
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1, source=McorSource.CAIMAN)

    group.add_mcor_files(source="patchwarp")

    assert stored(db) == {}


def test_unknown_source_is_refused(tmp_path):
    _, group = build(tmp_path, acquisitions=1)

    with pytest.raises(ValueError, match="source must be one of"):
        group.add_mcor_files(source="normcorre")


# --------------------------------------------------------------------------- #
# Refusing bad files
# --------------------------------------------------------------------------- #


def test_a_missing_file_does_not_stop_the_others(tmp_path):
    db, group = build(tmp_path)
    write_mcor(tmp_path, 1)
    write_mcor(tmp_path, 3)

    group.add_mcor_files()

    assert set(stored(db)) == {1, 3}
    assert group.method_calls.iloc[-1]["call_flag"] & McorFlag.FILE_NOT_FOUND


def test_wrong_frame_size_is_refused(tmp_path):
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1, height=HEIGHT + 8)

    group.add_mcor_files()

    assert stored(db) == {}
    assert group.method_calls.iloc[-1]["call_flag"] & McorFlag.WRONG_SHAPE


def test_wrong_frame_count_is_refused(tmp_path):
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1, frames=FRAMES + 3)

    group.add_mcor_files()

    assert stored(db) == {}
    assert group.method_calls.iloc[-1]["call_flag"] & McorFlag.WRONG_FRAME_COUNT


# --------------------------------------------------------------------------- #
# One run per group
# --------------------------------------------------------------------------- #


def test_an_empty_group_accepts_a_partial_set(tmp_path):
    """Partial is fine, as long as everything came from the same run."""
    db, group = build(tmp_path)
    write_mcor(tmp_path, 1)
    write_mcor(tmp_path, 3)

    group.add_mcor_files()

    assert set(stored(db)) == {1, 3}


def test_a_group_with_files_refuses_to_take_more(tmp_path):
    """Half from one run and half from another cannot be compared."""
    db, group = build(tmp_path)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()

    write_mcor(tmp_path, 2)
    group.add_mcor_files()

    assert set(stored(db)) == {1}
    assert group.method_calls.iloc[-1]["call_flag"] & McorFlag.ALREADY_HAS_FILES


def test_overwrite_replaces_every_file_in_the_group(tmp_path):
    db, group = build(tmp_path)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()
    first_call = db.con.execute("SELECT last_updated_by FROM mcor_files;").fetchone()[0]

    write_mcor(tmp_path, 2)
    write_mcor(tmp_path, 3)
    group.add_mcor_files(overwrite=True)

    owners = {
        row[0] for row in db.con.execute("SELECT last_updated_by FROM mcor_files;")
    }

    assert set(stored(db)) == {1, 2, 3}
    assert owners != {first_call} and len(owners) == 1
    assert group.method_calls.iloc[-1]["call_flag"] & McorFlag.REPLACED_EXISTING


def test_overwrite_clears_approval(tmp_path):
    """A replaced file has not been reviewed again."""
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()

    with db.con as con:
        con.execute("UPDATE mcor_files SET approved = TRUE;")

    group.add_mcor_files(overwrite=True)

    assert db.con.execute("SELECT approved FROM mcor_files;").fetchone()[0] == 0


def test_overwrite_keeps_what_it_has_if_it_finds_nothing(tmp_path):
    """Never drop a usable set to put nothing in its place."""
    db, group = build(tmp_path, acquisitions=1)
    path = write_mcor(tmp_path, 1)
    group.add_mcor_files()

    path.unlink()
    group.add_mcor_files(overwrite=True)

    assert set(stored(db)) == {1}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_the_call_owns_the_files_it_added(tmp_path):
    """`last_updated_by` is what tells imported files from computed ones."""
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1)

    group.add_mcor_files()

    call_id = db.con.execute("SELECT last_updated_by FROM mcor_files;").fetchone()[0]
    method = db.con.execute(
        "SELECT method_name FROM method_calls WHERE method_call_id = ?;", [call_id]
    ).fetchone()[0]

    assert method == "Group.add_mcor_files"


def test_output_records_what_happened(tmp_path):
    db, group = build(tmp_path)
    write_mcor(tmp_path, 1)
    write_mcor(tmp_path, 2, frames=FRAMES + 1)

    group.add_mcor_files()
    output = group.method_calls.iloc[-1]["call_output"]

    assert output["source"] == "caiman"
    assert output["added"] == 1
    assert output["file_not_found"] == 1
    assert output["wrong_frame_count"] == 1
    assert output["added_acq_ids"] == [1]


def test_nothing_to_add_leaves_the_table_empty(tmp_path):
    db, group = build(tmp_path, acquisitions=1)

    group.add_mcor_files()

    assert stored(db) == {}
    assert db.con.execute("SELECT count(*) FROM mcor_files;").fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# Groups that share an experiment
# --------------------------------------------------------------------------- #


def mcor_group(db, acq_id=1):
    """Which group motion corrected this acquisition, the way the code asks."""
    row = db.con.execute(
        """
        SELECT mc.group_id
            FROM mcor_files   AS m
            JOIN method_calls AS mc ON mc.method_call_id = m.last_updated_by
            WHERE m.acq_id = ?;
        """,
        [acq_id],
    ).fetchone()

    return row["group_id"]


def share_experiment(db):
    """A second group over the same experiment, as happens in the real data."""
    with db.con as con:
        con.execute("INSERT INTO groups (group_id) VALUES (2);")
        con.execute("INSERT INTO group_experiments (group_id, exp_id) VALUES (2, 1);")

    return db.groups[2]


def test_replacing_files_another_group_also_uses_is_flagged(tmp_path):
    """
    Most groups share an experiment with a neighbour, so overwriting reaches
    further than the group you called it on.
    """
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()

    other = share_experiment(db)
    group.add_mcor_files(overwrite=True)

    assert group.method_calls.iloc[-1]["call_flag"] & McorFlag.SHARED_WITH_OTHER_GROUPS
    assert set(other.mcor_files.index) == {1}


def test_an_analysis_group_spanning_runs_is_not_an_error(tmp_path):
    """
    Each experiment is corrected by one group, so a group made for analysis
    routinely holds experiments from different runs. That is normal, and
    replacing one of them must not be refused because of it.
    """
    db, group = build(tmp_path, acquisitions=2)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()

    other = share_experiment(db)
    write_mcor(tmp_path, 2)
    group.add_mcor_files(overwrite=True)

    assert set(stored(db)) == {1, 2}
    assert set(other.mcor_files.index) == {1, 2}


def test_replacing_files_another_group_mcor_is_refused(tmp_path):
    """
    The group that produced the files is recorded through `last_updated_by`,
    so replacing them from anywhere else is a mistake, not a choice: overwrite
    says "replace this group's files", not "take another group's".
    """
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()

    other = share_experiment(db)

    with pytest.raises(RuntimeError):
        other.add_mcor_files(overwrite=True)

    # Flagged even though it was refused, and the files did not change hands
    assert other.method_calls.iloc[-1]["call_flag"] & McorFlag.OWNED_BY_OTHER_GROUP
    assert mcor_group(db) == group.group_id


def test_the_mcor_group_can_be_changed_on_purpose(tmp_path):
    """
    The way out when the first motion correction went to the wrong group. The
    flag is set either way, so a takeover stays visible in method_calls.
    """
    db, group = build(tmp_path, acquisitions=1)
    write_mcor(tmp_path, 1)
    group.add_mcor_files()

    other = share_experiment(db)
    other.add_mcor_files(overwrite=True, _change_mcor_group=True)

    assert other.method_calls.iloc[-1]["call_flag"] & McorFlag.OWNED_BY_OTHER_GROUP
    assert set(other.mcor_files.index) == {1}
    assert mcor_group(db) == other.group_id
