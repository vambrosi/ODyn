"""
`Database.set_mouse`: the animal behind the recordings.

The details arrive from a colony spreadsheet, late and in pieces, so the point
here is that a second call fills gaps without erasing what a first one wrote.
The line is the awkward part: it is a cross of several mutations and the animal
has a separate genotype for each, so it is rows in `mouse_lines` rather than two
lists that have to stay the same length.
"""

import sqlite3

import pandas as pd
import pytest

from odyn import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path, project="test")


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_a_mouse_is_recorded_by_its_number(db):
    db.set_mouse(mouse_id=442, sex="F", dob="2026-01-22", sensor="GCaMP8s")

    row = db.mice.loc[442]

    assert row["mouse_sex"] == "F"
    assert row["mouse_dob"].date().isoformat() == "2026-01-22"
    assert row["sensor"] == "GCaMP8s"


def test_the_name_from_the_folders_is_accepted(db):
    """Recordings are filed under 'm442', so that is what callers have."""
    assert db.set_mouse(mouse_id="m442") == 442
    assert 442 in db.mice.index


def test_only_the_number_is_required(db):
    """Everything else is written down by a person and may never arrive."""
    db.set_mouse(mouse_id=442)

    assert pd.isna(db.mice.loc[442, "mouse_sex"])
    assert db.mouse_lines.empty


def test_sex_is_read_in_any_case(db):
    """The colony sheets write it as 'm' and 'f'."""
    db.set_mouse(mouse_id=442, sex="f")

    assert db.mice.loc[442, "mouse_sex"] == "F"


def test_a_sex_that_is_not_one_is_refused(db):
    with pytest.raises(ValueError, match="'M' or 'F'"):
        db.set_mouse(mouse_id=442, sex="female-ish")


def test_a_bad_date_of_birth_is_refused(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.set_mouse(mouse_id=442, dob="2026-01-32")


# --------------------------------------------------------------------------- #
# Filling in over time
# --------------------------------------------------------------------------- #


def test_a_later_call_fills_gaps_without_erasing(db):
    """Sex comes from one sheet and the sensor from another person's notes."""
    db.set_mouse(mouse_id=442, sex="F")
    db.set_mouse(mouse_id=442, sensor="GCaMP8s")

    row = db.mice.loc[442]

    assert row["mouse_sex"] == "F"
    assert row["sensor"] == "GCaMP8s"


def test_a_later_call_can_correct_a_field(db):
    db.set_mouse(mouse_id=442, sex="M")
    db.set_mouse(mouse_id=442, sex="F")

    assert db.mice.loc[442, "mouse_sex"] == "F"
    assert len(db.mice) == 1


# --------------------------------------------------------------------------- #
# The line
# --------------------------------------------------------------------------- #


def test_the_line_is_one_row_per_mutation(db):
    db.set_mouse(mouse_id=442, lines={"DAT-Cre": "het", "TIGRE": "het"})

    carried = db.mouse_lines.loc[442]

    assert sorted(carried.index) == ["DAT-Cre", "TIGRE"]
    assert carried.loc["TIGRE", "genotype"] == "het"


def test_a_mutation_may_be_carried_without_being_genotyped(db):
    """Plenty of animals are known by line and genotyped later, or never."""
    db.set_mouse(mouse_id=462, lines={"TH-Cre": None, "TIGRE": None})

    assert len(db.mouse_lines.loc[462]) == 2
    assert db.mouse_lines.loc[(462, "TH-Cre"), "genotype"] is None


def test_genotype_spellings_are_read(db):
    db.set_mouse(
        mouse_id=442,
        lines={"DAT-Cre": "Het", "TIGRE": "homozygous", "Thy1-GCaMP": "WT"},
    )

    stored = db.mouse_lines.loc[442, "genotype"].to_dict()

    assert stored == {"DAT-Cre": "het", "TIGRE": "hom", "Thy1-GCaMP": "wt"}


def test_a_genotype_that_means_nothing_is_refused(db):
    """Guessing would put an animal in the wrong group for every analysis."""
    with pytest.raises(ValueError, match="hemi"):
        db.set_mouse(mouse_id=442, lines={"TIGRE": "hemi"})


def test_the_line_is_replaced_as_a_whole(db):
    """A correction usually rewrites more than one of them."""
    db.set_mouse(mouse_id=442, lines={"TH-Cre": "het", "TIGRE": "het"})
    db.set_mouse(mouse_id=442, lines={"DAT-Cre": "het", "TIGRE": "hom"})

    carried = db.mouse_lines.loc[442]

    assert sorted(carried.index) == ["DAT-Cre", "TIGRE"]
    assert carried.loc["TIGRE", "genotype"] == "hom"


def test_leaving_the_lines_out_keeps_them(db):
    """A call about the sensor should not quietly drop the line."""
    db.set_mouse(mouse_id=442, lines={"TIGRE": "het"})
    db.set_mouse(mouse_id=442, sensor="GCaMP8s")

    assert len(db.mouse_lines.loc[442]) == 1


def test_an_empty_mapping_clears_the_line(db):
    """That is how a wild type animal is recorded once a line was wrong."""
    db.set_mouse(mouse_id=442, lines={"TIGRE": "het"})
    db.set_mouse(mouse_id=442, lines={})

    assert db.mouse_lines.empty


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_the_call_is_recorded(db):
    """Who said an animal is this line, and when, is worth keeping."""
    db.set_mouse(mouse_id=442, lines={"TIGRE": "het"})

    row = db.con.execute(
        """
        SELECT user, parameters_used FROM method_calls
            WHERE method_name = 'Database.set_mouse';
        """
    ).fetchone()

    assert row["user"]
    assert "TIGRE" in row["parameters_used"]
