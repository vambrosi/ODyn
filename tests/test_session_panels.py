"""
Odor panels: what sits in each vial, and which session ran which set.

A panel is a fixed recipe -- which odor sits in which vial, at what
concentration and flow -- so it is registered once and named by every session
that runs it. The mixing date is the part that changes: the vials are remade
every few weeks.
"""

import sqlite3

import pandas as pd
import pytest

from odyn import Database

from helpers import seed_rows

PANEL = [
    {
        "vial_position": 1,
        "odor_id": 1,
        "odor_sccm": 200.0,
        "total_sccm": 1500.0,
        "total_volume_ml": 5.0,
        "solvent_volume_ml": 4.65,
        "components": [{"odor_id": 1, "target_ppm": 0.3, "liquid_ul": 346.8}],
    },
    {
        "vial_position": 2,
        "odor_id": 17,
        "components": [
            {"odor_id": 4, "target_ppm": 44.2, "percent_vv": 3.0},
            {"odor_id": 10, "target_ppm": 13.3, "percent_vv": 2.0},
        ],
    },
    {"vial_position": 3, "odor_id": 0},
]


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path, project="test")
    database.seeded = seed_rows(database)
    database.add_panel(panel_name="panel_16", vials=PANEL)

    return database


# --------------------------------------------------------------------------- #
# The recipe
# --------------------------------------------------------------------------- #


def test_a_vial_resolves_to_the_odor_it_delivers(db):
    """This is what the panel is for: a vial number means an odor."""
    row = db.panel_vials.loc[("panel_16", 1)]

    assert row["odor_id"] == 1
    assert row["odor_name"] == "eugenol"
    assert row["odor_sccm"] == 200.0


def test_a_mix_points_at_the_mix_not_at_its_parts(db):
    """Vial 2 delivers 'alpha', which is its own odor, made of two others."""
    assert db.panel_vials.loc[("panel_16", 2), "odor_name"] == "alpha"

    parts = db.vial_components.loc[("panel_16", 2)]

    assert sorted(parts.index) == [4, 10]
    assert parts.loc[4, "target_ppm"] == 44.2


def test_a_vial_may_have_no_components(db):
    """Mineral oil is the blank: a vial with nothing dissolved in it."""
    assert db.panel_vials.loc[("panel_16", 3), "odor_name"] == "mineral oil"
    assert ("panel_16", 3) not in db.vial_components.index.droplevel("odor_id")


def test_the_panel_gets_an_id_of_its_own(db):
    """Named for people, keyed by id so the panel can be referred to and grown."""
    panel_id = db.add_panel(panel_name="panel_16", vials=PANEL)

    assert db.panels.loc[panel_id, "panel_name"] == "panel_16"
    assert db.panel_vials.loc["panel_16", "panel_id"].eq(panel_id).all()


def test_re_registering_keeps_the_id(db):
    """Sessions point at the id, so a corrected recipe must not orphan them."""
    first = db.panels.index[0]
    again = db.add_panel(panel_name="panel_16", vials=PANEL)

    assert again == first
    assert len(db.panels) == 1


def test_flows_left_out_stay_empty(db):
    """Most of the recipe is optional, so a partial panel is still a panel."""
    row = db.panel_vials.loc[("panel_16", 2)]

    assert pd.isna(row["odor_sccm"])


def test_registering_the_panel_again_replaces_its_vials(db):
    """A corrected recipe is one call, not a merge of old and new rows."""
    db.add_panel(
        panel_name="panel_16",
        vials=[{"vial_position": 1, "odor_id": 2}],
    )

    assert len(db.panel_vials.loc["panel_16"]) == 1
    assert db.panel_vials.loc[("panel_16", 1), "odor_name"] == "methyl salicylate"

    # The components of the vials that went away go with them.
    assert db.vial_components.empty


def test_a_recipe_a_session_ran_cannot_be_rewritten(db):
    """
    Once trials exist, the recipe is what they mean. Editing it in place would
    silently restate what the animal was given, so a change needs a new name.
    """
    db.set_session_panel(session_id=db.seeded.session_id, panel_name="panel_16")

    with pytest.raises(ValueError, match="cannot be changed"):
        db.add_panel(
            panel_name="panel_16",
            vials=[{"vial_position": 1, "odor_id": 2}],
        )

    assert db.panel_vials.loc[("panel_16", 1), "odor_name"] == "eugenol"


def test_re_registering_an_unchanged_panel_does_nothing(db):
    """So the importer can be run again over a workbook that has not moved."""
    db.set_session_panel(session_id=db.seeded.session_id, panel_name="panel_16")

    db.add_panel(panel_name="panel_16", vials=PANEL)

    assert len(db.panel_vials.loc["panel_16"]) == len(PANEL)
    assert len(db.session_vials.loc[db.seeded.session_id]) == len(PANEL)


def test_an_unknown_odor_is_refused(db):
    """A vial has to deliver something the database can name."""
    with pytest.raises(sqlite3.IntegrityError):
        db.add_panel(
            panel_name="panel_bad",
            vials=[{"vial_position": 1, "odor_id": 9999}],
        )


# --------------------------------------------------------------------------- #
# Which session ran it
# --------------------------------------------------------------------------- #


def test_a_session_records_its_panel(db):
    db.set_session_panel(
        session_id=db.seeded.session_id, panel_name="panel_16", made_on="2026-07-06"
    )

    assert db.session_panels.loc[db.seeded.session_id, "panel_name"] == "panel_16"


def test_one_mixing_date_covers_every_vial(db):
    """The usual case: a whole rack is remade at once."""
    db.set_session_panel(
        session_id=db.seeded.session_id, panel_name="panel_16", made_on="2026-07-06"
    )

    mixed = db.session_vials.loc[db.seeded.session_id]

    assert sorted(mixed.index) == [1, 2, 3]
    assert set(mixed["made_on"].dt.date.astype(str)) == {"2026-07-06"}


def test_a_vial_refilled_on_its_own_keeps_its_own_date(db):
    """
    The reason the date is per vial. Flattening this session to one date would
    put the wrong mixing against two thirds of its trials.
    """
    db.set_session_panel(
        session_id=db.seeded.session_id,
        panel_name="panel_16",
        made_on="2026-07-06",
        vial_dates={2: "2026-07-21"},
    )

    mixed = db.session_vials.loc[db.seeded.session_id, "made_on"]

    assert mixed.loc[1].date().isoformat() == "2026-07-06"
    assert mixed.loc[2].date().isoformat() == "2026-07-21"
    assert mixed.loc[3].date().isoformat() == "2026-07-06"


def test_dates_may_be_given_for_some_vials_only(db):
    """`made_on` is not required when the per-vial dates carry the answer."""
    db.set_session_panel(
        session_id=db.seeded.session_id,
        panel_name="panel_16",
        vial_dates={2: "2026-07-21"},
    )

    mixed = db.session_vials.loc[db.seeded.session_id, "made_on"]

    assert pd.isna(mixed.loc[1])
    assert mixed.loc[2].date().isoformat() == "2026-07-21"


def test_the_mixing_date_is_optional(db):
    """The panel may be known when the day it was mixed is not."""
    db.set_session_panel(session_id=db.seeded.session_id, panel_name="panel_16")

    mixed = db.session_vials.loc[db.seeded.session_id]

    # A row per vial even so: not recorded and recorded-as-unknown differ.
    assert len(mixed) == 3
    assert mixed["made_on"].isna().all()


def test_a_vial_the_panel_does_not_have_is_refused(db):
    """Otherwise a typo silently dates nothing."""
    with pytest.raises(ValueError, match="no vial"):
        db.set_session_panel(
            session_id=db.seeded.session_id,
            panel_name="panel_16",
            vial_dates={9: "2026-07-21"},
        )


def test_setting_it_again_replaces_it(db):
    """A session runs one panel, so this corrects rather than accumulates."""
    db.add_panel(panel_name="panel_12", vials=[{"vial_position": 1, "odor_id": 0}])

    db.set_session_panel(
        session_id=db.seeded.session_id, panel_name="panel_12", made_on="2026-06-19"
    )
    db.set_session_panel(
        session_id=db.seeded.session_id, panel_name="panel_16", made_on="2026-07-06"
    )

    assert len(db.session_panels) == 1
    assert db.session_panels.loc[db.seeded.session_id, "panel_name"] == "panel_16"

    # The dates of the panel that went away go with it.
    mixed = db.session_vials.loc[db.seeded.session_id]

    assert len(mixed) == 3
    assert set(mixed["made_on"].dt.date.astype(str)) == {"2026-07-06"}


def test_a_session_that_does_not_exist_is_refused(db):
    with pytest.raises(ValueError, match="no session 9999"):
        db.set_session_panel(session_id=9999, panel_name="panel_16")


def test_an_unregistered_panel_is_refused(db):
    """
    The recipe is in the database, so naming one it does not hold is a typo.
    Without this a session could point at a panel nobody can look up.
    """
    with pytest.raises(ValueError, match="no panel"):
        db.set_session_panel(
            session_id=db.seeded.session_id, panel_name="panel_typo"
        )


def test_a_bad_date_is_refused(db):
    """The column only holds dates, so a typo cannot pass for one."""
    with pytest.raises(sqlite3.IntegrityError):
        db.set_session_panel(
            session_id=db.seeded.session_id,
            panel_name="panel_16",
            made_on="2026-07-32",
        )
