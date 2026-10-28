"""
The registries the GUI fills before any recording: users, mice, sessions,
odors, panels and vials.
"""

import pytest

from odyn import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path, can_create=True, user="tester")


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #


def test_users_are_picked_by_name(db):
    db.add_user(user_name="erin", display_name="Erin")
    db.add_user(user_name="dave")

    users = db.users.set_index("user_name")
    assert users.loc["erin", "display_name"] == "Erin"
    assert users.loc["dave", "display_name"] == "dave"
    assert not users["retired"].any()

    db.retire_user(user_name="dave")
    assert bool(db.users.set_index("user_name").loc["dave", "retired"])

    with pytest.raises(ValueError, match="No user"):
        db.retire_user(user_name="nobody")


def test_the_database_user_authors_every_call(db):
    db.add_user(user_name="x")
    assert set(db.method_calls["user"]) == {"tester"}


# --------------------------------------------------------------------------- #
# Mice and sessions
# --------------------------------------------------------------------------- #


def test_mouse_details_accumulate_across_calls(db):
    db.add_line(line_name="Line-A", jax_strain_number="000001")
    assert db.lines.loc["Line-A", "jax_strain_number"] == "000001"

    assert db.set_mouse(mouse_id="m442", sex="f") == 442

    # a line is matched in any case, and stored as the registry spells it
    db.set_mouse(
        mouse_id=442, dob="2026-01-22", lines={"dat-cre": "Het", "TIGRE2": "na"}
    )
    db.set_mouse(mouse_id="m442", lines=None)

    mouse = db.mice.loc[442]
    assert mouse["mouse_sex"] == "F"
    assert mouse["mouse_dob"] == "2026-01-22"

    lines = db.mouse_lines.loc[[442]].set_index("line_name")["genotype"]
    assert lines.to_dict() == {"DAT-Cre": "het", "TIGRE2": "na"}

    # `lines={}` clears, `lines=None` leaves alone
    db.set_mouse(mouse_id=442, lines={})
    assert 442 not in db.mouse_lines.index


def test_a_mouse_carries_only_registered_lines(db):
    # the lines are seeded when the database is made
    assert {"Wild Type", "TH-Cre", "Thy1-GCaMP"} <= set(db.lines.index)

    with pytest.raises(ValueError, match="Unknown line 'Th-Cre-2'"):
        db.set_mouse(mouse_id=1, lines={"Th-Cre-2": "het"})

    with pytest.raises(ValueError, match="needs a name"):
        db.add_line(line_name=" ")


def test_viruses_are_a_registry_too(db):
    # seeded when the database is made, in alphabetical order
    assert "pAAV9-CAG-dLight1.3b" in db.viruses.index
    assert len(db.viruses) == 21

    # matched in any case, stored as registered, replaced as a whole
    db.set_mouse(mouse_id=5, viruses=["paav9-cag-dlight1.3b", "pAAVrg-EF1a-Flpo"])
    assert sorted(db.mouse_viruses.loc[[5], "virus_name"]) == [
        "pAAV9-CAG-dLight1.3b",
        "pAAVrg-EF1a-Flpo",
    ]
    db.set_mouse(mouse_id=5, sex="M")  # `viruses=None` leaves them alone
    assert len(db.mouse_viruses.loc[[5]]) == 2

    db.set_mouse(mouse_id=5, viruses=[])
    assert 5 not in db.mouse_viruses.index

    with pytest.raises(ValueError, match="Unknown virus 'AAV-nothing'"):
        db.set_mouse(mouse_id=5, viruses=["AAV-nothing"])

    with pytest.raises(ValueError, match="listed twice"):
        db.set_mouse(mouse_id=5, viruses=["pAAVrg-EF1a-Flpo", "paavrg-ef1a-flpo"])

    db.add_virus(virus_name="AAV-new")
    db.set_mouse(mouse_id=5, viruses=["AAV-new"])
    assert list(db.mouse_viruses.loc[[5], "virus_name"]) == ["AAV-new"]


def test_a_genotype_is_always_stated(db):
    # not genotyped is 'na', never a blank
    with pytest.raises(ValueError, match="None is not a valid genotype"):
        db.set_mouse(mouse_id=1, lines={"TH-Cre": None})

    # a wild-type mouse has nothing to genotype
    with pytest.raises(ValueError, match="genotype must be .na."):
        db.set_mouse(mouse_id=1, lines={"Wild Type": "het"})

    db.set_mouse(mouse_id=1, lines={"wild type": "na", "TH-Cre": "N/A"})
    stored = db.mouse_lines.loc[[1]].set_index("line_name")["genotype"]
    assert stored.to_dict() == {"TH-Cre": "na", "Wild Type": "na"}


@pytest.mark.parametrize("bad", ["m0", "mouse", "", "x-1"])
def test_a_mouse_needs_a_number(db, bad):
    with pytest.raises(ValueError, match="not a valid mouse name"):
        db.set_mouse(mouse_id=bad)


def test_sessions_need_a_registered_mouse(db):
    with pytest.raises(ValueError, match="not registered"):
        db.add_session(mouse_id=442, session_date="2026-07-08")

    db.set_mouse(mouse_id=442)
    session_id = db.add_session(mouse_id="m442", session_date="2026-07-08")

    assert db.session_for("m442", "2026-07-08") == session_id
    assert db.session_for(442, "2026-07-09") is None
    assert db.sessions.loc[session_id, "session_date"] == "2026-07-08"

    with pytest.raises(ValueError, match="already has a session"):
        db.add_session(mouse_id=442, session_date="2026-07-08")

    with pytest.raises(ValueError, match="Not a date"):
        db.add_session(mouse_id=442, session_date="July 8")


# --------------------------------------------------------------------------- #
# Odors
# --------------------------------------------------------------------------- #


def test_the_registry_is_seeded_with_the_olfactometer_numbers(db):
    odors = db.odors
    assert odors.loc[1, "odor_name"] == "eugenol"
    assert odors.loc[17, "odor_name"] == "alpha" and odors.loc[17, "mix"]
    assert not odors.loc[1, "mix"]
    assert sorted(db.odor_components.loc[17, "component_id"]) == [4, 10]
    assert odors.loc[4, "vapor_pressure_mmhg"] == pytest.approx(7.78)
    assert odors.loc[0, "vapor_pressure_source"] is None  # mineral oil, on every pandas


def test_odors_are_added_and_corrected(db):
    db.add_odor(odor_id=45, odor_name="2-hexanone")
    assert not db.odors.loc[45, "mix"]
    assert db.odors.loc[45, "vapor_pressure_mmhg"] is None or (
        db.odors.loc[45, "vapor_pressure_mmhg"]
        != db.odors.loc[45, "vapor_pressure_mmhg"]
    )

    db.update_odor(
        odor_id=45, vapor_pressure_mmhg=11.6, vapor_pressure_source="PubChem"
    )
    assert db.odors.loc[45, "vapor_pressure_mmhg"] == pytest.approx(11.6)
    assert db.odors.loc[45, "vapor_pressure_source"] == "PubChem"

    db.update_odor(odor_id=45, retired=True)
    assert bool(db.odors.loc[45, "retired"])

    with pytest.raises(ValueError, match="No odor"):
        db.update_odor(odor_id=999, vapor_pressure_mmhg=1.0)

    with pytest.raises(TypeError):  # a stored odor's name never changes
        db.update_odor(odor_id=45, odor_name="x")

    with pytest.raises(ValueError, match="Nothing to change"):
        db.update_odor(odor_id=45)


def test_a_mix_is_its_components(db):
    db.add_odor(odor_id=45, odor_name="sigma", components=[4, 10])
    assert db.odors.loc[45, "mix"]
    assert sorted(db.odor_components.loc[45, "component_id"]) == [4, 10]

    with pytest.raises(ValueError, match="at least two"):
        db.add_odor(odor_id=46, odor_name="one", components=[4])

    with pytest.raises(ValueError, match="itself a mix"):
        db.add_odor(odor_id=46, odor_name="nested", components=[45, 4])

    with pytest.raises(ValueError, match="No odor"):
        db.add_odor(odor_id=46, odor_name="ghost", components=[4, 999])

    with pytest.raises(ValueError, match="listed twice"):
        db.add_odor(odor_id=46, odor_name="twice", components=[4, 4])

    assert 46 not in db.odors.index


# --------------------------------------------------------------------------- #
# Panels and vials
# --------------------------------------------------------------------------- #

# What the test panels hold: position -> odor
POSITIONS = {1: 1, 3: 0, 4: 17}

RACK = [
    {
        "vial_position": 1,
        "odor_id": 1,  # eugenol, pure
        "vial_sccm": 200,
        "total_sccm": 1500,
        "total_volume_ml": 5,
        "solvent_volume_ml": 4.99,
        "components": [
            {"odor_id": 1, "target_ppm": 0.3, "liquid_ul": 12.5, "percent_vv": 0.25}
        ],
    },
    {
        "vial_position": 3,
        "odor_id": 0,  # mineral oil
        "vial_sccm": 200,
        "total_sccm": 1500,
        "total_volume_ml": 5,
    },
    {
        "vial_position": 4,
        "odor_id": 17,  # alpha, a mix
        "vial_sccm": 200,
        "total_sccm": 1500,
        "total_volume_ml": 5,
        "made_on": "2026-07-01",
        "components": [
            {"odor_id": 4, "target_ppm": 44.2},
            {"odor_id": 10, "target_ppm": 13.3},
        ],
    },
]


def test_a_panel_fixes_which_odor_each_position_holds(db):
    db.add_panel(panel_name="print_v3", positions=POSITIONS)
    positions = db.panel_positions.loc[[1]].set_index("vial_position")["odor_id"]
    assert positions.to_dict() == POSITIONS

    # a vial's odor comes from its position, and can only be that one
    ids = db.add_vials(
        panel_name="print_v3",
        vials=[{k: v for k, v in RACK[1].items() if k != "odor_id"}],
    )
    assert db.vials.loc[ids[0], "odor_id"] == 0

    with pytest.raises(ValueError, match="holds odor 1, not 2"):
        db.add_vials(panel_name="print_v3", vials=[{**RACK[0], "odor_id": 2}])

    with pytest.raises(ValueError, match="no position 2"):
        db.add_vials(panel_name="print_v3", vials=[{**RACK[0], "vial_position": 2}])

    with pytest.raises(ValueError, match="same position"):
        db.add_vials(panel_name="print_v3", vials=[RACK[0], RACK[0]])

    with pytest.raises(ValueError, match="No odors with ids"):
        db.add_panel(panel_name="ghost", positions={1: 999})

    assert len(db.vials) == 1 and "ghost" not in set(db.panels["panel_name"])


def test_vials_record_the_rack_as_made(db):
    db.add_panel(panel_name="print_v3", positions=POSITIONS, description="16 odors")
    ids = db.add_vials(panel_name="print_v3", vials=RACK, made_on="2026-07-06")

    vials = db.vials.loc[ids]
    assert list(vials["vial_position"]) == [1, 3, 4]
    assert list(vials["made_on"].dt.strftime("%Y-%m-%d")) == [
        "2026-07-06",
        "2026-07-06",
        "2026-07-01",
    ]
    assert vials.loc[ids[0], "solvent_volume_ml"] == pytest.approx(4.99)

    components = db.vial_odors.loc[[ids[2]]].set_index("odor_id")
    assert components["target_ppm"].to_dict() == {4: 44.2, 10: 13.3}
    assert ids[1] not in db.vial_odors.index

    # Recorded as they are made: today unless said otherwise
    later = db.add_vials(panel_name="print_v3", vials=RACK[:1])
    assert (
        db.vials.loc[later[0], "made_on"].date() == __import__("datetime").date.today()
    )


@pytest.mark.parametrize(
    "vial, message",
    [
        (
            {**RACK[0], "components": [{"odor_id": 2, "target_ppm": 1}]},
            "only component",
        ),
        ({**RACK[2], "components": []}, "list all of those"),
        (
            {**RACK[2], "components": [{"odor_id": 4, "target_ppm": 1}]},
            "list all of those",
        ),
        (
            {**RACK[2], "components": [{"odor_id": 18, "target_ppm": 1}]},
            "not a monomolecular odor",
        ),
        ({**RACK[0], "odor_id": 999}, "holds odor 1, not 999"),
        ({"vial_position": 1, "odor_id": 1}, "A vial needs"),
    ],
)
def test_a_recipe_that_makes_no_sense_is_refused(db, vial, message):
    db.add_panel(panel_name="p", positions=POSITIONS)

    with pytest.raises(ValueError, match=message):
        db.add_vials(panel_name="p", vials=[vial])

    assert len(db.vials) == 0


def test_vials_need_a_panel(db):
    with pytest.raises(ValueError, match="No panel"):
        db.add_vials(panel_name="nope", vials=RACK)


def test_a_loop_runs_one_rack(db, tmp_path):
    db.add_panel(panel_name="a", positions=POSITIONS)
    db.add_panel(panel_name="b", positions=POSITIONS)
    rack_a = db.add_vials(panel_name="a", vials=RACK)
    rack_b = db.add_vials(panel_name="b", vials=RACK[:1])
    remade = db.add_vials(panel_name="a", vials=RACK[:1])

    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-01-01")

    with db._writing() as cur:
        cur.execute(
            """
            INSERT INTO experiments
                ( exp_id, session_id, exp_number, exp_path, exp_start
                , height_px, width_px, height_um, width_um, frames_per_slice
                , frame_rate, objective, exp_type
                ) VALUES (7, ?, 1, 'p', '2026-01-01 10:00:00'
                         , 1, 1, 1, 1, 1, 1, 20, 'loop');
            """,
            [session_id],
        )

    db.set_experiment_vials(exp_id=7, vial_ids=rack_a)
    assert sorted(db.experiment_vials.loc[[7], "vial_id"]) == sorted(rack_a)

    # Declarative: the new set replaces the old one
    db.set_experiment_vials(exp_id=7, vial_ids=rack_a[1:])
    assert sorted(db.experiment_vials.loc[[7], "vial_id"]) == sorted(rack_a[1:])

    with pytest.raises(ValueError, match="one panel"):
        db.set_experiment_vials(exp_id=7, vial_ids=rack_a + rack_b)

    with pytest.raises(ValueError, match="same position"):
        db.set_experiment_vials(exp_id=7, vial_ids=rack_a + remade)

    with pytest.raises(ValueError, match="No vials"):
        db.set_experiment_vials(exp_id=7, vial_ids=[999])
