"""
`Database.add_experiment`, `add_grabs` and `add_zstacks`: what lands where when a
recording folder is added to a session.
"""

import json

from datetime import datetime

import pandas as pd
import pytest

from odyn import Database
from odyn.database import OBJECTIVE_SCALE, IngestFlag
from generate_data import EXP_START, UM_PER_PX, generate

# The 2026-08-11 z-stack of m357, as its header has it: 21 slices 5 um apart
# from -5.7 to -105.7, 35% at -55.7 with a -300 um length constant
STACK = dict(slices=21, step_um=5, top_um=-5.7, power=35, length_um=-300)
STACK["reference_um"] = -55.7


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """One synthetic loop, generated once for the module."""
    folder = tmp_path_factory.mktemp("recording")
    generate(folder, acquisitions=3, frames=56, height=32, width=40, z_um=-120.5)

    return folder


@pytest.fixture
def db(recording):
    """A fresh project database over the recording, with its session made."""
    db = Database(recording, project="p", can_create=True, user="tester")
    db.set_mouse(mouse_id="m1")
    db.add_session(mouse_id="m1", session_date="2026-01-01")

    yield db

    # The recording is shared by the module, the database is not
    db.path.unlink()


def add(db, **kwargs):
    return db.add_experiment(
        session_id=1, exp_path="20260101/m1/e1", objective=20, **kwargs
    )


def a_day(tmp_path, folder: str, count: int = 2, **options) -> tuple[Database, int]:
    """`count` grabs (or z-stacks) of m1 on 2026-03-04, and their session."""
    generate(
        tmp_path,
        acquisitions=count,
        frames=56,
        height=32,
        width=40,
        exp=folder,
        start=datetime(2026, 3, 4, 9, 0, 0),
        acq_state="grab",
        **options,
    )
    db = Database(tmp_path, can_create=True)
    db.set_mouse(mouse_id=1)

    return db, db.add_session(mouse_id=1, session_date="2026-03-04")


# --------------------------------------------------------------------------- #
# What lands
# --------------------------------------------------------------------------- #


def test_the_loop_points_at_its_session(db):
    exp_id = add(db)

    experiment = db.experiments.loc[exp_id]
    assert experiment["session_id"] == 1
    assert experiment["exp_number"] == 1
    assert experiment["exp_path"] == "20260101/m1/e1"
    assert experiment["exp_start"] == EXP_START
    assert experiment["objective"] == 20
    assert experiment["frames_per_slice"] == 56
    assert experiment["z_um"] == -120.5

    assert db.experiment_for(1, 1) == exp_id
    assert db.experiment_for(1, 2) is None


def test_the_field_of_view_is_stored_in_real_microns(db):
    exp_id = add(db)

    # ScanImage's microns assume 20x, so a 20x experiment keeps them as they are
    experiment = db.experiments.loc[exp_id]
    assert experiment["width_um"] == pytest.approx(40 * UM_PER_PX)
    assert experiment["height_um"] == pytest.approx(32 * UM_PER_PX)
    assert db.latest_calls("add_experiment").iloc[0]["input_um_scale"] == 1.0


def test_every_raw_tiff_becomes_an_acquisition(db):
    exp_id = add(db)

    acquisitions = db.acquisitions[db.acquisitions["exp_id"] == exp_id]
    assert len(acquisitions) == 3
    assert list(acquisitions["raw_path"].str.endswith(".tif")) == [True] * 3

    # No sync timing until the sync file is processed
    assert acquisitions["sync_odor_on_frame"].isna().all()
    assert "sync_frame_count" in db.acquisition_trials.columns


def test_rig_settings_became_annotations(db):
    exp_id = add(db)

    settings = db.annotations_for("experiment").loc[exp_id]
    assert settings["laser_power_920"] == 10
    assert settings["laser_power_1040"] == 0
    assert settings["loop_acq_interval_s"] == 10.0

    # ...written by the same call that added the loop
    (call_id,) = set(db.annotations["method_call_id"])
    assert db.method_calls.loc[call_id, "method_name"] == "Database.add_experiment"


def test_the_loop_gets_a_group_of_its_own(db):
    exp_id = add(db)

    (group_id,) = db.group_experiments[db.group_experiments["exp_id"] == exp_id].index
    group = db.groups[group_id]

    assert list(group.experiments.index) == [exp_id]
    assert list(group.exp_names) == ["20260101_m1_e1"]
    assert len(group.acquisitions) == 3
    assert len(group.sessions) == 1
    assert len(group.sync_blocks) == 0


def test_the_call_is_recorded_with_its_output(db):
    exp_id = add(db)

    call = db.latest_calls("add_experiment").iloc[0]
    assert call["user"] == "tester"
    assert call["input_exp_path"] == "20260101/m1/e1"
    assert call["input_objective"] == 20
    assert call["output_exp_ids"] == [exp_id]
    assert call["call_flag"] == 0


def test_the_number_comes_from_the_folder_unless_given(db, recording):
    (recording / "20260101/m1/e1").rename(recording / "20260101/m1/odd")

    try:
        with pytest.raises(ValueError, match="not named eN"):
            db.add_experiment(session_id=1, exp_path="20260101/m1/odd", objective=20)

        exp_id = db.add_experiment(
            session_id=1, exp_path="20260101/m1/odd", objective=20, exp_number=4
        )
        assert db.experiments.loc[exp_id, "exp_number"] == 4

    finally:
        (recording / "20260101/m1/odd").rename(recording / "20260101/m1/e1")


# --------------------------------------------------------------------------- #
# What is refused
# --------------------------------------------------------------------------- #


def test_nothing_is_inserted_twice(db):
    first = add(db)
    assert add(db) is None

    assert list(db.experiments.index) == [first]
    assert (
        db.latest_calls("add_experiment").iloc[0]["call_flag"]
        == IngestFlag.ALREADY_IN_DB
    )


def test_the_files_must_agree_with_the_session(db):
    db.set_mouse(mouse_id=2)
    other = db.add_session(mouse_id=2, session_date="2026-01-01")

    with pytest.raises(RuntimeError, match="but the session is m2 on 2026-01-01"):
        db.add_experiment(session_id=other, exp_path="20260101/m1/e1", objective=20)

    assert len(db.experiments) == 0
    assert (
        db.latest_calls("add_experiment").iloc[0]["call_flag"]
        & IngestFlag.NOT_THE_SESSION
    )


def test_a_session_that_does_not_exist_is_refused(db):
    with pytest.raises(ValueError, match="no session 9"):
        db.add_experiment(session_id=9, exp_path="20260101/m1/e1", objective=20)


def test_an_objective_without_a_scale_is_refused(db):
    with pytest.raises(ValueError, match="No micron scale for objective 16"):
        db.add_experiment(session_id=1, exp_path="20260101/m1/e1", objective=16)

    assert len(db.experiments) == 0


def test_a_loop_is_not_a_grab(db):
    added = db.add_grabs(session_id=1, folder="20260101/m1/e1", objective=20)

    assert added == []
    assert (
        db.latest_calls("add_grabs").iloc[0]["call_flag"] & IngestFlag.WRONG_ACQ_STATE
    )


def test_a_loop_of_z_stacks_is_refused_for_now(tmp_path):
    generate(tmp_path, acquisitions=2, frames=56, height=32, width=40, zstack=STACK)
    db = Database(tmp_path, can_create=True)
    db.set_mouse(mouse_id=1)
    db.add_session(mouse_id=1, session_date="2026-01-01")

    with pytest.raises(RuntimeError, match="z-stack loop with 21 slices"):
        db.add_experiment(session_id=1, exp_path="20260101/m1/e1", objective=20)

    assert (
        db.latest_calls("add_experiment").iloc[0]["call_flag"] & IngestFlag.STACK_LOOP
    )
    assert len(db.experiments) == 0


# --------------------------------------------------------------------------- #
# Grabs and z-stacks
# --------------------------------------------------------------------------- #


def test_each_grab_is_a_row_of_its_own(tmp_path):
    db, session_id = a_day(tmp_path, "grabs", z_um=-80)

    added = db.add_grabs(
        session_id=session_id, folder="20260304/m1/grabs", objective=10
    )
    assert len(added) == 2

    grabs = db.grabs.loc[added]
    assert list(grabs["raw_path"]) == [
        "20260304/m1/grabs/raw/20260304_m1_grabs_00001.tif",
        "20260304/m1/grabs/raw/20260304_m1_grabs_00002.tif",
    ]
    assert set(grabs["session_id"]) == {session_id}
    assert set(grabs["z_um"]) == {-80}
    assert set(grabs["objective"]) == {10}
    assert grabs["width_um"].tolist() == pytest.approx(
        [40 * UM_PER_PX * OBJECTIVE_SCALE[10]] * 2
    )
    assert db.latest_calls("add_grabs").iloc[0]["input_um_scale"] == OBJECTIVE_SCALE[10]
    assert grabs["sync_frame_count"].isna().all()  # until a sync file is processed

    assert db.annotations_for("grab").loc[added[0], "laser_power_920"] == 10

    # Running it again only reports what is already there
    assert (
        db.add_grabs(session_id=session_id, folder="20260304/m1/grabs", objective=10)
        == []
    )

    # ...and a grab is not a z-stack
    assert (
        db.add_zstacks(session_id=session_id, folder="20260304/m1/grabs", objective=10)
        == []
    )


def test_a_zstack_keeps_its_stack_and_the_powers_at_both_ends(tmp_path):
    db, session_id = a_day(tmp_path, "zstacks", count=1, zstack=STACK)

    (added,) = db.add_zstacks(
        session_id=session_id, folder="20260304/m1/zstacks", objective=20
    )
    stack = db.zstacks.loc[added]

    assert stack["slice_count"] == 21
    assert stack["z_step_um"] == 5
    assert stack["frames_per_slice"] == 56
    assert (stack["z_start_um"], stack["z_end_um"]) == (-5.7, -105.7)
    assert pd.isna(stack["exp_id"])  # until it is set for one

    # What the lab aims at (30% near the top, more below), from the header
    powers = db.annotations_for("zstack").loc[added]
    assert powers["power_start_pct"] == pytest.approx(29.63)
    assert powers["power_end_pct"] == pytest.approx(41.35)
    assert powers["power_length_um"] == -300

    # A z-stack is not a grab either
    assert (
        db.add_grabs(session_id=session_id, folder="20260304/m1/zstacks", objective=20)
        == []
    )


def test_a_zstack_is_for_a_loop_of_its_own_session(tmp_path):
    db, session_id = a_day(tmp_path, "zstacks", count=1, zstack=STACK)
    (zstack_id,) = db.add_zstacks(
        session_id=session_id, folder="20260304/m1/zstacks", objective=20
    )
    generate(
        tmp_path,
        acquisitions=1,
        frames=56,
        height=32,
        width=40,
        start=datetime(2026, 3, 4, 10, 0, 0),
    )
    exp_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e1", objective=20
    )

    db.set_zstack_experiment(zstack_id=zstack_id, exp_id=exp_id)
    assert db.zstacks.loc[zstack_id, "exp_id"] == exp_id

    db.set_mouse(mouse_id=2)
    other = db.add_session(mouse_id=2, session_date="2026-03-04")

    with db._writing() as cur:
        cur.execute(
            "INSERT INTO experiments (session_id, exp_number, exp_path, exp_start, "
            "height_px, width_px, height_um, width_um, frames_per_slice, frame_rate, "
            "objective) VALUES (?, 1, 'x', '2026-03-04 12:00:00', 1, 1, 1, 1, 1, "
            "1, 20);",
            [other],
        )
        foreign = cur.lastrowid

    with pytest.raises(ValueError, match="not from the same session"):
        db.set_zstack_experiment(zstack_id=zstack_id, exp_id=foreign)

    db.set_zstack_experiment(zstack_id=zstack_id, exp_id=None)
    assert db.zstacks["exp_id"].isna().all()


# --------------------------------------------------------------------------- #
# Olfactometer program definitions
# --------------------------------------------------------------------------- #


def test_program_definitions_are_read_by_name(tmp_path):
    from odyn.database import _program_configs

    def node(name, program=None, items=(), folder=False):
        return {
            "Name": name,
            "IsFolder": folder,
            "Program": program,
            "Items": list(items),
        }

    passive = {"Name": "16odors passive", "Trials": []}
    nodes = [
        node("16odors passive", passive),
        node(
            "4.x series",
            folder=True,
            items=[
                node("Buffer program", {"Name": "Buffer program", "Odor": 1}),
                node("Quick test", {"Name": "Quick test", "Odor": 1}),
            ],
        ),
        node(
            "5.x series",
            folder=True,
            items=[
                node("Buffer program", {"Name": "Buffer program", "Odor": 1}),
                node("Quick test", {"Name": "Quick test", "Odor": 2}),
            ],
        ),
    ]
    folder = tmp_path / "config"
    folder.mkdir()
    # The olfactometer writes it with a byte-order mark
    (folder / "Program.config").write_text(json.dumps(nodes), encoding="utf-8-sig")

    configs = _program_configs(folder)

    # Found in folders too; a name kept by two different programs is left out
    assert set(configs) == {"16odors passive", "Buffer program"}
    assert json.loads(configs["16odors passive"]) == passive
    assert _program_configs(tmp_path / "missing") == {}

    (folder / "Program.config").write_text("{not json")
    assert _program_configs(folder) == {}
