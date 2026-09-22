"""
`Database.add_experiment` and `add_grabs`: what lands where when a recording
folder is added to a session.
"""

import json

from datetime import datetime

import pytest

from odyn import Database
from odyn.database import ExpFlag
from generate_data import EXP_START, generate


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """One synthetic loop, generated once for the module."""
    folder = tmp_path_factory.mktemp("recording")
    generate(folder, acquisitions=3, frames=56, height=32, width=40)

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


# --------------------------------------------------------------------------- #
# What lands
# --------------------------------------------------------------------------- #


def test_the_experiment_points_at_its_session(db):
    exp_id = add(db)

    experiment = db.experiments.loc[exp_id]
    assert experiment["session_id"] == 1
    assert experiment["exp_name"] == "20260101_m1_e1"
    assert experiment["exp_type"] == "loop"
    assert experiment["exp_path"] == "20260101/m1/e1"
    assert experiment["exp_start"] == EXP_START
    assert experiment["objective"] == 20
    assert experiment["frame_count"] == 56

    assert db.experiment_for(1, "20260101_m1_e1") == exp_id


def test_every_raw_tiff_becomes_an_acquisition(db):
    exp_id = add(db)

    acquisitions = db.acquisitions[db.acquisitions["exp_id"] == exp_id]
    assert len(acquisitions) == 3
    assert list(acquisitions["raw_path"].str.endswith(".tif")) == [True] * 3

    # No sync timing until the sync file is decoded
    assert acquisitions["sync_odor_on_frame"].isna().all()
    assert "sync_frame_count" in db.acquisition_trials.columns


def test_rig_settings_became_annotations(db):
    exp_id = add(db)

    settings = db.annotations_for("experiment").loc[exp_id]
    assert settings["laser_power_920"] == 10
    assert settings["laser_power_1040"] == 0
    assert settings["loop_acq_interval_s"] == 10.0

    # ...written by the same call that added the experiment
    (call_id,) = set(db.annotations["method_call_id"])
    assert db.method_calls.loc[call_id, "method_name"] == "Database.add_experiment"


def test_the_experiment_gets_a_group_of_its_own(db):
    exp_id = add(db)

    (group_id,) = db.group_experiments[db.group_experiments["exp_id"] == exp_id].index
    group = db.groups[group_id]

    assert list(group.experiments.index) == [exp_id]
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


# --------------------------------------------------------------------------- #
# What is refused
# --------------------------------------------------------------------------- #


def test_nothing_is_inserted_twice(db):
    first = add(db)
    assert add(db) is None

    assert list(db.experiments.index) == [first]
    assert (
        db.latest_calls("add_experiment").iloc[0]["call_flag"] == ExpFlag.ALREADY_IN_DB
    )


def test_the_files_must_agree_with_the_session(db):
    db.set_mouse(mouse_id=2)
    other = db.add_session(mouse_id=2, session_date="2026-01-01")

    with pytest.raises(RuntimeError, match="mouse 1 on 2026-01-01, but session 2"):
        db.add_experiment(session_id=other, exp_path="20260101/m1/e1", objective=20)

    assert len(db.experiments) == 0
    assert (
        db.latest_calls("add_experiment").iloc[0]["call_flag"] & ExpFlag.NOT_THE_SESSION
    )


def test_a_session_that_does_not_exist_is_refused(db):
    with pytest.raises(ValueError, match="no session 9"):
        db.add_experiment(session_id=9, exp_path="20260101/m1/e1", objective=20)


def test_a_loop_is_not_a_grab(db):
    added = db.add_grabs(session_id=1, exp_path="20260101/m1/e1", objective=20)

    assert added == []
    assert db.latest_calls("add_grabs").iloc[0]["call_flag"] & ExpFlag.WRONG_ACQ_STATE


# --------------------------------------------------------------------------- #
# Grabs and z-stacks
# --------------------------------------------------------------------------- #


def test_each_grab_is_its_own_experiment(tmp_path):
    generate(
        tmp_path,
        acquisitions=2,
        frames=56,
        height=32,
        width=40,
        exp="grabs",
        start=datetime(2026, 3, 4, 9, 0, 0),
        acq_state="grab",
    )
    db = Database(tmp_path, can_create=True)
    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-03-04")

    added = db.add_grabs(
        session_id=session_id, exp_path="20260304/m1/grabs", objective=10
    )
    assert len(added) == 2

    experiments = db.experiments.loc[added]
    assert set(experiments["exp_type"]) == {"grab"}
    assert set(experiments["exp_path"]) == {"20260304/m1/grabs"}
    assert list(experiments["exp_name"]) == [
        "20260304_m1_grabs_00001",
        "20260304_m1_grabs_00002",
    ]

    # Running it again only reports what is already there
    assert (
        db.add_grabs(session_id=session_id, exp_path="20260304/m1/grabs", objective=10)
        == []
    )

    # Z-stacks look the same to ScanImage; the caller says which they are
    with pytest.raises(ValueError, match="'grab' or 'zstack'"):
        db.add_grabs(
            session_id=session_id,
            exp_path="20260304/m1/grabs",
            objective=10,
            exp_type="stack",
        )


# --------------------------------------------------------------------------- #
# Olfactometer program definitions
# --------------------------------------------------------------------------- #


def test_program_definitions_are_read_by_name(tmp_path):
    from odyn.database import _program_configs

    folder = tmp_path / "olfactometer"
    folder.mkdir()

    (folder / "a.json").write_text(
        json.dumps({"Name": "16odors passive", "Trials": []})
    )
    (folder / "b.json").write_text(
        json.dumps({"Id": "x", "Program": {"Name": "Empty channels", "Trials": []}})
    )
    (folder / "c.json").write_text("{not json")
    (folder / "d.json").write_text(json.dumps({"no": "name"}))

    configs = _program_configs(folder)

    assert set(configs) == {"16odors passive", "Empty channels"}
    assert json.loads(configs["Empty channels"]) == {
        "Name": "Empty channels",
        "Trials": [],
    }
    assert _program_configs(tmp_path / "missing") == {}
