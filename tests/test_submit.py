"""
Submitting a draft: recordings ingested and typed-in fields attached, together.

A drafted session has no rows anywhere until this runs, so the risk is not a
wrong value but a lost one -- a day someone typed that quietly does not arrive.
Most of what is checked here is therefore what `check` refuses to submit.
"""

import shutil

import pytest

from odyn import Database
from odyn.app.draft import Draft
from odyn.app.submit import SubmitRefused, check, session_path, submit

from generate_data import EXP, EXP_DATE, EXP_START, MOUSE, generate

ACQUISITIONS = 2
FRAMES = 64

PANEL = [
    {"vial_position": 1, "odor_id": 1},
    {"vial_position": 2, "odor_id": 0},
]


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """One synthetic recording, generated once: writing TIFFs is the slow part."""
    folder = tmp_path_factory.mktemp("recording")

    generate(
        folder, acquisitions=ACQUISITIONS, frames=FRAMES,
        height=32, width=32, motion=0.0,
    )

    return folder


@pytest.fixture
def db(recording, tmp_path):
    """A database over this test's own copy of the recording, with a panel."""
    folder = tmp_path / "main"
    shutil.copytree(recording, folder)

    database = Database(folder, project="test")
    database.add_panel(panel_name="print_v3", vials=PANEL)

    return database


@pytest.fixture
def draft(tmp_path):
    """A draft for the session the recording belongs to."""
    return Draft.open(
        tmp_path / "drafts",
        mouse_id=int(MOUSE.lstrip("m")),
        date=EXP_START.date().isoformat(),
    )


@pytest.fixture
def filled(draft):
    """A draft with the kind of thing someone types during a session."""
    draft.update_session(goal="10x pre/post ket/xyl", mouse_weight_g=25.1)
    draft.add_note("genteal drops before starting")
    draft.add_flag("pmt off for first 2 acquisitions")
    draft.set_experiment(EXP, fov_depth_um=-55.0, objective=10, goal="baseline")
    draft.set_panel(panel_name="print_v3", made_on="2026-07-06")

    return draft


# --------------------------------------------------------------------------- #
# Where the recordings should be
# --------------------------------------------------------------------------- #


def test_the_session_folder_is_found_not_guessed(db, draft):
    """
    A mouse id is the number alone, so it cannot say whether the folder was
    written `m1`, `m001` or `M1`. Reconstructing the name would miss.
    """
    assert session_path(db.main_folder, draft) == f"{EXP_DATE}/{MOUSE}"


def test_a_folder_named_any_other_way_is_still_found(db, tmp_path):
    """The same animal, filed under a differently padded name."""
    (db.main_folder / "20260202" / "M0001" / "e1" / "raw").mkdir(parents=True)

    other = Draft.open(tmp_path / "drafts", mouse_id=1, date="2026-02-02")

    assert session_path(db.main_folder, other) == "20260202/M0001"


def test_two_folders_for_one_mouse_and_day_are_refused(db, tmp_path, filled):
    """Ambiguous: a person has to say which, rather than the app picking."""
    (db.main_folder / EXP_DATE / "m0001").mkdir(parents=True)

    assert session_path(db.main_folder, filled) is None
    assert any("several folders" in problem.what for problem in check(filled, db))


def test_an_unusual_folder_can_be_named(db, draft):
    """A session filed somewhere else is still submittable."""
    (db.main_folder / "somewhere" / "else").mkdir(parents=True)

    draft.data["session_path"] = "somewhere/else"
    draft.save()

    assert session_path(db.main_folder, draft) == "somewhere/else"


# --------------------------------------------------------------------------- #
# What check refuses
# --------------------------------------------------------------------------- #


def test_an_empty_draft_has_nothing_to_submit(db, draft):
    assert [problem.what for problem in check(draft, db)] == [
        "nothing has been filled in"
    ]


def test_recordings_that_were_never_copied_are_reported(db, tmp_path, filled):
    """
    The common case: someone fills the app in during a session and submits
    before moving the files off the rig.
    """
    other = Draft.open(tmp_path / "drafts", mouse_id=999, date=EXP_START.date().isoformat())
    other.update_session(goal="never copied")

    problems = [problem for problem in check(other, db) if problem.blocking]

    assert any("no folder for this mouse" in problem.what for problem in problems)


def test_a_filled_draft_with_recordings_is_accepted(db, filled):
    assert [problem for problem in check(filled, db) if problem.blocking] == []


def test_an_experiment_that_was_never_recorded_blocks(db, filled):
    """Submitting would drop what was typed for it, silently."""
    filled.set_experiment("e99", fov_depth_um=-120.0)

    problems = [problem for problem in check(filled, db) if problem.blocking]

    assert any("e99" in problem.where for problem in problems)
    assert any("would lose it" in problem.what for problem in problems)


def test_a_recording_nobody_filled_in_is_only_a_warning(db, draft):
    """Worth saying, but not worth refusing: the recording is still ingested."""
    draft.update_session(goal="did not describe the experiment")

    problems = [problem for problem in check(draft, db) if not problem.blocking]

    assert any("nothing filled in" in problem.what for problem in problems)


def test_an_unregistered_key_blocks(db, filled):
    """A field the registry does not hold would be refused at write time."""
    filled.update_session(mouse_wieght_g=25.1)

    assert any("not a session annotation" in problem.what
               for problem in check(filled, db))


def test_an_unregistered_panel_blocks(db, filled):
    filled.set_panel(panel_name="print_v9")

    assert any("not registered" in problem.what for problem in check(filled, db))


def test_a_vial_the_panel_does_not_have_blocks(db, filled):
    filled.set_panel(panel_name="print_v3", vial_dates={9: "2026-07-21"})

    assert any("no vial" in problem.what for problem in check(filled, db))


def test_no_panel_is_only_a_warning(db, draft):
    draft.update_session(goal="forgot to say which panel")

    problems = check(draft, db)

    assert any("no odor panel" in problem.what for problem in problems)
    assert not any(
        problem.blocking and "panel" in problem.what for problem in problems
    )


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


def test_submit_refuses_a_draft_with_blocking_problems(db, filled):
    filled.set_experiment("e99", fov_depth_um=-120.0)

    with pytest.raises(SubmitRefused) as refused:
        submit(filled, db)

    assert any(problem.blocking for problem in refused.value.problems)

    # Nothing was written, so the draft is still there to fix.
    assert filled.path.exists()
    assert db.con.execute("SELECT count(*) FROM sessions;").fetchone()[0] == 0


def test_submit_ingests_and_annotates_together(db, filled):
    written = submit(filled, db)

    assert written.session_id is not None
    assert EXP in written.experiments

    session = db.con.execute("SELECT * FROM sessions;").fetchone()

    assert session["mouse_id"] == int(MOUSE.lstrip("m"))
    assert db.con.execute("SELECT count(*) FROM acquisitions;").fetchone()[0] == (
        ACQUISITIONS
    )


def test_the_typed_fields_arrive_as_annotations(db, filled):
    written = submit(filled, db)

    stored = {
        row["key"]: row["value"]
        for row in db.con.execute(
            "SELECT key, value FROM annotations WHERE target_type = 'session';"
        )
    }

    assert stored["goal"] == "10x pre/post ket/xyl"
    assert stored["mouse_weight_g"] == 25.1

    depth = db.con.execute(
        """
        SELECT value FROM annotations
            WHERE target_type = 'experiment' AND key = 'fov_depth_um';
        """
    ).fetchone()

    assert depth["value"] == -55.0


def test_notes_become_one_annotation_each(db, filled):
    """A multi-valued key holds separate observations, not one list."""
    filled.add_note("adjusted thermal probe")

    submit(filled, db)

    notes = db.con.execute(
        """
        SELECT value FROM annotations
            WHERE target_type = 'session' AND key = 'note' ORDER BY annotation_id;
        """
    ).fetchall()

    assert [row["value"] for row in notes] == [
        "genteal drops before starting", "adjusted thermal probe"
    ]


def test_the_objective_becomes_a_column(db, filled):
    """It is not an annotation: the micron-per-pixel scale depends on it."""
    submit(filled, db)

    assert db.con.execute("SELECT objective FROM experiments;").fetchone()[0] == 10


def test_the_panel_and_its_mixing_are_recorded(db, filled):
    written = submit(filled, db)

    assert written.panel
    assert db.session_panels.loc[written.session_id, "panel_name"] == "print_v3"

    mixed = db.session_vials.loc[written.session_id, "made_on"]

    assert set(mixed.dt.date.astype(str)) == {"2026-07-06"}


def test_the_mouse_block_reaches_the_mice_table(db, filled):
    filled.update_mouse(sex="F", dob="2026-01-22", lines={"TIGRE": "het"})

    written = submit(filled, db)

    assert written.mouse_id == int(MOUSE.lstrip("m"))
    assert db.mice.loc[written.mouse_id, "mouse_sex"] == "F"
    assert db.mouse_lines.loc[written.mouse_id].index.tolist() == ["TIGRE"]


def test_a_submitted_draft_is_not_offered_again(db, filled, tmp_path):
    submit(filled, db)

    assert Draft.pending(tmp_path / "drafts") == []
    assert (tmp_path / "drafts" / "submitted" / filled.path.name).exists()


def test_submitting_can_keep_the_draft(db, filled, tmp_path):
    """For a caller that wants to check the result before letting go of it."""
    submit(filled, db, archive=False)

    assert filled.path.exists()


def test_force_writes_what_it_can(db, filled):
    """An experiment with no recordings should not cost the rest of the day."""
    filled.set_experiment("e99", fov_depth_um=-120.0)

    written = submit(filled, db, force=True)

    assert written.session_id is not None
    assert "e99" not in written.experiments
    assert EXP in written.experiments
