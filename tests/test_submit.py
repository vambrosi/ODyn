"""
Submitting a draft: recordings ingested and typed-in fields attached, together.

A drafted session has no rows anywhere until this runs, so the risk is not a
wrong value but a lost one -- a day someone typed that quietly does not arrive.
Most of what is checked here is therefore what `check` refuses to submit.
"""

import shutil

from datetime import timedelta

import pytest

from odyn import Database
from odyn.app.draft import Draft
from odyn.app.submit import (
    SubmitRefused,
    check,
    check_all,
    session_folder,
    session_path,
    submit,
    submit_all,
)

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
        folder,
        acquisitions=ACQUISITIONS,
        frames=FRAMES,
        height=32,
        width=32,
        motion=0.0,
    )

    return folder


@pytest.fixture(scope="module")
def second_mouse(tmp_path_factory):
    """A second animal the same day, an hour later, in its own main folder."""
    folder = tmp_path_factory.mktemp("second")

    generate(
        folder,
        acquisitions=ACQUISITIONS,
        frames=FRAMES,
        height=32,
        width=32,
        motion=0.0,
        mouse="m2",
        start=EXP_START + timedelta(hours=1),
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
    return Draft.start(
        tmp_path / "drafts",
        mouse_id=int(MOUSE.lstrip("m")),
        date=EXP_START.date().isoformat(),
    )


@pytest.fixture
def filled(draft):
    """A draft with the kind of thing someone types during a session."""
    draft.update_session(goal="10x pre/post ket/xyl", mouse_weight_g=25.1)
    draft.update_session(
        note="genteal drops before starting\npmt off for first 2 acquisitions",
        flag=True,
    )
    draft.set_experiment(EXP, fov_depth_um=-55.0, objective=10, goal="baseline")
    draft.set_panel(panel_name="print_v3", made_on="2026-07-06")

    return draft


# --------------------------------------------------------------------------- #
# Where the recordings should be
# --------------------------------------------------------------------------- #


def test_the_session_folder_is_the_day_and_the_mouse(db, draft):
    """`20260914/m442`, which is how the rig names them."""
    assert session_folder(draft) == f"{EXP_DATE}/{MOUSE}"
    assert session_path(db.main_folder, draft) == f"{EXP_DATE}/{MOUSE}"


def test_a_folder_that_is_not_there_reads_as_missing(db, tmp_path):
    """Rather than as some other folder of the same animal."""
    other = Draft.start(tmp_path / "drafts", mouse_id=1, date="2026-02-02")

    assert session_folder(other) == "20260202/m1"
    assert session_path(db.main_folder, other) is None


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
    other = Draft.start(
        tmp_path / "drafts", mouse_id=999, date=EXP_START.date().isoformat()
    )
    other.update_session(goal="never copied")

    problems = [problem for problem in check(other, db) if problem.blocking]

    assert any(f"no folder '{EXP_DATE}/m999'" == problem.what for problem in problems)


def test_every_problem_fits_on_one_line(db, tmp_path, filled):
    """
    They are read as a list beside the session each is about, so a problem that
    wraps into a paragraph buries the ones after it. What to do about one is
    deliberately not in the text.
    """
    filled.set_experiment("e99", fov_depth_um=-120.0)
    filled.update_session(mouse_wieght_g=25.1)
    filled.set_panel(panel_name="not_a_panel")

    stranded = Draft.start(tmp_path / "drafts", mouse_id=999)
    stranded.update_session(goal="never copied")

    seen = [problem for draft in (filled, stranded) for problem in check(draft, db)]

    assert len(seen) >= 5, "the sample has to reach most of the messages"

    for problem in seen:
        assert "\n" not in problem.what
        assert len(f"{problem.where}: {problem.what}") <= 60, problem


def test_a_filled_draft_with_recordings_is_accepted(db, filled):
    assert [problem for problem in check(filled, db) if problem.blocking] == []


def test_an_experiment_that_was_never_recorded_blocks(db, filled):
    """Submitting would drop what was typed for it, silently."""
    filled.set_experiment("e99", fov_depth_um=-120.0)

    problems = [problem for problem in check(filled, db) if problem.blocking]

    assert any("e99" in problem.where for problem in problems)
    assert any("not recorded" in problem.what for problem in problems)


def test_a_recording_nobody_filled_in_is_only_a_warning(db, draft):
    """Worth saying, but not worth refusing: the recording is still ingested."""
    draft.update_session(goal="did not describe the experiment")

    problems = [problem for problem in check(draft, db) if not problem.blocking]

    assert any("not filled in" in problem.what for problem in problems)


def test_an_unregistered_key_blocks(db, filled):
    """A field the registry does not hold would be refused at write time."""
    filled.update_session(mouse_wieght_g=25.1)

    assert any(
        "not a session annotation" in problem.what for problem in check(filled, db)
    )


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
    assert not any(problem.blocking and "panel" in problem.what for problem in problems)


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
    submit(filled, db)

    stored = {
        row["key"]: row["value"]
        for row in db.con.execute(
            "SELECT key, value FROM annotations WHERE target_type = 'session';"
        )
    }

    assert stored["goal"] == "10x pre/post ket/xyl"
    assert stored["mouse_weight_g"] == 25.1

    depth = db.con.execute("""
        SELECT value FROM annotations
            WHERE target_type = 'experiment' AND key = 'fov_depth_um';
        """).fetchone()

    assert depth["value"] == -55.0


def test_the_note_arrives_as_one_annotation(db, filled):
    """One field, one row -- however many lines someone typed into it."""
    submit(filled, db)

    notes = db.con.execute("""
        SELECT value FROM annotations
            WHERE target_type = 'session' AND key = 'note';
        """).fetchall()

    assert len(notes) == 1
    assert notes[0]["value"].splitlines() == [
        "genteal drops before starting",
        "pmt off for first 2 acquisitions",
    ]


def test_the_flag_is_a_checkbox(db, filled):
    """A pointer to the note, not a second place to describe the problem."""
    submit(filled, db)

    flag = db.con.execute(
        "SELECT value FROM annotations WHERE key = 'flag';"
    ).fetchone()

    assert flag["value"] == 1


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


# --------------------------------------------------------------------------- #
# A whole day at once
# --------------------------------------------------------------------------- #


def test_a_day_is_checked_before_any_of_it_is_written(db, filled, tmp_path):
    """
    Several mice run in sequence, and their recordings are copied off once at
    the end, so every session is unsubmittable until then and reviewing them
    one at a time is four rounds of the same answer.
    """
    second = Draft.start(tmp_path / "drafts", mouse_id=999, date="2026-03-03")
    second.update_session(goal="recordings not copied yet")

    checked = check_all([filled, second], db)

    assert [result.blocked for result in checked] == [False, True]
    assert db.con.execute("SELECT count(*) FROM sessions;").fetchone()[0] == 0


def test_a_blocked_session_does_not_stop_the_others(db, filled, tmp_path):
    second = Draft.start(tmp_path / "drafts", mouse_id=999, date="2026-03-03")
    second.update_session(goal="recordings not copied yet")

    results = submit_all([filled, second], db)

    assert results[0].submitted
    assert not results[1].submitted
    assert results[1].blocked

    # The one that could not go is still on disk to be fixed and sent again.
    assert second.path.exists()
    assert not filled.path.exists()


def test_each_session_of_a_day_lands_separately(db, filled, tmp_path, second_mouse):
    """
    A second mouse on the same day is its own session row.

    Its recording needs a start time of its own: `experiments.exp_start` is
    UNIQUE, so two recordings made at the same instant read as one experiment
    and the second is skipped as already ingested. Two mice cannot share a
    scope at the same second anyway.
    """
    shutil.copytree(second_mouse, db.main_folder, dirs_exist_ok=True)

    second = Draft.start(
        tmp_path / "drafts", mouse_id=2, date=EXP_START.date().isoformat()
    )
    second.update_session(goal="second mouse of the day")
    second.set_experiment(EXP, fov_depth_um=-70.0)

    results = submit_all([filled, second], db)

    assert all(result.submitted for result in results), [r.problems for r in results]
    assert {row[0] for row in db.con.execute("SELECT mouse_id FROM sessions;")} == {
        1,
        2,
    }


def test_recordings_named_for_another_mouse_are_refused(db, filled, tmp_path):
    """
    Ingestion takes the animal from the file name, not the folder. If they
    disagree the recordings land under one mouse and the annotations look for
    another, so nothing would connect and nothing would say so.
    """
    second = Draft.start(
        tmp_path / "drafts", mouse_id=2, date=EXP_START.date().isoformat()
    )
    second.update_session(goal="folder says m2, files say m1")

    # Folder renamed, file names left alone -- which is what happens when a
    # recording is started before the mouse on the rig is changed over.
    (db.main_folder / EXP_DATE / MOUSE).rename(db.main_folder / EXP_DATE / "m2")

    problems = [problem for problem in check(second, db) if problem.blocking]

    assert any("named m1" in problem.what for problem in problems)


def test_an_empty_day_reports_rather_than_raises(db):
    assert submit_all([], db) == []
    assert check_all([], db) == []
