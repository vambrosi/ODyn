"""
Drafts: a session being filled in, before anyone submits it.

The app writes one of these all day in place of the spreadsheet, so the things
worth pinning down are that an interrupted day comes back intact, that the same
animal on the same day is one draft however it was reached, and that one bad
file cannot hide the good ones.
"""

import json
import re

from datetime import date

import pytest

from odyn.app.draft import DRAFT_VERSION, Draft


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "drafts"


@pytest.fixture
def draft(folder):
    return Draft.start(folder, mouse_id=442, date="2026-07-08")


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_a_draft_is_named_for_when_it_was_started(draft):
    """
    Not for the animal: a session is usually opened before anyone types the
    number, and the number can be corrected afterwards.
    """
    assert re.fullmatch(r"\d{8}-\d{6}\.json", draft.path.name)


def test_a_session_can_be_started_with_no_mouse(folder):
    started = Draft.start(folder)

    assert started.mouse_id is None
    assert started.date == date.today().isoformat()
    assert started.label.startswith("no mouse yet")


def test_the_mouse_can_be_named_afterwards(folder):
    started = Draft.start(folder)
    started.identify(mouse_id=442)

    assert Draft.load(started.path).mouse_id == 442


def test_naming_the_mouse_does_not_move_the_draft(folder):
    """The file keeps its name, so nothing it is open in has to follow it."""
    started = Draft.start(folder)
    started.update_session(goal="a real day of work")

    before = started.path
    started.identify(mouse_id=442)

    assert started.path == before
    assert Draft.load(before).session["goal"] == "a real day of work"


def test_two_unidentified_sessions_do_not_collide(folder):
    """Both are 'no mouse yet', so a name built from the animal would clash."""
    first, second = Draft.start(folder), Draft.start(folder)

    assert first.path != second.path
    assert len(Draft.pending(folder)) == 2


def test_correcting_the_mouse_cannot_overwrite_another_session(folder):
    """Two sessions of the same animal stay two drafts."""
    first = Draft.start(folder, mouse_id=442)
    second = Draft.start(folder, mouse_id=357)

    second.identify(mouse_id=442)

    assert first.path != second.path
    assert {d.mouse_id for d in Draft.pending(folder)} == {442}
    assert len(Draft.pending(folder)) == 2


def test_a_date_that_is_not_one_is_refused(folder):
    """A mistyped date would file the day under the wrong name silently."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        Draft.start(folder, date="08/07/2026")


def test_a_new_draft_is_empty(draft):
    assert draft.is_empty
    assert draft.experiments == []


# --------------------------------------------------------------------------- #
# Surviving the day
# --------------------------------------------------------------------------- #


def test_every_change_is_on_disk_immediately(folder, draft):
    """Nothing is held in memory waiting for a save the user might never do."""
    draft.update_session(mouse_weight_g=25.1)

    stored = json.loads(draft.path.read_text())

    assert stored["session"]["mouse_weight_g"] == 25.1


def test_an_interrupted_draft_comes_back(folder, draft):
    draft.update_session(goal="10x pre/post ket/xyl", headplate="A")
    draft.update_session(note="genteal drops before starting")
    draft.set_experiment("e1", fov_depth_um=-55)

    # The app died here.
    reopened = Draft.load(draft.path)

    assert reopened.session["headplate"] == "A"
    assert reopened.session["note"] == "genteal drops before starting"
    assert reopened.experiment("e1")["fov_depth_um"] == -55


def test_a_half_written_file_does_not_replace_a_good_one(draft):
    """
    Saving writes a temporary file and renames it, so a crash partway through
    costs the last edit rather than the whole day.
    """
    draft.update_session(goal="a real day of work")

    draft.path.with_suffix(".json.writing").write_text("{ truncated")

    assert Draft.load(draft.path).session["goal"] == "a real day of work"
    assert [entry.mouse_id for entry in Draft.pending(draft.path.parent)] == [442]


# --------------------------------------------------------------------------- #
# What it holds
# --------------------------------------------------------------------------- #


def test_the_note_is_one_block_of_text(draft):
    """
    Written as one field, not several. The separate cells in the spreadsheet
    were how a spreadsheet looks tidy, not a distinction anyone wanted.
    """
    draft.update_session(note="first day on scope\nadjusted thermal probe")
    draft.update_session(flag=True)

    assert draft.session["note"].splitlines() == [
        "first day on scope",
        "adjusted thermal probe",
    ]
    assert draft.session["flag"] is True


def test_a_rewritten_note_replaces_the_one_before(draft):
    """It is one field, so editing it is editing it."""
    draft.update_session(note="first day on scope")
    draft.update_session(note="first day on scope\nand the probe was adjusted")

    assert draft.session["note"].endswith("and the probe was adjusted")


def test_setting_a_field_to_none_clears_it(draft):
    """Clearing a box must remove the field, not store an empty one."""
    draft.update_session(headplate="A")
    draft.update_session(headplate=None)

    assert "headplate" not in draft.session


def test_experiments_stay_in_order(draft):
    """They are entered as the day goes, and read back as a list."""
    for name in ("e10", "e2", "e1"):
        draft.set_experiment(name)

    assert [entry["name"] for entry in draft.experiments] == ["e1", "e2", "e10"]


def test_an_experiment_is_updated_not_duplicated(draft):
    draft.set_experiment("e1", fov_depth_um=-55)
    draft.set_experiment("e1", objective=20)

    assert len(draft.experiments) == 1
    assert draft.experiment("e1") == {
        "name": "e1",
        "fov_depth_um": -55,
        "objective": 20,
    }


def test_an_experiment_can_be_removed(draft):
    draft.set_experiment("e1")
    draft.set_experiment("e2")
    draft.remove_experiment("e1")

    assert [entry["name"] for entry in draft.experiments] == ["e2"]


def test_the_panel_keeps_per_vial_dates(draft):
    """JSON has no integer keys, so positions come back as strings."""
    draft.set_panel(
        panel_name="print_v3", made_on="2026-07-06", vial_dates={4: "2026-07-21"}
    )

    assert draft.panel["panel_name"] == "print_v3"
    assert draft.panel["vial_dates"] == {"4": "2026-07-21"}


# --------------------------------------------------------------------------- #
# Listing what is unfinished
# --------------------------------------------------------------------------- #


def test_pending_lists_unfinished_drafts(folder):
    Draft.start(folder, mouse_id=442).update_session(goal="one")
    Draft.start(folder, mouse_id=357).update_session(goal="two")

    assert {entry.mouse_id for entry in Draft.pending(folder)} == {442, 357}


def test_a_submitted_draft_is_not_offered_again(folder, draft):
    draft.update_session(goal="done and submitted")
    archived = draft.archive()

    assert Draft.pending(folder) == []

    # Kept rather than deleted: it is what was typed, as opposed to what the
    # database made of it.
    assert archived.exists()


def test_one_bad_file_does_not_hide_the_good_ones(folder, draft):
    """
    The failure this prevents is the worst one available: someone opens the app
    after a crash, is told there is nothing unfinished, and loses a day.
    """
    draft.update_session(goal="a real day of work")

    (folder / "20260101-090000.json").write_text(
        json.dumps({"version": DRAFT_VERSION + 1, "mouse_id": 999})
    )
    (folder / "20260102-090000.json").write_text("{ not json at all")

    assert [entry.mouse_id for entry in Draft.pending(folder)] == [442]

    broken = {path.name for path, _ in Draft.unreadable(folder)}

    assert broken == {"20260101-090000.json", "20260102-090000.json"}


def test_a_draft_from_a_newer_app_is_refused(folder):
    """Reading the parts we recognize would quietly drop the rest."""
    path = folder / "20260101-090000.json"
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": DRAFT_VERSION + 1, "mouse_id": 999}))

    with pytest.raises(ValueError, match="newer version"):
        Draft.load(path)


def test_unrelated_files_are_ignored(folder, draft):
    draft.save()
    (folder / "notes.txt").write_text("not a draft")
    (folder / "20260103-090000.json.writing").write_text("{ half written")

    assert len(Draft.pending(folder)) == 1
    assert Draft.unreadable(folder) == []


def test_no_folder_yet_is_not_an_error(tmp_path):
    """First run, before anyone has drafted anything."""
    assert Draft.pending(tmp_path / "nothing" / "here") == []
