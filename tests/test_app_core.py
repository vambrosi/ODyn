"""
The entry app without Qt: fields from the registry, drafts on disk, and
check/submit against a database with a synthetic recording.
"""

import json

from datetime import datetime, timedelta

import pytest

from odyn import Database
from odyn.app.draft import Draft, Removals
from odyn.app.fields import Field, form_fields, missing_required
from odyn.app.panel import ANSI
from odyn.app.submissions import queue_folder
from odyn.app.submit import Problem, check, check_files, ingest, report, submit
from generate_data import generate
from generate_sync import generate_sync

START = datetime(2026, 1, 1, 10, 0, 0)


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #


def test_forms_come_from_the_registry(tmp_path):
    db = Database(tmp_path, can_create=True)

    fields = form_fields(db, "experiment")
    keys = [f.key for f in fields]

    assert keys[0] == "objective" and fields[0].options == ("10", "20")
    assert "pmt_gain" in keys and "treatment" in keys
    # what a file fills in is not asked for
    assert all(f.editable for f in fields)
    assert "laser_power_920" not in keys

    db.retire_annotation_key(applies_to="experiment", key="treatment")
    assert "treatment" not in [f.key for f in form_fields(db, "experiment")]

    db.add_annotation_key(
        applies_to="experiment",
        key="new",
        label="New",
        value_type="enum",
        allowed_values=["a"],
        description="d",
        required=True,
    )
    new = form_fields(db, "experiment")[-1]
    assert new.options == ("a",) and new.required

    assert form_fields(db, "vial") == []


@pytest.mark.parametrize(
    "field_, written, expected",
    [
        (Field("k", "K", "integer", "", prefix="m"), "m442", 442),
        (Field("k", "K", "integer", ""), "3.0", 3),
        (Field("k", "K", "real", "", unit="g"), "25.1", 25.1),
        (Field("k", "K", "boolean", ""), "Yes", True),
        (Field("k", "K", "boolean", ""), True, True),
        (Field("k", "K", "date", ""), "2026-07-08", "2026-07-08"),
        (Field("k", "K", "enum", "", options=("a", "b")), "B", "b"),
        (Field("k", "K", "text", ""), "  x ", "x"),
        (Field("k", "K", "text", ""), "", None),
    ],
)
def test_typed_values_are_read_as_their_type(field_, written, expected):
    assert field_.parse(written) == expected


@pytest.mark.parametrize(
    "field_, written",
    [
        (Field("k", "K", "integer", ""), "3.5"),
        (Field("k", "K", "real", ""), "heavy"),
        (Field("k", "K", "boolean", ""), "maybe"),
        (Field("k", "K", "date", ""), "July 8"),
        (Field("k", "K", "enum", "", options=("a",)), "c"),
    ],
)
def test_wrong_values_name_the_field(field_, written):
    with pytest.raises(ValueError, match="K should be"):
        field_.parse(written)


def test_missing_required_and_display():
    fields = [
        Field("a", "A", "real", "", required=True),
        Field("b", "B", "boolean", ""),
    ]
    assert [f.key for f in missing_required(fields, {"b": True})] == ["a"]
    assert fields[1].show(True) == "yes" and fields[0].show(None) == ""
    assert Field("k", "K", "integer", "").show(3.0) == "3"
    assert Field("k", "K", "real", "", unit="g").prompt == "K (g)"


# --------------------------------------------------------------------------- #
# Drafts
# --------------------------------------------------------------------------- #


def test_a_draft_is_saved_on_every_change(tmp_path):
    draft = Draft.start(tmp_path, user="tester")
    assert draft.path.exists() and draft.is_untouched

    draft.identify(mouse_id="442", date="2026-07-08")
    draft.update_session(goal="x", mouse_weight_g=24.0)
    draft.set_part("e1", objective=20, pmt_gain=70)
    draft.set_part("e2", exp_type="grab")
    draft.set_part("zstacks")
    draft.set_vials([3, 1, 1])

    again = Draft.load(draft.path)
    assert again.mouse_id == 442 and again.date == "2026-07-08"
    assert again.folder == "20260708/m442"
    assert again.session == {"goal": "x", "mouse_weight_g": 24.0}
    assert [e["name"] for e in again.parts] == ["e1", "e2", "zstacks"]
    assert again.part("e1")["kind"] == "experiment"
    assert again.part("e2")["exp_type"] == "grab"
    assert again.part("zstacks")["kind"] == "zstack"
    assert again.describe("e1") == "loop 20260708/m442/e1"
    assert again.describe("e2") == "grab 20260708/m442/e2"
    assert again.vial_ids == [1, 3]
    assert again.user == "tester"

    # `None` clears, and removing a part drops it
    again.update_session(goal=None).remove_part("e2")
    assert Draft.load(draft.path).session == {"mouse_weight_g": 24.0}
    assert len(Draft.load(draft.path).parts) == 2


def test_pending_skips_the_unreadable_and_the_archived(tmp_path):
    folder = tmp_path / "drafts"
    good = Draft.start(folder)
    (folder / "20260101-000000.json").write_text("{not json")
    (folder / "20260102-000000.json").write_text(json.dumps({"version": 99}))
    (folder / "notes.json").write_text("{}")

    assert [d.path for d in Draft.pending(folder)] == [good.path]
    assert [p.name for p, _ in Draft.unreadable(folder)] == [
        "20260101-000000.json",
        "20260102-000000.json",
    ]

    # Beside the drafts folder, so clearing the drafts keeps the record
    moved = good.archive()
    assert moved.parent == tmp_path / "submitted" and good.path == moved
    assert Draft.pending(folder) == []

    # A save after archiving stays archived
    good.update_session(goal="late")
    assert Draft.pending(folder) == []


def test_two_drafts_in_one_second_get_different_names(tmp_path):
    first = Draft.start(tmp_path)
    second = Draft.start(tmp_path)
    assert first.path != second.path


# --------------------------------------------------------------------------- #
# Check and submit
# --------------------------------------------------------------------------- #


@pytest.fixture
def share(tmp_path):
    """A share with one recorded loop, its sync file, and a database."""
    generate(tmp_path, acquisitions=3, frames=56, height=32, width=40)

    # A sync file whose bursts line up with the TIFFs' acquisition starts
    generate_sync(tmp_path / "20260101/m1/e1/sync/m1_e1.h5", acquisitions=3, frames=56)

    db = Database(tmp_path, can_create=True, user="tester")
    db.add_user(user_name="tester")
    db.set_mouse(mouse_id=1)
    db.add_panel(panel_name="p", positions={1: 0})
    vials = db.add_vials(
        panel_name="p",
        vials=[
            {
                "vial_position": 1,
                "odor_id": 0,
                "vial_sccm": 200,
                "total_sccm": 1500,
                "total_volume_ml": 5,
            },
        ],
    )

    draft = Draft.start(tmp_path / "drafts", mouse_id=1, date="2026-01-01")
    draft.update_session(goal="test", mouse_weight_g=24)
    draft.set_part("e1", objective=20, pmt_gain=70, treatment="fine")
    draft.set_vials(vials)

    return db, draft


def test_problems_are_listed_one_block_per_page():
    problems = [
        Problem("grabs/a.tif", "PMT gain should be a number"),
        Problem("e1", "no TIFFs in 'e1/raw'"),
        Problem("vials", "the vials come from more than one panel"),
        Problem("session", "Goal is not filled in", blocking=False),
        Problem("grabs/b.tif", "not in the server's folder"),
    ]

    text = report("m3 2026-10-07", problems, ["e1", "grabs"], "fix errors to submit")

    # in a column seven wide, red for errors and yellow for warnings
    assert ANSI.sub("", text).split("\n") == [
        "CHECKED  m3 2026-10-07 (4 errors, 1 warning)",
        "",
        "  vials  the vials come from more than one panel",
        "session  Goal is not filled in",
        "",
        "     e1  no TIFFs in 'e1/raw'",
        "",
        "  grabs  PMT gain should be a number (a.tif)",
        "  grabs  not in the server's folder (b.tif)",
        "",
        " RESULT  fix errors to submit",
    ]
    assert "\033[1;33msession" in text and "\033[1;31m     e1" in text

    assert ANSI.sub("", report("m3 2026-10-07", [], ["e1"], "ready")).split("\n") == [
        "CHECKED  m3 2026-10-07 (0 errors, 0 warnings)",
        "",
        " RESULT  ready",
    ]


def test_check_lists_what_is_wrong(share):
    db, draft = share

    # Submitting needs someone from the Users page, not the computer's login
    object.__setattr__(db, "user", "lab-pc")
    assert str(check(db, draft)[0]) == "user: didn't pick a user yet (Users page)"
    object.__setattr__(db, "user", "tester")

    # Only the advisory: the synthetic share has no olfactometer files
    assert [str(p) for p in check(db, draft)] == [
        "e1: no olfactometer Events file found. (Not required to submit.)"
    ]

    draft.set_part("e2", objective=None, pmt_gain="deep")
    draft.set_vials([99])
    messages = [str(p) for p in check(db, draft)]

    assert any("e2: didn't pick an objective" in m for m in messages)
    assert any("e2: PMT gain should be a number" in m for m in messages)
    assert any("e2: folder 'e2' is not on the server" in m for m in messages)
    assert any("vials: no vials with ids 99" in m for m in messages)
    assert [m for m in messages if "Not required to submit" in m] == [
        "e1: no olfactometer Events file found. (Not required to submit.)"
    ]

    draft.remove_part("e2").set_vials([]).identify(mouse_id=7)
    messages = [str(p) for p in check(db, draft)]
    assert messages[0].startswith("session: mouse 7 is not registered")
    assert any("folder '20260101/m7' is not on the server" in m for m in messages)


def test_the_recordings_on_the_share_are_checked_apart(share, tmp_path):
    db, draft = share
    raw = tmp_path / "20260101/m1/e1/raw"
    for tiff in raw.glob("*.tif"):
        tiff.rename(tiff.with_suffix(".tif.away"))
    (tmp_path / "20260101/m1/e1/sync/extra.h5").write_bytes(b"")

    # The app's check does not look into the folders on the share...
    assert not any(p.blocking for p in check(db, draft))

    # ...the cluster's does, before it ingests
    assert [str(p) for p in check_files(db, draft)] == [
        "e1: no TIFFs in 'e1/raw'",
        "e1: 2 sync files in 'e1/sync'",
    ]
    result = ingest(db, draft)
    assert result.blocked and result.written is None

    # A folder missing from the share is said once
    draft.set_part("e2", objective=20)
    problems = [str(p) for p in ingest(db, draft).problems]
    assert problems.count("e2: folder 'e2' is not on the server yet") == 1


def test_ingest_leaves_the_draft_where_it_is(share):
    db, draft = share
    where = draft.path

    result = ingest(db, draft)

    assert result.submitted, result.error
    assert draft.path == where and where.exists()
    assert list(result.written.experiments) == ["e1"]


def test_submit_hands_the_draft_to_the_cluster(share):
    db, draft = share
    drafts = draft.path.parent

    result = submit(db, draft)

    assert result.submitted, result.error
    assert result.queued.parent == queue_folder(db.main_folder)
    assert Draft.load(result.queued).data["submitted_by"] == "tester"

    # Nothing in the database yet; the draft is set aside here
    assert len(db.sessions) == 0 and len(db.experiments) == 0
    assert draft.path.parent.name == "submitted"
    assert Draft.pending(drafts) == []


def test_ingest_does_the_whole_day(share):
    db, draft = share

    # The sync file's frame timings should match: acquisitions start where
    # the generator put the bursts, which is not where the TIFFs say. The
    # processing matches by order and only flags the skew, so it still lands.
    result = ingest(db, draft)

    assert result.submitted, result.error
    assert result.written.session_id == db.session_for(1, "2026-01-01")
    assert list(result.written.experiments) == ["e1"]
    assert result.written.annotations == 4
    assert result.written.processed == ["e1"]

    exp_id = result.written.experiments["e1"]
    assert db.annotations_for("experiment").loc[exp_id, "pmt_gain"] == 70
    assert (
        db.annotations_for("session").loc[result.written.session_id, "goal"] == "test"
    )
    assert list(db.experiment_vials.loc[[exp_id], "vial_id"]) == draft.vial_ids
    assert len(db.sync_blocks) == 3
    assert result.written.process_failed == {}


def test_a_blocked_submit_writes_nothing(share):
    db, draft = share
    draft.set_part("e2", objective=20)

    result = submit(db, draft)

    assert not result.submitted and result.blocked
    assert result.queued is None
    assert not queue_folder(db.main_folder).exists()
    assert (
        Draft.pending(draft.path.parent) == [draft]
        or Draft.pending(draft.path.parent)[0].path == draft.path
    )


def test_a_second_ingest_does_not_add_twice(share):
    db, draft = share

    first = ingest(db, draft)
    assert first.submitted

    # Something more to say about the same day
    draft.set_part("e1", treatment="updated")
    second = ingest(db, draft, force=True)

    assert second.submitted, second.error
    assert second.written.experiments == first.written.experiments
    assert len(db.experiments) == 1
    assert db.annotations_for("experiment").iloc[0]["treatment"] == "updated"
    assert db.latest_calls("add_experiment").shape[0] == 1


def test_conditions_are_checked_and_written_per_acquisition(share):
    db, draft = share  # e1 has three TIFFs, numbered 1 to 3

    def messages(conditions):
        draft.set_part("e1", conditions=conditions)
        found = check(db, draft) + check_files(db, draft)
        return [str(p) for p in found if "condition " in str(p)]

    assert messages([{"start": 2, "end": 1, "condition": "baseline"}]) == [
        "e1: condition baseline 2 to 1 ends before it starts"
    ]
    assert messages(
        [
            {"start": 1, "end": 2, "condition": "baseline"},
            {"start": 2, "end": 3, "condition": "ket/xyl"},
        ]
    ) == ["e1: condition ket/xyl 2 to 3 overlaps baseline 1 to 2 at acquisition 2"]
    assert messages([{"start": 2, "end": 5, "condition": "ket/xyl"}]) == [
        "e1: condition ket/xyl 2 to 5 names acquisitions with no TIFF: 4-5"
    ]

    draft.set_part(
        "e1",
        conditions=[
            {"start": 1, "end": 1, "condition": "baseline"},
            {"start": 2, "end": 3, "condition": "ket/xyl"},
        ],
    )
    result = ingest(db, draft)
    assert result.submitted, result.error

    acquisitions = db.acquisitions.sort_values("acq_start")
    conditions = db.annotations_for("acquisition")["condition"]
    assert [conditions[a] for a in acquisitions.index] == [
        "baseline",
        "ket/xyl",
        "ket/xyl",
    ]

    # a resubmit with the same conditions does not write them again
    def written():
        query = "SELECT COUNT(*) AS n FROM annotations WHERE target_type = ?;"
        return int(db.from_query(query, ["acquisition"])["n"].iloc[0])

    assert written() == 3
    ingest(db, draft, force=True)
    assert written() == 3


def test_submit_copies_a_recording_kept_on_this_computer(share, tmp_path):
    db, draft = share

    # the recorder wrote it locally and the share was away: it never got there
    on_share = tmp_path / "20260101/m1/e1/sync/m1_e1.h5"
    local = tmp_path / "local" / "20260101/m1/e1/m1_e1.h5"
    local.parent.mkdir(parents=True)
    on_share.rename(local)
    draft.add_recording(
        "e1",
        {
            "file": "m1_e1.h5",
            "local": str(local),
            "share": "20260101/m1/e1/sync/m1_e1.h5",
        },
    )

    messages = [str(p) for p in check(db, draft)]
    assert any("m1_e1.h5 was not copied to the server" in m for m in messages)
    assert not any("no sync file" in m for m in messages)

    # one still being written is left alone, and blocks the submit
    early = submit(db, draft, busy={str(local)})
    assert early.queued is None and not on_share.exists()
    assert "e1: still recording (stop it first)" in [str(p) for p in early.problems]

    result = submit(db, draft)
    assert result.submitted, result.error
    assert on_share.is_file() and local.is_file()
    assert draft.recordings("e1")[0]["copied"]

    # The queued copy knows it was copied, so the cluster processes it
    queued = Draft.load(result.queued)
    assert queued.recordings("e1")[0]["copied"]
    assert ingest(db, queued).written.processed == ["e1"]


def test_a_recorded_session_keeps_its_mouse_and_date(tmp_path):
    draft = Draft.start(tmp_path, mouse_id=442, date="2026-09-29")
    draft.set_part("e1").identify(mouse_id=443)  # not recorded yet: fine
    draft.add_recording("e1", {"file": "f.h5", "local": "x", "share": "y"})

    with pytest.raises(ValueError, match="fixed"):
        draft.identify(date="2026-09-30")

    draft.identify(mouse_id=443, session_id=5)  # the same, and the row: fine
    assert draft.recorded and draft.session_id == 5


def test_a_removed_part_is_hidden_and_keeps_its_name(tmp_path):
    draft = Draft.start(tmp_path, mouse_id=442, date="2026-09-29")
    draft.set_part("e1", pmt_gain=70).set_part("e2")
    draft.remove_part("e1")

    again = Draft.load(draft.path)
    assert [e["name"] for e in again.parts] == ["e2"]
    assert again.part("e1") is None and again.is_removed("e1")
    assert again.names_taken() == ["e1", "e2"]

    # Cannot use the same name, even after removal
    with pytest.raises(ValueError, match="was removed"):
        again.set_part("e1")

    assert again.restore_part("e1")
    assert again.part("e1")["pmt_gain"] == 70
    assert "removed" not in again.part("e1")
    assert not again.restore_part("e1")  # nothing left to restore


def test_undo_restores_the_latest_removal_that_still_applies(tmp_path):
    folder = tmp_path / "drafts"
    removals = Removals(folder)
    assert removals.next() is None and removals.undo() is None

    kept = Draft.start(folder, mouse_id=1)
    kept.set_part("e1")
    gone = Draft.start(folder, mouse_id=2)
    other = Draft.start(folder, mouse_id=3)
    submitted = Draft.start(folder, mouse_id=4)

    kept.remove_part("e1")
    removals.push(kept, "e1")
    gone.remove()
    removals.push(gone)
    other.remove()
    removals.push(other)
    submitted.remove()
    removals.push(submitted)
    assert [d.path for d in Draft.pending(folder)] == [kept.path]

    # the latest ones no longer apply: moved away, restored another way,
    # deleted by hand; each is dropped on the way to the one that does
    submitted.restore()
    submitted.archive()
    other.restore()
    gone.path.unlink()

    draft, part = removals.next()
    assert draft.path == kept.path and part == "e1"
    assert len(json.loads(removals.path.read_text())) == 1  # the rest dropped

    # restored on the caller's own copy of the draft, which sees it
    assert removals.undo([kept]) == (kept, "e1")
    assert kept.part("e1") is not None
    assert Draft.load(kept.path).part("e1") is not None
    assert removals.next() is None and removals.undo() is None


def test_an_undone_session_is_pending_again(tmp_path):
    removals = Removals(tmp_path)
    draft = Draft.start(tmp_path, mouse_id=7).update_session(goal="keep me")
    draft.remove()
    removals.push(draft)
    assert Draft.pending(tmp_path) == []
    assert Draft.unreadable(tmp_path) == []  # the stack is not a draft

    restored, part = removals.undo()
    assert part is None and restored.session == {"goal": "keep me"}
    assert [d.path for d in Draft.pending(tmp_path)] == [draft.path]


def _rig_file(path, monkeypatch, **options) -> dict:
    """A sync file at the recorder's own sample rate, as the rig writes them."""
    import generate_sync

    from odyn.app.recorder import RATE

    monkeypatch.setattr(generate_sync, "RATE_HZ", RATE)

    return generate_sync.generate_sync(path, **options)


def test_a_replayed_file_decodes_like_the_original(tmp_path, monkeypatch):
    import h5py

    from datetime import timedelta

    from odyn.app.recorder import RATE, Recording, ReplaySource
    from odyn.sync import decode

    original = tmp_path / "m7_e1_rig.h5"
    _rig_file(original, monkeypatch, acquisitions=3, frames=20, mouse="m7")
    source = ReplaySource(original, speed=1e6)  # all of it at the first read
    recording = Recording(
        tmp_path / "replayed.h5", mouse="m7", experiment="e1", source=source
    )
    recording.start()

    while not source.finished:
        recording.tick()

    recording.stop()

    with h5py.File(original) as rig, h5py.File(recording.path) as replayed:
        n = rig["2pFrameSync"].shape[0]
        assert replayed.attrs["n_samples"] == n
        assert replayed.attrs["replayed_from"] == original.name
        assert replayed.attrs["replay_speed"] == 1e6

        # the original's clock, so the decode's skew check sees the real day
        started = datetime.fromisoformat(rig.attrs["start_time"])
        assert replayed.attrs["start_time"] == started.isoformat()
        assert (
            replayed.attrs["stop_time"]
            == (started + timedelta(seconds=n / RATE)).isoformat()
        )

        for name in ("2pFrameSync", "odorPulse", "cameraFrameSync"):
            assert (replayed[name][:] == rig[name][:]).all()

    assert decode(recording.path).blocks == decode(original).blocks


def test_a_replay_needs_the_recorders_sample_rate(tmp_path):
    from generate_sync import generate_sync

    from odyn.app.recorder import ReplaySource

    generate_sync(tmp_path / "slow.h5")  # 1000 samples/s

    with pytest.raises(ValueError, match="1000 samples/s"):
        ReplaySource(tmp_path / "slow.h5")

    with pytest.raises(ValueError, match="above 0"):
        ReplaySource(tmp_path / "slow.h5", speed=0)


def test_a_version_3_draft_is_read_with_parts(tmp_path):
    path = tmp_path / "20261001-090000.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "mouse_id": 442,
                "date": "2026-10-01",
                "session": {},
                "experiments": [
                    {"name": "e1", "exp_type": "loop", "submitted_exp_id": 4},
                    {"name": "e2", "exp_type": "loop", "removed": True},
                ],
                "vial_ids": [],
            }
        )
    )
    Removals(tmp_path).path.write_text(
        json.dumps([{"draft": path.name, "experiment": "e2"}])
    )

    draft = Draft.load(path)
    assert draft.part("e1") == {
        "name": "e1",
        "kind": "experiment",
        "exp_type": "loop",
        "submitted_id": 4,
    }
    assert draft.is_removed("e2")

    # its undo entry still applies
    restored, part = Removals(tmp_path).undo()
    assert part == "e2" and restored.part("e2") is not None
    assert json.loads(path.read_text())["version"] == 5


def test_a_version_4_draft_has_experiments_and_no_grabs(tmp_path):
    path = tmp_path / "20261001-090000.json"
    path.write_text(
        json.dumps(
            {
                "version": 4,
                "mouse_id": 442,
                "date": "2026-10-01",
                "session": {},
                "parts": [
                    {"name": "e1", "kind": "loop", "objective": 20},
                    {"name": "grabs", "kind": "grab", "objective": 10},
                    {
                        "name": "zstacks",
                        "kind": "zstack",
                        "files": {"z_00001.tif": {"loop": "e1", "note": "x"}},
                    },
                ],
                "vial_ids": [],
            }
        )
    )

    draft = Draft.load(path)
    assert [e["name"] for e in draft.parts] == ["e1", "zstacks"]
    assert draft.part("e1")["kind"] == "experiment"
    assert draft.part("e1")["exp_type"] == "loop"
    assert draft.files("zstacks") == {"z_00001.tif": {"experiment": "e1", "note": "x"}}


# The 2026-08-11 z-stack of m357, as its header has it
STACK = dict(slices=21, step_um=5, top_um=-5.7, power=35, length_um=-300)
STACK["reference_um"] = -55.7


@pytest.fixture
def a_day(tmp_path):
    """
    A loop, a grab with its sync file, and a z-stack of m1 on 2026-03-04, and a
    draft for the day with each part added.
    """
    day = datetime(2026, 3, 4, 9, 0, 0)
    size = dict(frames=56, height=32, width=40)
    generate(tmp_path, acquisitions=2, start=day + timedelta(hours=1), **size)
    generate(
        tmp_path,
        acquisitions=1,
        exp="e2",
        acq_state="grab",
        start=day,
        z_um=-80,
        **size,
    )
    generate(
        tmp_path,
        acquisitions=1,
        exp="zstacks",
        acq_state="grab",
        start=day + timedelta(hours=2),
        zstack=STACK,
        **size,
    )

    # The grab's one burst, the sync file started 2 s before it
    sync = "20260304/m1/e2/sync/m1_e2.h5"
    generate_sync(
        tmp_path / sync,
        acquisitions=1,
        frames=56,
        start=day - timedelta(seconds=2),
        missing_pulse_in=(0,),
    )

    db = Database(tmp_path, can_create=True, user="tester")
    db.add_user(user_name="tester")
    db.set_mouse(mouse_id=1)

    draft = Draft.start(tmp_path / "drafts", mouse_id=1, date="2026-03-04")
    draft.set_part("e1", objective=20)
    draft.set_part("e2", exp_type="grab", objective=10)
    draft.add_recording(
        "e2", {"file": "m1_e2.h5", "local": "x", "share": sync, "copied": "yes"}
    )
    draft.set_part("zstacks", objective=20)

    return db, draft


def test_zstacks_notes_are_checked(a_day):
    db, draft = a_day
    draft.set_file("zstacks", "20260304_m1_zstacks_00001.tif", experiment="e9")
    draft.set_file("zstacks", "late_00009.tif", z_landmark="x")

    messages = [str(p) for p in check(db, draft) + check_files(db, draft)]
    assert (
        "zstacks/20260304_m1_zstacks_00001.tif: the z-stack's experiment e9 is "
        "not listed" in messages
    )
    assert (
        "zstacks/late_00009.tif: not in the server's folder. (Not required to "
        "submit.)" in messages
    )

    # A grab needs neither olfactometer files nor a sync file
    assert not [m for m in messages if m.startswith("e2")]


def test_the_type_must_be_what_scanimage_recorded(a_day):
    db, draft = a_day
    draft.set_part("e2", exp_type="loop")

    messages = [str(p) for p in check_files(db, draft)]
    assert "e2: the TIFFs were recorded as a grab, not a loop (change the type)" in (
        messages
    )


def test_ingest_adds_the_grab_and_zstacks_with_their_notes(a_day):
    db, draft = a_day
    stack = "20260304_m1_zstacks_00001.tif"
    draft.set_part(
        "e2",
        conditions=[{"start": 1, "end": 1, "condition": "ket/xyl"}],
        block_iti_s=10,
    )
    draft.set_file(
        "zstacks", stack, experiment="e2", z_landmark="surface of right bulb"
    )

    assert not any(p.blocking for p in check(db, draft))
    result = ingest(db, draft)
    assert result.submitted, result.error

    written = result.written
    assert list(written.experiments) == ["e1", "e2"]
    assert list(written.zstacks) == [stack]

    grab = written.experiments["e2"]
    assert db.experiments.loc[grab, "exp_type"] == "grab"
    assert written.processed == ["e2"]  # its sync file
    assert db.acquisitions.loc[
        db.acquisitions["exp_id"] == grab, "sync_block"
    ].tolist() == [0]

    # Its condition, on its one acquisition; no trial block for a grab
    conditions = db.annotations_for("acquisition")["condition"]
    assert conditions.dropna().tolist() == ["ket/xyl"]
    assert "block_iti_s" not in db.annotations_for("experiment").columns

    stack_id = written.zstacks[stack]
    assert db.zstacks.loc[stack_id, "exp_id"] == grab
    assert (
        db.annotations_for("zstack").loc[stack_id, "z_landmark"]
        == "surface of right bulb"
    )
    assert draft.files("zstacks")[stack]["submitted_id"] == stack_id

    # a second ingest adds nothing new
    again = Draft.load(draft.path)
    assert ingest(db, again, force=True).submitted
    assert len(db.experiments) == 2 and len(db.zstacks) == 1


def test_an_error_reads_as_odyn_wrote_it():
    from odyn.app.submit import described

    assert described(RuntimeError("The TIFFs x_* were recorded as m2.")) == (
        "The TIFFs x_* were recorded as m2"
    )
    assert described(FileNotFoundError("No olfactometer log")) == "No olfactometer log"
    assert described(KeyError("mouse_id")) == "KeyError: 'mouse_id'"
    assert described(ZeroDivisionError()) == "ZeroDivisionError"
