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
from odyn.app.submit import Problem, check, report, submit
from generate_data import generate
from generate_sync import generate_sync

START = datetime(2026, 1, 1, 10, 0, 0)


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #


def test_forms_come_from_the_registry(tmp_path):
    db = Database(tmp_path, can_create=True)

    fields = form_fields(db, "loop")
    keys = [f.key for f in fields]

    assert keys[0] == "objective" and fields[0].options == ("10", "20")
    assert "pmt_gain" in keys and "treatment" in keys
    # what a file fills in is not asked for
    assert all(f.editable for f in fields)
    assert "laser_power_920" not in keys

    db.retire_annotation_key(applies_to="loop", key="treatment")
    assert "treatment" not in [f.key for f in form_fields(db, "loop")]

    db.add_annotation_key(
        applies_to="loop",
        key="new",
        label="New",
        value_type="enum",
        allowed_values=["a"],
        description="d",
        required=True,
    )
    new = form_fields(db, "loop")[-1]
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
    draft.set_part("grabs")
    draft.set_vials([3, 1, 1])

    again = Draft.load(draft.path)
    assert again.mouse_id == 442 and again.date == "2026-07-08"
    assert again.folder == "20260708/m442"
    assert again.session == {"goal": "x", "mouse_weight_g": 24.0}
    assert [e["name"] for e in again.parts] == ["e1", "grabs"]
    assert again.part("e1")["kind"] == "loop"
    assert again.part("grabs")["kind"] == "grab"
    assert again.vial_ids == [1, 3]
    assert again.user == "tester"

    # `None` clears, and removing a part drops it
    again.update_session(goal=None).remove_part("grabs")
    assert Draft.load(draft.path).session == {"mouse_weight_g": 24.0}
    assert len(Draft.load(draft.path).parts) == 1


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


def test_submit_does_the_whole_day(share):
    db, draft = share

    # The sync file's frame timings should match: acquisitions start where
    # the generator put the bursts, which is not where the TIFFs say. The
    # processing matches by order and only flags the skew, so it still lands.
    result = submit(db, draft)

    assert result.submitted, result.error
    assert result.written.session_id == db.session_for(1, "2026-01-01")
    assert list(result.written.loops) == ["e1"]
    assert result.written.annotations == 4
    assert result.written.processed == ["e1"]

    exp_id = result.written.loops["e1"]
    assert db.annotations_for("loop").loc[exp_id, "pmt_gain"] == 70
    assert (
        db.annotations_for("session").loc[result.written.session_id, "goal"] == "test"
    )
    assert list(db.loop_vials.loc[[exp_id], "vial_id"]) == draft.vial_ids
    assert len(db.sync_blocks) == 3
    assert result.written.process_failed == {}

    assert draft.path.parent.name == "submitted"
    assert Draft.pending(draft.path.parent.parent / "drafts") == []


def test_a_blocked_submit_writes_nothing(share):
    db, draft = share
    draft.set_part("e2", objective=20)

    result = submit(db, draft)

    assert not result.submitted and result.blocked
    assert result.written is None
    assert len(db.sessions) == 0
    assert (
        Draft.pending(draft.path.parent) == [draft]
        or Draft.pending(draft.path.parent)[0].path == draft.path
    )


def test_a_second_submit_does_not_ingest_twice(share):
    db, draft = share

    first = submit(db, draft)
    assert first.submitted

    # Something more to say about the same day
    draft.set_part("e1", treatment="updated")
    second = submit(db, draft, force=True)

    assert second.submitted, second.error
    assert second.written.loops == first.written.loops
    assert len(db.loops) == 1
    assert db.annotations_for("loop").iloc[0]["treatment"] == "updated"
    assert db.latest_calls("add_loop").shape[0] == 1


def test_conditions_are_checked_and_written_per_acquisition(share):
    db, draft = share  # e1 has three TIFFs, numbered 1 to 3

    def messages(conditions):
        draft.set_part("e1", conditions=conditions)
        return [str(p) for p in check(db, draft) if "condition " in str(p)]

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
    result = submit(db, draft)
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
    submit(db, draft, force=True)
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
    assert early.written is None and not on_share.exists()
    assert "e1: still recording (stop it first)" in [str(p) for p in early.problems]

    result = submit(db, draft)
    assert result.submitted, result.error
    assert result.written.processed == ["e1"]
    assert on_share.is_file() and local.is_file()
    assert draft.recordings("e1")[0]["copied"]


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
    assert draft.part("e1") == {"name": "e1", "kind": "loop", "submitted_id": 4}
    assert draft.is_removed("e2")

    # its undo entry still applies
    restored, part = Removals(tmp_path).undo()
    assert part == "e2" and restored.part("e2") is not None
    assert json.loads(path.read_text())["version"] == 4


# The 2026-08-11 z-stack of m357, as its header has it
STACK = dict(slices=21, step_um=5, top_um=-5.7, power=35, length_um=-300)
STACK["reference_um"] = -55.7


@pytest.fixture
def a_day(tmp_path):
    """
    A loop, two grabs with a sync file running through them, and a z-stack
    of m1 on 2026-03-04, and a draft for the day with each part added.
    """
    day = datetime(2026, 3, 4, 9, 0, 0)
    size = dict(frames=56, height=32, width=40)
    generate(tmp_path, acquisitions=2, start=day + timedelta(hours=1), **size)
    generate(
        tmp_path,
        acquisitions=2,
        exp="grabs",
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

    # Grabs at 0 s and 10 s; the file's bursts at 0 s and 9 s
    sync = "20260304/m1/grabs/sync/m1_grabs.h5"
    generate_sync(
        tmp_path / sync,
        acquisitions=2,
        frames=56,
        start=day - timedelta(seconds=2),
        missing_pulse_in=(0, 1),
    )

    db = Database(tmp_path, can_create=True, user="tester")
    db.set_mouse(mouse_id=1)

    draft = Draft.start(tmp_path / "drafts", mouse_id=1, date="2026-03-04")
    draft.set_part("e1", objective=20)
    draft.set_part("grabs", objective=10)
    draft.add_recording(
        "grabs",
        {"file": "m1_grabs.h5", "local": "x", "share": sync, "copied": "yes"},
    )
    draft.set_part("zstacks", objective=20)

    return db, draft


def test_grabs_and_zstacks_notes_are_checked(a_day):
    db, draft = a_day
    draft.set_file("zstacks", "20260304_m1_zstacks_00001.tif", loop="e9")
    draft.set_file("grabs", "late_00009.tif", pmt_gain="high")

    messages = [str(p) for p in check(db, draft)]
    assert (
        "zstacks/20260304_m1_zstacks_00001.tif: the z-stack loop e9 is not listed"
        in messages
    )
    assert any(
        "grabs/late_00009.tif: PMT gain should be a number" in m for m in messages
    )
    assert (
        "grabs/late_00009.tif: not in the server's folder. (Not required to submit.)"
        in messages
    )


def test_submit_adds_the_grabs_and_zstacks_with_their_notes(a_day):
    db, draft = a_day
    grab, stack = "20260304_m1_grabs_00002.tif", "20260304_m1_zstacks_00001.tif"
    draft.set_file("grabs", grab, goal="reference", condition="ket/xyl")
    draft.set_file("zstacks", stack, loop="e1", z_landmark="surface of right bulb")

    assert not any(p.blocking for p in check(db, draft))
    result = submit(db, draft)
    assert result.submitted, result.error

    written = result.written
    assert list(written.loops) == ["e1"]
    assert sorted(written.grabs) == ["20260304_m1_grabs_00001.tif", grab]
    assert list(written.zstacks) == [stack]

    grab_id, stack_id = written.grabs[grab], written.zstacks[stack]
    notes = db.annotations_for("grab").loc[grab_id]
    assert (notes["goal"], notes["condition"]) == ("reference", "ket/xyl")
    assert db.zstacks.loc[stack_id, "loop_id"] == written.loops["e1"]
    assert (
        db.annotations_for("zstack").loc[stack_id, "z_landmark"]
        == "surface of right bulb"
    )
    assert draft.files("grabs")[grab]["submitted_id"] == grab_id

    # the grabs' sync file, matched by time
    assert written.processed == ["grabs/m1_grabs.h5"]
    assert db.grabs["sync_block"].tolist() == [0, 1]

    # a second submit ingests nothing new
    again = Draft.load(draft.path)
    assert submit(db, again, force=True).submitted
    assert len(db.grabs) == 2 and len(db.zstacks) == 1


def test_an_error_reads_as_odyn_wrote_it():
    from odyn.app.submit import described

    assert described(RuntimeError("'x' is not about m1 on 2026-01-01.")) == (
        "'x' is not about m1 on 2026-01-01"
    )
    assert described(FileNotFoundError("No olfactometer log")) == "No olfactometer log"
    assert described(KeyError("mouse_id")) == "KeyError: 'mouse_id'"
    assert described(ZeroDivisionError()) == "ZeroDivisionError"
