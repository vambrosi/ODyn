"""
The queue of submitted drafts: what the cluster's tick waits for, adds, and
refuses (`odyn.app.submissions`).
"""

from datetime import datetime, timedelta

from odyn import Database
from odyn.app.draft import Draft
from odyn.app.submissions import (
    add_now,
    enqueue,
    process_submissions,
    queue_folder,
    queued,
    submit_again,
    take_back,
)
from odyn.cluster import cluster_folder, status, tick
from generate_data import generate
from generate_sync import generate_sync

SYNC = "20260101/m1/e1/sync/m1_e1.h5"
RAW = "20260101/m1/e1/raw"
T0 = datetime(2026, 1, 1, 20, 0, 0)
SETTLED = T0 + timedelta(minutes=11)


def a_day(tmp_path, *, acqs_per_loop=3, bursts=3, objective=20):
    """
    Three TIFFs of loop e1 (ScanImage set to `acqs_per_loop`), a sync file
    with `bursts`, and the draft of that day, queued as Ana's.
    """
    generate(
        tmp_path,
        acquisitions=3,
        frames=56,
        height=32,
        width=40,
        acqs_per_loop=acqs_per_loop,
    )
    generate_sync(tmp_path / SYNC, acquisitions=bursts, frames=56)

    db = Database(tmp_path, can_create=True, user="ana")
    db.add_user(user_name="ana")
    db.set_mouse(mouse_id=1)

    draft = Draft.start(tmp_path / "drafts", mouse_id=1, date="2026-01-01")
    draft.set_part("e1", objective=objective)
    path = enqueue(draft, tmp_path, user="ana")

    return db, draft, path


def state_of(tmp_path):
    [(_, state)] = queued(tmp_path)
    return state


def test_a_complete_draft_is_added_once_the_copy_settles(tmp_path):
    db, draft, path = a_day(tmp_path)

    assert process_submissions(tmp_path, now=T0) == [
        "20260101/m1: waiting for the copy to finish"
    ]
    assert len(db.experiments) == 0

    [line] = process_submissions(tmp_path, now=SETTLED)
    assert line == "20260101/m1: added as session 1"

    # Added as the person who submitted it, and moved out of the queue
    assert len(db.experiments) == 1
    assert db.latest_calls("add_experiment").iloc[0]["user"] == "ana"
    assert queued(tmp_path) == []
    done = queue_folder(tmp_path) / "done"
    assert (done / path.name).exists()
    assert path.name == f"{draft.path.stem}-ana.json"  # two people's never clash
    assert '"state": "done"' in (done / f"{path.stem}.state.json").read_text()

    # The queued copy kept what ingest made of it
    assert Draft.load(done / path.name).part("e1")["submitted_id"] == 1


def test_a_loop_waits_for_all_its_tiffs(tmp_path):
    db, _, _ = a_day(tmp_path, acqs_per_loop=5, bursts=5)

    assert process_submissions(tmp_path, now=SETTLED) == [
        "20260101/m1: waiting: e1: 3 of 5 TIFFs"
    ]
    assert state_of(tmp_path).parts == {"e1": "3 of 5 TIFFs"}
    assert len(db.experiments) == 0


def test_a_loop_stopped_early_is_complete_with_its_bursts(tmp_path):
    db, _, _ = a_day(tmp_path, acqs_per_loop=5, bursts=3)

    process_submissions(tmp_path, now=T0)
    [line] = process_submissions(tmp_path, now=SETTLED)

    assert line.endswith("added as session 1")
    assert len(db.acquisitions) == 3


def test_nothing_new_for_12_hours_fails_until_the_files_change(tmp_path):
    a_day(tmp_path, acqs_per_loop=5, bursts=5)

    process_submissions(tmp_path, now=T0)
    [line] = process_submissions(tmp_path, now=T0 + timedelta(hours=13))

    assert line == "20260101/m1: failed: no new files for 12 h (e1: 3 of 5 TIFFs)"
    state = state_of(tmp_path)
    assert state.state == "failed" and state.retry_on_change

    # Left alone while nothing changes...
    assert process_submissions(tmp_path, now=T0 + timedelta(hours=14)) == []

    # ...and waiting again once the copy goes on
    (tmp_path / RAW / "20260101_m1_e1_00004.tif").write_bytes(b"")
    assert process_submissions(tmp_path, now=T0 + timedelta(hours=15)) == [
        "20260101/m1: waiting: e1: 4 of 5 TIFFs"
    ]


def test_a_typed_problem_waits_for_the_draft_to_come_again(tmp_path):
    db, draft, _ = a_day(tmp_path, objective=None)

    process_submissions(tmp_path, now=T0)
    [line] = process_submissions(tmp_path, now=SETTLED)

    assert line == "20260101/m1: failed: e1: didn't pick an objective"
    assert not state_of(tmp_path).retry_on_change

    # Changing the files does not help...
    (tmp_path / SYNC).touch()
    assert process_submissions(tmp_path, now=SETTLED + timedelta(hours=1)) == []

    # ...submitting the draft again, fixed, does
    draft.set_part("e1", objective=20)
    enqueue(draft, tmp_path, user="ana")
    process_submissions(tmp_path, now=SETTLED + timedelta(hours=2))
    [line] = process_submissions(tmp_path, now=SETTLED + timedelta(hours=3))
    assert line.endswith("added as session 1")


def test_a_dry_run_changes_nothing(tmp_path):
    db, _, path = a_day(tmp_path)

    process_submissions(tmp_path, now=T0, dry_run=True)

    assert not path.with_name(f"{path.stem}.state.json").exists()
    assert state_of(tmp_path).state == "waiting"
    assert len(db.experiments) == 0


def test_the_tick_looks_at_the_queue(tmp_path):
    a_day(tmp_path, acqs_per_loop=5, bursts=5)

    report = tick(tmp_path)

    assert report.submissions == ["20260101/m1: waiting: e1: 3 of 5 TIFFs"]
    assert "Submitted:  1 waiting, 0 failed" in status(tmp_path)


def test_add_now_does_not_wait_for_the_tiffs(tmp_path):
    db, _, path = a_day(tmp_path, acqs_per_loop=5, bursts=5)
    process_submissions(tmp_path, now=T0)

    # The loop stopped at 3 and the sync file says 5: the user knows better.
    # The sync file is refused, the loop is still added.
    add_now(path)
    [line] = process_submissions(tmp_path, now=T0 + timedelta(minutes=1))

    assert line.startswith("20260101/m1: added as session 1; sync file of e1 refused")
    assert len(db.acquisitions) == 3
    assert not (queue_folder(tmp_path) / f"{path.stem}.now").exists()


def test_submit_again_starts_a_failed_draft_over(tmp_path):
    db, _, path = a_day(tmp_path)
    (tmp_path / "20260101/m1/e1/sync/extra.h5").write_bytes(b"")

    process_submissions(tmp_path, now=T0)
    [line] = process_submissions(tmp_path, now=SETTLED)
    assert line == "20260101/m1: failed: e1: 2 sync files in 'e1/sync'"

    # Fixed by hand, then submitted again from the QC page
    (tmp_path / "20260101/m1/e1/sync/extra.h5").rename(
        tmp_path / "20260101/m1/e1/sync/.extra.h5"
    )
    submit_again(path)
    later = SETTLED + timedelta(hours=1)

    assert process_submissions(tmp_path, now=later) == [
        "20260101/m1: waiting for the copy to finish"
    ]
    [line] = process_submissions(tmp_path, now=later + timedelta(minutes=11))
    assert line == "20260101/m1: added as session 1"


def test_take_back_puts_the_draft_among_this_computers_again(tmp_path):
    db, draft, path = a_day(tmp_path, objective=None)
    process_submissions(tmp_path, now=T0)
    process_submissions(tmp_path, now=SETTLED)

    local = take_back(path, tmp_path / "drafts")

    assert queued(tmp_path) == []
    assert (queue_folder(tmp_path) / "withdrawn" / path.name).exists()
    [back] = Draft.pending(tmp_path / "drafts")
    assert back.path == local and back.part("e1").get("objective") is None
    assert "submitted_by" not in back.data


def test_the_tick_tests_each_loop_it_adds_once(tmp_path, slurm):
    a_day(tmp_path)
    add_now(next(iter(queued(tmp_path)))[0])

    report = tick(tmp_path)

    assert report.submissions == ["20260101/m1: added as session 1"]
    assert report.tests == ["20260101/m1/e1: test submitted (group 1, job 4242)"]
    assert (
        "run-mcor-test --group 1 --user ana"
        in (cluster_folder(tmp_path) / "jobs" / "mcor-test-g1.sh").read_text()
    )

    # Never twice
    assert tick(tmp_path).tests == []
    assert slurm.read_text().count("sbatch") == 1

    # status says where it is
    (tmp_path / "queue.txt").write_text("4242 RUNNING\n")
    assert "Tests:      1 submitted, 1 running" in status(tmp_path)
    assert "  running   group 1 for ana, job 4242" in status(tmp_path)

    (tmp_path / "queue.txt").write_text("")
    log = cluster_folder(tmp_path) / "jobs" / "odyn-mcor-main-g1-4242.out"
    log.write_text("started  ...\nTraceback ...\n")
    assert "group 1 for ana, job 4242" in status(tmp_path)
    assert ": see odyn-mcor-main-g1-4242.out" in status(tmp_path)

    log.write_text("started  ...\nfinished 2026-01-01T21:00:00\n")
    assert "Tests:      1 submitted, 1 finished" in status(tmp_path)


def test_the_tick_can_leave_the_tests_out(tmp_path, slurm):
    a_day(tmp_path)
    add_now(next(iter(queued(tmp_path)))[0])
    (tmp_path / ".odyn" / "cluster.json").write_text('{"test_new_experiments": false}')

    assert tick(tmp_path).tests == []
    assert not (cluster_folder(tmp_path) / "mcor").exists()
