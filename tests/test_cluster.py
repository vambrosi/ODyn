"""
The cluster side: requests, the tick that applies them and processes the sync
files that arrived late, its heartbeat, and the chain of ticks (with stand-ins
for `sbatch`, `squeue` and `scancel`).
"""

import json
import os
import shutil
import time

import pytest

from odyn.cluster import (
    ClusterConfig,
    cluster_folder,
    heartbeat,
    main,
    pending_requests,
    send_request,
    start_chain,
    status,
    stop_chain,
    tick,
)
from generate_sync import generate_sync
from test_decode_sync import build, grabs_day

SYNC = "20260101/m001/e1/sync/m001_e1.h5"
GRABS_SYNC = "20260101/m001/grabs/sync/m001_grabs.h5"
HOUR_AGO = time.time() - 3600


def a_loop(tmp_path, **options):
    """A database with loop 1 and its sync file, an hour old."""
    truth = generate_sync(tmp_path / SYNC, acquisitions=4)
    os.utime(tmp_path / SYNC, (HOUR_AGO, HOUR_AGO))

    return build(tmp_path, truth, **options)


def test_a_request_runs_as_the_person_who_sent_it(tmp_path):
    db = a_loop(tmp_path)
    path = send_request(tmp_path, "process_sync", user="ana", exp_id=1)
    assert pending_requests(tmp_path) == [path]

    report = tick(tmp_path)

    assert report.applied == ["process_sync(exp_id=1) for ana"]
    assert report.processed == []  # processed by the request, not again
    assert pending_requests(tmp_path) == []
    done = json.loads((path.parent / "done" / path.name).read_text())
    assert done["error"] is None and done["user"] == "ana"

    call = db.latest_calls("process_sync").iloc[0]
    assert call["user"] == "ana" and call["call_flag"] == 0


def test_requests_for_anything_else_are_refused(tmp_path):
    a_loop(tmp_path)

    with pytest.raises(ValueError, match="cannot be requested"):
        send_request(tmp_path, "add_user", user_name="x")

    with pytest.raises(ValueError, match="Wrong arguments"):
        send_request(tmp_path, "process_sync", loop=1)

    # A request written by hand is checked again at the tick
    folder = cluster_folder(tmp_path) / "requests"
    folder.mkdir(parents=True)
    (folder / "by-hand.json").write_text(
        json.dumps({"method": "unlock", "kwargs": {}, "user": "x"})
    )
    report = tick(tmp_path)

    assert report.rejected and "cannot be requested" in report.rejected[0]
    rejected = json.loads((folder / "rejected" / "by-hand.json").read_text())
    assert "cannot be requested" in rejected["reason"]


def test_a_sync_file_copied_late_is_processed_once(tmp_path):
    db = a_loop(tmp_path)

    # Submitted before the file was there...
    shutil.move(tmp_path / SYNC, tmp_path / "away.h5")

    with pytest.raises(RuntimeError, match="No sync file"):
        db.process_sync(exp_id=1)

    # ...then copied in, keeping its old modification time (as Windows does)
    shutil.move(tmp_path / "away.h5", tmp_path / SYNC)

    assert tick(tmp_path).processed == [SYNC]
    assert len(db.streams) == 1
    assert tick(tmp_path).processed == []


def test_a_sync_file_still_arriving_waits(tmp_path):
    db = a_loop(tmp_path)
    os.utime(tmp_path / SYNC)  # modified just now

    assert tick(tmp_path).processed == []
    assert db.streams.empty

    os.utime(tmp_path / SYNC, (HOUR_AGO, HOUR_AGO))
    assert tick(tmp_path).processed == [SYNC]


def test_a_file_that_does_not_match_is_tried_once(tmp_path):
    db = a_loop(tmp_path, acquisitions=2)  # the file has 4 bursts

    report = tick(tmp_path)
    assert report.processed == [SYNC]
    assert "cannot be matched" in report.errors[0]

    assert tick(tmp_path).processed == []
    assert len(db.latest_calls("process_sync")) == 1


def test_a_grabs_file_is_processed_and_again_when_it_changes(tmp_path):
    truth = generate_sync(
        tmp_path / GRABS_SYNC, acquisitions=3, missing_pulse_in=(0, 1, 2)
    )
    os.utime(tmp_path / GRABS_SYNC, (HOUR_AGO, HOUR_AGO))
    db = grabs_day(tmp_path, truth)

    with db._writing() as cur:
        cur.execute(
            "UPDATE grabs SET raw_path = '20260101/m001/grabs/raw/' || raw_path;"
        )

    assert tick(tmp_path).processed == [GRABS_SYNC]
    assert db.grabs["sync_block"].notna().sum() == 3
    assert tick(tmp_path).processed == []

    os.utime(tmp_path / GRABS_SYNC)  # changed (no wait for it to settle)
    assert tick(tmp_path, settle_s=0).processed == [GRABS_SYNC]


def test_a_dry_run_changes_nothing(tmp_path):
    db = a_loop(tmp_path)
    send_request(tmp_path, "process_grabs_sync", session_id=1, sync_path="x.h5")

    report = tick(tmp_path, dry_run=True)

    assert report.applied[0].startswith("would apply")
    assert report.processed == [f"would process {SYNC}"]
    assert len(pending_requests(tmp_path)) == 1
    assert db.streams.empty and heartbeat(tmp_path) is None


def test_the_heartbeat_says_what_the_last_tick_did(tmp_path):
    a_loop(tmp_path)
    tick(tmp_path)

    beat = heartbeat(tmp_path)
    assert beat["processed"] == [SYNC] and beat["errors"] == []
    assert beat["user"] and beat["host"]

    log = (cluster_folder(tmp_path) / "tick.log").read_text().splitlines()
    assert len(log) == 1 and "1 sync files processed" in log[0]

    text = status(tmp_path)
    assert "Requests:   0 waiting" in text and "Stopped:    no" in text


def test_the_config_file_overrides_the_defaults(tmp_path):
    (tmp_path / ".odyn").mkdir()
    (tmp_path / ".odyn" / "cluster.json").write_text('{"partition": "short"}')
    config = ClusterConfig.read(tmp_path)
    assert config.partition == "short" and config.conda_env == "caiman"

    (tmp_path / ".odyn" / "cluster.json").write_text('{"partishun": "short"}')

    with pytest.raises(ValueError, match="Unknown settings"):
        ClusterConfig.read(tmp_path)


def test_a_chained_tick_submits_the_next_one(tmp_path, slurm):
    a_loop(tmp_path)

    report = tick(tmp_path, chain=True)

    assert report.next_tick is not None
    sbatch = slurm.read_text().splitlines()[-1]
    assert "--dependency=singleton" in sbatch and "--begin=now+30minutes" in sbatch

    script = (cluster_folder(tmp_path) / "jobs" / "tick.sh").read_text()
    assert "--job-name=odyn-tick-main" in script
    assert "conda activate caiman" in script
    assert f"odyn.cluster {tmp_path.resolve()} tick --chain" in script

    # Stopped: the running tick does not submit another
    stop_chain(tmp_path)
    assert "scancel --name=odyn-tick-main --state=PENDING" in slurm.read_text()
    assert tick(tmp_path, chain=True).next_tick is None
    assert slurm.read_text().count("sbatch") == 1


def test_starting_the_chain_once(tmp_path, slurm):
    a_loop(tmp_path)
    stop_chain(tmp_path)

    assert start_chain(tmp_path) == "4242"
    assert not (cluster_folder(tmp_path) / "STOP").exists()
    assert "--begin" not in slurm.read_text().splitlines()[-1]  # starts now

    # A tick already in the queue: nothing more
    (tmp_path / "queue.txt").write_text("4242 PENDING 2026-10-08T23:00:00\n")
    assert start_chain(tmp_path) is None
    assert slurm.read_text().count("sbatch") == 1
    assert "4242 PENDING" in status(tmp_path)


def test_the_command_line(tmp_path, capsys):
    db = a_loop(tmp_path)

    assert main([str(tmp_path), "request", "process_sync", "exp_id=1"]) == 0
    assert "Left" in capsys.readouterr().out

    assert main([str(tmp_path), "tick"]) == 0
    assert "process_sync(exp_id=1)" in capsys.readouterr().out
    assert len(db.streams) == 1
