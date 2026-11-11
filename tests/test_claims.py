"""
QC decisions as claim files: one per experiment and step, the first one
taken stands until the cluster records it as its author.
"""

import json
import os

import pytest

from generate_data import a_tested_experiment
from test_add_mcor_files import build, write_mcor

from odyn.claims import ClaimTaken, claim, claims, pending_claims
from odyn.cluster import TickReport, apply_claims


def test_the_first_claim_stands_until_the_cluster_records_it(tmp_path):
    path = claim(tmp_path, None, 42, "approve", "alice", excluded=[3])

    with pytest.raises(ClaimTaken, match="alice did it already"):
        claim(tmp_path, None, 42, "approve", "bob")

    # Its author can change it; another step or experiment is free
    claim(tmp_path, None, 42, "approve", "alice", excluded=[3, 7])
    claim(tmp_path, None, 42, "raw", "bob", excluded=[])
    claim(tmp_path, None, 7, "approve", "bob")

    found = pending_claims(tmp_path)
    assert found[(42, "approve")]["excluded"] == [3, 7]
    assert found[(42, "approve")]["at"] == os.stat(path).st_mtime  # the server's
    assert set(found) == {(42, "approve"), (42, "raw"), (7, "approve")}

    with pytest.raises(ValueError, match="Unknown QC step"):
        claim(tmp_path, None, 42, "delete", "alice")


def test_a_refused_claim_stays_with_why_and_can_be_made_again(tmp_path):
    path = claim(tmp_path, None, 42, "pick", "alice", run="rigid", folder="x")
    path.write_text(json.dumps({**json.loads(path.read_text()), "failed": "no"}))

    assert pending_claims(tmp_path)[(42, "pick")]["failed"] == "no"
    claim(tmp_path, None, 42, "pick", "bob", run="rigid", folder="x")  # replaced
    assert "failed" not in pending_claims(tmp_path)[(42, "pick")]


def test_the_tick_records_each_claim_as_its_author(tmp_path, monkeypatch):
    from odyn.qc import latest_raw_checks

    monkeypatch.setenv("PATH", "")
    db, group_id = a_tested_experiment(tmp_path)
    claim(tmp_path, None, group_id, "raw", "alice", excluded=[2])
    folder = claims(tmp_path).folder
    (folder / "g9-approve.json").write_text("")  # still being written: left
    report = TickReport()

    apply_claims(tmp_path, None, report)

    assert latest_raw_checks(db) == {group_id: [2]}
    calls = db.calls
    assert calls.loc[calls["method_name"] == "Group.check_raw", "user"].tolist() == [
        "alice"
    ]
    assert report.claims == [f"raw of group {group_id} for alice"]
    [done] = claims(tmp_path).items("done")
    assert done.name.startswith(f"g{group_id}-raw-")  # one per decision
    assert [p.name for p in claims(tmp_path).items()] == ["g9-approve.json"]


def test_a_claim_the_tick_refuses_is_kept_for_its_author(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    db, group_id = a_tested_experiment(tmp_path)
    claim(tmp_path, None, group_id, "again", "alice", runs=[])  # nothing to run
    report = TickReport()

    apply_claims(tmp_path, None, report)

    found = pending_claims(tmp_path)[(group_id, "again")]
    assert found["failed"] and report.rejected and not report.claims

    apply_claims(tmp_path, None, TickReport())  # left until decided again
    assert pending_claims(tmp_path)[(group_id, "again")]["failed"] == found["failed"]


def test_an_approval_submits_the_mean_images_then_the_cell_candidates(tmp_path, slurm):
    from odyn.cluster import cluster_folder

    db, group = build(tmp_path)
    group_id = group.group_id

    # Nothing saved yet: recorded, with nothing to make images of
    claim(tmp_path, None, group_id, "approve", "alice")
    report = TickReport()
    apply_claims(tmp_path, None, report)
    assert report.claims and not report.tests and not report.errors

    write_mcor(tmp_path, 1)
    group.add_mcor_files()
    claim(tmp_path, None, group_id, "approve", "alice")
    report = TickReport()
    apply_claims(tmp_path, None, report)

    assert report.claims == [f"approve of group {group_id} for alice"]
    assert report.tests == [
        f"mean images and Cellpose candidates of group {group_id} submitted"
        " (job 4242)"
    ]
    jobs = cluster_folder(tmp_path) / "jobs"
    script = (jobs / f"cellpose-g{group_id}.sh").read_text()
    means = script.index(f"mean-images --group {group_id} --user alice")
    cells = script.index(f"cellpose --group {group_id} --user alice")
    assert means < cells  # the candidates come from the mean image
    assert f"sbatch --parsable --export=NONE {jobs}" in slurm.read_text()
