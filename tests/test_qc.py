"""
The QC view: built where the share is quick to list, read by the app as one
file, and the same wherever the main folder is mounted.
"""

import json
import os

from generate_data import a_tested_experiment

from odyn.cluster.slurm import _job_state
from odyn.qc import (
    QC_VERSION,
    qc_path,
    qc_summary,
    read_qc_summary,
    write_qc_summary,
)


def test_the_view_holds_paths_inside_the_main_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, group_id = a_tested_experiment(tmp_path / "data")
    path = write_qc_summary(db)
    summary = read_qc_summary(db.main_folder)

    assert path == qc_path(db.main_folder) and summary["version"] == QC_VERSION
    top = db.groups[group_id].mcor_test_folder().relative_to(db.main_folder)

    # Each machine mounts the share elsewhere: nothing absolute in it
    assert {row["experiment_folder"] for row in summary["tests"]} == {top.as_posix()}
    assert top.as_posix() in summary["files"]
    assert str(db.main_folder) not in path.read_text()

    # Every user's rows, each saying whose
    assert {row["user"] for row in summary["recordings"]} == {"tester"}


def test_a_view_of_another_version_is_not_read(tmp_path):
    db = a_database(tmp_path)
    assert read_qc_summary(tmp_path) is None  # none yet

    summary = qc_summary(db)
    qc_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    qc_path(tmp_path).write_text(json.dumps({**summary, "version": QC_VERSION + 1}))

    assert read_qc_summary(tmp_path) is None


def test_a_finished_job_is_finished_while_still_in_squeue(tmp_path):
    log = tmp_path / "job.out"
    log.write_text("started\nfinished 2026-11-03T12:00:00\n")

    # Past its work, updating the view at its end: `squeue` still lists it
    assert _job_state("11", log, {"11": "RUNNING"}) == "finished"
    assert _job_state("11", None, {"11": "RUNNING"}) == "running"


def a_database(main):
    from odyn import Database

    return Database(main, can_create=True, user="alice")


def test_the_page_reads_the_view_not_the_share(tmp_path, monkeypatch):
    """The test folders can be gone from the share: the view still has them."""
    import shutil

    import pytest

    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    monkeypatch.setenv("PATH", "")
    db, group_id = a_tested_experiment(tmp_path / "data")
    write_qc_summary(db)
    top = db.groups[group_id].mcor_test_folder()
    shutil.rmtree(top)  # what the page shows comes from the view alone

    from odyn.app.pane_qc import qc_view

    summary = read_qc_summary(db.main_folder)
    name, mine, missing, experiments, segmentation, cluster = qc_view(
        summary, db.main_folder, "tester", base=tmp_path / "base"
    )
    assert segmentation.empty  # not approved

    assert experiments["path"].tolist() == ["20260304/m1/e1"]
    assert experiments["bytes"].iloc[0] > 0  # sizes from the file lists
    assert [run.name for run in experiments.iloc[0]["finished"]] == ["test", "rigid"]
    assert not os.path.exists(top)


def test_a_decision_shows_until_the_cluster_records_it():
    import pandas as pd
    import pytest

    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from odyn.app.pane_qc import with_claims

    experiments = pd.DataFrame(
        {
            "group_id": [1, 2, 3],
            "status": ["ready to compare"] * 3,
            "raw check": ["not yet"] * 3,
            "excluded": [[], [], []],
        }
    )
    claimed = {
        (1, "raw"): {"user": "bob", "excluded": [4]},
        (1, "pick"): {"user": "bob", "run": "rigid"},
        (2, "approve"): {"user": "alice"},
        (3, "pick"): {"user": "carol", "run": "s64", "failed": "still correcting"},
    }

    shown = with_claims(experiments, claimed)

    assert shown["status"].tolist() == [
        "rigid picked by bob, waiting for the cluster",
        "approved by alice, waiting for the cluster",
        "pick by carol refused: still correcting",
    ]
    assert (
        shown["raw check"].tolist()[0] == "done, 1 excluded (waiting for the cluster)"
    )
    assert shown["excluded"].tolist()[0] == [4]


def test_the_segmentation_of_an_approved_experiment_goes_step_by_step():
    import pytest

    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from odyn.app.pane_qc import segmentation_status

    approval = {"call_id": 10, "called_at": "2026-11-17 10:00:00"}

    def call(call_id, state="done", at="2026-11-17 11:00:00"):
        return {"call_id": call_id, "state": state, "called_at": at, "user": "bob"}

    def job(state, since="2026-11-17 12:00:00"):
        return {"state": state, "since": since}

    steps = {}
    assert segmentation_status(approval, None, steps) == ("finding candidates", False)

    # Candidates of an earlier approval do not count; a failed call, or a
    # job that ended unfinished since the approval, stops them
    steps["candidates"] = call(9)
    assert segmentation_status(approval, None, steps)[0] == "finding candidates"
    steps["means"] = call(11, "failed")
    assert segmentation_status(approval, None, steps)[0] == "candidates stopped"
    steps["means"] = call(11)
    steps["jobs"] = {"cellpose": job("timeout")}
    assert segmentation_status(approval, None, steps)[0] == "candidates stopped"
    steps["jobs"] = {"cellpose": job("timeout", since="2026-11-16 09:00:00")}
    assert segmentation_status(approval, None, steps)[0] == "finding candidates"

    steps["candidates"] = call(12)
    assert segmentation_status(approval, None, steps) == ("ready to curate", True)

    # ROIs recorded: their traces, then done
    steps["rois"] = call(13)
    assert segmentation_status(approval, None, steps) == ("extracting traces", True)
    steps["jobs"] = {"traces": job("out_of_memory")}
    assert segmentation_status(approval, None, steps)[0] == "traces stopped"
    steps["traces"] = call(14, "failed")
    assert segmentation_status(approval, None, steps)[0] == "traces stopped"
    steps["traces"] = call(15)
    assert segmentation_status(approval, None, steps) == ("done", True)

    # Candidates made again since the ROIs: curated again
    steps["candidates"] = call(16)
    assert segmentation_status(approval, None, steps) == ("ready to curate", True)

    # A test picked after the approval: nothing to curate until approved again
    pick = {"call_id": 17}
    assert segmentation_status(approval, pick, steps) == (
        "motion correction checked again",
        False,
    )


def test_the_segmentation_tab_has_the_approved_experiments_and_their_claims(
    tmp_path,
):
    import pandas as pd
    import pytest

    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from odyn.app import cache
    from odyn.app.curation_window import draft_name
    from odyn.app.pane_qc import segmentation_rows, with_claims
    from odyn.segmentation import group_files

    experiments = pd.DataFrame(
        {
            "group_id": [1, 2, 3],
            "path": ["a/e1", "b/e1", "c/e1"],
            "user": ["ana", "ana", "bob"],
            "status": ["approved", "approved", "ready to compare"],
            "raw check": ["done"] * 3,
            "excluded": [[], [], []],
        }
    )
    done = {"state": "done", "user": "ana", "called_at": "2026-11-17 11:00:00"}
    summary = {
        "approvals": {"1": {"call_id": 10}, "2": {"call_id": 20}},
        "picks": {},
        "segmentation": {
            "1": {
                "candidates": {**done, "call_id": 11},
                "rois": {**done, "call_id": 12, "count": 540},
                "traces": {**done, "call_id": 13},
            },
            "2": {"candidates": {**done, "call_id": 21}},
        },
    }
    claimed = {(2, "rois"): {"user": "bob", "rois": 7}}

    # A draft of ana here, for the second
    _, candidates = group_files(tmp_path, "b/e1", 2)
    here = cache.local_folder(tmp_path, candidates.parent, tmp_path / "base")
    here.mkdir(parents=True)
    (here / draft_name(2, "ana")).write_text("{}")

    rows = segmentation_rows(
        experiments, summary, tmp_path, "ana", tmp_path / "base", claimed
    )

    assert rows["path"].tolist() == ["a/e1", "b/e1"]  # the approved ones
    assert rows["status"].tolist() == [
        "done",
        "submitted by bob (waiting for cluster)",
    ]
    assert rows["ROIs"].tolist() == [540, 7]
    assert rows["curated by"].tolist() == ["ana", ""]
    assert rows["curated at"].tolist() == ["2026-11-17 11:00", ""]
    assert rows["draft here"].tolist() == ["", "yes"]
    assert rows["curatable"].tolist() == [True, True]

    # The motion correction rows are not about ROIs, even refused ones
    refused = {(1, "rois"): {"user": "bob", "failed": "new Cellpose candidates"}}
    assert with_claims(experiments, refused)["status"].tolist() == [
        "approved",
        "approved",
        "ready to compare",
    ]
    assert segmentation_rows(experiments, summary, tmp_path, "ana", None, refused)[
        "status"
    ].tolist() == ["refused submission: new Cellpose candidates", "ready to curate"]


def test_the_view_says_how_far_each_segmentation_is(tmp_path, monkeypatch):
    from test_claims import recorded

    from odyn.cluster import cluster_folder, segmentation_states

    monkeypatch.setenv("PATH", "")  # no squeue: off the cluster
    db, group_id = a_tested_experiment(tmp_path / "data")
    recorded(db, group_id, "Group.save_cellpose_candidates", {"candidates": "x.h5"})
    recorded(db, group_id, "Group.import_mask", {"mask_path": "r.h5", "labels": [1, 2]})

    # Two jobs of the traces: the newer one counts, from its log
    jobs = cluster_folder(db.main_folder) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"odyn-traces-main-g{group_id}-7.out").write_text("started\n")
    (jobs / f"odyn-traces-main-g{group_id}-8.out").write_text("finished now\n")
    (jobs / f"odyn-traces-other-g{group_id}-9.out").write_text("started\n")

    states = segmentation_states(db)[group_id]

    assert states["candidates"]["state"] == "done"
    assert (states["rois"]["count"], states["rois"]["path"]) == (2, "r.h5")
    assert states["jobs"]["traces"]["state"] == "finished"
    assert "means" not in states and "cellpose" not in states["jobs"]
    assert qc_summary(db)["segmentation"][str(group_id)] == states  # JSON keys
