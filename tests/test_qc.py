"""
The QC view: built where the share is quick to list, read by the app as one
file, and the same wherever the main folder is mounted.
"""

import json
import os

from generate_data import a_tested_experiment

from odyn.cluster import _job_state
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
    name, mine, missing, experiments, cluster = qc_view(
        summary, db.main_folder, "tester", base=tmp_path / "base"
    )

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
