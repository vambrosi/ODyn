"""
`Group.test_motion_correction`: a rigid test on acquisitions spread over the
loop, and the report it leaves, with a stand-in for caiman that copies each
TIFF as its "corrected" movie.
"""

import json
import sys
import types

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from odyn import Database
from odyn.registration import spread_acquisitions
from generate_data import generate


@pytest.fixture
def caiman(monkeypatch, tmp_path):
    """A stand-in for caiman; returns what it was asked to do."""
    seen = {}
    temp = tmp_path / "caiman-temp"
    temp.mkdir()

    class MotionCorrect:
        def __init__(self, fnames, dview=None, **settings):
            self.fname = [str(f) for f in fnames]
            self.max_shifts = settings["max_shifts"]
            seen["settings"] = settings

        def motion_correct(self, template=None, save_movie=False):
            seen["template"] = template
            self.mmap_file, self.shifts_rig = [], []

            for name in self.fname:
                movie = tifffile.imread(name).astype(np.float32)
                path = temp / f"{Path(name).stem}_rig.npy"
                np.save(path, movie)
                self.mmap_file.append(str(path))
                self.shifts_rig += [(0.5, -0.5)] * len(movie)

    module = types.ModuleType("caiman")
    module.cluster = types.SimpleNamespace(setup_cluster=lambda **_: (None, None, None))
    module.stop_server = lambda **_: None
    module.load = lambda path: np.load(path)
    motion = types.ModuleType("caiman.motion_correction")
    motion.MotionCorrect = MotionCorrect
    module.motion_correction = motion
    monkeypatch.setitem(sys.modules, "caiman", module)
    monkeypatch.setitem(sys.modules, "caiman.motion_correction", motion)

    return seen


def test_a_test_leaves_the_files_to_judge_it_by(tmp_path, caiman):
    generate(
        tmp_path,
        acquisitions=8,
        frames=56,
        height=48,
        width=40,
        start=datetime(2026, 3, 4, 9),
    )
    db = Database(tmp_path, can_create=True, user="ana")
    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-03-04")
    loop_id = db.add_loop(
        session_id=session_id, loop_path="20260304/m1/e1", objective=20
    )
    group_id = int(db.group_loops.index[db.group_loops["loop_id"] == loop_id][0])

    folder = db.groups[group_id].test_motion_correction(test_acqs=4, movie=False)

    name = f"20260304_m1_e1_g{group_id}"
    assert folder == db.files_folder / "cache" / "mcor-tests" / name

    # Rigid, and every acquisition aligned to the loop's own reference
    assert caiman["settings"]["pw_rigid"] is False
    assert caiman["template"].shape == (48, 40)

    summary = pd.read_csv(folder / "summary.csv")
    assert summary["position"].tolist() == spread_acquisitions(8, 4)
    assert summary["dy_median"].tolist() == [0.5] * 4
    assert tifffile.imread(folder / "side_by_side.tif").shape == (4, 48, 88)
    assert tifffile.imread(folder / "reference.tif").shape == (48, 40)
    assert (folder / "shifts.png").stat().st_size > 0

    # The stand-in leaves the movie as it was: the frames are compared with
    # the reference, and the motion left is measured, for every acquisition
    assert summary["test_frame_corr_median"].notna().all()
    assert (summary["test_frame_corr_p5"] <= summary["test_frame_corr_median"]).all()
    assert (summary["test_frame_corr_median"] <= summary["test_frame_corr_p95"]).all()
    assert not (folder / "means_test.tif").exists()  # side_by_side has them
    assert (summary["test_flow_median_px"] <= summary["test_flow_p90_px"]).all()
    assert tifffile.imread(folder / "flow_average.tif").shape == (48, 40)
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["flow_worst_acq_id"] in summary["acq_id"].tolist()
    assert tifffile.imread(folder / "flow_worst.tif").shape == (2, 48, 40)

    call = db.latest_calls("test_motion_correction").iloc[0]
    assert call["call_flag"] == 0
    assert call["output_folder"] == folder.relative_to(tmp_path).as_posix()
    assert call["output_acq_ids"] == summary["acq_id"].tolist()
    assert db.outputs["file_path"].str.endswith(f"{name}/summary.csv").any()
