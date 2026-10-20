"""
`Group.test_motion_correction`: a test on acquisitions spread over the
experiment, and the report it leaves, with a stand-in for caiman that copies
each TIFF as its "corrected" movie.
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
from odyn.cluster import (
    cluster_folder,
    run_mcor_sweep,
    run_mcor_test,
    submit_mcor_sweep,
    submit_mcor_test,
    sweep_runs,
)
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
            self.pw_rigid = settings["pw_rigid"]
            self.strides = settings.get("strides")
            self.overlaps = settings.get("overlaps")
            self.max_deviation_rigid = settings.get("max_deviation_rigid")
            self.shifts_opencv = settings["shifts_opencv"]
            self.gSig_filt = settings.get("gSig_filt")
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

            if self.pw_rigid and template is not None:
                del self.shifts_rig  # caiman skips its rigid pass

            if self.pw_rigid:
                # Four patches; two jump by 1 px every other frame
                frames = sum(len(np.load(p)) for p in self.mmap_file)
                self.y_shifts_els = np.full((frames, 4), 0.5)
                self.x_shifts_els = np.full((frames, 4), -0.5)
                self.y_shifts_els[1::2, :2] += 1

    module = types.ModuleType("caiman")
    module.cluster = types.SimpleNamespace(setup_cluster=lambda **_: (None, None, None))
    module.stop_server = lambda **_: None
    module.load = lambda path: np.load(path)
    motion = types.ModuleType("caiman.motion_correction")
    motion.MotionCorrect = MotionCorrect
    # A stand-in high-pass filter, recognizable in what caiman is given
    motion.high_pass_filter_space = lambda image, width: image - image.mean()
    module.motion_correction = motion
    monkeypatch.setitem(sys.modules, "caiman", module)
    monkeypatch.setitem(sys.modules, "caiman.motion_correction", motion)

    return seen


def a_loop(tmp_path, user="ana") -> tuple[Database, int]:
    """Eight acquisitions of loop e1, added by `user`, and its group."""
    generate(
        tmp_path,
        acquisitions=8,
        frames=56,
        height=48,
        width=40,
        start=datetime(2026, 3, 4, 9),
    )
    db = Database(tmp_path, can_create=True, user=user)
    db.set_mouse(mouse_id=1)
    session_id = db.add_session(mouse_id=1, session_date="2026-03-04")
    exp_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e1", objective=20
    )

    return db, int(
        db.group_experiments.index[db.group_experiments["exp_id"] == exp_id][0]
    )


def test_a_test_leaves_the_files_to_judge_it_by(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)

    folder = db.groups[group_id].test_motion_correction(
        test_acqs=4, movies=(), side_by_side=True
    )

    name = f"20260304_m1_e1_g{group_id}"
    assert folder == db.files_folder / "cache" / "mcor-tests" / name

    # Rigid, and every acquisition aligned to the loop's own reference
    assert caiman["settings"]["pw_rigid"] is False
    assert caiman["template"].shape == (48, 40)

    summary = pd.read_csv(folder / "summary.csv")
    assert summary["position"].tolist() == spread_acquisitions(8, 4)
    assert summary["dy_median"].tolist() == [0.5] * 4
    assert tifffile.imread(folder / "side_by_side.tif").shape == (4, 48, 88)
    with tifffile.TiffFile(folder / "side_by_side.tif") as tif:
        assert len(tif.pages) == 4  # pages, not one RGB image
    assert tifffile.imread(folder / "reference.tif").shape == (48, 40)
    assert (folder / "shifts.png").stat().st_size > 0

    # The stand-in leaves the movie as it was: the frames are compared with
    # the reference, and the motion left is measured, for every acquisition
    assert summary["test_frame_corr_with_ref_median"].notna().all()
    assert (
        summary["test_frame_corr_with_ref_p10"]
        <= summary["test_frame_corr_with_ref_median"]
    ).all()
    assert (
        summary["test_frame_corr_with_ref_median"]
        <= summary["test_frame_corr_with_ref_p90"]
    ).all()
    assert not (folder / "means_test.tif").exists()  # side_by_side has them
    assert (summary["test_flow_median_px"] <= summary["test_flow_p90_px"]).all()
    assert tifffile.imread(folder / "flow_average.tif").shape == (48, 40)
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["flow_worst_acq_id"] in summary["acq_id"].tolist()
    assert metrics["settings"]["pw_rigid"] is False
    assert "patch_jitter_median_px" not in metrics
    assert "test_patch_jitter_median_px" not in summary
    assert tifffile.imread(folder / "flow_worst.tif").shape == (48, 40)

    call = db.latest_calls("test_motion_correction").iloc[0]
    assert call["call_flag"] == 0
    assert call["output_folder"] == folder.relative_to(tmp_path).as_posix()
    assert call["output_acq_ids"] == summary["acq_id"].tolist()
    assert db.outputs["file_path"].str.endswith(f"{name}/summary.csv").any()


def test_a_piecewise_run_has_its_own_folder_and_measures_the_patches(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    base = group.test_motion_correction(test_acqs=4, movies=())
    reference = tifffile.imread(base / "reference.tif")
    caiman["template"] = None

    folder = group.test_motion_correction(
        test_acqs=4,
        pw_rigid=True,
        strides_um=[24, 24],
        overlap_um=[12, 12],
        max_deviation_um=3,
        run="s24_o12_d3",
        movies=(),
        keep_patch_shifts=True,
        reuse_reference=True,
    )

    assert folder == base / "s24_o12_d3"
    assert caiman["settings"]["pw_rigid"] is True
    # The saved reference, not a new one
    np.testing.assert_array_equal(caiman["template"], reference)
    assert not (folder / "side_by_side.tif").exists()  # not by default

    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["settings"]["run"] == "s24_o12_d3"
    assert metrics["settings"]["strides_um"] == [24, 24]
    # Two patches of four move 1 px every frame: 0.5 px over the patches
    assert metrics["patch_jitter_median_px"] == 0.5
    summary = pd.read_csv(folder / "summary.csv")
    assert (summary["test_patch_jitter_p90_px"] == 0.5).all()
    # Without caiman's rigid pass, a frame's shift is its patches' median
    assert summary["dy_median"].tolist() == [0.75] * 4

    shifts = np.load(folder / "shifts.npz")
    assert shifts["patch_dy"].shape == (4 * 56, 4)

    # Without asking, the patches are measured but not kept
    folder = group.test_motion_correction(
        test_acqs=4,
        pw_rigid=True,
        strides_um=[24, 24],
        overlap_um=[12, 12],
        max_deviation_um=3,
        run="lean",
        movies=(),
        reuse_reference=True,
    )
    shifts = np.load(folder / "shifts.npz")
    assert "patch_dy" not in shifts and shifts["shifts"].shape == (4 * 56, 2)
    assert "patch_jitter_median_px" in json.loads((folder / "metrics.json").read_text())


def test_a_filtered_run_filters_the_reference_too(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    base = group.test_motion_correction(test_acqs=4, movies=())
    reference = tifffile.imread(base / "reference.tif")

    folder = group.test_motion_correction(
        test_acqs=4,
        pw_rigid=True,
        strides_um=[24, 24],
        overlap_um=[12, 12],
        max_deviation_um=3,
        gsig_filt_um=4,
        run="s24_o12_d3_g4",
        movies=(),
        side_by_side=False,
        reuse_reference=True,
    )

    # caiman filters the frames itself; the reference it gets is filtered
    # here, and its shifts are OpenCV's (it refuses the filter otherwise)
    settings = caiman["settings"]
    assert settings["shifts_opencv"] is True
    assert settings["gSig_filt"][0] == settings["gSig_filt"][1] >= 1
    np.testing.assert_allclose(caiman["template"], reference - reference.mean())

    recorded = json.loads((folder / "metrics.json").read_text())["settings"]
    assert recorded["gsig_filt_um"] == 4 and recorded["shifts_opencv"] is True
    assert recorded["gsig_filt_px"] == settings["gSig_filt"]

    # The report compares with the reference itself, not the filtered one
    np.testing.assert_array_equal(tifffile.imread(folder / "reference.tif"), reference)


def test_the_videos_of_a_run_share_their_levels(tmp_path, caiman, monkeypatch):
    from odyn.groups import Group

    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    asked = []

    def save_preview_movie(self, **kwargs):
        asked.append(kwargs)
        self.preview_levels = kwargs["levels"] or (100.0, 900.0)
        path = Path(kwargs["save_folder"]) / f"{'_'.join(kwargs['grid'])}.avi"
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(Group, "save_preview_movie", save_preview_movie)

    folder = group.test_motion_correction(
        test_acqs=2, run="r1", movies=("raw", "test", "raw_test")
    )

    # The first sets the levels and the rest use them; raw is the test's only
    assert [a["grid"] for a in asked] == [["raw"], ["test"], ["raw", "test"]]
    assert [a["levels"] for a in asked] == [None, (100.0, 900.0), (100.0, 900.0)]
    assert all(a["raw_of_test"] for a in asked)
    assert [a["title"] for a in asked] == ["raw", "r1", "r1"]
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["movie_levels"] == [100.0, 900.0]


def test_a_video_takes_the_levels_and_the_title_it_is_given(tmp_path):
    from odyn.groups import _write_preview_movie

    movie = np.random.default_rng(0).uniform(0, 1000, (6, 64, 96)).astype(np.float32)

    levels = _write_preview_movie(
        tmp_path / "a.avi", movie, frame_rate=10, levels=(10, 20), title="s128_o64_d6"
    )
    assert levels == (10, 20)

    levels = _write_preview_movie(tmp_path / "b.avi", movie, frame_rate=10)
    assert 0 < levels[0] < levels[1] < 1000
    assert (tmp_path / "b.avi").stat().st_size > 0

    # Half the size, from the same frames and levels
    import cv2

    halved = _write_preview_movie(
        tmp_path / "c.avi", movie, frame_rate=10, um_per_px=2.0, size=0.5
    )
    assert halved == levels
    video = cv2.VideoCapture(str(tmp_path / "c.avi"))
    assert (
        video.get(cv2.CAP_PROP_FRAME_WIDTH),
        video.get(cv2.CAP_PROP_FRAME_HEIGHT),
    ) == (
        48,
        32,
    )
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    video.release()


# --------------------------------------------------------------------------- #
# As a job on the cluster
# --------------------------------------------------------------------------- #


def test_a_test_is_a_job_of_its_own_for_who_added_the_loop(tmp_path, slurm):
    _, group_id = a_loop(tmp_path, user="ana")

    job_id = submit_mcor_test(tmp_path, None, group_id)

    assert job_id == "4242"
    assert "sbatch --parsable" in slurm.read_text()
    script = (
        cluster_folder(tmp_path) / "jobs" / f"mcor-test-g{group_id}.sh"
    ).read_text()
    assert "#SBATCH --gres=disk:128" in script
    assert 'CAIMAN_TEMP="/mnt/scratch/${SLURM_JOB_ID}"' in script
    assert "trap 'rm -rf \"$CAIMAN_TEMP\"' EXIT" in script
    assert f"run-mcor-test --group {group_id} --user ana" in script

    sent = json.loads(
        (cluster_folder(tmp_path) / "mcor" / f"{group_id}.json").read_text()
    )
    assert (sent["job_id"], sent["user"]) == ("4242", "ana")


def test_a_group_that_does_not_exist_is_refused_before_anything_is_written(
    tmp_path, slurm
):
    a_loop(tmp_path)

    with pytest.raises(ValueError, match="no group 999"):
        submit_mcor_test(tmp_path, None, 999)

    assert not (cluster_folder(tmp_path) / "jobs").exists()
    assert "sbatch" not in (slurm.read_text() if slurm.exists() else "")


def test_the_job_records_the_test_as_the_owners(tmp_path, monkeypatch):
    from odyn.groups import Group

    _, group_id = a_loop(tmp_path, user="ana")
    seen = []
    monkeypatch.setattr(
        Group,
        "test_motion_correction",
        lambda self: seen.append((self.group_id, self.db.user)) or tmp_path,
    )

    assert run_mcor_test(tmp_path, None, group_id, "ana") == tmp_path
    assert seen == [(group_id, "ana")]


# --------------------------------------------------------------------------- #
# A sweep
# --------------------------------------------------------------------------- #

GRID = {
    "test_acqs": 4,
    "max_shift_um": [20, 20],
    "runs": [
        {"strides_um": [128, 128], "overlap_um": [64, 64], "max_deviation_um": 6},
        {"strides_um": [128, 96], "overlap_um": [32, 32], "max_deviation_um": 2.5},
        {
            "strides_um": [64, 64],
            "overlap_um": [32, 32],
            "max_deviation_um": 3,
            "run": "small",
        },
    ],
}


def test_runs_are_named_from_their_settings():
    assert [run["run"] for run in sweep_runs(GRID)] == [
        "s128_o64_d6",
        "s128x96_o32_d2.5",
        "small",
    ]

    patches = GRID["runs"][0]
    filtered = sweep_runs(
        {
            "runs": [
                {**patches, "gsig_filt_um": 4},
                {**patches, "shifts_opencv": True},
                {**patches, "gsig_filt_um": 2.5, "shifts_opencv": True},
            ]
        }
    )
    assert [run["run"] for run in filtered] == [
        "s128_o64_d6_g4",
        "s128_o64_d6_cv",
        "s128_o64_d6_g2.5",  # the filter implies OpenCV shifts
    ]
    assert filtered[0]["gsig_filt_um"] == 4 and "shifts_opencv" not in filtered[0]


@pytest.mark.parametrize(
    "runs, message",
    [
        ([], "no runs"),
        ([{**GRID["runs"][0], "gSig_filt": 3}], "Unknown run settings gSig_filt"),
        ([{"strides_um": [64, 64], "overlap_um": [32, 32]}], "no max_deviation_um"),
        ([GRID["runs"][0], GRID["runs"][0]], "s128_o64_d6"),
        ([{**GRID["runs"][0], "run": "rigid"}], "not be 'rigid'"),
    ],
)
def test_a_grid_that_cannot_run_is_refused(runs, message):
    with pytest.raises(ValueError, match=message):
        sweep_runs({"runs": runs})


def test_a_sweep_is_a_baseline_then_every_run_at_once(tmp_path, slurm):
    _, group_id = a_loop(tmp_path, user="ana")

    assert submit_mcor_sweep(tmp_path, None, group_id, GRID) == ("4242", "4242")

    baseline, array = [
        line for line in slurm.read_text().splitlines() if line.startswith("sbatch")
    ]
    assert "--array" not in baseline
    assert "--array=1-3 --dependency=afterok:4242" in array
    assert "%x-%A_%a.out" in array
    assert "%" not in array.replace("%x-%A_%a", "")  # no throttle

    folder = cluster_folder(tmp_path)
    script = (folder / "jobs" / f"mcor-sweep-g{group_id}.sh").read_text()
    assert 'CAIMAN_TEMP="/mnt/scratch/${SLURM_JOB_ID}"' in script
    assert '--task "${SLURM_ARRAY_TASK_ID:-0}"' in script

    sweep = json.loads((folder / "sweeps" / f"g{group_id}.json").read_text())
    assert (sweep["user"], sweep["baseline_job"], sweep["array_job"]) == (
        "ana",
        "4242",
        "4242",
    )
    assert sweep["runs"][0]["run"] == "s128_o64_d6"


def test_the_runs_take_the_baselines_reference_and_levels(tmp_path, slurm, monkeypatch):
    from odyn.groups import Group

    db, group_id = a_loop(tmp_path, user="ana")
    submit_mcor_sweep(tmp_path, None, group_id, GRID)
    sweep = cluster_folder(tmp_path) / "sweeps" / f"g{group_id}.json"
    asked = []
    monkeypatch.setattr(
        Group,
        "test_motion_correction",
        lambda self, **kwargs: asked.append((self.db.user, kwargs)) or tmp_path,
    )

    run_mcor_sweep(sweep, 0)
    baseline = db.groups[group_id].mcor_test_folder("rigid")
    baseline.mkdir(parents=True)
    (baseline / "metrics.json").write_text(json.dumps({"movie_levels": [10, 90]}))
    run_mcor_sweep(sweep, 2)

    (user, first), (_, second) = asked
    assert user == "ana"
    assert first["run"] == "rigid" and "pw_rigid" not in first
    assert first["movies"] == ("raw", "test", "raw_test")
    assert first["side_by_side"] is False
    assert second["keep_patch_shifts"] is False
    assert second["run"] == "s128x96_o32_d2.5"
    assert second["pw_rigid"] is True and second["reuse_reference"] is True
    assert second["strides_um"] == [128, 96]
    assert (second["movies"], second["levels"]) == (("test",), [10, 90])
    assert (second["test_acqs"], second["max_shift_um"]) == (4, [20, 20])
