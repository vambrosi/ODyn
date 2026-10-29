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
            seen.setdefault("batches", []).append([Path(f).name for f in fnames])

        def motion_correct(self, template=None, save_movie=False):
            seen["template"] = template
            seen.setdefault("templates", []).append(template)
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
    # caiman reads its own files and TIFFs alike
    module.load = lambda path: (
        tifffile.imread(path) if str(path).endswith(".tif") else np.load(path)
    )
    motion = types.ModuleType("caiman.motion_correction")
    motion.MotionCorrect = MotionCorrect
    # A stand-in high-pass filter, recognizable in what caiman is given
    motion.high_pass_filter_space = lambda image, width: image - image.mean()
    module.motion_correction = motion
    monkeypatch.setitem(sys.modules, "caiman", module)
    monkeypatch.setitem(sys.modules, "caiman.motion_correction", motion)

    return seen


def reference_of(folder: Path) -> np.ndarray:
    """
    The reference a test named in its `metrics.json`, checked against its
    SHA-1.
    """
    import hashlib

    named = json.loads((folder / "metrics.json").read_text())["reference"]
    main = next(p for p in folder.parents if (p / ".odyn").is_dir())
    path = main / named["file"]
    assert hashlib.sha1(path.read_bytes()).hexdigest() == named["sha1"]
    assert path.parent.name == "mcor" and path.parent.parent.name == "processed"

    return tifffile.imread(path)


def a_loop(tmp_path, user="alice") -> tuple[Database, int]:
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


def test_a_final_correction_goes_in_batches_with_one_reference(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]

    group.run_motion_correction(
        use_gui_parameters=False, is_test=False, pw_rigid=False, batch_acqs=3
    )

    # Eight acquisitions, three at a time, all to the same reference
    assert [len(names) for names in caiman["batches"]] == [3, 3, 2]
    assert sum(caiman["batches"], []) == sorted(sum(caiman["batches"], []))
    first = caiman["templates"][0]
    assert first is not None
    assert all(t is first for t in caiman["templates"])

    # Every acquisition saved and recorded, caiman's copies gone
    files = db.groups[group_id].mcor_files
    assert len(files) == 8
    assert all((tmp_path / path).is_file() for path in files["mcor_path"])
    assert not list((tmp_path / "caiman-temp").glob("*.npy"))
    saved = tifffile.imread(tmp_path / files["mcor_path"].iloc[0])
    assert saved.shape == (56, 48, 40)

    with db._locked() as con:
        used = con.execute(
            "SELECT parameters_used FROM method_calls"
            " WHERE method_name = 'Group.run_motion_correction'"
        ).fetchone()[0]
    assert json.loads(used)["batch_acqs"] == 3


def test_a_whole_test_saves_every_acquisition_and_reports_on_the_test_ones(
    tmp_path, caiman
):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]

    folder = group.test_motion_correction(
        test_acqs=4, movie=False, whole=True, batch_acqs=3
    )

    # Every acquisition corrected (in batches) and saved as the final
    assert [len(names) for names in caiman["batches"]] == [3, 3, 2]
    assert len(db.groups[group_id].mcor_files) == 8

    # The report is a test's: on the spread acquisitions, in the test folder
    assert folder == group.mcor_test_folder()
    summary = pd.read_csv(folder / "summary.csv")
    assert summary["position"].tolist() == spread_acquisitions(8, 4)
    assert summary["dy_median"].tolist() == [0.5] * 4
    shifts = np.load(folder / "shifts.npz")
    assert shifts["shifts"].shape == (4 * 56, 2)
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["settings"]["whole"] is True

    # And nothing of caiman's is left behind
    assert not list((tmp_path / "caiman-temp").glob("*.npy"))

    # The full run's numbers: every frame of every acquisition, shifts and
    # matches with the reference, raw and corrected
    full = np.load(folder / "full" / "shifts_summary.npz")
    assert full["frame_counts"].tolist() == [56] * 8
    assert len(full["acq_ids"]) == 8
    assert len(full["dy"]) == len(full["corr"]) == len(full["raw_corr"]) == 8 * 56
    assert "dy_p10" not in full  # rigid

    # The report on every acquisition, as a test's on a few: flow and all
    every = pd.read_csv(folder / "full" / "summary.csv")
    assert every["position"].tolist() == list(range(8))
    assert every["dy_median"].tolist() == [0.5] * 8
    assert every["test_flow_p90_px"].notna().all()
    assert (every["test_corr_with_ref"] >= every["raw_corr_with_ref"] - 1e-3).all()
    checked = json.loads((folder / "full" / "metrics.json").read_text())
    assert checked["acquisitions"] == 8
    assert checked["flow_worst_acq_id"] in every["acq_id"].tolist()
    assert tifffile.imread(folder / "full" / "flow_average.tif").shape == (48, 40)

    # Written again from the saved files (for a correction from before):
    # the same numbers, and listed until then
    from odyn.cluster import groups_without_full_report

    assert groups_without_full_report(db) == []
    (folder / "full" / "summary.csv").unlink()
    assert [g for g, _ in groups_without_full_report(db)] == [group_id]
    assert db.groups[group_id].full_report() == folder / "full"
    again = pd.read_csv(folder / "full" / "summary.csv")
    pd.testing.assert_frame_equal(again, every, atol=1e-3)


def test_a_whole_correction_stopped_partway_goes_on_from_there(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    motion = sys.modules["caiman.motion_correction"].MotionCorrect
    correct = motion.motion_correct
    calls = []

    def stopped_at_the_second(self, **kwargs):
        calls.append(1)

        if len(calls) == 2:  # the job's time limit, say
            raise RuntimeError("time limit")

        return correct(self, **kwargs)

    motion.motion_correct = stopped_at_the_second

    with pytest.raises(RuntimeError, match="time limit"):
        group.test_motion_correction(test_acqs=4, movie=False, whole=True, batch_acqs=3)

    # The first batch saved and written down, by the reference
    progress_file = group._progress_path()
    assert progress_file.name == f"progress_group_{group_id}.json"
    assert progress_file.parent == group._reference_path().parent
    progress = json.loads(progress_file.read_text())
    assert [b["batch"] for b in progress["done"]] == [1]
    assert not progress["finished"] and progress["batches"] == 3
    assert len(db.groups[group_id].mcor_files) == 3

    # Again, the same: only the batches left, and every number as if whole
    motion.motion_correct = correct
    caiman["batches"].clear()
    folder = db.groups[group_id].test_motion_correction(
        test_acqs=4, movie=False, whole=True, batch_acqs=3
    )

    assert [len(names) for names in caiman["batches"]] == [3, 2]
    assert len(db.groups[group_id].mcor_files) == 8
    summary = pd.read_csv(folder / "summary.csv")
    assert summary["position"].tolist() == spread_acquisitions(8, 4)
    assert summary["dy_median"].tolist() == [0.5] * 4
    full = np.load(folder / "full" / "shifts_summary.npz")
    assert full["frame_counts"].tolist() == [56] * 8
    assert len(full["dy"]) == len(full["corr"]) == 8 * 56
    every = pd.read_csv(folder / "full" / "summary.csv")
    assert every["position"].tolist() == list(range(8))
    assert (
        json.loads((folder / "full" / "metrics.json").read_text())["acquisitions"] == 8
    )
    progress = json.loads(progress_file.read_text())
    assert [b["batch"] for b in progress["done"]] == [1, 2, 3]
    assert progress["finished"] and len(progress["call_ids"]) == 2
    with db._locked() as con:
        used = con.execute(
            "SELECT parameters_used FROM method_calls WHERE method_name ="
            " 'Group.run_motion_correction' ORDER BY method_call_id DESC"
        ).fetchone()[0]
    assert json.loads(used)["resumed_after_batch"] == 1

    # The final files the report read instead of caiman's are kept
    files = db.groups[group_id].mcor_files["mcor_path"]
    assert all((tmp_path / path).is_file() for path in files)

    # Finished: a run again starts over, as do other settings
    caiman["batches"].clear()
    db.groups[group_id].test_motion_correction(
        test_acqs=4, movie=False, whole=True, batch_acqs=3
    )
    assert [len(names) for names in caiman["batches"]] == [3, 3, 2]

    motion.motion_correct = stopped_at_the_second
    calls.clear()
    with pytest.raises(RuntimeError):
        db.groups[group_id].test_motion_correction(
            test_acqs=4, movie=False, whole=True, batch_acqs=3
        )
    motion.motion_correct = correct
    caiman["batches"].clear()
    db.groups[group_id].test_motion_correction(
        test_acqs=4, movie=False, whole=True, batch_acqs=4
    )
    assert [len(names) for names in caiman["batches"]] == [4, 4]

    # And asked to: resume=False
    motion.motion_correct = stopped_at_the_second
    calls.clear()
    with pytest.raises(RuntimeError):
        db.groups[group_id].test_motion_correction(
            test_acqs=4, movie=False, whole=True, batch_acqs=3
        )
    motion.motion_correct = correct
    caiman["batches"].clear()
    db.groups[group_id].test_motion_correction(
        test_acqs=4, movie=False, whole=True, batch_acqs=3, resume=False
    )
    assert [len(names) for names in caiman["batches"]] == [3, 3, 2]


def group_folder(db, group_id):
    return db.groups[group_id].mcor_test_folder()


def test_a_whole_piecewise_test_reports_its_patches(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)

    folder = db.groups[group_id].test_motion_correction(
        test_acqs=4,
        movie=False,
        whole=True,
        batch_acqs=5,
        pw_rigid=True,
        strides_um=[24, 24],
        overlap_um=[12, 12],
        max_deviation_um=3,
        run="pw",
    )

    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["patch_jitter_median_px"] == 0.5
    full = np.load(group_folder(db, group_id) / "full" / "shifts_summary.npz")
    assert len(full["dy_p10"]) == len(full["dy"]) == 8 * 56
    summary = pd.read_csv(folder / "summary.csv")
    assert summary["dy_median"].tolist() == [0.75] * 4


def test_a_final_correction_is_saved_as_the_raw_files_are(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)

    # Overshoot as FFT shifts do: past each frame's raw range, and not whole
    stand_in = sys.modules["caiman"]
    stand_in.load = lambda path: np.load(path) * 1.5 - 100.3

    db.groups[group_id].run_motion_correction(
        use_gui_parameters=False, is_test=False, pw_rigid=False
    )

    files = db.groups[group_id].mcor_files.sort_index()
    raw_paths = db.groups[group_id].acquisitions.sort_index()["raw_path"]
    saved = tifffile.imread(tmp_path / files["mcor_path"].iloc[0])
    raw = tifffile.imread(tmp_path / raw_paths.iloc[0])

    assert saved.dtype == raw.dtype == np.int16
    # Every frame within its raw frame's range, as NoRMCorre clips
    assert (saved.min(axis=(1, 2)) >= raw.min(axis=(1, 2))).all()
    assert (saved.max(axis=(1, 2)) <= raw.max(axis=(1, 2))).all()
    expected = np.rint(
        np.clip(
            raw * 1.5 - 100.3,
            raw.min(axis=(1, 2))[:, None, None],
            raw.max(axis=(1, 2))[:, None, None],
        )
    )
    np.testing.assert_array_equal(saved, expected)

    output = db.latest_calls("run_motion_correction").iloc[0]
    assert output["output_saved_dtype"] == "int16"
    assert output["output_clipped_values"] > 0


def test_as_raw_rounds_and_counts_what_it_clips():
    from odyn.groups import as_raw

    raw = np.array([[[0, 10], [20, 30]], [[-5, 0], [5, 100]]], np.int16)
    movie = np.array(
        [[[-1.0, 10.4], [20.6, 31.0]], [[-5.0, 0.49], [4.5, 100.0]]], np.float32
    )

    saved, clipped = as_raw(movie, raw)

    assert saved.dtype == np.int16
    assert saved.tolist() == [[[0, 10], [21, 30]], [[-5, 0], [4, 100]]]
    assert clipped == 2

    with pytest.raises(ValueError, match="is"):
        as_raw(movie[:1], raw)


def test_a_final_correction_leaves_every_frames_match_to_check(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    group.run_motion_correction(
        use_gui_parameters=False, is_test=False, pw_rigid=False, batch_acqs=3
    )

    # Every frame of every acquisition, in order, raw and corrected
    folder = group.mcor_test_folder()
    full = np.load(folder / "full" / "shifts_summary.npz")
    assert full["acq_ids"].tolist() == sorted(group.acquisitions.index)
    assert len(full["corr"]) == len(full["raw_corr"]) == 8 * 56
    assert np.all(np.abs(full["corr"]) <= 1)
    # The stand-in leaves the movies as they were
    np.testing.assert_allclose(full["corr"], full["raw_corr"], atol=1e-5)

    # No means any more: the full run's videos show motion within each one
    assert not (folder / "final").exists()


def test_without_a_reference_a_final_correction_is_one_batch(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)

    # caiman then carries its template from file to file: no batches
    db.groups[group_id].run_motion_correction(
        use_gui_parameters=False,
        is_test=False,
        pw_rigid=False,
        reference_acqs=0,
        batch_acqs=3,
    )

    assert [len(names) for names in caiman["batches"]] == [8]
    assert caiman["templates"] == [None]


def test_a_test_leaves_the_files_to_judge_it_by(tmp_path, caiman, monkeypatch):
    import odyn.mcor_report as report

    db, group_id = a_loop(tmp_path)
    titles = []
    plot = report.plot_shifts
    monkeypatch.setattr(
        report,
        "plot_shifts",
        lambda *args: titles.append(args[-1]) or plot(*args),
    )

    folder = db.groups[group_id].test_motion_correction(
        test_acqs=4, movie=False, side_by_side=True
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
    # The reference: one file for the experiment, named, not copied here
    assert not (folder / "reference.tif").exists()
    assert reference_of(folder).shape == (48, 40)
    assert (folder / "shifts.png").stat().st_size > 0

    # The per-frame numbers the app plots, every frame of the test
    summary_npz = np.load(folder / "shifts_summary.npz")
    counts = summary_npz["frame_counts"]
    assert len(summary_npz["dy"]) == counts.sum() == len(summary_npz["corr"])
    assert summary_npz["acq_ids"].tolist() == summary["acq_id"].tolist()
    assert "dy_p10" not in summary_npz  # rigid: no patches
    assert titles == [f"20260304/m1/e1 — group {group_id} — test"]

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
    base = group.test_motion_correction(test_acqs=4, movie=False)
    reference = reference_of(base)
    caiman["template"] = None

    folder = group.test_motion_correction(
        test_acqs=4,
        pw_rigid=True,
        strides_um=[24, 24],
        overlap_um=[12, 12],
        max_deviation_um=3,
        run="s24_o12_d3",
        movie=False,
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
        movie=False,
        reuse_reference=True,
    )
    shifts = np.load(folder / "shifts.npz")
    assert "patch_dy" not in shifts and shifts["shifts"].shape == (4 * 56, 2)
    assert "patch_jitter_median_px" in json.loads((folder / "metrics.json").read_text())


def test_a_filtered_run_filters_the_reference_too(tmp_path, caiman):
    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    base = group.test_motion_correction(test_acqs=4, movie=False)
    reference = reference_of(base)

    folder = group.test_motion_correction(
        test_acqs=4,
        pw_rigid=True,
        strides_um=[24, 24],
        overlap_um=[12, 12],
        max_deviation_um=3,
        gsig_filt_um=4,
        run="s24_o12_d3_g4",
        movie=False,
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

    # The report compares with the reference itself, not the filtered one:
    # the same file as the run before, unchanged
    np.testing.assert_array_equal(reference_of(folder), reference)
    assert (
        json.loads((folder / "metrics.json").read_text())["reference"]
        == json.loads((base / "metrics.json").read_text())["reference"]
    )


def test_every_test_video_shares_the_raw_videos_frames_and_levels(
    tmp_path, caiman, monkeypatch
):
    from odyn.groups import Group

    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    asked = []

    def save_preview_movie(self, **kwargs):
        asked.append(kwargs)
        self.preview_levels = kwargs.get("levels") or (100.0, 900.0)
        path = Path(kwargs["save_folder"]) / f"G_{'_'.join(kwargs['grid'])}.avi"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(Group, "save_preview_movie", save_preview_movie)

    first = group.test_motion_correction(test_acqs=2, run="r1")
    group.test_motion_correction(test_acqs=2, run="r2")

    # The raw video once, in raw/ of the experiment's tests; each test its own
    raw = group.mcor_test_folder() / "raw"
    assert [a["grid"] for a in asked] == [["raw"], ["test"], ["test"]]
    assert asked[0]["save_folder"] == str(raw) and asked[0]["levels"] is None
    assert [a["levels"] for a in asked[1:]] == [[100.0, 900.0]] * 2
    assert all(a["raw_of_test"] for a in asked)
    assert [a["title"] for a in asked] == ["raw", "r1", "r2"]
    saved = json.loads((raw / "levels.json").read_text())
    assert saved["movie_levels"] == [100.0, 900.0] and len(saved["acq_ids"]) == 2
    metrics = json.loads((first / "metrics.json").read_text())
    assert metrics["movie_levels"] == [100.0, 900.0]

    # Other acquisitions: the raw video is written again for them
    group.test_motion_correction(test_acqs=4, run="r3")
    assert [a["grid"] for a in asked[3:]] == [["raw"], ["test"]]


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
    _, group_id = a_loop(tmp_path, user="alice")

    job_id = submit_mcor_test(tmp_path, None, group_id)

    assert job_id == "4242"
    assert "sbatch --parsable" in slurm.read_text()
    script = (
        cluster_folder(tmp_path) / "jobs" / f"mcor-test-g{group_id}.sh"
    ).read_text()
    assert "#SBATCH --gres=disk:128" in script
    assert 'CAIMAN_TEMP="/mnt/scratch/${SLURM_JOB_ID}"' in script
    assert "trap 'rm -rf \"$CAIMAN_TEMP\"' EXIT" in script
    assert f"run-mcor-test --group {group_id} --user alice" in script

    sent = json.loads(
        (cluster_folder(tmp_path) / "mcor" / f"{group_id}.json").read_text()
    )
    assert (sent["job_id"], sent["user"], sent["whole"]) == ("4242", "alice", False)


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

    _, group_id = a_loop(tmp_path, user="alice")
    seen = []
    monkeypatch.setattr(
        Group,
        "test_motion_correction",
        lambda self, whole: seen.append((self.group_id, self.db.user, whole))
        or tmp_path,
    )

    assert run_mcor_test(tmp_path, None, group_id, "alice") == tmp_path
    assert run_mcor_test(tmp_path, None, group_id, "alice", whole=True) == tmp_path
    assert seen == [(group_id, "alice", False), (group_id, "alice", True)]


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
    _, group_id = a_loop(tmp_path, user="alice")

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
        "alice",
        "4242",
        "4242",
    )
    assert sweep["runs"][0]["run"] == "s128_o64_d6"


def test_the_runs_take_the_baselines_reference_and_levels(tmp_path, slurm, monkeypatch):
    from odyn.groups import Group

    db, group_id = a_loop(tmp_path, user="alice")
    submit_mcor_sweep(tmp_path, None, group_id, GRID)
    sweep = cluster_folder(tmp_path) / "sweeps" / f"g{group_id}.json"
    asked = []
    monkeypatch.setattr(
        Group,
        "test_motion_correction",
        lambda self, **kwargs: asked.append((self.db.user, kwargs)) or tmp_path,
    )

    run_mcor_sweep(sweep, 0)
    run_mcor_sweep(sweep, 2)

    (user, first), (_, second) = asked
    assert user == "alice"
    assert first["run"] == "rigid" and "pw_rigid" not in first
    assert "movie" not in first and "levels" not in first  # the raw video's
    assert first["side_by_side"] is False
    assert second["keep_patch_shifts"] is False
    assert second["run"] == "s128x96_o32_d2.5"
    assert second["pw_rigid"] is True and second["reuse_reference"] is True
    assert second["strides_um"] == [128, 96]
    assert "levels" not in second
    assert (second["test_acqs"], second["max_shift_um"]) == (4, [20, 20])


def a_whole_report(db, group_id, **settings) -> Path:
    """The report of a whole correction of `group_id`, with `settings`."""
    folder = db.groups[group_id].mcor_test_folder()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(
        json.dumps({"settings": {"run": None, "whole": True, **settings}})
    )

    return folder


RIGID = {"pw_rigid": False, "max_shift_um": [80.0, 80.0], "shifts_opencv": False}
PIECEWISE = {
    **RIGID,
    "pw_rigid": True,
    "strides_um": [128.0, 128.0],
    "overlap_um": [64.0, 64.0],
    "max_deviation_um": 6.0,
}


def test_a_picked_test_is_recorded_and_only_corrected_when_new(tmp_path):
    from odyn.groups import candidate_settings

    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]

    # A test's report has more than the settings picked; a rigid one's
    # patches do not count
    assert (
        candidate_settings(
            {**RIGID, "run": "rigid", "max_shifts_px": [9, 9], "strides_um": [1, 1]}
        )
        == RIGID
    )

    # Nothing corrected whole yet: picking takes a correction
    assert group.pick_mcor_candidate(run="rigid", settings=RIGID) is True

    # The last whole correction had the same settings: nothing to redo
    a_whole_report(db, group_id, **RIGID, max_shifts_px=[9, 9])
    assert group.pick_mcor_candidate(run="rigid", settings=RIGID) is False
    assert group.pick_mcor_candidate(run="s128", settings=PIECEWISE) is True

    calls = db.latest_calls("pick_mcor_candidate")
    assert calls["user"].tolist() == ["alice"] * 3
    assert calls["output_correct"].tolist() == [True, False, True]

    with pytest.raises(ValueError, match="Unknown settings run"):
        group.pick_mcor_candidate(run="x", settings={"run": "x"})


def test_the_window_knows_if_a_pick_repeats_the_last_whole_run(tmp_path):
    from odyn.cluster import pick_repeats

    db, group_id = a_loop(tmp_path)
    folders = {}

    for run, settings in (("rigid", RIGID), ("s128", PIECEWISE)):
        folders[run] = tmp_path / run
        folders[run].mkdir()
        (folders[run] / "metrics.json").write_text(
            json.dumps({"settings": {"run": run, **settings}})
        )

    # Nothing whole yet: a pick corrects (said as over the rigid run)
    assert pick_repeats(db, group_id, folders["rigid"]) == (False, True)

    # The rigid full run: picking rigid again goes on to the final check
    a_whole_report(db, group_id, **RIGID)
    assert pick_repeats(db, group_id, folders["rigid"]) == (True, True)
    assert pick_repeats(db, group_id, folders["s128"]) == (False, True)

    # After a piecewise one: rigid overwrites its files
    a_whole_report(db, group_id, **PIECEWISE)
    assert pick_repeats(db, group_id, folders["rigid"]) == (False, False)


def test_the_tick_corrects_with_the_test_picked_once(tmp_path, slurm, monkeypatch):
    from odyn.cluster import TickReport, pick_candidate, submit_picks

    db, group_id = a_loop(tmp_path)
    a_whole_report(db, group_id, **RIGID)
    run = db.groups[group_id].mcor_test_folder("s128_o64_d6")
    run.mkdir()
    (run / "metrics.json").write_text(
        json.dumps({"settings": {"run": "s128_o64_d6", **PIECEWISE}})
    )

    # Refused while the last whole correction runs: both write the same files
    submit_mcor_test(tmp_path, None, group_id, user="alice", whole=True)
    (tmp_path / "queue.txt").write_text("4242 RUNNING\n")
    with pytest.raises(ValueError, match="has not finished yet"):
        pick_candidate(db, group_id, "s128_o64_d6", run)

    (tmp_path / "queue.txt").write_text("")
    log = cluster_folder(tmp_path) / "jobs" / f"odyn-mcor-main-g{group_id}-4242.out"
    log.write_text("started  ...\nfinished now\n")
    picker = Database(tmp_path, user="bob")
    assert pick_candidate(picker, group_id, "s128_o64_d6", run) is True

    def submitted():
        report = TickReport(started_at="now", dry_run=False)
        submit_picks(Database(tmp_path), None, report, dry_run=False)
        return report.tests

    # Submitted as who picked it, piecewise, with the longer time
    assert submitted() == [
        f"s128_o64_d6 (all) of group {group_id} submitted (job 4242)"
    ]
    script = (
        cluster_folder(tmp_path) / "jobs" / f"mcor-test-g{group_id}.sh"
    ).read_text()
    assert f"--group {group_id} --user bob --whole --settings" in script
    assert "--label 's128_o64_d6 (all)'" in script
    assert "#SBATCH --time=12:00:00" in script
    sent = json.loads(
        (cluster_folder(tmp_path) / "mcor" / f"{group_id}.json").read_text()
    )
    assert sent["settings"] == PIECEWISE and sent["label"] == "s128_o64_d6 (all)"

    # Never twice
    assert submitted() == []
    assert slurm.read_text().count("sbatch") == 2

    # The job runs the test whole with those settings
    seen = {}
    monkeypatch.setattr(
        "odyn.groups.Group.test_motion_correction",
        lambda self, **kwargs: seen.update(kwargs) or tmp_path,
    )
    run_mcor_test(
        tmp_path, None, group_id, "bob", True, sent["settings"], sent["label"]
    )
    assert seen == {"whole": True, **PIECEWISE, "label": "s128_o64_d6 (all)"}


def test_a_video_frame_map_says_where_each_frame_is_from(tmp_path):
    from odyn.groups import frame_map_path, read_frame_map, write_frame_map

    video = tmp_path / "e1_test.avi"
    assert frame_map_path(video) == tmp_path / "e1_test.frames.csv"
    write_frame_map(frame_map_path(video), [5, 5, 9], [0, 50, 0])
    frames = read_frame_map(frame_map_path(video))
    assert frames.values.tolist() == [[0, 5, 0], [1, 5, 50], [2, 9, 0]]
    assert read_frame_map(tmp_path / "missing.frames.csv") is None


def test_the_shifts_summary_has_the_patches_spread(tmp_path):
    from odyn.mcor_report import write_shifts_summary

    shifts = np.array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
    patches = (
        np.array([[0, 1, 2, 10]] * 3, float),
        np.array([[2, 2, 2, 2]] * 3, float),
    )
    path = tmp_path / "shifts_summary.npz"
    write_shifts_summary(path, shifts, patches, [0.2] * 3, [0.5] * 3, [7], [3])
    data = np.load(path)
    assert data["dy"].tolist() == [1.0, 1.5, 2.0]
    assert data["dy_p90"][0] > data["dy_p10"][0] and data["dx_p10"][0] == 2
    assert data["raw_corr"].tolist() == pytest.approx([0.2] * 3)


def test_frame_correlations_match_the_reports(tmp_path):
    from odyn.mcor_report import _summarize, frame_correlations

    rng = np.random.default_rng(0)
    reference = rng.random((30, 40))
    movie = reference + rng.normal(0, 0.5, (150, 30, 40))
    _, expected = _summarize(movie, reference, 3)
    assert frame_correlations(movie, reference, 3, chunk=64) == pytest.approx(
        expected, abs=1e-5
    )


def test_the_full_previews_share_the_raw_levels(tmp_path, monkeypatch):
    from odyn.groups import Group

    db, group_id = a_loop(tmp_path)
    asked = []

    def save_preview_movie(self, **kwargs):
        asked.append(kwargs)
        self.preview_levels = kwargs.get("levels") or (100.0, 900.0)
        return Path(kwargs["save_folder"]) / f"G_{kwargs['grid'][0]}.avi"

    monkeypatch.setattr(Group, "save_preview_movie", save_preview_movie)
    paths = db.groups[group_id].save_full_previews(label="s128 (all)")

    assert [a["grid"] for a in asked] == [["raw"], ["mcor"]]
    assert [a["size"] for a in asked] == [0.5, 0.5]
    assert asked[1]["levels"] == (100.0, 900.0) and asked[1]["title"] == "s128 (all)"
    assert all(Path(a["save_folder"]).name == "full" for a in asked)
    assert [p.name for p in paths] == ["G_raw.avi", "G_mcor.avi"]


def test_the_raw_check_records_what_is_left_out(tmp_path):
    from odyn.app.pane_qc import _raw_check_line, latest_raw_checks

    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    acq_ids = [int(i) for i in group.acquisitions.index]
    assert group.raw_exclusions() is None
    assert latest_raw_checks(db) == {}

    assert group.check_raw(exclude_acq_ids=[acq_ids[3], acq_ids[1]]) == sorted(
        [acq_ids[1], acq_ids[3]]
    )
    assert group.raw_exclusions() == sorted([acq_ids[1], acq_ids[3]])

    # The latest counts; one that failed does not
    group.check_raw()
    with pytest.raises(ValueError, match="not acquisitions"):
        group.check_raw(exclude_acq_ids=[9999])

    assert group.raw_exclusions() == []
    assert latest_raw_checks(db) == {group_id: []}
    assert [_raw_check_line(x) for x in (None, [], [3, 5])] == [
        "not yet",
        "done",
        "done, 2 excluded",
    ]


def test_an_approval_is_found_for_the_qc_page(tmp_path, caiman):
    from odyn.app.pane_qc import latest_approvals

    db, group_id = a_loop(tmp_path)
    group = db.groups[group_id]
    assert latest_approvals(db) == {}
    group.approve_mcor_files()  # nothing saved yet: not an approval
    assert latest_approvals(db) == {}

    group.run_motion_correction(use_gui_parameters=False, is_test=False, pw_rigid=False)
    acq_ids = [int(i) for i in group.acquisitions.index]
    group.approve_mcor_files(exclude_acq_ids=[acq_ids[0]])
    calls = db.latest_calls("approve_mcor_files")
    assert latest_approvals(db) == {group_id: int(calls.index[0])}
    assert len(db.groups[group_id].approved_mcor_files) == len(acq_ids) - 1


def test_the_tick_records_which_jobs_stopped_for_the_app(tmp_path, slurm, monkeypatch):
    from odyn.cluster import mcor_tests, record_job_states

    db, group_id = a_loop(tmp_path)
    submit_mcor_test(tmp_path, None, group_id, user="alice", whole=True)
    logs = cluster_folder(tmp_path) / "jobs"
    (logs / f"odyn-mcor-main-g{group_id}-4242.out").write_text("started\nKilled\n")

    # Running, then ended without finishing, as squeue says on the cluster
    (tmp_path / "queue.txt").write_text("4242 RUNNING\n")
    assert record_job_states(tmp_path, None) == 1
    assert record_job_states(tmp_path, None) == 0  # nothing new
    (tmp_path / "queue.txt").write_text("")
    assert record_job_states(tmp_path, None) == 1
    assert mcor_tests(tmp_path)[0]["state"] == "stopped"

    # Off the cluster (no squeue), the state recorded stands in
    monkeypatch.setenv("PATH", "")
    assert mcor_tests(tmp_path)[0]["state"] == "stopped"
    assert record_job_states(tmp_path, None) == 0


def test_stopped_jobs_run_again_once_with_their_settings(tmp_path, slurm):
    from odyn.cluster import (
        TickReport,
        mcor_sweep_runs,
        sweep_runs,
        submit_reruns,
    )

    db, group_id = a_loop(tmp_path)
    folder = cluster_folder(tmp_path)
    submit_mcor_test(
        tmp_path,
        None,
        group_id,
        user="alice",
        whole=True,
        settings=PIECEWISE,
        label="s128 (all)",
        pick_call_id=7,
    )
    grid = {
        "runs": [
            {"strides_um": [128, 128], "overlap_um": [64, 64], "max_deviation_um": 6},
            {"strides_um": [64, 64], "overlap_um": [32, 32], "max_deviation_um": 3},
        ]
    }
    submit_mcor_sweep(tmp_path, None, group_id, grid, user="alice")
    names = [run["run"] for run in sweep_runs(grid)]

    def submit():
        report = TickReport(started_at="now", dry_run=False)
        submit_reruns(Database(tmp_path), None, report, dry_run=False)
        assert report.errors == []
        return report.tests

    # Nothing asked: nothing submitted
    before = slurm.read_text().count("sbatch")
    assert submit() == []

    # The correction again, as it was; once
    db.groups[group_id].run_again(runs=["correction"])
    assert submit() == [f"group {group_id}: s128 (all) again submitted (job 4242)"]
    sent = json.loads((folder / "mcor" / f"{group_id}.json").read_text())
    assert sent["settings"] == PIECEWISE and sent["label"] == "s128 (all)"
    assert sent["pick_call_id"] == 7 and "rerun_call_id" in sent
    assert submit() == []

    # The sweep's baseline again: the runs waiting on it go with it, after it
    db.groups[group_id].run_again(runs=["rigid"])
    assert submit() == [
        f"group {group_id}: sweep runs again submitted "
        f"({', '.join(sorted(['rigid', *names]))})"
    ]
    calls = slurm.read_text()
    assert calls.count("sbatch") == before + 3
    assert "scancel 4242" in calls  # the first array, waiting on nothing
    assert "--array=1,2 --output=" in calls and "--dependency=afterok:4242" in calls
    runs = mcor_sweep_runs(tmp_path)
    assert {run["job_id"] for run in runs} == {"4242", "4242_1", "4242_2"}
    assert submit() == []

    # One run alone: no baseline, no dependency
    db.groups[group_id].run_again(runs=[names[1]])
    submit()
    last = slurm.read_text().strip().splitlines()[-1]
    assert "--array=2 " in last and "dependency" not in last

    with pytest.raises(ValueError, match="Name the runs"):
        db.groups[group_id].run_again(runs=[])


def test_old_experiments_are_corrected_in_batches(tmp_path, slurm):
    from odyn.cluster import ClusterConfig, mcor_backfill, sweep_grid_for

    # Three experiments never tested, by two people, through two objectives
    for seed, (mouse, objective, user) in enumerate(
        ((1, 20, "alice"), (2, 10, "bob"), (3, 20, "bob"))
    ):
        generate(
            tmp_path,
            acquisitions=2,
            frames=56,
            height=32,
            width=40,
            mouse=f"m{mouse}",
            seed=seed,
            start=datetime(2026, 7, mouse, 9),
        )
        db = Database(tmp_path, can_create=True, user=user)
        db.set_mouse(mouse_id=mouse)
        session_id = db.add_session(mouse_id=mouse, session_date=f"2026-07-0{mouse}")
        db.add_experiment(
            session_id=session_id,
            exp_path=f"2026070{mouse}/m{mouse}/e1",
            objective=objective,
        )

    run = {"strides_um": [96, 96], "overlap_um": [48, 48], "max_deviation_um": 6}
    grid = {"test_acqs": 2, "runs": [run]}
    config = tmp_path / ".odyn" / "cluster.json"
    config.write_text(json.dumps({"sweep_grid": {"20": grid}}))

    # One grid for all, or one per objective
    assert sweep_grid_for(ClusterConfig(sweep_grid=grid), 10) == grid
    assert sweep_grid_for(ClusterConfig(sweep_grid={"20": grid}), 10) is None
    assert sweep_grid_for(ClusterConfig(), 20) is None

    # Said first; the 20x ones, oldest first, one at a time
    assert mcor_backfill(tmp_path, None, batch=1, objective=20, dry_run=True) == [
        "20260701/m1/e1 (20x, group 1): would submit the whole correction and a sweep",
        "1 more waiting for the next batch",
    ]
    assert not slurm.exists()  # nothing submitted

    done = mcor_backfill(tmp_path, None, batch=1, objective=20)
    assert done == [
        "20260701/m1/e1 (20x, group 1): whole correction job 4242, sweep jobs "
        "4242 and 4242",
        "1 more waiting for the next batch",
    ]
    sent = json.loads((cluster_folder(tmp_path) / "mcor" / "1.json").read_text())
    assert sent["whole"] and sent["user"] == "alice"

    # The next batch: what was sent is not again; no grid for 10x yet
    assert mcor_backfill(tmp_path, None, batch=4) == [
        "20260702/m2/e1 (10x, group 2): no sweep grid for 10x in cluster.json, "
        "skipped",
        "20260703/m3/e1 (20x, group 3): whole correction job 4242, sweep jobs "
        "4242 and 4242",
    ]
    sent = json.loads((cluster_folder(tmp_path) / "mcor" / "3.json").read_text())
    assert sent["user"] == "bob"
    assert slurm.read_text().count("sbatch") == 6
