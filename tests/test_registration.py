"""
The reference image `run_motion_correction` aligns a loop to, built without
caiman: a synthetic scene, drifting by known amounts, with noise and some
frames from somewhere else.
"""

import numpy as np
import tifffile

from scipy import ndimage

from odyn.registration import (
    _align,
    reference_acquisitions,
    reference_image,
    sample_frames,
    spread_acquisitions,
)


def scene(rng, size=96):
    """Blobs, like glomeruli, on a dim background."""
    image = ndimage.gaussian_filter(rng.random((size, size)), 3)
    return (image - image.min()) / np.ptp(image)


def correlation(a, b, margin=8):
    a = a[margin:-margin, margin:-margin].ravel()
    b = b[margin:-margin, margin:-margin].ravel()
    return np.corrcoef(a, b)[0, 1]


def test_a_shift_is_found_to_a_fraction_of_a_pixel():
    rng = np.random.default_rng(0)
    truth = scene(rng)
    moved = ndimage.shift(truth, (2.4, -3.7), order=3, mode="nearest")

    (dy, dx), score = _align(moved, truth, (8, 8))

    assert abs(dy + 2.4) < 0.25 and abs(dx - 3.7) < 0.25  # undoes the shift
    assert score > 0.95


def test_the_reference_is_sharp_despite_drift_noise_and_strangers():
    rng = np.random.default_rng(1)
    truth = scene(rng)
    drift = np.cumsum(rng.normal(0, 0.4, size=(60, 2)), axis=0)  # a slow walk
    frames = [
        ndimage.shift(truth, d, order=3, mode="nearest")
        + rng.normal(0, 0.15, truth.shape)
        for d in drift
    ]
    # A fifth of the frames from another plane
    elsewhere = scene(rng)
    frames += [elsewhere + rng.normal(0, 0.15, truth.shape) for _ in range(15)]

    reference = reference_image(np.array(frames), (12, 12))

    # Aligned to the scene (up to one overall shift), and sharper than any frame
    (dy, dx), _ = _align(reference, truth, (12, 12))
    placed = ndimage.shift(reference, (dy, dx), order=1, mode="nearest")
    assert correlation(placed, truth) > 0.97
    assert correlation(placed, truth) > max(correlation(f, truth) for f in frames[:5])
    assert correlation(placed, elsewhere) < 0.5


def test_the_reference_acquisitions_avoid_the_start_and_the_exact_middle():
    assert reference_acquisitions(160) == [60, 73, 86, 100]
    assert 112 not in reference_acquisitions(224)
    assert reference_acquisitions(3) == [1]
    assert reference_acquisitions(0) == []


def test_a_test_spreads_evenly_around_the_middle():
    assert spread_acquisitions(10, 4) == [1, 3, 6, 8]

    picks = spread_acquisitions(224, 24)
    assert len(picks) == 24
    assert set(np.diff(picks)) <= {9, 10}  # even steps of 224 / 24
    assert picks[0] + picks[-1] == 223  # centered: as far from either end

    # A short loop is tested whole
    assert spread_acquisitions(8, 24) == list(range(8))
    assert spread_acquisitions(0, 24) == []


def test_frames_are_sampled_evenly_from_each_file(tmp_path):
    paths = []

    for index in range(2):
        path = tmp_path / f"acq{index}.tif"
        tifffile.imwrite(
            path, np.arange(10)[:, None, None] * np.ones((10, 4, 5)) + index * 100
        )
        paths.append(path)

    frames = sample_frames(paths, per_file=3)

    assert frames.shape == (6, 4, 5) and frames.dtype == np.float32
    assert list(frames[:, 0, 0]) == [0, 4, 9, 100, 104, 109]  # rounded positions


def test_run_motion_correction_hands_caiman_one_reference(tmp_path, monkeypatch):
    """With a stand-in for caiman, which this test only watches."""
    import sys
    import types
    from datetime import datetime

    from generate_data import generate

    from odyn import Database

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
    exp_id = db.add_experiment(
        session_id=session_id, exp_path="20260304/m1/e1", objective=20
    )
    group_id = int(
        db.group_experiments.index[db.group_experiments["exp_id"] == exp_id][0]
    )
    group = db.groups[group_id]

    seen = {}

    class MotionCorrect:
        def __init__(self, fnames, dview=None, **settings):
            self.fname, self.mmap_file = fnames, []
            seen["files"] = len(fnames)

        def motion_correct(self, template=None, save_movie=False):
            seen["template"] = template

    caiman = types.ModuleType("caiman")
    caiman.cluster = types.SimpleNamespace(setup_cluster=lambda **_: (None, None, None))
    caiman.stop_server = lambda **_: None
    motion = types.ModuleType("caiman.motion_correction")
    motion.MotionCorrect = MotionCorrect
    caiman.motion_correction = motion
    monkeypatch.setitem(sys.modules, "caiman", caiman)
    monkeypatch.setitem(sys.modules, "caiman.motion_correction", motion)

    # A test on two acquisitions still uses the reference of the whole loop
    group.run_motion_correction(use_gui_parameters=False, is_test=True, test_acqs=2)

    assert seen["files"] == 2 and seen["template"].shape == (48, 40)
    call = db.latest_calls("run_motion_correction").iloc[0]
    assert call["input_reference_acq_ids"] == [4, 5, 6]  # middle third of 8 (3 of them)
    assert call["input_acq_ids"] == [3, 7]  # positions 2 and 6 of 8
    outputs = db.outputs
    assert outputs["file_path"].str.endswith(f"reference_group_{group_id}.tif").any()

    # 0 leaves the template to caiman, as before
    group.run_motion_correction(
        use_gui_parameters=False, is_test=True, test_acqs=2, reference_acqs=0
    )
    assert seen["template"] is None
