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
    condition_positions,
    reference_acquisitions,
    reference_by_condition,
    reference_image,
    sample_frames,
    segments,
    spread_acquisitions,
    spread_by_condition,
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

    # Or from some of the frames of a file (a segment of a long acquisition)
    frames = sample_frames(paths, per_file=3, ranges=[(5, 5), None])
    assert list(frames[:, 0, 0]) == [5, 7, 9, 100, 104, 109]


def test_a_long_acquisition_is_cut_into_segments():
    # A loop is not cut: an acquisition shorter than one and a half segments
    # is one
    assert segments([56, 596, 749]) == [(0, 0, 56), (1, 0, 596), (2, 0, 749)]

    # A grab is cut into segments of about 500 frames, one after the other
    cut = segments([400, 13000])
    assert cut[0] == (0, 0, 400) and len(cut) == 27
    assert {frames for _, _, frames in cut[1:]} == {500}
    assert [first for _, first, _ in cut[1:4]] == [0, 500, 1000]
    assert segments([1201]) == [(0, 0, 600), (0, 600, 601)]
    assert segments([750], length=500) == [(0, 0, 375), (0, 375, 375)]


def test_each_condition_gets_its_share(tmp_path):
    conditions = ["baseline"] * 6 + ["ket/xyl"] * 2

    assert condition_positions(conditions) == [[0, 1, 2, 3, 4, 5], [6, 7]]
    assert condition_positions([None, "a", None]) == [[0, 2], [1]]

    # The test: half of the acquisitions from each, the earlier ones first
    assert spread_by_condition(conditions, 4) == [1, 4, 6, 7]
    assert spread_by_condition(conditions, 5) == [1, 3, 5, 6, 7]
    assert spread_by_condition(["a"] * 3 + ["b"] * 3 + ["c"] * 3, 2) == [1, 4]

    # The reference: the middle third of each
    assert reference_by_condition(conditions) == [[2, 3], [6, 7]]

    # One condition, or none: as before
    for one in ([None] * 224, ["baseline"] * 224):
        assert spread_by_condition(one, 24) == spread_acquisitions(224, 24)
        assert reference_by_condition(one) == [reference_acquisitions(224)]

    # As many frames from each file as asked
    paths = []

    for index in range(2):
        path = tmp_path / f"acq{index}.tif"
        tifffile.imwrite(path, np.arange(10)[:, None, None] * np.ones((10, 4, 5)))
        paths.append(path)

    assert len(sample_frames(paths, per_file=[2, 5])) == 7


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
    db = Database(tmp_path, can_create=True, user="alice")
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

    # Before and after anesthesia: the test and the reference from each alike
    for acq_id in group.acquisitions.index:
        db.add_annotations(
            target_type="acquisition",
            target_id=int(acq_id),
            values={"condition": "baseline" if acq_id <= 6 else "ket/xyl"},
        )

    group.run_motion_correction(use_gui_parameters=False, is_test=True, test_acqs=4)
    call = db.latest_calls("run_motion_correction").iloc[0]
    assert call["input_acq_ids"] == [2, 5, 7, 8]
    assert call["input_reference_acq_ids"] == [3, 4, 7, 8]
    assert call["input_reference_conditions"] == ["baseline", "ket/xyl"]

    # Unless asked not to
    group.run_motion_correction(
        use_gui_parameters=False, is_test=True, test_acqs=4, by_condition=False
    )
    call = db.latest_calls("run_motion_correction").iloc[0]
    assert call["input_acq_ids"] == [2, 4, 6, 8]
    assert call["input_reference_acq_ids"] == [4, 5, 6]
