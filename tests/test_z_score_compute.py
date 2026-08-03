"""
Array maths behind the z-score movies (`_z_score_from_average`).

Everything here runs on synthetic movies, so the properties can be checked
against a known answer: pure noise must score near zero with a unit noise
floor, an injected response must show up in the odor map and nowhere else, and
pixels that never vary must come out as exactly zero rather than infinity.
"""

import numpy as np
import pytest

from odyn.groups import _resolve_windows, _z_score_from_average


FRAME_RATE = 14.0
ONSET = 70
ODOR = 56
COUNT = 280

# Where the injected response lives
PATCH = (slice(8, 16), slice(8, 16))


def make_windows(**kwargs):
    return _resolve_windows(
        **{
            "frame_rate": FRAME_RATE,
            "onset_frames": [ONSET],
            "odor_frames": [ODOR],
            "frame_counts": [COUNT],
            **kwargs,
        }
    )


def make_average(windows, *, height=32, width=32, noise=1.0, response=0.0, seed=0):
    """
    A synthetic averaged movie, indexed from `first_raw_frame`.

    `response` is added to `PATCH` for the whole odor presentation, on top of a
    flat baseline of 1000 (roughly a real fluorescence level).
    """
    first = windows.first_frame - windows.half_frames
    last = windows.last_frame + windows.half_frames
    rng = np.random.default_rng(seed)

    movie = 1000.0 + noise * rng.standard_normal((last - first + 1, height, width))

    if response:
        odor = slice(-first, -first + ODOR)
        movie[(odor, *PATCH)] += response

    return movie.astype(np.float32), first


def z_score(windows, **kwargs):
    average, first = make_average(windows, **kwargs)
    return _z_score_from_average(average, first_raw_frame=first, windows=windows)


# --------------------------------------------------------------------------- #
# Shapes and consistency
# --------------------------------------------------------------------------- #


def test_stack_has_one_frame_per_window():
    w = make_windows()
    result = z_score(w)

    assert result.stack.shape == (len(w.stack_centres), 32, 32)
    assert result.odor_map.shape == (32, 32)
    assert result.stack.dtype == np.float32


def test_odor_map_is_the_mean_of_the_odor_windows():
    """The map must be derivable from the stack, or the two can disagree."""
    w = make_windows()
    result = z_score(w)

    first = w.onset_window_index + w.odor_first_k
    expected = result.stack[first : first + w.odor_windows].mean(axis=0)

    assert np.allclose(result.odor_map, expected)


def test_baseline_statistics_have_frame_shape():
    result = z_score(make_windows())

    assert result.baseline_mean.shape == (32, 32)
    assert result.baseline_std.shape == (32, 32)
    assert np.all(result.baseline_std > 0)


# --------------------------------------------------------------------------- #
# Pure noise
# --------------------------------------------------------------------------- #


def test_pure_noise_scores_near_zero():
    result = z_score(make_windows(), noise=1.0)

    assert abs(float(result.odor_map.mean())) < 0.5
    assert float(np.abs(result.odor_map).max()) < 6.0


def test_sanity_spreads_are_near_one_on_pure_noise():
    """Correctly normalised noise scores at about unit SD either side of the odor."""
    stats = z_score(make_windows(), noise=1.0).stats

    assert 0.7 < stats["pre_odor_sd"] < 1.4
    assert 0.7 < stats["post_odor_sd"] < 1.4
    assert stats["pre_odor_windows"] >= 2
    assert stats["post_odor_windows"] >= 2


def test_post_odor_spread_excludes_the_straddling_window():
    """
    With an odor that is not a whole number of windows, the window covering the
    offset is part response and part not. Counting it would make a clean
    recording look like it was drifting.
    """
    w = make_windows(odor_frames=[51])
    average, first = make_average(w, noise=1.0, response=6.0)
    # The response is added for ODOR frames, so trim it to the real 51
    average[(slice(-first + 51, -first + ODOR), *PATCH)] -= 6.0

    stats = _z_score_from_average(average, first_raw_frame=first, windows=w).stats

    assert stats["post_odor_windows"] == len(w.stack_centres) - (
        w.onset_window_index + w.post_odor_first_k
    )
    assert stats["post_odor_sd"] < 1.4


def test_post_odor_spread_detects_a_drifting_baseline():
    """
    A ramp is the signature of photobleaching or z-drift. It does *not* show up
    before the odor, because the baseline was fitted over that same period and
    absorbs it — which is exactly why the held-out number is the useful one.
    """
    w = make_windows()
    average, first = make_average(w, noise=1.0)
    ramp = np.arange(len(average), dtype=np.float32) * 0.5
    average += ramp[:, None, None]

    stats = _z_score_from_average(average, first_raw_frame=first, windows=w).stats

    assert stats["post_odor_sd"] > 3.0
    assert stats["pre_odor_sd"] < 1.5


# --------------------------------------------------------------------------- #
# Injected response
# --------------------------------------------------------------------------- #


def test_response_shows_up_in_the_odor_map():
    w = make_windows()
    result = z_score(w, noise=1.0, response=3.0)

    patch = result.odor_map[PATCH]
    background = np.delete(result.odor_map.ravel(), _patch_indices())

    assert patch.min() > 3.0
    assert patch.mean() > 5.0 * abs(background.mean())


def test_response_does_not_leak_before_onset():
    """Baseline windows end half a window before onset, so they stay clean."""
    w = make_windows()
    result = z_score(w, noise=1.0, response=3.0)

    before_odor = result.stack[: w.onset_window_index]

    assert np.abs(before_odor[(slice(None), *PATCH)]).max() < 4.0


def test_larger_response_scores_higher():
    w = make_windows()
    small = z_score(w, noise=1.0, response=1.0).odor_map[PATCH].mean()
    large = z_score(w, noise=1.0, response=4.0).odor_map[PATCH].mean()

    assert large > small > 0


def test_response_scales_with_the_inverse_of_the_noise():
    """z is a signal-to-noise ratio: double the noise, halve the score."""
    w = make_windows()
    quiet = z_score(w, noise=1.0, response=4.0).odor_map[PATCH].mean()
    noisy = z_score(w, noise=2.0, response=4.0).odor_map[PATCH].mean()

    assert noisy == pytest.approx(quiet / 2, rel=0.25)


# --------------------------------------------------------------------------- #
# Degenerate pixels
# --------------------------------------------------------------------------- #


def test_constant_pixels_become_zero_not_infinity():
    """Masked borders have no variation, so they would divide by zero."""
    w = make_windows()
    average, first = make_average(w, noise=1.0, response=3.0)
    average[:, 0, :] = 1000.0

    result = _z_score_from_average(average, first_raw_frame=first, windows=w)

    assert np.all(np.isfinite(result.stack))
    assert np.all(result.odor_map[0, :] == 0.0)
    assert result.stats["zeroed_pixel_fraction"] == pytest.approx(1 / 32)


def test_non_finite_pixels_become_zero():
    """`border_nan` in motion correction can leave NaN edges."""
    w = make_windows()
    average, first = make_average(w, noise=1.0)
    average[:, :, -1] = np.nan

    result = _z_score_from_average(average, first_raw_frame=first, windows=w)

    assert np.all(np.isfinite(result.odor_map))
    assert np.all(result.odor_map[:, -1] == 0.0)


def test_stats_report_the_map_spread():
    result = z_score(make_windows(), noise=1.0, response=3.0)
    percentiles = result.stats["odor_map_percentiles"]

    assert percentiles["p1"] < percentiles["p50"] < percentiles["p99"]
    assert result.stats["odor_map_max_abs"] >= percentiles["p99_9"]
    assert result.stats["baseline_std_median"] > 0


def test_stats_keys_survive_json_normalize():
    """
    `latest_calls` flattens `call_output` with `json_normalize`, which splits on
    ".", so no stats key may contain one.
    """
    stats = z_score(make_windows(), noise=1.0).stats

    def keys(value):
        if not isinstance(value, dict):
            return
        for key, nested in value.items():
            assert "." not in key, key
            keys(nested)

    keys(stats)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _patch_indices():
    flat = np.zeros((32, 32), dtype=bool)
    flat[PATCH] = True
    return np.flatnonzero(flat.ravel())
