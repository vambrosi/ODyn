"""
Rendering of the z-score movie (`_write_movie`).

The movie is a picture, not data, so what matters is that it is readable: right
number of frames, right size, zero always the neutral colour, a fixed range so
two conditions can be compared by eye, and the odor marked while it is on.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from odyn.groups import (  # noqa: E402
    _resolve_windows,
    _write_movie,
    _z_score_from_average,
)


FRAME_RATE = 14.0
ONSET = 70
ODOR = 56
COUNT = 280


@pytest.fixture
def rendered(tmp_path):
    """A rendered movie plus the pieces that went into it."""
    windows = _resolve_windows(
        frame_rate=FRAME_RATE,
        onset_frames=[ONSET],
        odor_frames=[ODOR],
        frame_counts=[COUNT],
    )
    first = windows.first_frame - windows.half_frames
    last = windows.last_frame + windows.half_frames

    rng = np.random.default_rng(0)
    average = (
        1000.0 + rng.standard_normal((last - first + 1, 24, 32))
    ).astype(np.float32)
    average[(slice(-first, -first + ODOR), slice(6, 12), slice(6, 12))] += 8.0

    result = _z_score_from_average(average, first_raw_frame=first, windows=windows)
    path = tmp_path / "nested" / "movie.avi"
    _write_movie(path, average, result, first_raw_frame=first)

    return path, windows, result, average, first


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    return frames


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_movie_has_one_frame_per_usable_centre(rendered):
    path, windows, *_ = rendered
    frames = read_frames(path)

    assert len(frames) == windows.last_frame - windows.first_frame + 1


def test_movie_matches_the_recording_size(rendered):
    path, _, result, *_ = rendered

    assert read_frames(path)[0].shape == (*result.baseline_mean.shape, 3)


def test_parent_directory_is_created(rendered):
    path, *_ = rendered

    assert path.exists() and path.parent.name == "nested"


# --------------------------------------------------------------------------- #
# Colour mapping
# --------------------------------------------------------------------------- #


def test_zero_maps_to_the_neutral_colour(rendered):
    """
    A diverging map centred on zero is the whole point: a pixel that did not
    change must not look like one that did, in either direction.
    """
    path, windows, result, average, first = rendered

    # A pre-odor frame is noise around zero, so it should be near-neutral
    frame = read_frames(path)[0].astype(int)
    spread = np.abs(frame[..., 2] - frame[..., 0])  # red minus blue

    assert np.median(spread) < 60


def test_range_is_fixed_not_auto_scaled(tmp_path):
    """
    Auto-scaling per condition would make a weak response look like a strong
    one. Two conditions differing only in response size must render differently.

    Note the amplitudes are kept inside the display range: past it both
    saturate, and scaling the *whole* movie changes nothing at all, because z is
    a ratio and divides the noise out along with the signal.
    """

    def render(response, name):
        windows = _resolve_windows(
            frame_rate=FRAME_RATE,
            onset_frames=[ONSET],
            odor_frames=[ODOR],
            frame_counts=[COUNT],
        )
        first = windows.first_frame - windows.half_frames
        last = windows.last_frame + windows.half_frames

        rng = np.random.default_rng(0)
        average = (
            1000.0 + rng.standard_normal((last - first + 1, 24, 32))
        ).astype(np.float32)
        average[(slice(-first, -first + ODOR), slice(6, 12), slice(6, 12))] += response

        result = _z_score_from_average(average, first_raw_frame=first, windows=windows)
        path = tmp_path / name
        _write_movie(path, average, result, first_raw_frame=first)

        odor_frame = windows.half_frames - windows.first_frame
        patch = read_frames(path)[odor_frame][6:12, 6:12].astype(int)

        return float(np.mean(patch[..., 2] - patch[..., 0]))  # red minus blue

    assert render(1.5, "strong.avi") > render(0.5, "faint.avi") > 0


def test_display_range_changes_saturation(tmp_path):
    """A narrower range saturates sooner, so more pixels hit the extremes."""
    windows = _resolve_windows(
        frame_rate=FRAME_RATE,
        onset_frames=[ONSET],
        odor_frames=[ODOR],
        frame_counts=[COUNT],
    )
    first = windows.first_frame - windows.half_frames
    last = windows.last_frame + windows.half_frames

    rng = np.random.default_rng(1)
    average = (
        1000.0 + rng.standard_normal((last - first + 1, 24, 32))
    ).astype(np.float32)
    average[(slice(-first, -first + ODOR), slice(6, 12), slice(6, 12))] += 8.0
    result = _z_score_from_average(average, first_raw_frame=first, windows=windows)

    def extreme_fraction(display_range):
        path = tmp_path / f"r{display_range}.avi"
        _write_movie(
            path, average, result, first_raw_frame=first, display_range=display_range
        )
        stacked = np.stack(read_frames(path))
        return float(np.mean((stacked == 0) | (stacked == 255)))

    assert extreme_fraction(2.0) > extreme_fraction(20.0)


# --------------------------------------------------------------------------- #
# Odor marker
# --------------------------------------------------------------------------- #


def test_odor_marker_is_on_during_the_odor_and_off_outside(rendered):
    path, windows, *_ = rendered
    frames = read_frames(path)

    def corner_is_red(frame):
        radius = max(4, frame.shape[0] // 60)
        patch = frame[2 * radius, 2 * radius].astype(int)
        return patch[2] > 200 and patch[0] < 80 and patch[1] < 80

    def at(relative_frame):
        return frames[relative_frame - windows.first_frame]

    assert not corner_is_red(at(-1))
    assert corner_is_red(at(0))
    assert corner_is_red(at(windows.odor_duration - 1))
    assert not corner_is_red(at(windows.odor_duration))


# --------------------------------------------------------------------------- #
# Degenerate pixels
# --------------------------------------------------------------------------- #


def test_constant_pixels_do_not_crash_the_renderer(tmp_path):
    """Zero-variance and NaN pixels would otherwise become NaN colour indices."""
    windows = _resolve_windows(
        frame_rate=FRAME_RATE,
        onset_frames=[ONSET],
        odor_frames=[ODOR],
        frame_counts=[COUNT],
    )
    first = windows.first_frame - windows.half_frames
    last = windows.last_frame + windows.half_frames

    rng = np.random.default_rng(2)
    average = (
        1000.0 + rng.standard_normal((last - first + 1, 24, 32))
    ).astype(np.float32)
    average[:, 0, :] = 1000.0
    average[:, :, -1] = np.nan

    result = _z_score_from_average(average, first_raw_frame=first, windows=windows)
    path = tmp_path / "degenerate.avi"
    _write_movie(path, average, result, first_raw_frame=first)

    frames = read_frames(path)

    assert len(frames) == windows.last_frame - windows.first_frame + 1
