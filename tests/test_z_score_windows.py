"""
Frame arithmetic behind the z-score movies (`_resolve_windows`).

Every index is a *centre* of the smoothing window, in frames relative to odor
onset. The properties worth pinning down are that no baseline window ever
reaches past onset, that the stack tiles onset with non-overlapping windows,
and that every window a caller is handed is one that all acquisitions can
actually supply.
"""

import pytest

from odyn.groups import _resolve_windows


# Roughly a real acquisition: 14 Hz, 280 frames, 4 s odor starting at ~5 s
FRAME_RATE = 14.0
ONSET = 70
ODOR = 56
COUNT = 280


def windows(**kwargs):
    """`_resolve_windows` on one nominal acquisition, overridable per test."""
    return _resolve_windows(
        **{
            "frame_rate": FRAME_RATE,
            "onset_frames": [ONSET],
            "odor_frames": [ODOR],
            "frame_counts": [COUNT],
            **kwargs,
        }
    )


# --------------------------------------------------------------------------- #
# Smoothing window
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "smoothing_window_s, expected",
    [(0.5, 7), (0.43, 7), (1.0, 15), (0.0, 1), (0.07, 1)],
)
def test_smoothing_window_is_always_odd(smoothing_window_s, expected):
    """Even windows have no unambiguous centre, so they are rounded up."""
    w = windows(smoothing_window_s=smoothing_window_s)

    assert w.smoothing_frames == expected
    assert w.smoothing_frames % 2 == 1
    assert w.half_frames == expected // 2


def test_photobleach_is_floored_at_half_a_window():
    """Every retained frame must have a full window behind it."""
    w = windows(photobleach_window_s=0.0, smoothing_window_s=1.0)

    assert w.photobleach_frames == w.half_frames == 7


def test_photobleach_is_respected_when_larger():
    w = windows(photobleach_window_s=2.0, smoothing_window_s=0.5)

    assert w.photobleach_frames == 28


# --------------------------------------------------------------------------- #
# Usable range
# --------------------------------------------------------------------------- #


def test_first_and_last_frame_have_full_windows():
    w = windows()

    assert w.first_frame == w.photobleach_frames + w.half_frames - ONSET
    assert w.last_frame == COUNT - 1 - w.half_frames - ONSET


def test_range_is_the_intersection_over_acquisitions():
    """A late-onset acquisition shortens the front, a short one the back."""
    w = _resolve_windows(
        frame_rate=FRAME_RATE,
        onset_frames=[70, 72],
        odor_frames=[ODOR, ODOR],
        frame_counts=[280, 260],
    )

    # Front is set by the *earliest* onset, back by the tightest tail
    assert w.first_frame == w.photobleach_frames + w.half_frames - 70
    assert w.last_frame == 260 - 1 - w.half_frames - 72


def test_impossible_range_raises():
    """Dropping more leading frames than the recording has leaves nothing."""
    with pytest.raises(ValueError, match="shared by every acquisition"):
        _resolve_windows(
            frame_rate=FRAME_RATE,
            onset_frames=[70],
            odor_frames=[ODOR],
            frame_counts=[75],
            photobleach_window_s=5.0,
        )


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #


def test_baseline_never_reaches_past_onset():
    """
    The last baseline centre is half a window before onset, so the window it
    averages ends at relative frame -1.
    """
    w = windows()

    assert w.baseline_stop + w.half_frames == -1
    assert w.baseline_start == w.first_frame


def test_baseline_gap_shifts_the_end_back():
    plain = windows()
    gapped = windows(baseline_gap_s=0.5)

    assert gapped.baseline_stop == plain.baseline_stop - 7
    assert gapped.baseline_start == plain.baseline_start


def test_baseline_frames_and_effective_samples():
    w = windows()

    assert w.baseline_frames == w.baseline_stop - w.baseline_start + 1
    assert w.effective_baseline_samples == pytest.approx(
        w.baseline_frames / w.smoothing_frames
    )


def test_baseline_too_short_raises():
    """Two independent samples is the minimum for an SD to mean anything."""
    with pytest.raises(ValueError, match="under the .* needed"):
        windows(photobleach_window_s=4.5, smoothing_window_s=1.0)


# --------------------------------------------------------------------------- #
# Independent-measurement stack
# --------------------------------------------------------------------------- #


def test_stack_tiles_onset_with_non_overlapping_windows():
    w = windows()
    centres = w.stack_centres

    # Window k covers [k*w, (k+1)*w - 1], so centres step by exactly w
    steps = {b - a for a, b in zip(centres, centres[1:])}
    assert steps == {w.smoothing_frames}

    # The k = 0 window starts exactly at onset
    onset_centre = centres[w.onset_window_index]
    assert onset_centre - w.half_frames == 0


def test_stack_covers_before_and_after_onset():
    """Pre-odor windows are the visual null; they must be there."""
    w = windows()

    assert w.stack_first_k < 0 < w.stack_last_k
    assert w.onset_window_index > 0


def test_stack_centres_stay_inside_the_usable_range():
    w = windows()

    assert all(w.first_frame <= c <= w.last_frame for c in w.stack_centres)

    # And they are maximal: one more window on either side would fall outside
    step = w.smoothing_frames
    assert w.stack_centres[0] - step < w.first_frame
    assert w.stack_centres[-1] + step > w.last_frame


def test_onset_window_index_locates_k_zero():
    w = windows()

    assert w.stack_centres[w.onset_window_index] == w.half_frames
    assert w.onset_window_index == -w.stack_first_k


# --------------------------------------------------------------------------- #
# Odor windows (the 2D map)
# --------------------------------------------------------------------------- #


def test_odor_windows_lie_entirely_inside_the_odor():
    w = windows()
    step = w.smoothing_frames

    for k in range(w.odor_first_k, w.odor_last_k + 1):
        assert k * step >= 0
        assert (k + 1) * step - 1 < ODOR

    # 56 frames of odor hold exactly 8 whole 7-frame windows
    assert (w.odor_first_k, w.odor_last_k) == (0, 7)


def test_odor_windows_start_at_onset():
    w = windows()

    assert w.odor_first_k == 0
    assert w.stack_centres[w.onset_window_index] == w.half_frames


def test_odor_windows_are_a_slice_of_the_stack():
    """The map must be derivable from the stack, or the two can disagree."""
    w = windows()

    assert w.stack_first_k <= w.odor_first_k <= w.odor_last_k <= w.stack_last_k
    assert w.odor_windows == w.odor_last_k - w.odor_first_k + 1


def test_no_gap_when_the_odor_is_a_whole_number_of_windows():
    """56 frames of odor is exactly 8 windows of 7."""
    w = windows()

    assert w.post_odor_first_k == w.odor_last_k + 1


def test_the_window_straddling_odor_offset_belongs_to_neither_side():
    """51 frames of odor is 7 windows plus 2 frames, so window 7 is split."""
    w = windows(odor_frames=[51])

    assert w.odor_last_k == 6
    assert w.post_odor_first_k == 8


def test_shortest_odor_presentation_sets_the_bound():
    """No map window may run past the odor in *any* acquisition."""
    w = _resolve_windows(
        frame_rate=FRAME_RATE,
        onset_frames=[ONSET, ONSET],
        odor_frames=[ODOR, 21],
        frame_counts=[COUNT, COUNT],
    )

    assert w.odor_last_k == 2  # 21 frames hold three whole 7-frame windows


def test_short_odor_still_yields_a_map():
    """Group 133 has a 1 s odor where every other group has 4 s."""
    w = _resolve_windows(
        frame_rate=29.78,
        onset_frames=[149],
        odor_frames=[30],
        frame_counts=[357],
    )

    assert w.odor_first_k == 0
    assert w.odor_last_k == 1  # 30 frames hold two whole 15-frame windows


def test_odor_shorter_than_one_window_raises():
    with pytest.raises(ValueError, match="no whole"):
        windows(odor_frames=[5])


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


def test_no_acquisitions_raises():
    with pytest.raises(ValueError, match="No acquisitions"):
        _resolve_windows(
            frame_rate=FRAME_RATE, onset_frames=[], odor_frames=[], frame_counts=[]
        )


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="equal length"):
        _resolve_windows(
            frame_rate=FRAME_RATE,
            onset_frames=[ONSET],
            odor_frames=[ODOR],
            frame_counts=[COUNT, COUNT],
        )


@pytest.mark.parametrize("frame_rate", [0.0, -14.0])
def test_non_positive_frame_rate_raises(frame_rate):
    with pytest.raises(ValueError, match="frame_rate must be positive"):
        windows(frame_rate=frame_rate)
