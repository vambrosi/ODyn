"""
The spreadsheet calculators, checked against the numbers the workbooks show.
"""

import pytest

from odyn.recipes import block_timing, solvent_volume_ml, vial_recipe


def test_a_pure_odor_matches_the_odor_workbook():
    # `Print v3`, vial 1: eugenol at 0.3 ppm, 200 of 1500 sccm, 5 ml
    recipe = vial_recipe(
        target_ppm=0.3,
        vapor_pressure_mmhg=0.0226,
        vial_sccm=200,
        total_sccm=1500,
        total_volume_ml=5,
    )

    # The workbook says 346.8445 ul; the pipette does 347
    assert recipe.exact_ul == pytest.approx(346.8445, abs=1e-3)
    assert recipe.liquid_ul == 347
    assert recipe.percent_vv == pytest.approx(6.94)
    assert recipe.dilution_per_sccm == pytest.approx(0.925333, abs=1e-5)
    assert recipe.headspace_ppm == pytest.approx(0.3, abs=1e-3)

    assert solvent_volume_ml(5, [recipe.liquid_ul]) == pytest.approx(4.653)


def test_a_mix_is_one_recipe_per_component():
    # `Print v3`, vial 4 (alpha): 1-butanol at 44.2 ppm + (-) alpha pinene at 13.3
    butanol = vial_recipe(
        target_ppm=44.2,
        vapor_pressure_mmhg=7.78,
        vial_sccm=200,
        total_sccm=1500,
        total_volume_ml=5,
    )
    pinene = vial_recipe(
        target_ppm=13.3,
        vapor_pressure_mmhg=3.489,
        vial_sccm=200,
        total_sccm=1500,
        total_volume_ml=5,
    )

    assert butanol.exact_ul == pytest.approx(149.9603, abs=1e-3)
    assert pinene.exact_ul == pytest.approx(100.0575, abs=1e-3)
    assert (butanol.liquid_ul, pinene.liquid_ul) == (150, 100)
    assert butanol.percent_vv == pytest.approx(3.0)
    assert pinene.percent_vv == pytest.approx(2.0)
    assert solvent_volume_ml(5, [butanol.liquid_ul, pinene.liquid_ul]) == pytest.approx(
        4.75
    )


def test_impossible_recipes_are_refused():
    with pytest.raises(ValueError, match="above zero"):
        vial_recipe(
            target_ppm=1,
            vapor_pressure_mmhg=0,
            vial_sccm=200,
            total_sccm=1500,
            total_volume_ml=5,
        )

    with pytest.raises(ValueError, match="exceed"):
        solvent_volume_ml(1, [600, 600])


def test_block_timing_matches_the_session_log():
    # The `mouse` sheet of 20260708_m442_Log_in_vivo.xlsx
    timing = block_timing(
        baseline_s=5,
        odor_s=4,
        post_odor_delay_s=5,
        iti_s=16,
        post_odor_acq_s=11,
        n_odors=16,
        n_trials=112,
        frame_rate_hz=12.86,
    )

    assert timing.trial_s == 30
    assert timing.acquisition_s == 20
    assert timing.frames == 258  # 257.2, rounded up
    assert timing.acquisitions == 224
    assert timing.trials_per_odor == 7
    assert timing.program_min == 56
    assert timing.loop_min == 112
    assert timing.overrun_s == 0  # 20 s of imaging inside a 30 s trial
    assert timing.gap_s == 10


def test_an_acquisition_longer_than_its_trial_is_flagged():
    # imaging wants 11 s after the odor, the program only leaves 5 + 4 = 9 s
    timing = block_timing(
        baseline_s=5,
        odor_s=4,
        post_odor_delay_s=5,
        iti_s=4,
        post_odor_acq_s=11,
        n_odors=16,
        n_trials=112,
        frame_rate_hz=12.86,
    )

    assert timing.trial_s == 18 and timing.acquisition_s == 20
    assert timing.overrun_s == 2 and timing.gap_s == 0

    # A whole number of frames is not rounded up past itself
    assert (
        block_timing(
            baseline_s=5,
            odor_s=4,
            post_odor_delay_s=5,
            iti_s=16,
            post_odor_acq_s=11,
            n_odors=16,
            n_trials=112,
            frame_rate_hz=19.0,
        ).frames
        == 380
    )

    with pytest.raises(ValueError):
        block_timing(
            baseline_s=5,
            odor_s=4,
            post_odor_delay_s=5,
            iti_s=16,
            post_odor_acq_s=11,
            n_odors=0,
            n_trials=112,
            frame_rate_hz=12.86,
        )


def test_camera_frames_cover_the_loop_and_a_minute():
    from odyn.recipes import camera_frames

    # a 101 min loop at 20 Hz: (101 * 60 + 60) * 20
    assert camera_frames(101, 20) == 122400
    assert camera_frames(0, 20) == 1200

    with pytest.raises(ValueError):
        camera_frames(10, 0)
