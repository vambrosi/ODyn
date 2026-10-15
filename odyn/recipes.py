"""
The two calculators the lab used to keep in spreadsheets.

Both are plain functions of plain numbers: no database, no GUI. The GUI shows
their results as the user types and stores what it stores; nothing here is
read back from the database.

- `vial_recipe` turns a target odor concentration into how much odorant to
  pipette into a vial of mineral oil, given the odor's vapor pressure and the
  flows the olfactometer will use.
- `block_timing` turns the layout of a trial into what ScanImage has to be
  told: frames per acquisition and the number of acquisitions.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

# Ideal-gas volume of a mole at 25 C, in liters
MOLAR_VOLUME_L = 24.45

# Density and molar mass of air, so a headspace fraction converts to ppm
AIR_DENSITY_G_L = 1.292
AIR_MOLAR_MASS = 28.97

# Atmospheric pressure the vapor pressures are taken against, in mmHg
ATMOSPHERE_MMHG = 759.75


@dataclass(frozen=True)
class VialRecipe:
    """What to pipette for one odorant, and what that makes."""

    liquid_ul: float  # volume of odorant to pipette, whole microliters
    percent_vv: float  # the dilution, % v/v, that volume makes
    dilution_per_sccm: float  # % v/v per unit of flow ratio, the lab's "% v/v / sccm"
    headspace_ppm: float  # the concentration the pipetted volume reaches
    exact_ul: float  # the unrounded volume, for the display


def vial_recipe(
    *,
    target_ppm: float,
    vapor_pressure_mmhg: float,
    vial_sccm: float,
    total_sccm: float,
    total_volume_ml: float,
) -> VialRecipe:
    """
    How much of an odorant to pipette to reach `target_ppm` at the nose.

    **PARAMETERS**
    - `target_ppm` is the wanted concentration at the animal
    - `vapor_pressure_mmhg` is the odorant's, at 25 C
    - `vial_sccm` is the flow drawn through the vial, `total_sccm` the flow it
    is diluted into at the nose
    - `total_volume_ml` is the liquid the vial holds once made up

    **EXAMPLE**
    ```python
    vial_recipe(
        target_ppm=0.3,
        vapor_pressure_mmhg=0.0226,   # eugenol
        vial_sccm=200,
        total_sccm=1500,
        total_volume_ml=5,
    ).liquid_ul
    ```

    The saturated headspace over the pure liquid holds a fraction
    `VP / (atm + VP)` of odorant; diluting the liquid `p` % v/v in an odorless
    solvent scales that (Raoult's law), and the olfactometer dilutes the
    headspace again by `vial_sccm / total_sccm`. Solving that for `p` and then
    for the volume that gives `p` in `total_volume_ml` is what the lab's
    workbook does.
    """
    if (
        min(target_ppm, vapor_pressure_mmhg, vial_sccm, total_sccm, total_volume_ml)
        <= 0
    ):
        raise ValueError("Every input has to be above zero.")

    # ppm at the nose -> % v/v per unit of flow ratio
    wanted_per_sccm = (
        target_ppm
        / 1e6
        / MOLAR_VOLUME_L
        / AIR_DENSITY_G_L
        / vapor_pressure_mmhg
        * AIR_MOLAR_MASS
        * (ATMOSPHERE_MMHG + vapor_pressure_mmhg)
        * 100
    )

    # ...times the flow ratio is the % v/v of the liquid; times the volume in
    # ul (ml * 1000) over 100 is the odorant volume, pipetted in whole ul
    exact_ul = 10 * wanted_per_sccm * total_volume_ml * total_sccm / vial_sccm
    liquid_ul = float(max(1, round(exact_ul)))
    percent_vv = 100 * liquid_ul / 1000 / total_volume_ml
    dilution_per_sccm = liquid_ul / (10 * total_volume_ml * total_sccm / vial_sccm)

    # ...and back to ppm, as a check the workbook also shows
    headspace_ppm = (
        1e6
        * MOLAR_VOLUME_L
        * AIR_DENSITY_G_L
        * dilution_per_sccm
        * vapor_pressure_mmhg
        / AIR_MOLAR_MASS
        / (ATMOSPHERE_MMHG + vapor_pressure_mmhg)
        / 100
    )

    return VialRecipe(liquid_ul, percent_vv, dilution_per_sccm, headspace_ppm, exact_ul)


def solvent_volume_ml(
    total_volume_ml: float, liquid_ul: list[float] | tuple[float, ...]
) -> float:
    """The mineral oil that makes the vial up to `total_volume_ml`."""
    oil = total_volume_ml - sum(liquid_ul) / 1000

    if oil < 0:
        raise ValueError(
            f"The odorants alone ({sum(liquid_ul) / 1000:g} ml) exceed the vial's "
            f"{total_volume_ml:g} ml."
        )

    return oil


@dataclass(frozen=True)
class BlockTiming:
    """What a trial layout means for the recording, in the units ScanImage wants."""

    acquisition_s: float  # how long each acquisition images
    acquisitions: int  # acquisitions in the loop, all programs together
    frames: int  # frames per acquisition to set in ScanImage
    trial_s: float  # one trial, baseline to the end of the interval
    trials_per_odor: float
    program_min: float  # one program
    loop_min: float  # all programs, without the pause between them
    overrun_s: float  # how far an acquisition outlasts its trial (0 when it fits)
    gap_s: float  # what is left of the trial after the acquisition (0 when it does not fit)


def block_timing(
    *,
    baseline_s: float,
    odor_s: float,
    post_odor_delay_s: float,
    iti_s: float,
    post_odor_acq_s: float,
    n_odors: int,
    n_trials: int,
    frame_rate_hz: float,
    n_programs: int = 2,
) -> BlockTiming:
    """
    What to type into ScanImage for a trial layout.

    **PARAMETERS**
    - `baseline_s`, `odor_s`, `post_odor_delay_s`, `iti_s` lay out one
    olfactometer trial: imaging starts at the baseline, the odor comes on, the
    trial ends after the delay, and the interval separates it from the next
    - `post_odor_acq_s` is how long imaging goes on after the odor goes off,
    which sets the acquisition length
    - `n_odors` and `n_trials` describe one program; `n_programs` says how
    many run in the loop (two for a before/after manipulation)
    - `frame_rate_hz` is ScanImage's

    **EXAMPLE**
    ```python
    block_timing(
        baseline_s=5, odor_s=4, post_odor_delay_s=5, iti_s=16,
        post_odor_acq_s=11, n_odors=16, n_trials=112, frame_rate_hz=12.86,
    )
    # BlockTiming(trial_s=30, acquisition_s=20, frames=258, acquisitions=224, ...)
    ```

    `frames` is rounded up, so the acquisition covers the whole window.

    The olfactometer only knows the first four durations; `post_odor_acq_s`
    is what the imaging wants. When it runs past the trial (`overrun_s > 0`,
    i.e. it is longer than the post-odor delay plus the interval), ScanImage
    is still acquiring when the next trial starts and misses its start: the
    olfactometer program has to change, not the frame count.
    """
    if n_odors <= 0 or n_trials <= 0 or frame_rate_hz <= 0 or n_programs <= 0:
        raise ValueError("Odors, trials, programs and the frame rate must be positive.")

    trial_s = baseline_s + odor_s + post_odor_delay_s + iti_s
    acquisition_s = baseline_s + odor_s + post_odor_acq_s

    return BlockTiming(
        trial_s=trial_s,
        acquisition_s=acquisition_s,
        frames=math.ceil(acquisition_s * frame_rate_hz - 1e-9),
        acquisitions=n_programs * n_trials,
        trials_per_odor=n_trials / n_odors,
        program_min=trial_s * n_trials / 60,
        loop_min=trial_s * n_trials * n_programs / 60,
        overrun_s=max(0.0, acquisition_s - trial_s),
        gap_s=max(0.0, trial_s - acquisition_s),
    )


# The camera keeps rolling a little past its run, so the last acquisition
# and the pause after it are covered (the quick maths lets it be changed)
CAMERA_MARGIN_S = 15.0


def camera_frames(
    run_min: float, camera_rate_hz: float, margin_s: float = CAMERA_MARGIN_S
) -> int:
    """
    How many frames to tell Micromanager to record for one run of the
    camera (one program, or one grab) lasting `run_min`: the run plus
    `margin_s`, at the camera's frame rate.
    """
    if run_min < 0 or camera_rate_hz <= 0:
        raise ValueError(
            "The loop cannot be negative and the camera rate must be positive."
        )

    return math.ceil((run_min * 60 + margin_s) * camera_rate_hz - 1e-9)
