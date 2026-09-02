#!/usr/bin/env python3
"""
Compare the average z-score of every pair of programs, pixel by pixel.

One figure per odor: the diagonal shows each program's average response, the
lower triangle scatters every pair against each other, and the upper triangle
reports the correlation and the slope of the principal axis. A pair that lies
on the diagonal encodes the odor the same way in both programs.

examples:
    python program_scatter.py --groups 48 49 53
    python program_scatter.py --groups 48 --outcomes all --per-outcome
    python program_scatter.py --groups 48 --trials-per-cell 0 --draws 1

The run is recorded in `method_calls` (one row per group), so the figures can
be traced back to the parameters and the commit that made them.
"""

import argparse
import json
import logging
import os
import sys

from enum import IntFlag
from pathlib import Path

import numpy as np

MAIN_FOLDER = "/home/groups/MossLab/ImagingData"

BLOCKS = ["fine 1", "coarse 1", "fine 2", "coarse 2"]

# Programs are compared as averages, so a cell needs a few trials to mean
# anything. Cells below this are dropped with a warning rather than plotted
MIN_TRIALS = 3


class ScatterFlag(IntFlag):
    """
    call_flag bits for `ProgramScatter.run` (bit 0 reserved by `CallFlag.RAISED`).
    """

    NO_TRIALS = 1 << 1  # nothing matched the filters
    REPEATED_PROGRAM = 1 << 2  # a program type appears twice in the group
    NO_SHARED_ODOR = 1 << 3  # no odor is present in every program
    TOO_FEW_TRIALS = 1 << 4  # asked for more trials per cell than exist
    DROPPED_CELLS = 1 << 5  # some program/odor pairs were too small to use
    DROPPED_ODORS = 1 << 6  # some odors were missing from a program
    NO_REGIONS = 1 << 7  # ran over the whole picture, not the drawn regions


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Point odyn's logger at stdout, with timestamps and no color.

    Same reasoning as the job scripts in the project repos: SLURM keeps the
    escape codes, and odyn's logger does not propagate, so its handler is
    replaced rather than added to.
    """
    from odyn.utils import logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #


def principal_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Slope of the line through the origin closest to the points.

    Least squares would answer a different question: it minimizes the error in
    `y` alone, so noise in `x` pulls the slope towards zero (by how much
    depends on how many trials went into `x`, which differs between programs).
    This minimizes the perpendicular distance instead, so both axes count the
    same and the number can be compared across panels.
    """
    xx, xy, yy = float(x @ x), float(x @ y), float(y @ y)
    angle = 0.5 * np.arctan2(2 * xy, xx - yy)

    # Vertical principal axis: no finite slope, and no sensible answer either
    if abs(np.cos(angle)) < 1e-12:
        return float("inf")

    return float(np.tan(angle))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, or NaN when one of the two does not vary."""
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def jsonable(value: float) -> None | float:
    """
    `None` for anything not finite, the number itself otherwise.

    `json.dumps` writes NaN and Infinity, which are not JSON, and the database
    guards its JSON columns with a `json_valid` CHECK. So a vertical fit or an
    empty correlation would fail the call at the very end, after all the work.
    """
    return None if not np.isfinite(value) else float(value)


# --------------------------------------------------------------------------- #
# The recorded call
# --------------------------------------------------------------------------- #


def build_analysis_class():
    """
    Build the class that carries the recorded call.

    Defined in a function so that the odyn import stays inside `main`, and the
    `--help` of this script does not wait for caiman to load.
    """
    from odyn.utils import record_call, CallRecorder, logger
    from odyn.regions import mask_outside

    class ProgramScatter(CallRecorder):
        """
        A `Group` analysis that records itself in `method_calls`.

        `record_call` only needs a `group_id`, a database and a call stack, so
        an analysis living outside `odyn` can be recorded the same way as the
        methods inside it. The recorded name is `ProgramScatter.run`.
        """

        def __init__(self, group):
            self.group = group
            self.db = group.db
            self.group_id = group.group_id

            self._call_stack = []
            self._method_calls = None
            self._outputs = None

        # ------------------------------------------------------------------ #

        def cells(self, blocks, odors, outcomes, only_approved):
            """
            Acquisitions for every (program, odor) pair that was asked for.

            Returns `({(block, odor): [acq_id, ...]}, trials)`, with the trials
            that survived every filter, so the odor window can be measured on
            exactly the trials that are averaged.
            """
            group = self.group

            programs = group.programs[group.programs["program_type"].isin(blocks)]
            repeated = programs["program_type"].duplicated(keep=False)

            if repeated.any():
                self.fail(
                    ScatterFlag.REPEATED_PROGRAM,
                    f"{group!r} has more than one program of type "
                    f"{sorted(set(programs['program_type'][repeated]))}. Name "
                    "them apart before comparing them.",
                )

            block_of = dict(zip(programs.index, programs["program_type"]))
            usable = self.group._mcors(only_approved).index

            trials = group.trials[
                group.trials["program_id"].isin(programs.index)
                & group.trials["acq_id"].isin(usable)
            ]

            if odors:
                trials = trials[trials["odor_id"].isin(odors)]

            if outcomes:
                trials = trials[trials["outcome"].isin(outcomes)]

            found = {}
            for (program_id, odor_id), rows in trials.groupby(
                ["program_id", "odor_id"]
            ):
                acq_ids = sorted(rows["acq_id"].astype(int).unique())
                found[(block_of[program_id], int(odor_id))] = [int(a) for a in acq_ids]

            return found, trials

        def odor_window(
            self, trials, photobleach_window_s, only_approved, pre_s, post_s
        ):
            """
            One frame window for every program, so the averages are comparable.

            The odor is not exactly the same length on every trial, so the
            shortest one sets the window; widening it per program would mean
            comparing different amounts of response.
            """
            first, last, frame_rate = self.group._z_score_frames(
                photobleach_window_s, only_approved
            )

            odor_s = (
                (trials["odor_end"] - trials["odor_start"]).dt.total_seconds().min()
            )

            onset = -first  # the odor starts here, in the z-scored movie
            start = max(0, onset - round(pre_s * frame_rate))
            stop = min(
                last - first + 1,
                onset + round(odor_s * frame_rate) + round(post_s * frame_rate),
            )

            return slice(start, stop), float(frame_rate), float(odor_s)

        def acquisition_images(
            self, acq_ids, window, photobleach_window_s, only_approved
        ):
            """
            Average z-score over the odor window, one image per acquisition.

            Computed once and kept, because every resampling draw below reuses
            them and the movies are the slow part.
            """
            images = {}

            for acq_id in acq_ids:
                z_score = self.group.z_score_acquisition(
                    acq_id=acq_id,
                    photobleach_window_s=photobleach_window_s,
                    only_approved=only_approved,
                )
                images[acq_id] = z_score[window].mean(axis=0).astype(np.float32)
                del z_score

            return images

        # ------------------------------------------------------------------ #

        @record_call
        def run(
            self,
            *,
            blocks: list[str] = BLOCKS,
            odors: list[int] = [],
            outcomes: list[str] = ["hit"],
            photobleach_window_s: float = 1.0,
            pre_s: float = 0.0,
            post_s: float = 0.0,
            only_approved: bool = True,
            average: str = "mean",
            trials_per_cell: int = -1,
            draws: int = 50,
            seed: int = 0,
            responsive_quantile: float = 0.9,
            use_regions: bool = True,
            save_folder: str = r"./outputs/scatters",
            figures: bool = True,
            image_format: str = "png",
            dpi: int = 150,
            max_points: int = 20000,
        ) -> None:
            """
            Scatter every program against every other, for each odor

            **PARAMETERS**
            - `blocks`: which `program_type`s to compare, in plotting order
            - `odors`: odor ids to plot, empty for every odor in every block
            - `outcomes`: trial outcomes to average, empty for all of them
            - `photobleach_window_s`: seconds dropped from the start
            - `pre_s`, `post_s`: seconds added around the odor
            - `only_approved`: use only approved motion corrections
            - `average`: `"mean"` keeps response size, `"sqrt"` keeps noise at 1
            - `trials_per_cell`: trials drawn per program and odor, `-1` for the
            largest number every cell can supply, `0` to use all of them
            - `draws`: how many times to redraw when subsampling
            - `seed`: for reproducible draws
            - `responsive_quantile`: keep this fraction of pixels out of the
            statistics, ranked by how much the odor moves them, `0` for all
            - `use_regions`: restrict to the regions drawn with `pick_regions`
            - `save_folder`: `"."` is the main_folder
            - `figures`, `image_format`, `dpi`, `max_points`: how to draw

            **ALERT**: `average="mean"` is the default because the diagonal of
            each plot should mean "same response". With `"sqrt"` a program with
            more trials is scaled up, so equal responses land off the diagonal.
            """
            group = self.group

            folder = (
                (self.db.main_folder / save_folder).resolve()
                if save_folder[0] == "."
                else Path(save_folder).resolve()
            )
            folder.mkdir(parents=True, exist_ok=True)

            recordable = folder.is_relative_to(self.db.main_folder)

            if not recordable:
                logger.warning(
                    f"'{folder}' is outside the main folder, so the files are "
                    "saved but not recorded in the database."
                )

            # -- what to average ------------------------------------------- #

            found, trials = self.cells(blocks, odors, outcomes, only_approved)

            if not found:
                self.fail(
                    ScatterFlag.NO_TRIALS,
                    f"{group!r} has no trials matching those filters.",
                )

            small = {
                key: len(ids) for key, ids in found.items() if len(ids) < MIN_TRIALS
            }

            if small:
                self.add_flag(ScatterFlag.DROPPED_CELLS)

            for key in small:
                logger.warning(f"Dropping {key}: only {small[key]} trials.")
                del found[key]

            present = sorted({block for block, _ in found}, key=blocks.index)
            odor_ids = sorted({odor for _, odor in found})

            # A pair only means something when both halves exist, so odors that
            # are missing from a program are dropped rather than half plotted
            complete = [
                odor
                for odor in odor_ids
                if all((block, odor) in found for block in present)
            ]

            missing = set(odor_ids) - set(complete)

            if missing:
                self.add_flag(ScatterFlag.DROPPED_ODORS)

            for odor in sorted(missing):
                logger.warning(f"Dropping odor {odor}: not in every program.")

            if not complete:
                self.fail(
                    ScatterFlag.NO_SHARED_ODOR,
                    "No odor is present in all of the programs asked for.",
                )

            # Dropped odors are out of the counts too, or one of them could set
            # the number of trials every remaining cell is cut down to
            found = {key: ids for key, ids in found.items() if key[1] in complete}
            trials = trials[trials["odor_id"].isin(complete)]

            window, frame_rate, odor_s = self.odor_window(
                trials, photobleach_window_s, only_approved, pre_s, post_s
            )

            counts = {
                f"{block}|{odor}": len(found[(block, odor)]) for block, odor in found
            }
            smallest = min(counts.values())
            drawn = smallest if trials_per_cell < 0 else trials_per_cell

            if drawn and drawn > smallest:
                self.fail(
                    ScatterFlag.TOO_FEW_TRIALS,
                    f"Asked for {drawn} trials per cell but the smallest has "
                    f"{smallest}. Lower 'trials_per_cell' or widen the filters.",
                )

            repeats = draws if drawn else 1

            self.update_parameters_used(
                {
                    "frame_rate": frame_rate,
                    "odor_s": odor_s,
                    "window": [window.start, window.stop],
                    "programs_compared": present,
                    "odors_compared": complete,
                    "trials_per_cell_used": drawn,
                    "draws_used": repeats,
                    "trial_counts": counts,
                }
            )

            logger.info(
                f"{len(present)} programs x {len(complete)} odors, frames "
                f"{window.start}-{window.stop} of the z-scored movie "
                f"({odor_s:.1f} s odor at {frame_rate:.2f} Hz)."
            )
            logger.info(
                f"{drawn or 'all'} trials per cell"
                + (f" over {repeats} draws." if drawn else ".")
            )

            # -- one image per acquisition --------------------------------- #

            every_acq = sorted({a for ids in found.values() for a in ids})
            logger.info(f"Averaging the odor window of {len(every_acq)} acquisitions.")

            images = self.acquisition_images(
                every_acq, window, photobleach_window_s, only_approved
            )
            shape = next(iter(images.values())).shape

            # -- where to look --------------------------------------------- #

            if use_regions:
                mask = group.region_mask(shape)
            else:
                mask = np.ones(shape, dtype=bool)
                self.add_flag(ScatterFlag.NO_REGIONS)
                logger.warning("Regions ignored, so every pixel is included.")

            # -- averages and statistics ----------------------------------- #

            rng = np.random.default_rng(seed)
            scale = np.sqrt if average == "sqrt" else (lambda n: n)

            def cell_average(acq_ids):
                total = sum(images[acq_id] for acq_id in acq_ids)
                return total / scale(len(acq_ids))

            # Full-data averages: what gets saved and drawn
            averages = {key: cell_average(ids) for key, ids in found.items()}

            statistics = {}
            selected = {}

            for odor in complete:
                # Selected on every program at once, so no single panel is
                # favoured and no axis is chosen by the data plotted on it
                reference = np.mean([averages[(b, odor)] for b in present], axis=0)

                if responsive_quantile > 0:
                    inside = np.abs(reference[mask])
                    threshold = float(np.quantile(inside, responsive_quantile))
                    chosen = mask & (np.abs(reference) >= threshold)
                else:
                    chosen = mask

                logger.info(
                    f"Odor {odor}: {chosen.sum()} of {mask.sum()} pixels in the "
                    "regions are used for the statistics."
                )
                selected[odor] = chosen

                for i, first_block in enumerate(present):
                    for second_block in present[i + 1 :]:
                        rows = []

                        for _ in range(repeats):
                            if drawn:
                                pick = {
                                    block: rng.choice(
                                        found[(block, odor)], drawn, replace=False
                                    )
                                    for block in (first_block, second_block)
                                }
                                x = cell_average(pick[first_block])[chosen]
                                y = cell_average(pick[second_block])[chosen]
                            else:
                                x = averages[(first_block, odor)][chosen]
                                y = averages[(second_block, odor)][chosen]

                            rows.append((correlation(x, y), principal_slope(x, y)))

                        values = np.asarray(rows, dtype=float)

                        # nanmean of an all-NaN column warns and returns NaN,
                        # which 'jsonable' then turns into null
                        with np.errstate(invalid="ignore"):
                            summary = [
                                np.nanmean(values[:, 0]),
                                np.nanstd(values[:, 0]),
                                np.nanmean(values[:, 1]),
                                np.nanstd(values[:, 1]),
                            ]

                        statistics[f"{odor}|{first_block}|{second_block}"] = {
                            "odor_id": odor,
                            "x": first_block,
                            "y": second_block,
                            "correlation": jsonable(summary[0]),
                            "correlation_sd": jsonable(summary[1]),
                            "slope": jsonable(summary[2]),
                            "slope_sd": jsonable(summary[3]),
                            "pixels": int(chosen.sum()),
                            "trials_x": len(found[(first_block, odor)]),
                            "trials_y": len(found[(second_block, odor)]),
                        }

            # -- save ------------------------------------------------------ #

            tag = "-".join(outcomes) if outcomes else "all"
            tag = tag.replace(" ", "_")
            stem = f"scatter_group{self.group_id}_{tag}"

            arrays = {
                f"average|{block}|{odor}": mask_outside(averages[(block, odor)], mask)
                for block, odor in found
            }
            arrays["mask"] = mask
            arrays.update({f"selected|{odor}": selected[odor] for odor in complete})

            array_path = folder / f"{stem}.npz"
            np.savez_compressed(array_path, **arrays)
            logger.info(f"Saved the averages in '{array_path}'.")

            stats_path = folder / f"{stem}_stats.json"
            stats_path.write_text(json.dumps(statistics, indent=2))
            logger.info(f"Saved the statistics in '{stats_path}'.")

            written = [array_path, stats_path]

            if figures:
                for odor in complete:
                    figure_path = folder / f"{stem}_odor{odor}.{image_format}"
                    draw_grid(
                        path=figure_path,
                        title=(
                            f"group {self.group_id}, odor {odor}, "
                            f"{tag.replace('-', ' / ')}"
                            + (f", {drawn} trials per program" if drawn else "")
                        ),
                        blocks=present,
                        averages={b: averages[(b, odor)] for b in present},
                        mask=mask,
                        chosen=selected[odor],
                        statistics=statistics,
                        odor=odor,
                        counts=found,
                        max_points=max_points,
                        rng=rng,
                        dpi=dpi,
                    )
                    written.append(figure_path)
                    logger.info(f"Saved '{figure_path}'.")

            if recordable:
                for path in written:
                    self.add_output_file(path)

            self.set_output(
                {
                    "programs": present,
                    "odors": complete,
                    "trial_counts": counts,
                    "trials_per_cell": drawn,
                    "draws": repeats,
                    "statistics": statistics,
                    "files": [p.name for p in written],
                }
            )

    return ProgramScatter


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #


def draw_grid(
    *,
    path,
    title,
    blocks,
    averages,
    mask,
    chosen,
    statistics,
    odor,
    counts,
    max_points,
    rng,
    dpi,
):
    """
    One panel per pair: images on the diagonal, scatters below, numbers above.

    The scatters share one pair of limits, so the identity line is at 45
    degrees in every panel and the panels can be compared by eye.
    """
    import matplotlib.pyplot as plt

    from odyn.regions import mask_outside

    size = len(blocks)
    figure, axes = plt.subplots(size, size, figsize=(3.1 * size, 3.1 * size))
    axes = np.atleast_2d(axes)

    # Odor responses are mostly positive, so symmetric limits would waste half
    # of every panel. Taken from the data instead, but always including zero so
    # the identity line still crosses the origin
    inside = np.concatenate([averages[b][chosen] for b in blocks])
    low, high = np.quantile(inside, [0.001, 0.999])
    pad = 0.05 * (high - low)
    span = (min(0.0, low - pad), max(0.0, high + pad) or 1.0)

    picture_limit = (
        float(
            np.quantile(
                np.abs(np.concatenate([averages[b][mask] for b in blocks])), 0.99
            )
        )
        or 1.0
    )

    def number(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    for row, y_block in enumerate(blocks):
        for column, x_block in enumerate(blocks):
            axis = axes[row, column]

            if row == column:
                axis.imshow(
                    mask_outside(averages[x_block], mask),
                    cmap="RdBu_r",
                    vmin=-picture_limit,
                    vmax=picture_limit,
                )
                axis.set_title(
                    f"{x_block}  (n = {len(counts[(x_block, odor)])})", fontsize=10
                )
                axis.set_xticks([])
                axis.set_yticks([])
                continue

            # Statistics are stored once per pair, with x the earlier program.
            # The lower triangle is drawn the same way round, so both triangles
            # can read the same entry and nothing has to be inverted
            low, high = (x_block, y_block) if column < row else (y_block, x_block)
            entry = statistics.get(f"{odor}|{low}|{high}")

            if row < column:
                # Upper triangle: the numbers, so the eye can scan them
                axis.axis("off")

                if entry:
                    axis.text(
                        0.5,
                        0.5,
                        f"{high} vs {low}\n\n"
                        f"r = {number(entry['correlation'])}"
                        f" ± {number(entry['correlation_sd'])}\n"
                        f"slope = {number(entry['slope'])}"
                        f" ± {number(entry['slope_sd'])}\n"
                        f"{entry['pixels']} pixels",
                        ha="center",
                        va="center",
                        fontsize=10,
                        transform=axis.transAxes,
                    )
                continue

            x = averages[x_block][chosen]
            y = averages[y_block][chosen]

            if x.size > max_points:
                keep = rng.choice(x.size, max_points, replace=False)
                x, y = x[keep], y[keep]

            axis.plot(span, span, color="0.6", linewidth=1, zorder=1)
            axis.scatter(x, y, s=3, alpha=0.15, linewidths=0, color="#1f77b4", zorder=2)

            if entry and entry["slope"] is not None:
                axis.plot(
                    span,
                    [entry["slope"] * span[0], entry["slope"] * span[1]],
                    color="#d62728",
                    linewidth=1.2,
                    zorder=3,
                )

            axis.set_xlim(span)
            axis.set_ylim(span)
            axis.set_aspect("equal")

            if row == size - 1:
                axis.set_xlabel(x_block)
            if column == 0:
                axis.set_ylabel(y_block)

    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--main-folder", default=MAIN_FOLDER, help="where the DB lives")
    parser.add_argument("--project", help="pick ODyn project DB (optional)")
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        required=True,
        help="run these groups in sequence",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=BLOCKS,
        help="program types to compare",
    )
    parser.add_argument(
        "--odors",
        type=int,
        nargs="*",
        default=[],
        help="odor ids (default: every odor present in all the programs)",
    )
    parser.add_argument(
        "--outcomes",
        nargs="*",
        default=["hit"],
        help="trial outcomes to average, or 'all' to pool every outcome",
    )
    parser.add_argument(
        "--per-outcome",
        action="store_true",
        help="one run per outcome instead of pooling them",
    )
    parser.add_argument("--photobleach-s", type=float, default=1.0)
    parser.add_argument("--pre-s", type=float, default=0.0)
    parser.add_argument("--post-s", type=float, default=0.0)
    parser.add_argument(
        "--all-mcors",
        action="store_true",
        help="pass to use all mcors (instead of only approved ones)",
    )
    parser.add_argument(
        "--average",
        choices=("mean", "sqrt"),
        default="mean",
        help="'mean' keeps response size, 'sqrt' keeps the noise level at 1",
    )
    parser.add_argument(
        "--trials-per-cell",
        type=int,
        default=0,
        help="-1 matches the smallest cell, 0 uses every trial",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=50,
        help="resampling repeats",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--responsive-quantile",
        type=float,
        default=0.9,
        help="keep the top pixels by response, 0 to keep the whole region",
    )
    parser.add_argument(
        "--no-regions",
        action="store_true",
        help="ignore the drawn regions",
    )
    parser.add_argument("--save-folder", default=r"./outputs/scatters")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="save arrays only",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-points", type=int, default=20000)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Headless, and set before pyplot is imported anywhere
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Delay import so --help doesn't have a long pause
    logger = setup_logging()

    from odyn import Database

    ProgramScatter = build_analysis_class()

    db = Database(args.main_folder, project=args.project)

    outcomes = [] if args.outcomes == ["all"] else args.outcomes
    rounds = [[outcome] for outcome in outcomes] if args.per_outcome else [outcomes]

    for group_id in args.groups:
        for chosen in rounds:
            logger.info(f"=== group {group_id}, outcomes {chosen or 'all'} {'=' * 20}")
            analysis = ProgramScatter(db.groups[group_id])

            analysis.run(
                blocks=args.blocks,
                odors=args.odors,
                outcomes=chosen,
                photobleach_window_s=args.photobleach_s,
                pre_s=args.pre_s,
                post_s=args.post_s,
                only_approved=not args.all_mcors,
                average=args.average,
                trials_per_cell=args.trials_per_cell,
                draws=args.draws,
                seed=args.seed,
                responsive_quantile=args.responsive_quantile,
                use_regions=not args.no_regions,
                save_folder=args.save_folder,
                figures=not args.no_figures,
                image_format=args.format,
                dpi=args.dpi,
                max_points=args.max_points,
            )


if __name__ == "__main__":
    main()
