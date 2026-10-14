"""
Comparing the runs of a sweep of test motion corrections
(`python -m odyn.cluster ... mcor-sweep`), once its jobs have finished.

Every run has a folder in the experiment's test folder
(`cache/mcor-tests/<experiment>_g<group>/<run>/`, see `odyn.mcor_report`):
- `collect` puts their settings and numbers side by side, a row per run, in
`sweep.csv` there: the whole-test numbers of `metrics.json`, and the median
over the acquisitions of each column of `summary.csv`;
- `tile` puts their videos side by side in one video, in a grid, each with
its run's name in the top right corner. The runs share the baseline's black
and white, so the tiles compare. "raw" is the baseline's raw video.

**COMMAND LINE**
```
python -m odyn.mcor_sweep <test folder> collect
python -m odyn.mcor_sweep <test folder> tile --runs raw rigid s128_o64_d6
python -m odyn.mcor_sweep <test folder> tile --where stride=128 overlap=64
python -m odyn.mcor_sweep <test folder> tile --all --scale 0.5
```
`--where` takes the runs with these settings (any of `stride`, `overlap`,
`deviation`, `gsig`, in µm), ordered by the others, after raw and rigid: one
setting left free shows what it changes.

**PYTHON**
```python
from odyn.mcor_sweep import collect, tile

table = collect(folder)
tile(folder, ["raw", "rigid", "s128_o64_d6"], columns=3)
```
"""

from __future__ import annotations

import argparse
import json
import math
import sys

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .cluster import BASELINE_RUN
from .utils import logger

RAW = "raw"
TABLE_FILE = "sweep.csv"

# What `--where` takes, and the column it reads
SETTINGS = {
    "stride": "stride_um",
    "overlap": "overlap_um",
    "deviation": "max_deviation_um",
    "gsig": "gsig_filt_um",
}

# The whole-test numbers of metrics.json kept in the table
METRICS = (
    "crispness_raw",
    "crispness_test",
    "frames_at_limit",
    "flow_worst_median_px",
    "patch_jitter_median_px",
    "patch_jitter_p90_px",
)


def collect(folder: str | Path) -> pd.DataFrame:
    """
    A row per run in `folder` (an experiment's test folder): its settings,
    the numbers of its `metrics.json` and the median over the acquisitions of
    its `summary.csv`, the baseline first and then by stride, overlap and
    deviation. Written to `sweep.csv` in `folder` too.
    """
    folder = Path(folder)
    rows = [run_numbers(path.parent) for path in sorted(folder.glob("*/metrics.json"))]

    if not rows:
        raise FileNotFoundError(f"No runs (folders with metrics.json) in '{folder}'.")

    table = pd.DataFrame(rows)
    table["baseline"] = table["run"] == BASELINE_RUN
    table = (
        table.sort_values(
            [
                "baseline",
                "stride_um",
                "overlap_um",
                "max_deviation_um",
                "gsig_filt_um",
                "shifts_opencv",
            ],
            ascending=[False, True, True, True, True, True],
            na_position="first",
        )
        .drop(columns="baseline")
        .reset_index(drop=True)
    )
    table.to_csv(folder / TABLE_FILE, index=False)
    logger.info(f"Wrote {len(table)} runs to {folder / TABLE_FILE}")

    return table


def run_numbers(folder: str | Path) -> dict:
    """
    One run's row of `collect`, from its folder: its settings, the numbers of
    its `metrics.json` and the median over the acquisitions of its
    `summary.csv`.
    """
    folder = Path(folder)
    metrics = json.loads((folder / "metrics.json").read_text())
    settings = metrics.get("settings") or {}
    row = {
        "run": settings.get("run") or folder.name,
        "pw_rigid": bool(settings.get("pw_rigid", False)),
        "stride_um": _first(settings.get("strides_um")),
        "overlap_um": _first(settings.get("overlap_um")),
        "max_deviation_um": settings.get("max_deviation_um"),
        "gsig_filt_um": settings.get("gsig_filt_um"),
        "shifts_opencv": bool(settings.get("shifts_opencv", False)),
        "max_shift_um": _first(settings.get("max_shift_um")),
    }
    row |= {key: metrics.get(key) for key in METRICS}

    summary = pd.read_csv(folder / "summary.csv")
    numbers = summary.drop(
        columns=["position", "acq_id", "file"], errors="ignore"
    ).select_dtypes("number")
    row["acquisitions"] = len(summary)
    row |= {
        f"{name}_median": round(float(numbers[name].median()), 4) for name in numbers
    }

    return row


def run_video(folder: str | Path) -> None | Path:
    """
    The video to judge a run by, in its folder: raw | test when there is one
    (a test or a sweep's baseline), else the test alone. `None` without one.
    """
    videos = [
        path
        for path in sorted(Path(folder).glob("*_test.*"))
        if path.suffix in (".avi", ".mp4")
    ]
    both = [path for path in videos if path.stem.endswith("_raw_test")]

    return (both or videos or [None])[0]


def select(table: pd.DataFrame, **where: float) -> list[str]:
    """
    The runs of `table` with the settings in `where` (`stride`, `overlap`,
    `deviation`), ordered by the other settings, after raw and the baseline.
    """
    unknown = sorted(set(where) - set(SETTINGS))

    if unknown:
        raise ValueError(
            f"Unknown settings {', '.join(unknown)}; "
            f"the known ones are {', '.join(SETTINGS)}."
        )

    chosen = table[table["pw_rigid"]]

    for key, value in where.items():
        chosen = chosen[np.isclose(chosen[SETTINGS[key]], value)]

    if chosen.empty:
        raise ValueError(f"No run has {_describe(where)}.")

    free = [column for key, column in SETTINGS.items() if key not in where]
    chosen = chosen.sort_values([*free, "shifts_opencv"], na_position="first")

    return [RAW, BASELINE_RUN, *chosen["run"]]


def tile(
    folder: str | Path,
    runs: Sequence[str],
    *,
    columns: None | int = None,
    scale: float = 1.0,
    out: None | str | Path = None,
    codec: str = "MJPG",
) -> Path:
    """
    Put the videos of `runs` (in `folder`, an experiment's test folder) side
    by side in one video, `columns` wide (about square by default), each
    scaled by `scale`. Empty tiles are black. Returns the video, written to
    `out` (`tiles_<runs>.avi` in `folder` by default).
    """
    import cv2

    folder = Path(folder)
    paths = [_video(folder, run) for run in runs]
    columns = columns or math.ceil(math.sqrt(len(paths)))
    rows = math.ceil(len(paths) / columns)
    out = Path(out) if out else folder / f"tiles_{'_'.join(runs)}.avi"

    if len(out.name) > 120:
        out = folder / f"tiles_{len(runs)}_runs.avi"

    captures = [cv2.VideoCapture(str(path)) for path in paths]
    frame_rate = captures[0].get(cv2.CAP_PROP_FPS) or 30
    counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in captures]

    if len(set(counts)) > 1:
        logger.warning(
            f"The videos have {min(counts)} to {max(counts)} frames; "
            f"the tiles stop at {min(counts)}."
        )

    sizes = {
        (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        for c in captures
    }
    width, height = max(sizes)  # the raw | test video is the widest
    tile_w, tile_h = round(width * scale) // 2 * 2, round(height * scale) // 2 * 2
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*codec),
        frame_rate,
        (columns * tile_w, rows * tile_h),
    )
    written = 0

    try:
        while True:
            canvas = np.zeros((rows * tile_h, columns * tile_w, 3), np.uint8)

            for i, capture in enumerate(captures):
                ok, frame = capture.read()

                if not ok:
                    break

                frame = _fit(cv2, frame, tile_w, tile_h)
                y, x = (i // columns) * tile_h, (i % columns) * tile_w
                canvas[y : y + tile_h, x : x + tile_w] = frame
            else:
                writer.write(canvas)
                written += 1
                continue

            break
    finally:
        writer.release()

        for capture in captures:
            capture.release()

    logger.info(f"Wrote {written} frames of {len(paths)} runs to {out}")

    return out


def _video(folder: Path, run: str) -> Path:
    """The test video of `run`, or the baseline's raw one for "raw"."""
    kind, run_folder = ("raw", BASELINE_RUN) if run == RAW else ("test", run)
    found = [
        path
        for path in sorted((folder / run_folder).glob(f"*_{kind}.*"))
        if path.suffix in (".avi", ".mp4") and not path.stem.endswith("_raw_test")
    ]

    if not found:
        raise FileNotFoundError(f"No {kind} video in '{folder / run_folder}'.")

    return found[0]


def _fit(cv2, frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """`frame` scaled to fit `width` × `height`, top left, on black."""
    ratio = min(width / frame.shape[1], height / frame.shape[0])
    size = (round(frame.shape[1] * ratio), round(frame.shape[0] * ratio))
    scaled = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    fitted = np.zeros((height, width, 3), np.uint8)
    fitted[: size[1], : size[0]] = scaled

    return fitted


def _first(values) -> None | float:
    if values is None:
        return None

    return float(values[0]) if isinstance(values, (list, tuple)) else float(values)


def _describe(where: dict[str, float]) -> str:
    return ", ".join(f"{key} {value:g} µm" for key, value in where.items())


def main(argv: None | list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odyn.mcor_sweep",
        description="Compare the runs of a sweep of test motion corrections.",
    )
    parser.add_argument("folder", type=Path, help="the experiment's test folder")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("collect", help="every run's numbers in sweep.csv")

    tiles = commands.add_parser("tile", help="runs' videos side by side")
    chosen = tiles.add_mutually_exclusive_group(required=True)
    chosen.add_argument("--runs", nargs="+", help="by name, raw for the raw video")
    chosen.add_argument(
        "--where", nargs="+", metavar="SETTING=UM", help="stride, overlap, deviation"
    )
    chosen.add_argument("--all", action="store_true", help="raw and every run")
    tiles.add_argument("--columns", type=int)
    tiles.add_argument("--scale", type=float, default=1.0)
    tiles.add_argument("--out", type=Path)

    args = parser.parse_args(argv)

    if args.command == "collect":
        table = collect(args.folder)
        print(table.to_string(index=False, max_colwidth=24))
        return 0

    runs = args.runs

    if runs is None:
        table = collect(args.folder)

        if args.all:
            runs = [RAW, *table["run"]]
        else:
            where = {}

            for item in args.where:
                key, equals, value = item.partition("=")

                if not equals:
                    parser.error(f"--where takes SETTING=UM, got {item!r}.")

                where[key] = float(value)

            runs = select(table, **where)

    path = tile(args.folder, runs, columns=args.columns, scale=args.scale, out=args.out)
    print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
