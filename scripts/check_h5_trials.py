#!/usr/bin/env python3
"""
Report which experiments have an incomplete trial at either end of their H5.

USAGE
    python scripts/check_h5_trials.py /path/to/main_folder
    python scripts/check_h5_trials.py /path/to/main_folder --project pa_k99
    python scripts/check_h5_trials.py /path/to/main_folder --csv h5_trials.csv

WHAT IT IS FOR
    A recording that starts after an imaging window has already opened has no
    rise for that window, so `_align_h5_events` drops the odor pulse left in
    front of the first trial start. Whether that window had an acquisition
    behind it is not something the H5 can answer, and it decides how to read
    the result:

        n_tiff == n_trial     the acquisition was never saved, and the H5 and
                              the TIFFs still line up one for one
        n_tiff == n_trial + 1 the acquisition exists with no trigger of its
                              own, so it gets no odor timing and everything
                              after it has to still be paired correctly

    `offset_ms` is the check on that pairing: acquisition start minus H5 trial
    start, index by index, after the usual shift onto the first trial. Near
    zero throughout means the two agree; a column that grows or sits at one
    loop interval means the H5 is anchored on the wrong window.

WHAT IT LOOKS AT
    Every experiment folder in the database, plus the folders of `add_experiment`
    calls that raised on the trial and odor counts -- those rolled back, so they
    have no experiment row and can only be found through `method_calls`.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd

from odyn import Database
from odyn.database import ExpFlag, _align_h5_events, _check_h5_alignment, _h5_edges
from odyn.utils import CallFlag

# The message `_check_h5_alignment` raises with, and the one the assert it
# replaced used, so calls recorded before that change are found too.
COUNT_MISMATCH = "The following do not match"


def candidates(db: Database) -> dict[str, str]:
    """`rel_path` -> where it was found, for every folder worth opening."""
    folders: dict[str, str] = {}

    for (raw_path,) in db.con.execute("SELECT DISTINCT raw_path FROM acquisitions;"):
        folders[Path(raw_path).parent.parent.as_posix()] = "added"

    rows = db.con.execute(
        """
        SELECT parameters_used FROM method_calls
            WHERE method_name LIKE '%.add_experiment'
              AND ( (call_flag & ? AND call_log LIKE ?)
                 OR (call_flag & ?) );
        """,
        [
            int(CallFlag.RAISED),
            f"%{COUNT_MISMATCH}%",
            int(ExpFlag.H5_DROPPED_FIRST | ExpFlag.H5_DROPPED_LAST),
        ],
    ).fetchall()

    for (parameters_used,) in rows:
        rel_path = json.loads(parameters_used).get("rel_path")

        # A raised add rolled back, so it has no rows of its own to find later
        if rel_path and rel_path not in folders:
            folders[rel_path] = "raised"

    return folders


def h5_report(exp_path: Path) -> dict:
    """Everything one H5 can say on its own, without reading any TIFF."""
    h5_paths = sorted(exp_path.glob("[!.]?*.h5"))
    report = {"n_h5": len(h5_paths)}

    if not h5_paths:
        return report | {"check": "no h5 file"}

    if len(h5_paths) > 1:
        return report | {"check": "more than one h5 file"}

    samplerate, trials, odor_starts, odor_ends, high_start, high_end = _h5_edges(
        h5_paths[0]
    )

    report |= {
        "n_trial_raw": len(trials),
        "n_odor_start_raw": len(odor_starts),
        "n_odor_end_raw": len(odor_ends),
        "high_at_start": high_start,
        "high_at_end": high_end,
    }

    if len(trials) == 0:
        return report | {"check": "no trial triggers"}

    trials, odor_starts, odor_ends, dropped_first, dropped_last = _align_h5_events(
        trials, odor_starts, odor_ends
    )

    report |= {
        "n_trial": len(trials),
        "dropped_first": dropped_first,
        "dropped_last": dropped_last,
        "check": "ok",
        "_times_s": (trials - trials[0]) / samplerate,
    }

    try:
        _check_h5_alignment(trials, odor_starts, odor_ends, samplerate)

    except RuntimeError as error:
        report["check"] = str(error).splitlines()[0]

    return report


def tiff_offsets(db: Database, raw_paths: list[Path], times_s: np.ndarray) -> dict:
    """
    Acquisition start minus H5 trial start, index by index, in milliseconds.

    Reads the TIFF metadata, which is slow, so it is only worth doing for a
    folder the H5 has already flagged as odd.
    """
    starts = []
    exp_start = None

    for raw_path in raw_paths:
        parsed = db._get_raw_metadata(raw_path)

        if parsed is None:
            continue

        experiment, acquisition = parsed
        exp_start = experiment["exp_start"]
        starts.append(acquisition["acq_start"])

    if exp_start is None or not starts:
        return {}

    starts.sort()
    paired = min(len(starts), len(times_s))

    offsets = np.array(
        [
            (starts[index] - exp_start).total_seconds() * 1000 - times_s[index] * 1000
            for index in range(paired)
        ]
    )

    return {
        "median_offset_ms": round(float(np.median(offsets)), 1),
        "max_offset_ms": round(float(np.abs(offsets).max()), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("--project", default=None)
    parser.add_argument("--csv", default=None, type=Path)
    parser.add_argument(
        "--limit", default=0, type=int, help="stop after this many folders"
    )
    args = parser.parse_args()

    db = Database(args.main_folder, project=args.project)
    folders = candidates(db)

    if args.limit:
        folders = dict(list(folders.items())[: args.limit])

    print(f"Looking at {len(folders)} experiment folders...\n")
    rows = []

    for rel_path, source in folders.items():
        exp_path = db.main_folder / rel_path
        row = {"rel_path": rel_path, "source": source}

        if not exp_path.is_dir():
            rows.append(row | {"check": "folder is gone", "clean": False})
            continue

        raw_paths = sorted(exp_path.glob("raw/[!.]?*.tif"))
        row |= {"n_tiff": len(raw_paths)}
        row |= h5_report(exp_path)

        times_s = row.pop("_times_s", None)

        if "n_trial" in row:
            row["tiff_minus_trials"] = row["n_tiff"] - row["n_trial"]

        row["clean"] = row.get("check") == "ok" and not (
            row.get("dropped_first")
            or row.get("dropped_last")
            or row.get("high_at_start")
            or row.get("high_at_end")
        )

        # Only the odd ones are worth the TIFF reads
        if not row["clean"] and times_s is not None:
            row |= tiff_offsets(db, raw_paths, times_s)

        rows.append(row)

    table = pd.DataFrame(rows).set_index("rel_path")

    print(table.to_string(), "\n")
    print(f"{len(table)} folders, {int((~table['clean']).sum())} of them not clean.\n")

    if "tiff_minus_trials" in table:
        print("TIFFs minus H5 trials, after dropping incomplete ones:")
        print(table["tiff_minus_trials"].value_counts().to_string(), "\n")

    if args.csv:
        table.to_csv(args.csv)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
