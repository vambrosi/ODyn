"""
Add experiments recorded before the app to a project, each as the person who
recorded it, so their motion correction is checked by them (see
`odyn.cluster.mcor-backfill`).

    python scripts/add_old_experiments.py <main_folder> --project main-sync \\
        experiments.csv [--dry-run]

`experiments.csv` has a line per experiment: `exp_path` (`YYYYMMDD/<mouse>/eN`,
relative to the main folder), `user` (who recorded it) and `objective` (10,
20, ...). Any other column is left alone. A line with no objective is skipped
and said, since the scale of the images depends on it. Two columns may be
added:
- `exp_type`, `loop` (when left out) or `grab`: what ScanImage was set to;
- `condition`, written on every acquisition of the experiment (the
acquisition annotation `condition`, as Submit writes it). Left out, a grab
takes its file name without the number (`Baseline_00001.tif` is `Baseline`)
and a loop none. A condition the database does not know yet is added to the
options of the annotation.

For each experiment: the mouse is registered when it is new (its number only;
the rest is for its people to fill in, in the app), the session (mouse and
day) is added when it is not there yet, then the experiment (`add_experiment`:
its TIFFs and olfactometer files) and its condition, all recorded as `user`.
An experiment already in the database is skipped, so the script can run again
after a fix. The sync file is left to the cluster tick, which processes the
sync file of every experiment in the database.

Reads every TIFF header, so run it on the cluster (`srun`), near the files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

from pathlib import Path

from odyn import Database
from odyn.database import EXP_TYPES

EXP_PATH = re.compile(r"(\d{8})/([A-Za-z]*\d+)/e(\d+)")

# What a TIFF name holds before its number (`Baseline_00001.tif`)
RECORDING_NAME = re.compile(r"(.+)_\d+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("csv", type=Path, help="exp_path, user, objective")
    parser.add_argument("--project", required=True)
    parser.add_argument("--dry-run", action="store_true", help="only say what")
    args = parser.parse_args()

    with open(args.csv, newline="") as handle:
        rows = list(csv.DictReader(handle))

    databases: dict[str, Database] = {}
    added, skipped, failed = [], [], []

    for row in rows:
        exp_path = row["exp_path"].strip().strip("/")
        user = row["user"].strip()
        objective = (row.get("objective") or "").strip()
        exp_type = (row.get("exp_type") or "").strip().lower() or "loop"
        condition = (row.get("condition") or "").strip()
        match = EXP_PATH.fullmatch(exp_path)

        if match is None:
            failed.append(f"{exp_path}: not YYYYMMDD/<mouse>/eN")
            continue

        if exp_type not in EXP_TYPES:
            failed.append(f"{exp_path}: exp_type is one of {', '.join(EXP_TYPES)}")
            continue

        if not objective:
            skipped.append(f"{exp_path}: no objective")
            continue

        if not (args.main_folder / exp_path).is_dir():
            failed.append(f"{exp_path}: no such folder")
            continue

        if not condition and exp_type == "grab":
            condition = recording_name(args.main_folder / exp_path)

        day, mouse, number = match.groups()
        session_date = f"{day[:4]}-{day[4:6]}-{day[6:]}"

        if user not in databases:
            databases[user] = Database(
                args.main_folder, project=args.project, user=user
            )

        db = databases[user]
        session_id = db.session_for(mouse, session_date)

        if session_id is not None and db.experiment_for(session_id, int(number)):
            skipped.append(f"{exp_path}: already added")
            continue

        what = f"{exp_path} as {user}, {objective}x"
        what += ", a grab" if exp_type == "grab" else ""
        what += f", condition {condition}" if condition else ""

        if args.dry_run:
            added.append(f"would add {what}")
            continue

        try:
            db.set_mouse(mouse_id=mouse)  # changes nothing for a known one

            if session_id is None:
                session_id = db.add_session(mouse_id=mouse, session_date=session_date)

            exp_id = db.add_experiment(
                session_id=session_id,
                exp_path=exp_path,
                objective=int(objective),
                exp_type=exp_type,
            )

            if exp_id is not None and condition:
                write_condition(db, exp_id, condition)

        except Exception as error:  # said, and the next line is tried
            failed.append(f"{exp_path}: {type(error).__name__}: {error}")
            continue

        if exp_id is None:
            failed.append(f"{exp_path}: not added (see the log above)")
        else:
            added.append(f"{what}: experiment {exp_id}")

    for title, lines in (("Added", added), ("Skipped", skipped), ("Failed", failed)):
        print(f"\n{title} ({len(lines)})")

        for line in lines:
            print(f"  {line}")

    return 1 if failed else 0


def recording_name(folder: Path) -> str:
    """
    The name the first TIFF of `folder` (an experiment) was recorded under:
    its file name without the number, or `""` when it has none.
    """
    tiffs = sorted((folder / "raw").glob("[!.]?*.tif"))
    match = RECORDING_NAME.fullmatch(tiffs[0].stem) if tiffs else None

    return match.group(1) if match else ""


def write_condition(db: Database, exp_id: int, condition: str) -> None:
    """
    Write `condition` on every acquisition of experiment `exp_id`, first
    adding it to the options of the annotation when it is new (the options
    match whatever the case).
    """
    options = db.annotation_keys.loc[("acquisition", "condition"), "allowed_values"]

    if condition.lower() not in {str(option).lower() for option in options}:
        db.update_annotation_key(
            applies_to="acquisition", key="condition", add_values=[condition]
        )

    rows = db.from_query(
        "SELECT acq_id FROM acquisitions WHERE exp_id = ?;", [int(exp_id)]
    )

    for acq_id in rows["acq_id"]:
        db.add_annotations(
            target_type="acquisition",
            target_id=int(acq_id),
            values={"condition": condition},
        )


if __name__ == "__main__":
    sys.exit(main())
