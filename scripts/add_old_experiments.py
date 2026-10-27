"""
Add experiments recorded before the app to a project, each as the person who
recorded it, so their motion correction is checked by them (see
`odyn.cluster.mcor-backfill`).

    python scripts/add_old_experiments.py <main_folder> --project main-sync \\
        experiments.csv [--dry-run]

`experiments.csv` has a line per experiment: `exp_path` (`YYYYMMDD/<mouse>/eN`,
relative to the main folder), `user` (who recorded it) and `objective` (10,
20, ...). Any other column is left alone. A line with no objective is skipped
and said, since the scale of the images depends on it.

For each experiment: the mouse is registered when it is new (its number only;
the rest is for its people to fill in, in the app), the session (mouse and
day) is added when it is not there yet, then the experiment (`add_experiment`:
its TIFFs and olfactometer files), all recorded as `user`. An experiment
already in the database is skipped, so the script can run again after a fix.
The sync file is left to the cluster tick, which processes the sync file of
every experiment in the database.

Reads every TIFF header, so run it on the cluster (`srun`), near the files.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

from pathlib import Path

from odyn import Database

EXP_PATH = re.compile(r"(\d{8})/([A-Za-z]*\d+)/e(\d+)")


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
        objective = row.get("objective", "").strip()
        match = EXP_PATH.fullmatch(exp_path)

        if match is None:
            failed.append(f"{exp_path}: not YYYYMMDD/<mouse>/eN")
            continue

        if not objective:
            skipped.append(f"{exp_path}: no objective")
            continue

        if not (args.main_folder / exp_path).is_dir():
            failed.append(f"{exp_path}: no such folder")
            continue

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

        if args.dry_run:
            added.append(f"would add {what}")
            continue

        try:
            db.set_mouse(mouse_id=mouse)  # changes nothing for a known one

            if session_id is None:
                session_id = db.add_session(mouse_id=mouse, session_date=session_date)

            exp_id = db.add_experiment(
                session_id=session_id, exp_path=exp_path, objective=int(objective)
            )
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


if __name__ == "__main__":
    sys.exit(main())
