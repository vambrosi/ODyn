#!/usr/bin/env python3
"""
Build the sync-era database from scratch: recordings, odor panels, mice.

USAGE
    python create_main_sync.py MAIN_FOLDER --sessions sync_sessions.txt
                               [--odors ODORS_XLSX] [--surgery SURGERY_XLSX]
                               [--project NAME] [--dry-run]

Run it on the machine that holds the recordings. It creates the project
database if it is not there, ingests every experiment of the listed sessions,
and then fills in the panels and the mice. Each step reports what it did and
what it skipped.

Safe to run again: ingestion skips experiments already in the database, an
unchanged panel is a no-op, and the mice are filled in field by field. A second
run after copying more recordings over adds only the new ones.

INPUTS
    --sessions  one session folder per line, relative to MAIN_FOLDER, as
                `scripts/collect_session_logs.py` writes it. Lines starting
                with `#` and blank lines are ignored.
    --odors     the lab odor workbook, for `Print v3` (see import_panels.py)
    --surgery   the lab surgery workbook (see import_mice.py)

Annotations from the log workbooks are a separate step -- see
`import_session_logs.py` -- because they are worth reviewing before import.

REQUIRES
    openpyxl for the workbooks, and whatever `odyn` itself needs to read TIFFs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from odyn import Database  # noqa: E402
from odyn.utils import DEFAULT_PROJECT  # noqa: E402

import import_mice  # noqa: E402
import import_panels  # noqa: E402

# Only sheets that describe a rack of vials. The older ones are kept in the
# workbook for reference and are not what any of these sessions ran.
PANEL_SHEETS = ["Print v3"]

RULE = "-" * 72


def read_session_list(path: Path) -> list[str]:
    """The session folders to ingest, in the order the file lists them."""
    lines = (line.strip() for line in path.read_text().splitlines())

    return [line for line in lines if line and not line.startswith("#")]


def experiment_folders(main_folder: Path, session: str) -> list[str]:
    """
    Every experiment folder inside one session, relative to `main_folder`.

    An experiment is a folder with a `raw/` subfolder of TIFFs, which is what
    `Database.add_experiment` reads. A session folder laid out any other way
    yields nothing and is reported rather than guessed at.
    """
    folder = main_folder / session

    if not folder.is_dir():
        return []

    found = {
        tiff.parent.parent.relative_to(main_folder).as_posix()
        for tiff in folder.rglob("raw/[!.]?*.tif")
    }

    return sorted(found)


def ingest(db: Database, main_folder: Path, sessions: list[str], dry_run: bool):
    """Add every experiment of every listed session. Returns what went wrong."""
    problems: list[str] = []
    added = 0

    for number, session in enumerate(sessions, start=1):
        folders = experiment_folders(main_folder, session)

        if not folders:
            problems.append(f"{session}: no 'raw/' folder of TIFFs found")
            print(f"  [{number:3d}/{len(sessions)}] {session}: nothing to ingest")
            continue

        print(f"  [{number:3d}/{len(sessions)}] {session}: {len(folders)} experiments")

        for rel_path in folders:
            if dry_run:
                added += 1
                continue

            try:
                db.add_experiment(rel_path=rel_path)
                added += 1
            except Exception as failure:
                # One unreadable experiment should not stop the other 38
                # sessions; it is reported at the end instead.
                problems.append(f"{rel_path}: {type(failure).__name__}: {failure}")

    return added, problems


def load_panels(db: Database, odors_xlsx: Path, dry_run: bool) -> list[str]:
    """Register each panel sheet. Returns what went wrong."""
    problems: list[str] = []

    for sheet in PANEL_SHEETS:
        vials = import_panels.read_panel(odors_xlsx, sheet)
        name = import_panels.panel_name(sheet)

        if not vials:
            problems.append(f"{sheet}: no vial rows found")
            continue

        missing = import_panels.unknown_odors(db, vials)

        if missing:
            problems.append(f"{sheet}: odors {sorted(missing)} are not in `odors`")
            continue

        print(f"  {sheet} -> {name}: {len(vials)} vials")

        if not dry_run:
            try:
                db.add_panel(
                    panel_name=name,
                    vials=vials,
                    description=f"Imported from {odors_xlsx.name}, sheet {sheet!r}.",
                )
            except ValueError as refused:
                problems.append(f"{sheet}: {refused}")

    return problems


def load_mice(db: Database, surgery_xlsx: Path, dry_run: bool) -> list[str]:
    """Fill in every mouse that now has a session. Returns what went wrong."""
    problems: list[str] = []
    records = import_mice.read_workbook(surgery_xlsx)

    wanted = sorted(
        int(sid)
        for (sid,) in db.con.execute("SELECT DISTINCT mouse_id FROM sessions;")
    )

    for sid in wanted:
        record = records.get(sid)

        if record is None:
            problems.append(f"m{sid}: not in {surgery_xlsx.name}")
            continue

        carried, complaint = import_mice.parse_lines(
            record["line"], record["genotype"]
        )

        if complaint:
            problems.append(f"m{sid}: {complaint}")

        summary = " x ".join(
            f"{line}({genotype or '?'})" for line, genotype in carried.items()
        )

        print(f"  m{sid}: {summary or 'unknown line'}")

        if not dry_run:
            db.set_mouse(
                mouse_id=sid,
                sex=import_mice.parse_sex(record["sex"]),
                dob=import_mice.parse_dob(record["dob"]),
                lines=carried or None,
            )

    return problems


def counts(db: Database) -> dict[str, int]:
    """How many rows each table that this script fills ended up with."""
    tables = (
        "sessions", "experiments", "acquisitions", "groups",
        "mice", "mouse_lines", "panels", "panel_vials", "vial_components",
    )

    return {
        table: db.con.execute(f"SELECT count(*) FROM {table};").fetchone()[0]
        for table in tables
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--odors", type=Path)
    parser.add_argument("--surgery", type=Path)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    sessions = read_session_list(args.sessions)

    print(RULE)
    print(f"main folder   {args.main_folder}")
    print(f"project       {args.project}")
    print(f"sessions      {len(sessions)} listed in {args.sessions.name}")

    if args.dry_run:
        print("DRY RUN -- nothing is written")

    db = Database(args.main_folder, project=args.project)

    print(f"database      {db.path}")

    print(f"\n{RULE}\nINGESTING RECORDINGS")
    added, problems = ingest(db, args.main_folder, sessions, args.dry_run)
    print(f"  {added} experiments")

    if args.odors:
        print(f"\n{RULE}\nODOR PANELS")
        problems += load_panels(db, args.odors, args.dry_run)

    if args.surgery:
        print(f"\n{RULE}\nMICE")
        problems += load_mice(db, args.surgery, args.dry_run)

    print(f"\n{RULE}\nRESULT")

    for table, rows in counts(db).items():
        print(f"  {table:18s} {rows:6d}")

    if problems:
        print(f"\n{len(problems)} problems:")

        for problem in problems:
            print(f"  - {problem}")

    print(
        f"\n{RULE}\nNext: import_session_logs.py, for the annotations the log "
        f"workbooks hold."
    )

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
