#!/usr/bin/env python3
"""
Survey the recording folders and collate every session log into flat tables.

USAGE
    python collect_session_logs.py MAIN_FOLDER [--out FOLDER]

Reads nothing but the folder tree and the log workbooks, and writes nothing
back: the output is a folder of CSVs to read, correct and then import. Run it
on the machine that holds the data, then copy the output folder off.

OUTPUTS (all under `--out`, default `session_survey/`)
    sessions.csv        one row per mouse/day folder, and what it holds
    experiments.csv     one row per experiment, from the `expLog` sheet
    odor_panels.csv     one row per (session, odor), from the `odorLog` sheet
    programs.csv        one row per block, from the `acqLog` sheet
    problems.csv        everything missing, unreadable or ambiguous
    sync_files.txt      sync H5 paths, one per line
    log_files.txt       workbook paths, one per line
    summary.txt         counts, sizes, and the commands to copy files off

The two `.txt` lists are relative to MAIN_FOLDER so they can be fed straight to
`rsync --files-from=`; `summary.txt` spells the command out.

REQUIRES
    openpyxl, to read the workbooks.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

from datetime import date, datetime, time
from pathlib import Path

# A recording folder is '<YYYYMMDD>/<mouse>/<experiment>', so a session -- one
# mouse on one day -- is the folder above the experiments.
DATE_FOLDER = re.compile(r"^\d{8}$")
MOUSE_FOLDER = re.compile(r"^[A-Za-z]+\d+$")

# The sync recording is copied in beside the experiment it belongs to.
SYNC_SUBFOLDER = "sync"

# Workbooks are named '<date>_<mouse>_Log_in_vivo.xlsx', but they are looked up
# by suffix and searched for in both the session and experiment folders, since
# the name is typed by hand.
LOG_PATTERN = "*.xlsx"

# Sheet -> the columns worth collecting from it. Column names are matched after
# lowercasing and dropping non-alphanumerics, so 'goal ppm', 'goalPPM' and
# 'Goal_ppm' all reach the same field.
SHEETS = {
    "mouse": None,  # key/value down two columns rather than a table
    "expLog": (
        "expNum", "objective", "pmtGain", "roiDepth_um", "roiDescription",
        "goal", "acqInterval_s", "laserPower_%", "frameRate_hz", "frames",
        "acqs", "flag",
    ),
    "odorLog": ("vial #", "odor #", "odor name", "made on", "goal ppm",
                "% v/v", "sccm"),
    "acqLog": ("mouseSID", "expNum", "acqNum", "description",
               "olfactometer program", "start at"),
}

# The `mouse` sheet is a column of labels beside a column of values.
MOUSE_FIELDS = (
    "goal", "flag", "mouseSID", "mouse weight (g)", "s.q. injection vol (ml)",
    "headplate", "pitch angle", "left-right-correction", "notes",
)


def normalized(name) -> str:
    """Comparable form of a column label: lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def as_text(value) -> str:
    """A cell as it should appear in a CSV, with dates kept as dates."""
    if value is None:
        return ""

    if isinstance(value, datetime):
        # Workbook dates carry a midnight time that was never recorded.
        return value.date().isoformat()

    if isinstance(value, (date, time)):
        return value.isoformat()

    return str(value).strip()


def find_sessions(main_folder: Path) -> list[dict]:
    """Every '<date>/<mouse>' folder, with the experiments and files under it."""
    sessions = []

    for date_folder in sorted(p for p in main_folder.iterdir() if p.is_dir()):
        if not DATE_FOLDER.match(date_folder.name):
            continue

        for mouse_folder in sorted(p for p in date_folder.iterdir() if p.is_dir()):
            if not MOUSE_FOLDER.match(mouse_folder.name):
                continue

            experiments = [
                p for p in sorted(mouse_folder.iterdir())
                if p.is_dir() and (p / "raw").is_dir()
            ]

            # A workbook may sit beside the experiments or inside one of them.
            logs = sorted(mouse_folder.glob(LOG_PATTERN))
            logs += sorted(
                path
                for experiment in experiments
                for path in experiment.glob(LOG_PATTERN)
            )

            sync_files = sorted(
                path
                for experiment in experiments
                for path in (experiment / SYNC_SUBFOLDER).glob("*.h5")
                if (experiment / SYNC_SUBFOLDER).is_dir()
            )

            sessions.append({
                "session_date": date_folder.name,
                "mouse_id": mouse_folder.name,
                "folder": mouse_folder,
                "experiments": experiments,
                "logs": logs,
                "sync_files": sync_files,
            })

    return sessions


def read_workbook(path: Path) -> tuple[dict, list[dict]]:
    """
    Everything worth keeping from one log workbook.

    Returns `(fields, problems)`, where `fields` holds one entry per sheet:
    `mouse` is a flat dict, the others are lists of row dicts. Sheets that are
    absent are simply missing from it.
    """
    import openpyxl

    problems = []

    try:
        book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as error:
        return {}, [{"detail": f"cannot read workbook: {type(error).__name__}"}]

    fields: dict = {}

    try:
        if "mouse" in book.sheetnames:
            fields["mouse"] = _read_key_value_sheet(book["mouse"])

        for sheet_name, columns in SHEETS.items():
            if columns is None or sheet_name not in book.sheetnames:
                continue

            rows, missing = _read_table_sheet(book[sheet_name], columns)
            fields[sheet_name] = rows

            problems += [
                {"detail": f"{sheet_name}: no column for {name!r}"} for name in missing
            ]

        for sheet_name in SHEETS:
            if sheet_name not in book.sheetnames:
                problems.append({"detail": f"no {sheet_name!r} sheet"})

    finally:
        book.close()

    return fields, problems


def _read_key_value_sheet(sheet) -> dict:
    """A sheet laid out as labels in one column and values in the next."""
    wanted = {normalized(name): name for name in MOUSE_FIELDS}
    found: dict = {}
    notes = []

    for row in sheet.iter_rows(values_only=True):
        if not row:
            continue

        label, *rest = row
        value = next((v for v in rest if v is not None), None)
        key = wanted.get(normalized(label or ""))

        if key == "notes":
            # Notes run down the value column for many rows under one label.
            notes.append(as_text(value))
        elif key is not None:
            found[key] = as_text(value)
        elif notes and label is None and value is not None:
            notes.append(as_text(value))

    if notes:
        found["notes"] = " | ".join(note for note in notes if note)

    return found


def _read_table_sheet(sheet, columns) -> tuple[list[dict], list[str]]:
    """A sheet with a header row, read into one dict per row."""
    rows = list(sheet.iter_rows(values_only=True))

    if not rows:
        return [], list(columns)

    header = {normalized(name): index for index, name in enumerate(rows[0]) if name}
    missing = [name for name in columns if normalized(name) not in header]

    collected = []

    for row in rows[1:]:
        entry = {
            name: as_text(row[header[normalized(name)]])
            for name in columns
            if normalized(name) in header and header[normalized(name)] < len(row)
        }

        if any(entry.values()):
            collected.append(entry)

    return collected, missing


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """One CSV, with `columns` in that order and blanks where a key is absent."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def collect(main_folder: Path, out_folder: Path) -> dict:
    """Walk the folders, read every workbook, and write the output files."""
    out_folder.mkdir(parents=True, exist_ok=True)

    sessions = find_sessions(main_folder)

    session_rows, experiment_rows, odor_rows, program_rows, problems = [], [], [], [], []
    sync_paths, log_paths = [], []

    for session in sessions:
        key = {
            "session_date": session["session_date"],
            "mouse_id": session["mouse_id"],
        }
        relative = session["folder"].relative_to(main_folder).as_posix()

        sync_paths += [p.relative_to(main_folder).as_posix() for p in session["sync_files"]]
        log_paths += [p.relative_to(main_folder).as_posix() for p in session["logs"]]

        def note(detail: str) -> None:
            problems.append({**key, "folder": relative, "detail": detail})

        if not session["logs"]:
            note("no log workbook")
        elif len(session["logs"]) > 1:
            note(f"{len(session['logs'])} log workbooks, using none")

        if not session["sync_files"]:
            note("no sync H5")

        if not session["experiments"]:
            note("no experiment folders (nothing with a 'raw' subfolder)")

        fields: dict = {}

        if len(session["logs"]) == 1:
            fields, workbook_problems = read_workbook(session["logs"][0])
            problems += [{**key, "folder": relative, **p} for p in workbook_problems]

            # The mouse is written in the workbook and in the folder name. They
            # disagree when a workbook is copied from another session and only
            # partly edited, which is worth catching before any of it is
            # believed.
            written = fields.get("mouse", {}).get("mouseSID", "")
            in_folder = re.sub(r"^[A-Za-z]+", "", session["mouse_id"])

            if written and written != in_folder:
                note(f"workbook says mouse {written!r}, folder says {in_folder!r}")

            # One row per experiment is what the sheet is for, so a mismatch
            # means either a missing row or a folder the log does not cover.
            logged = len(fields.get("expLog", []))

            if logged and logged != len(session["experiments"]):
                note(
                    f"{logged} rows in expLog but {len(session['experiments'])} "
                    f"experiment folders"
                )

        session_rows.append({
            **key,
            "folder": relative,
            "experiments": len(session["experiments"]),
            "experiment_names": " ".join(p.name for p in session["experiments"]),
            "sync_files": len(session["sync_files"]),
            "log_files": len(session["logs"]),
            **fields.get("mouse", {}),
        })

        for row in fields.get("expLog", []):
            experiment_rows.append({**key, **row})

        for row in fields.get("odorLog", []):
            odor_rows.append({**key, **row})

        for row in fields.get("acqLog", []):
            program_rows.append({**key, **row})

    write_csv(
        out_folder / "sessions.csv",
        session_rows,
        ["session_date", "mouse_id", "folder", "experiments", "experiment_names",
         "sync_files", "log_files", *MOUSE_FIELDS],
    )
    write_csv(
        out_folder / "experiments.csv",
        experiment_rows,
        ["session_date", "mouse_id", *SHEETS["expLog"]],
    )
    write_csv(
        out_folder / "odor_panels.csv",
        odor_rows,
        ["session_date", "mouse_id", *SHEETS["odorLog"]],
    )
    write_csv(
        out_folder / "programs.csv",
        program_rows,
        ["session_date", "mouse_id", *SHEETS["acqLog"]],
    )
    write_csv(
        out_folder / "problems.csv",
        problems,
        ["session_date", "mouse_id", "folder", "detail"],
    )

    (out_folder / "sync_files.txt").write_text("\n".join(sync_paths) + "\n")
    (out_folder / "log_files.txt").write_text("\n".join(log_paths) + "\n")

    sync_bytes = sum(
        (main_folder / p).stat().st_size for p in sync_paths
        if (main_folder / p).exists()
    )

    return {
        "sessions": len(sessions),
        "experiments": sum(len(s["experiments"]) for s in sessions),
        "with_log": sum(1 for s in sessions if len(s["logs"]) == 1),
        "with_sync": sum(1 for s in sessions if s["sync_files"]),
        "odor_rows": len(odor_rows),
        "program_rows": len(program_rows),
        "problems": len(problems),
        "sync_files": len(sync_paths),
        "sync_gb": sync_bytes / 1e9,
        "log_files": len(log_paths),
    }


def write_summary(main_folder: Path, out_folder: Path, counts: dict) -> str:
    """A readable summary, including the commands to copy the files off."""
    lines = [
        "SESSION SURVEY",
        f"  main folder     {main_folder}",
        "",
        f"  sessions        {counts['sessions']}",
        f"    with a log    {counts['with_log']}",
        f"    with a sync   {counts['with_sync']}",
        f"  experiments     {counts['experiments']}",
        f"  odor rows       {counts['odor_rows']}",
        f"  program rows    {counts['program_rows']}",
        f"  problems        {counts['problems']}  (see problems.csv)",
        "",
        f"  sync H5 files   {counts['sync_files']}  ({counts['sync_gb']:.1f} GB)",
        f"  log workbooks   {counts['log_files']}",
        "",
        "COPY THE RESULTS OFF",
        f"  scp -r USER@HOST:{out_folder} .",
        "",
        "COPY THE SYNC FILES AND WORKBOOKS",
        f"  rsync -av --files-from={out_folder / 'sync_files.txt'} \\",
        f"      USER@HOST:{main_folder} ./sync/",
        f"  rsync -av --files-from={out_folder / 'log_files.txt'} \\",
        f"      USER@HOST:{main_folder} ./logs/",
    ]
    text = "\n".join(lines) + "\n"
    (out_folder / "summary.txt").write_text(text)

    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("main_folder", type=Path, help="folder holding the dates")
    parser.add_argument(
        "--out", type=Path, default=Path("session_survey"),
        help="where to write the CSVs (default: session_survey)",
    )
    arguments = parser.parse_args()

    main_folder = arguments.main_folder.resolve()

    if not main_folder.is_dir():
        sys.exit(f"Not a folder: {main_folder}")

    counts = collect(main_folder, arguments.out.resolve())

    print(write_summary(main_folder, arguments.out.resolve(), counts))


if __name__ == "__main__":
    main()
