#!/usr/bin/env python3
"""
Survey the recording folders and collate every session log into flat tables.

USAGE
    python collect_session_logs.py MAIN_FOLDER [--out FOLDER] [--skip-sync]

Reads the folder tree, the log workbooks and the sync recordings, and writes
nothing back: the output is a folder of CSVs to read, correct and then import.
Run it on the machine that holds the data, then copy the output folder off.

OUTPUTS (all under `--out`, default `session_survey/`)
    sessions.csv        one row per session, and what it holds
    experiments.csv     one row per experiment, from the `expLog` sheet
    odor_panels.csv     one row per (session, odor), from the `odorLog` sheet
    programs.csv        one row per block, from the `acqLog` sheet
    sync_files.csv      one row per sync recording, read from the signals
    vocabulary.csv      every sheet, column and label seen, and how often
    problems.csv        everything missing, unreadable or inconsistent
    sync_files.txt      sync H5 paths, one per line
    log_files.txt       workbook paths, one per line
    summary.txt         counts, sizes, and the commands to copy files off

`vocabulary.csv` is the one to read first when deciding what a session is worth
recording: the other tables only hold fields this script was told to look for,
while that one holds everything the workbooks actually use.

The two `.txt` lists are relative to MAIN_FOLDER so they can be fed straight to
`rsync --files-from=`; `summary.txt` spells the command out.

REQUIRES
    openpyxl to read the workbooks, h5py and numpy to read the sync recordings.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys

from collections import Counter, defaultdict
from datetime import date, datetime, time
from pathlib import Path

DATE_FOLDER = re.compile(r"^\d{8}$")

# An experiment folder is one holding a `raw` folder of TIFFs. Sessions are
# found by looking for those rather than by matching folder names, because a
# session may be a `<date>/<mouse>` folder holding experiments or, where there
# is no mouse level, the date folder itself.
RAW_SUBFOLDER = "raw"
SYNC_SUBFOLDER = "sync"

# Raw files are named `<date>_<mouse>_<experiment>_<number>.tif`, which is where
# the mouse comes from when the folders do not name it.
RAW_STEM_PARTS = 4

LOG_PATTERN = "*.xlsx"

# Columns to pull from each sheet. `vocabulary.csv` reports everything else so
# this list can be extended from evidence.
SHEETS = {
    "expLog": (
        "expNum",
        "objective",
        "pmtGain",
        "roiDepth_um",
        "roiDescription",
        "goal",
        "acqInterval_s",
        "laserPower_%",
        "frameRate_hz",
        "frames",
        "acqs",
        "flag",
    ),
    "odorLog": (
        "vial #",
        "odor #",
        "odor name",
        "made on",
        "goal ppm",
        "% v/v",
        "sccm",
    ),
    "acqLog": (
        "mouseSID",
        "expNum",
        "acqNum",
        "description",
        "olfactometer program",
        "start at",
    ),
}

# The `mouse` sheet is a column of labels beside a column of values.
MOUSE_FIELDS = (
    "goal",
    "flag",
    "mouseSID",
    "mouse weight (g)",
    "s.q. injection vol (ml)",
    "headplate",
    "pitch angle",
    "left-right-correction",
    "notes",
)

# Channels rise to about 5 V, so half way is a safe edge threshold.
LOGIC_THRESHOLD_V = 2.5

# Frames arrive at tens of hertz, so anything far below a millisecond apart is
# contact bounce on the clock line rather than a frame.
MIN_FRAME_INTERVAL_S = 0.005

# Imaging pauses between acquisitions for much longer than a frame period.
BURST_GAP_S = 1.0

# A pause this long is an interruption in the session rather than the usual
# wait between acquisitions.
LONG_PAUSE_S = 60.0


def normalized(name) -> str:
    """Comparable form of a label: lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def mouse_number(name) -> str:
    """Comparable form of a mouse, so that `m442`, `M442` and `442` agree."""
    return re.sub(r"^[A-Za-z]+", "", str(name).strip()).lstrip("0")


def as_text(value) -> str:
    """A cell as it should appear in a CSV, with dates kept as dates."""
    if value is None:
        return ""

    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, (date, time)):
        return value.isoformat()

    return str(value).strip()


# --------------------------------------------------------------------------- #
# Folders
# --------------------------------------------------------------------------- #


def experiment_folders(folder: Path) -> list[Path]:
    """Folders directly inside `folder` that hold raw TIFFs."""
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_dir() and (path / RAW_SUBFOLDER).is_dir()
    ]


def mouse_from_raw(experiments: list[Path]) -> str:
    """The mouse named by the raw file names, or '' if they do not say."""
    for experiment in experiments:
        for raw in sorted((experiment / RAW_SUBFOLDER).glob("*.tif")):
            parts = raw.stem.split("_")

            if len(parts) >= RAW_STEM_PARTS:
                return parts[1]

    return ""


def session_from(
    folder: Path,
    session_date: str,
    mouse_id: str,
    mouse_source: str,
    experiments: list[Path],
) -> dict:
    """One session, with the files found under it."""
    logs = sorted(folder.glob(LOG_PATTERN))
    logs += sorted(path for e in experiments for path in e.glob(LOG_PATTERN))

    sync_files = sorted(
        path
        for experiment in experiments
        if (experiment / SYNC_SUBFOLDER).is_dir()
        for path in (experiment / SYNC_SUBFOLDER).glob("*.h5")
    )

    return {
        "session_date": session_date,
        "mouse_id": mouse_id,
        "mouse_source": mouse_source,
        "folder": folder,
        "experiments": experiments,
        "logs": logs,
        "sync_files": sync_files,
    }


def find_sessions(main_folder: Path) -> list[dict]:
    """
    Every session under `main_folder`, however its folders are arranged.

    A session is one mouse on one day. It is usually a `<date>/<mouse>` folder
    holding the experiments, but where there is no mouse level the date folder
    holds them directly and the mouse is read from the raw file names.
    """
    sessions = []

    for date_folder in sorted(p for p in main_folder.iterdir() if p.is_dir()):
        if not DATE_FOLDER.match(date_folder.name):
            continue

        direct = experiment_folders(date_folder)

        if direct:
            sessions.append(
                session_from(
                    date_folder,
                    date_folder.name,
                    mouse_from_raw(direct),
                    "raw file name",
                    direct,
                )
            )

        for child in sorted(p for p in date_folder.iterdir() if p.is_dir()):
            if child in direct:
                continue

            experiments = experiment_folders(child)

            if experiments or list(child.glob(LOG_PATTERN)):
                sessions.append(
                    session_from(
                        child,
                        date_folder.name,
                        child.name,
                        "folder",
                        experiments,
                    )
                )

    return sessions


# --------------------------------------------------------------------------- #
# Workbooks
# --------------------------------------------------------------------------- #


def read_workbook(path: Path) -> tuple[dict, dict, list[str]]:
    """
    Everything worth keeping from one log workbook.

    Returns `(fields, vocabulary, problems)`. `fields` holds one entry per
    sheet: `mouse` is a flat dict, the others are lists of row dicts, and sheets
    that are absent are missing from it. `vocabulary` maps a sheet name to the
    labels it uses and how many carried a value.
    """
    import openpyxl

    try:
        book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as error:
        return {}, {}, [f"cannot read workbook: {type(error).__name__}"]

    fields: dict = {}
    vocabulary: dict = {}
    problems: list[str] = []

    try:
        for sheet_name in book.sheetnames:
            sheet = book[sheet_name]

            if normalized(sheet_name) == "mouse":
                values, seen = _read_key_value_sheet(sheet)
                fields["mouse"] = values
            else:
                rows, seen = _read_table_sheet(sheet)
                fields[sheet_name] = rows

            vocabulary[sheet_name] = seen

        for sheet_name, columns in SHEETS.items():
            if sheet_name not in book.sheetnames:
                problems.append(f"no {sheet_name!r} sheet")
                continue

            present = {normalized(name) for name in vocabulary[sheet_name]}
            problems += [
                f"{sheet_name}: no column for {name!r}"
                for name in columns
                if normalized(name) not in present
            ]

        if "mouse" not in {normalized(name) for name in book.sheetnames}:
            problems.append("no 'mouse' sheet")

    finally:
        book.close()

    return fields, vocabulary, problems


def _read_key_value_sheet(sheet) -> tuple[dict, Counter]:
    """A sheet laid out as labels in one column and values in the next."""
    wanted = {normalized(name): name for name in MOUSE_FIELDS}
    found: dict = {}
    seen: Counter = Counter()
    notes: list[str] = []
    in_notes = False

    for row in sheet.iter_rows(values_only=True):
        if not row:
            continue

        label, *rest = row
        value = next((v for v in rest if v is not None), None)

        if label is not None:
            seen[str(label).strip()] += 1 if value is not None else 0
            in_notes = normalized(label) == "notes"

            key = wanted.get(normalized(label))

            if key == "notes":
                notes.append(as_text(value))
            elif key is not None:
                found[key] = as_text(value)

        elif in_notes and value is not None:
            # Notes run down the value column for many rows under one label.
            notes.append(as_text(value))

    if notes:
        found["notes"] = " | ".join(note for note in notes if note)

    return found, seen


def _read_table_sheet(sheet) -> tuple[list[dict], Counter]:
    """A sheet with a header row, read into one dict per row."""
    rows = list(sheet.iter_rows(values_only=True))

    if not rows:
        return [], Counter()

    header = {index: str(name).strip() for index, name in enumerate(rows[0]) if name}
    seen: Counter = Counter({name: 0 for name in header.values()})
    collected = []

    for row in rows[1:]:
        entry = {
            name: as_text(row[index])
            for index, name in header.items()
            if index < len(row)
        }

        if any(entry.values()):
            collected.append(entry)

            for name, value in entry.items():
                if value:
                    seen[name] += 1

    return collected, seen


# --------------------------------------------------------------------------- #
# Sync recordings
# --------------------------------------------------------------------------- #


def rising_edges(signal) -> "list":
    """Sample indices where a channel crosses the logic threshold going up."""
    import numpy as np

    high = signal > LOGIC_THRESHOLD_V

    return np.flatnonzero(~high[:-1] & high[1:]) + 1


def read_sync(path: Path) -> dict:
    """
    What one sync recording holds, read from its header and its signals.

    Reports the imaging frame clock as bursts of frames, since a burst is one
    acquisition, along with the odor valve and camera channels. An absent
    `stop_time` means the recording was interrupted before it was saved.
    """
    import h5py
    import numpy as np

    found: dict = {"file": path.name}

    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)

        found["mouse"] = str(attrs.get("mouse", ""))
        found["experiment"] = str(attrs.get("experiment", ""))
        found["rate_hz"] = attrs.get("rate_hz", "")
        found["start_time"] = str(attrs.get("start_time", ""))
        found["stop_time"] = str(attrs.get("stop_time", ""))
        found["duration_s"] = round(float(attrs.get("duration_s", 0)), 1)
        found["interrupted"] = "stop_time" not in attrs
        found["channels"] = " ".join(sorted(handle))

        rate = float(attrs.get("rate_hz", 0)) or 1.0

        if "filter_changes" in handle:
            found["filter_changes"] = len(handle["filter_changes"])

        if "2pFrameSync" not in handle:
            return found

        frames = rising_edges(handle["2pFrameSync"][:])
        found["frame_pulses"] = len(frames)

        if len(frames) < 2:
            return found

        gaps = np.diff(frames) / rate
        breaks = np.flatnonzero(gaps > BURST_GAP_S)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(frames) - 1]])
        sizes = ends - starts + 1

        found["frame_rate_hz"] = round(
            rate / float(np.median(np.diff(frames[starts[0] : ends[0] + 1]))), 3
        )
        found["bursts"] = len(sizes)
        found["burst_sizes"] = " ".join(
            f"{size}x{count}" for size, count in sorted(Counter(sizes.tolist()).items())
        )
        found["long_pauses_s"] = " ".join(
            str(round(float(gap), 0))
            for gap in (frames[starts[1:]] - frames[ends[:-1]]) / rate
            if gap > LONG_PAUSE_S
        )

        if "odorPulse" in handle:
            odor = handle["odorPulse"][:]
            high = (odor > LOGIC_THRESHOLD_V).astype(np.int8)
            on = np.flatnonzero(np.diff(high) == 1) + 1
            off = np.flatnonzero(np.diff(high) == -1) + 1

            found["odor_pulses"] = len(on)
            found["odor_starts_high"] = bool(high[0])

            if len(on) and len(off):
                width = (off[: len(on)] - on[: len(off)]) / rate
                found["odor_width_s"] = round(float(np.median(width)), 3)

            # Each acquisition should contain exactly one odor presentation.
            per_burst = Counter(
                int(((on >= frames[a]) & (on <= frames[b])).sum())
                for a, b in zip(starts, ends)
            )
            found["odor_per_burst"] = " ".join(
                f"{count}x{n}" for count, n in sorted(per_burst.items())
            )
            found["odor_outside_bursts"] = int(
                len(on) - sum(k * v for k, v in per_burst.items())
            )

            latencies = [
                float(on[(on >= frames[a]) & (on <= frames[b])][0] - frames[a]) / rate
                for a, b in zip(starts, ends)
                if ((on >= frames[a]) & (on <= frames[b])).any()
            ]

            if latencies:
                found["odor_latency_s"] = round(float(np.median(latencies)), 3)

        if "cameraFrameSync" in handle:
            camera = rising_edges(handle["cameraFrameSync"][:])
            found["camera_pulses"] = len(camera)

            if len(camera) > 1:
                camera_breaks = np.flatnonzero(np.diff(camera) / rate > BURST_GAP_S)
                found["camera_bursts"] = len(camera_breaks) + 1
                found["camera_rate_hz"] = round(
                    rate / float(np.median(np.diff(camera[: min(1000, len(camera))]))),
                    3,
                )

                # Acquisitions the camera did not cover have no pupil recording.
                blocks = list(
                    zip(
                        np.concatenate([[0], camera_breaks + 1]),
                        np.concatenate([camera_breaks, [len(camera) - 1]]),
                    )
                )
                found["acquisitions_without_camera"] = sum(
                    1
                    for a, b in zip(starts, ends)
                    if not any(
                        camera[i] <= frames[a] and camera[j] >= frames[b]
                        for i, j in blocks
                    )
                )

    return found


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """One CSV, with `columns` in that order and blanks where a key is absent."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def collect(main_folder: Path, out_folder: Path, read_sync_files: bool) -> dict:
    """Walk the folders, read every workbook and recording, write the output."""
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"Looking for sessions under {main_folder} ...", flush=True)
    sessions = find_sessions(main_folder)
    print(f"  found {len(sessions)} sessions", flush=True)

    session_rows, experiment_rows, odor_rows, program_rows = [], [], [], []
    problems, sync_rows = [], []
    sync_paths, log_paths = [], []
    vocabulary: dict = defaultdict(Counter)
    vocabulary_files: dict = defaultdict(Counter)

    for number, session in enumerate(sessions, start=1):
        key = {
            "session_date": session["session_date"],
            "mouse_id": session["mouse_id"],
        }
        relative = session["folder"].relative_to(main_folder).as_posix()

        sync_paths += [
            p.relative_to(main_folder).as_posix() for p in session["sync_files"]
        ]
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
            note("no experiment folders")

        if not session["mouse_id"]:
            note("no mouse in the folder names or the raw file names")

        fields: dict = {}

        if len(session["logs"]) == 1:
            print(f"  [{number}/{len(sessions)}] {relative}", flush=True)
            fields, seen, workbook_problems = read_workbook(session["logs"][0])

            for sheet_name, counts in seen.items():
                vocabulary[sheet_name].update(counts)
                vocabulary_files[sheet_name].update(counts.keys())

            for detail in workbook_problems:
                note(detail)

            written = fields.get("mouse", {}).get("mouseSID", "")

            if written and mouse_number(written) != mouse_number(session["mouse_id"]):
                note(
                    f"workbook says mouse {written!r}, folder says "
                    f"{session['mouse_id']!r}"
                )

            logged = len(fields.get("expLog", []))

            if logged and logged != len(session["experiments"]):
                note(
                    f"{logged} rows in expLog but {len(session['experiments'])} "
                    f"experiment folders"
                )

        session_rows.append(
            {
                **key,
                "folder": relative,
                "mouse_source": session["mouse_source"],
                "experiments": len(session["experiments"]),
                "experiment_names": " ".join(p.name for p in session["experiments"]),
                "sync_files": len(session["sync_files"]),
                "log_files": len(session["logs"]),
                **fields.get("mouse", {}),
            }
        )

        for sheet_name, target in (
            ("expLog", experiment_rows),
            ("odorLog", odor_rows),
            ("acqLog", program_rows),
        ):
            target += [{**key, **row} for row in fields.get(sheet_name, [])]

    if read_sync_files:
        print(f"\nReading {len(sync_paths)} sync recordings ...", flush=True)

        for number, relative in enumerate(sync_paths, start=1):
            print(f"  [{number}/{len(sync_paths)}] {relative}", flush=True)

            try:
                sync_rows.append(
                    {"path": relative, **read_sync(main_folder / relative)}
                )
            except Exception as error:
                sync_rows.append({"path": relative, "file": Path(relative).name})
                problems.append(
                    {
                        "session_date": "",
                        "mouse_id": "",
                        "folder": relative,
                        "detail": f"cannot read sync H5: {type(error).__name__}: {error}",
                    }
                )

    print("\nWriting the survey ...", flush=True)

    write_csv(
        out_folder / "sessions.csv",
        session_rows,
        [
            "session_date",
            "mouse_id",
            "mouse_source",
            "folder",
            "experiments",
            "experiment_names",
            "sync_files",
            "log_files",
            *MOUSE_FIELDS,
        ],
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

    vocabulary_rows = [
        {
            "sheet": sheet_name,
            "label": label,
            "workbooks": vocabulary_files[sheet_name][label],
            "values": count,
            "collected": label
            in (
                MOUSE_FIELDS
                if normalized(sheet_name) == "mouse"
                else SHEETS.get(sheet_name, ())
            ),
        }
        for sheet_name, counts in sorted(vocabulary.items())
        for label, count in counts.most_common()
    ]
    write_csv(
        out_folder / "vocabulary.csv",
        vocabulary_rows,
        ["sheet", "label", "workbooks", "values", "collected"],
    )

    sync_columns = [
        "path",
        "file",
        "mouse",
        "experiment",
        "interrupted",
        "rate_hz",
        "start_time",
        "stop_time",
        "duration_s",
        "channels",
        "filter_changes",
        "frame_pulses",
        "frame_rate_hz",
        "bursts",
        "burst_sizes",
        "long_pauses_s",
        "odor_pulses",
        "odor_width_s",
        "odor_latency_s",
        "odor_per_burst",
        "odor_outside_bursts",
        "odor_starts_high",
        "camera_pulses",
        "camera_bursts",
        "camera_rate_hz",
        "acquisitions_without_camera",
    ]
    write_csv(out_folder / "sync_files.csv", sync_rows, sync_columns)

    (out_folder / "sync_files.txt").write_text("\n".join(sync_paths) + "\n")
    (out_folder / "log_files.txt").write_text("\n".join(log_paths) + "\n")

    sync_bytes = sum(
        (main_folder / p).stat().st_size
        for p in sync_paths
        if (main_folder / p).exists()
    )

    return {
        "sessions": len(sessions),
        "experiments": sum(len(s["experiments"]) for s in sessions),
        "with_log": sum(1 for s in sessions if len(s["logs"]) == 1),
        "with_sync": sum(1 for s in sessions if s["sync_files"]),
        "from_raw_names": sum(
            1 for s in sessions if s["mouse_source"] == "raw file name"
        ),
        "odor_rows": len(odor_rows),
        "program_rows": len(program_rows),
        "labels": len(vocabulary_rows),
        "uncollected": sum(1 for row in vocabulary_rows if not row["collected"]),
        "problems": len(problems),
        "sync_files": len(sync_paths),
        "sync_read": len(sync_rows),
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
        f"    mouse from    {counts['from_raw_names']} raw file names,"
        f" {counts['sessions'] - counts['from_raw_names']} folder names",
        f"  experiments     {counts['experiments']}",
        f"  odor rows       {counts['odor_rows']}",
        f"  program rows    {counts['program_rows']}",
        "",
        f"  labels seen     {counts['labels']}  ({counts['uncollected']} not"
        f" collected yet, see vocabulary.csv)",
        f"  problems        {counts['problems']}  (see problems.csv)",
        "",
        f"  sync H5 files   {counts['sync_files']}  ({counts['sync_gb']:.1f} GB,"
        f" {counts['sync_read']} read)",
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
        "--out",
        type=Path,
        default=Path("session_survey"),
        help="where to write the CSVs (default: session_survey)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="list the sync recordings without opening them",
    )
    arguments = parser.parse_args()

    main_folder = arguments.main_folder.resolve()

    if not main_folder.is_dir():
        sys.exit(f"Not a folder: {main_folder}")

    out_folder = arguments.out.resolve()
    counts = collect(main_folder, out_folder, not arguments.skip_sync)

    print()
    print(write_summary(main_folder, out_folder, counts))


if __name__ == "__main__":
    main()
