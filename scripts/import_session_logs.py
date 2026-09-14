#!/usr/bin/env python3
"""
Import the log-workbook fields collated by `collect_session_logs.py`.

USAGE
    python import_session_logs.py MAIN_FOLDER SURVEY_FOLDER
                                  [--sessions sync_sessions.txt]
                                  [--project NAME] [--dry-run]

Reads `sessions.csv` and `experiments.csv` from the survey folder and writes
what they hold as annotations, plus the objective as a column. The survey
folder is the place to correct a workbook typo before it reaches the database:
this script reads those CSVs and never the workbooks.

WHAT IT WILL NOT DO
    It never guesses which experiment a row belongs to. A row whose `expNum`
    does not name a folder that ingestion created is reported and skipped, as
    is a session with two rows for one experiment. `expNum` written `e1` and
    written `1` are the same experiment; that is the only liberty taken.

    Z-stacks are logged in `expLog` but are not experiments in the database --
    they have no `raw/` folder, and `exp_type` has no 'zstack' yet. They are
    listed separately rather than as problems, and are the input for whenever
    z-stacks do get ingested.

    Values it cannot read are reported with the row they came from rather than
    coerced. Nothing is a best guess.

Run it after `create_main_sync.py`, which is what creates the sessions and
experiments these annotations attach to. Safe to run again: annotations are
append-only and the latest value of a key wins, so a second run after a
correction supersedes the first without losing what it replaced.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odyn import Database  # noqa: E402
from odyn.utils import DEFAULT_PROJECT  # noqa: E402

# Notes and flags arrive already joined by the survey, one entry per note.
SEPARATOR = " | "

# Column in the survey CSV -> annotation key. Everything here is a plain value
# that needs no interpretation beyond its type.
SESSION_TEXT = {
    "goal": "goal",
    "headplate": "headplate",
}

SESSION_LIST = {
    "notes": "note",
    "flag": "flag",
}

EXPERIMENT_TEXT = {
    "roiDescription": "fov_description",
    "goal": "goal",
}

EXPERIMENT_LIST = {
    "flag": "flag",
}

# Columns deliberately not imported, and why. Reported so that the reason is
# visible at the point someone wonders where the field went.
SKIPPED = {
    "laserPower_%": (
        "the workbook does not say which wavelength, and `add_experiment`"
        " already writes laser_power_920 and laser_power_1040 from the TIFF"
    ),
    "acqInterval_s": (
        "`add_experiment` already writes loop_acq_interval_s from the TIFF,"
        " which is what the rig was actually set to"
    ),
    "frameRate_hz": "already a column, read from the TIFF",
    "frames": "already a column, read from the TIFF",
    "acqs": "counted from the TIFFs at ingestion",
    "mouseSID": "the mouse is identified by the folder name",
}


class Report:
    """Everything that did not become an annotation, grouped by why."""

    def __init__(self):
        self.problems: list[str] = []
        self.zstacks: list[str] = []
        self.written = 0

    def problem(self, where: str, what: str) -> None:
        self.problems.append(f"{where}: {what}")

    def zstack(self, where: str, what: str) -> None:
        self.zstacks.append(f"{where}: {what}")


# --------------------------------------------------------------------------- #
# Reading the values people typed
# --------------------------------------------------------------------------- #


def experiment_number(written: str) -> None | str:
    """
    `'e1'` and `'1'` both name the folder `e1`; anything else names no folder.

    **EXAMPLE**
    ```python
    experiment_number("e1")          # "e1"
    experiment_number("zstackROI")   # None
    ```
    """
    match = re.fullmatch(r"[eE]?0*(\d+)", str(written).strip())

    return f"e{match.group(1)}" if match else None


def is_zstack(row: dict) -> bool:
    """
    Whether an `expLog` row describes a z-stack rather than an imaging run.

    Z-stacks are written into the same sheet, sometimes under the number of the
    experiment whose field they cover, so the number cannot tell them apart.
    The goal says so instead, and no z-stack row in the workbooks carries the
    per-acquisition timing an imaging run has.
    """
    said = f"{row.get('goal', '')} {row.get('expNum', '')}".lower()

    return "stack" in said


def number(written: str) -> None | float:
    """
    The leading number of a cell, ignoring a trailing unit.

    Returns `None` when the cell does not start with one, so that `'27g'` reads
    as 27 while `'right side down slightly'` reads as nothing at all. A leading
    `~` is allowed: the approximation is kept in the `_raw` annotation beside
    it, so dropping the mark here loses nothing.

    **EXAMPLE**
    ```python
    number("27g")     # 27.0
    number(".26cc")   # 0.26
    number("~200")    # 200.0
    ```
    """
    match = re.match(r"\s*~?\s*([-+]?(?:\d+\.?\d*|\.\d+))", str(written))

    return float(match.group(1)) if match else None


# A subcutaneous injection for a mouse is a fraction of a millilitre. Anything
# at or above this is a typo -- '26 cc' for '.26 cc' -- and is reported rather
# than recorded, because the number would look deliberate afterwards.
IMPLAUSIBLE_ML = 1.0

# Written after the volume and the drug: 'at 11:25am', '@1:13pm', '11:39am'.
# The introducer is optional because people leave it out, but the clock has to
# end the cell, so a dose written '2mg' is never read as a time.
INJECTION_TIME = re.compile(
    r"(?:\bat\b\s*|@\s*)?(\d{1,2}:\d{2}\s*(?:[ap]\.?m\.?)?)\s*$", re.I
)


def volume_ml(written: str) -> None | float:
    """
    An injection volume in millilitres, from the units people write.

    `cc` is a millilitre; `uL` is a thousandth of one. A bare number is taken
    as millilitres, which is what the column is headed.

    **EXAMPLE**
    ```python
    volume_ml("250 uL ketamine/xylazine")   # 0.25
    volume_ml(".26cc ket/xyl")              # 0.26
    ```
    """
    amount = number(written)

    if amount is None:
        return None

    unit = str(written)[len(re.match(r"\s*[-+]?[\d.]*", str(written)).group(0)):]

    return amount / 1000 if unit.strip().lower().startswith(("ul", "µl")) else amount


def drug_and_time(written: str) -> tuple[None | str, None | str]:
    """
    What was injected and when, from whatever follows the volume.

    The time is only taken when it is introduced by `at` or `@`, so a drug name
    that happens to contain digits is not mistaken for a clock.

    **EXAMPLE**
    ```python
    drug_and_time("0.33 ket/xyl at 11:25am")   # ("ket/xyl", "11:25am")
    drug_and_time(".26cc")                     # (None, None)
    ```
    """
    rest = re.sub(
        r"^\s*[-+]?[\d.]+\s*(ml|cc|ul|µl)?\s*", "", str(written), flags=re.I
    ).strip()

    found = INJECTION_TIME.search(rest)

    if found is None:
        return rest or None, None

    return rest[: found.start()].strip() or None, found.group(1).strip()


def yes_no(written: str) -> None | bool:
    """
    `True`, `False`, or `None` for a cell that answers something else.

    The column is a yes/no question but people also describe what they did, and
    a description is not an answer to it.
    """
    said = str(written).strip().lower()

    if said in ("y", "yes", "true", "1"):
        return True

    if said in ("n", "no", "false", "0"):
        return False

    return None


def entries(written: str) -> list[str]:
    """One note or flag per entry, as the survey joined them."""
    return [part.strip() for part in str(written).split(SEPARATOR) if part.strip()]


def filled(row: dict, column: str) -> str:
    """A cell's text, or `''` if the column is absent or blank."""
    return (row.get(column) or "").strip()


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def annotate(db, report, where, *, target_type, target_id, key, value, dry_run):
    """One annotation, counted, with a failed write reported rather than raised."""
    if value is None or value == "":
        return

    if dry_run:
        report.written += 1
        return

    try:
        db.add_annotation(
            target_type=target_type, target_id=target_id, key=key, value=value
        )
        report.written += 1
    except Exception as failure:
        report.problem(where, f"{key}: {type(failure).__name__}: {failure}")


def import_session(db, report, row, session_id, dry_run):
    """Everything the `mouse` sheet holds about one session."""
    where = row["folder"]

    def write(key, value):
        annotate(db, report, where, target_type="session", target_id=session_id,
                 key=key, value=value, dry_run=dry_run)

    for column, key in SESSION_TEXT.items():
        text = filled(row, column)

        # One headplate written 'a' and another 'A' is one headplate.
        write(key, text.upper() if key == "headplate" else text)

    for column, key in SESSION_LIST.items():
        for entry in entries(filled(row, column)):
            write(key, entry)

    weight = filled(row, "mouse weight (g)")

    if weight:
        grams = number(weight)

        if grams is None:
            report.problem(where, f"mouse weight {weight!r} is not a number")
        else:
            write("mouse_weight_g", grams)

    pitch = filled(row, "pitch angle")

    if pitch:
        degrees = number(pitch)

        if degrees is None:
            report.problem(where, f"pitch angle {pitch!r} is not a number")
        else:
            write("pitch_angle", degrees)

    correction = filled(row, "left-right-correction")

    if correction:
        answer = yes_no(correction)

        if answer is None:
            # The registry types this as a boolean, and these cells describe
            # what was done instead of answering. Recorded as a note so the
            # observation is not lost, and reported so the mismatch is visible.
            report.problem(
                where,
                f"left-right-correction {correction!r} is not yes or no;"
                f" kept as a note",
            )
            write("note", f"left-right correction: {correction}")
        else:
            write("left_right_correction", answer)

    injection = filled(row, "s.q. injection vol (ml)")

    if injection:
        millilitres = volume_ml(injection)
        substance, given_at = drug_and_time(injection)

        if millilitres is None:
            report.problem(where, f"injection {injection!r} has no volume in it")
            write("note", f"s.q. injection: {injection}")
        elif millilitres >= IMPLAUSIBLE_ML:
            # Recording it would put a dose an order of magnitude too large in
            # a typed field, where it stops looking like a typo.
            report.problem(
                where,
                f"injection {injection!r} reads as {millilitres:g} ml, too much"
                f" for a mouse; kept as a note",
            )
            write("note", f"s.q. injection: {injection}")
        else:
            write("injection_volume", millilitres)

        write("injection_drug", substance)
        write("injection_time", given_at)


def import_experiment(db, report, row, exp_id, where, dry_run):
    """Everything the `expLog` sheet holds about one experiment."""

    def write(key, value):
        annotate(db, report, where, target_type="experiment", target_id=exp_id,
                 key=key, value=value, dry_run=dry_run)

    for column, key in EXPERIMENT_TEXT.items():
        write(key, filled(row, column))

    for column, key in EXPERIMENT_LIST.items():
        for entry in entries(filled(row, column)):
            write(key, entry)

    gain = filled(row, "pmtGain")

    if gain:
        value = number(gain)

        if value is None:
            report.problem(where, f"pmtGain {gain!r} is not a number")
        else:
            write("pmt_gain", value)

    depth = filled(row, "roiDepth_um")

    if depth:
        # The raw text is kept whatever happens: people write '70 um', '~100'
        # and '31 steps', and the number alone would lose what they meant.
        write("fov_depth_raw", depth)
        microns = number(depth)

        if microns is None:
            report.problem(where, f"roiDepth_um {depth!r} has no number in it")
        else:
            write("fov_depth_um", microns)

    lens = filled(row, "objective")

    if lens:
        magnification = number(lens.rstrip("xX"))

        if magnification is None:
            report.problem(where, f"objective {lens!r} is not a magnification")
        elif not dry_run:
            try:
                db.set_objective(exp_id=exp_id, objective=int(magnification))
            except Exception as failure:
                report.problem(where, f"objective: {failure}")


# --------------------------------------------------------------------------- #
# Matching the survey rows to the database
# --------------------------------------------------------------------------- #


def session_rows(survey: Path, wanted: None | set[str]) -> list[dict]:
    """The session rows to import, in folder order."""
    rows = list(csv.DictReader(open(survey / "sessions.csv", newline="")))

    if wanted is not None:
        rows = [row for row in rows if row["folder"] in wanted]

    return sorted(rows, key=lambda row: row["folder"])


def experiment_rows(survey: Path) -> dict[tuple[str, str], list[dict]]:
    """`expLog` rows grouped by the session they belong to."""
    grouped: dict[tuple[str, str], list[dict]] = {}

    for row in csv.DictReader(open(survey / "experiments.csv", newline="")):
        grouped.setdefault((row["session_date"], row["mouse_id"]), []).append(row)

    return grouped


def database_sessions(db) -> dict[str, int]:
    """`session_id` by session folder, for the sessions ingestion created."""
    return {
        path: session_id
        for session_id, path in db.con.execute(
            "SELECT session_id, session_path FROM sessions;"
        )
    }


def database_experiments(db, session_id: int) -> dict[str, int]:
    """`exp_id` by the trailing `e<n>` of the experiment name."""
    found = {}

    for exp_id, name in db.con.execute(
        "SELECT exp_id, exp_name FROM experiments WHERE session_id = ?;",
        [session_id],
    ):
        match = re.search(r"_(e\d+)$", name)

        if match:
            found[match.group(1)] = exp_id

    return found


def import_all(db, survey, sessions, dry_run) -> Report:
    """Walk every listed session, writing what matches and reporting what does not."""
    report = Report()
    by_session = experiment_rows(survey)
    known = database_sessions(db)

    for row in sessions:
        folder = row["folder"]
        session_id = known.get(folder)

        if session_id is None:
            report.problem(folder, "no such session in the database; ingest it first")
            continue

        import_session(db, report, row, session_id, dry_run)

        experiments = database_experiments(db, session_id)
        logged = by_session.get((row["session_date"], row["mouse_id"]), [])
        seen: dict[str, dict] = {}

        for entry in logged:
            written = filled(entry, "expNum")

            if is_zstack(entry):
                report.zstack(folder, f"expNum {written!r}, {filled(entry, 'goal')}")
                continue

            name = experiment_number(written)

            if name is None:
                report.problem(folder, f"expNum {written!r} names no experiment")
                continue

            if name in seen:
                report.problem(
                    folder, f"two log rows both say {name}; neither imported"
                )
                seen[name] = None
                continue

            seen[name] = entry

        for name, entry in seen.items():
            if entry is None:
                continue

            exp_id = experiments.get(name)

            if exp_id is None:
                report.problem(
                    folder, f"log has {name} but no such experiment was ingested"
                )
                continue

            import_experiment(db, report, entry, exp_id, f"{folder}/{name}", dry_run)

        for name in sorted(set(experiments) - set(seen)):
            report.problem(folder, f"{name} was ingested but the log does not list it")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("survey_folder", type=Path)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    wanted = None

    if args.sessions:
        lines = (line.strip() for line in args.sessions.read_text().splitlines())
        wanted = {line for line in lines if line and not line.startswith("#")}

    db = Database(args.main_folder, project=args.project)
    sessions = session_rows(args.survey_folder, wanted)

    print(f"{len(sessions)} sessions from {args.survey_folder}")

    if args.dry_run:
        print("DRY RUN -- nothing is written")

    report = import_all(db, args.survey_folder, sessions, args.dry_run)

    print(f"\n{report.written} annotations {'would be ' if args.dry_run else ''}written")

    if report.zstacks:
        print(f"\n{len(report.zstacks)} z-stack rows, not imported (no experiment yet):")

        for entry in report.zstacks:
            print(f"  - {entry}")

    if report.problems:
        print(f"\n{len(report.problems)} problems:")

        for problem in report.problems:
            print(f"  - {problem}")

    print("\nNot imported by design:")

    for column, why in SKIPPED.items():
        print(f"  {column}: {why}")

    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
