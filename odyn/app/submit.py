"""
Turning a draft into database rows: the app checks and submits it, the
cluster ingests it.

The checks come in two parts, saying which problems block and which are only
worth knowing:
- `check`: what the computer submitting can tell, from the form, the database
and its own folders (the olfactometer's, the recordings still on it). Quick,
and it reads nothing from the recordings on the share.
- `check_files`: what only the recordings on the share can tell (their TIFFs,
sync files, the acquisition numbers the conditions name).

`ingest` runs both, then does the day's work in order: the session row, its
annotations, each loop (ingested, annotated, its vials recorded, its sync file
processed), then the grabs and z-stacks (each TIFF ingested and annotated, the
grabs' sync files processed, each z-stack linked to its loop). The cluster
runs it once the recordings are on the share (`odyn.app.submissions`).

`submit` is the app's part: it copies the files only this computer has, runs
`check`, puts the draft in the cluster's queue, and moves it aside. It writes
nothing to the database.

Every step that writes is a recorded `Database` call, and ingesting skips what
is already in, so an ingest that failed halfway can be run again. What each
part or file became is remembered in the draft (`submitted_id`).

**USAGE**
```python
result = submit(db, draft)          # the app: refuses while `check` blocks
problems = check(db, draft) + check_files(db, draft)
result = ingest(db, draft)          # the cluster: refuses while either blocks
```

No Qt in here: the window calls these from a worker thread and shows what
comes back.
"""

from __future__ import annotations

import re

from collections.abc import Container
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..database import OLFACTOMETER_FOLDER, SYNC_FOLDER, read_tiff
from ..utils import INFO_FOLDER, logger
from .draft import FILE_SETTINGS, KINDS, PART_SETTINGS, Draft
from .fields import annotation_fields, form_fields, missing_required
from .olfactometer import OlfactometerFolders, copy_for_loop, loop_window
from .recordings import pending, sync_pending

RAW_FOLDER = "raw"
EVENTS_PATTERN = "*Events.csv"


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a draft, and whether it stops the submit."""

    where: str  # 'session', 'e1', 'vials', ...
    message: str
    blocking: bool = True

    def __str__(self) -> str:
        kind = "" if self.blocking else ". (Not required to submit.)"

        return f"{self.where}: {self.message}{kind}"


@dataclass
class Written:
    """What one submit put in the database."""

    session_id: None | int = None
    loops: dict[str, int] = field(default_factory=dict)  # by folder name
    grabs: dict[str, int] = field(default_factory=dict)  # by TIFF name
    zstacks: dict[str, int] = field(default_factory=dict)  # by TIFF name
    annotations: int = 0
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    process_failed: dict[str, str] = field(default_factory=dict)


@dataclass
class SubmitResult:
    """
    The outcome of `submit` or `ingest`: what was wrong, and where the draft
    went (`queued`, by `submit`) or what it became (`written`, by `ingest`).
    """

    label: str
    problems: list[Problem] = field(default_factory=list)
    written: None | Written = None
    error: None | str = None
    queued: None | Path = None

    @property
    def blocked(self) -> bool:
        return any(problem.blocking for problem in self.problems)

    @property
    def submitted(self) -> bool:
        done = self.written is not None or self.queued is not None

        return done and self.error is None


# The Check tab's report is colored as the log is, with terminal codes (the
# panel shows them as colors): red for what stops a submit, yellow for what
# does not, blue for the report's own lines
RED, YELLOW, BLUE, RESET = "\033[1;31m", "\033[1;33m", "\033[1;34m", "\033[0m"

# The report's left column is at least this wide
LABEL_WIDTH = 7


def report(session: str, problems: list[Problem], parts: list[str], result: str) -> str:
    """
    The Check tab's report on `session`: a header counting errors (problems
    that stop a submit) and warnings, then the problems in one block per page
    (the user's, the session's with `vials`, then each experiment in the order of
    `parts`), then `result` on the last line.

    **USAGE**
    ```python
    report("m3 2026-10-07", problems, ["e1", "grabs"], f"ready to submit {CHECK}")
    ```
    """
    blocks: dict[str, list[Problem]] = {}

    for problem in problems:
        page = problem.where.split("/")[0]
        blocks.setdefault("session" if page == "vials" else page, []).append(problem)

    order = ["user", "session", *parts]
    groups = sorted(
        blocks, key=lambda group: order.index(group) if group in order else len(order)
    )
    width = max([LABEL_WIDTH, *(len(p.where.split("/")[0]) for p in problems)])

    def line(label: str, color: str, text: str) -> str:
        return f"{color}{label:>{width}}{RESET}  {text}"

    errors = sum(problem.blocking for problem in problems)
    warnings = len(problems) - errors
    counts = (
        f"({errors} {RED}error{'' if errors == 1 else 's'}{RESET}, "
        f"{warnings} {YELLOW}warning{'' if warnings == 1 else 's'}{RESET})"
    )
    lines = [line("CHECKED", BLUE, f"{session} {counts}"), ""]

    for group in groups:
        for problem in blocks[group]:
            page, _, file = problem.where.partition("/")
            text = problem.message + (f" ({file})" if file else "")
            lines.append(line(page, RED if problem.blocking else YELLOW, text))

        lines.append("")

    lines.append(line("RESULT", BLUE, result))

    return "\n".join(lines)


def running(session: str, doing: str) -> str:
    """The Check tab's line while a check or a submit runs."""
    return f"{BLUE}{'RUNNING':>{LABEL_WIDTH}}{RESET}  {session} ({doing}...)"


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def check(
    db, draft: Draft, olfactometer: None | OlfactometerFolders = None
) -> list[Problem]:
    """
    What is wrong with a draft that this computer can tell, blocking problems
    first.

    Looks at the form, at the database (is the one submitting a user, is the
    mouse registered, are the vials real), at whether each part's folder is on
    the share, and, given `olfactometer`, at the olfactometer's folders on this
    computer (are there files to copy). The recordings themselves are left to
    `check_files`. Writes nothing.
    """
    problems: list[Problem] = []

    # The QC page lists a session under who submitted it, so that has to be
    # someone on the Users page (`db.user` is the computer's login until then)
    users = db.users
    active = users.loc[~users["retired"].astype(bool), "user_name"]

    if db.user not in set(active):
        problems.append(Problem("user", "didn't pick a user yet (Users page)"))

    if draft.mouse_id is None:
        problems.append(Problem("session", "didn't pick a mouse yet"))
        return problems

    if draft.mouse_id not in db.mice.index:
        problems.append(Problem("session", f"mouse {draft.mouse_id} is not registered"))

    session_fields = form_fields(db, "session")
    problems += _field_problems(
        "session",
        session_fields,
        {**draft.session, "mouse_id": draft.mouse_id, "session_date": draft.date},
    )

    folder = Path(db.main_folder) / draft.folder
    local_only = pending(draft)

    if not draft.parts:
        problems.append(Problem("session", "no experiments listed", blocking=False))

    elif not folder.is_dir():
        problems.append(
            Problem(
                "session",
                f"folder '{draft.folder}' is not on the server (start copying the "
                "TIFFs first)",
            )
        )

    loop_fields = form_fields(db, "loop")
    loops = {str(e["name"]) for e in draft.parts if e.get("kind") == "loop"}

    for entry in draft.parts:
        name = str(entry.get("name", ""))
        kind = entry.get("kind")
        where = name or "experiment"

        if not name:
            problems.append(Problem("experiment", "the experiment has no name"))
            continue

        if kind not in KINDS:
            problems.append(Problem(where, f"kind must be one of {', '.join(KINDS)}"))

        if entry.get("objective") in (None, ""):
            problems.append(Problem(where, "didn't pick an objective"))

        if kind == "loop":
            problems += _field_problems(where, loop_fields, entry)
            problems += _condition_problems(where, entry)

        else:
            problems += _file_problems(db, entry, loops)

        if kind == "loop" and entry.get("submitted_id") is not None:
            continue  # already in the database; the files were checked then

        if name in local_only:
            problems.append(
                Problem(
                    where,
                    f"sync file {local_only[name]['file']} was not copied to the "
                    "server (submit copies it)",
                    blocking=False,
                )
            )

        if folder.is_dir() and not (folder / name).is_dir():
            problems.append(Problem(where, f"folder '{name}' is not on the server yet"))

        if kind == "loop" and (folder / name).is_dir():
            problems += _olfactometer_problems(
                where, entry, folder / name, db, olfactometer
            )

    if draft.vial_ids:
        known = set(db.vials.index)
        missing = sorted(set(draft.vial_ids) - known)

        if missing:
            problems.append(Problem("vials", f"no vials with ids {_spans(missing)}"))

        else:
            panels = set(db.vials.loc[draft.vial_ids, "panel_id"])

            if len(panels) > 1:
                problems.append(
                    Problem("vials", "the vials come from more than one panel")
                )

    problems.sort(key=lambda problem: not problem.blocking)

    return problems


def _field_problems(where: str, fields, values: dict) -> list[Problem]:
    """Values that do not read as their type, and required ones left empty."""
    problems = []

    for field_ in fields:
        if not field_.editable or field_.key not in values:
            continue

        try:
            field_.parse(values[field_.key])
        except ValueError as error:
            problems.append(Problem(where, str(error)))

    for field_ in missing_required(fields, values):
        if field_.key == "objective":
            continue  # blocking above, since ingestion needs it

        problems.append(
            Problem(where, f"{field_.label} is not filled in", blocking=False)
        )

    return problems


def _file_problems(db, entry: dict, loops: set[str]) -> list[Problem]:
    """
    What is wrong with the notes on a grabs or z-stacks part's TIFFs: values
    that do not read as their type, and a z-stack's loop that is not in the
    draft.
    """
    name, kind = str(entry["name"]), entry["kind"]
    fields = annotation_fields(db, kind)
    problems = []

    for file, noted in entry.get("files", {}).items():
        where = f"{name}/{file}"
        problems += _field_problems(where, fields, noted)
        loop = noted.get("loop")

        if loop is not None and loop not in loops:
            problems.append(Problem(where, f"the z-stack loop {loop} is not listed"))

    return problems


def list_tiffs(main_folder: Path, folder: str, kind: str) -> pd.DataFrame:
    """
    The TIFFs in a grabs or z-stacks folder's `raw/`, one row each with what
    its header says: when it started, and its frames and z (a grab) or its
    stack, frames per slice and powers (a z-stack). Empty when the folder is
    not there.
    """
    columns = {
        "grab": ["file", "started", "frames", "z_um"],
        "zstack": [
            "file",
            "started",
            "slices",
            "frames_per_slice",
            "z_start_um",
            "z_end_um",
            "power_start_pct",
            "power_end_pct",
        ],
    }[kind]
    raw = Path(main_folder) / folder / RAW_FOLDER
    rows = []

    try:
        paths = sorted(raw.glob("[!.]?*.tif"))
    except OSError:
        paths = []

    for path in paths:
        row = {"file": path.name}

        try:
            read = read_tiff(path, Path(main_folder))
        except Exception as error:  # a file still being copied, say
            logger.warning(f"Could not read '{path.name}': {error}")
            read = None

        if read is not None:
            recording, tiff = read
            stack = recording["stack"] or {}
            row.update(
                started=tiff["acq_start"].strftime("%H:%M:%S"),
                frames=recording["frames_per_slice"],
                frames_per_slice=recording["frames_per_slice"],
                z_um=recording["z_um"],
                slices=stack.get("slice_count"),
                z_start_um=stack.get("z_start_um"),
                z_end_um=stack.get("z_end_um"),
                power_start_pct=stack.get("power_start_pct"),
                power_end_pct=stack.get("power_end_pct"),
            )

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def acquisition_number(path: str | Path) -> None | int:
    """
    The number ScanImage ends a TIFF's name with (`..._00003.tif` is 3), which
    is how the rig counts acquisitions.
    """
    match = re.search(r"_(\d+)$", Path(path).stem)

    return None if match is None else int(match.group(1))


def _tiff_numbers(raw: Path) -> None | set[int]:
    """The acquisition numbers in a raw folder, or None while it has none."""
    numbers = {acquisition_number(p) for p in raw.glob("[!.]?*.tif")} - {None}

    return numbers or None


def _condition_problems(where: str, entry: dict) -> list[Problem]:
    """
    Conditions that cannot be written: an end before its start, and ranges
    that overlap (an acquisition has one condition).
    """
    conditions = entry.get("conditions") or []

    if not conditions:
        return []

    if entry.get("kind", "loop") != "loop":
        return [Problem(where, "conditions can only be applied to loops")]

    problems = []
    taken: dict[int, str] = {}

    for condition in conditions:
        start, end = int(condition["start"]), int(condition["end"])
        span = f"{condition['condition']} {start} to {end}"

        if end < start:
            problems.append(Problem(where, f"condition {span} ends before it starts"))
            continue

        overlap = sorted(set(range(start, end + 1)) & set(taken))

        if overlap:
            problems.append(
                Problem(
                    where,
                    f"condition {span} overlaps {taken[overlap[0]]} at "
                    f"acquisition {overlap[0]}",
                )
            )

        taken.update({n: span for n in range(start, end + 1)})

    return problems


def _condition_tiff_problems(
    where: str, entry: dict, numbers: None | set[int]
) -> list[Problem]:
    """Conditions naming acquisition numbers that no TIFF has (once some are there)."""
    if numbers is None:
        return []

    problems = []

    for condition in entry.get("conditions") or []:
        start, end = int(condition["start"]), int(condition["end"])
        missing = sorted(set(range(start, end + 1)) - numbers)

        if missing:
            span = f"{condition['condition']} {start} to {end}"
            problems.append(
                Problem(
                    where,
                    f"condition {span} names acquisitions with no TIFF: "
                    f"{_spans(missing)}",
                )
            )

    return problems


def _spans(numbers: list[int]) -> str:
    """`[3, 4, 5, 9]` as `'3-5, 9'`."""
    spans: list[list[int]] = []

    for n in numbers:
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])

    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in spans)


def _olfactometer_problems(
    where: str,
    entry: dict,
    folder: Path,
    db,
    olfactometer: None | OlfactometerFolders,
) -> list[Problem]:
    """
    What a loop's olfactometer files are missing. Submit copies the runs that
    started while the loop's sync file was recording, so a loop is fine when
    it has its Events files already or there are runs to copy; and it then
    needs the day's program log, in the loop or on this computer.
    """
    window = loop_window(entry)
    runs = (
        olfactometer.runs_during(*window)
        if olfactometer is not None and window is not None
        else []
    )

    if not runs and not any(folder.rglob(EVENTS_PATTERN)):
        searched = olfactometer is not None and window is None
        why = " (no recording to find its runs by)" if searched else ""
        return [
            Problem(where, f"no olfactometer Events file found{why}", blocking=False)
        ]

    day = window[0] if window is not None else None
    logs = list((folder / OLFACTOMETER_FOLDER / "log").glob("Program_*.txt"))

    if day is not None and not logs:
        name = f"Program_{day:%Y%m%d}.txt"
        shared = Path(db.main_folder) / INFO_FOLDER / f"{day:%Y%m}" / name
        here = olfactometer is not None and olfactometer.program_log(day).is_file()

        if not here and not shared.is_file():
            return [Problem(where, f"no olfactometer log for {day:%Y-%m-%d} ({name})")]

    return []


def _folder_problems(where: str, folder: Path, kind) -> list[Problem]:
    """What a part folder on the share is missing, or has too many of."""
    problems = []

    if not folder.is_dir():
        return [Problem(where, f"folder '{folder.name}' is not on the server yet")]

    tiffs = sorted((folder / RAW_FOLDER).glob("[!.]?*.tif"))

    if not tiffs:
        problems.append(Problem(where, f"no TIFFs in '{folder.name}/{RAW_FOLDER}'"))

    # A grab's sync recording is optional, and its files add up

    if kind == "loop":
        syncs = [
            p for p in (folder / SYNC_FOLDER).glob("*.h5") if not p.name.startswith(".")
        ]

        if not syncs:
            problems.append(
                Problem(
                    where,
                    f"no sync file in '{folder.name}/{SYNC_FOLDER}' (the odor "
                    "frames can be added later)",
                    blocking=False,
                )
            )

        elif len(syncs) > 1:
            problems.append(
                Problem(
                    where, f"{len(syncs)} sync files in '{folder.name}/{SYNC_FOLDER}'"
                )
            )

    return problems


def check_files(db, draft: Draft) -> list[Problem]:
    """
    What is wrong between a draft and its recordings on the share, blocking
    problems first: part folders that are missing, with no TIFFs or with
    more than one sync file, and conditions or notes that name TIFFs that are
    not there. It lists the share's folders, which is slow from far away, so
    it runs where the data is (when the cluster ingests). Writes nothing.
    """
    folder = Path(db.main_folder) / draft.folder
    problems: list[Problem] = []

    for entry in draft.parts:
        name, kind = str(entry.get("name", "")), entry.get("kind")
        part = folder / name

        if not name or kind not in KINDS:
            continue  # check says so

        # Conditions are written again on a resubmit, the loop is not
        if kind == "loop":
            problems += _condition_tiff_problems(
                name, entry, _tiff_numbers(part / RAW_FOLDER)
            )

            if entry.get("submitted_id") is not None:
                continue

        if not part.is_dir():
            problems.append(Problem(name, f"folder '{name}' is not on the server yet"))
            continue

        problems += _folder_problems(name, part, kind)

        if kind != "loop":
            on_share = {p.name for p in (part / RAW_FOLDER).glob("[!.]?*.tif")}
            problems += [
                Problem(f"{name}/{file}", "not in the server's folder", blocking=False)
                for file in entry.get("files", {})
                if file not in on_share
            ]

    problems.sort(key=lambda problem: not problem.blocking)

    return problems


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


def submit(
    db,
    draft: Draft,
    *,
    force: bool = False,
    busy: Container[str] = (),
    olfactometer: None | OlfactometerFolders = None,
) -> SubmitResult:
    """
    Copy what only this computer has, then put the draft in the cluster's
    queue as `db.user`'s (`submissions.enqueue`) and move it aside. The
    cluster ingests it once the recordings are on the share.

    The copies are the sync recordings that are only on this computer (any in
    `busy` are still being written, and are left) and, given `olfactometer`,
    each loop's olfactometer files (see `copy_olfactometer`). A copy that
    fails blocks, as a blocking problem of `check` does, unless `force`.
    """
    from .submissions import enqueue  # it imports this module

    recording = [
        name for name, record in pending(draft).items() if record["local"] in busy
    ]
    not_copied = sync_pending(Path(db.main_folder), draft, busy)
    olfactometer_failed = (
        {} if olfactometer is None else copy_olfactometer(db, draft, olfactometer)
    )
    found = [
        *(Problem(name, "still recording (stop it first)") for name in recording),
        *(
            Problem(name, f"could not copy the sync file to the server: {why}")
            for name, why in not_copied.items()
        ),
        *(
            Problem(name, f"could not copy the olfactometer files: {why}")
            for name, why in olfactometer_failed.items()
        ),
        *check(db, draft, olfactometer),
    ]
    result = SubmitResult(
        draft.label, sorted(found, key=lambda problem: not problem.blocking)
    )

    if result.blocked and not force:
        return result

    try:
        result.queued = enqueue(draft, db.main_folder, db.project, user=db.user)

    except OSError as error:
        logger.exception(f"Submit of {draft.label} failed")
        result.error = f"could not put it in the cluster's queue: {error}"
        return result

    draft.archive()

    return result


def ingest(
    db,
    draft: Draft,
    *,
    force: bool = False,
    olfactometer: None | OlfactometerFolders = None,
) -> SubmitResult:
    """
    Put a finished draft in the database.

    Refuses while `check` or `check_files` finds a blocking problem, unless
    `force`.
    Otherwise creates (or finds) the session, writes its annotations, then
    each loop and the grabs and z-stacks (see the module's description).
    Steps already done by an earlier attempt are skipped, so the call is safe
    to repeat after fixing what stopped it. Leaves the draft file where it is.
    """
    # Both checks report a part folder that is not there; it is said once
    found = [*check(db, draft, olfactometer), *check_files(db, draft)]
    result = SubmitResult(
        draft.label,
        sorted(dict.fromkeys(found), key=lambda problem: not problem.blocking),
    )

    if result.blocked and not force:
        return result

    written = Written()

    try:
        _submit(db, draft, written)

    except Exception as error:  # a failed step is reported, not raised
        logger.exception(f"Submit of {draft.label} failed")
        result.error = described(error)

    result.written = written

    return result


def described(error: Exception) -> str:
    """
    An error as the Check report shows it, without a final period. The ones
    ODyn raises to explain a refusal (`RuntimeError`, `ValueError`, `OSError`)
    read as they are; any other keeps its type, which is all a bare
    `KeyError: 'x'` says.
    """
    text = str(error).rstrip().rstrip(".")

    if isinstance(error, (RuntimeError, ValueError, OSError)) and text:
        return text

    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def copy_olfactometer(
    db, draft: Draft, olfactometer: OlfactometerFolders
) -> dict[str, str]:
    """
    Copy into each loop folder not submitted yet the olfactometer files it
    needs (`olfactometer.copy_for_loop`), for the loops with a recording and a
    folder on the share. Returns why it failed, by loop name.
    """
    failed = {}
    session = Path(db.main_folder) / draft.folder

    for entry in draft.parts:
        if entry.get("kind") != "loop" or entry.get("submitted_id") is not None:
            continue

        name = str(entry["name"])
        window = loop_window(entry)

        if window is None or not (session / name).is_dir():
            continue

        try:
            copied = copy_for_loop(olfactometer, session / name, window)
            logger.info(
                f"Copied the olfactometer files of {name}: {len(copied.runs)} runs, "
                f"log {copied.log or 'not found'}, {len(copied.config)} config files."
            )

        except OSError as error:
            failed[name] = str(error)

    return failed


def _submit(db, draft: Draft, written: Written) -> None:
    assert draft.mouse_id is not None

    session_id = draft.session_id

    if session_id is None:
        session_id = db.session_for(draft.mouse_id, draft.date)

    if session_id is None:
        session_id = db.add_session(mouse_id=draft.mouse_id, session_date=draft.date)

    draft.identify(session_id=session_id)
    written.session_id = session_id

    values = _annotation_values(db, "session", draft.session)

    if values:
        db.add_annotations(target_type="session", target_id=session_id, values=values)
        written.annotations += len(values)

    loops = [e for e in draft.parts if e.get("kind", "loop") == "loop"]
    others = [e for e in draft.parts if e.get("kind") in ("grab", "zstack")]

    for entry in loops:
        _submit_loop(db, draft, entry, session_id, written)

    # After the loops, so a z-stack can be linked to the loop it was for
    for entry in others:
        _submit_files(db, draft, entry, session_id, written)


def _submit_loop(db, draft: Draft, entry: dict, session_id: int, written) -> None:
    """Ingest one loop, annotate it, record its vials and process its sync file."""
    name = str(entry["name"])
    loop_path = f"{draft.folder}/{name}"
    loop_id = entry.get("submitted_id")

    if loop_id is None:
        loop_id = db.add_loop(
            session_id=session_id,
            loop_path=loop_path,
            objective=int(entry["objective"]),
        )

    if loop_id is None:
        # Already in the database (an earlier attempt, or someone else's)
        rows = db.from_query(
            "SELECT loop_id FROM loops WHERE session_id = ? AND loop_path = ?;",
            [session_id, loop_path],
        )
        loop_id = None if rows.empty else int(rows["loop_id"].iloc[0])

    if loop_id is None:
        written.skipped.append(name)
        return

    draft.set_part(name, submitted_id=int(loop_id))
    written.loops[name] = int(loop_id)

    values = _annotation_values(db, "loop", entry)

    if values:
        db.add_annotations(target_type="loop", target_id=loop_id, values=values)
        written.annotations += len(values)

    written.annotations += _write_conditions(db, draft, name, loop_id, entry)

    if draft.vial_ids:
        db.set_loop_vials(loop_id=loop_id, vial_ids=draft.vial_ids)

    if _has_sync(db, loop_path):
        # The loop is in either way; a sync file that is refused (a burst count
        # that does not match, say) is reported and can be rerun once the
        # file is sorted out.
        try:
            db.process_sync(loop_id=loop_id)
            written.processed.append(name)

        except Exception as error:
            logger.exception(f"Processing the sync file of {name} failed")
            written.process_failed[name] = described(error)


def _submit_files(db, draft: Draft, entry: dict, session_id: int, written) -> None:
    """
    Ingest every TIFF of the grabs or z-stacks folder (those already in are
    skipped), annotate the ones with notes, link each z-stack to its loop,
    and process the grabs' sync files.
    """
    name, kind = str(entry["name"]), entry["kind"]
    folder = f"{draft.folder}/{name}"
    add = db.add_grabs if kind == "grab" else db.add_zstacks
    add(session_id=session_id, folder=folder, objective=int(entry["objective"]))

    rows = db.from_query(
        f"SELECT {kind}_id AS id, raw_path FROM {kind}s WHERE session_id = ?;",
        [session_id],
    )
    ids = {Path(path).name: int(i) for i, path in zip(rows["id"], rows["raw_path"])}
    found = written.grabs if kind == "grab" else written.zstacks
    found.update(ids)
    notes = draft.files(name)

    for file, noted in notes.items():
        if file not in ids:
            continue  # not on the share (check said so)

        row_id = ids[file]
        values = _annotation_values(db, kind, noted)

        if values:
            db.add_annotations(target_type=kind, target_id=row_id, values=values)
            written.annotations += len(values)

        loop = noted.get("loop")

        if kind == "zstack" and loop is not None and loop in written.loops:
            db.set_zstack_loop(zstack_id=row_id, loop_id=written.loops[loop])

        draft.set_file(name, file, submitted_id=row_id)

    if kind != "grab":
        return

    for record in entry.get("recordings", []):
        if not record.get("copied") or record.get("replaced"):
            continue

        try:
            db.process_grabs_sync(session_id=session_id, sync_path=record["share"])
            written.processed.append(f"{name}/{record['file']}")

        except Exception as error:
            logger.exception(f"Processing {record['file']} failed")
            written.process_failed[f"{name}/{record['file']}"] = described(error)


def _write_conditions(db, draft: Draft, name: str, loop_id: int, entry: dict) -> int:
    """
    Each acquisition in a condition's range gets it as an annotation. Only
    when the conditions changed since the last submit, so a resubmit does
    not write them all again. Returns how many were written.
    """
    conditions = entry.get("conditions") or []

    if not conditions or conditions == entry.get("conditions_submitted"):
        return 0

    rows = db.from_query(
        "SELECT acq_id, raw_path FROM acquisitions WHERE loop_id = ?;", [int(loop_id)]
    )
    acq_ids = {
        acquisition_number(path): int(acq_id)
        for acq_id, path in zip(rows["acq_id"], rows["raw_path"])
    }
    written = 0

    for condition in conditions:
        for number in range(int(condition["start"]), int(condition["end"]) + 1):
            if number in acq_ids:  # (check refuses numbers with no TIFF)
                db.add_annotations(
                    target_type="acquisition",
                    target_id=acq_ids[number],
                    values={"condition": condition["condition"]},
                )
                written += 1

    draft.set_part(name, conditions_submitted=[dict(c) for c in conditions])

    return written


def _annotation_values(db, applies_to: str, values: dict) -> dict:
    """The typed values a draft section holds for the registry's keys."""
    parsed = {}

    for field_ in form_fields(db, applies_to):
        if field_.key in (*PART_SETTINGS, *FILE_SETTINGS) or not field_.editable:
            continue

        if field_.key in ("mouse_id", "session_date", "objective"):
            continue  # columns, not annotations

        if field_.key in values:
            parsed[field_.key] = field_.parse(values[field_.key])

    return parsed


def _has_sync(db, loop_path: str) -> bool:
    folder = Path(db.main_folder) / loop_path / SYNC_FOLDER

    return folder.is_dir() and any(
        p for p in folder.glob("*.h5") if not p.name.startswith(".")
    )
