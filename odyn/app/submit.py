"""
Turning a draft into database rows: check first, then submit.

`check` looks at a draft and the folders it points to and lists what is wrong,
saying which problems block a submit and which are only worth knowing.
`submit` runs the checks, then does the day's work in order: the session row,
its annotations, each experiment folder (ingested, annotated, its vials
recorded, its sync file decoded), and finally moves the draft aside.

Every step that writes is a recorded `Database` call, and a step that has
already been done is remembered in the draft (`submitted_exp_id`), so a
submit that failed halfway can be run again without ingesting twice.

**USAGE**
```python
problems = check(db, draft)
result = submit(db, draft)          # refuses while a blocking problem stands
result = submit(db, draft, force=True)
```

No Qt in here: the window calls these from a worker thread and shows what
comes back.
"""

from __future__ import annotations

import re

from collections.abc import Container
from dataclasses import dataclass, field
from pathlib import Path

from ..database import SYNC_FOLDER
from ..utils import logger
from .draft import EXPERIMENT_SETTINGS, EXPERIMENT_TYPES, Draft
from .fields import form_fields, missing_required
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
        kind = "" if self.blocking else " (can submit anyway)"

        return f"{self.where}: {self.message}{kind}"


@dataclass
class Written:
    """What one submit put in the database."""

    session_id: None | int = None
    experiments: dict[str, int] = field(default_factory=dict)
    annotations: int = 0
    decoded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    decode_failed: dict[str, str] = field(default_factory=dict)


@dataclass
class SubmitResult:
    """The outcome of `submit`: what was written, and what was wrong."""

    label: str
    problems: list[Problem] = field(default_factory=list)
    written: None | Written = None
    error: None | str = None

    @property
    def blocked(self) -> bool:
        return any(problem.blocking for problem in self.problems)

    @property
    def submitted(self) -> bool:
        return self.written is not None and self.error is None


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def check(db, draft: Draft) -> list[Problem]:
    """
    Everything wrong with a draft, blocking problems first.

    Looks at the database (is the mouse registered, are the vials real) and at
    the share (are the recordings there), but writes nothing.
    """
    problems: list[Problem] = []

    if draft.mouse_id is None:
        problems.append(Problem("session", "No mouse picked."))
        return problems

    if draft.mouse_id not in db.mice.index:
        problems.append(
            Problem("session", f"Mouse {draft.mouse_id} is not registered.")
        )

    session_fields = form_fields(db, "session")
    problems += _field_problems(
        "session",
        session_fields,
        {**draft.session, "mouse_id": draft.mouse_id, "session_date": draft.date},
    )

    folder = Path(db.main_folder) / draft.folder
    local_only = pending(draft)

    if not draft.experiments:
        problems.append(Problem("session", "No experiments listed.", blocking=False))

    elif not folder.is_dir():
        problems.append(
            Problem(
                "session",
                f"Folder '{draft.folder}' is not on the share yet. Copy the "
                "recordings over first.",
            )
        )

    experiment_fields = form_fields(db, "experiment")

    for entry in draft.experiments:
        name = str(entry.get("name", ""))
        where = name or "experiment"

        if not name:
            problems.append(Problem("experiment", "An experiment has no name."))
            continue

        if entry.get("exp_type") not in EXPERIMENT_TYPES:
            problems.append(Problem(where, f"Type must be one of {EXPERIMENT_TYPES}."))

        if entry.get("objective") in (None, ""):
            problems.append(Problem(where, "No objective set."))

        problems += _field_problems(where, experiment_fields, entry)
        problems += _condition_problems(
            where, entry, _tiff_numbers(folder / name / RAW_FOLDER)
        )

        if entry.get("submitted_exp_id") is not None:
            continue  # already in the database; the files were checked then

        if name in local_only:
            problems.append(
                Problem(
                    where,
                    f"Sync file {local_only[name]['file']} is only on this computer "
                    "so far. Submit copies it to the share first.",
                    blocking=False,
                )
            )

        if folder.is_dir():
            problems += _folder_problems(
                where, folder / name, entry.get("exp_type"), name in local_only
            )

    if draft.vial_ids:
        known = set(db.vials.index)
        missing = sorted(set(draft.vial_ids) - known)

        if missing:
            problems.append(Problem("vials", f"No vials with ids {missing}."))

        else:
            panels = set(db.vials.loc[draft.vial_ids, "panel_id"])

            if len(panels) > 1:
                problems.append(
                    Problem("vials", "The vials come from more than one panel.")
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
            Problem(where, f"{field_.label} is not filled in.", blocking=False)
        )

    return problems


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


def _condition_problems(where: str, entry: dict, numbers: None | set[int]):
    """
    Conditions that cannot be written: an end before its start, ranges that
    overlap (an acquisition has one condition), and, once the TIFFs are
    there, numbers that no TIFF has.
    """
    conditions = entry.get("conditions") or []

    if not conditions:
        return []

    if entry.get("exp_type", "loop") != "loop":
        return [Problem(where, "Conditions are only noted for loop experiments.")]

    problems = []
    taken: dict[int, str] = {}

    for condition in conditions:
        start, end = int(condition["start"]), int(condition["end"])
        span = f"{condition['condition']} {start} to {end}"

        if end < start:
            problems.append(Problem(where, f"Condition {span} ends before it starts."))
            continue

        overlap = sorted(set(range(start, end + 1)) & set(taken))

        if overlap:
            problems.append(
                Problem(
                    where,
                    f"Condition {span} overlaps {taken[overlap[0]]} at "
                    f"acquisition {overlap[0]}: each acquisition has one condition.",
                )
            )

        taken.update({n: span for n in range(start, end + 1)})

        if numbers is not None:
            missing = sorted(set(range(start, end + 1)) - numbers)

            if missing:
                problems.append(
                    Problem(
                        where,
                        f"Condition {span} names acquisitions with no TIFF: "
                        f"{_spans(missing)}.",
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


def _folder_problems(
    where: str, folder: Path, exp_type, sync_pending: bool = False
) -> list[Problem]:
    """What an experiment folder is missing (its sync file may be on its way)."""
    problems = []

    if not folder.is_dir():
        return [Problem(where, f"Folder '{folder.name}' is not on the share yet.")]

    tiffs = sorted((folder / RAW_FOLDER).glob("[!.]?*.tif"))

    if not tiffs:
        problems.append(Problem(where, f"No TIFFs in '{folder.name}/{RAW_FOLDER}'."))

    if exp_type == "loop":
        if not any(folder.rglob(EVENTS_PATTERN)):
            problems.append(
                Problem(where, "No olfactometer Events file found.", blocking=False)
            )

        syncs = [
            p for p in (folder / SYNC_FOLDER).glob("*.h5") if not p.name.startswith(".")
        ]

        if not syncs and not sync_pending:
            problems.append(
                Problem(
                    where,
                    f"No sync file in '{folder.name}/{SYNC_FOLDER}'; the odor "
                    "frames stay unknown until one is decoded.",
                    blocking=False,
                )
            )

        elif len(syncs) > 1:
            problems.append(
                Problem(
                    where, f"{len(syncs)} sync files in '{folder.name}/{SYNC_FOLDER}'."
                )
            )

    return problems


# --------------------------------------------------------------------------- #
# Submitting
# --------------------------------------------------------------------------- #


def submit(
    db, draft: Draft, *, force: bool = False, busy: Container[str] = ()
) -> SubmitResult:
    """
    Put a finished draft in the database.

    First copies to the share the sync recordings that are only on this
    computer (any in `busy` are still being written, and are left); one that
    cannot be copied blocks. Then refuses while `check` finds a blocking
    problem, unless `force`. Otherwise creates (or finds) the session, writes
    its annotations, and for each experiment folder ingests it, annotates it,
    records its vials and decodes its sync file. Steps already done by an
    earlier attempt are skipped, so the call is safe to repeat after fixing
    what stopped it.
    """
    recording = [
        name for name, record in pending(draft).items() if record["local"] in busy
    ]
    not_copied = sync_pending(Path(db.main_folder), draft, busy)
    result = SubmitResult(
        draft.label,
        [Problem(name, "Still recording: stop it first.") for name in recording]
        + [
            Problem(name, f"Could not copy the sync file to the share: {why}")
            for name, why in not_copied.items()
        ]
        + check(db, draft),
    )

    if result.blocked and not force:
        return result

    written = Written()

    try:
        _submit(db, draft, written)

    except Exception as error:  # a failed step is reported, not raised
        logger.exception(f"Submit of {draft.label} failed")
        result.error = f"{type(error).__name__}: {error}"

    result.written = written

    if result.error is None:
        draft.archive()

    return result


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

    for entry in draft.experiments:
        name = str(entry["name"])
        exp_path = f"{draft.folder}/{name}"
        exp_id = entry.get("submitted_exp_id")

        if exp_id is None:
            exp_id = _ingest(db, session_id, exp_path, entry)

        if exp_id is None:
            written.skipped.append(name)
            continue

        draft.set_experiment(name, submitted_exp_id=int(exp_id))
        written.experiments[name] = int(exp_id)

        values = _annotation_values(db, "experiment", entry)

        if values:
            db.add_annotations(
                target_type="experiment", target_id=exp_id, values=values
            )
            written.annotations += len(values)

        written.annotations += _write_conditions(db, draft, name, exp_id, entry)

        if draft.vial_ids:
            db.set_experiment_vials(exp_id=exp_id, vial_ids=draft.vial_ids)

        if entry.get("exp_type") == "loop" and _has_sync(db, exp_path):
            # The experiment is in either way; a decode that refuses (a burst
            # count that does not match, say) is reported and can be rerun
            # once the file is sorted out.
            try:
                db.decode_sync(exp_id=exp_id)
                written.decoded.append(name)

            except Exception as error:
                logger.exception(f"Decoding the sync file of {name} failed")
                written.decode_failed[name] = f"{type(error).__name__}: {error}"


def _ingest(db, session_id: int, exp_path: str, entry: dict) -> None | int:
    """Add the folder's recordings, or find them if they are already in."""
    objective = int(entry["objective"])
    exp_type = entry.get("exp_type", "loop")

    if exp_type == "loop":
        exp_id = db.add_experiment(
            session_id=session_id, exp_path=exp_path, objective=objective
        )

    else:
        added = db.add_grabs(
            session_id=session_id,
            exp_path=exp_path,
            objective=objective,
            exp_type=exp_type,
        )
        # A grab folder is several experiments; the annotations go on the
        # folder's first one, which is what the draft entry stands for
        exp_id = added[0] if added else None

    if exp_id is not None:
        return exp_id

    # Already in the database (an earlier attempt, or someone else's)
    rows = db.from_query(
        "SELECT exp_id FROM experiments WHERE session_id = ? AND exp_path = ? "
        "ORDER BY exp_id;",
        [session_id, exp_path],
    )

    return None if rows.empty else int(rows["exp_id"].iloc[0])


def _write_conditions(db, draft: Draft, name: str, exp_id: int, entry: dict) -> int:
    """
    Each acquisition in a condition's range gets it as an annotation. Only
    when the conditions changed since the last submit, so a resubmit does
    not write them all again. Returns how many were written.
    """
    conditions = entry.get("conditions") or []

    if not conditions or conditions == entry.get("conditions_submitted"):
        return 0

    rows = db.from_query(
        "SELECT acq_id, raw_path FROM acquisitions WHERE exp_id = ?;", [int(exp_id)]
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

    draft.set_experiment(name, conditions_submitted=[dict(c) for c in conditions])

    return written


def _annotation_values(db, applies_to: str, values: dict) -> dict:
    """The typed values a draft section holds for the registry's keys."""
    parsed = {}

    for field_ in form_fields(db, applies_to):
        if field_.key in EXPERIMENT_SETTINGS or not field_.editable:
            continue

        if field_.key in ("mouse_id", "session_date", "objective"):
            continue  # columns, not annotations

        if field_.key in values:
            parsed[field_.key] = field_.parse(values[field_.key])

    return parsed


def _has_sync(db, exp_path: str) -> bool:
    folder = Path(db.main_folder) / exp_path / SYNC_FOLDER

    return folder.is_dir() and any(
        p for p in folder.glob("*.h5") if not p.name.startswith(".")
    )
