"""
Turn a finished draft into database rows.

Submitting is two steps, and the first one writes nothing: `check` reports
everything wrong with a draft, and `submit` refuses to run while any of it is
blocking. That split is what lets the app show someone their mistakes while
they can still fix them, rather than failing halfway through writing.

Submitting **ingests and annotates together**. A drafted session has no rows
anywhere until this runs, so the recordings are read first and the drafted
fields are attached to what that produced. The common failure is a session
whose recordings have not been copied off the rig yet, which `check` reports as
`no recordings` so the app can say so and wait.

**USAGE**
```python
problems = check(draft, db)

if not any(problem.blocking for problem in problems):
    written = submit(draft, db)
```

Field names in a draft are annotation keys as the registry spells them, with
two exceptions handled here: `objective` is a column on `experiments`, and the
`mouse` block goes to `Database.set_mouse`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..database import _mouse_number
from .draft import Draft

# Written as a column rather than an annotation, because the micron-per-pixel
# scale depends on it. Everything else in an experiment entry is an annotation.
EXPERIMENT_COLUMNS = ("objective",)

# Not annotations either: they say which experiment the entry is about.
EXPERIMENT_KEYS = ("name",)

# Keys `Database.set_mouse` takes, as the draft's `mouse` block spells them.
MOUSE_FIELDS = ("sex", "dob", "lines", "stax_injection", "sensor")


@dataclass(frozen=True)
class Problem:
    """Something wrong with a draft. `blocking` ones stop the submit."""

    where: str
    what: str
    blocking: bool = True

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


@dataclass
class Written:
    """What a submit put in the database."""

    session_id: None | int = None
    experiments: dict[str, int] = field(default_factory=dict)
    annotations: int = 0
    mouse_id: None | int = None
    panel: bool = False


class SubmitRefused(Exception):
    """Raised when a draft still has blocking problems. Carries all of them."""

    def __init__(self, problems: list[Problem]):
        self.problems = problems

        super().__init__(
            "This draft cannot be submitted yet:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


# --------------------------------------------------------------------------- #
# Where the recordings should be
# --------------------------------------------------------------------------- #


def session_folders(main_folder: Path | str, draft: Draft) -> list[str]:
    """
    Every folder that could hold this draft's recordings.

    Found rather than constructed. A mouse id is the number alone, so it cannot
    say whether the folder was named `m442`, `m0442` or `M442` -- the day's
    folder is read instead and each child resolved back to a number.

    Normally one folder; none means the recordings are not there yet, and more
    than one means the day is ambiguous and a person has to say which.
    """
    main_folder = Path(main_folder)
    stated = draft.data.get("session_path")

    if stated:
        path = str(stated).strip("/")

        return [path] if (main_folder / path).is_dir() else []

    day = main_folder / draft.date.replace("-", "")

    if not day.is_dir():
        return []

    return sorted(
        f"{day.name}/{child.name}"
        for child in day.iterdir()
        if child.is_dir() and _mouse_number_of(child.name) == draft.mouse_id
    )


def session_path(main_folder: Path | str, draft: Draft) -> None | str:
    """The one folder holding this draft's recordings, or `None` if unclear."""
    folders = session_folders(main_folder, draft)

    return folders[0] if len(folders) == 1 else None


def _mouse_number_of(name: str) -> None | int:
    """The mouse a folder is named for, or `None` if it names no mouse."""
    try:
        return _mouse_number(name)
    except ValueError:
        return None


def experiment_folders(main_folder: Path, session: str) -> list[str]:
    """
    Every experiment folder in the session, relative to `main_folder`.

    An experiment is a folder with a `raw/` subfolder of TIFFs, which is what
    ingestion reads. Returned sorted, so `e1` is added before `e2`.
    """
    folder = Path(main_folder) / session

    if not folder.is_dir():
        return []

    return sorted(
        {
            tiff.parent.parent.relative_to(Path(main_folder)).as_posix()
            for tiff in folder.rglob("raw/[!.]?*.tif")
        }
    )


def _experiment_name(rel_path: str) -> str:
    """`'20260708/m442/e1'` -> `'e1'`."""
    return rel_path.rsplit("/", 1)[-1]


def recorded_mice(main_folder: Path | str, rel_path: str) -> set[int]:
    """
    The mice the raw file *names* say an experiment is of.

    Ingestion takes the animal from the file name rather than the folder, so
    the two can disagree -- a folder renamed after the fact, or a recording
    started under the previous mouse. Read from the names alone, so this costs
    a directory listing and no file opens.
    """
    folder = Path(main_folder) / rel_path / "raw"

    if not folder.is_dir():
        return set()

    found = set()

    for tiff in folder.glob("[!.]?*.tif"):
        parts = tiff.stem.split("_")
        number = _mouse_number_of(parts[1]) if len(parts) > 1 else None

        if number is not None:
            found.add(number)

    return found


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #


def check(draft: Draft, db) -> list[Problem]:
    """
    Everything wrong with a draft, without writing anything.

    Blocking problems are the ones where submitting would lose or misplace what
    someone typed: no recordings to attach to, a drafted experiment that was
    never recorded, an unregistered panel, a key the registry does not hold.
    Non-blocking ones are worth showing but do not stop a submit.
    """
    problems: list[Problem] = []
    where = f"m{draft.mouse_id} {draft.date}"

    if draft.is_empty:
        return [Problem(where, "nothing has been filled in")]

    sessions = session_folders(db.main_folder, draft)
    folders: list[str] = []

    if not sessions:
        problems.append(
            Problem(
                where,
                f"no folder for this mouse under '{draft.date.replace('-', '')}'."
                f" Copy the recordings off the rig, then submit again",
            )
        )
    elif len(sessions) > 1:
        problems.append(
            Problem(
                where,
                f"several folders could be this session ({', '.join(sessions)});"
                f" say which one in the draft",
            )
        )
    else:
        folders = experiment_folders(db.main_folder, sessions[0])

        if not folders:
            problems.append(
                Problem(
                    where,
                    f"'{sessions[0]}' holds no experiment with a 'raw/' folder of"
                    f" TIFFs. Copy the recordings over, then submit again",
                )
            )

    for rel_path in folders:
        mice = recorded_mice(db.main_folder, rel_path)

        # Ingestion would file these under the mouse the file names give, and
        # the annotations are looked up under the drafted one, so they would
        # land on different sessions -- or on none at all.
        if mice and mice != {draft.mouse_id}:
            problems.append(
                Problem(
                    f"{where} {_experiment_name(rel_path)}",
                    f"the recordings are named for "
                    f"{', '.join(f'm{number}' for number in sorted(mice))}, not "
                    f"m{draft.mouse_id}",
                )
            )

    recorded = {_experiment_name(path) for path in folders}
    drafted = {entry.get("name") for entry in draft.experiments}

    for name in sorted(drafted - recorded):
        problems.append(
            Problem(
                f"{where} {name}",
                "was filled in but has no recordings; submitting would lose it",
            )
        )

    for name in sorted(recorded - drafted):
        problems.append(
            Problem(
                f"{where} {name}",
                "was recorded but has nothing filled in",
                blocking=False,
            )
        )

    problems += _check_keys(draft, db, where)
    problems += _check_panel(draft, db, where)

    return problems


def _check_keys(draft: Draft, db, where: str) -> list[Problem]:
    """Every drafted field has to be a key the registry holds."""
    problems = []
    registry = db.annotation_keys

    for key in sorted(draft.session):
        if ("session", key) not in registry.index:
            problems.append(Problem(where, f"'{key}' is not a session annotation"))

    for entry in draft.experiments:
        name = entry.get("name", "?")

        for key in sorted(entry):
            if key in EXPERIMENT_KEYS or key in EXPERIMENT_COLUMNS:
                continue

            if ("experiment", key) not in registry.index:
                problems.append(
                    Problem(
                        f"{where} {name}", f"'{key}' is not an experiment annotation"
                    )
                )

    for key in sorted(draft.mouse):
        if key not in MOUSE_FIELDS:
            problems.append(
                Problem(where, f"'{key}' is not something recorded about a mouse")
            )

    return problems


def _check_panel(draft: Draft, db, where: str) -> list[Problem]:
    """The panel has to be registered, and its vials have to exist."""
    name = draft.panel.get("panel_name")

    if not name:
        if draft.panel:
            return [Problem(where, "mixing dates were given but no panel", False)]

        return [Problem(where, "no odor panel was chosen", blocking=False)]

    if name not in set(db.panels["panel_name"]):
        return [
            Problem(
                where,
                f"panel '{name}' is not registered; add it before submitting",
            )
        ]

    vials = db.panel_vials
    known = set(vials.index.get_level_values("panel_name"))
    positions = set(vials.loc[name].index) if name in known else set()

    unknown = sorted(
        int(position)
        for position in draft.panel.get("vial_dates", {})
        if int(position) not in positions
    )

    if unknown:
        return [Problem(where, f"panel '{name}' has no vial {unknown}")]

    return []


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def submit(draft: Draft, db, *, force: bool = False, archive: bool = True) -> Written:
    """
    Ingest the session's recordings and attach everything the draft holds.

    **PARAMETERS**
    - `draft` is the finished draft
    - `db` is the `Database` to write to
    - `force` submits despite blocking problems, writing what it can
    - `archive` moves the draft aside afterwards, which is what stops it being
      offered again

    Returns a `Written` saying what was created. Raises `SubmitRefused`, which
    carries every problem, when `check` found a blocking one and `force` is not
    set.

    Each step records its own call, so a failure partway through leaves what
    came before it in place rather than rolling back. `check` first: it is what
    makes that rare.
    """
    problems = check(draft, db)
    blocking = [problem for problem in problems if problem.blocking]

    if blocking and not force:
        raise SubmitRefused(problems)

    written = Written()

    if draft.mouse:
        written.mouse_id = db.set_mouse(
            mouse_id=draft.mouse_id,
            **{key: draft.mouse[key] for key in MOUSE_FIELDS if key in draft.mouse},
        )

    session = session_path(db.main_folder, draft)

    # Only reachable under `force`: without one folder there is nothing to
    # ingest, and searching from the main folder would sweep the whole share.
    if session is not None:
        for rel_path in experiment_folders(db.main_folder, session):
            db.add_experiment(rel_path=rel_path)

    written.session_id = _session_of(db, draft)

    if written.session_id is None:
        # Nothing was ingested, so there is no session to hang anything on.
        # Only reachable under `force`, since `check` blocks on it.
        return written

    written.annotations += _write_annotations(
        db, "session", written.session_id, draft.session
    )

    for entry in draft.experiments:
        name = entry.get("name")
        exp_id = _experiment_of(db, written.session_id, name)

        if exp_id is None:
            continue

        written.experiments[name] = exp_id

        if "objective" in entry:
            db.set_objective(exp_id=exp_id, objective=entry["objective"])

        fields = {
            key: value
            for key, value in entry.items()
            if key not in EXPERIMENT_KEYS and key not in EXPERIMENT_COLUMNS
        }

        written.annotations += _write_annotations(db, "experiment", exp_id, fields)

    if draft.panel.get("panel_name"):
        db.set_session_panel(
            session_id=written.session_id,
            panel_name=draft.panel["panel_name"],
            made_on=draft.panel.get("made_on"),
            vial_dates={
                int(position): date
                for position, date in draft.panel.get("vial_dates", {}).items()
            }
            or None,
        )
        written.panel = True

    if archive:
        draft.archive()

    return written


# --------------------------------------------------------------------------- #
# A whole day at once
# --------------------------------------------------------------------------- #


@dataclass
class SessionResult:
    """What became of one session in a day-long submit."""

    draft: Draft
    written: None | Written = None
    problems: list[Problem] = field(default_factory=list)
    error: None | str = None

    @property
    def submitted(self) -> bool:
        return self.written is not None

    @property
    def blocked(self) -> bool:
        return any(problem.blocking for problem in self.problems)


def check_all(drafts: list[Draft], db) -> list[SessionResult]:
    """
    Check every session of a day, writing nothing.

    Recordings are usually copied off the rig only once, at the end of the day,
    so until then every session reports missing recordings. Checking them
    together is what makes that one review rather than four.
    """
    return [SessionResult(draft, problems=check(draft, db)) for draft in drafts]


def submit_all(
    drafts: list[Draft], db, *, force: bool = False, archive: bool = True
) -> list[SessionResult]:
    """
    Submit a day's sessions, one after another.

    Each is independent: one that is blocked, or that fails partway, does not
    stop the others, and stays on disk to be fixed and submitted again. The
    results say which is which -- `submitted`, `blocked`, or carrying an
    `error`.

    Check the whole day first (`check_all`) and show it, so that a run of
    ingestion is not started on sessions that were never going to be written.
    """
    results = []

    for draft in drafts:
        result = SessionResult(draft, problems=check(draft, db))

        if result.blocked and not force:
            results.append(result)
            continue

        try:
            result.written = submit(draft, db, force=True, archive=archive)
        except Exception as failure:
            # One session's bad TIFF must not cost the other three, so this is
            # recorded against the session rather than raised.
            result.error = f"{type(failure).__name__}: {failure}"

        results.append(result)

    return results


def _write_annotations(db, target_type: str, target_id: int, fields: dict) -> int:
    """Write one target's fields, skipping the ones nobody filled in."""
    count = 0

    for key, value in fields.items():
        if value is None or value == "":
            continue

        db.add_annotation(
            target_type=target_type, target_id=target_id, key=key, value=value
        )
        count += 1

    return count


def _session_of(db, draft: Draft) -> None | int:
    """The session ingestion created for this draft, if it created one."""
    row = db.con.execute(
        "SELECT session_id FROM sessions WHERE mouse_id = ? AND session_date = ?;",
        [_mouse_number(str(draft.mouse_id)), draft.date],
    ).fetchone()

    return None if row is None else row[0]


def _experiment_of(db, session_id: int, name: None | str) -> None | int:
    """The experiment of a session whose name ends in `name`, e.g. `'e1'`."""
    if not name:
        return None

    for exp_id, exp_name in db.con.execute(
        "SELECT exp_id, exp_name FROM experiments WHERE session_id = ?;",
        [session_id],
    ):
        if exp_name.rsplit("_", 1)[-1] == name:
            return exp_id

    return None
