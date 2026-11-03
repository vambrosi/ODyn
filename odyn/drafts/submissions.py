"""
Submitted drafts waiting on the share, and how the cluster works through them.

At Submit the app puts the draft in the database's queue (`enqueue`), in
`<main_folder>/.odyn/cluster/<name>/submissions/`, and writes nothing to the
database itself. A tick on the cluster then looks at each queued draft
(`process_submissions`):

- **waiting** while the TIFFs are still arriving. A loop is complete when it
has as many TIFFs as ScanImage was set to take (`SI.acqsPerLoop`, from the
first TIFF's header), or as many as its sync file has bursts if that is fewer
(the loop was stopped early). A grab is complete once it has TIFFs, and
the z-stacks once they have a TIFF per z-stack listed (one at least). Then
nothing may change for `SETTLE_S` (a TIFF may be half copied), and the newest
TIFF must read.
- **failed** when nothing changed for `GIVE_UP_S` before the draft was
complete (a copy that broke, a share that ran out of space), or when
`ingest` refused it or stopped. A failure the files caused is tried again
once they change; one the typed draft caused waits for it to be submitted
again.
- **done** once `ingest` put the whole draft in the database, as the person
who submitted it. The draft and its state then move to `done/`.

The state is a file beside each draft (`<draft>.state.json`, see
`Submission`); the queued draft itself is what `ingest` updates, so an ingest
that stopped halfway resumes where it was. Nothing in the queue is deleted.

What the QC page can do with a queued draft:
- `add_now`: ingest it at the next tick without waiting for its TIFFs (a
loop stopped early with no sync file to say so, say);
- `submit_again`: start a failed one over, once what stopped it is fixed;
- `take_back`: move it out of the queue (to `withdrawn/`) and back among the
drafts on this computer, to edit it and submit it again.

**USAGE**
```python
path = enqueue(draft, main_folder, project="main-sync", user="ana")
lines = process_submissions(main_folder, project="main-sync")   # on the cluster
for path, state in queued(main_folder, project="main-sync"):
    print(path.name, state.state, state.message)
```

No Qt in here: the cluster runs it, and the app only reads the states.
"""

from __future__ import annotations

import json
import os
import socket

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

from tifffile import TiffFile

from ..cluster import TIME_FORMAT, cluster_folder
from ..database import SYNC_FOLDER, Database
from ..sync import decode
from ..utils import logger
from .draft import Draft
from .submit import RAW_FOLDER, check_files, ingest

QUEUE_FOLDER = "submissions"
DONE_FOLDER = "done"
WITHDRAWN_FOLDER = "withdrawn"
STATE_SUFFIX = ".state.json"
NOW_SUFFIX = ".now"

# What `enqueue` adds to a draft, taken off again when it comes back to edit
QUEUE_KEYS = ("submitted_by", "submitted_at", "submitted_on", "draft_file")

# A draft is ingested once its files stopped changing for this long, and
# fails when they did not change for this long before it was complete
SETTLE_S = 600
GIVE_UP_S = 12 * 3600


@dataclass
class Submission:
    """
    Where one queued draft stands (its `.state.json`).

    - `state`: "waiting", "failed" or "done"
    - `message`: what it waits for, why it failed, or what was added
    - `parts`: the progress of each experiment or z-stack ("45 of 160 TIFFs")
    - `files`: what the draft's folders looked like at `last_change_at`
    - `retry_on_change`: a failure the files caused, tried again once they change
    - `written`: the database ids `ingest` made (session, experiments, ...)
    """

    state: str = "waiting"
    message: str = ""
    parts: dict[str, str] = field(default_factory=dict)
    first_seen_at: str = ""
    last_change_at: str = ""
    updated_at: str = ""
    files: str = ""
    retry_on_change: bool = False
    written: dict = field(default_factory=dict)


def queue_folder(main_folder: str | Path, project: None | str = None) -> Path:
    """Where a database's submitted drafts wait."""
    return cluster_folder(main_folder, project) / QUEUE_FOLDER


def enqueue(
    draft: Draft,
    main_folder: str | Path,
    project: None | str = None,
    *,
    user: str,
) -> Path:
    """
    Put a copy of `draft` in the queue, as submitted by `user`, and return it.
    A draft submitted again replaces its queued copy and starts over as
    waiting (what an earlier ingest added is still skipped).
    """
    # A draft's name is when it was started on its computer, so two people's
    # can match: the queued copy also says whose it is (and stays in order)
    folder = queue_folder(main_folder, project)
    path = folder / f"{draft.path.stem}-{user}.json"
    content = {
        **draft.data,
        "submitted_by": user,
        "submitted_at": datetime.now().strftime(TIME_FORMAT),
        "submitted_on": socket.gethostname(),
        "draft_file": draft.path.name,
    }
    _write_json(path, content)
    _state_path(path).unlink(missing_ok=True)

    return path


def queued(
    main_folder: str | Path, project: None | str = None
) -> list[tuple[Path, Submission]]:
    """The drafts still in the queue (waiting or failed), oldest first."""
    folder = queue_folder(main_folder, project)

    if not folder.is_dir():
        return []

    return [(path, _read_state(path) or Submission()) for path in _drafts(folder)]


def added(
    main_folder: str | Path, project: None | str = None
) -> list[tuple[Path, Submission]]:
    """The drafts the queue added (in `done/`), oldest first."""
    folder = queue_folder(main_folder, project) / DONE_FOLDER

    return [(path, _read_state(path) or Submission()) for path in _drafts(folder)]


def add_now(path: Path) -> None:
    """Ask the next tick to ingest a queued draft without waiting for its files."""
    _now_path(path).write_text(datetime.now().strftime(TIME_FORMAT) + "\n")


def submit_again(path: Path) -> None:
    """Start a queued draft over as waiting (a failed one, once it is fixed)."""
    _state_path(path).unlink(missing_ok=True)


def take_back(path: Path, drafts: Path) -> Path:
    """
    Move a queued draft to `withdrawn/` and put a copy among the drafts in
    `drafts` (this computer's), to edit and submit again. Returns the copy.
    What an earlier ingest added stays in the database, and is skipped when
    the draft comes back.
    """
    content = json.loads(path.read_text())
    local = Path(drafts) / str(content.get("draft_file") or path.name)

    for key in QUEUE_KEYS:
        content.pop(key, None)

    _write_json(local, content)

    withdrawn = path.parent / WITHDRAWN_FOLDER
    withdrawn.mkdir(exist_ok=True)

    for source in (path, _state_path(path), _now_path(path)):
        if source.exists():
            os.replace(source, withdrawn / source.name)

    return local


def process_submissions(
    main_folder: str | Path,
    project: None | str = None,
    *,
    now: None | datetime = None,
    settle_s: float = SETTLE_S,
    give_up_s: float = GIVE_UP_S,
    dry_run: bool = False,
) -> list[str]:
    """
    One look at every queued draft (see the module's description): ingest the
    complete ones, and update the state of the rest. Returns a line per draft
    that is not left as it was, for the tick's report.
    """
    main_folder = Path(main_folder).resolve()
    now = now or datetime.now()
    stamp = now.strftime(TIME_FORMAT)
    lines = []

    for path in _drafts(queue_folder(main_folder, project)):
        state = _read_state(path) or Submission(
            first_seen_at=stamp, last_change_at=stamp
        )

        now_asked = _now_path(path).exists()

        if state.state == "failed" and not state.retry_on_change and not now_asked:
            continue  # until the draft is submitted again

        try:
            draft = Draft.load(path)
        except (OSError, ValueError) as error:
            message = f"could not read the draft: {error}"
            lines.append(f"{path.name}: failed: {message}")

            if not dry_run:
                _fail(path, state, stamp, message, False)

            continue

        complete, parts, files = _progress(main_folder, draft)

        if files != state.files:
            state.files, state.last_change_at = files, stamp

        elif state.state == "failed" and not now_asked:
            continue  # the files did not change since it failed

        state.parts = parts
        state.updated_at = stamp
        quiet_s = (now - _time(state.last_change_at)).total_seconds()

        if now_asked or (complete and quiet_s >= settle_s):
            if dry_run:
                lines.append(f"{draft.where}: would be added")
                continue

            _now_path(path).unlink(missing_ok=True)
            lines.append(_ingest(main_folder, project, path, draft, state, stamp))

        elif quiet_s >= give_up_s:
            message = f"no new files for {give_up_s / 3600:g} h ({_listed(parts)})"
            lines.append(f"{draft.where}: failed: {message}")

            if not dry_run:
                _fail(path, state, stamp, message, True)

        else:
            state.state = "waiting"
            state.message = (
                _listed(parts) if not complete else "waiting for the copy to finish"
            )
            lines.append(
                f"{draft.where}: waiting: {state.message}"
                if not complete
                else f"{draft.where}: {state.message}"
            )

            if not dry_run:
                _write_state(path, state)

    return lines


# --------------------------------------------------------------------------- #
# Ingesting
# --------------------------------------------------------------------------- #


def _ingest(
    main_folder: Path,
    project: None | str,
    path: Path,
    draft: Draft,
    state: Submission,
    stamp: str,
) -> str:
    """Ingest a complete draft as its submitter, and record how it went."""
    user = str(draft.data.get("submitted_by") or draft.user or "")
    db = Database(main_folder, project=project, user=user or None)
    result = ingest(db, draft)

    if result.error is not None:
        _fail(path, state, stamp, result.error, True)
        return f"{draft.where}: failed: {state.message}"

    if result.written is None:
        blocking = [p for p in result.problems if p.blocking]
        on_share = set(check_files(db, draft))
        typed = [p for p in blocking if p not in on_share]
        _fail(path, state, stamp, "; ".join(str(p) for p in blocking), not typed)
        return f"{draft.where}: failed: {state.message}"

    written = result.written
    state.state = "done"
    state.written = {
        "session_id": written.session_id,
        "experiments": written.experiments,
        "zstacks": written.zstacks,
    }
    refused = [
        f"sync file of {name} refused: {why}"
        for name, why in written.process_failed.items()
    ]
    state.message = "; ".join([f"added as session {written.session_id}", *refused])
    state.updated_at = stamp
    _write_state(path, state)
    _move_to_done(path)
    logger.info(f"Added {draft.where} for {user}: {state.message}.")

    return f"{draft.where}: {state.message}"


def _fail(path: Path, state: Submission, stamp: str, message: str, retry: bool) -> None:
    state.state = "failed"
    state.message = message
    state.retry_on_change = retry
    state.updated_at = stamp
    _write_state(path, state)
    logger.warning(f"Submission '{path.name}' failed: {message}")


def _move_to_done(path: Path) -> None:
    done = path.parent / DONE_FOLDER
    done.mkdir(exist_ok=True)

    for source in (path, _state_path(path), _now_path(path)):
        if source.exists():
            os.replace(source, done / source.name)


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


def _progress(main_folder: Path, draft: Draft) -> tuple[bool, dict[str, str], str]:
    """
    Whether every part of `draft` has its files on the share, each part's
    progress as text, and what the folders look like (to tell a change).
    """
    session = main_folder / draft.folder
    complete = True
    parts: dict[str, str] = {}
    looks: list[str] = []

    for entry in draft.parts:
        name, kind = str(entry.get("name", "")), entry.get("kind", "experiment")
        folder = session / name
        loop = kind == "experiment" and entry.get("exp_type", "loop") == "loop"

        if kind == "experiment" and entry.get("submitted_id") is not None:
            parts[name] = "added"
            continue

        tiffs = _files(folder / RAW_FOLDER, "*.tif")
        syncs = _files(folder / SYNC_FOLDER, "*.h5")
        looks.append(f"{name}: {_looks(tiffs + syncs)}")

        if not tiffs:
            parts[name] = "no TIFFs yet"
            complete = False
            continue

        # The z-stacks wait for one TIFF per z-stack listed
        if kind == "zstack":
            wanted = max(1, len(entry.get("stacks", [])))
            parts[name] = f"{len(tiffs)} of {wanted} TIFFs"
            complete = complete and len(tiffs) >= wanted and _reads(tiffs[-1])
            continue

        # A grab's TIFFs still say the loop length last set, so only a loop
        # waits for a count
        if not loop:
            parts[name] = f"{len(tiffs)} TIFF{'' if len(tiffs) == 1 else 's'}"
            complete = complete and _reads(tiffs[-1])
            continue

        expected = _acqs_per_loop(tiffs[0])

        if expected is None:
            parts[name] = f"{len(tiffs)} TIFFs (the first does not read yet)"
            complete = False
            continue

        # A loop stopped early has fewer bursts in its sync file
        if len(tiffs) < expected and len(syncs) == 1:
            bursts = _bursts(syncs[0])

            if bursts is not None and bursts < expected:
                expected = bursts

        parts[name] = f"{len(tiffs)} of {expected} TIFFs"
        complete = complete and len(tiffs) >= expected and _reads(tiffs[-1])

    return complete, parts, "; ".join(looks)


def _files(folder: Path, pattern: str) -> list[Path]:
    """The visible files in `folder` (hidden ones are copies in progress)."""
    if not folder.is_dir():
        return []

    return sorted(p for p in folder.glob(pattern) if not p.name.startswith("."))


def _looks(paths: list[Path]) -> str:
    """Names, sizes and times of `paths`, to tell when any of them changes."""
    looks = []

    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue

        looks.append(f"{path.name} {stat.st_size} {stat.st_mtime_ns}")

    return ", ".join(looks)


def _acqs_per_loop(path: Path) -> None | int:
    """How many acquisitions ScanImage was set to take, from a TIFF's header."""
    try:
        with TiffFile(path) as tif:
            frame_data = (tif.scanimage_metadata or {}).get("FrameData") or {}
            value = frame_data.get("SI.acqsPerLoop")

    except Exception:  # a file still being copied
        return None

    return None if value is None else int(value)


def _reads(path: Path) -> bool:
    """Whether a TIFF's header and last page read (it is not half copied)."""
    try:
        with TiffFile(path) as tif:
            tif.pages[-1].asarray()

    except Exception:
        return False

    return True


def _bursts(path: Path) -> None | int:
    """How many acquisitions a sync file recorded, or `None` if it does not read."""
    try:
        return len(decode(path).blocks)
    except Exception:
        return None


def _listed(parts: dict[str, str]) -> str:
    return ", ".join(f"{name}: {text}" for name, text in parts.items()) or "no parts"


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def _drafts(folder: Path) -> list[Path]:
    """The queued drafts, oldest first (their names start with when they began)."""
    if not folder.is_dir():
        return []

    return sorted(
        p
        for p in folder.glob("*.json")
        if p.is_file()
        and not p.name.startswith(".")
        and not p.name.endswith(STATE_SUFFIX)
    )


def _state_path(path: Path) -> Path:
    return path.with_name(path.stem + STATE_SUFFIX)


def _now_path(path: Path) -> Path:
    return path.with_name(path.stem + NOW_SUFFIX)


def _read_state(path: Path) -> None | Submission:
    try:
        content = json.loads(_state_path(path).read_text())
    except (OSError, ValueError):
        return None

    known = {f.name for f in fields(Submission)}

    return Submission(**{k: v for k, v in content.items() if k in known})


def _write_state(path: Path, state: Submission) -> None:
    _write_json(_state_path(path), asdict(state))


def _write_json(path: Path, content: dict) -> None:
    """Write under a hidden name, then rename (readers see all or nothing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = path.parent / f".{path.name}.partial"
    hidden.write_text(json.dumps(content, indent=2, sort_keys=True, default=str))
    os.replace(hidden, path)


def _time(text: str) -> datetime:
    return datetime.strptime(text, TIME_FORMAT)
