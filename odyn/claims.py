"""
QC decisions taken in the app, as claim files the cluster records later.

A decision (a raw check, a pick, an approval, running stopped jobs again) is
one small file, `claims/g<group>-<step>.json` in the database's cluster
folder, created only if none is there (`claim`): a second person deciding the
same step of the same experiment learns at once who did it first. The tick
then records each claim in the database as the person who made it (with
`cluster.apply_claims`), starts what follows (a whole correction after a
pick), and files it in `claims/done/`. A claim the tick refuses stays, with
the reason, for its author to see on the QC page; deciding that step again
replaces it.

**USAGE**
```python
claim(main_folder, "main-sync", 42, "approve", "alice", excluded=[3, 7])
for (group_id, step), found in pending_claims(main_folder, "main-sync").items():
    print(group_id, step, found["user"], found.get("failed"))
```

No Qt in here: the app makes claims, the cluster applies them.
"""

from __future__ import annotations

import json
import os

from datetime import datetime
from pathlib import Path
from typing import Any

from .shared import Inbox, read_json

CLAIMS_FOLDER = "claims"

# The steps of an experiment's QC, and how the page says each
STEPS = {
    "raw": "raw check",
    "pick": "pick",
    "approve": "approval",
    "again": "run again",
}


class ClaimTaken(Exception):
    """Someone else decided this step already; the cluster has not recorded it."""

    def __init__(self, user: str, at: None | float):
        self.user, self.at = user, at
        when = "" if at is None else f" at {datetime.fromtimestamp(at):%H:%M}"
        super().__init__(f"{user} did it already{when}")


def claims(main_folder: str | Path, project: None | str = None) -> Inbox:
    """Where the claims wait for the cluster."""
    from .cluster import cluster_folder  # it imports this module

    return Inbox(cluster_folder(main_folder, project) / CLAIMS_FOLDER)


def claim(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    step: str,
    user: str,
    **details: Any,
) -> Path:
    """
    Claim `step` (see `STEPS`) of group `group_id` for `user`, with what the
    cluster needs to record it (`details`: `excluded`, `run`, ...). Replaces
    a claim of this user or one the cluster refused. Raises `ClaimTaken` when
    someone else claimed it and the cluster has not recorded it yet.
    """
    if step not in STEPS:
        raise ValueError(f"Unknown QC step {step!r}; one of {', '.join(STEPS)}.")

    folder = claims(main_folder, project).folder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"g{int(group_id)}-{step}.json"
    content = {"group_id": int(group_id), "step": step, "user": user, **details}

    for _ in range(2):
        try:
            # Created only if not there: the one step every machine on the
            # share sees the same way (as the database lock is)
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            other = _read(path)

            if other is not None and (other.get("failed") or other["user"] == user):
                path.unlink(missing_ok=True)
                continue

            raise ClaimTaken(
                "someone" if other is None else str(other["user"]),
                None if other is None else other["at"],
            ) from None

        with os.fdopen(handle, "w", encoding="utf-8", newline="") as file:
            json.dump(content, file, indent=2, default=str)

        return path

    raise ClaimTaken("someone", None)  # taken again in between


def pending_claims(
    main_folder: str | Path, project: None | str = None
) -> dict[tuple[int, str], dict]:
    """
    The claims not recorded yet, by `(group_id, step)`, each with its `at`
    (the file time, by the server clock) and, when the cluster refused it,
    `failed` (why). One listing, and a read per claim.
    """
    found = {}

    for path in claims(main_folder, project).items():
        content = _read(path)

        if content is not None:
            found[(int(content["group_id"]), str(content["step"]))] = content

    return found


def _read(path: Path) -> None | dict:
    """A claim with its `at`, or `None` while it is being written (or gone)."""
    try:
        content = read_json(path)
        at = path.stat().st_mtime
    except (OSError, ValueError):
        return None

    if not isinstance(content, dict) or "user" not in content:
        return None

    return {**content, "at": at}
