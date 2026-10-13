"""
Sync recordings kept on this computer, and their copies on the share.

The recorder writes each file locally first, so a share that drops mid-session
never stops a recording, then copies it to where `submit` looks for it:

    ~/.odyn/sync/20261001/m442/e1/m442_e1_2026-10-01_13-02-11.h5     (local)
    <main folder>/20261001/m442/e1/sync/m442_e1_2026-10-01_13-02-11.h5

The local folders mirror the share's, and the file keeps its final name, so
either copy says what it is without the draft. The draft keeps the link (one
entry per recording, see `Draft.add_recording`), and the local copy is never
deleted here.

A copy is written under a hidden name (`.<name>.partial`) and renamed once its
size matches, so a half-copied file never looks like a recording. A file on
the share is never overwritten. A recording that was replaced by a newer one
of the same part is hidden on the share (renamed with a leading dot,
which `process_sync` ignores) rather than deleted.

No Qt in here: the recorder runs `copy_to_share` on a thread of its own, and
`submit` runs `sync_pending` on the database thread before adding anything.
"""

from __future__ import annotations

import os
import shutil

from collections.abc import Container
from datetime import datetime
from pathlib import Path
from typing import Any

from ..database import SYNC_FOLDER
from .draft import Draft

LOCAL_FOLDER = "sync"  # beside the drafts folder: ~/.odyn/sync

# Folders each kind of part gets, so ScanImage and the pipeline find their
# place. Z-stacks have no sync recording.
PART_FOLDERS = {
    "experiment": ("raw", SYNC_FOLDER, "processed"),
    "zstack": ("raw", "processed"),
}


def local_path(root: Path, folder: str, part: str, name: str) -> Path:
    """Where a recording is written on this computer."""
    return Path(root) / folder / part / name


def share_path(folder: str, part: str, name: str) -> str:
    """Where its copy goes, relative to the main folder."""
    return f"{folder}/{part}/{SYNC_FOLDER}/{name}"


def make_part_folders(part_folder: Path, kind: str = "experiment") -> None:
    """A part's folder and the ones inside it, as its kind needs."""
    for name in PART_FOLDERS[kind]:
        (part_folder / name).mkdir(parents=True, exist_ok=True)


def copy_to_share(local: Path, target: Path) -> None:
    """
    Copy one recording to the share. Raises `OSError` when it could not.

    A file already at `target` with the same size is taken as an earlier copy
    of this one; a different one is never overwritten.
    """
    local, target = Path(local), Path(target)
    size = local.stat().st_size

    if target.exists():
        if target.stat().st_size == size:
            return

        raise FileExistsError(f"A different file is already at '{target}'")

    make_part_folders(target.parent.parent)
    partial = target.with_name(f".{target.name}.partial")
    shutil.copyfile(local, partial)

    if partial.stat().st_size != size:
        raise OSError(f"The copy of '{local.name}' came out incomplete.")

    os.replace(partial, target)


def hide_on_share(target: Path) -> None:
    """Take a replaced recording out of `process_sync`'s sight, keeping it."""
    target = Path(target)

    if target.exists():
        os.replace(target, target.with_name(f".replaced-{target.name}"))


def sync_pending(
    main_folder: Path, draft: Draft, busy: Container[str] = ()
) -> dict[str, str]:
    """
    Bring the share in line with the draft's recordings: copy each
    part's latest one if it is not there yet, and hide the replaced
    ones that were copied. Files in `busy` (still being written) are left.

    Returns what could not be done, as `{part: why}`.
    """
    failed: dict[str, str] = {}

    for entry in draft.parts:
        name = str(entry["name"])

        for record in entry.get("recordings", []):
            target = Path(main_folder) / record["share"]

            try:
                if record.get("replaced"):
                    if record.get("copied") and not record.get("hidden"):
                        hide_on_share(target)
                        draft.update_recording(name, record["file"], hidden=True)

                elif not record.get("copied") and record["local"] not in busy:
                    copy_to_share(Path(record["local"]), target)
                    draft.update_recording(name, record["file"], copied=_now())

            except OSError as error:
                failed[name] = f"{record['file']}: {error}"

    return failed


def pending(draft: Draft) -> dict[str, dict[str, Any]]:
    """
    Each experiment's recording in use (its latest) that is not on the share
    yet.
    """
    found = {}

    for entry in draft.parts:
        latest = Draft.latest(entry)

        if latest is not None and not latest.get("copied"):
            found[str(entry["name"])] = latest

    return found


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")
