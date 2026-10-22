"""
Copies of test motion corrections on this computer, for QC.

Played from the share, a test's video is slow, so the QC page downloads a
test's small files first (by hand: nothing is copied unless someone asks)
and opens them from here. They go to
`~/.odyn/cache/<main folder name>-<hash of its path>/` (beside the drafts
folder, `base` below), at the test folder's
path inside the main folder: the tests folder and the real share have the same
paths inside them (`projects/main-sync/cache/...`), and the hash keeps them
apart.

Only a test's own files are copied, not the folders inside it (a sweep's
runs, each downloaded on its own), and not its large ones
(`side_by_side.tif`, `shifts.npz`).

No Qt in here: the QC page calls it off the database thread.
"""

from __future__ import annotations

import hashlib
import os
import shutil

from collections.abc import Callable
from pathlib import Path

from .draft import FOLDER_NAME

CACHE_FOLDER = "cache"

# The files of a test folder worth having here (as `Path.match` patterns)
SMALL_FILES = (
    "*.avi",
    "*.mp4",
    "shifts.png",
    "reference.tif",
    "flow_average.tif",
    "flow_worst.tif",
    "summary.csv",
    "metrics.json",
    "shifts_summary.npz",
    "*.frames.csv",
)

# What `state` says of a test
NOT_HERE = "no"
HERE = "yes"
NEWER = "newer on server"

# File times on a network share are coarse (2 s on some), so closer than this
# is the same time
SAME_TIME_S = 2.0


def default_base() -> Path:
    """The folder the cache goes in, beside the drafts: `~/.odyn`."""
    return Path.home() / FOLDER_NAME


def cache_root(main_folder: str | Path, base: None | Path = None) -> Path:
    """Where the copies of `main_folder`'s tests go on this computer."""
    main = Path(os.path.abspath(main_folder))
    digest = hashlib.sha1(str(main).encode()).hexdigest()[:8]

    return (base or default_base()) / CACHE_FOLDER / f"{main.name}-{digest}"


def local_folder(main_folder: str | Path, folder: str | Path, base=None) -> Path:
    """Where test `folder` (inside `main_folder`) is copied to."""
    main = Path(os.path.abspath(main_folder))
    inside = Path(os.path.abspath(folder)).relative_to(main)

    return cache_root(main, base) / inside


def small_files(folder: str | Path) -> list[Path]:
    """The files of test `folder` that get copied, by name."""
    folder = Path(folder)

    if not folder.is_dir():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and any(path.match(pattern) for pattern in SMALL_FILES)
    )


def remote_size(folder: str | Path) -> int:
    """How many bytes downloading test `folder` copies (all of it)."""
    return sum(path.stat().st_size for path in small_files(folder))


def _same(remote: Path, local: Path) -> bool:
    if not local.is_file():
        return False

    theirs, ours = remote.stat(), local.stat()

    return (
        theirs.st_size == ours.st_size
        and abs(theirs.st_mtime - ours.st_mtime) < SAME_TIME_S
    )


def state(main_folder: str | Path, folder: str | Path, base=None) -> str:
    """
    Whether test `folder` is here: `"no"` (not downloaded, or nothing to
    download yet), `"yes"`, or `"newer on server"` (a file there changed or
    was added since).
    """
    files = small_files(folder)
    local = local_folder(main_folder, folder, base)

    # The folder can be there for the runs inside it, without this test's files
    if not files or not any((local / path.name).is_file() for path in files):
        return NOT_HERE

    if all(_same(path, local / path.name) for path in files):
        return HERE

    return NEWER


def download(
    main_folder: str | Path,
    folder: str | Path,
    base=None,
    progress: None | Callable[[str], None] = None,
) -> Path:
    """
    Copy test `folder`'s small files here, those not here already or changed
    on the share, and return the local folder. Each file is written under a
    temporary name and renamed when complete, so a copy cut short is never
    taken for a whole one. `progress` hears each file's name before it is
    copied.
    """
    local = local_folder(main_folder, folder, base)
    local.mkdir(parents=True, exist_ok=True)

    for path in small_files(folder):
        target = local / path.name

        if _same(path, target):
            continue

        if progress is not None:
            progress(path.name)

        partial = target.with_name(f".{target.name}.part")
        shutil.copy2(path, partial)  # copy2 keeps the time, which `state` reads
        os.replace(partial, target)

    return local


def delete(main_folder: str | Path, folder: str | Path, base=None) -> int:
    """
    Delete the local copy of test `folder`: its files only, as the folders
    inside it are other tests (a sweep's runs). Returns how many bytes it
    freed.
    """
    local = local_folder(main_folder, folder, base)
    freed = 0

    if not local.is_dir():
        return freed

    for path in local.iterdir():
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()

    # The folder goes when nothing else is in it
    for empty in (local, *local.parents):
        if empty == cache_root(main_folder, base).parent:
            break

        try:
            empty.rmdir()
        except OSError:
            break

    return freed


def cache_folder(base: None | Path = None) -> Path:
    """The folder of every copy on this computer, all main folders."""
    return (base or default_base()) / CACHE_FOLDER


def cache_size(base: None | Path = None) -> int:
    """How many bytes every copy on this computer takes, all main folders."""
    root = cache_folder(base)

    if not root.is_dir():
        return 0

    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def human_size(size: float) -> str:
    """`1234567` as `"1.2 MB"`."""
    units = ("B", "KB", "MB", "GB", "TB")
    step = 0

    while size >= 1000 and step < len(units) - 1:
        size /= 1000
        step += 1

    return f"{size:.0f} B" if step == 0 else f"{size:.1f} {units[step]}"
