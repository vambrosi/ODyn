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

# How much a download copies at a time, between two progress reports
CHUNK_BYTES = 4 * 1024 * 1024

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


def pending(main_folder: str | Path, folder: str | Path, base=None) -> list[Path]:
    """The files of test `folder` a download would copy: not here, or changed."""
    local = local_folder(main_folder, folder, base)

    return [path for path in small_files(folder) if not _same(path, local / path.name)]


def download(
    main_folder: str | Path,
    folder: str | Path,
    base=None,
    progress: None | Callable[[str, int], None] = None,
) -> Path:
    """
    Copy test `folder`'s small files here, those not here already or changed
    on the share (`pending`), and return the local folder. Each file is
    written under a temporary name and renamed when complete, so a copy cut
    short is never taken for a whole one. `progress(name, copied)` hears each
    file as it starts (`copied` 0) and every chunk copied (its bytes); an
    exception it raises stops the download, the partial file deleted.
    """
    local = local_folder(main_folder, folder, base)
    local.mkdir(parents=True, exist_ok=True)

    for path in pending(main_folder, folder, base):
        target = local / path.name

        if progress is not None:
            progress(path.name, 0)

        partial = target.with_name(f".{target.name}.part")

        def copied(count: int, name: str = path.name) -> None:
            if progress is not None:
                progress(name, count)

        try:
            _copy(path, partial, copied)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

        # The time only, which `state` reads. Not `copy2`: it also copies the
        # file flags, and macOS refuses a user the flags a network share sets
        # ("Operation not permitted")
        times = path.stat()
        os.utime(partial, ns=(times.st_atime_ns, times.st_mtime_ns))
        os.replace(partial, target)

    return local


def _copy(source: Path, target: Path, copied: Callable[[int], None]) -> None:
    """Copy `source`'s data to `target`, telling `copied` each chunk's bytes."""
    with open(source, "rb") as reading, open(target, "wb") as writing:
        while chunk := reading.read(CHUNK_BYTES):
            writing.write(chunk)
            copied(len(chunk))


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
