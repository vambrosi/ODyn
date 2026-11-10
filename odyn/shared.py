"""
JSON files the programs share on the network drive (requests, records, the
queue, views) and those kept on one computer (drafts), all read and written
here.

A file is written whole under a hidden name, then renamed over the old one,
so a reader sees the old file or the new one, never half of either. Readers
therefore take no lock; a file with several writers takes `file_lock(path)`
around its read-modify-write. Files are UTF-8 with `\\n` line ends on every
OS, so a file written on Windows reads the same on the cluster.

File times are set by the server that keeps the share, so files written from
every machine share one clock; `server_now` reads that clock. Compare a file
time with it, never with this machine's own clock.

**USAGE**
```python
write_json(folder / "heartbeat.json", {"started_at": "2026-11-03 12:00:00"})
beat = read_json(folder / "heartbeat.json")
```

Messages that a program works through (requests to the cluster, submitted
drafts, QC claims) wait in an `Inbox`.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Windows refuses to replace a file another program has open, and a reader
# holds one open for a moment: the rename is tried again for this long
REPLACE_RETRY_S = 2.0


def read_json(path: str | Path) -> Any:
    """What the JSON file at `path` holds. Raises `OSError` or `ValueError`."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, content: Any, **dump: Any) -> Path:
    """
    Write `content` to `path` as indented JSON, all or nothing (see the module
    docstring), creating the folder if needed. Values JSON has no type for
    (a date, a `Path`) are written as text. `dump` goes to `json.dump`
    (`sort_keys=True`, ...). Returns `path`.
    """
    with atomic_write(path) as hidden:
        with open(hidden, "w", encoding="utf-8", newline="") as file:
            json.dump(content, file, **{"indent": 2, "default": str, **dump})

    return Path(path)


@contextmanager
def atomic_write(path: str | Path) -> Generator[Path]:
    """
    A hidden path beside `path` to write the file to, put in place of `path`
    when the block ends, or deleted if the block raises: a reader finds the
    old file or the new one, never half of either. Creates the folder if
    needed.

    **USAGE**
    ```python
    with atomic_write(folder / "mean.tif") as hidden:
        tifffile.imwrite(hidden, image)
    ```
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # A name of its own, so two writers never share a half-written file; not
    # ending as the file does, so no listing of the folder takes it for one
    hidden = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.partial"

    try:
        yield hidden
        replace(hidden, path)
    except BaseException:
        hidden.unlink(missing_ok=True)
        raise


def server_now(folder: str | Path) -> float:
    """
    The time now on the clock of the server that keeps `folder`, as a Unix
    time: a small file is written there, its time read back, and the file
    deleted (three round trips).
    """
    probe = Path(folder) / f".now.{uuid.uuid4().hex[:8]}"

    try:
        probe.write_bytes(b"")
        return probe.stat().st_mtime
    finally:
        probe.unlink(missing_ok=True)


def replace(source: str | Path, target: str | Path) -> None:
    """`os.replace`, tried again for a moment while a reader has `target` open."""
    give_up = time.monotonic() + REPLACE_RETRY_S

    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() > give_up:
                raise

            time.sleep(0.05)


# --------------------------------------------------------------------------- #
# Inboxes
# --------------------------------------------------------------------------- #

DONE_FOLDER = "done"
FAILED_FOLDER = "failed"


class Inbox:
    """
    A folder of messages a program works through, oldest first: requests to
    the cluster, submitted drafts, QC claims. Each is a JSON file whose name
    sorts by age; the program files each away when done with it (`done/`,
    `failed/`, or another folder), together with its `sidecars`: files
    beside it named `<its name><suffix>` (a submitted draft has its
    `.state.json`).

    **USAGE**
    ```python
    inbox = Inbox(folder)
    path = inbox.put("20261103-120000-alice", {"step": "approve"})

    for path in inbox.items():
        ...
        inbox.file(path, DONE_FOLDER, {"applied_at": "2026-11-03 12:05:00"})
    ```
    """

    def __init__(self, folder: str | Path, sidecars: tuple[str, ...] = ()):
        self.folder = Path(folder)
        self.sidecars = sidecars

    def items(self, folder: None | str = None) -> list[Path]:
        """The messages waiting (or those filed in `folder`), oldest first."""
        where = self.folder if folder is None else self.folder / folder

        try:
            names = sorted(entry.name for entry in os.scandir(where))
        except OSError:  # no message yet
            return []

        return [
            where / name
            for name in names
            if name.endswith(".json")
            and not name.startswith(".")  # being written (`write_json`)
            and not name.endswith(self.sidecars)
        ]

    def put(self, name: str, content: Any, **dump: Any) -> Path:
        """Leave the message `<name>.json`, replacing one of that name."""
        return write_json(self.folder / f"{name}.json", content, **dump)

    def file(
        self,
        path: Path,
        folder: str,
        extra: None | dict[str, Any] = None,
        name: None | str = None,
    ) -> Path:
        """
        Move the message at `path` and its sidecars into `folder`, with
        `extra` added to the message when it reads, as `<name>.json` when
        given (for messages whose names come back). Returns its new path.
        """
        target = path.parent / folder
        target.mkdir(exist_ok=True)
        moved = target / (path.name if name is None else f"{name}.json")

        try:
            if extra is None:
                raise ValueError

            write_json(moved, {**read_json(path), **extra})
            path.unlink()

        except (OSError, ValueError, TypeError):  # unreadable: moved as it is
            replace(path, moved)

        stem = path.name[: -len(".json")]
        new_stem = moved.name[: -len(".json")]

        for suffix in self.sidecars:
            sidecar = path.parent / f"{stem}{suffix}"

            if sidecar.exists():
                replace(sidecar, target / f"{new_stem}{suffix}")

        return moved
