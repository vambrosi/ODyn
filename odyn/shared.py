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
"""

from __future__ import annotations

import json
import os
import time
import uuid

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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # A name of its own, so two writers never share a half-written file; not
    # ending in `.json`, so no listing of the folder takes it for one
    hidden = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.partial"

    try:
        with open(hidden, "w", encoding="utf-8", newline="") as file:
            json.dump(content, file, **{"indent": 2, "default": str, **dump})

        replace(hidden, path)

    except BaseException:
        hidden.unlink(missing_ok=True)
        raise

    return path


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
