"""
The olfactometer's own files, copied into each loop folder at Submit.

The olfactometer software keeps three folders on its computer, and a loop
needs a piece of each to be read later, anywhere:
- `config`, its settings, among them `Program.config` with every program's
definition (copied whole, it is small);
- `log`, whose `Info/<yyyymm>/Program_<yyyymmdd>.txt` says when each program
started (copied for the loop's day);
- `data`, one folder per program run, named for when it started
(`2026_08_14-13_58_59/...`), holding its Events file (copied for the runs
that started while the loop's sync file was recording).

They land in `<loop>/olfactometer/config`, `log` and `data`, where ingestion
reads them, so nothing has to be copied by hand or by a nightly script.

Where the three folders are on this computer comes from `~/.odyn/settings.json`
(keys `olfactometer_config`, `olfactometer_log`, `olfactometer_data`), with the
olfactometer PC's folders as defaults:
```json
{"olfactometer_data": "D:/OM_Data"}
```

No Qt in here: `submit` and `check` call it on the database thread.
"""

from __future__ import annotations

import json
import re
import shutil

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..database import OLFACTOMETER_FOLDER
from .draft import FOLDER_NAME, Draft

SETTINGS_FILE = "settings.json"

DEFAULT_FOLDERS = {
    "olfactometer_config": r"C:\Users\MossLab\Documents\Olfactometer",
    "olfactometer_log": r"C:\Program Files (x86)\RWD\Olfactometer\Log",
    "olfactometer_data": r"C:\Program Files (x86)\RWD\OM_Data",
}

# A program run's folder is named for when it started
RUN_NAME = re.compile(r"(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})")

# A run started up to this long before the recording still belongs to it
# (the two are started by hand, in either order)
EARLY = timedelta(minutes=2)


@dataclass(frozen=True)
class OlfactometerFolders:
    """Where the olfactometer software keeps its files on this computer."""

    config: Path
    log: Path
    data: Path

    @classmethod
    def read(cls, folder: None | Path = None) -> OlfactometerFolders:
        """
        The folders `settings.json` names in `folder` (`~/.odyn` if left
        out), or the defaults.
        """
        path = (folder or Path.home() / FOLDER_NAME) / SETTINGS_FILE
        settings: dict[str, Any] = {}

        if path.is_file():
            loaded = json.loads(path.read_text())
            settings = loaded if isinstance(loaded, dict) else {}

        def folder(key: str) -> Path:
            return Path(str(settings.get(key) or DEFAULT_FOLDERS[key]))

        return cls(
            config=folder("olfactometer_config"),
            log=folder("olfactometer_log"),
            data=folder("olfactometer_data"),
        )

    def runs_during(self, start: datetime, stop: datetime) -> list[Path]:
        """The program runs (folders in `data`) that started in the window."""
        if not self.data.is_dir():
            return []

        runs = []

        for path in sorted(self.data.iterdir()):
            started = run_start(path.name)

            if (
                path.is_dir()
                and started is not None
                and start - EARLY <= started <= stop
            ):
                runs.append(path)

        return runs

    def program_log(self, day: datetime) -> Path:
        """The log of when each program started on `day`."""
        return self.log / "Info" / f"{day:%Y%m}" / f"Program_{day:%Y%m%d}.txt"


def run_start(name: str) -> None | datetime:
    """When the run a folder is named for started, or `None`."""
    match = RUN_NAME.fullmatch(name)

    if match is None:
        return None

    return datetime(*(int(part) for part in match.groups()))


def recording_window(record: dict[str, Any]) -> None | tuple[datetime, datetime]:
    """
    When a sync recording ran, from its file's header when the file is on
    this computer (a replay keeps the original's clock), else from the times
    the draft noted. `None` for a recording that never stopped.
    """
    local = Path(str(record.get("local", "")))

    if local.is_file():
        from ..sync import read_header

        try:
            header = read_header(local)
        except (OSError, KeyError, ValueError):
            header = None

        if header is not None and header.start_time and header.stop_time:
            return header.start_time, header.stop_time

    try:
        return (
            datetime.fromisoformat(str(record["started"])),
            datetime.fromisoformat(str(record["stopped"])),
        )
    except (KeyError, ValueError):
        return None


def loop_window(entry: dict[str, Any]) -> None | tuple[datetime, datetime]:
    """When a loop's current sync recording ran (`None` without one)."""
    current = Draft.latest(entry)

    return None if current is None else recording_window(current)


@dataclass
class Copied:
    """What `copy_for_loop` put in the loop folder."""

    runs: list[str]
    log: None | str
    config: list[str]


def copy_for_loop(
    folders: OlfactometerFolders, loop_folder: Path, window: tuple[datetime, datetime]
) -> Copied:
    """
    Copy what the loop recorded during `window` needs into
    `<loop_folder>/olfactometer/{config,log,data}`. A run already there is
    left as it is; the day's log is copied again (it only grows during the
    day); the config folder is copied once, as it was when the loop was
    first submitted. Raises `OSError` when a copy fails.
    """
    target = loop_folder / OLFACTOMETER_FOLDER
    start, stop = window
    copied = Copied(runs=[], log=None, config=[])

    for run in folders.runs_during(start, stop):
        destination = target / "data" / run.name

        if not destination.exists():
            shutil.copytree(run, destination)
            copied.runs.append(run.name)

    log = folders.program_log(start)

    if log.is_file():
        (target / "log").mkdir(parents=True, exist_ok=True)
        shutil.copy2(log, target / "log" / log.name)
        copied.log = log.name

    config = target / "config"

    if folders.config.is_dir() and not config.exists():
        config.mkdir(parents=True)

        for path in sorted(folders.config.iterdir()):
            if path.is_file():
                shutil.copy2(path, config / path.name)
                copied.config.append(path.name)

    return copied
