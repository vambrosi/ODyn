"""
Work in progress for one session, held on disk until someone submits it.

A draft is what the app writes while an experiment day is running, in place
of the spreadsheet people used to leave open all day. Mice, odors, panels and
users go straight to the database when they are added; what stays here is
what only makes sense once the day is finished: the session's annotations,
each experiment's annotations and settings, and which vials were loaded.
Annotations are append-only, so half-typed values must not reach them, and
an experiment's annotations need its `exp_id`, which only exists once its
recordings have been copied and ingested at submit.

One JSON file per session, named for the moment it was started
(`20260914-113042.json`). Saving is atomic, so a crash costs at most the last
edit rather than the whole day, and an unfinished draft is simply still there
the next time the app opens.

**USAGE**
```python
draft = Draft.start(drafts_folder())        # today, animal not yet known
draft.identify(mouse_id=442)                # once someone picks it

draft.update_session(goal="10x pre/post ket/xyl", mouse_weight_g=25.1)
draft.set_experiment("e1", exp_type="loop", objective=20, fov_depth_um=-55)
draft.set_vials([1, 2, 3, 4])

Draft.pending(drafts_folder())   # everything not yet submitted
```

Submitting is elsewhere: this module knows nothing about the database, which is
what lets it be read and written without one.
"""

from __future__ import annotations

import json
import os
import re

from datetime import date as Date, datetime
from pathlib import Path
from typing import Any

# Bumped when the stored shape changes in a way a reader must notice. A draft
# written by a newer app is refused rather than half-read.
DRAFT_VERSION = 2

FOLDER_NAME = ".odyn"
DRAFTS = "drafts"

# The keys of an experiment entry that are not annotations
EXPERIMENT_SETTINGS = ("name", "exp_type", "objective", "submitted_exp_id")

# What an experiment folder is called for each kind of recording
EXPERIMENT_TYPES = ("loop", "grab", "zstack")

# A draft is named for when it was started, not for what it is about: the mouse
# is picked later and can be corrected, and two sessions can be open before
# either has been identified.
NAME_PATTERN = re.compile(r"\d{8}-\d{6}(-\d+)?\.json")


def drafts_folder() -> Path:
    """
    Where drafts live: `~/.odyn/drafts`.

    Local to the machine on purpose. Drafts are half-written and belong to the
    person typing them, so they do not go on the shared drive with the database.
    """
    return Path.home() / FOLDER_NAME / DRAFTS


def _check_date(value: str) -> str:
    """A date as `YYYY-MM-DD`, or `ValueError` saying what was wrong."""
    if isinstance(value, (Date, datetime)):
        return value.strftime("%Y-%m-%d")

    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"A session date is written 'YYYY-MM-DD', not {value!r}."
        ) from None


class Draft:
    """
    One session being filled in. Every change is written to disk immediately.

    Read the parts through `session`, `experiments` and `vial_ids`; change
    them through the `update_*` and `set_*` methods, which save as they go.
    """

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    # ----------------------------------------------------------------- #
    # Opening and listing
    # ----------------------------------------------------------------- #

    @classmethod
    def start(
        cls,
        folder: Path | str,
        *,
        mouse_id: None | int = None,
        date: None | str = None,
        user: None | str = None,
    ) -> Draft:
        """
        Begin a session, named for the moment it was started.

        `mouse_id` is optional because a session is usually opened before anyone
        picks the animal, and `date` defaults to today. Both can be set
        afterwards with `identify`. `user` is who opened it.
        """
        folder = Path(folder)
        started = datetime.now()

        draft = cls(
            folder / f"{started:%Y%m%d-%H%M%S}.json",
            {
                "version": DRAFT_VERSION,
                "mouse_id": None if mouse_id is None else int(mouse_id),
                "date": _check_date(date or started.date()),
                "user": user,
                "started_at": started.isoformat(sep=" ", timespec="seconds"),
                "saved_at": None,
                "session_id": None,
                "session": {},
                "experiments": [],
                "vial_ids": [],
            },
        )

        # Two sessions started in the same second would otherwise share a name.
        stem = draft.path.stem
        attempt = 0

        while draft.path.exists():
            attempt += 1
            draft.path = draft.path.with_name(f"{stem}-{attempt}.json")

        # Saved straight away, so a crash before the first keystroke still
        # leaves the session there to come back to.
        return draft.save()

    def identify(
        self,
        *,
        mouse_id: None | int = None,
        date: None | str = None,
        session_id: None | int = None,
    ) -> Draft:
        """
        Say which animal and which day this session is.

        Both are ordinary fields, so either can be corrected later without the
        draft changing its name or colliding with another. `session_id` is
        remembered once the session row exists, so a resubmit finds it.
        """
        if mouse_id is not None:
            self.data["mouse_id"] = int(mouse_id)

        if date is not None:
            self.data["date"] = _check_date(date)

        if session_id is not None:
            self.data["session_id"] = int(session_id)

        return self.save()

    @classmethod
    def load(cls, path: Path | str) -> Draft:
        """
        Read one draft file.

        Raises `ValueError` for a draft a newer version of the app wrote, rather
        than reading the parts it recognizes and silently dropping the rest.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        version = data.get("version")

        if version is None or version > DRAFT_VERSION:
            raise ValueError(
                f"{path.name} was written by a newer version of the app "
                f"(draft version {version}, this app reads {DRAFT_VERSION})."
            )

        return cls(path, data)

    @classmethod
    def pending(cls, folder: Path | str) -> list[Draft]:
        """
        Every unsubmitted draft that can be read, most recently saved first.

        One damaged file must not hide the rest, so anything unreadable is left
        out here and listed by `unreadable` instead. Check both before telling
        someone they have no unfinished work.
        """
        drafts = []

        for path in _draft_files(folder):
            try:
                drafts.append(cls.load(path))
            except (ValueError, OSError, json.JSONDecodeError):
                continue

        return sorted(drafts, key=lambda draft: draft.saved_at or "", reverse=True)

    @classmethod
    def unreadable(cls, folder: Path | str) -> list[tuple[Path, str]]:
        """
        The draft files that could not be read, and why.

        Show these: a draft nobody can open is still a day someone spent, and
        the file is right there to be recovered by hand.
        """
        broken = []

        for path in _draft_files(folder):
            try:
                cls.load(path)
            except (ValueError, OSError, json.JSONDecodeError) as failure:
                broken.append((path, str(failure)))

        return sorted(broken)

    # ----------------------------------------------------------------- #
    # What it holds
    # ----------------------------------------------------------------- #

    @property
    def mouse_id(self) -> None | int:
        """The animal, or `None` while the session is still unidentified."""
        stored = self.data.get("mouse_id")

        return None if stored is None else int(stored)

    @property
    def date(self) -> str:
        return self.data["date"]

    @property
    def user(self) -> None | str:
        return self.data.get("user")

    @property
    def session_id(self) -> None | int:
        """The session row, once it exists in the database."""
        stored = self.data.get("session_id")

        return None if stored is None else int(stored)

    @property
    def saved_at(self) -> None | str:
        return self.data.get("saved_at")

    @property
    def session(self) -> dict[str, Any]:
        """The fields that describe the day: goal, weight, headplate, notes."""
        return self.data.setdefault("session", {})

    @property
    def experiments(self) -> list[dict[str, Any]]:
        """One entry per experiment folder, each with a `name` like `'e1'`."""
        return self.data.setdefault("experiments", [])

    @property
    def vial_ids(self) -> list[int]:
        """The vials loaded in the olfactometer for the day's experiments."""
        return [int(v) for v in self.data.get("vial_ids", [])]

    @property
    def label(self) -> str:
        """How the session reads on screen, before and after it is identified."""
        who = "no mouse yet" if self.mouse_id is None else f"m{self.mouse_id}"

        return f"{who} {self.date}"

    @property
    def folder(self) -> str:
        """The session folder, `<date without dashes>/m<mouse>`, once identified."""
        if self.mouse_id is None:
            raise ValueError("The session has no mouse yet.")

        return f"{self.date.replace('-', '')}/m{self.mouse_id}"

    @property
    def is_empty(self) -> bool:
        """Whether any of the session's fields hold a value: is there anything
        here to submit?"""
        return not (self.session or self.experiments or self.vial_ids)

    @property
    def is_untouched(self) -> bool:
        """
        Whether the draft is still exactly as it was started: did anyone type
        into it at all?

        Weaker than `is_empty`, and the one to ask before throwing a draft
        away. The animal's number and the date are set through `identify`
        rather than into a section, and the number is usually the first thing
        picked, so a draft holding one has nothing to submit yet but is a
        session someone began rather than one opened by accident.
        """
        # `started_at` is `'2026-09-14 11:30:42'`, so its day is the first ten
        # characters -- what `date` holds unless someone changed it.
        started = str(self.data.get("started_at", ""))[:10]

        return self.is_empty and self.mouse_id is None and self.date == started

    def experiment(self, name: str) -> None | dict[str, Any]:
        """One experiment's entry, or `None` if it has not been started."""
        for entry in self.experiments:
            if entry.get("name") == name:
                return entry

        return None

    # ----------------------------------------------------------------- #
    # Changing it
    # ----------------------------------------------------------------- #

    def update_session(self, **fields: Any) -> Draft:
        """
        Set session fields. Passing `None` clears one rather than storing null.

        Returns the draft, so a caller can chain and still get it saved.
        """
        _merge(self.session, fields)

        return self.save()

    def set_experiment(self, name: str, **fields: Any) -> Draft:
        """
        Set fields on one experiment, creating its entry if it is new.

        `name` is the folder under the session (`e1`, `grabs`, `zstacks`);
        `exp_type` and `objective` are settings, everything else is an
        annotation.
        """
        entry = self.experiment(name)

        if entry is None:
            entry = {"name": name, "exp_type": _default_type(name)}
            self.experiments.append(entry)
            self.experiments.sort(key=_experiment_order)

        _merge(entry, fields)

        return self.save()

    def remove_experiment(self, name: str) -> Draft:
        """Drop an experiment entry that was added by mistake."""
        self.data["experiments"] = [
            entry for entry in self.experiments if entry.get("name") != name
        ]

        return self.save()

    def set_vials(self, vial_ids: list[int]) -> Draft:
        """Record which vials were loaded for the day's experiments."""
        self.data["vial_ids"] = sorted({int(v) for v in vial_ids})

        return self.save()

    # ----------------------------------------------------------------- #
    # Writing it out
    # ----------------------------------------------------------------- #

    def save(self) -> Draft:
        """
        Write the draft to disk, atomically.

        Written to a temporary file in the same folder and then renamed, so a
        crash midway leaves the previous save intact rather than a half file.
        """
        self.data["saved_at"] = datetime.now().isoformat(sep=" ", timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        temporary = self.path.with_suffix(".json.writing")
        temporary.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        os.replace(temporary, self.path)

        return self

    def archive(self) -> Path:
        """
        Move a submitted draft aside, and return where it went.

        Kept rather than deleted: it is the only record of what was typed, as
        opposed to what the database made of it, and it is small. The draft
        keeps pointing at the moved file, so a later save cannot bring it
        back among the pending ones.
        """
        destination = self.path.parent / "submitted" / self.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.path, destination)
        self.path = destination

        return destination

    def discard(self) -> None:
        """Delete the draft. For one started by mistake, not for a submitted one."""
        self.path.unlink(missing_ok=True)

    def __repr__(self) -> str:
        return f"<Draft {self.label} ({len(self.experiments)} exp)>"


def _draft_files(folder: Path | str) -> list[Path]:
    """
    The draft files in a folder, ignoring anything else that lives there.

    Only the top level, so archived drafts under `submitted/` are not offered
    again, and only names of the expected shape, so a half-written `.writing`
    file is never mistaken for one.
    """
    folder = Path(folder)

    if not folder.is_dir():
        return []

    return [path for path in folder.glob("*.json") if NAME_PATTERN.fullmatch(path.name)]


def _merge(into: dict[str, Any], fields: dict[str, Any]) -> None:
    """Apply changes, treating `None` as 'clear this' rather than a value."""
    for key, value in fields.items():
        if value is None:
            into.pop(key, None)
        else:
            into[key] = value


def _default_type(name: str) -> str:
    """What a folder name says the recording is: `e1` a loop, `grabs` grabs."""
    if name.startswith("zstack"):
        return "zstack"

    if name.startswith("grab"):
        return "grab"

    return "loop"


def _experiment_order(entry: dict[str, Any]):
    """`e2` sorts after `e1` and before `e10`; anything unnumbered goes last."""
    match = re.fullmatch(r"e(\d+)", str(entry.get("name", "")))

    return (0, int(match.group(1))) if match else (1, str(entry.get("name", "")))
