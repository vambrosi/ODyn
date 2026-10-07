"""
Work in progress for one session, held on disk until someone submits it.

A draft is what the app writes throughout an imaging day. This replaces the
log spreadsheet that was created for each mice and day.

What goes here: the session's annotations, its parts' (each loop folder `e1`,
`e2`, ..., and the `grabs` and `zstacks` folders) annotations and settings,
and which vials were loaded. What doesn't go here: Mice, odors, panels, and
users (those are added directly to the database).

More than buffers, these files serve as a temporary state where the data might
not satisfy the database checks, or is otherwise inconsistent, and so it can't
be submitted, but it still needs to be kept, in case the app is closed by
mistake, for example. In the same vein, since annotations are append-only, the
drafts help avoid half-typed values being added to the database.

**FORMAT**
One JSON file per session, named for the datetime it was started (e.g.
`20260914-113042.json`). Default drafts folder is `~/.odyn/drafts` (local).

**USAGE**
```python
draft = Draft.start(drafts_folder())        # today, animal not yet known
draft.identify(mouse_id=442)                # once someone picks it

draft.update_session(goal="10x pre/post ket/xyl", mouse_weight_g=25.1)
draft.set_part("e1", objective=20, z_landmark="surface of right bulb")
draft.set_part("e1", conditions=[
    {"start": 3, "end": 40, "condition": "baseline"},     # acquisitions by
    {"start": 41, "end": 80, "condition": "ket/xyl"},     # their TIFF number
])
draft.set_part("zstacks", objective=20)
draft.set_file("zstacks", "20260914_m442_zstack_00001.tif", loop="e1")
draft.set_vials([1, 2, 3, 4])

Draft.pending(drafts_folder())   # everything not yet submitted or removed

removals = Removals(drafts_folder())
draft.remove_part("e1")
removals.push(draft, "e1")
removals.undo()                  # e1 is back
```

**NOTES**
Saving is atomic: so a crash costs at most the last edit rather than the whole
day, and an unfinished draft is simply still there the next time the app opens.

Nothing typed is ever deleted: removing a session or a part only marks
it `removed` (hidden everywhere, kept on disk), and `Removals` is the stack
that undoes the latest of those. Submitted drafts move to `~/.odyn/submitted`,
outside the drafts folder, so the record survives clearing the drafts.

This module is database-independent, so submitting is not part of it. This lets
drafts be read and written even without a database.
"""

from __future__ import annotations

import json
import os
import re

from datetime import date as Date, datetime
from pathlib import Path
from typing import Any

# Bumped when the stored shape changes in a way a reader must notice.
# A draft written by a newer app is refused rather than half-read; an older
# one is brought up to date as it is read (`_upgrade`).
DRAFT_VERSION = 4

FOLDER_NAME = ".odyn"
DRAFTS = "drafts"
SUBMITTED = "submitted"

# The undo stack of removals, in the drafts folder
REMOVALS = "removed.json"

# The keys of a part entry that are not annotations
PART_SETTINGS = (
    "name",
    "kind",
    "objective",
    "submitted_id",
    "removed",
    "recordings",
    "conditions",
    "conditions_submitted",
    "files",
    "folder_made",
)

# The keys of a file's entry (in a grabs or z-stacks part) that are not
# annotations
FILE_SETTINGS = ("loop", "submitted_id")

# The kinds of part, and the folder of the two a session has one of
KINDS = ("loop", "grab", "zstack")
FOLDER_OF = {"grab": "grabs", "zstack": "zstacks"}

# How the app names each kind of part in a sentence ("Remove loop ...")
KIND_WORDS = {"loop": "loop", "grab": "grabs", "zstack": "z-stacks"}

# A draft is named for when it was started because:
# - the mouse is picked later and can be corrected;
# - two sessions can be open before either has been identified.
NAME_PATTERN = re.compile(r"\d{8}-\d{6}(-\d+)?\.json")


def drafts_folder() -> Path:
    """
    Where drafts live (currently, `~/.odyn/drafts`).

    These don't go in the shared drive because they are owned by the user
    producing them, and they can be written even without the DB connection.
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
            f"Session data format is 'YYYY-MM-DD', not {value!r}."
        ) from None


class Draft:
    """
    One session being filled in. Every change is written to disk immediately.

    Read the parts through `session`, `parts` and `vial_ids`; change
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

        `mouse_id` is optional to allow the user to write other entries
        before the mouse is picked. `date` defaults to today. Both can be
        set afterwards with `identify`.
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
                "parts": [],
                "vial_ids": [],
            },
        )

        # Two sessions started in the same second would otherwise share a name.
        # (Not likely to happen, but just in case.)
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
        Which animal and which day this session is about.

        Both are ordinary fields, so either can be corrected later without the
        draft changing its name or colliding with another, until a part's
        folder or a recording was made (`fixed`): those are named and placed for
        that mouse and day, so a change then raises `ValueError` (remove the
        session and start again).
        `session_id` is remembered once the session row exists, so a resubmit
        finds it.
        """
        changes = (mouse_id is not None and int(mouse_id) != self.mouse_id) or (
            date is not None and _check_date(date) != self.date
        )

        if changes and self.fixed:
            raise ValueError(
                "Folders or recordings were made for this session, so its mouse "
                "and date are fixed. Remove the session and start a new one to "
                "change them."
            )

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
        than reading what it recognizes and silently dropping the rest.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        version = data.get("version")

        if version is None or version > DRAFT_VERSION:
            raise ValueError(
                f"{path.name} was written by a newer version of the app "
                f"(draft version {version}, this app reads {DRAFT_VERSION})."
            )

        return cls(path, _upgrade(data))

    @classmethod
    def pending(cls, folder: Path | str) -> list[Draft]:
        """
        Every draft that was not submitted nor removed and can be read. Most
        recently saved first.

        Damaged files should not hide the valid ones, so anything unreadable is
        left out here and listed by `unreadable` instead. Should check both
        before telling the user that they have no unfinished work.
        """
        drafts = []

        for path in _draft_files(folder):
            try:
                draft = cls.load(path)

            except (ValueError, OSError, json.JSONDecodeError):
                continue

            if not draft.removed:
                drafts.append(draft)

        return sorted(drafts, key=lambda draft: draft.saved_at or "", reverse=True)

    @classmethod
    def unreadable(cls, folder: Path | str) -> list[tuple[Path, str]]:
        """
        The draft files that could not be read, and why.

        Relevant in case they need to be recovered by hand.
        """
        broken = []

        for path in _draft_files(folder):
            try:
                cls.load(path)
            except (ValueError, OSError, json.JSONDecodeError) as failure:
                broken.append((path, str(failure)))

        return sorted(broken)

    # ----------------------------------------------------------------- #
    # Draft fields
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
        """The fields that describe the session (the mouse and the setup)"""
        return self.data.setdefault("session", {})

    @property
    def removed(self) -> bool:
        """Whether the whole session was removed (and can still be restored)."""
        return bool(self.data.get("removed"))

    @property
    def parts(self) -> list[dict[str, Any]]:
        """
        One entry per part of the session (removed ones left out): each loop
        folder, named like `'e1'`, then `grabs` and `zstacks` if added. This
        gives entries themselves, so changing one changes the draft (then
        `save`).
        """
        return [e for e in self._entries if not e.get("removed")]

    @property
    def _entries(self) -> list[dict[str, Any]]:
        """Every part entry, removed ones included."""
        return self.data.setdefault("parts", [])

    @property
    def vial_ids(self) -> list[int]:
        """The vials loaded in the olfactometer for that session."""
        return [int(v) for v in self.data.get("vial_ids", [])]

    @property
    def label(self) -> str:
        """How the session appears on the screen."""
        who = "no mouse yet" if self.mouse_id is None else f"m{self.mouse_id}"

        return f"{who} {self.date}"

    @property
    def where(self) -> str:
        """
        How the app names the session everywhere: its folder, `20260929/m442`,
        or `20260929/m?` before it has a mouse.
        """
        if self.mouse_id is None:
            return f"{self.date.replace('-', '')}/m?"

        return self.folder

    @property
    def folder(self) -> str:
        """The session folder, `<date without dashes>/m<mouse>`, once identified."""
        if self.mouse_id is None:
            raise ValueError("The session has no mouse yet.")

        return f"{self.date.replace('-', '')}/m{self.mouse_id}"

    @property
    def is_empty(self) -> bool:
        """Whether any of the session's fields hold a value (used to see, for
        example, if there anything to submit)."""
        return not (self.session or self.parts or self.vial_ids)

    @property
    def is_untouched(self) -> bool:
        """
        Whether the draft is still exactly as it was started.

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

    def part(self, name: str) -> None | dict[str, Any]:
        """One part's entry, or `None` if it was not started (or removed)."""
        for entry in self.parts:
            if entry.get("name") == name:
                return entry

        return None

    def describe(self, name: str) -> str:
        """A part as the app says it: `loop 20260929/m442/e1`, `grabs ...`."""
        entry = next((e for e in self._entries if e.get("name") == name), {})
        kind = entry.get("kind", _default_type(name))

        return f"{KIND_WORDS.get(kind, 'part')} {self.where}/{name}"

    def files(self, name: str) -> dict[str, dict[str, Any]]:
        """What was noted per TIFF of part `name` (grabs, z-stacks), by file name."""
        entry = self.part(name)

        return {} if entry is None else entry.get("files", {})

    def next_loop(self, taken: tuple[int, ...] = ()) -> str:
        """
        The next loop folder's name, `e<n>`: one past every loop number in the
        draft (removed ones too, they keep their names) and in `taken` (the
        numbers already on the share, say).
        """
        numbers = [
            int(name[1:]) for name in self.names_taken() if re.fullmatch(r"e\d+", name)
        ]

        return f"e{max([*numbers, *taken], default=0) + 1}"

    @property
    def fixed(self) -> bool:
        """
        Whether the mouse and date can no longer change: a part's folder was
        made on the share, or a recording was (removed parts too).
        """
        return self.recorded or any(e.get("folder_made") for e in self._entries)

    @property
    def recorded(self) -> bool:
        """Whether any part (removed ones too) has a sync recording."""
        return any(e.get("recordings") for e in self._entries)

    @staticmethod
    def latest(entry: dict[str, Any]) -> None | dict[str, Any]:
        """A part entry's recording in use: the last not replaced."""
        kept = [r for r in entry.get("recordings", []) if not r.get("replaced")]

        return kept[-1] if kept else None

    def recordings(self, name: str) -> list[dict[str, Any]]:
        """Every recording of part `name` (removed or not), oldest first."""
        for entry in self._entries:
            if entry.get("name") == name:
                return entry.get("recordings", [])

        return []

    def is_removed(self, name: str) -> bool:
        """Whether part `name` was removed (its name stays taken)."""
        return any(e.get("name") == name and e.get("removed") for e in self._entries)

    def names_taken(self) -> list[str]:
        """Every part name in the draft, removed ones included."""
        return [str(e.get("name")) for e in self._entries]

    # ----------------------------------------------------------------- #
    # Changing the draft
    # ----------------------------------------------------------------- #

    def update_session(self, **fields: Any) -> Draft:
        """
        Set session fields. Passing `None` clears one rather than storing null.

        Returns the draft, so a caller can chain and still get it saved.
        """
        _merge(self.session, fields)

        return self.save()

    def set_part(self, name: str, **fields: Any) -> Draft:
        """
        Set fields on one part, creating its entry if it is new.

        `name` is the folder under the session (`e1`, `grabs`, `zstacks`), which
        says its `kind`; `objective` is a setting, everything else is an
        annotation. Raises `ValueError` for the name of a removed part:
        restore it instead, so it comes back as it was.
        """
        if self.is_removed(name):
            raise ValueError(
                f"'{name}' was removed: undo that to bring it back, "
                "or pick another name."
            )

        entry = self.part(name)

        if entry is None:
            entry = {"name": name, "kind": _default_type(name)}
            self._entries.append(entry)
            self._entries.sort(key=_part_order)

        _merge(entry, fields)

        return self.save()

    def set_file(self, name: str, file: str, **fields: Any) -> Draft:
        """
        Set fields on one TIFF of part `name` (grabs, z-stacks): its
        annotations, and for a z-stack the `loop` it was taken for.
        """
        entry = self.part(name)

        if entry is None:
            raise ValueError(f"There is no part '{name}'.")

        noted = entry.setdefault("files", {}).setdefault(file, {})
        _merge(noted, fields)

        return self.save()

    def add_recording(self, name: str, record: dict[str, Any]) -> Draft:
        """
        Note a sync recording of part `name`. A loop's replaces any earlier
        one (those are kept, marked `replaced`); the grabs' add up, since each
        covers the grabs taken while it ran. `record` holds at least the
        `file` name, its `local` path and its `share` path (relative to the
        main folder); `stopped` and `copied` are added as they happen.
        """
        entry = self.part(name)

        if entry is None:
            raise ValueError(f"There is no part '{name}' to record into.")

        if entry.get("kind") == "zstack":
            raise ValueError("Z-stacks are not recorded.")

        recordings = entry.setdefault("recordings", [])

        if entry.get("kind", "loop") == "loop":
            for earlier in recordings:
                earlier["replaced"] = True

        recordings.append(dict(record))

        return self.save()

    def update_recording(self, name: str, file: str, **fields: Any) -> Draft:
        """Set fields on one recording of part `name` (removed or not)."""
        for entry in self._entries:
            if entry.get("name") != name:
                continue

            for record in entry.get("recordings", []):
                if record.get("file") == file:
                    _merge(record, fields)
                    return self.save()

        raise ValueError(f"No recording '{file}' in part '{name}'.")

    def remove_part(self, name: str) -> Draft:
        """Hide a part, keeping what was typed into it."""
        entry = self.part(name)

        if entry is not None:
            entry["removed"] = True

        return self.save()

    def restore_part(self, name: str) -> bool:
        """
        Bring a removed part back and returns whether there was
        actually something to bring back.
        """
        for entry in self._entries:
            if entry.get("name") == name and entry.get("removed"):
                entry.pop("removed")
                self.save()
                return True

        return False

    def remove(self) -> Draft:
        """Hide the whole session, keeping its file and what was typed."""
        self.data["removed"] = True

        return self.save()

    def restore(self) -> bool:
        """Bring a removed session back (`False` if it was not removed)."""
        if not self.removed:
            return False

        self.data.pop("removed")
        self.save()

        return True

    def set_vials(self, vial_ids: list[int]) -> Draft:
        """Record which vials were loaded for the day's parts."""
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
        Move a submitted draft to the `submitted` folder beside the drafts
        folder, and return where it went.

        Kept rather than deleted: it is the only record of what was typed, as
        opposed to what the database made of it, and it is small (so worth
        keeping). The draft keeps pointing at the moved file, so a later
        save cannot bring it back among the pending ones.
        """
        destination = self.path.parent.parent / SUBMITTED / self.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.path, destination)
        self.path = destination

        return destination

    def __repr__(self) -> str:
        return f"<Draft {self.label} ({len(self.parts)} parts)>"


class Removals:
    """
    The undo stack of removed sessions and parts, kept in
    `removed.json` in the drafts folder so it outlives the app.

    Each entry names a draft file and, for a part, its name. `undo`
    restores the latest entry that still applies and drops the ones that do
    not (the draft deleted by hand or submitted, or restored another way).

    Read from disk on every call, so two windows on one machine share it.
    """

    def __init__(self, folder: Path | str):
        self.folder = Path(folder)
        self.path = self.folder / REMOVALS

    def push(self, draft: Draft, part: None | str = None) -> None:
        """Record that `draft` (or its `part`) was just removed."""
        entries = self._read()
        entries.append(
            {
                "draft": draft.path.name,
                "part": part,
                "removed_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
            }
        )
        self._write(entries)

    def next(self) -> None | tuple[Draft, None | str]:
        """
        What `undo` would restore: `(draft, part or None)`. Entries
        that no longer apply are dropped on the way.
        """
        entries = self._read()
        dropped = False
        found = None

        while entries and found is None:
            found = self._applies(entries[-1])

            if found is None:
                entries.pop()
                dropped = True

        if dropped:
            self._write(entries)

        return found

    def undo(self, known: list[Draft] = ()) -> None | tuple[Draft, None | str]:
        """
        Restore the latest removal that still applies, and say what it was.

        A draft in `known` with the same file is restored in place of a fresh
        copy, so a caller holding that draft sees the change.
        """
        found = self.next()

        if found is None:
            return None

        draft, part = found
        draft = next((d for d in known if d.path == draft.path), draft)

        if part is None:
            draft.restore()
        else:
            draft.restore_part(part)

        self._write(self._read()[:-1])

        return draft, part

    def _applies(self, entry: dict[str, Any]) -> None | tuple[Draft, None | str]:
        path = self.folder / str(entry.get("draft", ""))
        part = entry.get("part", entry.get("experiment"))  # version 3 called it that

        if not NAME_PATTERN.fullmatch(path.name) or not path.is_file():
            return None

        try:
            draft = Draft.load(path)
        except (ValueError, OSError, json.JSONDecodeError):
            return None

        if part is None:
            return (draft, None) if draft.removed else None

        # The session must be there too, or the part would come back
        # somewhere nobody can see it
        if draft.removed or not draft.is_removed(str(part)):
            return None

        return draft, str(part)

    def _read(self) -> list[dict[str, Any]]:
        try:
            entries = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return []

        return (
            [e for e in entries if isinstance(e, dict)]
            if isinstance(entries, list)
            else []
        )

    def _write(self, entries: list[dict[str, Any]]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.writing")
        temporary.write_text(json.dumps(entries, indent=2))
        os.replace(temporary, self.path)


def _draft_files(folder: Path | str) -> list[Path]:
    """
    The draft files in a folder, ignoring anything else that lives there.

    Only names of the expected shape, so a half-written `.writing` file or
    the `removed.json` stack is never mistaken for one.
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
    """What a folder name says the part is: `e1` a loop, `grabs` grabs."""
    if name.startswith("zstack"):
        return "zstack"

    if name.startswith("grab"):
        return "grab"

    return "loop"


def _part_order(entry: dict[str, Any]):
    """`e2` sorts after `e1` and before `e10`; anything unnumbered goes last."""
    match = re.fullmatch(r"e(\d+)", str(entry.get("name", "")))

    return (0, int(match.group(1))) if match else (1, str(entry.get("name", "")))


def _upgrade(data: dict[str, Any]) -> dict[str, Any]:
    """
    A draft from an older version of the app, in the current shape.

    Version 3 called a session's parts experiments (`experiments`, with
    `exp_type` and `submitted_exp_id`).
    """
    if data.get("version", DRAFT_VERSION) >= DRAFT_VERSION:
        return data

    parts = data.pop("experiments", [])

    for entry in parts:
        if "exp_type" in entry:
            entry["kind"] = entry.pop("exp_type")

        if "submitted_exp_id" in entry:
            entry["submitted_id"] = entry.pop("submitted_exp_id")

    data["parts"] = parts
    data["version"] = DRAFT_VERSION

    return data
