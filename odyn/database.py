import json
import math
import re
import sqlite3

from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, time, datetime, timedelta
from enum import IntEnum, IntFlag
from pathlib import Path
from sqlite3 import Cursor
from tifffile import TiffFile, TiffPage
from typing import Final

import pandas as pd

from tqdm.auto import tqdm

from .groups import Group
from .locking import DatabaseLock
from .migrate import SCHEMA_VERSION
from .sync import DecodedSync, decode, summary
from .utils import (
    ACQUISITION_TRIALS,
    ACQUISITIONS,
    ANNOTATION_KEYS,
    ANNOTATION_TARGETS,
    ANNOTATIONS,
    CHECK,
    CROSS,
    DB_TIMEOUT_S,
    EVENTS,
    GRABS,
    GROUP_LOOPS,
    INFO_FOLDER,
    MCOR_FILES,
    METHOD_CALLS,
    LINES,
    LOOP_VIALS,
    LOOPS,
    MICE,
    MOUSE_LINES,
    MOUSE_VIRUSES,
    ODORS,
    ODOR_COMPONENTS,
    OUTPUTS,
    OUTPUTS_FOLDER,
    PANEL_POSITIONS,
    PANELS,
    PROGRAMS,
    PROJECTS_FOLDER,
    SESSIONS,
    STREAMS,
    SYNC_BLOCKS,
    TRIALS,
    USERS,
    VIAL_ODORS,
    VIALS,
    VIRUSES,
    ZSTACKS,
    CallFrame,
    CallRecorder,
    Object,
    TableSpec,
    Value,
    database_path,
    get_user,
    logger,
    record_call,
)
from .utils import _method_calls_dataframe, _read_table

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# Tolerance for ScanImage settings silent rounding
WHOLE_TOLERANCE = 0.01

# TIFF metadata that is tier 3 in `create.sql`: recorded, but read by nothing in
# odyn, so it is stored as annotations instead of columns. Read out of the same
# dict as the columns so the across-TIFF consistency check still sees it.
RIG_ANNOTATIONS = ("laser_power_920", "laser_power_1040", "loop_acq_interval_s")

# The columns every recording table (loops, grabs, z-stacks) takes from a TIFF
CONTRACT = (
    "height_px",
    "width_px",
    "height_um",
    "width_um",
    "frames_per_slice",
    "frame_rate",
    "z_um",
)

# Real microns per micron ScanImage writes, by objective. ScanImage computes
# the field of view as if every recording were taken at 20x, so on 10x a real
# 100 um is written as 45 um (measured, not the nominal 2x). Recordings are
# stored in real microns; an objective missing here cannot be ingested.
OBJECTIVE_SCALE = {20: 1.0, 10: 100 / 45}

# Where the app copies the olfactometer's own files a loop needs, inside the
# loop folder: `config/` (its settings, `Program.config` among them), `log/`
# (the day's `Program_<yyyymmdd>.txt`) and `data/` (one folder per program
# run, with its Events file)
OLFACTOMETER_FOLDER = "olfactometer"

# Where the sync H5 of a loop lives, inside the loop folder (and of the grabs,
# inside the grabs folder)
SYNC_FOLDER = "sync"

# TODO: Make program types part of the database
PROGRAM_TYPES = [
    "fine 1",
    "fine 2",
    "coarse 1",
    "coarse 2",
    "passive",
    "warm-up",
    "short",
]


class IngestFlag(IntFlag):
    """
    call_flag bits for `Database.add_loop`, `add_grabs` and `add_zstacks` (bit
    0 reserved by `CallFlag.RAISED`).
    """

    ALREADY_IN_DB = 1 << 1  # recording already present, nothing inserted
    UNSUPPORTED_METADATA = 1 << 3  # TIFF does not have expected metadata format
    METADATA_CHANGED = 1 << 4  # TIFF metadata changed, skipped
    NOT_THE_SESSION = 1 << 5  # file names disagree with the session given
    WRONG_ACQ_STATE = 1 << 7  # a TIFF is not the kind asked for (loop/grab/z-stack)
    NO_PROGRAM_CONFIG = 1 << 8  # a program has no olfactometer definition
    STACK_LOOP = 1 << 9  # a loop of z-stacks, which cannot be processed yet


class SyncFlag(IntFlag):
    """
    call_flag bits for `Database.process_sync` and `process_grabs_sync` (bit 0
    reserved by `CallFlag.RAISED`).
    """

    NO_SYNC_FILE = 1 << 1  # nothing to process in <loop>/sync/
    MULTIPLE_SYNC_FILES = 1 << 2  # more than one H5 there and none was named
    INTERRUPTED = 1 << 3  # the recording was not stopped properly
    FRAGMENTS = 1 << 4  # short bursts were set aside
    BLOCK_COUNT_MISMATCH = 1 << 5  # bursts and acquisitions (or grabs) do not pair up
    TRIAL_COUNT_MISMATCH = 1 << 6  # odor bursts and trials do not pair up
    ODOR_PULSES = (
        1 << 7
    )  # a loop burst without exactly one valve pulse, a grab's with any
    CLOCK_SKEW = 1 << 8  # the two computers disagree beyond `max_skew_s`
    NO_CAMERA = 1 << 9  # an acquisition without camera strobes (no pupil)
    FRAME_COUNT = 1 << 10  # frames in a burst differ from the ScanImage setting
    NO_TRIALS = 1 << 11  # the loop has no trials to match the odor bursts to


class TrialPhase(IntEnum):
    NOT_IN_TRIAL = 0
    TRIAL_START = 1
    ODOR_WINDOW = 2
    INTERVAL = 3
    RESPONSE_WINDOW = 4
    TRIAL_END = 5


# --------------------------------------------------------------------------- #
# Main UI Class
# --------------------------------------------------------------------------- #


class Database(CallRecorder):
    """
    Creates and connects you to the database.

    **USAGE**
    ```python
    db = Database(main_folder, project="main-sync")
    ```

    **RELEVANT PROPERTIES**
    ```python
        db.groups             # `List` of `Group`s for processing/analysis

        db.mice               # `DataFrame` per table: mice, sessions, loops,
        db.loops              #   grabs, zstacks, acquisitions, streams,
        db.acquisitions       #   sync_blocks, programs, trials, events, odors,
        db.annotations        #   panels, vials, vial_odors, loop_vials,
        db.method_calls       #   annotation_keys, annotations, method_calls,
                              #   outputs, users, mcor_files
    ```

    **RELEVANT METHODS**
    ```python
        db.add_loop(...)                # Add a loop folder to a session
        db.add_grabs(...)               # ... or its grabs, or its z-stacks
        db.add_annotations(...)         # Record facts about a row
        db.annotations_for("loop")      # One column per annotation key
        db.latest_calls(method_name)    # `DataFrame` with `method_name` calls
        db.from_query(sql)              # `DataFrame` from a SQL `SELECT`
    ```
    """

    def __init__(
        self,
        path: str | Path,
        project: None | str = None,
        can_create: bool = False,
        user: None | str = None,
    ):
        """
        **PARAMETERS**
        - `path` is the main folder holding the session folders
        - `project` is a separate database in the same main folder, at
        `.odyn/projects/<project>.db`. `None` is the shared one.
        - `can_create` makes a new, empty database if none is found. Leave it
        `False` to get an error instead, so a typo in `path` or `project` does
        not quietly create a database somewhere else.
        - `user` is recorded as the author of every call made through this
        object. Leave it out to use the computer login (or `ODYN_USER`), pass
        it on a shared login or someone else's computer.

        **ALERT**
        Projects do not see each other. Two of them can hold the same
        loop and neither will know, so use one when work should be kept
        apart, not to split work that has to be compared later.
        """
        # Resolved so that every path built from it is absolute and symlink
        # free. Stored paths are relative to this, and the code that makes
        # them relative resolves its side, so leaving this one as given makes
        # relative_to fail on a relative main folder or through a symlink.
        self.main_folder = Path(path).resolve()
        self.project: Final[None | str] = project
        self.user: Final[str] = user or get_user()

        self.path: Final[Path] = database_path(self.main_folder, project)

        # Database has a default group to record its calls
        self.group_id: Final[int] = 0

        # Initialize "private" variables
        self._call_stack: list[CallFrame] = []
        self._tables: dict[str, pd.DataFrame] = {}
        self._groups: dict[int, Group] = {}  # Caches groups one-by-one

        # Checked before anything to not leave traces due to a typo.
        if not can_create and not _has_database(self.path):
            raise FileNotFoundError(
                f"No database at '{self.path}'. Check the main folder and the "
                "project name for typos, or use 'can_create=True'."
            )

        # Every use of the connection goes through `_locked()`.
        # See `locking.py` for more details.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = DatabaseLock(self.path)
        self._schema_checked = False

        with self._lock:
            # Decided under the lock, so two processes cannot both create it.
            # Connecting makes an empty file, hence the size check.
            is_new = not _has_database(self.path)

            if is_new and not can_create:
                raise FileNotFoundError(f"Could not find a database at '{self.path}'")

            self._con = sqlite3.connect(self.path, timeout=DB_TIMEOUT_S)
            self._con.row_factory = sqlite3.Row

            if is_new:
                logger.info("Did not find a database!")
                logger.info("Creating database...")

                # Create schema and add default values
                with self._con as con:
                    with open(Path(__file__).parent / "create.sql") as f:
                        con.executescript(f.read())

                    # Fresh DB is already at the latest schema
                    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

                logger.info(f"Database created at: '{self.path.resolve()}'")

            else:
                logger.info(f"Connected to the database at: '{self.path.resolve()}'")

            self._check_schema_version()
            self._schema_checked = True

            self._con.execute("PRAGMA foreign_keys = ON;")
            self._data_version = self._con.execute("PRAGMA data_version;").fetchone()[0]

    def __del__(self):
        # `__init__` can raise before the connection exists (a bad `project`,
        # a schema mismatch), and __del__ still runs on the half-built object.
        connection = getattr(self, "_con", None)

        if connection is not None:
            connection.close()

    @contextmanager
    def _locked(self) -> Generator[sqlite3.Connection]:
        """
        Hold the database lock and get the connection. Nests freely.

        ```python
        with self._locked() as con:
            rows = con.execute("SELECT ...").fetchall()   # read everything here

        with self._locked() as con, con:                  # one transaction
            con.execute("INSERT ...")
        ```
        """
        with self._lock as outermost:
            # A process can stay open across a migration (a notebook left
            # running), and must not keep writing through the old schema.
            if outermost and self._schema_checked:
                self._check_schema_version()

            try:
                yield self._con

            finally:
                # A write left uncommitted would keep going after the lock is
                # released, which is exactly what the lock is there to prevent.
                if outermost and self._con.in_transaction:
                    self._con.rollback()
                    logger.error("Rolled back changes that were never committed.")

    @contextmanager
    def _writing(self) -> Generator[Cursor]:
        """
        One transaction with foreign keys on, for the writers below.

        ```python
        with self._writing() as cur:
            _db_insert(cur, "mice", {...})
        ```

        The caches are dropped on the way out, since something changed.
        """
        with self._locked() as con, con:
            cur = con.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")

            yield cur

        self._reset_caches()

    @property
    def project_folder(self) -> Path:
        """
        Folder holding the project files

        **USAGE**
        ```python
            db = Database(main_folder, project="project_name")
            outputs = db.project_folder / "outputs"
        ```

        All files created by a project (except mcor files) live in this folder.
        """
        if self.project is None:
            raise RuntimeError(
                "Only a project has a folder of its own. Open the database "
                "with 'Database(main_folder, project=\"name\")' to use one."
            )

        return self.main_folder / PROJECTS_FOLDER / self.project

    @property
    def files_folder(self) -> Path:
        """
        "Root" folder for this database's files
        """
        return self.main_folder if self.project is None else self.project_folder

    @property
    def outputs_folder(self) -> Path:
        """
        Where functions of this database save what they produce

        The project's own `outputs` folder when the database belongs to one,
        and the main folder's otherwise. It is made when something is saved.
        """
        return self.files_folder / OUTPUTS_FOLDER

    # ----------------------------------------------------------------------- #
    # SQLite Tables as DataFrames
    # ----------------------------------------------------------------------- #

    def _table(self, spec: TableSpec) -> pd.DataFrame:
        """The cached DataFrame of `spec`, read on first use."""
        with self._locked() as con:
            self._refresh_if_stale()

            if spec.name not in self._tables:
                self._tables[spec.name] = _read_table(con, spec, spec.all_rows)

            return self._tables[spec.name]

    @property
    def users(self) -> pd.DataFrame:
        """`DataFrame` with the names that can author a call"""
        return self._table(USERS)

    @property
    def mice(self) -> pd.DataFrame:
        """`DataFrame` with one row per mouse"""
        return self._table(MICE)

    @property
    def lines(self) -> pd.DataFrame:
        """`DataFrame` with one row per mouse line, indexed by its name"""
        return self._table(LINES)

    @property
    def viruses(self) -> pd.DataFrame:
        """`DataFrame` with one row per virus, indexed by its name"""
        return self._table(VIRUSES)

    @property
    def mouse_viruses(self) -> pd.DataFrame:
        """`DataFrame` with one row per mouse and virus it was injected with"""
        return self._table(MOUSE_VIRUSES)

    @property
    def mouse_lines(self) -> pd.DataFrame:
        """`DataFrame` with one row per mouse and line it carries"""
        return self._table(MOUSE_LINES)

    @property
    def sessions(self) -> pd.DataFrame:
        """`DataFrame` with one row per mouse and day"""
        return self._table(SESSIONS)

    @property
    def loops(self) -> pd.DataFrame:
        """`DataFrame` with loop metadata"""
        return self._table(LOOPS)

    @property
    def group_loops(self) -> pd.DataFrame:
        """`DataFrame` with both group and loop data"""
        return self._table(GROUP_LOOPS)

    @property
    def grabs(self) -> pd.DataFrame:
        """
        `DataFrame` with grab metadata

        The `sync_*` columns come from the processed sync file and are empty
        until `process_grabs_sync` has run for its recording.
        """
        return self._table(GRABS)

    @property
    def zstacks(self) -> pd.DataFrame:
        """`DataFrame` with z-stack metadata"""
        return self._table(ZSTACKS)

    @property
    def acquisitions(self) -> pd.DataFrame:
        """
        `DataFrame` with acquisition metadata

        The `sync_*` columns come from the processed sync file and are empty
        until `process_sync` has run for the loop.
        """
        return self._table(ACQUISITIONS)

    @property
    def acquisition_trials(self) -> pd.DataFrame:
        """
        `DataFrame` with each acquisition and the associated trial data.

        `sync_*` columns come from the DAQ sync file and `trial_*` ones from
        the olfactometer event files. Acquisitions that are not associated
        with a trial are kept, just with null entries.

        To compute `events` timedeltas use the trial (olfactometer) timings.
        """
        return self._table(ACQUISITION_TRIALS)

    @property
    def streams(self) -> pd.DataFrame:
        """`DataFrame` with the whole-file recordings (sync H5, videos)"""
        return self._table(STREAMS)

    @property
    def sync_blocks(self) -> pd.DataFrame:
        """`DataFrame` with the frame bursts of every processed sync file"""
        return self._table(SYNC_BLOCKS)

    @property
    def mcor_files(self) -> pd.DataFrame:
        """`DataFrame` with mcor files metadata"""
        return self._table(MCOR_FILES)

    @property
    def programs(self) -> pd.DataFrame:
        """`DataFrame` with one entry per olfactometer run"""
        return self._table(PROGRAMS)

    @property
    def trials(self) -> pd.DataFrame:
        """`DataFrame` with all olfactometer trials"""
        return self._table(TRIALS)

    @property
    def events(self) -> pd.DataFrame:
        """`DataFrame` with olfactometer events"""
        return self._table(EVENTS)

    @property
    def odors(self) -> pd.DataFrame:
        """`DataFrame` with the odor registry, ids as in the olfactometer"""
        return self._table(ODORS)

    @property
    def odor_components(self) -> pd.DataFrame:
        """`DataFrame` with the single chemicals each mix is made of"""
        return self._table(ODOR_COMPONENTS)

    @property
    def panels(self) -> pd.DataFrame:
        """`DataFrame` with the named sets of vials"""
        return self._table(PANELS)

    @property
    def panel_positions(self) -> pd.DataFrame:
        """`DataFrame` with the odor each position of each panel holds"""
        return self._table(PANEL_POSITIONS)

    @property
    def vials(self) -> pd.DataFrame:
        """
        `DataFrame` with one row per physical vial as made up, with the
        `odor_id` of its panel position
        """
        return self._table(VIALS)

    @property
    def vial_odors(self) -> pd.DataFrame:
        """`DataFrame` with what was pipetted into each vial"""
        return self._table(VIAL_ODORS)

    @property
    def loop_vials(self) -> pd.DataFrame:
        """`DataFrame` with which vials each loop ran"""
        return self._table(LOOP_VIALS)

    @property
    def annotation_keys(self) -> pd.DataFrame:
        """`DataFrame` with what can be annotated, indexed by `(applies_to, key)`"""
        return self._table(ANNOTATION_KEYS)

    @property
    def annotations(self) -> pd.DataFrame:
        """
        `DataFrame` with every annotation ever written, oldest first.

        Use `annotations_for(...)` for the current value of each key.
        """
        return self._table(ANNOTATIONS)

    @property
    def method_calls(self) -> pd.DataFrame:
        """`DataFrame` with `@record_call` functions"""
        return self._table(METHOD_CALLS)

    @property
    def outputs(self) -> pd.DataFrame:
        """`DataFrame` with output files of functions"""
        return self._table(OUTPUTS)

    @property
    def groups(self) -> dict[int, Group]:
        """`Group`s (indexed by `group_id`) for processing/analysis."""
        with self._locked() as con:
            self._refresh_if_stale()

            query = "SELECT group_id FROM groups WHERE group_id != ?;"
            rows = con.execute(query, [self.group_id]).fetchall()

            return {row["group_id"]: self._group(row["group_id"]) for row in rows}

    def _group(self, group_id: int) -> Group:
        """Return the cached `Group` for `group_id`, creating it if missing."""
        if group_id not in self._groups:
            self._groups[group_id] = Group(group_id, self)

        return self._groups[group_id]

    def annotations_for(self, target_type: str) -> pd.DataFrame:
        """
        One row per annotated `session`/`loop`/`grab`/... , one column per key.

        **EXAMPLE**
        ```python
        superficial = db.annotations_for("loop").query("depth_goal == 'superficial'")
        db.loops.join(superficial, how="inner")
        ```

        Every key holds one value, so each column is a plain number, string or
        date and the table behaves like any other. Annotations are append-only,
        so what appears here is the latest value written for each key (a key
        cleared with `None` reads as missing); the earlier ones are in
        `db.annotations`.
        """
        keys = self.annotation_keys

        if target_type not in keys.index.get_level_values("applies_to"):
            raise ValueError(
                f"Nothing can be annotated on a {target_type!r}. Use one of: "
                f"{sorted(set(keys.index.get_level_values('applies_to')))}."
            )

        wanted = keys.xs(target_type, level="applies_to")

        rows = self.annotations
        rows = rows[
            (rows["target_type"] == target_type) & rows["key"].isin(wanted.index)
        ]

        # Append-only, so the last row written for a key is its current value.
        # `annotation_id` rises with time and is the index, hence sorting on it.
        current = rows.sort_index().drop_duplicates(
            subset=["target_id", "key"], keep="last"
        )

        frame = current.pivot(index="target_id", columns="key", values="value")
        frame.columns.name = None
        frame.index.name = ANNOTATION_TARGETS[target_type].index

        # `value` is an ANY column, so a column arrives as object dtype even when
        # every entry in it is a number. The registry says which is which.
        for key, declared in wanted["value_type"].items():
            if key not in frame.columns:
                continue

            if declared in ("integer", "real"):
                frame[key] = pd.to_numeric(frame[key], errors="coerce")

            elif declared == "boolean":
                # Kept as `object` of Python bools and None: a nullable boolean
                # dtype would turn the missing ones into `pd.NA`, which is
                # neither True nor False and trips `if value:` checks.
                frame[key] = (
                    frame[key]
                    .map(lambda x: None if pd.isna(x) else bool(x))
                    .astype(object)
                )

            elif declared == "date":
                frame[key] = pd.to_datetime(frame[key], errors="coerce")

        return frame

    def missing_annotations(self) -> pd.DataFrame:
        """
        Everything still to be filled in before this project's data is finished.

        One row per `(target_type, target_id, key)` that is marked `required` in
        `db.annotation_keys` and holds no value yet (never written, or cleared).
        """
        keys = self.annotation_keys
        required = keys[keys["required"].astype(bool) & ~keys["retired"].astype(bool)]

        rows = []

        with self._locked() as con:
            for (applies_to, key), entry in required.iterrows():
                spec = ANNOTATION_TARGETS[applies_to]

                # Group 0 is the database's own row, not something to annotate
                skip = "WHERE t.group_id != 0" if applies_to == "group" else ""

                found = con.execute(
                    f"""
                    SELECT t.{spec.index} AS target_id FROM {spec.name} AS t
                        {skip}
                        GROUP BY t.{spec.index}
                        HAVING (
                            SELECT a.value FROM annotations AS a
                                WHERE a.target_type = ? AND a.key = ?
                                  AND a.target_id = t.{spec.index}
                                ORDER BY a.annotation_id DESC LIMIT 1
                        ) IS NULL;
                    """,
                    [applies_to, key],
                ).fetchall()

                rows.extend(
                    {
                        "target_type": applies_to,
                        "target_id": row["target_id"],
                        "key": key,
                        "label": entry["label"],
                    }
                    for row in found
                )

        return pd.DataFrame(rows, columns=["target_type", "target_id", "key", "label"])

    def session_for(self, mouse_id: int | str, session_date: Value) -> None | int:
        """
        The `session_id` of a mouse on a day, or `None` if there is none.

        **EXAMPLE**
        ```python
        db.session_for("m442", "2026-07-08")
        ```
        """
        with self._locked() as con:
            row = con.execute(
                "SELECT session_id FROM sessions WHERE mouse_id = ? AND session_date = ?;",
                [_mouse_number(mouse_id), _date_string(session_date)],
            ).fetchone()

        return None if row is None else int(row["session_id"])

    def loop_for(self, session_id: int, loop_number: int) -> None | int:
        """
        The `loop_id` of a session's loop by number (`eN`), or `None`.

        **EXAMPLE**
        ```python
        db.loop_for(db.session_for("m442", "2026-07-08"), 2)
        ```
        """
        with self._locked() as con:
            row = con.execute(
                "SELECT loop_id FROM loops WHERE session_id = ? AND loop_number = ?;",
                [int(session_id), int(loop_number)],
            ).fetchone()

        return None if row is None else int(row["loop_id"])

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _check_schema_version(self) -> None:
        """
        Throws error if DB schema version is not what the code expects.
        """

        # Called with the lock held.
        version = self._con.execute("PRAGMA user_version;").fetchone()[0]
        if version == SCHEMA_VERSION:
            return

        if version < SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema is v{version} but the code expects "
                f"v{SCHEMA_VERSION}. Run the migration:\n"
                f"    python -m odyn.admin '{self.main_folder}'"
                + (f" --project {self.project}" if self.project else "")
                + " migrate"
            )

        raise RuntimeError(
            f"Database schema is v{version} but the code expects v{SCHEMA_VERSION}. "
            "Your code is out of date! Pull the latest version and restart Python."
        )

    def _read_tiff(self, path: Path) -> None | tuple[Object, Object]:
        return read_tiff(path, self.main_folder)

    def _refresh_if_stale(self) -> None:
        """Reset caches if another connection has committed since the last check."""
        with self._locked() as con:
            version = con.execute("PRAGMA data_version;").fetchone()[0]

        if version != self._data_version:
            self._data_version = version
            self._reset_caches()

    def _reset_caches(self) -> None:
        self._tables.clear()

        # In case a specific group can still be accessed
        for group in self._groups.values():
            group._tables.clear()

        self._groups.clear()

    # ----------------------------------------------------------------------- #
    # Database Queries
    # ----------------------------------------------------------------------- #

    def add_group(self, loop_ids: list[int]) -> Group:
        """
        Add a group with the loops listed and return it.

        **EXAMPLE**
        ```python
        group = db.add_group([3, 10, 12])
        ```

        Every loop already has a group of its own; this makes one that
        processes several together (a field of view imaged in several loops).
        An existing group with exactly these loops is returned instead.
        """
        missing = set(int(i) for i in loop_ids) - set(self.loops.index)

        if missing:
            raise ValueError(f"No loops with ids: {sorted(missing)}")

        target = frozenset(int(i) for i in loop_ids)

        # Check if there is a group with exactly those loops
        members: dict[int, set[int]] = {}

        for group_id, loop_id in self.group_loops["loop_id"].items():
            members.setdefault(group_id, set()).add(int(loop_id))

        for group_id, member_ids in members.items():
            if member_ids == set(target):
                logger.info(
                    f"Group already exists (group_id = {group_id}), returning it."
                )
                return self._group(group_id)

        with self._writing() as cur:
            cur.execute("INSERT INTO groups DEFAULT VALUES;")
            group_id = cur.lastrowid

            cur.executemany(
                "INSERT INTO group_loops (group_id, loop_id) VALUES (?, ?);",
                [(group_id, loop_id) for loop_id in target],
            )

        assert group_id is not None

        return self._group(group_id)

    def from_query(self, query: str, params: list = []) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from a SQL query (reading only).

        **USAGE**
        ```python
        db = Database(main_folder)
        df = db.from_query(query_as_a_string)
        ```

        **EXAMPLE**
        ```python
        db.from_query("SELECT loop_id, loop_path FROM loops;")
        ```
        """
        with self._locked() as con:
            frame = pd.read_sql_query(query, con, params=params)

            if con.in_transaction:
                con.rollback()
                raise ValueError(
                    "'from_query' only reads, so this change was not saved."
                )

        return frame

    def latest_calls(self, method_name: str) -> pd.DataFrame:
        """
        `DataFrame` with all calls to `method_name`, newest first.

        The inputs of each call are the `input_*` columns and its recorded
        output the `output_*` ones.
        """

        query = """
            SELECT * FROM method_calls
                WHERE method_name LIKE ?
                ORDER BY method_call_id DESC
            """
        with self._locked() as con:
            return _method_calls_dataframe(con, query, [f"%{method_name}"])

    def latest_output(self, method_name: str) -> None | Object:
        """Return output of the most recent call to 'method_name'."""

        with self._locked() as con:
            row = con.execute(
                """
                SELECT method_call_id, call_output FROM method_calls
                    WHERE group_id = ? AND method_name = ? AND call_output IS NOT NULL
                    ORDER BY method_call_id DESC LIMIT 1
                """,
                [self.group_id, method_name],
            ).fetchone()

        if row is None:
            return None

        # Links this value to the call reading it
        self.note_consumed(row["method_call_id"])

        return json.loads(row["call_output"])

    # ----------------------------------------------------------------------- #
    # People and subjects
    # ----------------------------------------------------------------------- #

    @record_call
    def add_user(self, *, user_name: str, display_name: None | str = None) -> int:
        """
        Add a name to the user picker.

        **USAGE**
        ```python
        db.add_user(user_name="vinicius", display_name="Vinicius")
        ```

        `user_name` is what `method_calls.user` records; `display_name` is what
        the GUI shows, and defaults to the same. Returns the `user_id`.
        """
        name = str(user_name).strip()
        shown = name if display_name is None else str(display_name).strip()

        if not name or not shown:
            raise ValueError("A user needs a name.")

        with self._writing() as cur:
            user_id = _db_insert(
                cur, "users", {"user_name": name, "display_name": shown}
            )

        logger.info(f"Added user '{name}'. {CHECK}")

        return user_id

    @record_call
    def retire_user(self, *, user_name: str, retired: bool = True) -> None:
        """
        Take a name out of the picker (`retired=False` puts it back).

        What the person recorded keeps their name; only the picker changes.
        """
        with self._writing() as cur:
            changed = cur.execute(
                "UPDATE users SET retired = ? WHERE user_name = ?;",
                [bool(retired), str(user_name).strip()],
            ).rowcount

        if not changed:
            raise ValueError(f"No user named {user_name!r}.")

        was = "Retired" if retired else "Brought back"
        logger.info(f"{was} user '{user_name}'. {CHECK}")

    @record_call
    def add_line(self, *, line_name: str, jax_strain_number: None | str = None) -> str:
        """
        Add a mouse line to the ones a mouse can be recorded as carrying.

        **USAGE**
        ```python
        db.add_line(line_name="TH-Cre", jax_strain_number="008601")
        ```

        `jax_strain_number` is kept as text (JAX numbers have leading zeros).
        Returns the name as stored.
        """
        name = str(line_name).strip()
        number = None if jax_strain_number is None else str(jax_strain_number).strip()

        if not name:
            raise ValueError("A line needs a name.")

        with self._writing() as cur:
            _db_insert(cur, "lines", {"line_name": name, "jax_strain_number": number})

        logger.info(f"Added line '{name}'. {CHECK}")

        return name

    @record_call
    def add_virus(self, *, virus_name: str) -> str:
        """
        Add a virus to the list of viruses a mouse can be injected with.

        **USAGE**
        ```python
        db.add_virus(virus_name="pAAV9-CAG-dLight1.3b")
        ```

        Returns the name as stored.
        """
        name = str(virus_name).strip()

        if not name:
            raise ValueError("A virus needs a name.")

        with self._writing() as cur:
            _db_insert(cur, "viruses", {"virus_name": name})

        logger.info(f"Added virus '{name}'. {CHECK}")

        return name

    @record_call
    def set_mouse(
        self,
        *,
        mouse_id: int | str,
        sex: None | str = None,
        dob: None | Value = None,
        lines: None | dict[str, str] = None,
        viruses: None | list[str] = None,
    ) -> int:
        """
        Record what is known about an animal, creating its row if it is new.

        **USAGE**
        ```python
        db.set_mouse(
            mouse_id="m442",
            sex="F",
            dob="2026-01-22",
            lines={"DAT-Cre": "het", "TIGRE2": "het"},
            viruses=["pAAV9-CAG-dLight1.3b"],
        )
        ```

        **PARAMETERS**
        - `mouse_id` is the SID or a name like `'m442'`
        - `sex` is `'M'` or `'F'`
        - `dob` is the date of birth, `YYYY-MM-DD`
        - `lines` maps each line the animal carries (a name from `db.lines`)
        to `'wt'`, `'het'`, `'hom'`, or `'na'` when it was not genotyped
        (always `'na'` for `'Wild Type'`)
        - `viruses` lists the viruses the animal was injected with (names
        from `db.viruses`)

        Returns the `mouse_id`. Anything left out keeps the value already
        stored, so this can be called repeatedly as details arrive. Passing
        `lines={}` (or `viruses=[]`) clears them; `None` leaves them alone.
        """
        number = _mouse_number(mouse_id)
        stored_sex = None if sex is None else str(sex).strip().upper()

        if stored_sex not in (None, "M", "F"):
            raise ValueError(f"Sex must be 'M' or 'F', not {sex!r}.")

        born = None if dob is None else _date_string(dob)
        carried = None if lines is None else _known_lines(self.lines, lines)
        injected = (
            None
            if viruses is None
            else [_registered(self.viruses, "virus", v) for v in viruses]
        )

        if injected is not None and len(set(injected)) < len(injected):
            raise ValueError("A virus is listed twice.")

        with self._writing() as cur:
            # COALESCE so that a field left out of this call does not erase what
            # an earlier one wrote: the details arrive from different people.
            cur.execute(
                """
                INSERT INTO mice (mouse_id, mouse_sex, mouse_dob)
                    VALUES (:mouse_id, :sex, :dob)
                ON CONFLICT(mouse_id) DO UPDATE SET
                      mouse_sex = COALESCE(:sex, mouse_sex)
                    , mouse_dob = COALESCE(:dob, mouse_dob);
                """,
                {"mouse_id": number, "sex": stored_sex, "dob": born},
            )

            if carried is not None:
                # Replaced as a whole: a mouse carries one set of mutations, and
                # a correction usually rewrites more than one of them.
                cur.execute("DELETE FROM mouse_lines WHERE mouse_id = ?;", [number])

                for line, genotype in carried.items():
                    _db_insert(
                        cur,
                        "mouse_lines",
                        {"mouse_id": number, "line_name": line, "genotype": genotype},
                    )

            if injected is not None:
                # Replaced as a whole, like the lines
                cur.execute("DELETE FROM mouse_viruses WHERE mouse_id = ?;", [number])

                for virus in injected:
                    _db_insert(
                        cur, "mouse_viruses", {"mouse_id": number, "virus_name": virus}
                    )

        cross = "unknown line" if not carried else " x ".join(carried)

        logger.info(f"Mouse {number} recorded ({cross}). {CHECK}")

        return number

    @record_call
    def add_session(self, *, mouse_id: int | str, session_date: Value) -> int:
        """
        Start a session: one mouse, one day. Returns its `session_id`.

        **USAGE**
        ```python
        session_id = db.add_session(mouse_id="m442", session_date="2026-07-08")
        ```

        The mouse must be registered (`set_mouse`) first. A session that already
        exists is an error; look it up with `session_for` instead.
        """
        number = _mouse_number(mouse_id)
        day = _date_string(session_date)

        try:
            with self._writing() as cur:
                session_id = _db_insert(
                    cur, "sessions", {"mouse_id": number, "session_date": day}
                )

        except sqlite3.IntegrityError as error:
            if "FOREIGN KEY" in str(error):
                raise ValueError(
                    f"Mouse {number} is not registered. Add it with `set_mouse` first."
                ) from error

            raise ValueError(
                f"Mouse {number} already has a session on {day} (use "
                f"`session_for` to find it)."
            ) from error

        logger.info(f"Session {session_id}: mouse {number} on {day}. {CHECK}")

        return session_id

    # ----------------------------------------------------------------------- #
    # Odors and panels
    # ----------------------------------------------------------------------- #

    @record_call
    def add_odor(
        self,
        *,
        odor_id: int,
        odor_name: str,
        components: None | list[int] = None,
        vapor_pressure_mmhg: None | float = None,
        vapor_pressure_source: None | str = None,
    ) -> None:
        """
        Register an odor, with the number the olfactometer knows it by.

        **USAGE**
        ```python
        db.add_odor(
            odor_id=45,
            odor_name="2-hexanone",
            vapor_pressure_mmhg=11.6,
            vapor_pressure_source="PubChem",
        )
        db.add_odor(odor_id=46, odor_name="sigma", components=[45, 4])
        ```

        `components` makes it a mix: the numbers of the single chemicals it is
        made of, two or more, fixed from here on (a vial of the mix only picks
        each one's concentration). Left out, the odor is a single chemical,
        whose vapor pressure (mmHg at 25 C) is what the vial recipe needs.
        """
        component_ids = [int(c) for c in components or []]

        if len(component_ids) == 1:
            raise ValueError(
                "A mix has at least two components (a monomolecular odor has none)."
            )

        if len(set(component_ids)) != len(component_ids):
            raise ValueError("A component is listed twice.")

        with self._writing() as cur:
            mixes = {
                row["odor_id"]: bool(row["mix"])
                for row in cur.execute("SELECT odor_id, mix FROM odors;")
            }

            for component_id in component_ids:
                if component_id not in mixes:
                    raise ValueError(
                        f"No odor with id {component_id} to add to the mix."
                    )

                if mixes[component_id]:
                    raise ValueError(
                        f"Component {component_id} is itself a mix (list its "
                        "components instead)."
                    )

            _db_insert(
                cur,
                "odors",
                {
                    "odor_id": int(odor_id),
                    "odor_name": str(odor_name).strip(),
                    "mix": bool(component_ids),
                    "vapor_pressure_mmhg": _optional_real(vapor_pressure_mmhg),
                    "vapor_pressure_source": vapor_pressure_source,
                },
            )

            for component_id in component_ids:
                _db_insert(
                    cur,
                    "odor_components",
                    {"odor_id": int(odor_id), "component_id": component_id},
                )

        logger.info(f"Added odor {odor_id} '{odor_name}'. {CHECK}")

    @record_call
    def update_odor(
        self,
        *,
        odor_id: int,
        vapor_pressure_mmhg: None | float = None,
        vapor_pressure_source: None | str = None,
        retired: None | bool = None,
    ) -> None:
        """
        Correct what is recorded about an odor. Only what is passed changes.

        The number, the name and what a mix is made of never change. The
        number is the olfactometer's, and the trials, vials and loop
        ingestion go by the number and name together. A wrong one is a new
        odor. `retired=True` takes the odor out of the dropdown menus.
        """
        changes = {
            "vapor_pressure_mmhg": _optional_real(vapor_pressure_mmhg),
            "vapor_pressure_source": vapor_pressure_source,
            "retired": None if retired is None else bool(retired),
        }
        changes = {
            column: value for column, value in changes.items() if value is not None
        }

        if not changes:
            raise ValueError("Nothing to change.")

        with self._writing() as cur:
            changed = cur.execute(
                f"UPDATE odors SET {', '.join(f'{c} = :{c}' for c in changes)} "
                f"WHERE odor_id = :odor_id;",
                {**changes, "odor_id": int(odor_id)},
            ).rowcount

        if not changed:
            raise ValueError(f"No odor with id {odor_id}.")

        logger.info(f"Updated odor {odor_id}: {changes}. {CHECK}")

    @record_call
    def add_panel(
        self,
        *,
        panel_name: str,
        positions: dict[int, int],
        description: None | str = None,
    ) -> int:
        """
        Add a panel, which is determined by its name and by which odors are in
        which positions in the olfactometer. Returns the `panel_id`.

        **USAGE**
        ```python
        db.add_panel(
            panel_name="16 Odors",
            positions={1: 1, 2: 2, 3: 0, 4: 17},  # position: odor_id
            description="3 mixes at 60:40 and 40:60 plus their components",
        )
        ```

        The relationship between positions and odors can never change (add a
        new panel for that), but vials mixes can be remade (see `add_vials`).
        """
        if not positions:
            raise ValueError("A panel needs at least one position.")

        with self._writing() as cur:
            known = {
                row["odor_id"] for row in cur.execute("SELECT odor_id FROM odors;")
            }
            unknown = sorted(set(int(o) for o in positions.values()) - known)

            if unknown:
                raise ValueError(f"No odors with ids {unknown}.")

            taken = cur.execute(
                "SELECT panel_name FROM panels WHERE panel_name = ? COLLATE NOCASE;",
                [str(panel_name).strip()],
            ).fetchone()

            if taken is not None:
                raise ValueError(
                    f"There is already a panel named {taken['panel_name']!r} "
                    "(names are case-insensitive)."
                )

            panel_id = _db_insert(
                cur,
                "panels",
                {"panel_name": str(panel_name).strip(), "description": description},
            )

            for position, odor_id in sorted(
                (int(p), int(o)) for p, o in positions.items()
            ):
                if position < 1:
                    raise ValueError(f"Positions start at 1, not {position}.")

                _db_insert(
                    cur,
                    "panel_positions",
                    {
                        "panel_id": panel_id,
                        "vial_position": position,
                        "odor_id": odor_id,
                    },
                )

        logger.info(f"Added panel '{panel_name}' ({len(positions)} vials). {CHECK}")

        return panel_id

    @record_call
    def add_vials(
        self,
        *,
        panel_name: str,
        vials: list[Object],
        made_on: None | Value = None,
    ) -> list[int]:
        """
        Record a new batch of vials for a panel. Returns their `vial_id`s.

        **USAGE**
        ```python
        db.add_vials(
            panel_name="16 Odors",
            made_on="2026-07-06",
            vials=[
                {
                    "vial_position": 4,           # alpha, a mix
                    "vial_sccm": 200,
                    "total_sccm": 1500,
                    "total_volume_ml": 5,
                    "solvent_volume_ml": 4.9,
                    "components": [
                        {
                            "odor_id": 4,
                            "target_ppm": 44.2,
                            "liquid_ul": 60,
                            "percent_vv": 1.2,
                        },
                        {
                            "odor_id": 10,
                            "target_ppm": 13.3,
                            "liquid_ul": 40,
                            "percent_vv": 0.8,
                        },
                    ],
                },
                {"vial_position": 3, "vial_sccm": 200,
                 "total_sccm": 1500, "total_volume_ml": 5},   # mineral oil
            ],
        )
        ```

        **PARAMETERS**
        - `panel_name` names a panel added with `add_panel`
        - `vials` is one dict per vial: `vial_position` (as in the panel's)
        `vial_sccm`, `total_sccm`, `total_volume_ml`, optionally
        `solvent_volume_ml`, `made_on`, and `components`. An `odor_id` may be
        given too, and must be the one associated with the position
        - `components` is one dict per chemical pipetted in, with `odor_id`,
        `target_ppm` and the computed `liquid_ul` and `percent_vv`. A mix
        lists exactly its `odor_components`, a pure odor lists itself once,
        mineral oil lists nothing
        - `made_on` is the date for every vial without one of its own; it
        defaults to today, since vials are usually recorded as they are made

        Remaking vials is a new call, for some of the panel's positions or all
        of them. A vial a loop has used is never edited, so what was in
        the olfactometer stays on record.
        """
        day = _date_string(datetime.now() if made_on is None else made_on)
        taken = [int(v["vial_position"]) for v in vials]  # type: ignore[arg-type]

        if len(set(taken)) < len(taken):
            raise ValueError("Two vials in this batch are in the same position.")

        with self._writing() as cur:
            panel_id = _panel_id(cur, panel_name)
            holds = {
                row["vial_position"]: row["odor_id"]
                for row in cur.execute(
                    "SELECT vial_position, odor_id FROM panel_positions "
                    "WHERE panel_id = ?;",
                    [panel_id],
                )
            }
            mixes = {
                row["odor_id"]: bool(row["mix"])
                for row in cur.execute("SELECT odor_id, mix FROM odors;")
            }
            made_of: dict[int, set[int]] = {}

            for row in cur.execute(
                "SELECT odor_id, component_id FROM odor_components;"
            ):
                made_of.setdefault(row["odor_id"], set()).add(row["component_id"])

            vial_ids = []

            for vial in vials:
                vial_ids.append(
                    _insert_vial(cur, panel_id, holds, vial, day, mixes, made_of)
                )

        logger.info(f"Added {len(vial_ids)} vials to panel '{panel_name}'. {CHECK}")

        return vial_ids

    @record_call
    def set_loop_vials(self, *, loop_id: int, vial_ids: list[int]) -> None:
        """
        Record which vials were in the olfactometer for a loop.

        **USAGE**
        ```python
        db.set_loop_vials(loop_id=12, vial_ids=[1, 2, 3, 4])
        ```

        Replaces the whole set: a rack is only true as a whole. The vials must
        all come from one panel and sit in different positions.
        """
        ids = sorted({int(v) for v in vial_ids})

        with self._writing() as cur:
            placeholders = ", ".join("?" * len(ids))
            rows = cur.execute(
                f"SELECT vial_id, panel_id, vial_position FROM vials "
                f"WHERE vial_id IN ({placeholders});",
                ids,
            ).fetchall()

            missing = set(ids) - {row["vial_id"] for row in rows}
            if missing:
                raise ValueError(f"No vials with ids {sorted(missing)}.")

            if len({row["panel_id"] for row in rows}) > 1:
                raise ValueError("A loop runs vials from one panel only.")

            positions = [row["vial_position"] for row in rows]
            if len(positions) != len(set(positions)):
                raise ValueError("Two of these vials sit in the same position.")

            cur.execute("DELETE FROM loop_vials WHERE loop_id = ?;", [int(loop_id)])
            cur.executemany(
                "INSERT INTO loop_vials (loop_id, vial_id) VALUES (?, ?);",
                [(int(loop_id), vial_id) for vial_id in ids],
            )

        logger.info(f"Loop {loop_id} ran vials {ids}. {CHECK}")

    # ----------------------------------------------------------------------- #
    # Annotations
    # ----------------------------------------------------------------------- #

    @record_call
    def add_annotations(
        self, *, target_type: str, target_id: int, values: dict[str, Value]
    ) -> None:
        """
        Record things about a loop, session, mouse, program, etc.

        **PARAMETERS**
        - `target_type` is what this is about: `'mouse'`, `'session'`,
        `'loop'`, `'grab'`, `'zstack'`, `'program'`, `'acquisition'` or
        `'group'`.
        - `target_id` is that row's id (on the target_type table)
        - `values` maps keys from `db.annotation_keys` to their values, which
        have to match the type each key was registered with. `None` clears a
        key.

        **EXAMPLE**
        ```python
        db.add_annotations(
            target_type="loop",
            target_id=12,
            values={
                "z_landmark": "surface of right bulb",
                "fov_description": "medial-rostral left bulb",
            },
        )
        ```

        **ALERT**
        odyn's processing does not use annotations. Those are for analysis
        only. Anything the pipeline has to use must be a column instead.

        Annotations are never overwritten. Writing a key twice keeps both, and
        the later one is used (so the whole history stays visible).
        """
        if not values:
            raise ValueError("Nothing to annotate.")

        with self._writing() as cur:
            for key, value in values.items():
                _db_annotate(
                    cur,
                    target_type=target_type,
                    target_id=target_id,
                    key=key,
                    value=value,
                    method_call_id=self.current_call_id,
                )

        logger.info(
            f"Annotated {target_type} {target_id}: {', '.join(values)}. {CHECK}"
        )

    @record_call
    def add_annotation_key(
        self,
        *,
        applies_to: str,
        key: str,
        label: str,
        value_type: str,
        description: str,
        unit: None | str = None,
        allowed_values: None | list = None,
        source: str = "user",
        required: bool = False,
    ) -> None:
        """
        Register something new that can be annotated in this project.

        **PARAMETERS**
        - `applies_to` is `'mouse'`, `'session'`, `'loop'`, `'grab'`,
        `'zstack'`, `'program'`, `'acquisition'` or `'group'`
        - `key` is what `add_annotations` will be called with
        - `value_type` is `'text'`, `'long_text'`, `'integer'`, `'real'`,
        `'boolean'`, `'date'`, or `'enum'`; an `'enum'` needs `allowed_values`.
        `'long_text'` is stored like `'text'` and says a form should give it a
        box rather than a line
        - `source` says how the value arrives: `'user'` (typed in),
        `'metadata'` (read from a file) or `'computed'` (from other fields)
        - `required` means the data is not finished until this is filled in

        **EXAMPLE**
        ```python
        db.add_annotation_key(
            applies_to="loop",
            key="uses_odor_batch",
            label="Odor batch used",
            value_type="text",
            description="Which batch of odor was used.",
        )
        ```

        Keys are never deleted, because old values would stop making sense. If
        you want to stop using a particular key, use `retire_annotation_key`.
        """
        with self._writing() as cur:
            _db_insert(
                cur,
                "annotation_keys",
                {
                    "applies_to": applies_to,
                    "key": key,
                    "label": label,
                    "value_type": value_type,
                    "allowed_values": (
                        None if allowed_values is None else json.dumps(allowed_values)
                    ),
                    "unit": unit,
                    "description": description,
                    "source": source,
                    "required": bool(required),
                    "retired": False,
                },
            )

        logger.info(f"Registered annotation '{key}' for a {applies_to}. {CHECK}")

    @record_call
    def update_annotation_key(
        self,
        *,
        applies_to: str,
        key: str,
        label: None | str = None,
        description: None | str = None,
        unit: None | str = None,
        add_values: None | list = None,
        source: None | str = None,
        required: None | bool = None,
    ) -> None:
        """
        Change how an annotation is presented, without touching its values.

        **USAGE**
        ```python
        db.update_annotation_key(
            applies_to="loop",
            key="depth_class",
            add_values=["intermediate"],
        )
        ```

        Only what is passed changes. `add_values` appends options to an enum;
        options are never removed, since values may already use them. The type
        is not changeable either, for the same reason: register a new key.
        """
        with self._writing() as cur:
            row = cur.execute(
                "SELECT value_type, allowed_values FROM annotation_keys "
                "WHERE applies_to = ? AND key = ?;",
                [applies_to, key],
            ).fetchone()

            if row is None:
                raise ValueError(f"No annotation key '{key}' for a {applies_to}.")

            changes: Object = {
                "label": label,
                "description": description,
                "unit": unit,
                "source": source,
                "required": None if required is None else bool(required),
            }
            changes = {c: v for c, v in changes.items() if v is not None}

            if add_values:
                if row["value_type"] != "enum":
                    raise ValueError(f"'{key}' is not an enum.")

                options = json.loads(row["allowed_values"])
                options += [str(v) for v in add_values if str(v) not in options]
                changes["allowed_values"] = json.dumps(options)

            if not changes:
                raise ValueError("Nothing to change.")

            cur.execute(
                f"UPDATE annotation_keys SET {', '.join(f'{c} = :{c}' for c in changes)} "
                "WHERE applies_to = :applies_to AND key = :key;",
                {**changes, "applies_to": applies_to, "key": key},
            )

        logger.info(f"Updated annotation '{key}' for a {applies_to}. {CHECK}")

    @record_call
    def retire_annotation_key(
        self, *, applies_to: str, key: str, retired: bool = True
    ) -> None:
        """
        Stop offering an annotation, without losing what was already written.

        **USAGE**
        ```python
        db.retire_annotation_key(applies_to="loop", key="cohort")
        db.retire_annotation_key(
            applies_to="loop",
            key="cohort",
            retired=False,
        )
        ```

        **PARAMETERS**
        - `applies_to` and `key` name the annotation, as in `db.annotation_keys`
        - `retired=False` brings it back, for when one is retired by mistake

        A retired key is refused by `add_annotations` and dropped from
        `db.missing_annotations()`, but everything written under it stays in
        `db.annotations` and keeps showing up in `db.annotations_for(...)`.
        """
        with self._writing() as cur:
            changed = cur.execute(
                """
                UPDATE annotation_keys
                    SET retired = ?
                    WHERE applies_to = ? AND key = ?;
                """,
                [bool(retired), applies_to, key],
            ).rowcount

        if not changed:
            raise ValueError(
                f"No annotation key '{key}' for a {applies_to}. Check "
                f"'db.annotation_keys' for the ones this project has."
            )

        was = "Retired" if retired else "Brought back"
        logger.info(f"{was} annotation '{key}' for a {applies_to}. {CHECK}")

    # ----------------------------------------------------------------------- #
    # Recordings
    # ----------------------------------------------------------------------- #

    @record_call
    def add_loop(
        self,
        *,
        session_id: int,
        loop_path: str,
        objective: int,
        loop_number: None | int = None,
        rel_raw_paths: None | list[str] = None,
    ) -> None | int:
        """
        Add a loop folder to a session. Returns its `loop_id`.

        **PARAMETERS**
        - `session_id` is the session (mouse and day) the loop belongs to
        - `loop_path` is the loop folder, relative to the `main_folder`
        - `objective` is the objective it was imaged through (10, 20, ...)
        - `loop_number` is the loop's number in the session. Left out, it is
        the N of a folder named `eN`
        - `rel_raw_paths` is the raw files list (if None it searches the raw folder)

        **EXAMPLE**
        ```python
        db.add_loop(session_id=3, loop_path="20260623/m462/e2", objective=20)
        ```

        Reads the TIFFs, the olfactometer event files and the program
        definitions in `<loop_path>/olfactometer/`, and inserts the loop, its
        acquisitions, programs, trials and events in one transaction. The
        sync file is processed separately (`process_sync`).

        Returns `None` when nothing was added: the loop was already in
        the database, or the files did not pass the checks (see the log).
        """
        logger.info("Adding a loop to the database...")
        scale = self._um_scale(objective)

        if loop_number is None:
            match = re.fullmatch(r"e(\d+)", Path(loop_path).name)

            if match is None:
                raise ValueError(
                    f"'{Path(loop_path).name}' is not named eN: give `loop_number`."
                )

            loop_number = int(match.group(1))

        folder, session, tiffs = self._recordings_folder(
            session_id, loop_path, rel_raw_paths
        )

        # ----------------------------------------------------------------- #
        # Phase 1: Load
        # ----------------------------------------------------------------- #

        loop: None | Object = None
        acquisitions: list[Object] = []
        checks_failed = 0

        for path in tqdm(tiffs, desc="Loading TIFF Metadata"):
            read = self._read_kind(path, "loop")

            if read is None:
                continue

            recording, acquisition = read

            if loop is None:
                loop = recording

            elif recording != loop:
                checks_failed += 1
                logger.warning(
                    f"'{path.relative_to(self.main_folder)}' metadata"
                    " is inconsistent with the previous acquisition."
                )

            acquisitions.append(acquisition)

        if checks_failed > 0:
            logger.error(
                f"TIFF metadata changed {checks_failed} or more times in the raw "
                f"folder. {CROSS}"
            )
            logger.info("Are there multiple loops in the same folder?")
            logger.error("The loop will not be added to the DB.")
            self.add_flag(IngestFlag.METADATA_CHANGED)
            return None

        if loop is None:
            logger.error(f"No loop TIFF with the expected metadata found. {CROSS}")
            return None

        if loop["stack"] is not None:
            self.fail(
                IngestFlag.STACK_LOOP,
                f"'{loop_path}' is a z-stack loop with {loop['stack']['slice_count']} "
                "slices (odyn cannot process it yet).",
            )

        # The session is given, not read off the files; but the file names say
        # which mouse and day ScanImage was told, and a disagreement means the
        # folder is not what the caller thinks it is.
        _check_session(loop, session, self)

        logger.info(f"Passed all TIFF metadata checks! {CHECK}")

        programs_data = self._load_programs(folder)

        # ----------------------------------------------------------------- #
        # Phase 2: Insert
        # ----------------------------------------------------------------- #

        rig_settings = {key: loop[key] for key in RIG_ANNOTATIONS}
        loop_start = _dt_string(loop["start"])

        with self._writing() as cur:
            exists = cur.execute(
                "SELECT EXISTS(SELECT 1 FROM loops WHERE loop_start = ?);",
                [loop_start],
            ).fetchone()[0]

            if exists:
                logger.info(f"'{loop_path}' already in DB.")
                self.add_flag(IngestFlag.ALREADY_IN_DB)
                return None

            loop_id = _db_insert(
                cur,
                "loops",
                {
                    **_contract(loop, scale),
                    "session_id": session_id,
                    "loop_number": int(loop_number),
                    "loop_path": folder.resolve()
                    .relative_to(self.main_folder)
                    .as_posix(),
                    "loop_start": loop_start,
                    "objective": int(objective),
                },
            )

            # Every loop starts as a group of its own
            cur.execute("INSERT INTO groups DEFAULT VALUES;")
            cur.execute(
                "INSERT INTO group_loops (group_id, loop_id) VALUES (?, ?);",
                [cur.lastrowid, loop_id],
            )

            for acquisition in acquisitions:
                _db_insert(
                    cur,
                    "acquisitions",
                    {
                        **acquisition,
                        "loop_id": loop_id,
                        "acq_start": _dt_string(acquisition["acq_start"]),
                    },
                )

            for key, value in rig_settings.items():
                _db_annotate(
                    cur,
                    target_type="loop",
                    target_id=loop_id,
                    key=key,
                    value=value,
                    method_call_id=self.current_call_id,
                )

            for program_data in programs_data:
                _insert_program(cur, loop_id, program_data)

        self.set_output({"loop_ids": [loop_id]})
        logger.info(f"Added '{loop_path}' (loop_id = {loop_id}). {CHECK}")

        return loop_id

    @record_call
    def add_grabs(
        self,
        *,
        session_id: int,
        folder: str,
        objective: int,
        rel_raw_paths: None | list[str] = None,
    ) -> list[int]:
        """
        Add a session's grabs, one per TIFF. Returns the `grab_id`s added.

        **PARAMETERS**
        - `session_id` is the session (mouse and day) the grabs belong to
        - `folder` is the grabs folder relative to the `main_folder`
        - `objective` is the objective they were imaged through
        - `rel_raw_paths` is the raw files list (if None it searches the raw folder)

        **EXAMPLE**
        ```python
        db.add_grabs(session_id=3, folder="20260623/m462/grabs", objective=20)
        ```

        Files already in the database are skipped, so running this again after
        more grabs were taken only adds the new ones. Their sync recordings are
        processed separately (`process_grabs_sync`).
        """
        return self._add_singles("grab", session_id, folder, objective, rel_raw_paths)

    @record_call
    def add_zstacks(
        self,
        *,
        session_id: int,
        folder: str,
        objective: int,
        rel_raw_paths: None | list[str] = None,
    ) -> list[int]:
        """
        Add a session's z-stacks, one per TIFF. Returns the `zstack_id`s added.

        **PARAMETERS**
        - `session_id` is the session (mouse and day) the z-stacks belong to
        - `folder` is the z-stacks folder relative to the `main_folder`
        - `objective` is the objective they were imaged with
        - `rel_raw_paths` is the raw files list (if None it searches the raw folder)

        **EXAMPLE**
        ```python
        db.add_zstacks(session_id=3, folder="20260623/m462/zstacks", objective=20)
        ```

        The stack (slices, step, z range) is read from each TIFF, and the laser
        power at its shallowest and deepest slices stored as annotations. Files
        already in the database are skipped. Which loop a z-stack was taken for
        is set with `set_zstack_loop`.
        """
        return self._add_singles("zstack", session_id, folder, objective, rel_raw_paths)

    @record_call
    def set_zstack_loop(self, *, zstack_id: int, loop_id: None | int) -> None:
        """
        Say which loop a z-stack was taken for (`None`: none).

        **EXAMPLE**
        ```python
        db.set_zstack_loop(zstack_id=4, loop_id=12)
        ```

        The loop must be from the z-stack's session.
        """
        with self._writing() as cur:
            row = cur.execute(
                "SELECT session_id FROM zstacks WHERE zstack_id = ?;", [int(zstack_id)]
            ).fetchone()

            if row is None:
                raise ValueError(f"There is no z-stack {zstack_id}.")

            if loop_id is not None:
                loop = cur.execute(
                    "SELECT session_id FROM loops WHERE loop_id = ?;", [int(loop_id)]
                ).fetchone()

                if loop is None:
                    raise ValueError(f"There is no loop {loop_id}.")

                if loop["session_id"] != row["session_id"]:
                    raise ValueError(
                        f"Loop {loop_id} is not from the same session as z-stack "
                        f"{zstack_id}."
                    )

            cur.execute(
                "UPDATE zstacks SET loop_id = ? WHERE zstack_id = ?;",
                [None if loop_id is None else int(loop_id), int(zstack_id)],
            )

        logger.info(f"Z-stack {zstack_id} is associated with loop {loop_id}. {CHECK}")

    def _recordings_folder(
        self, session_id: int, folder_path: str, rel_raw_paths: None | list[str]
    ) -> tuple[Path, sqlite3.Row, list[Path]]:
        """The folder, its session's row and the TIFFs to read."""
        folder = self.main_folder / folder_path
        assert folder.is_dir(), f"Folder not found: '{folder.resolve()}'"

        tiffs = (
            sorted(folder.glob("raw/[!.]?*.tif"))
            if rel_raw_paths is None
            else [self.main_folder / p for p in rel_raw_paths]
        )
        assert tiffs, "Did not find any raw/*.tif files."

        logger.info(f"Processing folder: '{folder.resolve()}'")

        with self._locked() as con:
            session = con.execute(
                "SELECT session_id, mouse_id, session_date FROM sessions "
                "WHERE session_id = ?;",
                [int(session_id)],
            ).fetchone()

        if session is None:
            raise ValueError(f"There is no session {session_id}.")

        return folder, session, tiffs

    def _read_kind(self, path: Path, kind: str) -> None | tuple[Object, Object]:
        """A TIFF's metadata, or `None` (flagged) when it is not a `kind`."""
        read = self._read_tiff(path)

        if read is None:
            logger.info(f"  Skipped file {path} (metadata format not supported).")
            self.add_flag(IngestFlag.UNSUPPORTED_METADATA)
            return None

        recording = read[0]
        state = "loop" if kind == "loop" else "grab"
        stacked = recording["stack"] is not None

        # A loop may still be a stack (refused by the caller, with the reason)
        if recording["acq_state"] != state or (
            kind != "loop" and stacked != (kind == "zstack")
        ):
            what = {"loop": "loop", "grab": "grab", "zstack": "z-stack"}[kind]
            logger.warning(f"  Skipped '{path.name}' (not a {what}).")
            self.add_flag(IngestFlag.WRONG_ACQ_STATE)
            return None

        return read

    def _um_scale(self, objective: int) -> float:
        """
        The `OBJECTIVE_SCALE` for `objective`, recorded as `um_scale` in the
        running call's parameters. Raises for an objective it does not know.
        """
        scale = OBJECTIVE_SCALE.get(int(objective))

        if scale is None:
            known = ", ".join(str(o) for o in OBJECTIVE_SCALE)
            raise ValueError(
                f"No micron scale for objective {objective} (known: {known})."
            )

        self.update_parameters_used({"um_scale": scale})

        return scale

    def _add_singles(
        self,
        kind: str,
        session_id: int,
        folder_path: str,
        objective: int,
        rel_raw_paths: None | list[str],
    ) -> list[int]:
        """
        Shared body of `add_grabs` and `add_zstacks`: one row per TIFF.

        One transaction per file, so a folder that fails on one file keeps the
        others.
        """
        what = "grab" if kind == "grab" else "z-stack"
        logger.info(f"Adding {what}s to the database...")
        scale = self._um_scale(objective)
        _, session, tiffs = self._recordings_folder(
            session_id, folder_path, rel_raw_paths
        )

        added: list[int] = []
        seen = 0

        for path in tqdm(tiffs, desc="Loading TIFF Metadata"):
            read = self._read_kind(path, kind)

            if read is None:
                continue

            recording, tiff = read
            seen += 1
            _check_session(recording, session, self)

            row = {
                **_contract(recording, scale),
                "session_id": int(session_id),
                "raw_path": tiff["raw_path"],
                f"{kind}_start": _dt_string(tiff["acq_start"]),
                "objective": int(objective),
            }

            if kind == "grab":
                notes = {
                    key: recording[key]
                    for key in ("laser_power_920", "laser_power_1040")
                }

            else:
                stack = recording["stack"]
                row.pop("z_um")
                row.update(
                    slice_count=stack["slice_count"],
                    z_step_um=stack["z_step_um"],
                    z_start_um=stack["z_start_um"],
                    z_end_um=stack["z_end_um"],
                )
                notes = {
                    key: stack[key]
                    for key in ("power_start_pct", "power_end_pct", "power_length_um")
                    if stack[key] is not None
                }

            with self._writing() as cur:
                exists = cur.execute(
                    f"SELECT EXISTS(SELECT 1 FROM {kind}s WHERE raw_path = ?);",
                    [tiff["raw_path"]],
                ).fetchone()[0]

                if exists:
                    logger.info(f"'{path.name}' already in DB.")
                    self.add_flag(IngestFlag.ALREADY_IN_DB)
                    continue

                row_id = _db_insert(cur, f"{kind}s", row)

                for key, value in notes.items():
                    _db_annotate(
                        cur,
                        target_type=kind,
                        target_id=row_id,
                        key=key,
                        value=value,
                        method_call_id=self.current_call_id,
                    )

            added.append(row_id)
            logger.info(f"Added '{path.name}' ({kind}_id = {row_id}).")

        self.set_output({f"{kind}_ids": added})
        logger.info(f"Added {len(added)} of {seen} {what}s. {CHECK}")

        return added

    @record_call
    def process_sync(
        self,
        *,
        loop_id: int,
        sync_path: None | str = None,
        max_skew_s: float = 5.0,
    ) -> None:
        """
        Process a loop's sync file into `streams` and `sync_blocks`.

        **PARAMETERS**
        - `loop_id` is the loop the file was recorded for
        - `sync_path` is the H5 file, relative to the `main_folder`. Left out,
        it is the one file in `<loop>/sync/`
        - `max_skew_s` is how far the ScanImage and olfactometer clocks may
        disagree before the call is flagged (the match itself is by order, so
        this is a sanity check, not a tolerance)

        **EXAMPLE**
        ```python
        db.process_sync(loop_id=12)
        ```

        Frame-clock bursts are matched to the loop's acquisitions in
        order, and the bursts with an odor pulse to its trials in order, which
        fills `trials.acq_id`. Running it again replaces what the earlier run
        wrote, so a sync file copied late or a corrected one is not a problem.
        """
        with self._locked() as con:
            loop = con.execute(
                "SELECT session_id, loop_path, frames_per_slice FROM loops "
                "WHERE loop_id = ?;",
                [int(loop_id)],
            ).fetchone()

            if loop is None:
                raise ValueError(f"There is no loop {loop_id}.")

            acquisitions = con.execute(
                "SELECT acq_id, acq_start FROM acquisitions "
                "WHERE loop_id = ? ORDER BY acq_start;",
                [int(loop_id)],
            ).fetchall()

            trials = con.execute(
                "SELECT trial_id, trial_odor_start FROM trials "
                "WHERE loop_id = ? ORDER BY trial_start;",
                [int(loop_id)],
            ).fetchall()

        path = self._sync_file(loop["loop_path"], sync_path)
        rel_path = path.relative_to(self.main_folder).as_posix()

        logger.info(f"Processing '{rel_path}'")

        # Read without the lock: a file is hundreds of megabytes
        decoded = decode(path)

        self._check_sync(decoded, acquisitions, trials, loop["frames_per_slice"])

        blocks = decoded.blocks
        acq_ids = [row["acq_id"] for row in acquisitions]
        odor_blocks = [(i, b) for i, b in enumerate(blocks) if b.odor_pulses]

        # Clock skew between the two computers, one number per acquisition.
        # Wall clock on the olfactometer side needs the file's start time.
        skews: list[None | float] = []

        for block, row in zip(blocks, acquisitions):
            started = decoded.time_at(block.start_sample)
            acq_start = datetime.strptime(row["acq_start"], DT_FORMAT)

            skews.append(
                None
                if started is None
                else (acq_start - started).total_seconds() * 1000
            )

        finite = [abs(s) for s in skews if s is not None]

        if finite and max(finite) > max_skew_s * 1000:
            self.add_flag(SyncFlag.CLOCK_SKEW)
            logger.warning(
                f"ScanImage and olfactometer clocks differ by up to "
                f"{max(finite) / 1000:.1f} s. {CROSS}"
            )

        # Trials share the olfactometer's clock with the sync file, so their
        # odor onsets should agree to well under a second.
        trial_skews = []

        for (_, block), row in zip(odor_blocks, trials):
            assert block.odor_on_sample is not None
            valve = decoded.time_at(block.odor_on_sample)

            if valve is not None:
                logged = datetime.strptime(row["trial_odor_start"], DT_FORMAT)
                trial_skews.append(abs((logged - valve).total_seconds()))

        if trial_skews and max(trial_skews) > max_skew_s:
            self.add_flag(SyncFlag.CLOCK_SKEW)
            logger.warning(
                f"Olfactometer and sync file odor onsets differ by up to "
                f"{max(trial_skews):.1f} s. {CROSS}"
            )

        with self._writing() as cur:
            # Replaced as a whole: the rows of one file are one call
            cur.execute(
                "DELETE FROM streams WHERE loop_id = ? AND stream_type = 'sync';",
                [int(loop_id)],
            )
            cur.execute(
                "UPDATE trials SET acq_id = NULL WHERE loop_id = ?;", [int(loop_id)]
            )

            stream_id = self._insert_stream(
                cur, decoded, rel_path, loop["session_id"], int(loop_id)
            )
            self._insert_blocks(
                cur,
                stream_id,
                decoded,
                [
                    (block, {"acq_id": acq_id}, skew)
                    for block, acq_id, skew in zip(blocks, acq_ids, skews)
                ],
            )

            cur.executemany(
                "UPDATE trials SET acq_id = ? WHERE trial_id = ?;",
                [
                    (acq_ids[index], row["trial_id"])
                    for (index, _), row in zip(odor_blocks, trials)
                ],
            )

        self.set_output(
            {
                **summary(decoded),
                "stream_id": stream_id,
                "matched_trials": len(trials),
                "clock_offset_ms": None if not finite else max(finite),
            }
        )

        logger.info(
            f"Processed {len(blocks)} acquisitions and matched {len(trials)} "
            f"trials. {CHECK}"
        )

    @record_call
    def process_grabs_sync(
        self,
        *,
        session_id: int,
        sync_path: str,
        max_skew_s: float = 5.0,
    ) -> None:
        """
        Process a sync file recorded during a session's grabs.

        **PARAMETERS**
        - `session_id` is the session the grabs belong to
        - `sync_path` is the H5 file, relative to the `main_folder` (one in
        `<session>/grabs/sync/`)
        - `max_skew_s` is how far a burst may start from a grab's first frame,
        the ScanImage and olfactometer clocks included

        **EXAMPLE**
        ```python
        db.process_grabs_sync(
            session_id=3,
            sync_path="20260623/m462/grabs/sync/m462_grabs_2026-06-23_14-02-11.h5",
        )
        ```

        Grabs are taken whenever needed, so a file may hold any of them: each
        frame-clock burst is matched to the grab whose first frame is nearest
        in time, within `max_skew_s`, and bursts with no grab are left out.
        Running it again replaces what the earlier run of the same file wrote.
        """
        with self._locked() as con:
            grabs = con.execute(
                "SELECT grab_id, grab_start, frames_per_slice FROM grabs "
                "WHERE session_id = ? ORDER BY grab_start;",
                [int(session_id)],
            ).fetchall()

        path = self._sync_file("", sync_path)
        rel_path = path.relative_to(self.main_folder).as_posix()
        logger.info(f"Processing '{rel_path}'")
        decoded = decode(path)

        if decoded.header.interrupted:
            self.add_flag(SyncFlag.INTERRUPTED)
            logger.warning("The recording was not stopped properly (no stop time).")

        if decoded.header.start_time is None:
            self.fail(
                SyncFlag.BLOCK_COUNT_MISMATCH,
                "Cannot compute the bursts' start times (the file has no start time).",
            )

        # Grabs vary in length, so a short burst is as likely a grab as not
        bursts = sorted(decoded.blocks + decoded.fragments, key=lambda b: b.sync_block)
        free = {row["grab_id"]: row for row in grabs}
        matched: list[tuple] = []

        for block in bursts:
            started = decoded.time_at(block.start_sample)
            assert started is not None
            nearest = min(
                free.values(),
                key=lambda row: abs(_parse_dt(row["grab_start"]) - started),
                default=None,
            )

            if nearest is None:
                break

            skew = (_parse_dt(nearest["grab_start"]) - started).total_seconds()

            if abs(skew) > max_skew_s:
                continue

            del free[nearest["grab_id"]]
            matched.append((block, {"grab_id": nearest["grab_id"]}, skew * 1000))

            if block.frame_count != nearest["frames_per_slice"]:
                self.add_flag(SyncFlag.FRAME_COUNT)
                logger.warning(
                    f"Burst {block.sync_block} has {block.frame_count} frames, but "
                    f"grab {nearest['grab_id']} has {nearest['frames_per_slice']}."
                )

            if block.odor_pulses:
                self.add_flag(SyncFlag.ODOR_PULSES)
                logger.warning(f"Burst {block.sync_block} has an odor presentation.")

            if not block.camera_frames:
                self.add_flag(SyncFlag.NO_CAMERA)
                logger.warning(
                    f"No camera strobes during burst {block.sync_block} (no pupil)."
                )

        if not matched:
            self.fail(
                SyncFlag.BLOCK_COUNT_MISMATCH,
                f"None of the {len(bursts)} frame bursts in '{path.name}' starts "
                f"within {max_skew_s:g} s of one of the session's {len(grabs)} "
                "grabs.",
            )

        with self._writing() as cur:
            # Replaced as a whole: the rows of one file are one call
            cur.execute("DELETE FROM streams WHERE file_path = ?;", [rel_path])
            cur.executemany(
                "DELETE FROM sync_blocks WHERE grab_id = ?;",
                [(ids["grab_id"],) for _, ids, _ in matched],
            )
            stream_id = self._insert_stream(
                cur, decoded, rel_path, int(session_id), None
            )
            self._insert_blocks(cur, stream_id, decoded, matched)

        self.set_output(
            {
                **summary(decoded),
                "stream_id": stream_id,
                "matched_grabs": len(matched),
                "unmatched_bursts": len(bursts) - len(matched),
            }
        )

        logger.info(f"Matched {len(matched)} of {len(bursts)} bursts to grabs. {CHECK}")

    def _insert_stream(
        self,
        cur: Cursor,
        decoded: DecodedSync,
        rel_path: str,
        session_id: int,
        loop_id: None | int,
    ) -> int:
        header = decoded.header

        return _db_insert(
            cur,
            "streams",
            {
                "session_id": session_id,
                "loop_id": loop_id,
                "stream_type": "sync",
                "file_path": rel_path,
                "rate_hz": header.rate_hz,
                "start_time": _optional_datetime_str(header.start_time),
                "stop_time": _optional_datetime_str(header.stop_time),
                "n_samples": header.n_samples,
                "attributes": json.dumps(header.attributes),
                "method_call_id": self.current_call_id,
            },
        )

    def _insert_blocks(
        self, cur: Cursor, stream_id: int, decoded: DecodedSync, pairs: list[tuple]
    ) -> None:
        """One `sync_blocks` row per `(block, {acq_id or grab_id}, skew_ms)`."""
        for block, target, skew in pairs:
            _db_insert(
                cur,
                "sync_blocks",
                {
                    "stream_id": stream_id,
                    "sync_block": block.sync_block,
                    **target,
                    "sync_start_sample": block.start_sample,
                    "sync_frame_count": block.frame_count,
                    "sync_frame_period_s": block.frame_period_s,
                    "sync_odor_pulses": block.odor_pulses,
                    "sync_odor_on_frame": block.odor_on_frame,
                    "sync_odor_off_frame": block.odor_off_frame,
                    "sync_odor_on_residual_s": block.odor_on_residual_s,
                    "sync_odor_start": _optional_datetime_str(
                        None
                        if block.odor_on_sample is None
                        else decoded.time_at(block.odor_on_sample)
                    ),
                    "sync_odor_end": _optional_datetime_str(
                        None
                        if block.odor_off_sample is None
                        else decoded.time_at(block.odor_off_sample)
                    ),
                    "sync_camera_frames": block.camera_frames,
                    "clock_offset_ms": skew,
                    "method_call_id": self.current_call_id,
                },
            )

    def _sync_file(self, folder_path: str, sync_path: None | str) -> Path:
        """The sync file to process, named or found in `<folder>/sync/`."""
        if sync_path is not None:
            path = self.main_folder / sync_path

            if not path.is_file():
                self.fail(SyncFlag.NO_SYNC_FILE, f"No sync file at '{path}'.")

            return path.resolve()

        folder = self.main_folder / folder_path / SYNC_FOLDER
        found = sorted(folder.glob("*.h5")) if folder.is_dir() else []
        found = [p for p in found if not p.name.startswith(".")]

        if not found:
            self.fail(
                SyncFlag.NO_SYNC_FILE,
                f"No sync file in '{folder}'. Copy the H5 there, or use `sync_path`.",
            )

        if len(found) > 1:
            self.fail(
                SyncFlag.MULTIPLE_SYNC_FILES,
                f"{len(found)} H5 files in '{folder}', pick one of them with "
                "`sync_path`.",
            )

        return found[0].resolve()

    def _check_sync(
        self,
        decoded: DecodedSync,
        acquisitions: list,
        trials: list,
        frames_per_slice: int,
    ) -> None:
        """Flag what is off about a sync file, and refuse what cannot be matched."""
        blocks = decoded.blocks

        if decoded.header.interrupted:
            self.add_flag(SyncFlag.INTERRUPTED)
            logger.warning("The recording was not stopped properly (no stop time).")

        if decoded.fragments:
            self.add_flag(SyncFlag.FRAGMENTS)
            sizes = [f.frame_count for f in decoded.fragments]
            logger.warning(f"Ignored {len(sizes)} short bursts of {sizes} frames.")

        if len(blocks) != len(acquisitions):
            self.fail(
                SyncFlag.BLOCK_COUNT_MISMATCH,
                f"The sync file has {len(blocks)} frame bursts but the loop "
                f"has {len(acquisitions)} acquisitions, so they cannot be matched "
                "by order.",
            )

        odd = [b.sync_block for b in blocks if b.odor_pulses != 1]

        if odd:
            self.add_flag(SyncFlag.ODOR_PULSES)
            logger.warning(
                f"Bursts {odd} have more or less than one odor presentation."
            )

        with_odor = sum(1 for b in blocks if b.odor_pulses)

        if not trials:
            self.add_flag(SyncFlag.NO_TRIALS)
            logger.warning(
                "The loop has no trials (no olfactometer Events file was "
                "processed), so the odor bursts are stored without matches."
            )

        elif with_odor != len(trials):
            self.fail(
                SyncFlag.TRIAL_COUNT_MISMATCH,
                f"{with_odor} bursts have an odor pulse but the loop has "
                f"{len(trials)} trials, so they cannot be matched by order.",
            )

        wrong = [b.sync_block for b in blocks if b.frame_count != frames_per_slice]

        if wrong:
            self.add_flag(SyncFlag.FRAME_COUNT)
            logger.warning(
                f"Bursts {wrong} do not have the {frames_per_slice} frames ScanImage "
                "was set to."
            )

        no_camera = [b.sync_block for b in blocks if not b.camera_frames]

        if no_camera:
            self.add_flag(SyncFlag.NO_CAMERA)
            logger.warning(f"No camera strobes during bursts {no_camera} (no pupil).")

    def _load_programs(self, folder: Path) -> list[dict]:
        """Parse the olfactometer event files of a loop folder."""
        event_files = sorted(
            folder.rglob("[!.]?*Events.csv"), key=lambda x: x.stat().st_mtime
        )

        logger.info(f"Found {len(event_files)} olfactometer event files.")

        if not event_files:
            return []

        with self._locked() as con:
            rows = con.execute("SELECT odor_name, odor_id FROM odors;").fetchall()

        odors: dict[str, int] = {name: id for name, id in rows}

        stem_split = event_files[0].stem.split("-")
        events_start = datetime.strptime(
            " ".join(stem_split[-3:-1]), "%Y_%m_%d %H_%M_%S"
        )
        program_starts = _parse_program_starts(self, events_start, folder)

        programs_data = _load_event_data(
            event_files, program_starts, odors, self.main_folder
        )

        configs = _program_configs(folder / OLFACTOMETER_FOLDER / "config")

        for program_data in programs_data:
            name = program_data["metadata"]["program_name"]
            config = configs.get(name)

            if config is None:
                logger.warning(f"No olfactometer config found for '{name}'.")
                self.add_flag(IngestFlag.NO_PROGRAM_CONFIG)

            program_data["metadata"]["program_config"] = config

        return programs_data


def _has_database(path: Path) -> bool:
    # Connecting to a missing file creates it empty, so an empty file is what a
    # failed or interrupted creation leaves behind, not a database.
    return path.is_file() and path.stat().st_size > 0


def read_tiff(path: Path, main_folder: Path) -> None | tuple[Object, Object]:
    """
    What a ScanImage TIFF says about its recording and about itself, or
    `None` when it is not a ScanImage file this can read. Paths are given
    relative to `main_folder`.

    The first dict (about the recording) must be the same across a loop's
    TIFFs; its `stack` is `None` unless ScanImage took a z-stack.
    """
    with TiffFile(path) as tif:
        return _read_open_tiff(tif, Path(path), Path(main_folder))


def _read_open_tiff(
    tif: TiffFile, path: Path, main_folder: Path
) -> None | tuple[Object, Object]:
    if (
        not tif.is_scanimage
        or tif.scanimage_metadata is None
        or tif.scanimage_metadata["FrameData"] is None
        or not isinstance(tif.pages[0], TiffPage)
    ):
        return None

    # Get file SI metadata
    SI_metadata = tif.scanimage_metadata["FrameData"]
    laser_powers = SI_metadata["SI.hBeams.powers"]

    # Get the data that must be the same across a loop
    file_stem_parts = path.stem.split("_")

    recording: Object = {
        "name": "_".join(file_stem_parts[:-1]),
        "acq_state": SI_metadata["SI.acqState"],
        "height_px": tif.pages[0].tags["ImageLength"].value,
        "width_px": tif.pages[0].tags["ImageWidth"].value,
        "frames_per_slice": _whole_number(
            SI_metadata["SI.hStackManager.framesPerSlice"],
            "Frames per slice",
            path,
        ),
        "frame_rate": SI_metadata["SI.hRoiManager.scanFrameRate"],
        "z_um": _sample_z(SI_metadata),
        "laser_power_920": _whole_number(
            laser_powers[0],
            "Laser power at 920 nm",
            path,
        ),
        "laser_power_1040": _whole_number(
            laser_powers[1],
            "Laser power at 1040 nm",
            path,
        ),
        "loop_acq_interval_s": SI_metadata["SI.loopAcqInterval"],
        "stack": _stack(SI_metadata),
    }

    # Assume unit is centimeters
    # TODO: Check that units are centimeters
    dx, nx = tif.pages[0].tags["XResolution"].value
    dy, ny = tif.pages[0].tags["YResolution"].value

    # um per pixels in each direction
    factor_x = round(1e4 * nx / dx, 4)
    factor_y = round(1e4 * ny / dy, 4)

    # Size of image in um
    recording["width_um"] = recording["width_px"] * factor_x
    recording["height_um"] = recording["height_px"] * factor_y

    # Parse ImageDescription
    image_description = dict(
        line.split(" = ")
        for line in tif.pages[0].tags["ImageDescription"].value.splitlines()
    )

    # Parse ImageDescription epoch as a datetime
    date_string = image_description["epoch"].strip("[]")
    date_string = " ".join(date_string.split())
    started = datetime.strptime(date_string, "%Y %m %d %H %M %S.%f")

    recording["start"] = started

    # Data specific to the TIFF: when its first frame was taken
    delta_sec = float(image_description["frameTimestamps_sec"])

    tiff: Object = {
        "raw_path": path.relative_to(main_folder).as_posix(),
        "acq_start": started + timedelta(seconds=delta_sec),
    }

    return recording, tiff


def _dt_string(value: datetime) -> str:
    """
    A datetime as the recording tables hold it (DT_FORMAT).

    Formatted rather than passed as a datetime: sqlite3's adapter writes
    isoformat, which drops '.000000' when the time lands exactly on a second.
    """
    assert isinstance(value, datetime)

    return value.strftime(DT_FORMAT)


def _contract(recording: Object, um_scale: float) -> Object:
    """
    The columns a recording table takes from `_read_tiff`'s first part, with
    the field of view in real microns (`um_scale` from `OBJECTIVE_SCALE`).
    """
    row = {key: recording[key] for key in CONTRACT}
    row["height_um"] = recording["height_um"] * um_scale
    row["width_um"] = recording["width_um"] * um_scale

    return row


def _sample_z(si: Object) -> None | float:
    """
    Where ScanImage's sample z was, in microns from the z = 0 the lab sets at
    the landmark (negative is deeper), or `None` when the file does not say.
    """
    position = si.get("SI.hMotors.samplePosition")

    try:
        return float(position[2])  # type: ignore[index]

    except (TypeError, IndexError, ValueError):
        return None


def _first(value) -> Value:
    """The first beam's entry of a ScanImage per-beam setting."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None

    if isinstance(value, str):
        parts = value.strip("[]").split()
        return parts[0] if parts else None

    return value


def _stack(si: Object) -> None | Object:
    """
    The z-stack ScanImage took, or `None` when it took a single plane.

    `z_start_um` is the shallowest slice and `z_end_um` the deepest (z is
    negative below the landmark). The 920 nm power at each end follows
    ScanImage's power-depth adjustment: exponential with
    `hBeams.lengthConstants`, from `hBeams.powers` at
    `hStackManager.zPowerReference`, capped at `powerFractionLimits`.
    """
    slices = si.get(
        "SI.hStackManager.actualNumSlices", si.get("SI.hStackManager.numSlices")
    )

    if not si.get("SI.hStackManager.enable") or slices is None or int(slices) < 2:
        return None

    zs = si.get("SI.hStackManager.zs")
    step = si.get(
        "SI.hStackManager.actualStackZStepSize",
        si.get("SI.hStackManager.stackZStepSize"),
    )

    if isinstance(zs, (int, float)):
        zs = [zs]

    if not zs or step is None:
        return None

    start, end = max(float(z) for z in zs), min(float(z) for z in zs)
    stack: Object = {
        "slice_count": int(slices),
        "z_step_um": round(abs(float(step)), 3),  # ScanImage's has float noise
        "z_start_um": start,
        "z_end_um": end,
        "power_start_pct": None,
        "power_end_pct": None,
        "power_length_um": None,
    }

    try:
        power = float(_first(si["SI.hBeams.powers"]))  # type: ignore[arg-type]

    except (KeyError, TypeError, ValueError):
        return stack

    limit = 100.0

    try:
        limit = 100.0 * float(_first(si.get("SI.hBeams.powerFractionLimits", 1)))  # type: ignore[arg-type]

    except (TypeError, ValueError):
        pass

    adjust = str(_first(si.get("SI.hBeams.pzAdjust")) or "")
    reference = si.get("SI.hStackManager.zPowerReference")

    try:
        length = float(_first(si.get("SI.hBeams.lengthConstants")))  # type: ignore[arg-type]

    except (TypeError, ValueError):
        length = math.inf

    def at(z: float) -> float:
        if (
            "Exponential" not in adjust
            or reference is None
            or not math.isfinite(length)
        ):
            return min(power, limit)

        return min(power * math.exp((z - float(reference)) / length), limit)

    stack["power_start_pct"] = round(at(start), 2)
    stack["power_end_pct"] = round(at(end), 2)

    if "Exponential" in adjust and math.isfinite(length):
        stack["power_length_um"] = length

    return stack


def _check_session(recording: Object, session: sqlite3.Row, db: Database) -> None:
    """
    Refuse a recording whose file names disagree with the session given.

    ScanImage names files `<date>_m<mouse>_<...>`, from what was typed into it.
    A folder submitted under the wrong session would land its acquisitions on
    the wrong mouse or day, with nothing to say so.
    """
    name = str(recording["name"])
    match = re.match(r"(\d{8})_([A-Za-z]*\d+)_", name)

    if match is None:
        logger.warning(f"Cannot read the mouse and date from the file name '{name}'")
        return

    day = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    mouse = _mouse_number(match.group(2))

    if (mouse, day) != (session["mouse_id"], session["session_date"]):
        db.fail(
            IngestFlag.NOT_THE_SESSION,
            f"The TIFFs {name}_* were recorded as m{mouse} on {day}, but the "
            f"session is m{session['mouse_id']} on {session['session_date']}.",
        )


def _insert_program(cur: Cursor, loop_id: int, program_data: dict) -> int:
    """Insert one program with its trials and events."""
    metadata = program_data["metadata"]
    assert isinstance(metadata["program_start"], datetime)

    program_id = _db_insert(
        cur,
        "programs",
        {
            **metadata,
            "loop_id": loop_id,
            "program_start": metadata["program_start"].strftime(DT_FORMAT),
        },
    )

    # Pass 1: insert trials, collect trial_ids by index
    trial_ids: dict[int, int] = {}

    for trial_idx, trial in enumerate(program_data["trials"]):
        trial_ids[trial_idx] = _db_insert(
            cur,
            "trials",
            {
                "trial_start": trial["trial_start"].strftime(DT_FORMAT),
                "trial_odor_start": trial["odor_start"].strftime(DT_FORMAT),
                "trial_odor_end": trial["odor_end"].strftime(DT_FORMAT),
                "odor_id": trial["odor_id"],
                "outcome": trial["outcome"],
                "acq_id": None,
                "program_id": program_id,
                "loop_id": loop_id,
            },
        )

    # Pass 2: insert all events in order
    for trial_idx, event in program_data["events"]:
        _db_insert(
            cur,
            "events",
            {**event, "program_id": program_id, "trial_id": trial_ids.get(trial_idx)},
        )

    return program_id


def _program_configs(folder: Path) -> dict[str, str]:
    """
    The program definitions in the olfactometer's `Program.config`, copied
    into `folder` (a loop's `olfactometer/config`), keyed by program name as
    JSON text ready to store.

    The file is a tree of folders and programs. A name used by programs that
    differ (the same name in two folders) is left out with a warning, since
    the log names a program only by its name.
    """
    path = folder / "Program.config"

    if not path.is_file():
        return {}

    try:
        nodes = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError:
        logger.warning(f"Could not read '{path.name}' as JSON.")
        return {}

    found: dict[str, set[str]] = {}

    def walk(items) -> None:
        for node in items if isinstance(items, list) else []:
            if not isinstance(node, dict):
                continue

            program = node.get("Program")

            if not node.get("IsFolder") and isinstance(program, dict):
                name = str(node.get("Name") or program.get("Name"))
                found.setdefault(name, set()).add(json.dumps(program, sort_keys=True))

            walk(node.get("Items"))

    walk(nodes)
    configs = {}

    for name, definitions in found.items():
        if len(definitions) > 1:
            logger.warning(
                f"'{name}' names {len(definitions)} different programs in "
                f"'{path.name}', so its definition is not stored."
            )
            continue

        configs[name] = definitions.pop()

    return configs


def _db_insert(cur: Cursor, table_name: str, data: Object | list[Object]) -> int:
    # HACK:
    #   ONLY FOR INTERNAL USE (CAN BE USED FOR SQL INJECTION)
    #   Column names are not validated, for simplicity

    template = data[0] if isinstance(data, list) else data

    insertion_query = (
        f"INSERT INTO {table_name} "
        f"({', '.join(template.keys())}) "
        f"VALUES (:{', :'.join(template.keys())});"
    )

    if isinstance(data, list):
        cur.executemany(insertion_query, data)

    else:
        cur.execute(insertion_query, data)

    # Check to make output type == int
    lastrowid = cur.lastrowid
    assert lastrowid is not None

    return lastrowid


# --------------------------------------------------------------------------- #
# Subjects and odors
# --------------------------------------------------------------------------- #


def _mouse_number(name: int | str) -> int:
    """
    The number a mouse is known by, from the name its folders and files use.

    Raises `ValueError` for a name that is not a number with an optional letter
    prefix, since guessing which animal was meant would attach a recording to
    the wrong one.

    **EXAMPLE**
    ```python
    _mouse_number("m442")   # 442
    ```
    """
    match = re.fullmatch(r"[A-Za-z]*0*(\d+)", str(name).strip())

    # Zero is the placeholder people write when a recording is not about a
    # particular animal, so it names no mouse either.
    if match is None or int(match.group(1)) == 0:
        raise ValueError(
            f"{name!r} is not a valid mouse name (which is an optional prefix and "
            f"a number above zero, as in 'm442')."
        )

    return int(match.group(1))


# The wild-type line (C57BL/6) has no mutation to track so its genotype is
# always recorded as 'na'. (And the dropdown menu reflects that.)
WILD_TYPE = "Wild Type"

GENOTYPES = {
    "na": "na",
    "n/a": "na",
    "not genotyped": "na",
    "wt": "wt",
    "wildtype": "wt",
    "wild type": "wt",
    "+/+": "wt",
    "het": "het",
    "heterozygous": "het",
    "+/-": "het",
    "hom": "hom",
    "homozygous": "hom",
    "-/-": "hom",
}


def _mouse_lines(lines: dict[str, str]) -> dict[str, str]:
    """
    Clean a line-to-genotype mapping for storage.

    Line names keep the spelling they were given, minus surrounding spaces.
    The genotype is matched case-insensitively against the spellings people
    write and stored as `'na'` (which means not genotyped), `'wt'`, `'het'` or
    `'hom'`. One of the options must be given, there is no default.

    **EXAMPLE**
    ```python
    _mouse_lines({"TH-Cre ": "Het", "TIGRE2": "na"})
    # {"TH-Cre": "het", "TIGRE2": "na"}
    ```
    """
    cleaned: dict[str, str] = {}

    for line, genotype in lines.items():
        name = str(line).strip()

        if not name:
            raise ValueError("A line needs a name.")

        known = GENOTYPES.get(str(genotype).strip().lower())

        if known is None:
            raise ValueError(
                f"{genotype!r} is not a valid genotype for {name}. Use one of: "
                f"{', '.join(sorted(set(GENOTYPES.values())))}."
            )

        cleaned[name] = known

    return cleaned


def _registered(registry: pd.DataFrame, what: str, name: str) -> str:
    """
    `name` as `registry` (indexed by name) spells it, matched in any case.
    Raises `ValueError` naming it when it is not there.
    """
    spelled = {str(known).lower(): str(known) for known in registry.index}
    found = spelled.get(str(name).strip().lower())

    if found is None:
        raise ValueError(
            f"Unknown {what} {str(name).strip()!r}. Add it with `add_{what}`."
        )

    return found


def _known_lines(registry: pd.DataFrame, lines: dict[str, str]) -> dict[str, str]:
    """
    `_mouse_lines`, each name matched to one in `registry` and stored as
    spelled there. Raises `ValueError` naming an unknown line.
    """
    known: dict[str, str] = {}

    for name, genotype in _mouse_lines(lines).items():
        name = _registered(registry, "line", name)

        if name == WILD_TYPE and genotype != "na":
            raise ValueError(f"'{WILD_TYPE}' genotype must be 'na'.")

        known[name] = genotype

    return known


def _date_string(value: Value) -> str:
    """A date as `YYYY-MM-DD`, from a date, a datetime or an ISO string."""
    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    try:
        return date.fromisoformat(str(value).strip()).isoformat()

    except ValueError as error:
        raise ValueError(f"Not a date (YYYY-MM-DD): {value!r}.") from error


def _parse_dt(value: str) -> datetime:
    """A recording table's time (DT_FORMAT) back as a datetime."""
    return datetime.strptime(value, DT_FORMAT)


def _optional_datetime_str(value: None | datetime) -> None | str:
    return None if value is None else value.strftime(DT_FORMAT)


def _optional_real(value: Value) -> None | float:
    if value is None:
        return None

    number = float(value)  # type: ignore[arg-type]

    if number != number:  # NaN reads as "not given"
        return None

    return number


def _panel_id(cur: Cursor, panel_name: str) -> int:
    # Names are case-insensitive ("16 odors" ~ "16 Odors")
    row = cur.execute(
        "SELECT panel_id FROM panels WHERE panel_name = ? COLLATE NOCASE;",
        [str(panel_name).strip()],
    ).fetchone()

    if row is None:
        raise ValueError(f"No panel named {panel_name!r}. Add it with `add_panel`.")

    return int(row["panel_id"])


VIAL_COLUMNS = ("vial_sccm", "total_sccm", "total_volume_ml")


def _insert_vial(
    cur: Cursor,
    panel_id: int,
    holds: dict[int, int],
    vial: Object,
    day: str,
    mixes: dict[int, bool],
    made_of: dict[int, set[int]],
) -> int:
    """
    Insert one vial with its components, checking it is for one of the
    panel's positions (`holds` maps each to its odor) and the recipe makes
    sense for that odor.
    """
    missing = [c for c in ("vial_position", *VIAL_COLUMNS) if c not in vial]

    if missing:
        raise ValueError(f"A vial needs {missing} (got {sorted(vial)}).")

    position = int(vial["vial_position"])  # type: ignore[arg-type]

    if position not in holds:
        raise ValueError(
            f"The panel has no position {position} (to change positions, make a "
            "new panel)."
        )

    odor_id = holds[position]

    if vial.get("odor_id") is not None and int(vial["odor_id"]) != odor_id:  # type: ignore[arg-type]
        raise ValueError(
            f"Position {position} holds odor {odor_id}, not {vial['odor_id']} (to "
            "change positions/odors, make a new panel)."
        )

    components = vial.get("components") or []
    assert isinstance(components, list)

    component_ids = [int(c["odor_id"]) for c in components]  # type: ignore[index]

    for component_id in component_ids:
        if mixes.get(component_id, True):
            raise ValueError(
                f"Component {component_id} of vial {vial['vial_position']} is "
                "not a monomolecular odor."
            )

    if mixes[odor_id]:
        if set(component_ids) != made_of.get(odor_id, set()):
            raise ValueError(
                f"Vial {vial['vial_position']} holds odor {odor_id}, which is a mix "
                f"of {sorted(made_of.get(odor_id, ()))} (list all of those)."
            )

    elif set(component_ids) - {odor_id}:
        raise ValueError(
            f"Vial {vial['vial_position']} holds odor {odor_id}, which is not a "
            f"mix, so its only component can be itself."
        )

    made_on = vial.get("made_on")

    vial_id = _db_insert(
        cur,
        "vials",
        {
            "panel_id": panel_id,
            "vial_position": position,
            **{column: float(vial[column]) for column in VIAL_COLUMNS},  # type: ignore[arg-type]
            "solvent_volume_ml": _optional_real(vial.get("solvent_volume_ml")),
            "made_on": day if made_on is None else _date_string(made_on),
        },
    )

    for component in components:
        assert isinstance(component, dict)

        _db_insert(
            cur,
            "vial_odors",
            {
                "vial_id": vial_id,
                "odor_id": int(component["odor_id"]),  # type: ignore[arg-type]
                "target_ppm": float(component["target_ppm"]),  # type: ignore[arg-type]
                "liquid_ul": _optional_real(component.get("liquid_ul")),
                "percent_vv": _optional_real(component.get("percent_vv")),
            },
        )

    return vial_id


# --------------------------------------------------------------------------- #
# Annotations
# --------------------------------------------------------------------------- #


def _db_annotate(
    cur: Cursor,
    *,
    target_type: str,
    target_id: int,
    key: str,
    value: Value,
    method_call_id: int,
) -> int:
    """Write one annotation, checking the target and the registry first."""

    # `target_id` usually arrives as a numpy integer, because the obvious way to
    # get one is out of a DataFrame index. That is not an `int` to `isinstance`
    # and sqlite3 will not adapt it, so coerce rather than refuse.
    target_id = int(target_id)

    # No foreign key can cover a polymorphic target, so this stands in for one:
    # without it an annotation can name a row that does not exist, and nothing
    # would ever say so.
    if target_type not in ANNOTATION_TARGETS:
        raise ValueError(
            f"Cannot annotate a {target_type!r}. Use one of: "
            f"{sorted(ANNOTATION_TARGETS)}."
        )

    spec = ANNOTATION_TARGETS[target_type]

    exists = cur.execute(
        f"SELECT EXISTS(SELECT 1 FROM {spec.name} WHERE {spec.index} = ?);",
        [target_id],
    ).fetchone()[0]

    if not exists:
        raise ValueError(f"There is no {target_type} {target_id} to annotate.")

    # The registry declares a type per key and SQLite cannot enforce it: the
    # column is ANY, which is what lets a number stay a number. So the check
    # lives here, on the only path that writes.
    row = cur.execute(
        "SELECT value_type, allowed_values, retired "
        "  FROM annotation_keys "
        "  WHERE applies_to = ? AND key = ?;",
        [target_type, key],
    ).fetchone()

    if row is None:
        raise ValueError(
            f"No annotation key '{key}' for a {target_type}. Register it in "
            f"`annotation_keys` first, or check the spelling."
        )

    # Retiring is what a project has instead of deleting a key: old values stay
    # readable, new ones are not taken.
    if row["retired"]:
        raise ValueError(
            f"Annotation '{key}' is retired for a {target_type}, so it does not "
            f"take new values. Use `retire_annotation_key(..., retired=False)` to "
            f"bring it back."
        )

    # NULL clears the key: the latest row wins, so it reads as unset from now
    # on while what it used to say stays in the history.
    stored = (
        None
        if value is None
        else _annotation_value(key, value, row["value_type"], row["allowed_values"])
    )

    return _db_insert(
        cur,
        "annotations",
        {
            "target_type": target_type,
            "target_id": target_id,
            "key": key,
            "value": stored,
            "method_call_id": method_call_id,
        },
    )


BOOLEAN_WORDS = {
    "true": True,
    "yes": True,
    "y": True,
    "1": True,
    "false": False,
    "no": False,
    "n": False,
    "0": False,
}


def _annotation_value(
    key: str, value: Value, value_type: str, allowed_values: None | str
) -> Value:
    """Coerce `value` to what the registry says `key` holds, or explain why not."""

    try:
        match value_type:
            case "integer":
                number = float(value)  # type: ignore[arg-type]

                if not number.is_integer():
                    raise ValueError("not a whole number")

                return int(number)

            case "real":
                return float(value)  # type: ignore[arg-type]

            case "boolean":
                if isinstance(value, str):
                    if value.strip().lower() not in BOOLEAN_WORDS:
                        raise ValueError("not yes or no")

                    return int(BOOLEAN_WORDS[value.strip().lower()])

                return int(bool(value))

            case "date":
                return _date_string(value)

            case "enum":
                options = json.loads(allowed_values or "[]")
                chosen = [o for o in options if o.lower() == str(value).strip().lower()]

                if not chosen:
                    raise ValueError(f"expected one of {options}")

                return chosen[0]

            case _:
                # 'text' and 'long_text', stored as written
                return str(value)

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Annotation '{key}' is declared {value_type} "
            f"but got {value!r} ({error})."
        ) from error


# --------------------------------------------------------------------------- #
# Olfactometer files
# --------------------------------------------------------------------------- #


def _load_event_data(
    event_files: list[Path],
    program_starts: list[tuple[datetime, str]],
    odors: dict[str, int],
    main_folder: Path,
) -> list[dict]:
    """
    Parse all event files into structured program/trial/event dicts.

    Returns a list of program dicts, each containing:
        "metadata": fields for the programs table (no loop_id yet)
        "trials":   list of trial dicts (no events key)
        "events":   list of (trial_idx, event_record) in timeline order.
                    trial_idx is the index into "trials" for that event's trial,
                    or None for events before the first trial or after an
                    incomplete last trial.
    """
    programs = []

    for (program_start, name), event_file in zip(program_starts, event_files):
        stem_split = event_file.stem.split("-")
        program_name = "-".join(stem_split[:-3])

        assert name in program_name, (
            f"Program name from log ({name}) does not match"
            f" name from Events .csv ({program_name})"
        )

        if "buffer" in program_name.lower():
            continue

        program_type = "unknown"
        for t in PROGRAM_TYPES:
            if t in program_name:
                program_type = t

        metadata = {
            "program_name": program_name,
            "program_type": program_type,
            "program_start": program_start,
            "program_path": event_file.relative_to(main_folder).as_posix(),
        }

        df = _parse_event_file(event_file, program_start)

        trials: list[dict] = []
        events: list[tuple[None | int, Object]] = []

        trial: None | dict = None
        current_trial_idx: None | int = None
        trial_phase = TrialPhase.NOT_IN_TRIAL
        licks_count = 0

        for _, (et, event_name, event_type, event_tag) in df.iterrows():
            event_time: datetime = et.to_pydatetime()

            # Skip session start events
            if event_type == "Session":
                continue

            # Build event record for storage
            event_record: Object = {
                "event_time": event_time.strftime(DT_FORMAT),
                "event_type": event_type,
                "event_tag": event_tag,
            }

            if event_type == "Trial" and event_tag != "Interval":
                # Finalize previous trial
                if trial is not None:
                    trials.append(trial)

                # Stop processing if tag is not an integer
                try:
                    int(event_tag)

                except ValueError as e:
                    logger.error(f"Unexpected event: {event_name}")
                    logger.error("The loop will not be added to the DB.")
                    raise e

                current_trial_idx = len(trials)
                trial = {
                    "trial_start": event_time,
                    "odor_start": None,
                    "odor_end": None,
                    "odor_id": None,
                    "outcome": "na",
                }
                trial_phase = TrialPhase.TRIAL_START

            elif event_type == "Odor":
                assert trial is not None, "Odor presentation without trial"
                trial["odor_id"] = odors.get(event_tag.lower())
                trial["odor_start"] = event_time
                trial_phase = TrialPhase.ODOR_WINDOW

            elif trial_phase == TrialPhase.RESPONSE_WINDOW and event_type == "Lick":
                licks_count += 1

            elif trial_phase == TrialPhase.ODOR_WINDOW and event_type == "Delay":
                assert trial is not None, "Odor end without trial"
                trial["odor_end"] = event_time
                trial_phase = TrialPhase.INTERVAL

            elif event_type == "Response":
                assert trial is not None, "Response window without trial"
                trial["odor_end"] = event_time
                trial_phase = TrialPhase.RESPONSE_WINDOW

            elif event_type == "Reward" and trial_phase == TrialPhase.RESPONSE_WINDOW:
                assert trial is not None, "Reward without trial"
                trial["outcome"] = "hit"

            elif event_type == "Trial" and event_tag == "Interval":
                assert trial is not None, "Trial end without trial data"
                trial_phase = TrialPhase.TRIAL_END

                if program_type != "passive" and trial["outcome"] != "hit":
                    trial["outcome"] = "miss" if licks_count < 3 else "false choice"
                licks_count = 0

            # Record in timeline order. current_trial_idx is None before the first
            # trial, or points to a trial that may not be finalized yet (last trial
            # that didn't end) — in that case the insertion step maps it to NULL.
            events.append((current_trial_idx, event_record))

        # Attempts to adds last trial
        if trial is not None:

            # Trials that didn't reach INTERVAL are missing odor fields and cannot
            # satisfy the DB constraints, so they are skipped with a warning.
            if trial_phase < TrialPhase.INTERVAL:
                logger.warning(
                    f"Last trial discarded: ended at phase '{trial_phase.name}'"
                    " before odor delivery window ending event."
                )

            # Otherwise, we finish as if "Trial Interval" was emitted
            else:
                if program_type != "passive" and trial["outcome"] != "hit":
                    trial["outcome"] = "miss" if licks_count < 3 else "false choice"
                licks_count = 0

                if trial_phase < TrialPhase.RESPONSE_WINDOW:
                    logger.warning(
                        f"Last trial added with incomplete phase '{trial_phase.name}'."
                    )

                trials.append(trial)

        programs.append(
            {
                "metadata": metadata,
                "trials": trials,
                "events": events,
            }
        )

    return programs


def _parse_event_file(path: Path, program_start: datetime) -> pd.DataFrame:
    """
    Perform simple parsing into a DataFrame to be iterated over.
    """

    # Load file skipping the header "Mode: ..."
    # TODO: Check if this info is relevant and should be stored
    df = pd.read_csv(path, skiprows=1)

    # Split events that happen at the same time like:
    #   Odor I - ..., Output 4
    df["Events"] = df["Events"].str.split(",", n=1)
    df = df.explode("Events", ignore_index=True)

    # Split Events into simpler to parse columns
    df[["Type", "Tag"]] = df["Events"].str.split("[ _]", n=1, expand=True)
    df["Tag"] = df["Tag"].fillna("")

    # Simplify "Tag" value for df["Type"] == "Odor" to just the odor name
    mask = df["Tag"].str.startswith("I ")
    df.loc[mask, "Tag"] = df.loc[mask, "Tag"].str.split(" ").str[3:].str.join(" ")

    # Convert "TimeStamp" to datetime
    df["TimeStamp"] = df["TimeStamp"].apply(
        lambda ms: program_start + timedelta(milliseconds=ms)
    )

    return df


def _parse_program_starts(
    db: Database, start: datetime, loop_folder: Path
) -> list[tuple[datetime, str]]:
    """
    The programs started on `start`'s day from `start` on (1 s earlier, for
    rounding), read from the olfactometer's log for that day. That is the
    copy in the loop's `olfactometer/log/`, or else the one in the main
    folder's `.odyn/olfactometer/Log/Info/` (where older loops left it).
    """
    name = f"Program_{start.date().strftime('%Y%m%d')}.txt"
    candidates = [
        loop_folder / OLFACTOMETER_FOLDER / "log" / name,
        db.main_folder / INFO_FOLDER / start.date().strftime("%Y%m") / name,
    ]
    log_path = next((path for path in candidates if path.is_file()), None)

    if log_path is None:
        raise FileNotFoundError(
            f"No olfactometer log for {start.date()} ({name} in "
            f"'{loop_folder.name}/{OLFACTOMETER_FOLDER}/log' or "
            f"'{INFO_FOLDER}/{start.date():%Y%m}')."
        )

    starts = []

    # Olfactometer log entries format:
    # [TIMESTAMP] PROGRAM_EVENT: PROGRAM_NAME

    with open(log_path) as f:
        for line in f:
            timestamp, desc = line[1:].strip().split("] ", 1)

            if desc.startswith("Start program"):
                _, program_name = desc.split(": ", 1)

                t = time.fromisoformat(timestamp)
                dt = datetime.combine(start.date(), t)

                # 'start' is only precise up to seconds, so rounding might
                # have pushed 'start' to the second after the actual start.
                if dt >= start - timedelta(seconds=1):
                    starts.append((dt, program_name))

    return starts


def _whole_number(value, field: str, path: Path) -> int:
    """
    A ScanImage setting rounded to a whole number.

    Anything more than `WHOLE_TOLERANCE` from a whole number is logged as a warning.
    """
    number = float(value)
    nearest = round(number)

    if abs(number - nearest) > WHOLE_TOLERANCE:
        logger.warning(
            f"{field} is {number:g} in '{path.name}', which is not a whole "
            f"number. Recording it as {nearest}."
        )

    return nearest
