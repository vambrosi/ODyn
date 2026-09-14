# --------------------------------------------------------------------------- #
# NOTE:
#   - Programs with same name can be different
#   - RWD Olfactometer trigger
#   - All odors have concentration, even monomolecular ones (add to DB)
#   - Not in metadata: Odor Dilution (%v/v), Odor Made in Date
#   - Inconsistent a:b and a/b usage
#   - Record vial to odor association
#
# TODO:
#   - Add mcors that where already made to DB.
#   - Assert that non-passive trials have a non "na" outcome
#   - Start a test suite (so far only for DB tests)
#   - Add quality control plots for ported MATLAB code
#   - Add pavlovian reward to warm ups
#
#   - What to do about some particular experiments?
#       - '20250703/SID200' has more trials in .csv than in H5
#
# MAYBE TODO:
#   - Add 'computer' column/prefix when there is a folder in the table.
#   ( Paths starting with "." should be still relative to the
#       main_folder which is independent of the computer )
#   - Maybe use some env/global variables like $SERVER, $COMPUTER_NAME.
#   - Maybe should add small db for consolidation in folders that are not relative.
#   - Add git hash to every db entry? (To help db updates...)
#   - Make a function that creates .py file containing the whole
#     processing/analysis pipeline, to run on the server.
# --------------------------------------------------------------------------- #

import json
import re
import sqlite3

from collections import defaultdict
from datetime import time, datetime, timedelta
from pathlib import Path
from sqlite3 import Cursor
from tifffile import TiffFile, TiffPage
from typing import Final

import pandas as pd

from .groups import Group
from .migrate import SCHEMA_VERSION
from .utils import *
from .utils import CallFrame, CallRecorder
from .utils import _acquisition_trials, _method_calls_dataframe, _SYNC_COLUMNS

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TIMEDELTA_MS = timedelta(milliseconds=1)
DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# TIFF metadata that is tier 3 in `create.sql`: recorded, but read by nothing in
# odyn, so it is stored as annotations instead of columns. Read out of the same
# dict as the columns so the across-TIFF consistency check still sees it.
RIG_ANNOTATIONS = ("laser_power_920", "laser_power_1040", "loop_acq_interval_s")

# What an annotation can be attached to, and where that row lives. The target is
# polymorphic, so SQLite cannot check `annotations.target_id` with a foreign key
# and this is what stands in for one. Keep it in step with `applies_to` in
# `create.sql`.
ANNOTATION_TARGETS = {
    "mouse": ("mice", "mouse_id"),
    "session": ("sessions", "session_id"),
    "experiment": ("experiments", "exp_id"),
    "program": ("programs", "program_id"),
    "acquisition": ("acquisitions", "acq_id"),
    "group": ("groups", "group_id"),
}

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


class ExpFlag(IntFlag):
    """
    call_flag bits for `Database.add_experiment` (bit 0 reserved by `CallFlag.RAISED`).
    """

    # Bits 2, 5 and 6 belonged to the old olfactometer H5 and are left free
    # rather than reused, so an old log line cannot be misread as a new flag.
    ALREADY_IN_DB = 1 << 1  # experiment already present, nothing inserted
    UNSUPPORTED_METADATA = 1 << 3  # TIFF does not have expected metadata format
    METADATA_CHANGED = 1 << 4  # TIFF metadata changed, skipped
    NOT_A_GRAB = 1 << 7  # add_grab_folder found a file that was not a grab


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
    db = Database(main_folder)
    ```

    **RELEVANT PROPERTIES**
    ```python
        db.groups             # `List` of `Group`s for processing/analysis

        db.acquisitions       # `DataFrame` with acquisition metadata
        db.annotations        # `DataFrame` with everything people wrote down
        db.annotation_keys    # `DataFrame` with what may be annotated
        db.events             # `DataFrame` with olfactometer events
        db.experiments        # `DataFrame` with experiment metadata
        db.mcor_files         # `DataFrame` with mcor files metadata
        db.method_calls       # `DataFrame` with `@record_call` functions
        db.mice               # `DataFrame` with one row per animal
        db.mouse_lines        # `DataFrame` with the mutations each mouse carries
        db.odors              # `DataFrame` with current list of odors
        db.panels             # `DataFrame` with every known odor panel
        db.panel_vials        # `DataFrame` with what each vial delivers
        db.vial_components    # `DataFrame` with what was pipetted into each vial
        db.session_panels     # `DataFrame` with the panel each session ran
        db.session_vials      # `DataFrame` with each vial's mixing date
        db.programs           # `DataFrame` with one entry per _Event.csv_ file
        db.trials             # `DataFrame` with all olfactometer trials
    ```

    **RELEVANT METHODS**
    ```python
        db.add_experiment(...)          # Add a new experiment folder
        db.update(...)                  # Find and add all experiment folders
        db.latest_calls(method_name)    # `DataFrame` with `method_name` calls

        db.add_annotation(...)          # Save an annotation in the DB
        db.annotations_for(target)      # One row per target, one column per key
        db.missing_annotations()        # What is still to be filled in
        db.add_annotation_key(...)      # Allow a new kind of annotation
        db.retire_annotation_key(...)   # Stop offering one, keep its values

        db.set_mouse(...)               # Record an animal's sex, DOB and line
        db.add_panel(...)               # Register a rack of vials as a recipe
        db.set_session_panel(...)       # Say which panel a session ran
    ```
    """

    def __init__(
        self,
        path: str | Path,
        update=False,
        project: None | str = DEFAULT_PROJECT,
        _is_test=False,
    ):
        """
        **PARAMETERS**
        - `path` is the main folder holding the experiment folders
        - `update` searches the main folder for experiments to add
        - `project` is a separate database in the same main folder, at
        `.odyn/projects/<project>.db`. Leaving it out uses `main_sync`.

        **ALERT**
        Projects do not see each other. Two of them can hold the same
        experiment and neither will know, so use one when work should be kept
        apart, not to split work that has to be compared later.
        """
        # NOTE: _is_test is deliberately not documented above.
        #       See _copy_for_test for more details.

        # Resolved so that every path built from it is absolute and symlink
        # free. Stored paths are relative to this, and the code that makes
        # them relative resolves its side, so leaving this one as given makes
        # relative_to fail on a relative main folder or through a symlink.
        self.main_folder = Path(path).resolve()
        self.project: Final[None | str] = project

        odyn_folder = self.main_folder / ODYN_FOLDER

        if project is None:
            # `project=None` used to mean the shared '.odyn/odyn.db', which is
            # v2 and stays with the tagged release. Refusing it matters most
            # when that file does *not* exist yet: the old behavior would have
            # created one here, where everyone expects the shared database, and
            # nothing would notice until a workstation on the old release
            # failed to open it.
            raise ValueError(
                f"'project' cannot be None: the shared database is schema v2 "
                f"and this version writes v3. Leave it out to use "
                f"'{DEFAULT_PROJECT}', or name one:\n"
                f"    Database(main_folder, project='name')"
            )

        else:
            # Project name is the db file name so it needs to work everywhere
            # For simplicity, we allow only ASCII alphanumerics and underscores
            if not project or not all(
                letter.isascii() and (letter.isalnum() or letter == "_")
                for letter in project
            ):
                raise ValueError(
                    "'project' must be letters, digits and "
                    f"underscores, but instead got {project!r}."
                )

            live = odyn_folder / PROJECTS_FOLDER / f"{project}.db"

        self._is_test: Final[bool] = _is_test
        self.path: Final[Path] = self._copy_for_test(live) if _is_test else live

        # Database has a default group to record its calls
        self.group_id: Final[int] = 0

        # Initialize "private" variables
        self._call_stack: list[CallFrame] = []
        self._acquisitions: None | pd.DataFrame = None
        self._acquisition_trials: None | pd.DataFrame = None
        self._annotation_keys: None | pd.DataFrame = None
        self._annotations: None | pd.DataFrame = None
        self._session_panels: None | pd.DataFrame = None
        self._events: None | pd.DataFrame = None
        self._experiments: None | pd.DataFrame = None
        self._groups: dict[int, Group] = {}  # Caches groups one-by-one
        self._group_experiments: None | pd.DataFrame = None
        self._mcor_files: None | pd.DataFrame = None
        self._method_calls: None | pd.DataFrame = None
        self._mice: None | pd.DataFrame = None
        self._mouse_lines: None | pd.DataFrame = None
        self._odors: None | pd.DataFrame = None
        self._panels: None | pd.DataFrame = None
        self._panel_vials: None | pd.DataFrame = None
        self._session_vials: None | pd.DataFrame = None
        self._vial_components: None | pd.DataFrame = None
        self._outputs: None | pd.DataFrame = None
        self._programs: None | pd.DataFrame = None
        self._trials: None | pd.DataFrame = None

        # Get connection and create database if needed
        if not self.path.exists():
            logger.info("Did not find a database!")
            logger.info("Creating database...")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path, timeout=DB_TIMEOUT_S)

            # Create schema and add default values
            with self.con as con:
                with open(Path(__file__).parent / "create.sql") as f:
                    con.executescript(f.read())

                # Fresh DB is already at the latest schema
                con.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

                # Insert default database group
                query = "INSERT OR IGNORE INTO groups (group_id) VALUES (?);"
                con.execute(query, [self.group_id])

            logger.info(f"Database created at: '{self.path.resolve()}'")

        else:
            self.con = sqlite3.connect(self.path, timeout=DB_TIMEOUT_S)
            logger.info(f"Connected to the database at: '{self.path.resolve()}'")
            self._check_schema_version()

        self.con.execute("PRAGMA foreign_keys = ON;")
        self.con.row_factory = sqlite3.Row

        self._data_version = self.con.execute("PRAGMA data_version;").fetchone()[0]

        if update:
            self.update()

    def __del__(self):
        # `__init__` can raise before the connection exists (a bad `project`,
        # a schema mismatch), and __del__ still runs on the half-built object.
        # Without this the real error is followed by an AttributeError from
        # here, which is the one people read first.
        connection = getattr(self, "con", None)

        if connection is not None:
            connection.close()

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

    # ----------------------------------------------------------------------- #
    # SQLite Tables as DataFrames
    # ----------------------------------------------------------------------- #

    @property
    def acquisitions(self) -> pd.DataFrame:
        """`DataFrame` with acquisition metadata"""
        self._refresh_if_stale()

        if self._acquisitions is not None:
            return self._acquisitions

        # `acquisition_sync` joined back in; see `Group.acquisitions` for why.
        # LEFT, because those rows are missing until the sync file is decoded.
        query = f"""
            SELECT a.*
                 {_SYNC_COLUMNS}
                FROM acquisitions AS a
                LEFT JOIN acquisition_sync AS s ON s.acq_id = a.acq_id;
        """
        self._acquisitions = pd.read_sql_query(
            query,
            self.con,
            parse_dates=["acq_start", "sync_odor_start", "sync_odor_end"],
        )
        self._acquisitions.set_index("acq_id", inplace=True)

        return self._acquisitions

    @property
    def outputs_folder(self) -> Path:
        """
        Where functions of this database save what they produce

        The project's own `outputs` folder when the database belongs to one,
        and the main folder's otherwise. It is made when something is saved.
        """
        root = self.main_folder if self.project is None else self.project_folder

        return root / OUTPUTS_FOLDER

    @property
    def acquisition_trials(self) -> pd.DataFrame:
        """
        `DataFrame` with each acquisition and the associated trial data.

        Odor starts and ends are prepended with `acq` or `trial` depending
        on the table of origin. The former come from H5 timings and the
        later from olfactometer event files. Acquisitions that are not
        associated with a trial are kept, just with null entries.

        To compute `events` timedeltas use the trial (olfactometer) timings.
        """
        self._refresh_if_stale()

        if self._acquisition_trials is not None:
            return self._acquisition_trials

        self._acquisition_trials = _acquisition_trials(self.con, ACQUISITION_TRIALS)
        return self._acquisition_trials

    @property
    def events(self) -> pd.DataFrame:
        """`DataFrame` with olfactometer events"""
        self._refresh_if_stale()

        if self._events is not None:
            return self._events

        query = "SELECT * FROM events;"

        self._events = pd.read_sql_query(query, self.con, parse_dates=["event_time"])
        self._events.set_index("event_id", inplace=True)

        return self._events

    @property
    def experiments(self) -> pd.DataFrame:
        """`DataFrame` with experiment metadata"""
        self._refresh_if_stale()

        if self._experiments is not None:
            return self._experiments

        query = "SELECT * FROM experiments;"
        self._experiments = pd.read_sql_query(
            query, self.con, parse_dates=["exp_start", "added_to_db_at"]
        )
        self._experiments.set_index("exp_id", inplace=True)

        return self._experiments

    @property
    def groups(self) -> dict[int, Group]:
        """`Group`s (indexed by `group_id`) for processing/analysis."""
        self._refresh_if_stale()

        query = "SELECT group_id FROM groups WHERE group_id != ?;"
        rows = self.con.execute(query, [self.group_id]).fetchall()

        return {row["group_id"]: self._group(row["group_id"]) for row in rows}

    def _group(self, group_id: int) -> Group:
        """Return the cached `Group` for `group_id`, creating it if missing."""
        if group_id not in self._groups:
            self._groups[group_id] = Group(group_id, self)

        return self._groups[group_id]

    @property
    def group_experiments(self) -> pd.DataFrame:
        """`DataFrame` with both group and experiment data"""
        self._refresh_if_stale()

        if self._group_experiments is not None:
            return self._group_experiments

        query = """
            SELECT group_id, e.* FROM group_experiments AS ge
                JOIN experiments AS e ON e.exp_id = ge.exp_id
                WHERE group_id != ?;
        """
        self._group_experiments = pd.read_sql_query(
            query, self.con, params=[self.group_id]
        )
        self._group_experiments.set_index("group_id", inplace=True)

        return self._group_experiments

    @property
    def mcor_files(self) -> pd.DataFrame:
        """`DataFrame` with mcor files metadata"""
        self._refresh_if_stale()

        if self._mcor_files is not None:
            return self._mcor_files

        query = "SELECT * FROM mcor_files;"
        self._mcor_files = pd.read_sql_query(query, self.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._mcor_files

    @property
    def method_calls(self) -> pd.DataFrame:
        """`DataFrame` with `@record_call` functions"""
        self._refresh_if_stale()

        if self._method_calls is not None:
            return self._method_calls

        query = "SELECT * FROM method_calls;"

        self._method_calls = pd.read_sql_query(
            query, self.con, parse_dates=["called_at"]
        )
        self._method_calls.set_index("method_call_id", inplace=True)

        self._method_calls["parameter_inputs"] = self._method_calls[
            "parameter_inputs"
        ].apply(json.loads)
        self._method_calls["parameters_used"] = self._method_calls[
            "parameters_used"
        ].apply(json.loads)

        self._method_calls["call_output"] = self._method_calls["call_output"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else None
        )

        return self._method_calls

    @property
    def odors(self) -> pd.DataFrame:
        """`DataFrame` with current list of odors"""
        self._refresh_if_stale()

        if self._odors is not None:
            return self._odors

        query = "SELECT * FROM odors;"

        self._odors = pd.read_sql_query(query, self.con)
        self._odors.set_index("odor_id", inplace=True)

        return self._odors

    @property
    def mice(self) -> pd.DataFrame:
        """`DataFrame` with one row per animal. Its line is in `mouse_lines`"""
        self._refresh_if_stale()

        if self._mice is not None:
            return self._mice

        query = "SELECT * FROM mice;"

        self._mice = pd.read_sql_query(query, self.con, parse_dates=["mouse_dob"])
        self._mice.set_index("mouse_id", inplace=True)

        return self._mice

    @property
    def mouse_lines(self) -> pd.DataFrame:
        """`DataFrame` with one row per line a mouse carries, and its genotype"""
        self._refresh_if_stale()

        if self._mouse_lines is not None:
            return self._mouse_lines

        query = "SELECT * FROM mouse_lines ORDER BY mouse_id, line;"

        self._mouse_lines = pd.read_sql_query(query, self.con)
        self._mouse_lines.set_index(["mouse_id", "line"], inplace=True)

        return self._mouse_lines

    @property
    def panels(self) -> pd.DataFrame:
        """`DataFrame` with every odor panel this project knows"""
        self._refresh_if_stale()

        if self._panels is not None:
            return self._panels

        query = "SELECT * FROM panels;"

        self._panels = pd.read_sql_query(
            query, self.con, parse_dates=["added_to_db_at"]
        )
        self._panels.set_index("panel_id", inplace=True)

        return self._panels

    @property
    def panel_vials(self) -> pd.DataFrame:
        """
        `DataFrame` with what each vial of each panel delivers

        Indexed by panel *name* rather than id, since that is what a person
        reading it knows. `odor_name` is joined in for the same reason; the
        components pipetted into the vial are in `vial_components`.
        """
        self._refresh_if_stale()

        if self._panel_vials is not None:
            return self._panel_vials

        query = """
            SELECT p.panel_name, v.*, o.odor_name
                FROM panel_vials AS v
                JOIN panels AS p ON p.panel_id = v.panel_id
                JOIN odors  AS o ON o.odor_id  = v.odor_id
                ORDER BY p.panel_name, v.vial_position;
            """

        self._panel_vials = pd.read_sql_query(query, self.con)
        self._panel_vials.set_index(["panel_name", "vial_position"], inplace=True)

        return self._panel_vials

    @property
    def vial_components(self) -> pd.DataFrame:
        """`DataFrame` with what was pipetted into each vial of each panel"""
        self._refresh_if_stale()

        if self._vial_components is not None:
            return self._vial_components

        query = """
            SELECT p.panel_name, c.*, o.odor_name
                FROM vial_components AS c
                JOIN panels AS p ON p.panel_id = c.panel_id
                JOIN odors  AS o ON o.odor_id  = c.odor_id
                ORDER BY p.panel_name, c.vial_position, c.odor_id;
            """

        self._vial_components = pd.read_sql_query(query, self.con)
        self._vial_components.set_index(
            ["panel_name", "vial_position", "odor_id"], inplace=True
        )

        return self._vial_components

    @property
    def session_vials(self) -> pd.DataFrame:
        """
        `DataFrame` with every vial a session ran, and the day it was mixed

        One row per vial. `made_on` is NULL where the mixing was recorded as
        unknown; a vial the session did not run has no row at all.
        """
        self._refresh_if_stale()

        if self._session_vials is not None:
            return self._session_vials

        query = """
            SELECT v.*, p.panel_name, pv.odor_id, o.odor_name
                FROM session_vials AS v
                JOIN panels AS p ON p.panel_id = v.panel_id
                JOIN panel_vials AS pv
                    ON pv.panel_id = v.panel_id
                   AND pv.vial_position = v.vial_position
                JOIN odors AS o ON o.odor_id = pv.odor_id
                ORDER BY v.session_id, v.vial_position;
            """

        self._session_vials = pd.read_sql_query(
            query, self.con, parse_dates=["made_on"]
        )
        self._session_vials.set_index(["session_id", "vial_position"], inplace=True)

        return self._session_vials

    @property
    def session_panels(self) -> pd.DataFrame:
        """
        `DataFrame` with which panel each session ran

        When each of its vials was mixed is in `session_vials`.
        """
        self._refresh_if_stale()

        if self._session_panels is not None:
            return self._session_panels

        query = """
            SELECT s.*, p.panel_name
                FROM session_panels AS s
                JOIN panels AS p ON p.panel_id = s.panel_id;
            """

        self._session_panels = pd.read_sql_query(query, self.con)
        self._session_panels.set_index("session_id", inplace=True)

        return self._session_panels

    @property
    def annotation_keys(self) -> pd.DataFrame:
        """`DataFrame` with every annotation this project can record"""
        self._refresh_if_stale()

        if self._annotation_keys is not None:
            return self._annotation_keys

        query = "SELECT * FROM annotation_keys;"

        self._annotation_keys = pd.read_sql_query(query, self.con)
        self._annotation_keys.set_index(["applies_to", "key"], inplace=True)

        return self._annotation_keys

    @property
    def annotations(self) -> pd.DataFrame:
        """
        `DataFrame` with every annotation ever written, newest last

        Annotations are append-only, so a key that was corrected appears more
        than once here. Use `annotations_for` to get the current value of each.
        """
        self._refresh_if_stale()

        if self._annotations is not None:
            return self._annotations

        query = "SELECT * FROM annotations;"

        self._annotations = pd.read_sql_query(query, self.con)
        self._annotations.set_index("annotation_id", inplace=True)

        return self._annotations

    def annotations_for(self, target_type: str) -> pd.DataFrame:
        """
        One row per annotated `session`/`experiment`/`program`/... , one column per key.

        **EXAMPLE**
        ```python
        deep = db.annotations_for("experiment").query("fov_depth_um > 200")
        db.experiments.join(deep, how="inner")
        ```

        Only keys that hold a single value appear, so every column is a plain
        number or string and the table behaves like any other. Keys that hold a
        list (notes, flags) are in `db.annotations` instead.
        """
        keys = self.annotation_keys

        if target_type not in keys.index.get_level_values("applies_to"):
            raise ValueError(
                f"Nothing can be annotated on a {target_type!r}. Use one of: "
                f"{sorted(set(keys.index.get_level_values('applies_to')))}."
            )

        scalar = keys.xs(target_type, level="applies_to")
        scalar = scalar[~scalar["multi_valued"].astype(bool)]

        rows = self.annotations
        rows = rows[
            (rows["target_type"] == target_type) & rows["key"].isin(scalar.index)
        ]

        # Append-only, so the last row written for a key is its current value.
        # `annotation_id` rises with time and is the index, hence sorting on it.
        current = rows.sort_index().drop_duplicates(
            subset=["target_id", "key"], keep="last"
        )

        frame = current.pivot(index="target_id", columns="key", values="value")
        frame.columns.name = None
        frame.index.name = f"{target_type}_id"

        # `value` is an ANY column, so a column arrives as object dtype even when
        # every entry in it is a number. The registry says which is which.
        for key, declared in scalar["value_type"].items():
            if key in frame.columns and declared in ("integer", "real", "boolean"):
                frame[key] = pd.to_numeric(frame[key], errors="coerce")

        return frame

    @property
    def outputs(self) -> pd.DataFrame:
        """`DataFrame` with output files of functions"""
        self._refresh_if_stale()

        if self._outputs is not None:
            return self._outputs

        query = "SELECT * FROM outputs;"

        self._outputs = pd.read_sql_query(query, self.con)
        self._outputs.set_index("output_id", inplace=True)

        return self._outputs

    @property
    def programs(self) -> pd.DataFrame:
        """`DataFrame` with one entry per _Event.csv_ file"""
        self._refresh_if_stale()

        if self._programs is not None:
            return self._programs

        query = "SELECT * FROM programs;"

        self._programs = pd.read_sql_query(
            query, self.con, parse_dates=["program_start"]
        )
        self._programs.set_index("program_id", inplace=True)

        return self._programs

    @property
    def trials(self) -> pd.DataFrame:
        """`DataFrame` with all olfactometer trials"""
        self._refresh_if_stale()

        if self._trials is not None:
            return self._trials

        query = "SELECT * FROM trials;"

        self._trials = pd.read_sql_query(
            query,
            self.con,
            parse_dates=["trial_start", "trial_odor_start", "trial_odor_end"],
        )
        self._trials.set_index("trial_id", inplace=True)

        return self._trials

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _check_schema_version(self) -> None:
        """
        Throws error if DB schema version is not what the code expects.
        """

        version = self.con.execute("PRAGMA user_version;").fetchone()[0]
        if version == SCHEMA_VERSION:
            return

        if version < SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema is v{version} but the code expects "
                f"v{SCHEMA_VERSION}. Run the migration:\n"
                f"    python -m odyn.migrate '{self.main_folder}'"
            )

        raise RuntimeError(
            f"Database schema is v{version} but the code expects v{SCHEMA_VERSION}. "
            "Your code is out of date! Pull the latest version!"
        )

    def _copy_for_test(self, source: Path) -> Path:
        """
        Returns path to a fresh snapshot of the database. For tests only,
        via `Database(main_folder, project="name", _is_test=True)`.

        PROTECTS DATABASE, BUT ACCESS REAL DATA.

        `source` is whichever database was asked for, so testing against a
        project copies that project rather than the shared one.
        """
        if not source.exists():
            raise FileNotFoundError(f"No database at '{source}' to copy.")

        # Named after the source, so two projects cannot overwrite each
        # other's test copy
        copy = self.main_folder / ODYN_FOLDER / "tests" / source.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.unlink(missing_ok=True)

        # Use the online backup API rather than a file copy.
        #   (In case the DB is in use.)
        origin = sqlite3.connect(source, timeout=DB_TIMEOUT_S)
        destination = sqlite3.connect(copy)

        try:
            origin.backup(destination)
        finally:
            destination.close()
            origin.close()

        logger.warning(f"TEST COPY: '{copy.resolve()}'")
        logger.warning("The shared database will not see anything you do here.")

        return copy

    def _get_raw_metadata(self, path: Path) -> None | tuple[Object, Object]:
        tif = TiffFile(path)

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

        # Get the data that must be the same across the experiment
        file_stem_parts = path.stem.split("_")

        experiment: Object = {
            "exp_name": "_".join(file_stem_parts[:-1]),
            "exp_type": SI_metadata["SI.acqState"],
            "mouse_id": file_stem_parts[1],
            "height_px": tif.pages[0].tags["ImageLength"].value,
            "width_px": tif.pages[0].tags["ImageWidth"].value,
            "frame_count": SI_metadata["SI.hStackManager.framesPerSlice"],
            "frame_rate": SI_metadata["SI.hRoiManager.scanFrameRate"],
            "laser_power_920": laser_powers[0],
            "laser_power_1040": laser_powers[1],
            "loop_acq_interval_s": SI_metadata["SI.loopAcqInterval"],
        }

        # Assume unit is centimeters
        # TODO: Check that units are centimeters
        dx, nx = tif.pages[0].tags["XResolution"].value
        dy, ny = tif.pages[0].tags["YResolution"].value

        # um per pixels in each direction
        factor_x = round(1e4 * nx / dx, 4)
        factor_y = round(1e4 * ny / dy, 4)

        # Size of image in um
        experiment["width_um"] = experiment["width_px"] * factor_x
        experiment["height_um"] = experiment["height_px"] * factor_y

        # Parse ImageDescription
        image_description = dict(
            line.split(" = ")
            for line in tif.pages[0].tags["ImageDescription"].value.splitlines()
        )

        # Parse ImageDescription epoch as a datetime
        date_string = image_description["epoch"].strip("[]")
        date_string = " ".join(date_string.split())
        loop_start = datetime.strptime(date_string, "%Y %m %d %H %M %S.%f")

        experiment["exp_start"] = loop_start

        # Data specific to the acquisition
        delta_sec = float(image_description["frameTimestamps_sec"])
        acquisition_time = loop_start + timedelta(seconds=delta_sec)

        acquisition: Object = {
            "raw_path": path.relative_to(self.main_folder).as_posix(),
            "acq_start": acquisition_time,
        }

        return experiment, acquisition

    def _refresh_if_stale(self) -> None:
        """Reset caches if another connection has committed since the last check."""
        version = self.con.execute("PRAGMA data_version;").fetchone()[0]

        if version != self._data_version:
            self._data_version = version
            self._reset_caches()

    def _reset_caches(self) -> None:
        self._acquisitions = None
        self._acquisition_trials = None
        self._annotation_keys = None
        self._annotations = None
        self._session_panels = None
        self._events = None
        self._experiments = None
        self._group_experiments = None
        self._mcor_files = None
        self._method_calls = None
        self._mice = None
        self._mouse_lines = None
        self._odors = None
        self._outputs = None
        self._panels = None
        self._panel_vials = None
        self._session_vials = None
        self._programs = None
        self._trials = None
        self._vial_components = None

        # In case a specific group can still be accessed
        for group in self._groups.values():
            group._acquisitions = None
            group._acquisition_trials = None
            group._events = None
            group._experiments = None
            group._mcor_files = None
            group._method_calls = None
            group._outputs = None
            group._programs = None
            group._trials = None

        self._groups.clear()

    # ----------------------------------------------------------------------- #
    # Database Queries
    # ----------------------------------------------------------------------- #

    def add_group(self, exp_ids=None, exp_names=None) -> Group:
        """
        Add a group with the experiments listed and return it.

        **USAGE**
        ```python
        db = Database(main_folder)
        group = db.add_group(list of exp_ids or exp_names)
        ```

        **EXAMPLES**
        ```python
        group = db.add_group(
                    exp_names=[
                        "20250303_sid172_e1",
                        "20250303_sid172_e2"
                    ]
                )

        group = db.add_group(exp_ids=[3,10,12])
        """

        if (exp_ids is None) == (exp_names is None):
            raise ValueError("Provide exactly one of exp_ids or exp_names.")

        if exp_names is not None:
            missing = set(exp_names) - set(self.experiments["exp_name"])
            if missing:
                raise ValueError(f"No experiments named: {sorted(missing)}")

            matched = self.experiments[self.experiments["exp_name"].isin(exp_names)]
            target = frozenset(int(e) for e in matched.index)

        else:
            missing = set(exp_ids) - set(self.experiments.index)
            if missing:
                raise ValueError(f"No experiments with ids: {sorted(missing)}")

            target = frozenset(int(e) for e in exp_ids)

        # Check if there is a group with exactly those experiments
        members: dict[int, set[int]] = {}
        for group_id, exp_id in self.group_experiments["exp_id"].items():
            members.setdefault(group_id, set()).add(int(exp_id))

        for group_id, member_ids in members.items():
            if member_ids == set(target):
                logger.info(
                    f"Group already exists (group_id = {group_id}), returning it."
                )
                return self._group(group_id)

        with self.con as con:
            cur = con.cursor()

            cur.execute("INSERT INTO groups DEFAULT VALUES;")
            group_id = cur.lastrowid

            cur.executemany(
                "INSERT INTO group_experiments (group_id, exp_id) VALUES (?, ?);",
                [(group_id, exp_id) for exp_id in target],
            )

        # Reset group_experiments cache
        self._group_experiments = None

        return self._group(group_id)

    def from_query(self, query: str) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from a SQL query.
        Use db.run_query() for inserts/updates.

        **USAGE**
        ```python
        db = Database(main_folder)
        df = db.from_query(query_as_a_string)
        ```

        **EXAMPLE**
        ```python
        db.from_query("SELECT exp_id, exp_name FROM experiments;")
        ```
        """
        return pd.read_sql_query(query, self.con)

    def latest_calls(self, method_name: str) -> pd.DataFrame:
        """Return DataFrame with all calls to 'method_name'."""

        query = """
            SELECT * FROM method_calls
                WHERE method_name LIKE ?
                ORDER BY method_call_id DESC
            """
        return _method_calls_dataframe(self.con, query, [f"%{method_name}"])

    def latest_output(self, method_name: str) -> None | Object:
        """Return output of the most recent call to 'method_name'."""

        row = self.con.execute(
            """
            SELECT method_call_id, call_output FROM method_calls
                WHERE group_id = ? AND method_name = ? AND call_output IS NOT NULL
                ORDER BY method_call_id DESC LIMIT 1
            """,
            [self.group_id, method_name],
        ).fetchone()

        if row is None:
            return None

        # Link the call that produced this value to the call reading it, so the
        # tree of what fed what can be walked later.
        self.note_consumed(row["method_call_id"])

        return json.loads(row["call_output"])

    # ----------------------------------------------------------------------- #
    # Custom SQL INSERT/UPDATE
    # ----------------------------------------------------------------------- #

    def commit_changes(self):
        self.con.commit()

    def rollback_changes(self):
        self._reset_caches()
        self.con.rollback()

    def run_query(self, query: str) -> Cursor:
        """
        Run SQL query (be careful!).

        **USAGE**
        ```python
        db = Database(main_folder)
        db.run_query(query_as_a_string)
        ```

        **EXAMPLE**
        ```python
        db.run_query(\"\"\"
            UPDATE experiments
                SET exp_name = "test"
                WHERE exp_id = 10;
        \"\"\")
        ````
        """

        cur = self.con.execute(query)
        self._reset_caches()

        return cur

    # ----------------------------------------------------------------------- #
    # Updating the Database
    # ----------------------------------------------------------------------- #

    @record_call
    def add_grab_folder(
        self,
        *,
        rel_path: str,
        rel_raw_paths: None | list[str] = None,
    ) -> None:
        """
        Add a folder of grabs, each as its own experiment

        **PARAMETERS**
        - `rel_path` is the experiment folder path relative to the `main_folder`
        - `rel_raw_paths` is the raw files list (if None it searches the raw folder)

        **EXAMPLE**
        ```python
        db.add_grab_folder(rel_path="20260623/m462/e2")

        Files already in the database are skipped, so running this again after
        adding more grabs to the folder only adds the new ones.
        ```
        """
        logger.info("Adding grabs to database...")

        exp_path = self.main_folder / rel_path

        assert exp_path.is_dir(), f"Folder not found: '{exp_path.resolve()}'"

        raw_paths = (
            sorted(exp_path.glob("raw/[!.]?*.tif"))
            if rel_raw_paths is None
            else [self.main_folder / p for p in rel_raw_paths]
        )

        assert raw_paths, "Did not find any raw/*.tif files."

        logger.info(f"Processing folder: '{exp_path.resolve()}'")

        added = []

        for raw_path in tqdm(raw_paths, desc="Loading TIFF Metadata"):
            # Read outside the transaction (to not write lock).
            raw_metadata = self._get_raw_metadata(raw_path)

            if raw_metadata is None:
                logger.info(
                    f"  Skipped file {raw_path} (metadata format not supported)"
                )
                self.add_flag(ExpFlag.UNSUPPORTED_METADATA)
                continue

            experiment, acquisition = raw_metadata

            if experiment["exp_type"] != "grab":
                logger.warning(f"  Skipped '{raw_path.name}' (not a grab).")
                self.add_flag(ExpFlag.NOT_A_GRAB)
                continue

            # We keep the trailing number to differentiate grabs
            #   (differently from the add_experiment)
            experiment["exp_name"] = raw_path.stem

            # Type checking because Object is too generic
            assert isinstance(experiment["exp_start"], datetime)
            exp_start_str = experiment["exp_start"].strftime(DT_FORMAT)

            # One transaction per grab/experiment as in add_experiment
            # User can rerun it fails in a couple files (or just skip them)
            with self.con as con:
                cur = con.cursor()
                cur.execute("PRAGMA foreign_keys = ON;")

                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM experiments WHERE exp_start = ?);",
                    [exp_start_str],
                )

                if cur.fetchone()[0]:
                    logger.info(f"  '{raw_path.name}' already in DB.")
                    self.add_flag(ExpFlag.ALREADY_IN_DB)
                    continue

                exp_id = _insert_experiment(
                    cur,
                    experiment,
                    rel_path=rel_path,
                    method_call_id=self.current_call_id,
                )

                cur.execute("INSERT INTO groups DEFAULT VALUES;")
                group_id = cur.lastrowid

                cur.execute(
                    "INSERT INTO group_experiments (group_id, exp_id) VALUES (?, ?);",
                    [group_id, exp_id],
                )

                acq_start = acquisition["acq_start"]
                assert isinstance(acq_start, datetime)

                _db_insert(
                    cur,
                    "acquisitions",
                    {
                        **acquisition,
                        "exp_id": exp_id,
                        "acq_start": acq_start.strftime(DT_FORMAT),
                    },
                )

            added.append(raw_path.stem)

        self.set_output({"added": added})
        self._reset_caches()

        logger.info(f"Added {len(added)} of {len(raw_paths)} grabs. {CHECK}")

    @record_call
    def add_experiment(
        self,
        *,
        rel_path: str,
        rel_raw_paths: None | list[str] = None,
    ) -> None:
        """
        Add a new experiment folder to the database

        **PARAMETERS**
        - `rel_path` is the experiment folder path relative to the `main_folder`
        - `rel_raw_paths` is the raw files list (if None it searches the raw folder)

        **EXAMPLE**
        ```python
        db.add_experiment(rel_path="20260623/m462/e2")
        ```
        """

        logger.info("Adding experiment to database...")

        # Basically, everything in this function is done in a single transaction
        # because if something fails we rollback all insertions.
        exp_path = self.main_folder / rel_path

        assert exp_path.is_dir(), f"Folder not found: '{exp_path.resolve()}'"

        # Fetch raw file paths list if not provided
        raw_paths = (
            sorted(exp_path.glob("raw/[!.]?*.tif"))
            if rel_raw_paths is None
            else [self.main_folder / p for p in rel_raw_paths]
        )

        logger.info(f"Processing folder: '{exp_path.resolve()}'")

        with self.con as con:
            cur = con.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")

            # --------------------------------------------------------------- #
            # Phase 1: Load
            # --------------------------------------------------------------- #

            experiment: None | Object = None
            acquisitions: list[Object] = []
            event_files: list[Path] = []

            last_exp_data: None | Object = None
            checks_failed = 0

            assert raw_paths, "Did not find any raw/*.tif files."

            for raw_path in tqdm(raw_paths, desc="Loading TIFF Metadata"):
                raw_metadata = self._get_raw_metadata(raw_path)

                if raw_metadata is None:
                    logger.info(
                        f"  Skipped file {raw_path} (metadata format not supported)"
                    )
                    self.add_flag(ExpFlag.UNSUPPORTED_METADATA)
                    continue

                exp_data, acq = raw_metadata

                if last_exp_data is None:
                    # Don't do anything if experiment is already in the DB.
                    #
                    # Formatted rather than passed as a datetime: the column
                    # holds DT_FORMAT strings, and sqlite3's datetime adapter
                    # writes isoformat, which drops '.000000' when the epoch
                    # lands exactly on a second. The two then never compare
                    # equal, this check says "not present", and the insert
                    # fails on UNIQUE instead of returning quietly. (That
                    # adapter is also deprecated since Python 3.12.)
                    assert isinstance(exp_data["exp_start"], datetime)

                    cur.execute(
                        """
                        SELECT EXISTS(
                            SELECT 1 FROM experiments
                                WHERE exp_start = ?
                        );
                    """,
                        [exp_data["exp_start"].strftime(DT_FORMAT)],
                    )

                    if cur.fetchone()[0]:
                        logger.info("Experiment already in DB.")
                        self.add_flag(ExpFlag.ALREADY_IN_DB)
                        return

                    experiment = exp_data

                    # Load event files before metadata checks
                    # (Checks take some time so better to not do them if possible)
                    event_files = sorted(
                        exp_path.rglob("[!.]?*Events.csv"),
                        key=lambda x: x.stat().st_mtime,
                    )

                    logger.info(f"Found {len(event_files)} olfactometer event files.")

                elif last_exp_data != exp_data:
                    checks_failed += 1

                    logger.warning(
                        f"'{raw_path.relative_to(self.main_folder)}' metadata"
                        " is inconsistent with the previous acquisition."
                    )

                last_exp_data = exp_data
                acquisitions.append(acq)

            if checks_failed > 0:
                logger.error(
                    f"TIFF metadata changed {checks_failed} or more times in the raw folder. {CROSS}"
                )
                logger.info("Are there multiple loops or grabs in the same folder?")
                logger.error("Experiment will not be added to the DB.")
                self.add_flag(ExpFlag.METADATA_CHANGED)
                return

            logger.info(f"Passed all TIFF metadata checks! {CHECK}")

            assert (
                experiment is not None
            ), "Could not find any TIFF file with the expected metadata format."

            # Parse all event files into structured program/trial dicts
            programs_data: list[dict] = []

            if event_files:
                res = cur.execute("SELECT odor_name, odor_id FROM odors;")
                odors: dict[str, int] = {name: id for name, id in res.fetchall()}

                stem_split = event_files[0].stem.split("-")
                events_start = datetime.strptime(
                    " ".join(stem_split[-3:-1]), "%Y_%m_%d %H_%M_%S"
                )
                program_starts = _parse_program_starts(self, events_start)

                programs_data = _load_event_data(
                    event_files, program_starts, odors, self.main_folder
                )

            # --------------------------------------------------------------- #
            # Phase 2: Insert
            # --------------------------------------------------------------- #
            #
            # There is no matching phase here any more. Odor timing and the link
            # from a trial to its acquisition both come from the sync file, and
            # the decode that reads it is its own recorded call so it can be run
            # again on its own. Once that call exists this method runs it at the
            # end, warning and skipping when '<exp>/sync/' holds no H5.

            exp_id = _insert_experiment(
                cur,
                experiment,
                rel_path=rel_path,
                method_call_id=self.current_call_id,
            )

            cur.execute("INSERT INTO groups DEFAULT VALUES;")
            group_id = cur.lastrowid

            cur.execute(
                "INSERT INTO group_experiments (group_id, exp_id) VALUES (?, ?);",
                [group_id, exp_id],
            )

            for acq in acquisitions:
                # Formatted here for the same reason as `exp_start` above: the
                # column holds DT_FORMAT strings, and letting sqlite3's
                # (deprecated) adapter write the datetime drops the microseconds
                # whenever they are zero.
                acq_start = acq["acq_start"]
                assert isinstance(acq_start, datetime)

                _db_insert(
                    cur,
                    "acquisitions",
                    {
                        **acq,
                        "exp_id": exp_id,
                        "acq_start": acq_start.strftime(DT_FORMAT),
                    },
                )

            # Insert programs, trials, and events
            for program_idx, program_data in enumerate(programs_data):
                metadata = program_data["metadata"]
                assert isinstance(metadata["program_start"], datetime)

                program_id = _db_insert(
                    cur,
                    "programs",
                    {
                        **metadata,
                        "exp_id": exp_id,
                        # Formatted, as everywhere else: see `acq_start` above.
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
                            # Both come from the sync decode (see Phase 2).
                            "acq_id": None,
                            "sync_to_trial_ms": None,
                            "program_id": program_id,
                            "exp_id": exp_id,
                        },
                    )

                # Pass 2: insert all events in order
                for trial_idx, event in program_data["events"]:
                    _db_insert(
                        cur,
                        "events",
                        {
                            **event,
                            "program_id": program_id,
                            "trial_id": trial_ids.get(trial_idx),
                        },
                    )

            # --------------------------------------------------------------- #
            # Reporting
            # --------------------------------------------------------------- #

            n_trials = sum(len(p["trials"]) for p in programs_data)

            logger.info(
                f"Added {len(acquisitions)} acquisitions and {n_trials} trials "
                f"over {len(programs_data)} programs. {CHECK}"
            )
            # TODO: run the sync decode here once it exists, warning and
            #       skipping if '<exp>/sync/' holds no H5.
            logger.warning(
                "Odor timing and the trial-to-acquisition link are empty: the "
                "sync file is not read yet on this branch."
            )

            self._reset_caches()

    @record_call
    def add_annotation(
        self,
        *,
        target_type: str,
        target_id: int,
        key: str,
        value: Value,
    ) -> None:
        """
        Record something about an experiment, session, group, etc.

        **PARAMETERS**
        - `target_type` is what this annotation is about. Options are
        `'session'`, `'experiment'`, `'program'`, `'acquisition'` or `'group'`.
        - `target_id` is that row's id (on the target_type table)
        - `key` must already be in `db.annotation_keys`
        - `value` has to match the type the key was registered with

        **EXAMPLE**
        ```python
        db.add_annotation(
            target_type="experiment",
            target_id=12,
            key="fov_depth_um",
            value=70,
        )
        ```

        **ALERT**
        odyn's processing does not uses annotations. Those are for analysis
        only. Anything the pipeline has to use must be a column instead.

        Annotations are never overwritten. Writing a key twice keeps both, and
        the later one is used (so the whole history stays visible).
        """

        with self.con as con:
            cur = con.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")

            _db_annotate(
                cur,
                target_type=target_type,
                target_id=target_id,
                key=key,
                value=value,
                method_call_id=self.current_call_id,
            )

        logger.info(f"Annotated {target_type} {target_id}: {key} = {value!r}. {CHECK}")

        self._reset_caches()

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
        required: bool = False,
        multi_valued: bool = False,
    ) -> None:
        """
        Register something new that can be annotated in this project.

        **PARAMETERS**
        - `applies_to` is `'session'`, `'experiment'`, `'program'`,
        `'acquisition'` or `'group'`
        - `key` is what `add_annotation` will be called with
        - `value_type` is `'text'`, `'integer'`, `'real'`, `'boolean'`,
        `'date'`, or `'enum'`; an `'enum'` needs `allowed_values`
        - `required` means the data is not finished until this is filled in
        - `multi_valued` keeps every value written instead of only the latest,
        which is what notes and flags want

        **EXAMPLE**
        ```python
        db.add_annotation_key(
            applies_to="experiment",
            key="uses_odor_batch",
            label="Odor batch used",
            value_type="text",
            description="Which batch of odor was used.",
        )
        ```

        Keys are never deleted, because old values would stop making sense. If
        you want to stop using a particular key, use `retire_annotation_key`.
        """
        with self.con as con:
            cur = con.cursor()

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
                    "required": bool(required),
                    "multi_valued": bool(multi_valued),
                    "retired": False,
                },
            )

        logger.info(f"Registered annotation '{key}' for a {applies_to}. {CHECK}")

        self._reset_caches()

    @record_call
    def set_mouse(
        self,
        *,
        mouse_id: int | str,
        sex: None | str = None,
        dob: None | str = None,
        lines: None | dict[str, None | str] = None,
        stax_injection: None | str = None,
        sensor: None | str = None,
    ) -> int:
        """
        Record what is known about an animal, creating its row if it is new.

        **USAGE**
        ```python
        db.set_mouse(
            mouse_id="m442",
            sex="F",
            dob="2026-01-22",
            lines={"DAT-Cre": "het", "TIGRE": "het"},
            sensor="GCaMP8s",
        )
        ```

        **PARAMETERS**
        - `mouse_id` is the number, or a name like `'m442'` to take it from
        - `sex` is `'M'` or `'F'`, in either case
        - `dob` is the date of birth, `YYYY-MM-DD`
        - `lines` maps each mutation the animal carries to `'wt'`, `'het'`,
        `'hom'`, or `None` when the line is known but the genotyping is not
        - `stax_injection` and `sensor` are what was injected and what it expresses

        Returns the `mouse_id`. Anything left out keeps the value already
        stored, so this can be called repeatedly as details arrive. Passing
        `lines={}` clears the line; `lines=None` leaves it alone.
        """
        number = _mouse_number(str(mouse_id))
        stored_sex = None if sex is None else str(sex).strip().upper()

        if stored_sex not in (None, "M", "F"):
            raise ValueError(f"Sex must be 'M' or 'F', not {sex!r}.")

        born = dob.date().isoformat() if isinstance(dob, datetime) else dob
        carried = None if lines is None else _mouse_lines(lines)

        with self.con as con:
            cur = con.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")

            # COALESCE so that a field left out of this call does not erase what
            # an earlier one wrote: the details arrive from different people.
            cur.execute(
                """
                INSERT INTO mice
                    (mouse_id, mouse_sex, mouse_dob, stax_injection, sensor)
                    VALUES (:mouse_id, :sex, :dob, :stax, :sensor)
                ON CONFLICT(mouse_id) DO UPDATE SET
                      mouse_sex      = COALESCE(:sex, mouse_sex)
                    , mouse_dob      = COALESCE(:dob, mouse_dob)
                    , stax_injection = COALESCE(:stax, stax_injection)
                    , sensor         = COALESCE(:sensor, sensor);
                """,
                {
                    "mouse_id": number,
                    "sex": stored_sex,
                    "dob": None if born is None else str(born),
                    "stax": stax_injection,
                    "sensor": sensor,
                },
            )

            if carried is not None:
                # Replaced as a whole: a mouse carries one set of mutations, and
                # a correction usually rewrites more than one of them.
                cur.execute(
                    "DELETE FROM mouse_lines WHERE mouse_id = ?;", [number]
                )

                for line, genotype in carried.items():
                    _db_insert(
                        cur,
                        "mouse_lines",
                        {"mouse_id": number, "line": line, "genotype": genotype},
                    )

        cross = "unknown line" if not carried else " x ".join(carried)

        logger.info(f"Mouse {number} recorded ({cross}). {CHECK}")

        self._reset_caches()

        return number

    @record_call
    def add_panel(
        self,
        *,
        panel_name: str,
        vials: list[Object],
        description: None | str = None,
    ) -> int:
        """
        Register an odor panel: what sits in each vial, and how it was made up.

        **USAGE**
        ```python
        db.add_panel(
            panel_name="print_v3",
            vials=[
                {"vial_position": 1, "odor_id": 1, "odor_sccm": 200,
                 "components": [{"odor_id": 1, "target_ppm": 0.3}]},
                {"vial_position": 3, "odor_id": 0},
            ],
        )
        ```

        **PARAMETERS**
        - `panel_name` names the recipe, and is what `set_session_panel` refers to
        - `vials` is one dict per vial. `vial_position` and `odor_id` are
        required; `odor_sccm`, `total_sccm`, `total_volume_ml` and
        `solvent_volume_ml` are optional, as is `components`
        - each entry of `components` needs an `odor_id` and may carry
        `target_ppm`, `liquid_ul` and `percent_vv`
        - `description` is a free-text note about the panel

        Returns the `panel_id`. Every `odor_id` must already be in `odors`.
        Calling this again for the same name replaces that panel's vials, so a
        corrected recipe is one call -- until a session has run it, after which
        the recipe is what that session's trials mean and cannot be rewritten.
        Re-registering an unchanged panel is a no-op either way, so an importer
        can be run again safely. Raises `ValueError` on a changed recipe that a
        session already used; register it under a new name instead.
        """
        wanted = _panel_rows(vials)

        with self.con as con:
            cur = con.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")

            # RETURNING rather than lastrowid: on the update branch no row is
            # inserted, and the existing id is the one the vials must carry.
            panel_id = cur.execute(
                """
                INSERT INTO panels (panel_name, description)
                    VALUES (:name, :description)
                ON CONFLICT(panel_name) DO UPDATE SET
                    description = COALESCE(:description, description)
                RETURNING panel_id;
                """,
                {"name": str(panel_name), "description": description},
            ).fetchone()[0]

            if _stored_panel(cur, panel_id) == wanted:
                logger.info(f"Panel {panel_name} is already as given. {CHECK}")

                return panel_id

            sessions = [
                row[0]
                for row in cur.execute(
                    "SELECT DISTINCT session_id FROM session_vials"
                    " WHERE panel_id = ? ORDER BY session_id;",
                    [panel_id],
                )
            ]

            # A session's trials mean whatever was in its vials at the time, so
            # a used recipe is history. The foreign key from `session_vials`
            # would refuse the rewrite anyway; this says why.
            if sessions:
                raise ValueError(
                    f"Panel {panel_name!r} was already run by sessions "
                    f"{sessions}, so its recipe cannot be changed. Register the "
                    f"new one under a different name."
                )

            # Replaced rather than merged, so a recipe cannot end up half old
            # and half new. `vial_components` cascades from the vials.
            cur.execute("DELETE FROM panel_vials WHERE panel_id = ?;", [panel_id])

            for position, odor_id, scalars, components in wanted:
                _db_insert(cur, "panel_vials", {
                    "panel_id": panel_id,
                    "vial_position": position,
                    "odor_id": odor_id,
                    **dict(zip(VIAL_COLUMNS, scalars)),
                })

                for component_odor, values in components:
                    _db_insert(cur, "vial_components", {
                        "panel_id": panel_id,
                        "vial_position": position,
                        "odor_id": component_odor,
                        **dict(zip(COMPONENT_COLUMNS, values)),
                    })

        logger.info(f"Panel {panel_name} has {len(vials)} vials. {CHECK}")

        self._reset_caches()

        return panel_id

    @record_call
    def set_session_panel(
        self,
        *,
        session_id: int,
        panel_name: str,
        made_on: None | str = None,
        vial_dates: None | dict[int, None | str] = None,
    ) -> None:
        """
        Record which set of vials a session ran, and when each was mixed.

        **USAGE**
        ```python
        # A whole rack remade at once, which is the usual case.
        db.set_session_panel(
            session_id=3, panel_name="print_v3", made_on="2026-07-06"
        )

        # One vial refilled later than the rest.
        db.set_session_panel(
            session_id=3,
            panel_name="print_v3",
            made_on="2026-07-06",
            vial_dates={4: "2026-07-21"},
        )
        ```

        **PARAMETERS**
        - `session_id` is the session that ran this panel
        - `panel_name` names a panel registered with `add_panel`
        - `made_on` is the day the vials were mixed, applied to every vial the
        panel defines
        - `vial_dates` maps a vial position to its own date, for the vials that
        were not mixed with the rest. It overrides `made_on` for those
        positions, and may be given without it

        A session runs one panel, so calling this again replaces what it had,
        dates included. Positions in `vial_dates` must exist in the panel.
        """
        common = _optional_date(made_on)
        per_vial = {
            int(position): _optional_date(date)
            for position, date in (vial_dates or {}).items()
        }

        with self.con as con:
            cur = con.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")

            exists = cur.execute(
                "SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id = ?);",
                [int(session_id)],
            ).fetchone()[0]

            if not exists:
                raise ValueError(f"There is no session {session_id}.")

            panel = cur.execute(
                "SELECT panel_id FROM panels WHERE panel_name = ?;",
                [str(panel_name)],
            ).fetchone()

            if panel is None:
                raise ValueError(
                    f"There is no panel {panel_name!r}. Register it with "
                    f"`add_panel` before a session can name it."
                )

            panel_id = panel[0]
            positions = [
                row[0]
                for row in cur.execute(
                    "SELECT vial_position FROM panel_vials WHERE panel_id = ?"
                    " ORDER BY vial_position;",
                    [panel_id],
                )
            ]

            unknown = sorted(set(per_vial) - set(positions))

            if unknown:
                raise ValueError(
                    f"Panel {panel_name!r} has no vial {unknown}. It runs "
                    f"positions {positions[0]}-{positions[-1]}."
                )

            # Replaced whole, so a re-entered panel cannot keep a date from the
            # one it replaced. `session_vials` cascades from this row.
            cur.execute(
                "DELETE FROM session_panels WHERE session_id = ?;", [int(session_id)]
            )

            _db_insert(cur, "session_panels", {
                "session_id": int(session_id),
                "panel_id": panel_id,
            })

            # Every vial of the panel gets a row, so that "not recorded" (no
            # row) stays distinguishable from "recorded as unknown" (NULL).
            for position in positions:
                _db_insert(cur, "session_vials", {
                    "session_id": int(session_id),
                    "panel_id": panel_id,
                    "vial_position": position,
                    "made_on": per_vial.get(position, common),
                })

        dates = {per_vial.get(position, common) for position in positions}
        mixed = dates.pop() if len(dates) == 1 else f"{len(dates)} dates"

        logger.info(
            f"Session {session_id} ran {panel_name}, mixed {mixed or 'unknown'}."
            f" {CHECK}"
        )

        self._reset_caches()

    @record_call
    def retire_annotation_key(
        self, *, applies_to: str, key: str, retired: bool = True
    ) -> None:
        """
        Stop offering an annotation, without losing what was already written.

        **USAGE**
        ```python
        db.retire_annotation_key(applies_to="experiment", key="cohort")
        db.retire_annotation_key(
            applies_to="experiment",
            key="cohort",
            retired=False,
        )
        ```

        **PARAMETERS**
        - `applies_to` and `key` name the annotation, as in `db.annotation_keys`
        - `retired=False` brings it back, for when one is retired by mistake

        A retired key is refused by `add_annotation` and dropped from
        `db.missing_annotations()`, but everything written under it stays in
        `db.annotations` and keeps showing up in `db.annotations_for(...)`.
        """
        with self.con as con:
            changed = con.execute(
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

        self._reset_caches()

    def missing_annotations(self) -> pd.DataFrame:
        """
        Everything still to be filled in before this project's data is finished.

        One row per `(target_type, target_id, key)` that is marked `required` in
        `db.annotation_keys` and has nothing written for it yet.
        """
        keys = self.annotation_keys
        required = keys[keys["required"].astype(bool) & ~keys["retired"].astype(bool)]

        rows = []

        for (applies_to, key), entry in required.iterrows():
            table, id_column = ANNOTATION_TARGETS[applies_to]

            found = self.con.execute(
                f"""
                SELECT t.{id_column} FROM {table} AS t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM annotations AS a
                            WHERE a.target_type = ? AND a.key = ?
                              AND a.target_id = t.{id_column}
                    );
                """,
                [applies_to, key],
            ).fetchall()

            rows.extend(
                {
                    "target_type": applies_to,
                    "target_id": row[id_column],
                    "key": key,
                    "label": entry["label"],
                }
                for row in found
            )

        return pd.DataFrame(rows, columns=["target_type", "target_id", "key", "label"])

    @record_call
    def update(self) -> None:
        """
        Find and add all experiments folders in the `main_folder` to the database.

        Experiment folders are folders that contain a `raw` subfolder with
        ScanImage TIFFs.

        It skips the ones that are already included in database.
        """

        # ----------------------------------------------------------------------- #
        #
        # Collect all TIFF files that satisfy:
        #   1) File is inside a "raw" folder that don't start with a '.'
        #   2) File was made by scanimage (and has metadata)
        #
        # We will log all files that satisfy (1) but not (2), and we will
        # group acquisitions with the same loop start time. We assume that:
        #
        #     acqs share a loop start time <=> acqs share a raw folder
        #
        # We will also collect data from all *Events.csv that share an experiment
        # folder with a raw TIFF file.
        #
        # Check schema.svg to see the current database schema diagram!
        #
        # I used preProcessing_v2.m and other scripts in that file as a baseline
        # for what metadata has to be collected, and what needs to be checked.
        #
        # TODO:
        #   1) Make sure all relevant data is added to the db.
        #   2) (SEE NOTE) Add option to overwrite experiment data?
        #
        # NOTE:
        #   - Should not overwrite experiment data and keep calls, because calls
        #     will not be reproducible. Better to add an added_by tag and create
        #     a new experiment every time the metadata is recomputed. Only the
        #     experiment with the latest added_by would be reproducible.
        #
        # ----------------------------------------------------------------------- #

        logger.info("Updating the database...")
        logger.info("Searching for raw files ('**/raw/*.tif')...")

        raw_paths = sorted(self.main_folder.rglob("raw/[!.]?*.tif"))
        assert raw_paths, f"Found no .tif files in: '{self.main_folder.resolve()}'"

        logger.info(f"Found {len(raw_paths)} raw TIFF files.")

        # Split files into experiments
        experiments: defaultdict[str, list[str]] = defaultdict(list)

        for raw_path in raw_paths:
            exp_path = raw_path.parent.parent

            # Path are relative to main_folder to be computer independent
            # This makes the DB method_call parameters reusable

            rel_path = exp_path.relative_to(self.main_folder).as_posix()
            raw_path_rel = raw_path.relative_to(self.main_folder).as_posix()
            experiments[rel_path].append(raw_path_rel)

        # Add experiments to the database
        for path in experiments:
            try:
                self.add_experiment(rel_path=path, rel_raw_paths=experiments[path])

            except Exception:
                logger.exception("Failed to add experiment")

        logger.info("Database updated!")


def _insert_experiment(
    cur: Cursor,
    experiment: Object,
    *,
    rel_path: str,
    method_call_id: int,
) -> int:
    """
    Store one experiment's TIFF metadata, splitting it as the schema does.

    What `_get_raw_metadata` reads out of a TIFF lands in three places under
    this schema: the mouse names a **session**, the rig settings are
    **annotations**, and what is left are the experiment's own **columns**. It
    is read as one dict so the across-TIFF consistency check in `add_experiment`
    still compares every field, and split here, on the way in.
    """
    experiment = dict(experiment)

    exp_start = experiment["exp_start"]
    assert isinstance(exp_start, datetime)

    mouse_id = experiment.pop("mouse_id")
    rig = {key: experiment.pop(key) for key in RIG_ANNOTATIONS}

    session_id = _session_id(
        cur,
        mouse_id=_mouse_number(str(mouse_id)),
        session_date=exp_start.date().isoformat(),
        # The session is the folder above the experiment: '20260708/m442'.
        session_path=Path(rel_path).parent.as_posix(),
    )

    exp_id = _db_insert(
        cur,
        "experiments",
        {
            **experiment,
            "session_id": session_id,
            "exp_start": exp_start.strftime(DT_FORMAT),
        },
    )

    for key, value in rig.items():
        _db_annotate(
            cur,
            target_type="experiment",
            target_id=exp_id,
            key=key,
            value=value,
            method_call_id=method_call_id,
        )

    return exp_id


def _mouse_number(name: str) -> int:
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
            f"Cannot tell which mouse {name!r} is. A name is an optional prefix "
            f"and a number above zero, as in 'm442'."
        )

    return int(match.group(1))


GENOTYPES = {
    "wt": "wt", "wildtype": "wt", "wild type": "wt", "+/+": "wt",
    "het": "het", "heterozygous": "het", "+/-": "het",
    "hom": "hom", "homozygous": "hom", "-/-": "hom",
}


def _mouse_lines(lines: dict[str, None | str]) -> dict[str, None | str]:
    """
    Clean a line-to-genotype mapping for storage.

    Line names keep the spelling they were given, minus surrounding spaces.
    The genotype is matched case-insensitively against the spellings people
    write and stored as `'wt'`, `'het'` or `'hom'`; `None` stays `None`, meaning
    the mouse carries the mutation but was not genotyped for it.

    **EXAMPLE**
    ```python
    _mouse_lines({"TH-Cre ": "Het", "TIGRE": None})
    # {"TH-Cre": "het", "TIGRE": None}
    ```
    """
    cleaned: dict[str, None | str] = {}

    for line, genotype in lines.items():
        name = str(line).strip()

        if not name:
            raise ValueError("A line needs a name.")

        if genotype is None:
            cleaned[name] = None
            continue

        known = GENOTYPES.get(str(genotype).strip().lower())

        if known is None:
            raise ValueError(
                f"Cannot tell what {genotype!r} means for {name}. Use one of: "
                f"{sorted(set(GENOTYPES.values()))}."
            )

        cleaned[name] = known

    return cleaned


VIAL_COLUMNS = ("odor_sccm", "total_sccm", "total_volume_ml", "solvent_volume_ml")
COMPONENT_COLUMNS = ("target_ppm", "liquid_ul", "percent_vv")


def _panel_rows(vials: list[Object]) -> list[tuple]:
    """
    A panel's vials as sorted, comparable tuples.

    The shape is `(position, odor_id, scalars, components)` with `components`
    sorted by odor. Two panels compare equal exactly when they would store the
    same rows, which is what lets an unchanged re-registration do nothing.
    """
    rows = []

    for vial in vials:
        components = sorted(
            (
                int(component["odor_id"]),
                tuple(
                    _optional_real(component.get(name))
                    for name in COMPONENT_COLUMNS
                ),
            )
            for component in vial.get("components") or []
        )

        rows.append((
            int(vial["vial_position"]),
            int(vial["odor_id"]),
            tuple(_optional_real(vial.get(name)) for name in VIAL_COLUMNS),
            components,
        ))

    return sorted(rows)


def _stored_panel(cur: Cursor, panel_id: int) -> list[tuple]:
    """The panel as it currently sits in the database, shaped like `_panel_rows`."""

    components: dict[int, list] = defaultdict(list)

    for row in cur.execute(
        f"SELECT vial_position, odor_id, {', '.join(COMPONENT_COLUMNS)}"
        f" FROM vial_components WHERE panel_id = ?;",
        [panel_id],
    ):
        components[row[0]].append((row[1], tuple(row)[2:]))

    return sorted(
        (row[0], row[1], tuple(row)[2:], sorted(components[row[0]]))
        for row in cur.execute(
            f"SELECT vial_position, odor_id, {', '.join(VIAL_COLUMNS)}"
            f" FROM panel_vials WHERE panel_id = ?;",
            [panel_id],
        )
    )


def _optional_date(value: Value) -> None | str:
    """A `YYYY-MM-DD` string for the database, or `None` for a missing entry."""

    if value is None:
        return None

    # Spreadsheets give a date as a datetime at midnight, and that time is not
    # real, so it is dropped rather than stored as zeros.
    if isinstance(value, datetime):
        return value.date().isoformat()

    return str(value).strip() or None


def _optional_real(value: Value) -> None | float:
    """A float for the database, or `None` for a missing or blank entry."""

    # Spreadsheets give an empty cell as None or as an empty string, and pandas
    # turns it into NaN, which STRICT stores happily and every later sum ruins.
    if value is None or (isinstance(value, str) and not value.strip()):
        return None

    number = float(value)

    return None if number != number else number


def _session_id(
    cur: Cursor, *, mouse_id: int, session_date: str, session_path: str
) -> int:
    """The session for this mouse on this day, creating it if it is new."""

    # A session holds every experiment a mouse did that day, so the second
    # experiment of a session must find the first one's row rather than make
    # another. `UNIQUE (mouse_id, session_date)` is what makes that safe.
    row = cur.execute(
        "SELECT session_id FROM sessions WHERE mouse_id = ? AND session_date = ?;",
        [mouse_id, session_date],
    ).fetchone()

    if row is not None:
        return row["session_id"]

    return _db_insert(
        cur,
        "sessions",
        {
            "mouse_id": mouse_id,
            "session_date": session_date,
            "session_path": session_path,
        },
    )


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

    table, id_column = ANNOTATION_TARGETS[target_type]

    exists = cur.execute(
        f"SELECT EXISTS(SELECT 1 FROM {table} WHERE {id_column} = ?);",
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
            f"take new values. What was written under it is still readable. Use "
            f"`retire_annotation_key(..., retired=False)` to bring it back."
        )

    value = _annotation_value(
        key,
        value,
        row["value_type"],
        row["allowed_values"],
    )

    return _db_insert(
        cur,
        "annotations",
        {
            "target_type": target_type,
            "target_id": target_id,
            "key": key,
            "value": value,
            "method_call_id": method_call_id,
        },
    )


def _annotation_value(
    key: str, value: Value, value_type: str, allowed_values: None | str
) -> Value:
    """Coerce `value` to what the registry says `key` holds, or explain why not."""

    try:
        match value_type:
            case "integer":
                return int(value)  # type: ignore[arg-type]

            case "real":
                return float(value)  # type: ignore[arg-type]

            case "boolean":
                return int(bool(value))

            case "enum":
                options = json.loads(allowed_values or "[]")

                if value not in options:
                    raise ValueError(f"expected one of {options}")

                return str(value)

            case _:
                # 'text' and 'date'. Dates are stored as written: the workbook
                # holds things like '70 um', and repairing that silently would
                # lose what was actually recorded.
                return str(value)

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Annotation '{key}' is declared {value_type} "
            f"but got {value!r} ({error})."
        ) from error


def _db_insert(cur: Cursor, table_name: str, data: Object | list[Object]) -> int:
    # HACK:
    #   ONLY FOR INTERNAL USE (CAN BE USED FOR SQL INJECTION)
    #   Column names are not validated, for simplicity

    template = data[0] if isinstance(data, list) else data

    insertion_query = (
        f"INSERT INTO {table_name} "
        f"({", ".join(template.keys())}) "
        f"VALUES (:{", :".join(template.keys())});"
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
# Data Parsing and Matching
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
        "metadata": fields for the programs table (no exp_id yet)
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
                    logger.error("Experiment will not be added to the DB.")
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


def _parse_program_starts(db: Database, start: datetime) -> list[tuple[datetime, str]]:
    """ "
    Gets program starts after a certain datetime (-1s) from olfactometer log file.
    """

    log_path = (
        db.main_folder
        / INFO_FOLDER
        / start.date().strftime("%Y%m")
        / f"Program_{start.date().strftime("%Y%m%d")}.txt"
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
