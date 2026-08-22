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
import sqlite3

from collections import defaultdict
from datetime import time, datetime, timedelta
from pathlib import Path
from scipy.signal import find_peaks
from sqlite3 import Cursor
from tifffile import TiffFile, TiffPage
from typing import Final

import h5py
import numpy as np
import pandas as pd

from .groups import Group
from .migrate import SCHEMA_VERSION
from .utils import *
from .utils import CallFrame, CallRecorder, _method_calls_dataframe

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TIMEDELTA_MS = timedelta(milliseconds=1)
H5_TOLERANCE = timedelta(milliseconds=100)
DT_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

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

    ALREADY_IN_DB = 1 << 1  # experiment already present, nothing inserted
    MULTIPLE_H5 = 1 << 2  # more than one H5 file in the folder, skipped
    UNSUPPORTED_METADATA = 1 << 3  # TIFF does not have expected metadata format
    METADATA_CHANGED = 1 << 4  # TIFF metadata changed, skipped
    H5_UNMATCHED_ACQ = 1 << 5  # some H5 trials had no matching acquisition
    TRIAL_NO_ACQ = 1 << 6  # some trials matched H5 but had no acquisition
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
        db.events             # `DataFrame` with olfactometer events
        db.experiments        # `DataFrame` with experiment metadata
        db.mcor_files         # `DataFrame` with mcor files metadata
        db.method_calls       # `DataFrame` with `@record_call` functions
        db.odors              # `DataFrame` with current list of odors
        db.programs           # `DataFrame` with one entry per _Event.csv_ file
        db.trials             # `DataFrame` with all olfactometer trials
    ```

    **RELEVANT METHODS**
    ```python
        db.add_experiment(...)          # Add a new experiment folder
        db.update(...)                  # Find and add all experiment folders
        db.latest_calls(method_name)    # `DataFrame` with `method_name` calls
    ```
    """

    def __init__(
        self,
        path: str | Path,
        update=False,
        project: None | str = None,
        _is_test=False,
    ):
        """
        **PARAMETERS**
        - `path` is the main folder holding the experiment folders
        - `update` searches the main folder for experiments to add
        - `project` is a separate database in the same main folder, at
        `.odyn/projects/<project>.db`. `None` is the shared one.

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
            live = odyn_folder / "odyn.db"

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

            live = odyn_folder / "projects" / f"{project}.db"

        self._is_test: Final[bool] = _is_test
        self.path: Final[Path] = self._copy_for_test(live) if _is_test else live

        # Database has a default group to record its calls
        self.group_id: Final[int] = 0

        # Initialize "private" variables
        self._call_stack: list[CallFrame] = []
        self._acquisitions: None | pd.DataFrame = None
        self._events: None | pd.DataFrame = None
        self._experiments: None | pd.DataFrame = None
        self._groups: dict[int, Group] = {}  # Caches groups one-by-one
        self._group_experiments: None | pd.DataFrame = None
        self._mcor_files: None | pd.DataFrame = None
        self._method_calls: None | pd.DataFrame = None
        self._odors: None | pd.DataFrame = None
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
        self.con.close()

    # ----------------------------------------------------------------------- #
    # SQLite Tables as DataFrames
    # ----------------------------------------------------------------------- #

    @property
    def acquisitions(self) -> pd.DataFrame:
        """`DataFrame` with acquisition metadata"""
        self._refresh_if_stale()

        if self._acquisitions is not None:
            return self._acquisitions

        query = "SELECT * FROM acquisitions;"
        self._acquisitions = pd.read_sql_query(
            query, self.con, parse_dates=["acq_start", "odor_start", "odor_end"]
        )
        self._acquisitions.set_index("acq_id", inplace=True)

        return self._acquisitions

    @property
    def events(self) -> pd.DataFrame:
        """`DataFrame` with olfactometer events"""
        self._refresh_if_stale()

        if self._events is not None:
            return self._events

        query = "SELECT * FROM events;"

        self._events = pd.read_sql_query(query, self.con)
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
            query, self.con, parse_dates=["trial_start", "odor_start", "odor_end"]
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
        Returns path to a fresh snapshot of the database.
        For tests only, via `Database(main_folder, _is_test=True)`.

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
        self._events = None
        self._experiments = None
        self._group_experiments = None
        self._mcor_files = None
        self._method_calls = None
        self._outputs = None
        self._programs = None
        self._trials = None

        # In case a specific group can still be accessed
        for group in self._groups.values():
            group._acquisitions = None
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
            SELECT call_output FROM method_calls
                WHERE group_id = ? AND method_name = ? AND call_output IS NOT NULL
                ORDER BY method_call_id DESC LIMIT 1
            """,
            [self.group_id, method_name],
        ).fetchone()

        return json.loads(row["call_output"]) if row else None

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

                exp_id = _db_insert(
                    cur, "experiments", {**experiment, "exp_start": exp_start_str}
                )

                cur.execute("INSERT INTO groups DEFAULT VALUES;")
                group_id = cur.lastrowid

                cur.execute(
                    "INSERT INTO group_experiments (group_id, exp_id) VALUES (?, ?);",
                    [group_id, exp_id],
                )

                _db_insert(cur, "acquisitions", {**acquisition, "exp_id": exp_id})

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
            h5_data: None | dict = None
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
                    # Don't do anything if experiment is already in the DB
                    cur.execute(
                        """
                        SELECT EXISTS(
                            SELECT 1 FROM experiments
                                WHERE exp_start = ?
                        );
                    """,
                        [exp_data["exp_start"]],
                    )

                    if cur.fetchone()[0]:
                        logger.info("Experiment already in DB.")
                        self.add_flag(ExpFlag.ALREADY_IN_DB)
                        return

                    experiment = exp_data

                    # Load H5 and event files before metadata checks
                    # (Checks take some time so better to not do them if possible)
                    h5_paths = list(exp_path.glob("[!.]?*.h5"))

                    if len(h5_paths) > 1:
                        logger.error(
                            f"There is more than one H5 file in this experiment folder. {CROSS}"
                        )

                        for path in h5_paths:
                            relative_path = path.relative_to(self.main_folder)
                            logger.warning(f"  {relative_path}")

                        logger.error("Experiment will not be added to the DB.")
                        self.add_flag(ExpFlag.MULTIPLE_H5)
                        return

                    # Type checking because Object is too generic
                    assert isinstance(experiment["exp_start"], datetime)
                    h5_data = _get_h5_metadata(h5_paths, experiment["exp_start"])

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
            # Phase 2: Match
            # --------------------------------------------------------------- #

            # Acquisitions <-> H5 trials
            # Result: h5_idx -> (acq_idx, h5_to_acq_ms)
            acq_to_h5: dict[int, tuple[int, float]] = {}

            if h5_data and acquisitions:
                acq_to_h5 = _match_acq_to_h5(acquisitions, h5_data)

            # Pool all event trial starts across programs
            # event_trial_pool[i] = (program_idx, trial_idx, trial_start)
            trials: list[tuple[int, int, datetime]] = [
                (program_idx, trial_idx, trial["trial_start"])
                for program_idx, program in enumerate(programs_data)
                for trial_idx, trial in enumerate(program["trials"])
            ]

            # CSV trials <-> H5 trials
            # Result: pool_idx -> (h5_idx, h5_to_trial_ms)
            csv_to_h5: dict[int, tuple[int, float]] = {}

            if h5_data and trials:
                trial_starts = [x[2] for x in trials]
                csv_to_h5 = _match_csv_to_h5(trial_starts, h5_data)

            # Build lookup: (program_idx, trial_idx) -> (h5_idx, h5_to_trial_ms)
            trial_to_h5: dict[tuple[int, int], tuple[int, float]] = {
                (trials[pool_idx][0], trials[pool_idx][1]): (
                    h5_idx,
                    h5_to_trial_ms,
                )
                for pool_idx, (h5_idx, h5_to_trial_ms) in csv_to_h5.items()
            }

            # --------------------------------------------------------------- #
            # Phase 3: Insert
            # --------------------------------------------------------------- #

            # Fix datetime format to include microseconds
            assert isinstance(experiment["exp_start"], datetime)
            exp_start_str = experiment["exp_start"].strftime(DT_FORMAT)

            exp_id = _db_insert(
                cur, "experiments", {**experiment, "exp_start": exp_start_str}
            )

            cur.execute("INSERT INTO groups DEFAULT VALUES;")
            group_id = cur.lastrowid

            cur.execute(
                "INSERT INTO group_experiments (group_id, exp_id) VALUES (?, ?);",
                [group_id, exp_id],
            )

            # Insert acquisitions matched to H5 trials (with odor timing from H5)
            h5_to_acq_id: dict[int, int] = {}
            matched_acq_indices: set[int] = set()

            if h5_data:
                for h5_idx, (acq_idx, delta_ms) in acq_to_h5.items():
                    # Type checking because Object is too generic
                    acq_start = acquisitions[acq_idx]["acq_start"]
                    assert isinstance(acq_start, datetime)

                    acq = {
                        **acquisitions[acq_idx],
                        "exp_id": exp_id,
                        "acq_start": acq_start.strftime(DT_FORMAT),
                        "odor_start": _to_datetime_str(h5_data["odor_starts"][h5_idx]),
                        "odor_end": _to_datetime_str(h5_data["odor_ends"][h5_idx]),
                        "h5_to_acq_ms": delta_ms,
                    }
                    h5_to_acq_id[h5_idx] = _db_insert(cur, "acquisitions", acq)
                    matched_acq_indices.add(acq_idx)

            # Fallback: insert acquisitions with no matching H5 trial
            for acq_idx, acq in enumerate(acquisitions):
                if acq_idx not in matched_acq_indices:
                    _db_insert(cur, "acquisitions", {**acq, "exp_id": exp_id})

            # Insert programs, trials, and events
            for program_idx, program_data in enumerate(programs_data):
                program_id = _db_insert(
                    cur, "programs", {**program_data["metadata"], "exp_id": exp_id}
                )

                # Pass 1: insert trials, collect trial_ids by index
                trial_ids: dict[int, int] = {}

                for trial_idx, trial in enumerate(program_data["trials"]):
                    acq_id = None
                    h5_to_trial_ms = None

                    if (program_idx, trial_idx) in trial_to_h5:
                        h5_idx, delta_ms = trial_to_h5[(program_idx, trial_idx)]
                        acq_id = h5_to_acq_id.get(h5_idx)
                        if acq_id is not None:
                            h5_to_trial_ms = delta_ms

                    trial_ids[trial_idx] = _db_insert(
                        cur,
                        "trials",
                        {
                            "trial_start": trial["trial_start"].strftime(DT_FORMAT),
                            "odor_start": trial["odor_start"].strftime(DT_FORMAT),
                            "odor_end": trial["odor_end"].strftime(DT_FORMAT),
                            "odor_id": trial["odor_id"],
                            "outcome": trial["outcome"],
                            "acq_id": acq_id,
                            "h5_to_trial_ms": h5_to_trial_ms,
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

            if h5_data:
                n_h5 = len(h5_data["trial_starts"])
                n_matched_acq = len(acq_to_h5)

                if n_matched_acq < n_h5:
                    self.add_flag(ExpFlag.H5_UNMATCHED_ACQ)
                    logger.warning(
                        f"{n_h5 - n_matched_acq} H5 trials without "
                        f"matching acquisition. {CROSS}"
                    )
                else:
                    logger.info(f"All H5 trials matched to acquisitions. {CHECK}")

            if trials:
                n_events = len(trials)
                n_matched = len(csv_to_h5)

                if n_matched < n_events:
                    logger.info(f"{n_events - n_matched} trials without H5 match.")

                n_with_acq = sum(
                    1
                    for (h5_idx, _) in trial_to_h5.values()
                    if h5_to_acq_id.get(h5_idx) is not None
                )

                if n_with_acq < n_matched:
                    self.add_flag(ExpFlag.TRIAL_NO_ACQ)
                    logger.warning(
                        f"{n_matched - n_with_acq} trials matched "
                        f"to H5 but no acquisition. {CROSS}"
                    )
                elif event_files:
                    logger.info(f"All matched trials have acquisitions. {CHECK}")

            self._reset_caches()

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


def _get_h5_metadata(
    paths: list[Path], exp_start: datetime
) -> None | dict[str, np.ndarray]:
    # There must be at most one path
    if not paths:
        return None

    path = paths[0]

    # Very similar to getScopeH5Timestamps
    logger.info(f"Getting timing data from: '{path}'")

    # Parse experiment start time
    exp_start_np = np.datetime64(exp_start)

    with h5py.File(path) as f:
        samplerate = f.attrs["samplerate"]

        imaging_TTL = f["ImagingWindow"][:]
        odor_TTL = f["OdorDelivery"][:]

        # TODO: (Vinicius)
        #   Maybe change the way we find starts? Because adding the distance argument
        #   picks the highest and not the first choice (both in MATLAB and Python).

        # NOTE: (Priscilla, from MATLAB code, adapted)
        #   Added distance to deal with problematic file where
        #   code found 2 peaks right next to each other

        trial_starts, _ = find_peaks(
            np.diff(imaging_TTL), height=2.0, distance=samplerate / 2
        )
        odor_starts, _ = find_peaks(
            np.diff(odor_TTL), height=2.0, distance=samplerate / 10
        )
        odor_ends, _ = find_peaks(
            -np.diff(odor_TTL), height=2.0, distance=samplerate / 10
        )

        # Returns None if no trials where found
        if len(trial_starts) == 0:
            logger.info("No trial triggers found in this H5 file.")
            return None

        # Shift everything by first trial start, to match FrameTimestamp_sec data
        # FrameTimestamp_sec always starts at zero, so they almost exactly match
        shift = trial_starts[0]

        trial_starts -= shift
        odor_starts -= shift
        odor_ends -= shift

        assert len(trial_starts) == len(odor_starts) == len(odor_ends), (
            f"The following do not match:\n"
            f"    Number of trials {len(trial_starts)}\n"
            f"    Odor presentation starts {len(odor_starts)}\n"
            f"    Odor presentation ends {len(odor_ends)}"
        )

        logger.info(f"Found {len(trial_starts)} trial starts in the H5 file.")

        # Convert to timedeltas
        trial_starts = (trial_starts / samplerate * 1e9).astype("timedelta64[ns]")
        odor_starts = (odor_starts / samplerate * 1e9).astype("timedelta64[ns]")
        odor_ends = (odor_ends / samplerate * 1e9).astype("timedelta64[ns]")

        # Return the datetimes to be matched with acquisition frame times
        return {
            "trial_starts": exp_start_np + trial_starts,
            "odor_starts": exp_start_np + odor_starts,
            "odor_ends": exp_start_np + odor_ends,
        }


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


def _match_acq_to_h5(
    acquisitions: list[Object],
    h5_data: dict[str, np.ndarray],
) -> dict[int, tuple[int, float]]:
    """
    Match acquisitions to H5 trials by nearest timestamp.

    h5_to_acq_ms stores the signed difference (acq_start - h5_trial_start) in ms.

    Returns dict: h5_idx -> (acq_idx, h5_to_acq_ms)
    """
    h5_dts = [_to_datetime(t) for t in h5_data["trial_starts"]]
    matches: dict[int, tuple[int, float]] = {}
    h5_ptr = 0

    for acq_idx, acq in enumerate(acquisitions):
        # Type checking because Object is too generic
        acq_start = acq["acq_start"]
        assert isinstance(acq_start, datetime)

        # Advance h5 pointer past trials clearly before this acquisition
        while h5_ptr < len(h5_dts) - 1 and h5_dts[h5_ptr] < acq_start - H5_TOLERANCE:
            h5_ptr += 1

        if h5_ptr < len(h5_dts):
            delta = acq_start - h5_dts[h5_ptr]
            if abs(delta) < H5_TOLERANCE:
                matches[h5_ptr] = (acq_idx, delta / TIMEDELTA_MS)
                h5_ptr += 1

    return matches


def _match_csv_to_h5(
    csv_starts: list[datetime],
    h5_data: dict[str, np.ndarray],
) -> dict[int, tuple[int, float]]:
    """
    Find the best alignment of CSV trial starts with H5 trial starts.

    This function searches for the starting position k in the csv_starts list
    such that csv_starts[k:k+n_h5] best aligns with h5 trial starts (minimizing
    std of pairwise differences). Unmatched trials at the boundaries are left out.

    h5_to_trial_ms stores the signed difference (csv_start - h5_start) in ms.

    Returns dict: pool_idx -> (h5_idx, h5_to_trial_ms)
    """
    h5_starts = [_to_datetime(t) for t in h5_data["trial_starts"]]
    n_h5 = len(h5_starts)
    n_events = len(csv_starts)

    if n_events == 0 or n_h5 == 0:
        return {}

    if n_events < n_h5:
        logger.warning(
            f"Fewer trial starts in the CSV files ({n_events})"
            f" than in the H5 file ({n_h5}). Matching as many as possible."
        )
        n_h5 = n_events

    # Use offsets from the first H5 trial for numerical stability
    base = h5_starts[0]
    h5_ms = np.array([(t - base).total_seconds() * 1000 for t in h5_starts[:n_h5]])
    trial_ms = np.array([(t - base).total_seconds() * 1000 for t in csv_starts])

    best_k = 0
    best_std = float("inf")

    for k in range(n_events - n_h5 + 1):
        std = float(np.std(trial_ms[k : k + n_h5] - h5_ms))
        if std < best_std:
            best_std = std
            best_k = k

    diffs = trial_ms[best_k : best_k + n_h5] - h5_ms

    logger.info(
        f"Average clock offset (event - h5):"
        f" {float(np.mean(diffs)):.1f} ms, std: {best_std:.1f} ms"
    )

    return {best_k + j: (j, float(diffs[j])) for j in range(n_h5)}


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


def _to_datetime(dt: np.datetime64) -> datetime:
    dt_str = np.datetime_as_string(dt).item()
    return datetime.fromisoformat(dt_str)


def _to_datetime_str(dt: np.datetime64) -> str:
    return _to_datetime(dt).strftime(DT_FORMAT)
