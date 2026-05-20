# --------------------------------------------------------------------------- #
# NOTE:
# - Programs with same name can be different
# - Parse log excel file possibly
# - RWD Olfactometer trigger
# - All odors have concentration, even monomolecular ones (add to DB)
# - Odor ID and odor name (future unchangeable but currently in flux)
# - Not in metadata: Odor Dilution (%v/v), Odor Made in Date
# - Inconsistent a:b and a/b usage
# - Record vial to odor association
#
# TODO:
#   - Use logging module instead of print (console + log file)
#   - Add log table and every method_call stores a log
#   - Start a test suite (so far only for DB tests)
#   - Add quality control plots for ported MATLAB code
#   - Reduce the conversions between strings and datetimes
#   - Match Odor ID and name (prefer name, fail test)
#
# NOTE:
#   How to deal with output files being in various computers and analysis
#   running locally? Given speed concerns and the different workflows of the
#   lab members, this should be requirement. Possible solutions and steps:
#       - Add 'computer' column/prefix when there is a folder in the table.
#         ( Paths starting with "." should be still relative to the
#           main_folder which is independent of the computer )
#       - Maybe use some env/global variables like $SERVER, $COMPUTER_NAME.
#       - Maybe should add small db for consolidation in folders that are
#         not relative.
#
# MAYBE TODO:
#   - Add git hash to every db entry? (To help db updates...)
#   - Make a function that creates .py file containing the whole
#     processing/analysis pipeline, to run on the server.
# --------------------------------------------------------------------------- #

import json
import sqlite3

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from scipy.signal import find_peaks
from sqlite3 import Cursor
from tifffile import TiffFile, TiffPage
from typing import Optional, Any

import h5py
import numpy as np
import pandas as pd

from .groups import Group
from .utils import ProgressBar, INFO, FAIL, PASS, CHECK, CROSS

type InsertData = dict[str, Any]


class Database:
    """
    \033[1;35mDATABASE\033[0m
    Creates and connects you to the database.

    \033[1;34mUSAGE\033[0m
        db = Database(main_folder)

    \033[1;34mRELEVANT PROPRIETIES/METHODS\033[0m
        db.groups          <- Return list of Groups for processing/analysis
        db.experiments     <- Returns DataFrame with experiment metadata
        db.acquisitions    <- Returns DataFrame with acquisition metadata
        db.mcor_files      <- Returns DataFrame with mcor files metadata

    Run Database.help('method_name') to know more about one of the methods above.

    \033[1;34mEXAMPLE\033[0m
        Database.help('experiment')
    """

    def __init__(self, path: str | Path, force_update=False):
        self.main_folder = Path(path)
        self.path = self.main_folder / ".odyn" / "odyn.db"

        # Initialize "private" variables
        self._acquisitions: Optional[pd.DataFrame] = None
        self._events: Optional[pd.DataFrame] = None
        self._experiments: Optional[pd.DataFrame] = None
        self._groups: Optional[list[Group]] = None
        self._mcor_files: Optional[pd.DataFrame] = None
        self._method_calls: Optional[pd.DataFrame] = None
        self._odors: Optional[pd.DataFrame] = None
        self._programs: Optional[pd.DataFrame] = None
        self._trials: Optional[pd.DataFrame] = None

        # Get connection or create database if needed
        if not self.path.exists():
            print(f"{INFO} Did not find a database!")
            print(f"{INFO} Creating database...")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path)

            self.update()

            print(f"{INFO} Database created at: {self.path.resolve()}")

        elif force_update:
            print(f"{INFO} Updating the database...")

            self.con = sqlite3.connect(self.path)
            self.update()

            print(f"{INFO} Database updated!")
            print(f"{INFO} Path to the database: {self.path.resolve()}")

        else:
            self.con = sqlite3.connect(self.path)
            print(f"{INFO} Connected to the database at: {self.path.resolve()}")

        self.con.execute("PRAGMA foreign_keys = ON;")
        self.con.row_factory = sqlite3.Row

    def __del__(self):
        self.con.close()

    @staticmethod
    def help(name="Database"):
        if name.lower() == "database":
            return print(Database.__doc__)

        attr = getattr(Database, name, None)

        if attr is not None:
            return print(attr.__doc__)

        print(f"{INFO} Method not found!")

    @property
    def acquisitions(self) -> pd.DataFrame:
        if self._acquisitions is not None:
            return self._acquisitions

        query = "SELECT * FROM acquisitions;"
        self._acquisitions = pd.read_sql_query(
            query, self.con, parse_dates=["acq_start"]
        )
        self._acquisitions.set_index("acq_id", inplace=True)

        return self._acquisitions

    @property
    def events(self) -> pd.DataFrame:
        if self._events is not None:
            return self._events

        query = "SELECT * FROM events;"

        self._events = pd.read_sql_query(query, self.con)
        self._events.set_index("event_id", inplace=True)

        return self._events

    @property
    def experiments(self) -> pd.DataFrame:
        if self._experiments is not None:
            return self._experiments

        query = "SELECT * FROM experiments;"
        self._experiments = pd.read_sql_query(
            query, self.con, parse_dates=["exp_start", "added_to_db_at"]
        )
        self._experiments.set_index("exp_id", inplace=True)

        return self._experiments

    @property
    def groups(self) -> list[Group]:
        if self._groups is not None:
            return self._groups

        query = "SELECT group_id FROM groups;"
        res = self.con.execute(query)

        group_ids = sorted(exp["group_id"] for exp in res.fetchall())
        self._groups = [Group(group_id, self) for group_id in group_ids]

        return self._groups

    @property
    def mcor_files(self) -> pd.DataFrame:
        if self._mcor_files is not None:
            return self._mcor_files

        query = "SELECT * FROM mcor_files;"
        self._mcor_files = pd.read_sql_query(query, self.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._mcor_files

    @property
    def method_calls(self) -> pd.DataFrame:
        if self._method_calls is not None:
            return self._method_calls

        query = "SELECT * FROM method_calls;"

        self._method_calls = pd.read_sql_query(
            query, self.con, parse_dates=["called_at"]
        )
        self._method_calls.set_index("method_call_id", inplace=True)
        self._method_calls["parameters"] = self._method_calls["parameters"].apply(
            json.loads
        )

        return self._method_calls

    @property
    def odors(self) -> pd.DataFrame:
        if self._odors is not None:
            return self._odors

        query = "SELECT * FROM odors;"

        self._odors = pd.read_sql_query(query, self.con)
        self._odors.set_index("odor_id", inplace=True)

        return self._odors

    @property
    def programs(self) -> pd.DataFrame:
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
        if self._trials is not None:
            return self._trials

        query = "SELECT * FROM trials;"

        self._trials = pd.read_sql_query(
            query, self.con, parse_dates=["trial_start", "odor_start", "odor_end"]
        )
        self._trials.set_index("trial_id", inplace=True)

        return self._trials

    def _get_raw_metadata(self, path: Path) -> Optional[tuple[InsertData, InsertData]]:
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

        experiment: InsertData = {
            "exp_name": "_".join(file_stem_parts[:-1]),
            "exp_type": tif.scanimage_metadata["FrameData"]["SI.acqState"],
            "mouse_id": file_stem_parts[1],
            "height_px": tif.pages[0].tags["ImageLength"].value,
            "width_px": tif.pages[0].tags["ImageWidth"].value,
            "frame_count": len(tif.pages),
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
        dt = datetime.strptime(date_string, "%Y %m %d %H %M %S.%f")

        loop_start = dt.isoformat(" ")
        experiment["exp_start"] = loop_start

        # Data specific to the acquisition
        delta_sec = float(image_description["frameTimestamps_sec"])
        acquisition_time = dt + timedelta(seconds=delta_sec)

        acquisition: InsertData = {
            "raw_path": str(path.relative_to(self.main_folder)),
            "acq_start": acquisition_time.isoformat(" "),
        }

        return experiment, acquisition

    def _insert_events(
        self, cur: Cursor, trial_id: Optional[int], events: list[InsertData]
    ):
        for event in events:
            event["trial_id"] = trial_id

            _db_insert(cur, "events", event)

        events.clear()

    def _insert_trial(self, cur: Cursor, trial: InsertData, events: list[InsertData]):
        trial_id = _db_insert(cur, "trials", trial)

        self._insert_events(cur, trial_id, events)

    def _reset_caches(self) -> None:
        self._acquisitions = None
        self._events = None
        self._experiments = None
        self._mcor_files = None
        self._method_calls = None
        self._programs = None
        self._trials = None

        if self._groups is None:
            return

        # In case a specific group can still be accessed
        for group in self._groups:
            group._acquisitions = None
            group._events = None
            group._experiments = None
            group._mcor_files = None
            group._method_calls = None
            group._programs = None
            group._trials = None

        self._groups = None

    def add_experiment(
        self,
        *,
        rel_path: str,
        raw_paths: Optional[list[Path]] = None,
    ) -> None:

        # Basically, everything in this function is done in a single transaction
        # because if something fails we rollback all insertions.

        exp_path = self.main_folder / rel_path

        assert exp_path.is_dir(), f"Folder not found: {exp_path.resolve()}"

        # Fetch raw file paths list if not provided
        if raw_paths is None:
            raw_paths = sorted(self.main_folder.glob("raw/[!.]?*.tif"))

        # ASSUMPTION: raw_paths are sorted

        print(f"{INFO} Processing folder: {exp_path.resolve()}")

        with self.con as con:
            # Create a cursor to get row ids
            cur = con.cursor()

            # Make sure foreign keys will be checked
            cur.execute("PRAGMA foreign_keys = ON;")

            # ----------------------------------------------------------------------- #
            # Load TIFF metadata
            # ----------------------------------------------------------------------- #

            # Initialize variables
            first_metadata = True
            checks_failed = 0
            acquisitions = []

            bar = ProgressBar(len(raw_paths))
            bar.show()

            for raw_path in raw_paths:
                last_raw_metadata = self._get_raw_metadata(raw_path)

                if last_raw_metadata is None:
                    msg = f"{INFO}   Skipped file {raw_path} (metadata format not supported)"
                    bar.message(msg)
                    continue

                last_exp_data, acq = last_raw_metadata

                # Don't do anything if experiment is already in the DB
                if first_metadata:
                    query = f"""
                        SELECT EXISTS(
                            SELECT 1 FROM experiments as e
                                WHERE e.exp_start = ?
                        );
                    """
                    cur.execute(query, [last_exp_data["exp_start"]])

                    if cur.fetchone()[0]:
                        bar.end(f"{INFO} Experiment already in DB.")
                        return

                    experiment = last_exp_data
                    first_metadata = False

                # Check if metadata is consistent across acquisitions
                elif experiment != last_exp_data:
                    checks_failed += 1

                    bar.message(
                        f"{FAIL}   {raw_path} metadata is inconsistent "
                        "with the first experiment acquisition."
                    )
                    bar.message(f"{experiment}")
                    bar.message(f"{last_exp_data}")

                acquisitions.append(acq)
                bar.step()

            bar.end()

            # Don't add the experiment if checks failed
            if checks_failed > 0:
                print(f"{FAIL} Failed {checks_failed} TIFF metadata checks. {CROSS}")
                return

            print(f"{PASS} Passed all TIFF metadata checks! {CHECK}")

            # ----------------------------------------------------------------------- #
            # Insert Experiment and Group
            # ----------------------------------------------------------------------- #

            # Add experiment to the database
            exp_id = _db_insert(cur, "experiments", experiment)

            # Add a group associated with this experiment
            cur.execute("INSERT INTO groups DEFAULT VALUES;")
            group_id = cur.lastrowid

            insertion_query = """
                INSERT INTO group_experiments
                    ( group_id
                    , exp_id
                    ) VALUES (?, ?);
            """
            cur.execute(insertion_query, [group_id, exp_id])

            # ----------------------------------------------------------------------- #
            # Load H5 file and Event files
            # ----------------------------------------------------------------------- #
            # TODO:
            #   - Simplify (and maybe vectorize) this section
            # ----------------------------------------------------------------------- #

            h5_paths = list(exp_path.glob("*.h5"))

            assert (
                len(h5_paths) <= 1
            ), "There must be at most one H5 file per experiment!"

            # Get the datetimes of trial starts and odor presentation starts and ends
            # It is None if there is no H5 file or if there are no trial starts
            timing_data = (
                _get_h5_metadata(h5_paths[0], experiment["exp_start"])
                if h5_paths
                else None
            )

            # Get event files
            event_files = sorted(
                exp_path.rglob("[!.]?*Events.csv"), key=lambda x: x.stat().st_mtime
            )

            print(f"{INFO} Found {len(event_files)} olfactometer event files.")

            # ----------------------------------------------------------------------- #
            # Insert Program, Events, Trials, and Acquisitions
            # ----------------------------------------------------------------------- #
            # TODO:
            #   - Simplify (and maybe vectorize) this section
            # ----------------------------------------------------------------------- #

            # List of recognized program types
            # TODO: Make it part of the database
            program_types = [
                "fine 1",
                "fine 2",
                "coarse 1",
                "coarse 2",
                "passive",
                "warm-up",
            ]

            # Next acquisition not currently matched to a trial
            next_acq = 0

            # Initialize trial variables
            events: list[InsertData] = []
            trials_without_acq = 0
            current_trial = 0

            for event_file in event_files:
                # All data should be in inside main_folder
                program_path = str(event_file.relative_to(self.main_folder))

                # Parse file name
                stem_split = event_file.stem.split("-")

                program_name = stem_split[0]
                program_start = datetime.strptime(
                    " ".join(stem_split[1:3]), "%Y_%m_%d %H_%M_%S"
                )

                # Find program type (DEFAULT: "unknown")
                program_type = "unknown"

                for t in program_types:
                    if t in program_name:
                        program_type = t

                # Skip "Buffer" programs
                if "buffer" in program_name.lower():
                    continue

                # Add program to the insertion list
                program = {
                    "exp_id": exp_id,
                    "program_name": program_name,
                    "program_type": program_type,
                    "program_start": program_start,
                    "program_path": program_path,
                }

                # Insert program into DB
                program_id = _db_insert(cur, "programs", program)

                # Simple iteration to parse file
                trial: Optional[InsertData] = None

                df = _parse_event_file(event_file)
                is_response_window = False
                licks_count = 0

                for _, (event_time, event_name, event_type, event_tag) in df.iterrows():
                    event: Optional[InsertData] = None

                    # Check if a trial is starting
                    if event_type == "Trial" and event_tag != "Interval":
                        # Insert the last trial (if not the first)
                        if trial is not None:
                            self._insert_trial(cur, trial, events)

                            current_trial += 1

                        # Stop processing if tag is not an integer
                        try:
                            int(event_tag)

                        except ValueError as e:
                            print(f"{FAIL} Unexpected event: {event_name}")
                            print(f"{FAIL} Experiment will not be added to DB.")

                            raise e

                        assert (
                            timing_data is not None
                        ), "There is a trial without corresponding H5 file data"

                        trial_start = timing_data["trial_starts"][current_trial]
                        odor_start = timing_data["odor_starts"][current_trial]
                        odor_end = timing_data["odor_ends"][current_trial]

                        # Finds acquisition that is associated with this trial
                        # NOTE: Assumes that acquisitions are ordered
                        acq_id = None
                        tolerance = np.timedelta64(1, "ms")

                        if next_acq < len(acquisitions):
                            acq = acquisitions[next_acq]
                            acq_start = np.datetime64(acq["acq_start"])

                            # Check if next acquisition matches the trial
                            if np.abs(trial_start - acq_start) < tolerance:
                                # Insert the acquisition into DB
                                acq["exp_id"] = exp_id
                                acq_id = _db_insert(cur, "acquisitions", acq)

                                next_acq += 1

                            else:
                                trials_without_acq += 1

                        else:
                            trials_without_acq += 1

                        trial = {
                            "trial_start": _format_np_datetime(trial_start),
                            "odor_start": _format_np_datetime(odor_start),
                            "odor_end": _format_np_datetime(odor_end),
                            "odor_id": None,
                            "outcome": "na",
                            "acq_id": acq_id,
                            "program_id": program_id,
                            "exp_id": exp_id,
                        }

                    elif event_type == "Lick" and is_response_window:
                        licks_count += 1

                    # Licks start to count after "Response"
                    elif event_type == "Response":
                        is_response_window = True

                    # End of reponse window
                    elif event_type == "Trial" and event_tag == "Interval":
                        is_response_window = False

                        # "Trial Interval" can only come after a trial is created
                        assert trial is not None, "Response window without trial"

                        if program_type != "passive" and trial["outcome"] != "hit":
                            trial["outcome"] = (
                                "miss" if licks_count < 3 else "false choice"
                            )

                        licks_count = 0

                    # Record licks inside response window
                    elif is_response_window and event_type == "Lick":
                        licks_count += 1

                    # Add odor_id to trial
                    elif event_type == "Odor":
                        # "Odor #" can only come after a trial is created
                        assert trial is not None, "Odor presentation without trial"

                        trial["odor_id"] = int(event_tag)

                    # Reward <=> hit
                    elif event_type == "Reward":
                        # "Reward" can only come after a trial is created
                        assert trial is not None, "Reward without trial"

                        trial["outcome"] = "hit"

                    # Don't add session start to DB
                    elif event_type == "Session":
                        continue

                    # Add event to event list
                    event = {
                        "program_id": program_id,
                        "event_time": event_time,
                        "event_type": event_type,
                        "event_tag": event_tag,
                    }

                    events.append(event)

                # Add last trial and its events
                if trial is not None:
                    self._insert_trial(cur, trial, events)
                    current_trial += 1

                # Add events without trial
                elif events:
                    self._insert_events(cur, None, events)

            if trials_without_acq > 0:
                print(
                    f"{FAIL} There are {trials_without_acq} trials without acquisition! {CROSS}"
                )
            else:
                print(f"{PASS} All trials have matching acquisitions. {CHECK}")

            # ----------------------------------------------------------------------- #
            # Fallback Acquisition Insertions
            # ----------------------------------------------------------------------- #
            #   If there are no event files, acquisitions must be inserted without a
            #   corresponding trial. We do this here.
            # ----------------------------------------------------------------------- #

            if not event_files:
                for acq in acquisitions:
                    acq["exp_id"] = exp_id

                _db_insert(cur, "acquisitions", acquisitions)

            # Make sure all properties will be recomputed
            self._reset_caches()

    def update(self) -> None:
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
        # TODO: 1) Make sure all relevant data is added to the db.
        #       2) Add option to overwrite experiment data? (Keeping calls.)
        #
        # ----------------------------------------------------------------------- #

        print(f"{INFO} Searching for raw files ('**/raw/*.tif')...")

        raw_paths = sorted(self.main_folder.rglob("raw/[!.]?*.tif"))
        assert raw_paths, f"Found no .tif files in: {self.main_folder.resolve()}"

        print(f"{INFO} Found {len(raw_paths)} raw TIFF files.")

        # Split files into experiments
        experiments: defaultdict[str, list[Path]] = defaultdict(list)

        for raw_path in raw_paths:
            exp_path = raw_path.parent.parent
            rel_path = str(exp_path.relative_to(self.main_folder))
            experiments[rel_path].append(raw_path)

        # Create DB if it doesn't exists
        with self.con:
            with open(Path(__file__).parent / "create.sql") as f:
                self.con.executescript(f.read())

        # Add experiments to the database
        for path in experiments:
            self.add_experiment(rel_path=path, raw_paths=experiments[path])

    def from_query(self, query: str):
        """
        \033[1;35mFROM_QUERY\033[0m
        Creates a pandas DataFrame from a SQL query.

        \033[1;34mUSAGE\033[0m
            db = Database(main_folder)
            df = db.from_query(query_as_a_string)

        \033[1;34mEXAMPLE\033[0m
            db.from_query("SELECT exp_id, exp_name FROM experiments;")
        """
        return pd.read_sql_query(query, self.con)


def _db_insert(cur: Cursor, table_name: str, data: InsertData | list[InsertData]):
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

    return cur.lastrowid


def _format_np_datetime(dt: np.datetime64) -> str:
    dt_str = np.datetime_as_string(dt).item()
    return datetime.fromisoformat(dt_str).isoformat(" ")


def _get_h5_metadata(path: Path, exp_start: str) -> Optional[dict[str, np.ndarray]]:
    # Very similar to getScopeH5Timestamps
    print(f"{INFO} Getting timing data from: {path}")

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
            print(f"{INFO} No trials triggers found in this H5 file.")
            return None

        # Shift everything by first trial start, to match FrameTimestamp_sec data
        # FrameTimestamp_sec always starts at zero, so they almost exactly match
        trial_starts -= trial_starts[0]
        odor_starts -= trial_starts[0]
        odor_ends -= trial_starts[0]

        assert len(trial_starts) == len(odor_starts) == len(odor_ends), print(
            f"{FAIL} The following do not match:"
            f"{FAIL}    Number of trials ({len(trial_starts)}"
            f"{FAIL}    Odor presentation starts ({len(trial_starts)}"
            f"{FAIL}    Odor presentation ends ({len(odor_ends)}"
        )

        print(f"{INFO} Found {len(trial_starts)} trial starts in the H5 file.")

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


def _parse_event_file(path: Path) -> pd.DataFrame:
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

    # Simplify "Tag" value for df["Type"] == "Odor"
    mask = df["Tag"].str.startswith("I ")
    df.loc[mask, "Tag"] = df.loc[mask, "Tag"].str.split(" ").str[1]

    return df
