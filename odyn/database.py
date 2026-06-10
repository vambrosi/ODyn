# --------------------------------------------------------------------------- #
# NOTE:
#   - Programs with same name can be different
#   - RWD Olfactometer trigger
#   - All odors have concentration, even monomolecular ones (add to DB)
#   - Not in metadata: Odor Dilution (%v/v), Odor Made in Date
#   - Inconsistent a:b and a/b usage
#   - Record vial to odor association
#   - GUI for strides and overlaps
#
# NOTE (Olfactometer vs Acquisition shift)
#   - Seems consistent across programs, but not whole experiments.
#
# TODO:
#   - Change h5 timing data assert position
#   - Add mcors that where already made to DB.
#   - Assert that non-passive trials have a non "na" outcome
#   - Possibly change to "nothing fails, just skip and log"
#   - Use logging module instead of print (console + log file)
#   - Add log table and every method_call stores a log
#   - Start a test suite (so far only for DB tests)
#   - Add quality control plots for ported MATLAB code
#   - ProgressBar is increasing sometimes
#
#   - What to do about some particular experiments?
#       - '20250703/SID200' has more trials in .csv than in H5
#
# NOTE:
#   How to deal with output files being in various computers and analysis
#   running locally? Given speed concerns and the different workflows of the
#   lab members, this should be a requirement. Possible solutions and steps:
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
from datetime import time, datetime, timedelta
from pathlib import Path
from scipy.signal import find_peaks
from sqlite3 import Cursor
from tifffile import TiffFile, TiffPage
from typing import Any, Final, Optional

import h5py
import numpy as np
import pandas as pd

from .groups import Group
from .utils import *

type InsertData = dict[str, Any]

TIMEDELTA_MS = timedelta(milliseconds=1)
H5_TOLERANCE = timedelta(milliseconds=100)

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

    def __init__(self, path: str | Path, update=False):
        self.main_folder = Path(path)
        self.path: Final[Path] = self.main_folder / ODYN_FOLDER / "odyn.db"

        # Database has a default group to record its calls
        self.group_id: Final[int] = 0

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

        # Get connection and create database if needed
        if not self.path.exists():
            logger.info("Did not find a database!")
            logger.info("Creating database...")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path)

            # Create schema and add default values
            with self.con as con:
                with open(Path(__file__).parent / "create.sql") as f:
                    con.executescript(f.read())

                # Insert default database group
                query = "INSERT OR IGNORE INTO groups (group_id) VALUES (?);"
                con.execute(query, [self.group_id])

            logger.info(f"Database created at: '{self.path.resolve()}'")

        else:
            self.con = sqlite3.connect(self.path)
            logger.info(f"Connected to the database at: '{self.path.resolve()}'")

        self.con.execute("PRAGMA foreign_keys = ON;")
        self.con.row_factory = sqlite3.Row

        if update:
            self.update()

    def __del__(self):
        self.con.close()

    @staticmethod
    def help(name="Database"):
        if name.lower() == "database":
            return print(Database.__doc__)

        attr = getattr(Database, name, None)

        if attr is not None:
            return print(attr.__doc__)

        logger.info("Method not found!")

    @property
    def acquisitions(self) -> pd.DataFrame:
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

        query = "SELECT group_id FROM groups WHERE group_id != ?;"
        res = self.con.execute(query, [self.group_id])

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

        acquisition: InsertData = {
            "raw_path": str(path.relative_to(self.main_folder)),
            "acq_start": acquisition_time,
        }

        return experiment, acquisition

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

    @record_call
    def add_experiment(
        self,
        *,
        rel_path: str,
        rel_raw_paths: Optional[list[str]] = None,
    ) -> None:

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

            experiment: Optional[InsertData] = None
            acquisitions: list[InsertData] = []
            h5_data: Optional[dict] = None
            event_files: list[Path] = []

            last_exp_data: Optional[InsertData] = None
            checks_failed = 0

            assert raw_paths, "Did not find any raw/*.tif files."

            bar = ProgressBar(len(raw_paths))
            bar.show()

            for raw_path in raw_paths:
                raw_metadata = self._get_raw_metadata(raw_path)

                if raw_metadata is None:
                    bar.message(
                        f"  Skipped file {raw_path} (metadata format not supported)"
                    )
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
                        bar.end()
                        return

                    experiment = exp_data

                    # Load H5 and event files before metadata checks
                    # (Checks take some time so better to not do them if possible)
                    h5_paths = list(exp_path.glob("[!.]?*.h5"))

                    if len(h5_paths) > 1:
                        bar.end()

                        logger.warning(
                            f"There is more than one H5 file in this experiment folder. {CROSS}"
                        )

                        for path in h5_paths:
                            relative_path = path.relative_to(self.main_folder).resolve()
                            logger.warning(f"  {relative_path}")

                        logger.warning("Experiment will not be added to the DB.")
                        return

                    h5_data = (
                        _get_h5_metadata(h5_paths[0], experiment["exp_start"])
                        if h5_paths
                        else None
                    )

                    event_files = sorted(
                        exp_path.rglob("[!.]?*Events.csv"),
                        key=lambda x: x.stat().st_mtime,
                    )

                    bar.message(f"Found {len(event_files)} olfactometer event files.")

                elif last_exp_data != exp_data:
                    checks_failed += 1

                    bar.message(
                        f"  '{raw_path.relative_to(self.main_folder)}' "
                        "metadata is inconsistent with the previous acquisition."
                    )

                last_exp_data = exp_data
                acquisitions.append(acq)
                bar.step()

            bar.end()

            if checks_failed > 0:
                logger.warning(
                    f"TIFF metadata changed {checks_failed} or more times in the raw folder. {CROSS}"
                )
                logger.info("Are there multiple loops or grabs in the same folder?")
                logger.warning("Experiment will not be added to the DB.")
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

            # Acquisitions <-> H5 trials (same imaging clock, tight tolerance)
            # Result: h5_idx -> (acq_idx, delta_ms)
            acq_to_h5: dict[int, tuple[int, float]] = {}

            if h5_data and acquisitions:
                acq_to_h5 = _match_acq_to_h5(acquisitions, h5_data)

            # Pool all event trial starts across programs (keeping program/trial indices)
            # event_trial_pool[i] = (program_idx, trial_idx, trial_start)
            event_trial_pool: list[tuple[int, int, datetime]] = [
                (p_idx, t_idx, trial["trial_start"])
                for p_idx, p in enumerate(programs_data)
                for t_idx, trial in enumerate(p["trials"])
            ]

            # Event trials <-> H5 trials (different clocks, find constant offset)
            # Result: pool_idx -> (h5_idx, timedelta_ms)
            event_to_h5: dict[int, tuple[int, float]] = {}

            if h5_data and event_trial_pool:
                event_starts = [x[2] for x in event_trial_pool]
                event_to_h5 = _match_events_to_h5(event_starts, h5_data)

            # Build lookup: (p_idx, t_idx) -> (h5_idx, timedelta_ms)
            trial_to_h5: dict[tuple[int, int], tuple[int, float]] = {
                (event_trial_pool[pool_idx][0], event_trial_pool[pool_idx][1]): (
                    h5_idx,
                    timedelta_ms,
                )
                for pool_idx, (h5_idx, timedelta_ms) in event_to_h5.items()
            }

            # --------------------------------------------------------------- #
            # Phase 3: Insert
            # --------------------------------------------------------------- #

            exp_id = _db_insert(cur, "experiments", experiment)

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
                    acq = {
                        **acquisitions[acq_idx],
                        "exp_id": exp_id,
                        "odor_start": _to_datetime(h5_data["odor_starts"][h5_idx]),
                        "odor_end": _to_datetime(h5_data["odor_ends"][h5_idx]),
                        "h5_timedelta_ms": delta_ms,
                    }
                    h5_to_acq_id[h5_idx] = _db_insert(cur, "acquisitions", acq)
                    matched_acq_indices.add(acq_idx)

            # Fallback: insert acquisitions with no matching H5 trial
            for acq_idx, acq in enumerate(acquisitions):
                if acq_idx not in matched_acq_indices:
                    _db_insert(cur, "acquisitions", {**acq, "exp_id": exp_id})

            # Insert programs, trials, and events
            for p_idx, p_data in enumerate(programs_data):
                program_id = _db_insert(
                    cur, "programs", {**p_data["meta"], "exp_id": exp_id}
                )

                # Pass 1: insert trials, collect trial_ids by index
                trial_ids: dict[int, int] = {}

                for t_idx, trial in enumerate(p_data["trials"]):
                    acq_id = None
                    acq_timedelta_ms = None

                    if (p_idx, t_idx) in trial_to_h5:
                        h5_idx, timedelta_ms = trial_to_h5[(p_idx, t_idx)]
                        acq_id = h5_to_acq_id.get(h5_idx)
                        if acq_id is not None:
                            acq_timedelta_ms = timedelta_ms

                    trial_ids[t_idx] = _db_insert(
                        cur,
                        "trials",
                        {
                            "trial_start": trial["trial_start"],
                            "odor_start": trial["odor_start"],
                            "odor_end": trial["odor_end"],
                            "odor_id": trial["odor_id"],
                            "outcome": trial["outcome"],
                            "acq_id": acq_id,
                            "acq_timedelta_ms": acq_timedelta_ms,
                            "program_id": program_id,
                            "exp_id": exp_id,
                        },
                    )

                # Pass 2: insert all events in timeline order
                # trial_ids.get(trial_idx) returns None for orphans and
                # unfinalized last trials, preserving timeline order.
                for trial_idx, event in p_data["events"]:
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
                    logger.warning(
                        f"{n_h5 - n_matched_acq} H5 trials without matching acquisition. {CROSS}"
                    )
                else:
                    logger.info(f"All H5 trials matched to acquisitions. {CHECK}")

            if event_trial_pool:
                n_events = len(event_trial_pool)
                n_matched = len(event_to_h5)

                if n_matched < n_events:
                    logger.info(f"{n_events - n_matched} trials without H5 match.")

                n_with_acq = sum(
                    1
                    for (h5_idx, _) in trial_to_h5.values()
                    if h5_to_acq_id.get(h5_idx) is not None
                )

                if n_with_acq < n_matched:
                    logger.warning(
                        f"{n_matched - n_with_acq} trials matched to H5 but no acquisition. {CROSS}"
                    )
                elif event_files:
                    logger.info(f"All matched trials have acquisitions. {CHECK}")

            self._reset_caches()

    @record_call
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
        experiments: defaultdict[str, list[Path]] = defaultdict(list)

        for raw_path in raw_paths:
            exp_path = raw_path.parent.parent

            # Path are relative to main_folder to be computer independent
            # This makes the DB method_call parameters reusable

            rel_path = str(exp_path.relative_to(self.main_folder))
            raw_path_rel = str(raw_path.relative_to(self.main_folder))
            experiments[rel_path].append(raw_path_rel)

        # Add experiments to the database
        for path in experiments:
            try:
                self.add_experiment(rel_path=path, rel_raw_paths=experiments[path])
            except Exception:
                logger.exception("Failed to add experiment")

        logger.info("Database updated!")

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


def _load_event_data(
    event_files: list[Path],
    program_starts: list[tuple[datetime, str]],
    odors: dict[str, int],
    main_folder: Path,
) -> list[dict]:
    """
    Parse all event files into structured program/trial/event dicts.

    Returns a list of program dicts, each containing:
        "meta":   fields for the programs table (no exp_id yet)
        "trials": list of trial dicts (no events key)
        "events": list of (trial_idx, event_record) in timeline order.
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

        meta = {
            "program_name": program_name,
            "program_type": program_type,
            "program_start": program_start,
            "program_path": str(event_file.relative_to(main_folder)),
        }

        df = _parse_event_file(event_file, program_start)

        trials: list[dict] = []
        events: list[tuple[Optional[int], InsertData]] = []

        trial: Optional[dict] = None
        current_trial_idx: Optional[int] = None
        trial_phase = TrialPhase.NOT_IN_TRIAL
        licks_count = 0

        for _, (et, event_name, event_type, event_tag) in df.iterrows():
            event_time: datetime = et.to_pydatetime()

            # Skip session start events
            if event_type == "Session":
                continue

            # Build event record for storage
            event_record: InsertData = {
                "event_time": event_time,
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
                    logger.warning(f"Unexpected event: {event_name}")
                    logger.warning("Experiment will not be added to the DB.")
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
                assert trial is not None, "Trial end without trial"
                trial_phase = TrialPhase.TRIAL_END

                if program_type != "passive" and trial["outcome"] != "hit":
                    trial["outcome"] = "miss" if licks_count < 3 else "false choice"
                licks_count = 0

            # Record in timeline order. current_trial_idx is None before the first
            # trial, or points to a trial that may not be finalized yet (last trial
            # that didn't end) — in that case the insertion step maps it to NULL.
            events.append((current_trial_idx, event_record))

        # Finalize last trial only if it ended successfully.
        # Otherwise current_trial_idx points to an index not in trials,
        # and those events get inserted with trial_id = None.
        if trial is not None and trial_phase == TrialPhase.TRIAL_END:
            trials.append(trial)

        programs.append(
            {
                "meta": meta,
                "trials": trials,
                "events": events,
            }
        )

    return programs


def _match_acq_to_h5(
    acquisitions: list[InsertData],
    h5_data: dict[str, np.ndarray],
) -> dict[int, tuple[int, float]]:
    """
    Match acquisitions to H5 trials by nearest timestamp (same imaging clock).

    Returns dict: h5_idx -> (acq_idx, delta_ms)
    """
    h5_times = [_to_datetime(t) for t in h5_data["trial_starts"]]
    matches: dict[int, tuple[int, float]] = {}
    h5_ptr = 0

    for acq_idx, acq in enumerate(acquisitions):
        acq_start: datetime = acq["acq_start"]

        # Advance h5 pointer past trials clearly before this acquisition
        while (
            h5_ptr < len(h5_times) - 1 and h5_times[h5_ptr] < acq_start - H5_TOLERANCE
        ):
            h5_ptr += 1

        if h5_ptr < len(h5_times):
            delta = acq_start - h5_times[h5_ptr]
            if abs(delta) < H5_TOLERANCE:
                matches[h5_ptr] = (acq_idx, delta / TIMEDELTA_MS)
                h5_ptr += 1  # each H5 trial matched at most once

    return matches


def _match_events_to_h5(
    event_starts: list[datetime],
    h5_data: dict[str, np.ndarray],
) -> dict[int, tuple[int, float]]:
    """
    Find the best alignment of event trial starts with H5 trial starts.

    The olfactometer and imaging clocks have a constant offset (possibly seconds).
    This function searches for the starting position k in the event list such that
    event_starts[k:k+n_h5] best aligns with h5 trial starts (minimizing std of
    pairwise differences). Unmatched events at the boundaries are left out.

    acq_timedelta_ms stores the signed raw difference (event_time - h5_time) in ms,
    so the clock offset is recoverable via e.g. median(acq_timedelta_ms).

    Returns dict: pool_idx -> (h5_idx, timedelta_ms)
    """
    h5_times = [_to_datetime(t) for t in h5_data["trial_starts"]]
    n_h5 = len(h5_times)
    n_events = len(event_starts)

    if n_events == 0 or n_h5 == 0:
        return {}

    if n_events < n_h5:
        logger.warning(
            f"Fewer trial starts in the CSV files ({n_events})"
            f" than in the H5 file ({n_h5}). Matching as many as possible."
        )
        n_h5 = n_events

    # Use offsets from the first H5 trial for numerical stability
    base = h5_times[0]
    h5_ms = np.array([(t - base).total_seconds() * 1000 for t in h5_times[:n_h5]])
    event_ms = np.array([(t - base).total_seconds() * 1000 for t in event_starts])

    best_k = 0
    best_std = float("inf")

    for k in range(n_events - n_h5 + 1):
        std = float(np.std(event_ms[k : k + n_h5] - h5_ms))
        if std < best_std:
            best_std = std
            best_k = k

    diffs = event_ms[best_k : best_k + n_h5] - h5_ms

    logger.info(
        f"Average clock offset (event - h5):"
        f" {float(np.mean(diffs)):.1f} ms, std: {best_std:.1f} ms"
    )

    # Store signed raw difference so offset is recoverable from a query
    return {best_k + j: (j, float(diffs[j])) for j in range(n_h5)}


def _db_insert(
    cur: Cursor, table_name: str, data: InsertData | list[InsertData]
) -> Optional[int]:
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


def _to_datetime(dt: np.datetime64) -> datetime:
    dt_str = np.datetime_as_string(dt).item()
    return datetime.fromisoformat(dt_str)


def _get_h5_metadata(path: Path, exp_start: str) -> Optional[dict[str, np.ndarray]]:
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
