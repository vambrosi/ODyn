# --------------------------------------------------------------------------- #
# TODO: - Fix the database updating (shouldn't insert rows again)
#       - Use logging module instead of print (console + log file)
# --------------------------------------------------------------------------- #

import sqlite3

from pathlib import Path
from datetime import date, datetime
from tifffile import TiffFile

import pandas as pd

from .groups import Group
from .utils import ProgressBar, INFO, FAIL, PASS, CHECK, CROSS


class Database:
    """
    \033[1;35mDATABASE\033[0m
    Creates and connects you to the database.

    \033[1;34mUSAGE\033[0m
        db = Database(main_folder)

    \033[1;34mRELEVANT PROPRIETIES/METHODS\033[0m
        db.groups          <- Return all Groups for processing/analysis
        db.experiments     <- Returns DataFrame with experiment metadata
        db.raw_files       <- Returns DataFrame with raw files metadata
        db.mcor_files      <- Returns DataFrame with mcor files metadata

    Run Database.help('method_name') to know more about one of the methods above.

    \033[1;34mEXAMPLE\033[0m
        Database.help('experiment')
    """

    def __init__(self, path: str | Path, force_update=False):
        self.main_folder = Path(path)
        self.path = self.main_folder / ".odyn" / "odyn.db"

        # Initialize "private" variables
        self._groups = None
        self._experiments = None
        self._raw_files = None
        self._mcor_files = None
        self._method_calls = None

        # Get connection or create database if needed
        if not self.path.exists():
            print(f"{INFO} Did not find a database!")
            print(f"{INFO} Creating database...")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path)

            self.update()

            print(f"{INFO} Database created at the following location:")
            print(f"{INFO}    {self.path.resolve()}")

        elif force_update:
            print(f"{INFO} Updating the database...")

            self.con = sqlite3.connect(self.path)
            self.update()

            print(f"{INFO} Database updated!")
            print(f"{INFO} Path to the database:")
            print(f"{INFO}    {self.path.resolve()}")

        else:
            self.con = sqlite3.connect(self.path)
            print(f"{INFO} Connected to the database at:")
            print(f"{INFO}    {self.path.resolve()}")

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

        return print(f"{INFO} Method not found!")

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
    def experiments(self) -> pd.DataFrame:
        query = "SELECT * FROM experiments;"
        self._experiments = pd.read_sql_query(query, self.con)
        self._experiments.set_index("exp_id", inplace=True)

        return self._experiments

    @property
    def raw_files(self) -> pd.DataFrame:
        query = "SELECT * FROM raw_files;"
        self._raw_files = pd.read_sql_query(query, self.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._raw_files

    @property
    def mcor_files(self) -> pd.DataFrame:
        query = "SELECT * FROM mcor_files;"
        self._mcor_files = pd.read_sql_query(query, self.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._mcor_files

    @property
    def method_calls(self) -> pd.DataFrame:
        if self._method_calls is not None:
            return self._method_calls

        query = "SELECT * FROM method_calls;"

        self._method_calls = pd.read_sql_query(query, self.con)
        self._method_calls.set_index("method_call_id", inplace=True)

        return self._method_calls

    def _reset_caches(self) -> None:
        self._experiments = None
        self._raw_files = None
        self._mcor_files = None
        self._method_calls = None

        if self._groups is None:
            return

        # In case a specific group can still be accessed
        for group in self._groups:
            group._experiments = None
            group._raw_files = None
            group._mcor_files = None
            group._method_calls = None

        self._groups = None

    def update(self) -> None:
        # ----------------------------------------------------------------------- #
        # Gather all raw file paths and their metadata
        # ----------------------------------------------------------------------- #
        #
        # Collect all TIFF files that satisfy:
        #   1) File is inside a "raw" folder that don't start with a '.'
        #   2) File was made by scanimage (and has metadata)
        #
        # We will log all files that satisfy (1) but not (2), and we will
        # group acquisitions with the same loop start time.
        #
        # I used preProcessing_v2.m and other scripts in that file as a baseline
        # for what metadata has to be collected, and what needs to be checked.
        #
        # TODO: 1) Make sure all relevant data is added to the db.
        #       2) Only get metadata of new experiments, unless "forced = True".
        #
        # ----------------------------------------------------------------------- #

        self.con.execute("PRAGMA foreign_keys = ON;")

        print(f"{INFO} Searching for raw files ('**/raw/*.tif')...")

        raw_paths = sorted(self.main_folder.rglob("raw/[!.]?*.tif"))
        assert raw_paths, f"Found no .tif files in: {raw_path.resolve()}"

        print(f"{INFO} Found {len(raw_paths)} raw TIFF files.")

        # Experiments are sets of acquisitions with the same loop start time
        experiments = {}

        # Add experiments and acquisitions
        print(f"{INFO} Getting metadata from files...")
        checks_failed = 0

        bar = ProgressBar(len(raw_paths))
        bar.show()

        for raw_path in raw_paths:
            file_stem_parts = raw_path.stem.split("_")
            tif = TiffFile(raw_path)

            if not tif.is_scanimage:
                bar.message(f"{INFO}   Skipped file {raw_path} (not a ScanImage TIFFs)")
                continue

            # Get file SI metadata
            SI_metadata = tif.scanimage_metadata["FrameData"]
            laser_powers = SI_metadata["SI.hBeams.powers"]

            # Get the data that must be the same across the experiment
            exp_data = {
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
            exp_data["width_um"] = exp_data["width_px"] * factor_x
            exp_data["height_um"] = exp_data["height_px"] * factor_y

            # Parse ImageDescription
            image_description = dict(
                line.split(" = ")
                for line in tif.pages[0].tags["ImageDescription"].value.splitlines()
            )

            # Data specific to the acquisition
            acq = {
                "raw_path": str(raw_path.relative_to(self.main_folder)),
                "first_frame_start_s": float(image_description["frameTimestamps_sec"]),
            }

            # Parse ImageDescription epoch as a datetime
            date_string = image_description["epoch"].strip("[]")
            date_string = " ".join(date_string.split())
            dt = datetime.strptime(date_string, "%Y %m %d %H %M %S.%f")

            loop_start = dt.isoformat(" ")
            exp_data["exp_start"] = loop_start

            # Add group if this is the first acquisition
            # NOTE: We name the experiment using the first acquisition stem
            if loop_start not in experiments:
                experiments[loop_start] = {
                    "data": exp_data,
                    "acquisitions": [acq],
                }

            else:
                # Check if metadata is consistent across acquisitions
                if exp_data != experiments[loop_start]["data"]:
                    checks_failed += 1

                    bar.message(
                        f"{FAIL}   {acq["raw_path"]} metadata is "
                        "inconsistent with the first experiment acquisition."
                    )

                experiments[loop_start]["acquisitions"].append(acq)

            bar.step()

        bar.end()

        if checks_failed > 0:
            print(f"{FAIL} Failed {checks_failed} checks. {CROSS}")
        else:
            print(f"{PASS} Passed all checks! {CHECK}")

        # ----------------------------------------------------------------------- #
        # Create SQL DB and store metadata
        # ----------------------------------------------------------------------- #

        with self.con:
            # Create the database with the specified format
            with open(Path(__file__).parent / "create.sql") as f:
                self.con.executescript(f.read())

            cur = self.con.cursor()

            for exp in experiments.values():
                # Add experiment to the database
                insertion_query = _create_insertion_query("experiments", exp["data"])
                cur.execute(insertion_query, exp["data"])

                # Store exp_id for later
                exp_id = cur.lastrowid

                # Add a group for each experiment
                cur.execute("INSERT INTO groups DEFAULT VALUES;")
                group_id = cur.lastrowid

                insertion_query = """
                    INSERT INTO group_experiments
                        ( group_id
                        , exp_id
                        ) VALUES (?, ?);
                """
                cur.execute(insertion_query, [group_id, exp_id])

                # Add raw acquisition files in the exp loop
                for acq in exp["acquisitions"]:
                    acq["exp_id"] = exp_id

                    insertion_query = _create_insertion_query("raw_files", acq)
                    cur.execute(insertion_query, acq)

            # Make sure all properties will be recomputed
            self._groups = None
            self._experiments = None
            self._raw_files = None
            self._mcor_files = None

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


def _create_insertion_query(table_name, data):
    # HACK: ONLY FOR INTERNAL USE (CAN BE USED FOR SQL INJECTION)
    #           Column_names are not validated

    template = data[0] if isinstance(data, list) else data

    return (
        f"INSERT INTO {table_name} "
        f"({", ".join(template.keys())}) "
        f"VALUES (:{", :".join(template.keys())});"
    )
