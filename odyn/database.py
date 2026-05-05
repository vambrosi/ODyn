# --------------------------------------------------------------------------- #
# TODO: - Fix the database updating (shouldn't insert rows again)
#       - Use logging module instead of print (console + log file)
# --------------------------------------------------------------------------- #

import sqlite3

from pathlib import Path
from datetime import date, datetime
from tifffile import TiffFile

import pandas as pd

from .experiment import Experiment
from .utils import ProgressBar, INFO

# Print a helpful string when user imports this library
print(f"[{INFO}] Run Database.help() to get examples of how to use ODyn.")


class Database:
    """
    \033[1;31mDATABASE\033[0m
    Creates and connects you to the database.

    \033[1;34mUSAGE\033[0m
        db = Database(main_folder)

    \033[1;34mRELEVANT PROPRIETIES/METHODS\033[0m
        db.experiments          <- Returns DataFrame with experiment data
        db.experiment(exp_id)   <- Return an Experiment to do processing/analysis
        db.update()          <- Add new data to the database

    Run Database.help('method_name') to know more about one of the methods above.

    \033[1;34mEXAMPLE\033[0m
        Database.help('experiment')
    """

    def __init__(self, path: str | Path, force_update=False):
        self.main_folder = Path(path)
        self.path = self.main_folder / ".odyn" / "odyn.db"
        self._experiments = None

        # Get connection or create database if needed
        if not self.path.exists():
            print(f"[{INFO}] Did not find a database!")
            print(f"[{INFO}] Creating database...")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(self.path)

            self.update()

            print(f"[{INFO}] Database created at the following location:")
            print(f"[{INFO}]    {self.path.resolve()}")

        elif force_update:
            print(f"[{INFO}] Updating the database...")

            self.con = sqlite3.connect(self.path)
            self.update()

            print(f"[{INFO}] Database updated!")
            print(f"[{INFO}] Path to the database:")
            print(f"[{INFO}]    {self.path.resolve()}")

        else:
            self.con = sqlite3.connect(self.path)
            print(f"[{INFO}] Connected to the database at:")
            print(f"[{INFO}]    {self.path.resolve()}")

        self.con.execute("PRAGMA foreign_keys = ON;")
        self.con.row_factory = sqlite3.Row

    def experiment(self, exp_id: int):
        return Experiment(exp_id, self.con)

    @property
    def experiments(self) -> pd.DataFrame:
        if self._experiments is not None:
            return self._experiments

        COLUMN_NAMES = [
            "e.exp_id",
            "exp_name",
            "exp_type",
            "loop_start",
            "height_px",
            "width_px",
            "height_um",
            "width_um",
            "frame_count",
            "frame_rate",
            "laser_power_920",
            "laser_power_1040",
            "loop_acq_interval_s",
        ]

        query = f"""
            SELECT DISTINCT {", ".join(COLUMN_NAMES)}
                FROM raw_files
                JOIN exp_files ON raw_files.acq_id = exp_files.acq_id
                JOIN experiments AS e ON e.exp_id = exp_files.exp_id;
            """

        # Cache result
        self._experiments = pd.read_sql_query(query, self.con).astype(
            {"exp_name": str, "exp_type": "category", "loop_start": "datetime64[ms]"}
        )

        return self._experiments

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

        print(f"[{INFO}] Searching for raw files ('**/raw/*.tif')...")

        raw_paths = sorted(self.main_folder.rglob("raw/[!.]?*.tif"))
        assert raw_paths, f"Found no .tif files in: {raw_path.resolve()}"

        print(f"[{INFO}] Found {len(raw_paths)} raw TIFF files.")

        # Experiments are sets of acquisitions with the same loop start time
        experiments = {}

        # Add experiments and acquisitions
        print(f"[{INFO}] Getting metadata from files...")

        bar = ProgressBar(len(raw_paths))
        bar.show()

        for raw_path in raw_paths:
            tif = TiffFile(raw_path)

            if not tif.is_scanimage:
                bar.message(
                    f"[{INFO}]   Skipped file {raw_path} (not a ScanImage TIFFs)"
                )
                continue

            # Get file SI metadata
            SI_metadata = tif.scanimage_metadata["FrameData"]
            laser_powers = SI_metadata["SI.hBeams.powers"]

            # TODO: What is better? And why the redundancy?
            #           len(tif.pages) or
            #           SI_metadata["SI.hStackManager.framesPerSlice"]
            acq = {
                "raw_path": str(raw_path.relative_to(self.main_folder)),
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
            acq["width_um"] = acq["width_px"] * factor_x
            acq["height_um"] = acq["height_px"] * factor_y

            # Parse ImageDescription
            image_description = dict(
                line.split(" = ")
                for line in tif.pages[0].tags["ImageDescription"].value.splitlines()
            )

            acq["first_frame_start_s"] = float(image_description["frameTimestamps_sec"])

            # Parse ImageDescription epoch as a datetime
            date_string = image_description["epoch"].strip("[]")
            date_string = " ".join(date_string.split())
            dt = datetime.strptime(date_string, "%Y %m %d %H %M %S.%f")

            loop_start = dt.isoformat(" ")

            # Add group if this is the first acquisition
            # NOTE: We name the experiment using the first acquisition stem
            if loop_start not in experiments:
                file_stem_parts = raw_path.stem.split("_")

                experiments[loop_start] = {
                    "columns": {
                        "exp_name": "_".join(file_stem_parts[:-1]),
                        "exp_type": tif.scanimage_metadata["FrameData"]["SI.acqState"],
                        "loop_start": loop_start,
                    },
                    "acquisitions": [acq],
                }
            else:
                experiments[loop_start]["acquisitions"].append(acq)

            bar.step()

        bar.end()

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
                insertion_query = _create_insertion_query("experiments", exp["columns"])
                cur.execute(insertion_query, exp["columns"])

                # Store exp_id for later
                exp_id = cur.lastrowid

                for acq in exp["acquisitions"]:
                    # Add raw acquisition files in the exp loop
                    insertion_query = _create_insertion_query("raw_files", acq)
                    cur.execute(insertion_query, acq)

                    # Add pair exp_id and acq_id to the join table
                    acq_id = cur.lastrowid
                    cur.execute(
                        "INSERT INTO exp_files (exp_id, acq_id) VALUES (?, ?)",
                        [exp_id, acq_id],
                    )

            # Make sure the corresponding property will be recomputed
            self._experiments = None

    @staticmethod
    def help(name="Database"):
        if name.lower() == "database":
            return print(Database.__doc__)

        attr = getattr(Database, name, None)

        if attr is not None:
            return print(attr.__doc__)

        return print(f"[{INFO}] Method not found!")


def _create_insertion_query(table_name, data):
    # HACK: ONLY FOR INTERNAL USE (CAN BE USED FOR SQL INJECTION)
    #           Column_names are not validated

    template = data[0] if isinstance(data, list) else data

    return (
        f"INSERT INTO {table_name} "
        f"({", ".join(template.keys())}) "
        f"VALUES (:{", :".join(template.keys())});"
    )
