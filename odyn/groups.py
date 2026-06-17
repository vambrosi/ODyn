# --------------------------------------------------------------------------- #
#
# TODO:
#   - TEST PLAY MOVIE!
#   - Change save directory default?
#   - Make list of files in play_movie optional?
#   - Add movies to outputs
#   - Add support for files (outputs) not in server (computer prefix?)
#   - Integrate docstrings with default values, to avoid copy-paste.
#   - Split docstrings and help functions to make VSCode hints usable.
#   - Remove added stuff in docstrings
#   - Use validated defaults on docstrings?
#   - Add help for expected file name and folder structure?
#   - Add delete_temp_files reminder in run_motion_correction
#   - Update raw_mmap_pairs when did final mcor?
#
# MAYBE TODO:
#   - Add pipeline function/classes to facilitate data analysis:
#       - Convolution layers (moving weighted averages)
#       - Thresholding (entrywise biased Heaviside or ReLU functions)
#       - All the steps in the MATLAB segmentation GUI?
#
# --------------------------------------------------------------------------- #

from __future__ import annotations

import json

from dataclasses import dataclass
from typing import cast, Final, TYPE_CHECKING

from pathlib import Path

import pandas as pd
import tifffile

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir

from .utils import *
from .utils import CallFrame

if TYPE_CHECKING:
    from .database import Database


class Group:
    """
    \033[1;35mGROUP\033[0m
    Class that runs data processing/analysis.

    \033[1;34mUSAGE\033[0m
        db  = Database(main_folder)
        group = db.groups[some_index]

    \033[1;34mRELEVANT METHODS\033[0m
        group.run_motion_correction(...)
        group.play_movie(...)
        group.delete_temp_files()

    Run Group.help('method_name') to know more about one of the methods above.

    \033[1;34mEXAMPLE\033[0m
        Group.help('play_movie')
    """

    is_first = True

    def __init__(self, group_id: int, db: Database) -> None:
        self.group_id: Final[int] = group_id
        self.db = db

        # Initialize "private" variables
        self._acquisitions: None | pd.DataFrame = None
        self._events: None | pd.DataFrame = None
        self._experiments: None | pd.DataFrame = None
        self._mcor_files: None | pd.DataFrame = None
        self._method_calls: None | pd.DataFrame = None
        self._programs: None | pd.DataFrame = None
        self._trials: None | pd.DataFrame = None

        self._call_stack: list[CallFrame] = []
        self._raw_mmap_pairs: None | tuple[list[str], list[str]] = None
        self.movies: dict[tuple[MovieType, ...], LazyMovie] = {}

        if Group.is_first:
            Group.short_help()
            Group.is_first = False

    def __repr__(self):
        msg = f"Group {self.group_id}"

        # Show the experiment name if there is only one
        if len(self.experiments) == 1:
            msg += f" (exp_name = {self.experiments["exp_name"].iloc[0]})"

        return msg

    @staticmethod
    def help(name="Group"):
        if name.lower() == "group":
            return print(Group.__doc__)

        attr = getattr(Group, name, None)

        if attr is not None:
            return print(attr.__doc__)

        return logger.info("Method not found!")

    @staticmethod
    def short_help():
        logger.info("Run Group.help() to get a list of useful functions.")
        logger.info("Run Group.help('function_name') to know more about a function.")

    # ----------------------------------------------------------------------- #
    # Database Interaction
    # ----------------------------------------------------------------------- #

    @property
    def acquisitions(self) -> pd.DataFrame:
        if self._acquisitions is not None:
            return self._acquisitions

        query = f"""
            SELECT a.* FROM group_experiments AS g
                JOIN experiments  AS e ON e.exp_id = g.exp_id
                JOIN acquisitions AS a ON a.exp_id = e.exp_id
                WHERE g.group_id = {self.group_id};
        """

        self._acquisitions = pd.read_sql_query(
            query, self.db.con, parse_dates=["acq_start", "odor_start", "odor_end"]
        )
        self._acquisitions.set_index("acq_id", inplace=True)

        return self._acquisitions

    @property
    def events(self) -> pd.DataFrame:
        if self._events is not None:
            return self._events

        query = f"""
            SELECT e.* FROM group_experiments AS g
                JOIN experiments AS x ON x.exp_id     = g.exp_id
                JOIN programs    AS p ON p.exp_id     = x.exp_id
                JOIN events      AS e ON e.program_id = p.program_id
                WHERE g.group_id = {self.group_id};
        """

        self._events = pd.read_sql_query(query, self.db.con)
        self._events.set_index("event_id", inplace=True)

        return self._events

    @property
    def experiments(self) -> pd.DataFrame:
        if self._experiments is not None:
            return self._experiments

        query = f"""
            SELECT e.* FROM group_experiments AS ge
                JOIN experiments AS e ON e.exp_id = ge.exp_id
                WHERE ge.group_id = {self.group_id};
        """
        self._experiments = pd.read_sql_query(
            query, self.db.con, parse_dates=["exp_start", "added_to_db_at"]
        )
        self._experiments.set_index("exp_id", inplace=True)

        return self._experiments

    @property
    def mcor_files(self) -> pd.DataFrame:
        if self._mcor_files is not None:
            return self._mcor_files

        query = f"""
            SELECT m.* FROM group_experiments AS g
                JOIN experiments  AS e ON e.exp_id = g.exp_id
                JOIN acquisitions AS a ON a.exp_id = e.exp_id
                JOIN mcor_files   AS m ON m.acq_id = a.acq_id
                WHERE g.group_id = {self.group_id};
        """

        self._mcor_files = pd.read_sql_query(query, self.db.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._mcor_files

    @property
    def method_calls(self) -> pd.DataFrame:
        if self._method_calls is not None:
            return self._method_calls

        query = f"""
            SELECT mc.* FROM group_experiments AS ge
                JOIN method_calls AS mc ON ge.group_id = mc.group_id
                WHERE ge.group_id = {self.group_id};
        """

        self._method_calls = pd.read_sql_query(
            query, self.db.con, parse_dates=["called_at"]
        )
        self._method_calls.set_index("method_call_id", inplace=True)
        self._method_calls["parameters"] = self._method_calls["parameters"].apply(
            json.loads
        )
        self._method_calls["call_output"] = self._method_calls["call_output"].apply(
            lambda s: json.loads(s) if isinstance(s, str) else None
        )

        return self._method_calls

    @property
    def programs(self) -> pd.DataFrame:
        if self._programs is not None:
            return self._programs

        query = f"""
            SELECT p.* FROM group_experiments AS g
                JOIN experiments AS e ON e.exp_id = g.exp_id
                JOIN programs    AS p ON p.exp_id = e.exp_id
                WHERE g.group_id = {self.group_id};
        """

        self._programs = pd.read_sql_query(
            query, self.db.con, parse_dates=["program_start"]
        )
        self._programs.set_index("program_id", inplace=True)

        return self._programs

    @property
    def trials(self) -> pd.DataFrame:
        if self._trials is not None:
            return self._trials

        query = f"""
            SELECT t.* FROM group_experiments AS g
                JOIN experiments AS x ON x.exp_id     = g.exp_id
                JOIN programs    AS p ON p.exp_id     = x.exp_id
                JOIN trials      AS t ON t.program_id = p.program_id
                WHERE g.group_id = {self.group_id};
        """

        self._trials = pd.read_sql_query(
            query, self.db.con, parse_dates=["trial_start", "odor_start", "odor_end"]
        )
        self._trials.set_index("trial_id", inplace=True)

        return self._trials

    @property
    def current_call_id(self) -> int:
        assert self._call_stack, (
            "Empty call stack — 'current_call_id' is only available inside a "
            "method decorated with '@record_call'."
        )
        return self._call_stack[-1].call_id

    def add_flag(self, flag) -> None:
        """Set bits on the current call's flag (bitwise OR). Use inside @record_call."""
        self._call_stack[-1].flag |= int(flag)

    def set_output(self, output: Object) -> None:
        """Record this call's output as JSON. Last write wins."""
        self._call_stack[-1].output = output

    def fail(self, flag, message: str = "") -> None:
        """Flag the current call and abort it by raising RuntimeError."""
        self.add_flag(flag)
        raise RuntimeError(message)

    def latest_output(self, method_name: str) -> None | Object:
        """Return the parsed output of the most recent call to 'method_name'."""
        row = self.db.con.execute(
            """
            SELECT call_output FROM method_calls
             WHERE group_id = ? AND method_name = ? AND call_output IS NOT NULL
             ORDER BY method_call_id DESC LIMIT 1
            """,
            [self.group_id, method_name],
        ).fetchone()
        return json.loads(row["call_output"]) if row else None

    def _reset_caches(self) -> None:
        self._acquisitions = None
        self._events = None
        self._experiments = None
        self._mcor_files = None
        self._method_calls = None
        self._programs = None
        self._trials = None

        self.db._acquisitions = None
        self.db._events = None
        self.db._experiments = None
        self.db._mcor_files = None
        self.db._method_calls = None
        self.db._programs = None
        self.db._trials = None

    # ----------------------------------------------------------------------- #
    # Motion Correction functions
    # ----------------------------------------------------------------------- #

    @record_call
    def pick_mcor_parameters(self) -> dict:
        """
        Open a GUI to pick motion-correction parameters and record them.

        The picked values are stored so
            run_motion_correction(use_gui_parameters=True)
        can read them back via latest_output("Group.pick_mcor_parameters").
        """

        # TODO: integrate the GUI prototype here (blocks until "Pick").
        params: dict = {}

        self.set_output(params)
        return params

    @memorize_params
    @record_call
    def run_motion_correction(
        self,
        *,
        use_gui_parameters: bool = True,
        is_test: bool = True,
        first_acq: int = 0,
        step_acq: int = 1,
        last_acq: int = 3,
        border_nan: bool | str = "copy",
        nonneg_movie: bool = False,
        pw_rigid: bool = True,
        shifts_opencv: bool = False,
        max_deviation_um: float = 12.0,
        max_shift_um: list[float] = [128.0, 128.0],
        overlap_um: list[float] = [96.0, 96.0],
        strides_um: list[float] = [128.0, 128.0],
    ) -> None:
        """
        \033[1;35mRUN_MOTION_CORRECTION\033[0m
        Method that does test/final motion correction

        \033[1;34mUSAGE\033[0m
            db    = Database(main_folder)
            group = db.groups[some_index]
            group.run_motion_correction(...)

        \033[1;34mLIST OF PARAMETERS\033[0m (WITH DEFAULT VALUES)

            \033[0;32mBasic Parameters\033[0m
            use_gui_parameters  = True              Override parameters with latest pick_mcor_parameters() values
            use_last_parameters = False             Use parameters from last run as the defaults
            is_test             = True              Whether to use a limited range of acquisitions in this run

            \033[0;32mParameters in this section will be ignored if is_test == False\033[0m
            first_acq           = 0                 Index of the first acquisition to motion correct
            step_acq            = 1                 Get one acquisition for every 'step_acq' acquisitions
            last_acq            = 3                 Index of the last acquisition to motion correct

            \033[0;32mCaImAn motion correction parameters (before metadata adjustments)\033[0m
            border_nan          = "copy"            copy along the boundary (if True, fill in with NaN)
            nonneg_movie        = False             make SAVED movie mostly non-negative
            pw_rigid            = True              Piecewise-rigid (True) or rigid motion correction
            shifts_opencv       = False             True = bicubic, False = FFT (True is faster)
            max_deviation_um    = 12.0              max deviation for patch with respect to rigid shifts
            max_shift_um        = [128.0, 128.0]    max allowed rigid shift
            overlap_um          = [96.0, 96.0]      overlap between patches (patch = strides + overlaps)
            strides_um          = [128.0, 128.0]    start a new patch every x or y um (only for pw-rigid)

        \033[1;34mEXAMPLES\033[0m
            group.run_motion_correction(is_test=True, last_acq=10)
        """

        # --- Optionally override parameters with GUI-picked values --- #
        # These override both the passed arguments and use_last_parameters.

        if use_gui_parameters:
            gui = self.latest_output("Group.pick_mcor_parameters")

            if gui is None:
                logger.warning(
                    "use_gui_parameters=True but no saved parameters found; "
                    "run pick_mcor_parameters() first. Using provided values."
                )
            else:
                logger.info("Loaded motion-correction parameters from the GUI.")
                max_deviation_um = cast(
                    float, gui.get("max_deviation_um", max_deviation_um)
                )
                max_shift_um = cast(list[float], gui.get("max_shift_um", max_shift_um))
                overlap_um = cast(list[float], gui.get("overlap_um", overlap_um))
                strides_um = cast(list[float], gui.get("strides_um", strides_um))

        # --- Validate and adjust parameters to reasonable values --- #

        # If is not test, include all acquisitions
        if not is_test:
            first_acq = 0
            step_acq = 1
            last_acq = len(self.acquisitions) - 1

        # max_shift_um has to less than size of image (in μm / 4)
        height_um = self.experiments["height_um"].min()
        width_um = self.experiments["width_um"].min()
        max_shift_um[0] = clamp(max_shift_um[0], 0, height_um / 4)
        max_shift_um[1] = clamp(max_shift_um[1], 0, width_um / 4)

        # --- Make sure movies will be updated next time they are played --- #

        if is_test:
            for movie_types in self.movies.keys():
                if MovieType.TEST in movie_types:
                    self.movies[movie_types].mark_as_outdated()
        else:
            for movie_types in self.movies.keys():
                if MovieType.MCOR in movie_types:
                    self.movies[movie_types].mark_as_outdated()

        # --- Get settings for CaImAn MotionCorrection class --- #

        # Get raw paths
        acquisitions_slice = self.acquisitions.iloc[first_acq : last_acq + 1 : step_acq]
        raw_paths = [self.db.main_folder / p for p in acquisitions_slice["raw_path"]]

        assert raw_paths, "No raw files within the index range."

        # CaImAn only uses pixels as units, so we make the conversion
        factor = [
            height_um / self.experiments["height_px"].min(),
            width_um / self.experiments["width_px"].min(),
        ]

        settings = {
            "border_nan": border_nan,
            "pw_rigid": pw_rigid,
            "shifts_opencv": shifts_opencv,
            "nonneg_movie": nonneg_movie,
            "max_deviation_rigid": int(max_deviation_um) / min(factor),
            "max_shifts": um_to_pixels(max_shift_um, factor),
            "overlaps": um_to_pixels(overlap_um, factor),
            "strides": um_to_pixels(strides_um, factor),
        }

        # --- Run the motion correction --- #

        logger.info("Starting motion correction...")

        _, dview, _ = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=None, single_thread=False
        )

        try:
            self.mc = MotionCorrect(raw_paths, dview=dview, **settings)
            self.mc.motion_correct(save_movie=True)

        except:
            raise

        # Always stop the server after motion correction
        finally:
            cm.stop_server(dview=dview)

        logger.info("Finished motion correction")

        # --- Save the results in TIFF files (if it is not a test) --- #
        if is_test:
            # Stores the list of raw and mmap paths to make a movie later
            self._raw_mmap_pairs = (
                [str(p) for p in raw_paths],
                self.mc.mmap_file.copy(),
            )
            return

        with self.db.con as con:
            logger.info("Saving mcor files...")

            # Load mmap files and save them as TIFFs
            # TODO: Confirm that mmap_file and fname have the same order
            for acq_id, mmap_path, raw_path in tqdm(
                zip(acquisitions_slice.index, self.mc.mmap_file, self.mc.fname),
                desc="Saving mcor files",
                total=len(self.mc.mmap_file),
            ):
                # Create folder if it doesn't exist
                mcor_folder = raw_path.parent.parent / "processed" / "mcor"
                mcor_folder.mkdir(parents=True, exist_ok=True)

                mcor_path = mcor_folder / (raw_path.stem + "_mcor.tif")

                mc = cm.load(mmap_path)

                # Saving TIFFs directly because caiman saves them as 64-bit
                with tifffile.TiffWriter(mcor_path) as tif:
                    tif.write(
                        [mc[i].copy() for i in range(mc.shape[0])],
                        shape=mc[0].shape,
                        dtype=mc.dtype,
                    )

                # Update the database with the latest mcor files
                insertion_query = """
                    INSERT OR REPLACE INTO mcor_files
                        ( acq_id
                        , mcor_path
                        , last_updated_by
                        ) VALUES (?, ?, ?);
                """
                con.execute(
                    insertion_query,
                    [
                        acq_id,
                        str(mcor_path.relative_to(self.db.main_folder)),
                        self.current_call_id,
                    ],
                )

            # Reset mcor DataFrames
            self._mcor_files = None
            self.db._mcor_files = None

    @memorize_params
    @record_call
    def play_movie(
        self,
        *,
        grid: list[str] = ["raw", "mcor"],
        downsample_ratio: float = 0.03,
        opencv_codec: str = "MJPG",
        save_movie: bool = True,
        save_folder: str = r"./movies",
        backend: str = "embed_opencv",
        do_loop: bool = False,
        fr: float = 30,
        magnification: float = 1,
        plot_text: bool = True,
        q_max: float = 99.5,
        q_min: float = 0.05,
    ) -> None:
        """
        \033[1;35mPLAY_MOVIE\033[0m
        Play and save movies for quality control

        \033[1;34mUSAGE\033[0m
            group = Group(experimentFolder)
            group.play_movie(...)

        \033[1;34mLIST OF PARAMETERS\033[0m (WITH DEFAULT VALUES)

            \033[0;32mBasic settings\033[0m
            use_last_parameters = False             Use parameters from last run as the defaults
            grid                = ["raw", "mcor"]   How to concatenate movies horizontally.
                                                    ("raw", "mcor", "test" in some order)

            \033[0;32mMovie loading settings\033[0m
            downsample_ratio    = 0.03              Percentage of frames to keep

            \033[0;32mVideo saving\033[0m
            opencv_codec        = "MJPG"            Codec used to encode the saved video
            save_movie          = True              Put "true" if you want to save the preview video to a file
            save_folder         = r"./movies"       "." is the main_folder (r is to use \\ in the path)

            \033[0;32mVideo settings\033[0m
            backend             = "embed_opencv"    "opencv" for popup and "embed_opencv" for inline player
            do_loop             = false             Loop the video or not
            fr                  = 30                How fast to play the video (frames/s)
            magnification       = 1                 Magnification of video
            plot_text           = true              Add current frame label on the video
            q_max               = 99.5              Quantile to consider as white
            q_min               = 0.05              Quantile to consider as black

        \033[1;34mEXAMPLES\033[0m
            group.play_movie(grid=["raw"], save_folder="~/TempData/20260101/e1/movies")
                Running this command would save a compilation of all raw movies

            group.play_movie(grid=["raw", "test"])
                This would save a video with raw movies on the left and test on the right
        """
        # --- Validate parameters --- #

        # Check grid input
        assert len(grid) == 1 or (
            len(grid) == 2
            and MovieType.RAW.value in grid
            and (MovieType.MCOR.value in grid or MovieType.TEST.value in grid)
        ), """The parameter 'grid' must be one of the following:
                ['raw'], ['mcor'], ['test'],
                ['raw', 'test'], ['raw', 'mcor'],
                ['test', 'raw'], ['mcor', 'raw']"""

        params = locals()

        # --- Check if movie needs to be updated --- #

        # Exclude the current call (already recorded by decorator) so we find
        # the most recent *previous* call with the same grid.
        play_movie_calls = self.method_calls[
            (self.method_calls["method_name"] == "Group.play_movie")
            & (self.method_calls.index != self.current_call_id)
        ]
        last_call_id = play_movie_calls[
            play_movie_calls["parameters"].apply(lambda x: x["grid"] == grid)
        ].index.max()

        # Trigger recompute if the parameters changed from the last call
        movie_types = tuple(MovieType(s) for s in grid)
        params.pop("self", None)

        if movie_types not in self.movies:
            self.movies[movie_types] = LazyMovie(self, movie_types)

        elif play_movie_calls.loc[last_call_id, "parameters"] != params:
            self.movies[movie_types].mark_as_outdated()

        # --- Transform parameters --- #

        # TODO: Change name and default folder
        movie_type_str = "_".join(t.value for t in movie_types)
        first_exp_name = self.experiments.iloc[0]["exp_name"]
        filename = f"Group_{self.group_id}_{first_exp_name}_{movie_type_str}.avi"

        filepath = (
            (self.db.main_folder / save_folder / filename).resolve()
            if save_folder[0] == "."
            else (Path(save_folder) / filename).resolve()
        )

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # --- Play and maybe update movies --- #
        # Only updated if run_motion_correction was called invalidating the
        # relevant files (raw or mcor TIFFs, or test MMAPs)
        movie_name = str(filepath)

        if save_movie:
            logger.info(f"Saving movie to {movie_name}")

        self.movies[movie_types].maybe_update(downsample_ratio).play(
            backend=backend,
            do_loop=do_loop,
            fr=fr,
            magnification=magnification,
            plot_text=plot_text,
            q_max=q_max,
            q_min=q_min,
            opencv_codec=opencv_codec,
            save_movie=save_movie,
            movie_name=movie_name,
        )

    def delete_temp_files(self) -> None:
        """
        \033[1;35mDELETE_TEMP_FILES\033[0m
        Deletes all temp files associated with this experiment

        \033[1;34mUSAGE\033[0m
            group = Group(experimentFolder)
            group.delete_temp_files()

        \033[1;34mEXAMPLES\033[0m
            group.delete_temp_files()
        """

        # Get file paths for mmaps in the temp folder
        path = Path(get_tempdir())

        exp_movies = []
        for exp_name in self.experiments["exp_name"]:
            exp_movies.append((exp_name, sorted(path.glob(f"{exp_name}*.mmap"))))

        # Remove files
        for exp_name, movie_paths in exp_movies:
            logger.info(f"Removing .mmap files that start with {exp_name}...")
            if movie_paths:
                total_size = 0.0  # in bytes
                for movie_path in movie_paths:
                    total_size += movie_path.stat().st_size
                    movie_path.unlink(missing_ok=True)

                total_size = total_size / (1_000_000_000)  # in GBs
                ending = "s" if len(movie_paths) > 1 else ""
                logger.info(
                    f"Deleted {len(movie_paths)} file{ending} ({total_size:.1f} GB)."
                )

            else:
                logger.info("No .mmap files found.")

        self._raw_mmap_pairs = None

    # ----------------------------------------------------------------------- #
    # Data Analysis
    # ----------------------------------------------------------------------- #

    # def compute_z_scores(self) -> cm.movie:
    #     # Check an sync config
    #     self._sync_config()

    #     # TODO: replace dummy values below
    #     odor_onset = 2
    #     odor_offset = 4
    #     # END OF DUMMY VALUES ------------

    #     # Get config parameters
    #     frame_rate = self.config["metadata"]["frame_rate"]
    #     post_odor_interval = self.config["z-scores"]["post_odor_interval"]
    #     baseline_pre_odor = self.config["z-scores"]["baseline_pre_odor"]

    #     frame_odor_onset = int(odor_onset * frame_rate)
    #     frame_onset = int((odor_onset + post_odor_interval) * frame_rate)
    #     frame_offset = int((odor_offset + post_odor_interval) * frame_rate)
    #     frame_duration = frame_offset - frame_onset + 1

    #     frame_baseline_start = (
    #         frame_odor_onset - frame_duration
    #         if baseline_pre_odor
    #         else frame_onset - frame_duration
    #     )

    #     mcor_folder = self.path / self.config["experiment"]["mcor_folder"]
    #     mcor_files = mcor_folder.glob(f"[!.]?*_mcor.tif")

    #     assert mcor_files, "No .tif files in the mcor folder."

    #     # TIFFs were originally in int16, so we add 32768 to make then non-negative
    #     baseline_range = range(frame_baseline_start, frame_duration)
    #     baseline_avg = cm.load(mcor_files[0], subindices=baseline_range).mean(axis=0)
    #     baseline_avg += 32768

    #     signal_range = range(frame_onset, frame_duration)
    #     signal_avg = cm.load(mcor_files[0], subindices=signal_range).mean(axis=0)
    #     signal_avg += 32768

    #     # Computes z-score of dF/F
    #     dFF = (signal_avg - baseline_avg) / baseline_avg
    #     z_score = (dFF - dFF.mean()) / dFF.std()

    #     return z_score


@dataclass
class LazyMovie:
    """
    Movies that only reload when needed.
    """

    owner: Group
    types: tuple[MovieType, ...]
    movie: None | cm.movie = None

    def mark_as_outdated(self):
        self.movie = None

    def maybe_update(self, downsample_ratio) -> cm.movie:
        if self.movie is not None:
            logger.info("Using cached movie...")
            return self.movie

        logger.info("Updating movie...")

        movie_chains = []

        if MovieType.TEST in self.types:
            # If test is included there must be test files
            assert (
                self.owner._raw_mmap_pairs is not None
            ), "There are no cached 'test' file paths."

        for movie_type in self.types:
            # Get the movie_paths
            if movie_type == MovieType.TEST:
                self.owner.method_calls

                # Redundant check for mypy
                assert self.owner._raw_mmap_pairs is not None
                _, movie_paths = self.owner._raw_mmap_pairs

            elif MovieType.TEST in self.types:
                # Redundant check for mypy
                assert self.owner._raw_mmap_pairs is not None

                # It must be raw in this case
                movie_paths, _ = self.owner._raw_mmap_pairs

            # If there is no test always include all files
            elif movie_type == MovieType.RAW:
                movie_paths = [
                    self.owner.db.main_folder / path
                    for path in self.owner.acquisitions["raw_path"]
                ]

            elif movie_type == MovieType.MCOR:
                movie_paths = [
                    self.owner.db.main_folder / path
                    for path in self.owner.mcor_files["mcor_path"]
                ]

            # There must be at least one movie
            # TODO: Gather movie_paths for every type first, so it fails
            #       before having spent time loading anything.
            assert movie_paths, f"Didn't find any {movie_type.value} files."

            logger.info(
                f"Adding {len(movie_paths)} {movie_type.value} files to the movie."
            )

            movie_chain = None
            for path in tqdm(movie_paths, desc=f"Loading {movie_type.value} movies"):
                movie = cm.load(path).resize(1, 1, downsample_ratio)
                movie_chain = (
                    movie
                    if movie_chain is None
                    else cm.concatenate([movie_chain, movie], axis=0)
                )
                logger.info(f"  {path}")

            movie_chains.append(movie_chain)

        movie_chain = cm.concatenate(movie_chains, axis=2)

        self.movie = movie_chain

        return self.movie


#     def play(self) -> None:
#         # TODO: - Use Caiman function as blueprint
#         #       - Add movie type, total time, current time, and speed (e.g. x2) to label
#         #       - Remove frame number from label
#         #       - Possibly add total time to config and remove fr in [play.load]
#         return
