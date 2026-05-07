# --------------------------------------------------------------------------- #
#
# TODO: - Integrate docstrings with default values, to avoid copy-paste.
#       - Add pipeline function/classes to facilitate data analysis:
#           - Convolution layers (moving weighted averages)
#           - Thresholding (entrywise biased Heaviside or ReLU functions)
#           - All the steps in the MATLAB segmentation GUI?
#       - Add git hash to every db entry? (To help db updates...)
#       - Add support for use_last_parameters
#       - Add delete_temp_files reminder in run_motion_correction
#       - Add help for expected file name and folder structure?
#
# NOTE: There is a question of how to integrate everything into a unique db
#       while preserving the ability to copy files to do analysis off the
#       server or to process subsets of the data. This would require being
#       able to call functions on query results, and possibly make db
#       consolidation easy (preferably automatic). Which steps would be most
#       relevant would depend on how much of the code would run on each
#       computer vs directly on the server. All possibilities should be
#       feasible given the different workflows of people in the lab.
#
# Related TODO: - Make a function that creates .py file containing the whole
#                 processing/analysis pipeline, to run on the server.
#
# --------------------------------------------------------------------------- #

import re
import sqlite3

from typing import Optional, Iterable
from dataclasses import dataclass

from pathlib import Path

import tifffile
import pandas as pd

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir

from .utils import ProgressBar, um_to_pixels, clamp, MovieType, INFO


class Experiment:
    """
    \033[1;31mEXPERIMENT\033[0m
    Class that runs data processing/analysis.

    \033[1;34mUSAGE\033[0m
        exp = Experiment(experimentFolder)

    \033[1;34mRELEVANT METHODS\033[0m
        exp.run_motion_correction(...)
        exp.play_movie(...)
        exp.delete_temp_files()

    Run Experiment.help('method_name') to know more about one of the methods above.

    \033[1;34mEXAMPLE\033[0m
        Experiment.help('play_movie')
    """

    is_first = True

    def __init__(self, exp_id: int, main_folder: Path, con: sqlite3.Connection) -> None:
        self.main_folder = main_folder
        self.con = con
        self.exp_id = exp_id

        self._metadata = None
        self._raw_files = None
        self._mcor_files = None

        # Add a movie of every type (and load it later)
        # self.movies = {(t,): LazyMovie(self, (t,)) for t in MovieType}

        # Add all movie comparisons
        # for t in [MovieType.TEST, MovieType.MCOR]:
        #     self.movies[(MovieType.RAW, t)] = LazyMovie(self, (MovieType.RAW, t))

        if Experiment.is_first:
            Experiment.short_help()
            Experiment.is_first = False

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        query = f"SELECT * FROM experiments WHERE exp_id = {self.exp_id};"
        self._metadata = pd.read_sql_query(query, self.con)
        self._metadata.set_index("exp_id", inplace=True)

        return self._metadata

    @property
    def mcor_files(self) -> Optional[pd.DataFrame]:
        query = f"SELECT * FROM mcor_files WHERE exp_id = {self.exp_id};"
        self._mcor_files = pd.read_sql_query(query, self.con)
        self._mcor_files.set_index("acq_id", inplace=True)

        return self._mcor_files

    @property
    def raw_files(self) -> Optional[pd.DataFrame]:
        query = f"SELECT * FROM raw_files WHERE exp_id = {self.exp_id};"
        self._raw_files = pd.read_sql_query(query, self.con)
        self._raw_files.set_index("acq_id", inplace=True)

        return self._raw_files

    @staticmethod
    def help(name="Experiment"):
        if name.lower() == "experiment":
            return print(Experiment.__doc__)

        attr = getattr(Experiment, name, None)

        if attr is not None:
            return print(attr.__doc__)

        return print(f"{INFO} Method not found!")

    @staticmethod
    def short_help():
        msg = (
            f"{INFO} Run Experiment.help() to get a list of useful functions.\n"
            f"{INFO} Run Experiment.help('function_name') to know more about a function."
        )
        print(msg)

    # ----------------------------------------------------------------------- #
    # Motion Correction functions
    # ----------------------------------------------------------------------- #

    def run_motion_correction(
        self,
        use_last_parameters: bool = False,
        is_test: bool = True,
        first_acq: int = 1,
        step_acq: int = 1,
        last_acq: int = 3,
        border_nan: bool | str = "copy",
        nonneg_movie: bool = False,
        pw_rigid: bool = True,
        shifts_opencv: bool = False,
        max_deviation_um: float = 12.0,
        max_shift_um: Iterable[float] = [128.0, 128.0],
        overlap_um: Iterable[float] = [96.0, 96.0],
        strides_um: Iterable[float] = [128.0, 128.0],
    ):
        """
        \033[1;31mRUN_MOTION_CORRECTION\033[0m
        Method that does test/final motion correction

        \033[1;34mUSAGE\033[0m
            db  = Database(main_folder)
            exp = db.experiments[some_index]
            exp.run_motion_correction(...)

        \033[1;34mLIST OF PARAMETERS\033[0m (WITH DEFAULT VALUES)

            \033[0;32mBasic Parameters\033[0m
            use_last_parameters = False             Use parameters from last run as the defaults
            is_test             = True              Whether to use a limited range of acquisitions in this run

            \033[0;32mParameters in this section will be ignored if is_test == False\033[0m
            first_acq           = 1                 Index of the first acquisition to motion correct
            step_acq            = 1                 Get one acquisition for every 'step_acq' acquisitions
            last_acq            = 3                 Index of the last acquisition to motion correct

            \033[0;32mCaImAn motion correction parameters\033[0m
            border_nan          = "copy"            copy along the boundary (if True, fill in with NaN)
            nonneg_movie        = False             make SAVED movie mostly non-negative
            pw_rigid            = True              Piecewise-rigid (True) or rigid motion correction
            shifts_opencv       = False             True = bicubic, False = FFT (True is faster)
            max_deviation_um    = 12.0              max deviation for patch with respect to rigid shifts
            max_shift_um        = [128.0, 128.0]    max allowed rigid shift
            overlap_um          = [96.0, 96.0]      overlap between patches (patch = strides + overlaps)
            strides_um          = [128.0, 128.0]    start a new patch every x or y um (only for pw-rigid)

        \033[1;34mEXAMPLES\033[0m
            exp.run_motion_correction(is_test=True, last_acq=10)
        """

        # --- Validate and adjust parameters to reasonable values --- #

        # If is not test, include all acquisitions
        if not is_test:
            first_acq = 0
            step_acq = 1
            last_acq = len(self.raw_files)

        # max_shift_um has to less than size of image (in μm / 4)
        height_um = self.metadata.at[self.exp_id, "height_um"]
        width_um = self.metadata.at[self.exp_id, "width_um"]
        max_shift_um[0] = clamp(max_shift_um[0], 0, height_um / 4)
        max_shift_um[1] = clamp(max_shift_um[1], 0, width_um / 4)

        # --- Save validated parameters in the database --- #

        # TODO: Implement this!!

        # --- Make sure movies will be updated next time they are played --- #

        # TODO: Uncomment this, when ready!!
        # if is_test:
        #     self.movies[(MovieType.TEST,)].mark_as_outdated()
        #     self.movies[(MovieType.RAW, MovieType.TEST)].mark_as_outdated()
        # else:
        #     self.movies[(MovieType.MCOR,)].mark_as_outdated()
        #     self.movies[(MovieType.RAW, MovieType.MCOR)].mark_as_outdated()

        # --- Get settings for CaImAn MotionCorrection class --- #

        # Get raw paths
        raw_files_slice = self.raw_files.iloc[first_acq:last_acq:step_acq]
        raw_paths = [self.main_folder / p for p in raw_files_slice["raw_path"]]

        # CaImAn only uses pixels as units, so we make the conversion
        factor = [
            height_um / self.metadata.at[self.exp_id, "height_px"],
            width_um / self.metadata.at[self.exp_id, "width_px"],
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

        print(f"{INFO} Starting motion correction...")

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

        print(f"{INFO} Finished motion correction")

        # --- Save the results in TIFF files (if it is not a test) --- #

        if is_test:
            return

        with self.con as con:
            print(f"{INFO} Saving mcor files...")

            bar = ProgressBar(len(self.mc.mmap_file))
            bar.show()

            # acq_id            INTEGER PRIMARY KEY
            # , mcor_path         TEXT NOT NULL
            # , approved          INTEGER NOT NULL DEFAULT FALSE
            # , last_updated_by   INTEGER

            # Load mmap files and save them as TIFFs
            # TODO: Confirm that mmap_file and fname have the same order

            for acq_id, mmap_path, raw_path in zip(
                raw_files_slice.index, self.mc.mmap_file, self.mc.fname
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
                query = "INSERT OR REPLACE INTO mcor_files (acq_id, mcor_path) VALUES (?,?);"
                con.execute(
                    query, [acq_id, str(mcor_path.relative_to(self.main_folder))]
                )

                bar.step()
            bar.end()

    def play_movie(
        self,
        use_last_parameters: bool = False,
        grid: Iterable[str] = ["raw", "mcor"],
        downsample_ratio: float = 0.03,
        rigid: bool = False,
        opencv_codec: str = "MJPG",
        save_movie: bool = True,
        save_folder: str = r"./movies",
        backend: str = "embed_opencv",
        do_loop: bool = False,
        fr: float = 30,
        magnification: float = 1,
        plot_text: bool = True,
        q_max: float = 99.5,
        q_min: float = 0.0,
    ) -> None:
        """
        \033[1;31mPLAY_MOVIE\033[0m
        Play and save movies for quality control

        \033[1;34mUSAGE\033[0m
            exp = Experiment(experimentFolder)
            exp.play_movie(...)

        \033[1;34mLIST OF PARAMETERS\033[0m (WITH DEFAULT VALUES)

            \033[0;32mBasic settings\033[0m
            use_last_parameters = False             Use parameters from last run as the defaults
            grid                = ["raw", "mcor"]   How to concatenate movies horizontally.
                                                    ("raw", "mcor", "test" in some order)

            \033[0;32mMovie loading settings\033[0m
            downsample_ratio    = 0.03              Percentage of frames to keep
            rigid               = false             Play rigid or non-rigid motion movies (on tests)

            \033[0;32mVideo saving\033[0m
            opencv_codec        = "MJPG"            Codec used to encode the saved video
            save_movie          = true              Put "true" if you want to save the preview video to a file
            save_folder         = r"./movies"       "." is the experiment folder (r is to use \\ in the path)

            \033[0;32mVideo settings\033[0m
            backend             = "embed_opencv"    "opencv" for popup and "embed_opencv" for inline player
            do_loop             = false             Loop the video or not
            fr                  = 30                How fast to play the video (frames/s)
            magnification       = 1                 Magnification of video
            plot_text           = true              Add current frame label on the video
            q_max               = 99.5              Quantile to consider as white
            q_min               = 0.0               Quantile to consider as black

        \033[1;34mEXAMPLES\033[0m
            exp.play_movie(grid=["raw"], save_folder="~/TempData/20260101/e1/movies")
                Running this command would save a compilation of all raw movies

            exp.play_movie(grid=["raw", "test"])
                This would save a video with raw movies on the left and test on the right
        """

        #     new_hash = self._sync_config()
        #     video_config = dict(self.config["player"]["video"])

        #     movie_type_str = "_".join(t.value for t in movie_types)
        #     filename = f"{self.config["experiment"]["tiff_stem"]}_{movie_type_str}.avi"
        #     filepath = (self.path / filename).resolve()
        #     video_config["movie_name"] = str(filepath)

        #     self.movies[movie_types].maybe_update(new_hash).play(**video_config)
        return

    def delete_temp_files(self) -> None:
        """
        \033[1;31mDELETE_TEMP_FILES\033[0m
        Deletes all temp files associated with this experiment

        \033[1;34mUSAGE\033[0m
            exp = Experiment(experimentFolder)
            exp.delete_temp_files()

        \033[1;34mEXAMPLES\033[0m
            exp.delete_temp_files()
        """

        # Get file paths for mmaps in the temp folder
        path = Path(get_tempdir())
        stem = self.metadata["tiff_stem"]
        movie_paths = sorted(path.glob(f"{stem}*.mmap"))

        # Remove files
        if movie_paths:
            print(f"{INFO} Removing .mmap files that start with {stem}...")
            total_size = 0  # in bytes
            for movie_path in movie_paths:
                total_size += movie_path.stat().st_size
                movie_path.unlink(missing_ok=True)

            total_size = total_size / (1_000_000_000)  # in GBs
            ending = "s" if len(movie_paths) > 1 else ""
            print(
                f"{INFO} Deleted {len(movie_paths)} file{ending} ({total_size:.1f} GB)."
            )
        else:
            print(f"{INFO} No .mmap files found.")

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


# @dataclass
# class LazyMovie:
#     """
#     Movies that only reload when the config changes.
#     """

#     owner: Experiment
#     types: tuple[MovieType, ...]
#     movie: Optional[cm.movie] = None
#     hash: str = ""

#     def mark_as_outdated(self):
#         self.movie = None
#         self.hash = ""

#     def maybe_update(self, new_hash) -> cm.movie:
#         if self.hash == new_hash:
#             print(f"{INFO} No changes to the config. Using cached movie...")
#             return self.movie

#         print(f"{INFO} Updating movie...")

#         load_config = self.owner.config["player"]["load"]
#         downsample_ratio = load_config["downsample_ratio"]

#         movie_chains = []

#         for movie_type in self.types:
#             if movie_type == MovieType.TEST:
#                 rigid = load_config["rigid"]
#                 path = Path(get_tempdir())

#                 stem = self.owner.config["experiment"]["tiff_stem"]
#                 file_identifier = "rig" if rigid else "els"

#                 movie_paths = sorted(path.glob(f"{stem}*{file_identifier}*.mmap"))

#                 mcor_type, other_type = "rigid", "non-rigid"
#                 if not rigid:
#                     mcor_type, other_type = other_type, mcor_type

#                 msg = (
#                     f"No {mcor_type} test movies found for this experiment.\n"
#                     + f"Change the 'rigid' setting to {str(not rigid).lower()} "
#                     + f"in [player.load] to play {other_type} test movies."
#                 )
#                 assert movie_paths, msg

#                 first_acq = self.owner.config["test"]["first_acq"]
#                 step_acq = self.owner.config["test"]["step_acq"]
#                 last_acq = self.owner.config["test"]["last_acq"]

#                 # Get only the slice of movies_paths determined by the parameters above
#                 updated_movie_paths = []

#                 for movie_path in movie_paths:
#                     # Get the acquisition number. It should be the only 5 digit number
#                     # in the filename with underscores around it.
#                     matches = re.findall(r"_\d{5}_", movie_path.name)
#                     assert (
#                         len(matches) == 1
#                     ), "Failed to get files by acquisition number"

#                     # Remove underscores and cast it to a integer
#                     i_acq = int(matches[0][1:6])

#                     if (
#                         first_acq <= i_acq <= last_acq
#                         and (i_acq - first_acq) % step_acq == 0
#                     ):
#                         updated_movie_paths.append(movie_path)

#                 # Compare predicted movie count with actual movie count
#                 # They disagree only when some temp files are missing.
#                 assert len(updated_movie_paths) == max(
#                     0, (last_acq - first_acq) // step_acq + 1
#                 ), "Missing temp files."

#                 movie_paths = updated_movie_paths

#             else:
#                 folder_type = movie_type.value + "_folder"
#                 path = self.owner.path / self.owner.config["experiment"][folder_type]
#                 movie_paths = sorted(path.glob(f"[!.]?*.tif"))

#                 assert movie_paths, f"No movies found in the folder: {path.resolve()}"

#                 # Should use a subset of the files if one the MovieTypes is TEST
#                 if any(mt == MovieType.TEST for mt in self.types):
#                     first_acq = self.owner.config["test"]["first_acq"]
#                     step_acq = self.owner.config["test"]["step_acq"]
#                     last_acq = self.owner.config["test"]["last_acq"]

#                     movie_paths = movie_paths[first_acq - 1 : last_acq : step_acq]

#             print(
#                 f"{INFO} Adding {len(movie_paths)} {movie_type.value} files to the movie."
#             )

#             bar = ProgressBar(len(movie_paths))
#             bar.show()

#             movie_chain = cm.load(movie_paths[0]).resize(1, 1, downsample_ratio)
#             bar.step()

#             for filename in movie_paths[1:]:
#                 movie = cm.load(filename).resize(1, 1, downsample_ratio)
#                 movie_chain = cm.concatenate([movie_chain, movie], axis=0)
#                 bar.step()

#             bar.end()
#             movie_chains.append(movie_chain)

#         movie_chain = cm.concatenate(movie_chains, axis=2)

#         self.movie = movie_chain
#         self.hash = new_hash

#         return self.movie

#     def play(self) -> None:
#         # TODO: - Use Caiman function as blueprint
#         #       - Add movie type, total time, current time, and speed (e.g. x2) to label
#         #       - Remove frame number from label
#         #       - Possibly add total time to config and remove fr in [play.load]
#         return
