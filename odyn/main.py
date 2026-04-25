import re
import sqlite3

from typing import Optional
from dataclasses import dataclass

from pathlib import Path

import tifffile

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir

from .const import MovieType, INFO
from .config import create_db

# Print a helpful string when user imports this library
print(f"[{INFO}] Run Experiment.help() to get examples of how to use ODyn.")


class Experiment:
    """
    \033[1;31mEXPERIMENT\033[0m
    Class that runs data processing/analysis.

    \033[1;34mUSAGE\033[0m
        exp = Experiment(experimentFolder)

    \033[1;34mRELEVANT METHODS\033[0m
        exp.run_motion_correction()
        exp.play_movie()
        exp.delete_temp_files()

    Run Experiment.help('method_name') to know more about one of the methods above.

    \033[1;34mEXAMPLE\033[0m
        Experiment.help('play_movie')
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.db_path = self.path / "odyn.db"

        # Add a movie of every type (and load it later)
        # self.movies = {(t,): LazyMovie(self, (t,)) for t in MovieType}

        # Add all movie comparisons
        # for t in [MovieType.TEST, MovieType.MCOR]:
        #     self.movies[(MovieType.RAW, t)] = LazyMovie(self, (MovieType.RAW, t))

        # Get connection or create database if needed
        self.db_con = (
            sqlite3.connect(self.db_path)
            if self.db_path.exists()
            else create_db(self.db_path)
        )
        self.db_con.row_factory = sqlite3.Row

        with self.db_con as con:
            res = con.execute("SELECT * FROM metadata;")
            self.metadata = dict(res.fetchone())

        Experiment.short_help()

    def __del__(self):
        self.db_con.close()

    def __str__(self):
        return (
            f"Experiment {self.metadata["exp_name"]}\n"
            f"  Date: {self.metadata["exp_date"]}\n"
            f"  Subject: {self.metadata["mouse_id"]}\n"
            f"  Folder: {str(self.path)}"
        )

    def __repr__(self):
        return str(self)

    @staticmethod
    def short_help():
        msg = (
            f"[{INFO}] Run Experiment.help() to get a list of useful functions.\n"
            f"[{INFO}] Run Experiment.help('function_name') to know more about a function."
        )
        print(msg)

    @staticmethod
    def help(name="Experiment"):
        if name.lower() == "experiment":
            return print(Experiment.__doc__)

        attr = getattr(Experiment, name, None)

        if attr is not None:
            return print(attr.__doc__)

        return print(f"[{INFO}] Class/method not found!")

    # ----------------------------------------------------------------- #
    # Motion Correction functions
    # ----------------------------------------------------------------- #

    # def _play_movie(self, movie_types: tuple[MovieType, ...]) -> None:
    #     new_hash = self._sync_config()
    #     video_config = dict(self.config["player"]["video"])

    #     movie_type_str = "_".join(t.value for t in movie_types)
    #     filename = f"{self.config["experiment"]["tiff_stem"]}_{movie_type_str}.avi"
    #     filepath = (self.path / filename).resolve()
    #     video_config["movie_name"] = str(filepath)

    #     self.movies[movie_types].maybe_update(new_hash).play(**video_config)

    # def _run_motion_correction(self, final=False) -> None:
    #     # Check and sync config
    #     self._sync_config()

    #     # Get config
    #     test_config = self.config["test"]
    #     mcor_config = test_config["motion_correction"]

    #     # Get some setup variables and possibly return early
    #     if final:
    #         # Get TIFFs destination folder (and creates it if it doesn't exist)
    #         mcor_folder = self.path / self.config["experiment"]["mcor_folder"]
    #         mcor_folder.mkdir(parents=True, exist_ok=True)

    #         # If it is a final motion correction run, there are previous motion corrected
    #         # files, and the user doesn't want to overwrite them, exit the function.

    #         # This for loop is executed at most once (if the iterator is non-empty)
    #         for _ in mcor_folder.glob(f"[!.]?*_mcor.tif"):
    #             answer = input(f"Overwrite motion corrected files? [y/N]")

    #             if answer.lower() != "y":
    #                 return

    #             break

    #         # Get acquisition range
    #         first_acq = self.config["experiment"]["first_acq"]
    #         last_acq = self.config["experiment"]["last_acq"]
    #         step_acq = 1

    #     else:
    #         first_acq = test_config["first_acq"]
    #         step_acq = test_config["step_acq"]
    #         last_acq = test_config["last_acq"]

    #     # Get raw movies
    #     raw_path = self.path / self.config["experiment"]["raw_folder"]
    #     raw_paths = sorted(raw_path.glob("[!.]?*.tif"))
    #     raw_paths = raw_paths[first_acq - 1 : last_acq : step_acq]

    #     if not final:
    #         print(f"[{INFO}] List of files included in the test:")
    #         for path in raw_paths:
    #             print(f"[{INFO}]   {path}")

    #     # Convert settings to pixel units
    #     factor = self.config["metadata"]["um_per_pixels"]
    #     settings = {
    #         "border_nan": mcor_config["border_nan"],
    #         "pw_rigid": mcor_config["pw_rigid"],
    #         "shifts_opencv": mcor_config["shifts_opencv"],
    #         "nonneg_movie": mcor_config["nonneg_movie"],
    #         "max_deviation_rigid": int(mcor_config["max_deviation_um"] / min(factor)),
    #         "max_shifts": um_to_pixels(mcor_config["max_shift_um"], factor),
    #         "overlaps": um_to_pixels(mcor_config["overlap_um"], factor),
    #         "strides": um_to_pixels(mcor_config["strides_um"], factor),
    #     }

    #     _, dview, _ = cm.cluster.setup_cluster(
    #         backend="multiprocessing", n_processes=None, single_thread=False
    #     )

    #     print(f"[{INFO}] Starting motion correction...")

    #     try:
    #         self.mc = MotionCorrect(raw_paths, dview=dview, **settings)
    #         self.mc.motion_correct(save_movie=True)

    #     except:
    #         raise

    #     # Always stop the server after motion correction
    #     finally:
    #         cm.stop_server(dview=dview)

    #     print(f"[{INFO}] Finished motion correction")

    #     # If final save settings and TIFF files
    #     if final:
    #         print(f"[{INFO}] Saving mcor files...")

    #         # Record settings in the motion_correction section
    #         temp = dict(test_config)
    #         self._sync_config()
    #         for key, value in temp["motion_correction"].items():
    #             self.config["motion_correction"][key] = value
    #         self._save_config()

    #         bar = ProgressBar(len(self.mc.mmap_file))
    #         bar.show()

    #         # Load mmap files and save them as TIFFs
    #         for mmap_path, raw_path in zip(self.mc.mmap_file, raw_paths):
    #             mcor_path = mcor_folder / (raw_path.stem + "_mcor.tif")

    #             mc = cm.load(mmap_path)

    #             # Saving TIFFs directly because caiman saves them as 64-bit
    #             with tifffile.TiffWriter(mcor_path) as tif:
    #                 tif.write(
    #                     [mc[i].copy() for i in range(mc.shape[0])],
    #                     shape=mc[0].shape,
    #                     dtype=mc.dtype,
    #                 )

    #             bar.step()
    #         bar.end()

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
            print(f"[{INFO}] Removing .mmap files that start with {stem}...")
            total_size = 0  # in bytes
            for movie_path in movie_paths:
                total_size += movie_path.stat().st_size
                movie_path.unlink(missing_ok=True)

            total_size = total_size / (1_000_000_000)  # in GBs
            ending = "s" if len(movie_paths) > 1 else ""
            print(
                f"[{INFO}] Deleted {len(movie_paths)} file{ending} ({total_size:.1f} GB)."
            )
        else:
            print(f"[{INFO}] No .mmap files found.")

    def run_motion_correction(self):
        """
        \033[1;31mRUN_MOTION_CORRECTION\033[0m
        Method that does test/final motion correction

        \033[1;34mUSAGE\033[0m
            exp = Experiment(experimentFolder)
            exp.run_motion_correction(...)

        \033[1;34mLIST OF PARAMETERS\033[0m (WITH DEFAULT VALUES)

            \033[0;32mBasic Parameters\033[0m
            use_last_parameters = False             Use parameters from last run as the defaults
            is_test             = True              Whether to use a limited range of acquisitions in this run

            \033[0;32mParameters in this section are ignored if is_test == False\033[0m
            first_acq           = 1                 Number of the first acquisition to motion correct
            step_acq            = 1                 Get one acquisition for every 'step_acq' acquisitions
            last_acq            = 3                 Number of the last acquisition to motion correct

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

        return

    # def play_raw_movies(self) -> None:
    #     self._play_movie(movie_types=(MovieType.RAW,))

    # def play_mcor_movies(self) -> None:
    #     self._play_movie(movie_types=(MovieType.MCOR,))

    # def play_test_movies(self) -> None:
    #     self._play_movie(movie_types=(MovieType.TEST,))

    # def play_test_comparison(self) -> None:
    #     self._play_movie(movie_types=(MovieType.RAW, MovieType.TEST))

    # def play_mcor_comparison(self) -> None:
    #     self._play_movie(movie_types=(MovieType.RAW, MovieType.MCOR))

    # def run_final_motion_correction(self) -> None:
    #     self._run_motion_correction(final=True)
    #     self.movies[(MovieType.MCOR,)].mark_as_outdated()
    #     self.movies[(MovieType.RAW, MovieType.MCOR)].mark_as_outdated()

    # def run_test_motion_correction(self) -> None:
    #     self._run_motion_correction(final=False)
    #     self.movies[(MovieType.TEST,)].mark_as_outdated()
    #     self.movies[(MovieType.RAW, MovieType.TEST)].mark_as_outdated()

    # ----------------------------------------------------------------- #
    # Data Analysis
    # ----------------------------------------------------------------- #

    # TODO: 1) Compute z-score compared with the baseline for each file
    #       2) Reproduce the pre-processing from the MATLAB code

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
#             print(f"[{INFO}] No changes to the config. Using cached movie...")
#             return self.movie

#         print(f"[{INFO}] Updating movie...")

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
#                 f"[{INFO}] Adding {len(movie_paths)} {movie_type.value} files to the movie."
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


# TODO: - Fix the logging interaction with this class
@dataclass
class ProgressBar:
    total: int
    current: int = 0

    def show(self) -> None:
        filled_squares = int(40 * self.current / self.total)

        progress_str = f"[{INFO}] [ "
        progress_str += "\u2588" * filled_squares
        progress_str += "-" * (40 - filled_squares)
        progress_str += f" ] {self.current:03d}/{self.total:03d} Files processed"

        print(progress_str, end="\r")

    def step(self) -> None:
        self.current = min(self.current + 1, self.total)
        self.show()

    def end(self) -> None:
        msg = f"[{INFO}] {self.total} files processed sucessfully."
        print(f"{msg:75s}")


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]
