import re
import sys

from enum import Enum
from typing import Optional
from dataclasses import dataclass

from pathlib import Path
from hashlib import file_digest
from tomlkit import load, dump

import numpy as np
import tifffile

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir
from caiman.utils.visualization import create_quilt_patches

from .config import create_config


class MovieType(Enum):
    RAW = "raw"
    MCOR = "mcor"
    TEST = "test"


class Experiment:
    """
    Class that creates/loads the config file and runs data processing/analysis.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.config_path = self.path / "odyn_config.toml"

        self.config_hash = ""

        # Add a movie of every type (and load it later)
        self.movies = {(t,): LazyMovie(self, (t,)) for t in MovieType}

        # Add all movie comparisons
        for t in [MovieType.TEST, MovieType.MCOR]:
            self.movies[(MovieType.RAW, t)] = LazyMovie(self, (MovieType.RAW, t))

        if not self.config_path.exists():
            create_config(self.config_path)

        self._sync_config()

    def __str__(self):
        exp_info = self["experiment"]
        return (
            f"Experiment {exp_info["name"]}\n"
            f"  Date: {exp_info["date"]}\n"
            f"  Subject: {exp_info["subject"]}\n"
            f"  Folder: {str(self.path)}"
        )

    def __repr__(self):
        return str(self)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def _get_config_file_hash(self) -> str:
        with open(self.config_path, "rb") as f:
            return file_digest(f, "sha256").hexdigest()

    def _sync_config(self) -> str:
        # Compute hash to see if something changed
        new_hash = self._get_config_file_hash()

        # Load file if config changed
        if self.config_hash != new_hash:
            with open(self.config_path) as file:
                print("[INFO] Loading new config...")
                self.config = load(file)
                self.config_hash = new_hash

        return new_hash

    def _save_config(self) -> None:
        with open(self.config_path, "w") as file:
            dump(self.config, file)

    # ----------------------------------------------------------------- #
    # Motion Correction functions
    # ----------------------------------------------------------------- #

    def _play_movie(self, movie_types: tuple[MovieType, ...]) -> None:
        new_hash = self._sync_config()
        video_config = dict(self.config["player"]["video"])

        movie_type_str = "_".join(t.value for t in movie_types)
        filename = f"{self.config["experiment"]["tiff_stem"]}_{movie_type_str}.avi"
        filepath = (self.path / filename).resolve()
        video_config["movie_name"] = str(filepath)

        self.movies[movie_types].maybe_update(new_hash).play(**video_config)

    def _run_motion_correction(self, final=False) -> None:
        # Check and sync config
        self._sync_config()

        # Get config
        test_config = self.config["test"]
        mcor_config = test_config["motion_correction"]

        # Get some setup variables and possibly return early
        if final:
            # Get TIFFs destination folder (and creates it if it doesn't exist)
            mcor_folder = self.path / self.config["experiment"]["mcor_folder"]
            mcor_folder.mkdir(parents=True, exist_ok=True)

            # If it is a final motion correction run, there are previous motion corrected
            # files, and the user doesn't want to overwrite them, exit the function.

            # This for loop is executed at most once (if the iterator is non-empty)
            for _ in mcor_folder.glob(f"[!.]?*_mcor.tif"):
                answer = input(f"Overwrite motion corrected files? [y/N]")

                if answer.lower() != "y":
                    return

                break

            # Get acquisition range
            first_acq = self.config["experiment"]["first_acq"]
            last_acq = self.config["experiment"]["last_acq"]
            step_acq = 1

        else:
            first_acq = test_config["first_acq"]
            step_acq = test_config["step_acq"]
            last_acq = test_config["last_acq"]

        # Get raw movies
        raw_path = self.path / self.config["experiment"]["raw_folder"]
        raw_paths = sorted(raw_path.glob("[!.]?*.tif"))
        raw_paths = raw_paths[first_acq - 1 : last_acq : step_acq]

        if not final:
            print("[INFO] List of files included in the test:")
            for path in raw_paths:
                print(f"[INFO]   {path}")

        # Convert settings to pixel units
        factor = self.config["metadata"]["um_per_pixels"]
        settings = {
            "border_nan": mcor_config["border_nan"],
            "pw_rigid": mcor_config["pw_rigid"],
            "shifts_opencv": mcor_config["shifts_opencv"],
            "nonneg_movie": mcor_config["nonneg_movie"],
            "max_deviation_rigid": int(mcor_config["max_deviation_um"] / min(factor)),
            "max_shifts": um_to_pixels(mcor_config["max_shift_um"], factor),
            "overlaps": um_to_pixels(mcor_config["overlap_um"], factor),
            "strides": um_to_pixels(mcor_config["strides_um"], factor),
        }

        _, dview, _ = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=None, single_thread=False
        )

        print("[INFO] Starting motion correction...")

        try:
            self.mc = MotionCorrect(raw_paths, dview=dview, **settings)
            self.mc.motion_correct(save_movie=True)

        except:
            raise

        # Always stop the server after motion correction
        finally:
            cm.stop_server(dview=dview)

        print("[INFO] Finished motion correction")

        # If final save settings and TIFF files
        if final:
            if self.config["metadata"]["size_gigabytes"] > 4:
                big_file = True
                print("[INFO] Saving mcor files with bigtiff = True...")
            else:
                big_file = False
                print("[INFO] Saving mcor files...")

            # Record settings in the motion_correction section
            temp = dict(test_config)
            self._sync_config()
            for key, value in temp["motion_correction"].items():
                self.config["motion_correction"][key] = value
            self._save_config()

            bar = ProgressBar(len(self.mc.mmap_file))
            bar.show()

            # Load mmap files and save them as TIFFs
            for mmap_path, raw_path in zip(self.mc.mmap_file, raw_paths):
                mcor_path = mcor_folder / (raw_path.stem + "_mcor.tif")

                mc = cm.load(mmap_path)

                # Saving TIFFs directly because caiman saves them as 64-bit
                with tifffile.TiffWriter(mcor_path, bigtiff=big_file) as tif:
                    tif.write(
                        [mc[i].copy() for i in range(mc.shape[0])],
                        shape=mc[0].shape,
                        dtype=mc.dtype,
                    )

                bar.step()
            bar.end()

    def delete_temp_files(self) -> None:
        # Check an sync config
        self._sync_config()

        # Get file paths for mmaps in the temp folder
        path = Path(get_tempdir())
        stem = self.config["experiment"]["tiff_stem"]
        movie_paths = sorted(path.glob(f"{stem}*.mmap"))

        # Remove files
        if movie_paths:
            print(f"[INFO] Removing .mmap files that start with {stem}...")
            total_size = 0  # in bytes
            for movie_path in movie_paths:
                total_size += movie_path.stat().st_size
                movie_path.unlink(missing_ok=True)

            total_size = total_size / (1_000_000_000)  # in GBs
            ending = "s" if len(movie_paths) > 1 else ""
            print(
                f"[INFO] Deleted {len(movie_paths)} file{ending} ({total_size:.1f} GB)."
            )
        else:
            print(f"[INFO] No .mmap files found.")

    def pick_mcor_parameters(self):
        import bokeh.plotting as bpl

        from bokeh.io import output_notebook
        from bokeh.models import (
            Button,
            ColumnDataSource,
            Slider,
            LinearColorMapper,
        )
        from bokeh.io.state import curstate
        from bokeh.palettes import Greys256

        is_notebook = "ipykernel" in sys.modules
        is_bokeh_initialized = curstate().notebook

        if is_notebook and not is_bokeh_initialized:
            # HACK: So things show in VSCode (change it)
            import os

            os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"

            output_notebook()

        def modify_doc(doc):
            # Check and sync config
            self._sync_config()

            # Get raw movies
            raw_path = self.path / self.config["experiment"]["raw_folder"]
            raw_paths = sorted(raw_path.glob("[!.]?*.tif"))

            assert raw_paths, "Could not find any raw TIFF files."

            # Create correlation image as in demo_pipeline.ipynb
            movie = cm.load(raw_paths[0])
            correlation_image = cm.local_correlations(movie, swap_dim=False)
            correlation_image[np.isnan(correlation_image)] = 0

            lower_quantile = np.quantile(correlation_image, 0.01)
            higher_quantile = np.quantile(correlation_image, 0.99)

            # Get current values
            size_pixels = self.config["metadata"]["size_pixels"]
            um_per_pixels = self.config["metadata"]["um_per_pixels"]

            strides_um = self.config["test"]["motion_correction"]["strides_um"]
            overlap_um = self.config["test"]["motion_correction"]["overlap_um"]

            strides = [int(s / c) for s, c in zip(strides_um, um_per_pixels)]
            overlaps = [int(o / c) for o, c in zip(overlap_um, um_per_pixels)]

            MAX = min(size_pixels) / 2
            MIN = 1

            p = bpl.figure(
                x_range=(0, size_pixels[1]),
                y_range=(size_pixels[0], 0),
                width=600,
                height=600,
            )

            stride_x = Slider(start=MIN, end=MAX, value=strides[1], title="Stride x")
            overlap_x = Slider(start=MIN, end=MAX, value=overlaps[1], title="Overlap x")

            stride_y = Slider(
                start=MIN,
                end=MAX,
                value=strides[0],
                title="Stride y",
            )
            overlap_y = Slider(
                start=MIN,
                end=MAX,
                value=overlaps[0],
                title="Overlap y",
            )

            source = ColumnDataSource(data=dict(x=[], y=[], w=[], h=[]))

            color_mapper = LinearColorMapper(
                palette=Greys256, low=lower_quantile, high=higher_quantile
            )

            p.image(
                image=[correlation_image],
                x=[0],
                y=[0],
                dh=[correlation_image.shape[0]],
                dw=[correlation_image.shape[1]],
                color_mapper=color_mapper,
            )

            p.rect(
                x="x",
                y="y",
                width="w",
                height="h",
                source=source,
                alpha=0.2,
                line_color="white",
                selection_color="red",
            )

            p.xaxis.visible = False
            p.yaxis.visible = False

            def update_data(attrs, old, new):
                patch_rows, patch_cols = get_rectangle_coords(
                    size_pixels,
                    stride_x.value,
                    stride_y.value,
                    overlap_x.value,
                    overlap_y.value,
                )
                patches = create_quilt_patches(patch_rows, patch_cols)

                source.data = dict(
                    x=[patch["center_x"] for patch in patches],
                    y=[patch["center_y"] for patch in patches],
                    w=[patch["width"] for patch in patches],
                    h=[patch["height"] for patch in patches],
                )

            stride_x.on_change("value", update_data)
            stride_y.on_change("value", update_data)
            overlap_x.on_change("value", update_data)
            overlap_y.on_change("value", update_data)

            p.add_tools("tap")

            # Force the first update
            update_data(None, None, None)

            save_button = Button(label="Save Parameters", button_type="success")

            # Save slider values to odyn_config.toml
            def save_callback():
                strides = [stride_y.value, stride_x.value]
                overlaps = [overlap_y.value, overlap_x.value]

                self.config["test"]["motion_correction"]["strides_um"] = [
                    float(s * c) for s, c in zip(strides, um_per_pixels)
                ]
                self.config["test"]["motion_correction"]["overlap_um"] = [
                    float(o * c) for o, c in zip(overlaps, um_per_pixels)
                ]
                self._save_config()

            save_button.on_click(save_callback)

            layout = bpl.row(
                bpl.column(
                    p,
                    bpl.row(stride_x, stride_y),
                    bpl.row(overlap_x, overlap_y),
                    save_button,
                ),
            )

            doc.add_root(layout)

        bpl.show(modify_doc)

    def play_raw_movies(self) -> None:
        self._play_movie(movie_types=(MovieType.RAW,))

    def play_mcor_movies(self) -> None:
        self._play_movie(movie_types=(MovieType.MCOR,))

    def play_test_movies(self) -> None:
        self._play_movie(movie_types=(MovieType.TEST,))

    def play_test_comparison(self) -> None:
        self._play_movie(movie_types=(MovieType.RAW, MovieType.TEST))

    def play_mcor_comparison(self) -> None:
        self._play_movie(movie_types=(MovieType.RAW, MovieType.MCOR))

    def run_final_motion_correction(self) -> None:
        self._run_motion_correction(final=True)
        self.movies[(MovieType.MCOR,)].mark_as_outdated()
        self.movies[(MovieType.RAW, MovieType.MCOR)].mark_as_outdated()

    def run_test_motion_correction(self) -> None:
        self._run_motion_correction(final=False)
        self.movies[(MovieType.TEST,)].mark_as_outdated()
        self.movies[(MovieType.RAW, MovieType.TEST)].mark_as_outdated()

    # ----------------------------------------------------------------- #
    # Data Analysis
    # ----------------------------------------------------------------- #

    # TODO: 1) Compute z-score compared with the baseline for each file
    #       2) Reproduce the pre-processing from the MATLAB code

    def compute_z_scores(self) -> cm.movie:
        # Check an sync config
        self._sync_config()

        # TODO: replace dummy values below
        odor_onset = 2
        odor_offset = 4
        # END OF DUMMY VALUES ------------

        # Get config parameters
        frame_rate = self.config["metadata"]["frame_rate"]
        post_odor_interval = self.config["z-scores"]["post_odor_interval"]
        baseline_pre_odor = self.config["z-scores"]["baseline_pre_odor"]

        frame_odor_onset = int(odor_onset * frame_rate)
        frame_onset = int((odor_onset + post_odor_interval) * frame_rate)
        frame_offset = int((odor_offset + post_odor_interval) * frame_rate)
        frame_duration = frame_offset - frame_onset + 1

        frame_baseline_start = (
            frame_odor_onset - frame_duration
            if baseline_pre_odor
            else frame_onset - frame_duration
        )

        mcor_folder = self.path / self.config["experiment"]["mcor_folder"]
        mcor_files = mcor_folder.glob(f"[!.]?*_mcor.tif")

        assert mcor_files, "No .tif files in the mcor folder."

        # TIFFs were originally in int16, so we add 32768 to make then non-negative
        baseline_range = range(frame_baseline_start, frame_duration)
        baseline_avg = cm.load(mcor_files[0], subindices=baseline_range).mean(axis=0)
        baseline_avg += 32768

        signal_range = range(frame_onset, frame_duration)
        signal_avg = cm.load(mcor_files[0], subindices=signal_range).mean(axis=0)
        signal_avg += 32768

        # Computes z-score of dF/F
        dFF = (signal_avg - baseline_avg) / baseline_avg
        z_score = (dFF - dFF.mean()) / dFF.std()

        return z_score


@dataclass
class LazyMovie:
    """
    Movies that only reload when the config changes.
    """

    owner: Experiment
    types: tuple[MovieType, ...]
    movie: Optional[cm.movie] = None
    hash: str = ""

    def mark_as_outdated(self):
        self.movie = None
        self.hash = ""

    def maybe_update(self, new_hash) -> cm.movie:
        if self.hash == new_hash:
            print("[INFO] No changes to the config. Using cached movie...")
            return self.movie

        print("[INFO] Updating movie...")

        load_config = self.owner.config["player"]["load"]
        downsample_ratio = load_config["downsample_ratio"]

        movie_chains = []

        for movie_type in self.types:
            if movie_type == MovieType.TEST:
                rigid = load_config["rigid"]
                path = Path(get_tempdir())

                stem = self.owner.config["experiment"]["tiff_stem"]
                file_identifier = "rig" if rigid else "els"

                movie_paths = sorted(path.glob(f"{stem}*{file_identifier}*.mmap"))

                mcor_type, other_type = "rigid", "non-rigid"
                if not rigid:
                    mcor_type, other_type = other_type, mcor_type

                msg = (
                    f"No {mcor_type} test movies found for this experiment.\n"
                    + f"Change the 'rigid' setting to {str(not rigid).lower()} "
                    + f"in [player.load] to play {other_type} test movies."
                )
                assert movie_paths, msg

                first_acq = self.owner.config["test"]["first_acq"]
                step_acq = self.owner.config["test"]["step_acq"]
                last_acq = self.owner.config["test"]["last_acq"]

                # Get only the slice of movies_paths determined by the parameters above
                updated_movie_paths = []

                for movie_path in movie_paths:
                    # Get the acquisition number. It should be the only 5 digit number
                    # in the filename with underscores around it.
                    matches = re.findall(r"_\d{5}_", movie_path.name)
                    assert (
                        len(matches) == 1
                    ), "Failed to get files by acquisition number"

                    # Remove underscores and cast it to a integer
                    i_acq = int(matches[0][1:6])

                    if (
                        first_acq <= i_acq <= last_acq
                        and (i_acq - first_acq) % step_acq == 0
                    ):
                        updated_movie_paths.append(movie_path)

                # Compare predicted movie count with actual movie count
                # They disagree only when some temp files are missing.
                assert len(updated_movie_paths) == max(
                    0, (last_acq - first_acq) // step_acq + 1
                ), "Missing temp files."

                movie_paths = updated_movie_paths

            else:
                folder_type = movie_type.value + "_folder"
                path = self.owner.path / self.owner.config["experiment"][folder_type]
                movie_paths = sorted(path.glob(f"[!.]?*.tif"))

                assert movie_paths, f"No movies found in the folder: {path.resolve()}"

                # Should use a subset of the files if one the MovieTypes is TEST
                if any(mt == MovieType.TEST for mt in self.types):
                    first_acq = self.owner.config["test"]["first_acq"]
                    step_acq = self.owner.config["test"]["step_acq"]
                    last_acq = self.owner.config["test"]["last_acq"]

                    movie_paths = movie_paths[first_acq - 1 : last_acq : step_acq]

            print(
                f"[INFO] Adding {len(movie_paths)} {movie_type.value} files to the movie."
            )

            bar = ProgressBar(len(movie_paths))
            bar.show()

            movie_chain = cm.load(movie_paths[0]).resize(1, 1, downsample_ratio)
            bar.step()

            for filename in movie_paths[1:]:
                movie = cm.load(filename).resize(1, 1, downsample_ratio)
                movie_chain = cm.concatenate([movie_chain, movie], axis=0)
                bar.step()

            bar.end()
            movie_chains.append(movie_chain)

        movie_chain = cm.concatenate(movie_chains, axis=2)

        self.movie = movie_chain
        self.hash = new_hash

        return self.movie

    def play(self) -> None:
        # TODO: - Use Caiman function as blueprint
        #       - Add movie type, total time, current time, and speed (e.g. x2) to label
        #       - Remove frame number from label
        #       - Possibly add total time to config and remove fr in [play.load]
        return


# TODO: - Fix the logging interaction with this class
@dataclass
class ProgressBar:
    total: int
    current: int = 0

    def show(self) -> None:
        filled_squares = int(40 * self.current / self.total)

        progress_str = "[INFO] [ "
        progress_str += "\u2588" * filled_squares
        progress_str += "-" * (40 - filled_squares)
        progress_str += f" ] {self.current:03d}/{self.total:03d} Files processed"

        print(progress_str, end="\r")

    def step(self) -> None:
        self.current = min(self.current + 1, self.total)
        self.show()

    def end(self) -> None:
        msg = f"[INFO] {self.total} files processed sucessfully."
        print(f"{msg:75s}")


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]


def get_rectangle_coords(
    size: list[int], stride_x: int, stride_y: int, overlap_x: int, overlap_y: int
) -> tuple[np.ndarray, np.ndarray]:

    patch_width = overlap_x + stride_x
    patch_height = overlap_y + stride_y

    patch_onset_rows = np.array(
        list(range(0, size[0] - patch_height, stride_y)) + [size[0] - patch_height]
    )
    patch_offset_rows = patch_onset_rows + patch_height
    patch_offset_rows[patch_offset_rows > size[0] - 1] = size[0] - 1
    patch_rows = np.column_stack((patch_onset_rows, patch_offset_rows))

    patch_onset_cols = np.array(
        list(range(0, size[1] - patch_width, stride_x)) + [size[1] - patch_width]
    )
    patch_offset_cols = patch_onset_cols + patch_width
    patch_offset_cols[patch_offset_cols > size[1] - 1] = size[1] - 1
    patch_cols = np.column_stack((patch_onset_cols, patch_offset_cols))

    return patch_rows, patch_cols
