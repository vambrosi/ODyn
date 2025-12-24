from typing import Optional
from pathlib import Path
from hashlib import file_digest
from tomlkit import load, dump

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import get_tempdir

from .config import create_config


class Experiment:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.config_path = self.path / "odyn_config.toml"

        self.config_hash = ""
        self.movies = {}

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

    def _did_config_update(self) -> bool:
        return self.config_hash != self._get_config_file_hash()

    def _sync_config(self) -> None:
        new_hash = self._get_config_file_hash()

        if self.config_hash != new_hash:
            with open(self.config_path) as file:
                print("Loading new config...")
                self.config = load(file)
                self.config_hash = new_hash

    def _get_movies(self, movie_type: str) -> None:
        load_config = self.config["player"]["load"]
        downsample_ratio = load_config["downsample_ratio"]

        if movie_type == "test":
            rigid = load_config["rigid"]
            path = Path(get_tempdir())
            file_identifier = "rig" if rigid else "els"
            movie_paths = sorted(list(path.glob(f"[!.]?*{file_identifier}*.mmap")))

            assert movie_paths, "No movies found in the caiman temp folder"
        else:
            path = self.path / self.config["experiment"][movie_type + "_folder"]
            movie_paths = sorted(list(path.glob(f"[!.]?*.tif")))

            assert movie_paths, "No movies found in the {movie_type} folder"

        movie_chain = cm.load(movie_paths[0]).resize(1, 1, downsample_ratio)

        for filename in movie_paths[1:]:
            movie = cm.load(filename).resize(1, 1, downsample_ratio)
            movie_chain = cm.concatenate([movie_chain, movie], axis=0)

        self.movies[movie_type] = movie_chain

    def _play_movies(self, movie_type) -> None:
        if movie_type not in self.movies or self._did_config_update():
            self._sync_config()
            self._get_movies(movie_type=movie_type)

        video_config = dict(self.config["player"]["video"])

        filename = f"{self.config["experiment"]["tiff_stem"]}_{movie_type}.avi"
        filepath = (self.path / filename).resolve()
        video_config["movie_name"] = str(filepath)

        self.movies[movie_type].play(**video_config)

    def _run_motion_correction(self, final=False) -> None:
        # Check and sync config
        self._sync_config()

        # Get config
        test_config = self.config["test"]
        mcor_config = test_config["motion_correction"]

        # Get acquisition range
        if final:
            first_acq = self.config["experiment"]["first_acq"]
            last_acq = self.config["experiment"]["last_acq"]
        else:
            first_acq = test_config["first_acq"]
            last_acq = test_config["last_acq"]

        # Get raw movies
        raw_path = self.path / self.config["experiment"]["raw_folder"]
        raw_paths = sorted([p for p in raw_path.glob("[!.]?*.tif")])
        raw_paths = raw_paths[first_acq - 1 : last_acq]

        # Convert settings to pixel units
        factor = self.config["imaging"]["um_per_pixels"]
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

        try:
            self.mc = MotionCorrect(raw_paths, dview=dview, **settings)
            self.mc.motion_correct(save_movie=True)
        except:
            cm.stop_server(dview=dview)
            raise

        cm.stop_server(dview=dview)

        # If final save settings and TIFFs
        # If not final skip the rest of the function
        if not final:
            return

        # Record settings in the motion_correction section
        temp = dict(test_config)
        self._sync_config()
        for key, value in temp["motion_correction"].items():
            self.config["motion_correction"][key] = value
        self._save_config()

        # Get TIFFs destination folder and caiman temp folder
        temp_folder = Path(get_tempdir())
        mcor_folder = self.path / self.config["experiment"]["mcor_folder"]
        mcor_folder.mkdir(parents=True, exist_ok=True)

        # Load mmap files and save them as TIFFs
        for mmap_path, raw_path in zip(self.mc.mmap_file, raw_paths):
            mcor_path = mcor_folder / (raw_path.stem + "_mcor.tif")

            # Check if file already exists
            if mcor_path.exists():
                answer = input(
                    f"File {mcor_path.resolve()} already exists. Overwrite? [y/N]"
                )

                if answer.lower() != "y":
                    continue

            mc = cm.load(mmap_path)
            mc.save(mcor_path)

    def _save_config(self) -> None:
        with open(self.config_path, "w") as file:
            dump(self.config, file)

    def play_raw_movies(self) -> None:
        self._play_movies(movie_type="raw")

    def play_mcor_movies(self) -> None:
        self._play_movies(movie_type="mcor")

    def play_test_movies(self) -> None:
        self._play_movies(movie_type="test")

    def run_final_motion_correction(self) -> None:
        self._run_motion_correction(final=True)

    def run_test_motion_correction(self) -> None:
        self._run_motion_correction(final=False)


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]
