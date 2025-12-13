from pathlib import Path
from hashlib import file_digest
from tomlkit import load, dump

import caiman as cm
from caiman.motion_correction import MotionCorrect

from .config import create_config


class Experiment:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.config_path = self.path / "odyn_config.toml"

        self.config_hash = ""
        self.temp_movie = None

        if not self.config_path.exists():
            create_config(self.config_path)

        self._did_config_update()

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def _did_config_update(self) -> bool:
        with open(self.config_path, "rb") as f:
            new_hash = file_digest(f, "sha256").hexdigest()

        if self.config_hash != new_hash:
            with open(self.config_path) as file:
                print("Loading new config...")
                self.config = load(file)
                self.config_hash = new_hash

            # Had to load the file
            return True

        # Kept the old file
        return False

    def _get_temp_movie(self) -> None:
        load_config = self.config["test"]["player"]["load"]
        downsample_ratio = load_config["downsample_ratio"]
        rigid = load_config["rigid"]

        path = Path.home() / "caiman_data" / "temp"

        file_identifier = "rig" if rigid else "els"
        filenames = sorted([p for p in path.rglob(f"[!.]?*{file_identifier}*.mmap")])

        assert filenames, "No movies found in the caiman temp folder"

        movie_chain = cm.load(filenames[0]).resize(1, 1, downsample_ratio)

        for filename in filenames[1:]:
            movie = cm.load(filename).resize(1, 1, downsample_ratio)
            movie_chain = cm.concatenate([movie_chain, movie], axis=0)

        self.temp_movie = movie_chain

    def _save_config(self) -> None:
        with open(self.config_path, "w") as file:
            dump(self.config, file)

    def play_test_movie(self) -> None:
        if self.temp_movie is None or self._did_config_update():
            self._get_temp_movie()

        video_config = self.config["test"]["player"]["video"]
        self.temp_movie.play(**video_config)

    def run_test_motion_correction(self) -> None:
        # Check and sync config
        self._did_config_update()

        # Get config
        test_config = self.config["test"]
        mcor_config = test_config["motion_correction"]

        # Get raw movies
        raw_path = self.path / self.config["experiment"]["raw_folder"]
        tif_files = sorted([p for p in raw_path.glob("[!.]?*.tif")])
        tif_files = tif_files[test_config["first_acq"] - 1 : test_config["last_acq"]]

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

        self.test_mc = MotionCorrect(tif_files, dview=dview, **settings)
        self.test_mc.motion_correct(save_movie=True)

        cm.stop_server(dview=dview)


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]
