from pathlib import Path
from hashlib import file_digest
from tomlkit import load, dump
import caiman as cm

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
