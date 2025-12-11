import datetime

from pathlib import Path
from shutil import copy
from tomlkit import load, dump


def create_config(path: str | Path) -> None:
    default_path = Path(__file__).parent / "odyn_config.toml"
    with open(default_path) as file:
        config = load(file)

    path = Path(path)
    exp_path = path.parent

    # Get files that have stem starting and ending with a number
    # In other words, tries to get all raw TIFF files (in the raw folder)
    file_paths = sorted([p for p in exp_path.glob("raw/[0-9]*[0-9].tif")])
    if not file_paths:
        print("Found no raw '.tif' files in this folder.")
        return

    # Get some metadata from the first and last filenames
    date, subject, name, *_, first_acq = file_paths[0].stem.split("_")
    *_, last_acq = file_paths[-1].stem.split("_")

    # Transform string into actual date
    date = datetime.date.fromisoformat(date)

    # Add metadata to config file
    config["experiment"]["date"] = date
    config["experiment"]["subject"] = subject
    config["experiment"]["name"] = name

    config["experiment"]["first_acq"] = int(first_acq)
    config["experiment"]["last_acq"] = int(last_acq)

    # Get metadata from the first raw TIFF file

    with open(path, "w") as file:
        dump(config, file)
