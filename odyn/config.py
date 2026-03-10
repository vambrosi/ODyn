import datetime

from pathlib import Path
from shutil import copy
from tomlkit import load, dump, string
from tifffile import TiffFile


def create_config(path: str | Path) -> None:
    default_path = Path(__file__).parent / "odyn_config.toml"
    with open(default_path) as file:
        config = load(file)

    path = Path(path)
    exp_path = path.parent

    # Assume raw files are in the 'raw' folder
    raw_path = exp_path / "raw"

    # Get TIFF files in the raw folder that don't start with a '.'
    # The last condition is to exclude some hidden files that MacOS creates
    file_paths = sorted(raw_path.glob("[!.]?*.tif"))
    assert file_paths, f"Found no .tif files in: {raw_path.resolve()}"

    # Get some metadata from the first and last filenames
    file_stem_parts = file_paths[0].stem.split("_")

    # Check if file names have the correct form
    msg = (
        "File names should be of the form: "
        + "MMDDYYYY_SubjectID_ExperimentName_Number.tif -- "
        + "Example: 20251014_sid309_e1_00001.tif"
    )
    assert len(file_stem_parts) >= 4, msg

    date, subject, name, *_, first_acq = file_stem_parts
    *_, last_acq = file_paths[-1].stem.split("_")

    # Transform string into actual date
    date = datetime.date.fromisoformat(date)

    # Add metadata to config file
    config["experiment"]["date"] = date
    config["experiment"]["subject"] = subject
    config["experiment"]["name"] = name

    config["experiment"]["first_acq"] = int(first_acq)
    config["experiment"]["last_acq"] = int(last_acq)
    config["experiment"]["n_acq"] = len(file_paths)

    config["experiment"]["tiff_stem"] = "_".join(file_stem_parts[:-1])

    # Get metadata from the first raw TIFF file
    tif = TiffFile(file_paths[0])
    SI_metadata = tif.scanimage_metadata["FrameData"]

    config["metadata"]["frames"] = len(tif.pages)
    config["metadata"]["size_pixels"] = tif.pages[0].shape
    config["metadata"]["frame_rate"] = SI_metadata["SI.hRoiManager.scanFrameRate"]

    # Assume unit is centimeters
    dx, nx = tif.pages[0].tags["XResolution"].value
    dy, ny = tif.pages[0].tags["YResolution"].value
    config["metadata"]["um_per_pixels"] = list(
        map(lambda x: round(1e4 * x, 4), [nx / dx, ny / dy])
    )

    # TODO: 1) Find max_shift limit.
    #       2) Write code to enforce this limit.

    # Save config
    with open(path, "w") as file:
        dump(config, file)
