import sqlite3

from pathlib import Path
from datetime import date
from tifffile import TiffFile


def create_db(path: str | Path) -> None:
    # Get all relevant paths
    db_path = Path(path)
    exp_path = path.parent
    raw_path = exp_path / "raw"

    # ----------------------------------------------------------------------- #
    # Gather all the metadata from raw files and file names
    # ----------------------------------------------------------------------- #

    # Get TIFF files in the raw folder that don't start with a '.'
    # The last condition is to exclude some hidden files that MacOS creates
    file_paths = sorted(raw_path.glob("[!.]?*.tif"))
    assert file_paths, f"Found no .tif files in: {raw_path.resolve()}"

    # Get some metadata from the first and last filenames
    file_stem_parts = file_paths[0].stem.split("_")

    # Check if file names have the correct form
    msg = (
        "File names should be of the form: "
        + "MMDDYYYY_SubjectID_ExperimentName_..._Number.tif -- "
        + "Example: 20251014_sid309_e1_00001.tif"
    )
    assert len(file_stem_parts) >= 4, msg

    # Create metadata dict for SQL INSERT
    metadata = {}

    # Reformat date to YYYY-MM-DD (from YYYYMMDD)
    metadata["exp_date"] = date.fromisoformat(file_stem_parts[0]).isoformat()

    metadata["mouse_id"] = file_stem_parts[1]
    metadata["exp_name"] = file_stem_parts[2]
    metadata["first_acq"] = file_stem_parts[-1]
    metadata["n_acq"] = len(file_paths)

    *_, metadata["last_acq"] = file_paths[-1].stem.split("_")

    # Get metadata from the first raw TIFF file
    tif = TiffFile(file_paths[0])
    SI_metadata = tif.scanimage_metadata["FrameData"]

    # Get size from tags (instead of shape)
    metadata["frame_count"] = len(tif.pages)

    metadata["width_px"] = tif.pages[0].tags["ImageWidth"].value
    metadata["height_px"] = tif.pages[0].tags["ImageLength"].value
    metadata["frame_rate"] = SI_metadata["SI.hRoiManager.scanFrameRate"]

    # Assume unit is centimeters
    dx, nx = tif.pages[0].tags["XResolution"].value
    dy, ny = tif.pages[0].tags["YResolution"].value

    # um per pixels in each direction
    factor_x = round(1e4 * nx / dx, 4)
    factor_y = round(1e4 * ny / dy, 4)

    # Size of image in um
    metadata["width_um"] = metadata["width_px"] * factor_x
    metadata["height_um"] = metadata["height_px"] * factor_y

    # Create acquisition dict for SQL INSERT
    acquisitions = [
        {"raw_filename": file_path.name, "should_include": True}
        for file_path in file_paths
    ]

    # ----------------------------------------------------------------------- #
    # Create SQL DB and store metadata
    # ----------------------------------------------------------------------- #

    con = sqlite3.connect(db_path)

    with con:
        # Create the database with the specified format
        with open(Path(__file__).parent / "create.sql") as f:
            con.executescript(f.read())

        # Add metadata to the database
        insertion_query = create_insertion_query("metadata", metadata)
        con.execute(insertion_query, metadata)

        # Add acquisitions to the database
        insertion_query = create_insertion_query("acquisitions", acquisitions)
        con.executemany(insertion_query, acquisitions)

    return con


def create_insertion_query(table_name, data):
    # HACK: ONLY FOR INTERNAL USE (CAN BE USED FOR SQL INJECTION)

    template = data[0] if isinstance(data, list) else data

    return (
        f"INSERT INTO {table_name} "
        f"({", ".join(template.keys())}) "
        f"VALUES (:{", :".join(template.keys())});"
    )
