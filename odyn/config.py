import sqlite3

from pathlib import Path
from datetime import date, datetime
from tifffile import TiffFile

from .const import INFO, FAIL
from .utils import ProgressBar


def create_db(path: str | Path) -> None:
    # Get all relevant paths
    db_path = Path(path)
    exp_path = path.parent
    raw_path = exp_path / "raw"

    # ----------------------------------------------------------------------- #
    # Gather all the metadata from raw files and file names
    # ----------------------------------------------------------------------- #
    # I used preProcessing_v2.m and other scripts in that file as a baseline
    # for what metadata has to be collected, and what needs to be checked.
    #
    # TODO: Make sure all relevant data is added to the db.
    # ----------------------------------------------------------------------- #

    print(f"[{INFO}] Getting metadata from files...")

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

    # Gets the part of the filename before the acquisition number
    metadata["tiff_stem"] = "_".join(file_stem_parts[:-1])

    # Reformat date to YYYY-MM-DD (from YYYYMMDD)
    metadata["exp_date"] = date.fromisoformat(file_stem_parts[0]).isoformat()

    metadata["mouse_id"] = file_stem_parts[1]
    metadata["exp_name"] = file_stem_parts[2]
    metadata["first_acq"] = file_stem_parts[-1]
    metadata["n_acq"] = len(file_paths)

    *_, metadata["last_acq"] = file_paths[-1].stem.split("_")

    # Get metadata from the first raw TIFF file
    tif = TiffFile(file_paths[0])

    # We only support ScanImage files
    assert (
        tif.scanimage_metadata is not None
    ), "Did not find ScanImage metadata in the first TIFF file"
    SI_metadata = tif.scanimage_metadata["FrameData"]

    # Get size from tags (instead of shape)
    # TODO: What is better? And why the redundancy?
    #           len(tif.pages) or
    #           SI_metadata["SI.hStackManager.framesPerSlice"]
    metadata["frame_count"] = len(tif.pages)

    metadata["width_px"] = tif.pages[0].tags["ImageWidth"].value
    metadata["height_px"] = tif.pages[0].tags["ImageLength"].value
    metadata["frame_rate"] = SI_metadata["SI.hRoiManager.scanFrameRate"]

    laser_powers = SI_metadata["SI.hBeams.powers"]
    metadata["laser_power_920"] = laser_powers[0]
    metadata["laser_power_1040"] = laser_powers[1]

    # TODO: Make variable name clearer
    metadata["loop_acq_interval_s"] = SI_metadata["SI.loopAcqInterval"]

    # Assume unit is centimeters
    # TODO: Check that units are centimeters (throw error otherwise)
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

    # Get metadata for each acquisition
    bar = ProgressBar(len(file_paths))
    bar.show()

    for acq, file_path in zip(acquisitions, file_paths):
        tif = TiffFile(file_path)

        # Get frame count for each acquisition
        acq["frame_count"] = len(tif.pages)

        # Parse ImageDescription
        first_frame_metadata = dict(
            line.split(" = ")
            for line in tif.pages[0].tags["ImageDescription"].value.splitlines()
        )

        # Store relevant metadata
        acq["first_frame_start_s"] = first_frame_metadata["frameTimestamps_sec"]

        date_string = first_frame_metadata["epoch"].strip("[]")
        date_string = " ".join(date_string.split())
        dt = datetime.strptime(date_string, "%Y %m %d %H %M %S.%f")

        acq["loop_start_datetime"] = dt.isoformat()

        bar.step()
    bar.end()

    # Perform a couple of sanity checks
    print(f"[{INFO}] Performing metadata sanity checks...")

    frame_count = acquisitions[0]["frame_count"]
    loop_start = acquisitions[0]["loop_start_datetime"]
    fails_count = 0

    for acq in acquisitions:
        if frame_count != acq["frame_count"]:
            print(
                f"[{FAIL}] File {acq["raw_filename"]} doesn't have the "
                "same number of frames as its predecessor!"
            )
            frame_count = acq["frame_count"]
        if loop_start != acq["loop_start_datetime"]:
            print(
                f"[{FAIL}] File {acq["raw_filename"]} doesn't have the "
                "same loop start datetime as its predecessor!"
            )
            loop_start = acq["loop_start_datetime"]

    # TODO: Maybe create a function to split an exp database along an
    #       acquisition. If this check is here, there might be an usecase.
    if fails_count > 0:
        print(
            f"[{INFO}] We will create the database despite the check "
            "fail. Split the database afterwards if necessary."
        )
    else:
        print(f"[{INFO}] Passed all checks!")

    # ----------------------------------------------------------------------- #
    # Create SQL DB and store metadata
    # ----------------------------------------------------------------------- #
    # So far, there is one database per experiment.
    #
    # TODO: figure out if we should keep that, and aggregate results from
    #       time to time, or if we should just have one db.
    #
    #       Things to consider: some times folders get moved. So having one
    #       db referencing those files might break code. One alternative is
    #       to regularly crawl the file tree in the server looking for
    #       odyn.db and aggregate those. It would also allow targeted tests,
    #       where you only add a selection of experiment folders.
    # ----------------------------------------------------------------------- #

    print(f"[{INFO}] Creating database...")

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

    print(f"[{INFO}] Database created.")

    return con


def create_insertion_query(table_name, data):
    # HACK: ONLY FOR INTERNAL USE (CAN BE USED FOR SQL INJECTION)

    template = data[0] if isinstance(data, list) else data

    return (
        f"INSERT INTO {table_name} "
        f"({", ".join(template.keys())}) "
        f"VALUES (:{", :".join(template.keys())});"
    )
