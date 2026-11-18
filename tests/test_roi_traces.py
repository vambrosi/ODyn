"""
`Group.save_roi_traces` and `Group.roi_traces`: the traces of ROIs and of
the neuropil around them, on movies known pixel by pixel, and the neuropil
taken as suite2p takes it (`neuropil_masks`).
"""

import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile

from test_add_mcor_files import FRAMES, HEIGHT, WIDTH, build, write_mcor

from odyn.groups import neuropil_masks
from odyn.segmentation import Curated, Edits

# Where the two ROIs and a cell not kept are (rows, columns)
FIRST = (slice(10, 14), slice(10, 14))
SECOND = (slice(40, 46), slice(50, 55))
OTHER = (slice(10, 14), slice(18, 21))  # near the first, inside its neuropil


def with_onsets(db, frame=2):
    """Every acquisition of the group with its odor on at `frame`."""
    with db._locked() as con, con:
        con.execute("""
            INSERT INTO method_calls
                ( method_call_id, group_id, user, method_name, module, code
                , parameter_inputs, parameters_used )
                VALUES (900, 0, 'test', 'Database.process_sync', 'odyn.database'
                       , '{}', '{}', '{}');
            """)
        con.execute(
            "INSERT INTO streams (stream_id, exp_id, stream_type, file_path, "
            "method_call_id) VALUES (1, 1, 'sync', 'sync.h5', 900);"
        )

        for acq_id in (1, 2, 3):
            con.execute(
                """
                INSERT INTO sync_blocks
                    ( stream_id, sync_block, acq_id, sync_start_sample
                    , sync_frame_count, sync_frame_period_s, sync_odor_pulses
                    , sync_odor_on_frame, method_call_id )
                    VALUES (1, ?, ?, 0, ?, 0.07, 1, ?, 900);
                """,
                [acq_id, acq_id, FRAMES, frame],
            )


def movie(acq_id: int) -> np.ndarray:
    """
    Frames of acquisition `acq_id`: the background at `10 * acq_id`, the
    first ROI at `100 + frame`, the second at `200 + frame`, and the cell
    not kept far brighter, so a neuropil that took it would show.
    """
    frames = np.full((FRAMES, HEIGHT, WIDTH), 10.0 * acq_id, np.float32)

    for frame in range(FRAMES):
        frames[frame][FIRST] = 100 + frame
        frames[frame][SECOND] = 200 + frame
        frames[frame][OTHER] = 5000

    return frames


@pytest.fixture
def group(tmp_path):
    """A group of three approved acquisitions with odor onsets and ROIs."""
    db, group = build(tmp_path)
    with_onsets(db)  # before the group reads its tables

    for acq_id in (1, 2, 3):
        tifffile.imwrite(write_mcor(tmp_path, acq_id), movie(acq_id))

    group.add_mcor_files()
    group.approve_mcor_files()

    labels = np.zeros((HEIGHT, WIDTH), np.uint32)
    labels[FIRST], labels[SECOND] = 1, 2
    cells = labels > 0
    cells[OTHER] = True
    table = pd.DataFrame(
        {"label": [1, 2], "kind": ["cellpose", "image"], "pixels": [16, 30]}
    )
    curated = Curated(labels, cells, table, Edits(), "fingerprint")
    curated.write(tmp_path / "rois.h5", user="alice")
    group.import_mask(mask_path="rois.h5")

    return group


def test_the_traces_of_rois_and_their_neuropil_are_saved_per_call(group):
    path = group.save_roi_traces(photobleach_window_s=0.0)
    assert path.name.endswith(
        f"_roi_traces_{group.latest_calls('save_roi_traces').index[0]}.h5"
    )

    traces = group.roi_traces()
    assert traces.F.shape == traces.Fneu.shape == (3, FRAMES, 2)
    np.testing.assert_array_equal(traces.times * 14.0, np.arange(-2, FRAMES - 2))

    # Each ROI its own pixels, frame by frame; the neuropil only background,
    # whatever the cell not kept next to the first ROI does
    np.testing.assert_allclose(traces.F[:, :, 0], [100 + np.arange(FRAMES)] * 3)
    np.testing.assert_allclose(traces.F[:, :, 1], [200 + np.arange(FRAMES)] * 3)
    np.testing.assert_allclose(traces.Fneu[:, 0, 0], [10, 20, 30])
    np.testing.assert_allclose(traces.Fneu[:, :, 1], traces.Fneu[:, :, 0])

    # The table of the mask file, with the pixels of each neuropil
    assert list(traces.rois["kind"]) == ["cellpose", "image"]
    assert (traces.rois["neuropil_pixels"] > 350).all()  # 1 um/px: 350 um2
    assert "odor" in traces.acquisitions.columns  # no trials here: empty

    with h5py.File(path) as handle:
        assert list(handle["odor_ids"]) == [-1, -1, -1]
        assert list(handle["odors"].asstr()[()]) == ["", "", ""]

    # A second call keeps the first file; the last one is read
    again = group.save_roi_traces(photobleach_window_s=0.0)
    assert again != path and path.is_file()
    assert group.roi_traces().method_call_id > traces.method_call_id

    # The approval read is linked to the call, as the mask is
    calls = group.latest_calls("save_roi_traces")
    consumed = calls["consumed_calls"].iloc[0]
    imported = group.latest_calls("import_mask").index[0]
    approved = group.latest_calls("approve_mcor_files").index[0]
    assert {int(imported), int(approved)} <= set(consumed)


def test_the_neuropil_leaves_out_the_cells_and_a_gap_and_grows_until_full():
    from scipy import ndimage as ndi

    labels = np.zeros((40, 40), np.uint32)
    labels[18:22, 18:22] = 1
    cells = np.zeros((40, 40), bool)
    cells[18:22, 24:28] = True

    # What may be taken: no ROI, no cell, more than 2 steps from the ROI
    steps = ndi.distance_transform_cdt(labels == 0, metric="taxicab")
    allowed = (labels == 0) & ~cells & (steps > 2)

    [ring] = neuropil_masks(labels, [1], cells, gap_px=2, min_px=100, step_px=3)
    taken = np.zeros((40, 40), bool)
    taken.flat[ring] = True

    assert len(ring) > 100 and not (taken & ~allowed).any()
    assert taken[16, 16] and not taken[17, 17]  # 4 steps away, and 2

    # A square from the ROI and its gap, grown by 3 px on every side
    rows, columns = np.nonzero(taken)
    assert (rows.min(), rows.max(), columns.min(), columns.max()) == (13, 26, 13, 26)

    # Asked for more than the image has, it takes all it can; a missing
    # label has none
    [whole, none] = neuropil_masks(
        labels, [1, 5], cells, gap_px=2, min_px=10**6, step_px=3
    )
    np.testing.assert_array_equal(np.sort(whole), np.flatnonzero(allowed))
    assert len(none) == 0
