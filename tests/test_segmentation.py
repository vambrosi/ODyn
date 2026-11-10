"""
`odyn.segmentation`: the cell candidates from Cellpose, here from flows made
out of known cells instead of the network, so no model weights are needed.
Following the flows needs Cellpose itself; those tests are skipped without
it. `Group.save_cellpose_candidates` is tested beside the mean images, in
`test_mcor_test.py`.
"""

import numpy as np
import pytest

from odyn import segmentation


def known_cells(height=48, width=40):
    """
    Three cells, the first two touching, with what the network would say
    about them: flows toward their centers (five times unit length, as the
    network gives them) and a cell probability falling from 6 at the center
    to about 0.4 at the edge, -6 outside.
    """
    import torch

    from cellpose import dynamics

    yy, xx = np.mgrid[:height, :width]
    labels = np.zeros((height, width), np.int32)
    prob = np.full((height, width), -6.0, np.float32)

    for number, (cy, cx, radius) in enumerate([(12, 12, 7), (12, 25, 6), (34, 20, 7)]):
        distance = np.hypot(yy - cy, xx - cx)
        inside = (distance <= radius) & (labels == 0)
        labels[inside] = number + 1
        prob[inside] = 6 - 0.8 * distance[inside]

    unit = dynamics.masks_to_flows_gpu(labels, device=torch.device("cpu"))
    flows = 5 * np.asarray(unit[0] if isinstance(unit, tuple) else unit, np.float32)

    return labels, flows, prob


def test_a_basin_cut_at_a_threshold_is_what_cellpose_finds_there():
    pytest.importorskip("cellpose")
    import torch

    from cellpose import dynamics

    labels, flows, prob = known_cells()
    found = segmentation.candidates_from_maps(flows, prob)

    # Each known cell is one basin, and a cell at the defaults
    assert list(found.cells) == [1, 2, 3]
    cells = segmentation.default_cells(found)

    for number in (1, 2, 3):
        basin = np.bincount(cells[labels == number]).argmax()
        both = ((cells == basin) & (labels == number)).sum()
        either = ((cells == basin) | (labels == number)).sum()
        assert both / either > 0.98

    # Lower and higher thresholds give the pixels Cellpose gives there
    for threshold in (-1.0, 2.0):
        ours = segmentation.cut(found, threshold, found.cells)
        theirs = dynamics.compute_masks(
            flows,
            prob,
            niter=segmentation.NITER,
            cellprob_threshold=threshold,
            flow_threshold=0,
            min_size=-1,
            device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(ours > 0, theirs > 0)


def test_candidates_read_back_as_written(tmp_path):
    rng = np.random.default_rng(0)
    found = segmentation.Candidates(
        prob=rng.normal(size=(6, 5)).astype(np.float32),
        flows=rng.normal(size=(2, 6, 5)).astype(np.float32),
        basins=rng.integers(0, 3, size=(6, 5)).astype(np.uint32),
        peak=np.array([4, 17]),
        area=np.array([20, 3]),
        flow_error=np.array([0.1, np.nan], np.float32),
        settings={"model": "cpdino", "tile_px": None},
    )
    found.write(tmp_path / "c.h5")
    back = segmentation.read_candidates(tmp_path / "c.h5")

    np.testing.assert_array_equal(back.basins, found.basins)
    np.testing.assert_allclose(back.prob, found.prob, atol=0.01)  # float16
    np.testing.assert_allclose(back.flows, found.flows, atol=0.01)
    np.testing.assert_array_equal(back.peak, found.peak)
    np.testing.assert_array_equal(back.flow_error, found.flow_error)
    assert back.settings == found.settings
    assert list(back.cells) == [1]  # the second is too small
