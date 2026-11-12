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


def test_the_network_runs_in_float32_on_every_cpu(monkeypatch):
    pytest.importorskip("cellpose")
    import torch

    from cellpose import models

    built = {}

    class Network:
        """Stands in for the model: what it was built with, and zeros."""

        device = torch.device("cpu")

        def __init__(self, **options):
            built.update(options, threads=torch.get_num_threads())

        def eval(self, image, **options):
            maps = np.zeros((3, *image.shape), np.float32)
            return None, (None, maps[:2], maps[2]), None

    monkeypatch.setattr(models, "CellposeModel", Network)
    before = torch.get_num_threads()
    flows, prob, settings = segmentation.run_network(np.ones((8, 9)), "cpdino")

    assert built["use_bfloat16"] is False and settings["dtype"] == "float32"
    assert built["threads"] == settings["threads"] == segmentation._usable_cpus()
    assert torch.get_num_threads() == before  # put back for what runs next
    assert flows.shape == (2, 8, 9) and prob.shape == (8, 9)


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


def drawn_candidates() -> segmentation.Candidates:
    """
    Candidates drawn by hand, so no Cellpose is needed: three round basins,
    the probability falling by 0.8 per px from the center (-5 outside them).
    Basins 1 (radius 7, peak 4.8) and 2 (radius 6, peak 4) are cells at the
    defaults, inside radius 6 and 5, so both can grow; basin 3 is a dim spot
    (radius 4, peak 2) whose cell fails the flow check. 1 um/px.
    """
    yy, xx = np.mgrid[:30, :44]
    basins = np.zeros((30, 44), np.uint32)
    prob = np.full((30, 44), -5.0, np.float32)

    for number, (cy, cx, radius, top) in enumerate(
        [(10, 10, 7, 4.8), (12, 27, 6, 4.0), (24, 38, 4, 2.0)], start=1
    ):
        distance = np.hypot(yy - cy, xx - cx)
        inside = distance <= radius
        basins[inside] = number
        prob[inside] = top - 0.8 * distance[inside]

    peak = np.array(
        [np.argmax(np.where(basins == b, prob, -np.inf)) for b in (1, 2, 3)]
    )
    found = segmentation.Candidates(
        prob=prob,
        flows=np.zeros((2, 30, 44), np.float32),
        basins=basins,
        peak=peak,
        area=np.zeros(3, np.int64),
        flow_error=np.array([0.1, 0.2, 0.9], np.float32),
        settings={"um_per_px": [1.0, 1.0]},
    )
    found.area[:] = np.bincount(segmentation.cut(found, 0.0).ravel(), minlength=4)[1:]

    return found


def drawn_image(found: segmentation.Candidates) -> np.ndarray:
    """
    A mean image like the drawn candidates (`drawn_candidates`), with a spot
    where no basin is: a Gaussian 4 high, of sigma 1.5 px, at (3, 39).
    """
    yy, xx = np.mgrid[:30, :44]
    spot = 4 * np.exp(-((yy - 3) ** 2 + (xx - 39) ** 2) / (2 * 1.5**2))

    return (found.prob + 6 + spot).astype(np.float32)


def test_the_curation_starts_from_the_cells_cellpose_keeps():
    found = drawn_candidates()
    curation = segmentation.Curation(found)

    assert sorted(curation.cells) == [1, 2]
    assert all(
        cell.kind == "cellpose" and cell.kept for cell in curation.cells.values()
    )
    np.testing.assert_array_equal(curation.owner, segmentation.default_cells(found))
    labels = curation.labels()
    assert sorted(np.unique(labels)) == [0, 1, 2]
    assert curation.cell_at(10, 10) == 1 and curation.cell_at(0, 0) is None

    # Without the flow check, the dim spot is a cell too
    curation.set_flow_check(False)
    assert sorted(curation.cells) == [1, 2, 3]


def test_a_flood_passes_through_every_threshold():
    found = drawn_candidates()

    for basin in (1, 2, 3):
        pixels, level = segmentation.flood(found, basin)
        assert np.all(np.diff(level) <= 0)  # never rises

        for threshold in (-3.0, -1.0, 0.0, 0.5, 2.0):
            size = int((level > threshold).sum())
            cut = segmentation.cut(found, threshold, [basin])
            assert set(pixels[:size]) == set(np.flatnonzero(cut))


def test_the_threshold_sizes_every_cell_but_those_given_a_size():
    curation = segmentation.Curation(drawn_candidates())
    start = {b: len(cell.pixels) for b, cell in curation.cells.items()}

    curation.resize(2, 20)
    curation.set_threshold(-2.0)
    assert len(curation.cells[1].pixels) > start[1]
    assert len(curation.cells[2].pixels) == 20
    assert curation.cells[2].kind == "resized"
    assert curation.cells[2].threshold > -2.0  # its weakest pixel

    curation.follow_threshold(2)
    assert len(curation.cells[2].pixels) > start[2]

    # A cell under 15 px at the threshold is dropped, as Cellpose drops it
    curation.set_threshold(4.0)
    assert not curation.cells


def test_a_click_adds_the_candidate_there_or_grows_the_cell_of_its_basin():
    found = drawn_candidates()
    curation = segmentation.Curation(found)

    # The dim spot, clicked at its edge (probability -0.4, under the
    # threshold): added, grown from the click, which it holds. It starts
    # halfway between the floor and its peak (-0.5), not as a speck.
    basin = curation.add(24, 41)
    cell = curation.cells[basin]
    assert basin == 3 and cell.kind == "added"
    assert curation.seed(3) == 24 * 44 + 41
    assert 24 * 44 + 41 in set(cell.pixels)
    assert cell.threshold > -0.5 and len(cell.pixels) > 20

    # Just outside cell 1, in its basin: cell 1 grows until it holds the click
    curation.set_threshold(2.0)
    before = len(curation.cells[1].pixels)
    y, x = 10, 10 + 4  # probability 4.8 - 3.2 = 1.6, under the threshold
    assert curation.cell_at(y, x) is None
    assert curation.add(y, x) == 1
    assert curation.cell_at(y, x) == 1
    assert len(curation.cells[1].pixels) > before

    # A click by a rejected cell brings it back
    curation.reject(1)
    assert curation.add(y, x + 1) == 1 and curation.cells[1].kept

    # Nowhere near a basin
    assert curation.add(0, 43) is None


def test_rejected_and_excluded_cells_stay_but_are_not_kept():
    curation = segmentation.Curation(drawn_candidates())
    areas = {b: len(cell.pixels) for b, cell in curation.cells.items()}

    curation.reject(1)
    assert curation.cells[1].rejected and [c.basin for c in curation.kept()] == [2]
    assert curation.cell_at(10, 10) == 1  # still there to click on
    assert list(curation.table()["basin"]) == [2]

    curation.restore(1)
    curation.set_area_range(min_um2=areas[2] + 1)
    assert curation.cells[2].excluded and [c.basin for c in curation.kept()] == [1]
    curation.set_area_range(max_um2=areas[2])
    assert [c.basin for c in curation.kept()] == [2]


def test_a_click_where_no_basin_is_grows_a_cell_from_the_image():
    found = drawn_candidates()
    curation = segmentation.Curation(found, image=drawn_image(found))

    # Next to the spot: a new cell, numbered above the basins, from the
    # brightest pixel near the click, as large as the spot at half its height
    cell = curation.add(4, 40)
    grown = curation.cells[cell]
    assert cell == 4 and grown.kind == "image" and np.isnan(grown.threshold)
    assert curation.seed(cell) == 3 * 44 + 39 and curation.from_image(cell)
    assert 10 < len(grown.pixels) < 40
    assert not found.basins.flat[grown.pixels].any()  # where no basin is
    assert curation.cell_at(3, 39) == cell

    # Resized, rejected and kept as the others are
    curation.resize(cell, 5)
    assert len(curation.cells[cell].pixels) == 5
    table = curation.table()
    assert list(table["kind"]) == ["cellpose", "cellpose", "image"]
    assert list(table["basin"]) == [1, 2, 0]  # no basin
    edits = segmentation.Edits.from_json(curation.edits.as_json())
    assert edits.grown == {cell: 3 * 44 + 39} and edits.sizes[cell] == 5
    curation.reject(cell)
    assert [c.basin for c in curation.kept()] == [1, 2]

    # Without the image, there is nothing to grow it from
    assert segmentation.Curation(found).add(4, 40) is None


def test_two_grown_cells_never_overlap_and_the_first_keeps_its_pixels():
    found = drawn_candidates()
    curation = segmentation.Curation(found, image=drawn_image(found))
    first = curation.grow(3, 39)
    second = curation.grow(3, 43)
    assert (first, second) == (4, 5)

    def overlap() -> int:
        sizes = sum(len(cell.pixels) for cell in curation.cells.values())
        return sizes - np.count_nonzero(curation.owner)

    assert overlap() == 0

    # The first grows over what the second had: the second gives it up
    taken = set(curation.cells[second].pixels)
    curation.resize(first, len(curation.flood(first)[0]))
    assert overlap() == 0 and taken & set(curation.cells[first].pixels)


def test_a_cell_switches_between_its_basin_and_the_image():
    found = drawn_candidates()
    curation = segmentation.Curation(found, image=drawn_image(found))
    before = curation.cells[1].pixels.copy()

    curation.use_basin(1, False)
    cell = curation.cells[1]
    assert cell.kind == "image" and curation.from_image(1)
    assert set(found.basins.flat[cell.pixels]) <= {0, 1}  # its basin, or none

    curation.use_basin(1, True)
    assert curation.cells[1].kind == "cellpose"
    np.testing.assert_array_equal(curation.cells[1].pixels, before)

    # A cell grown where no basin is has none to use
    spot = curation.grow(4, 40)
    curation.use_basin(spot, True)
    assert curation.from_image(spot)


def test_putting_edits_back_undoes_them_and_they_read_back_as_json():
    curation = segmentation.Curation(drawn_candidates())
    labels = curation.labels()
    before = curation.edits  # a copy

    curation.add(24, 38)
    curation.resize(1, 12)
    curation.reject(2)
    curation.set_area_range(5.0, 500.0)
    edits = segmentation.Edits.from_json(curation.edits.as_json())
    assert edits == curation.edits and before == segmentation.Edits()

    curation.edits = before
    np.testing.assert_array_equal(curation.labels(), labels)

    curation.edits = edits
    assert sorted(curation.cells) == [1, 2, 3] and curation.cells[2].rejected
    table = curation.table()
    assert list(table["basin"]) == [1, 3]
    assert list(table["kind"]) == ["resized", "added"]
    assert list(table["pixels"])[0] == 12


def test_a_cell_undone_or_left_out_by_the_flow_check_is_there_no_more():
    found = drawn_candidates()
    curation = segmentation.Curation(found, image=drawn_image(found))
    before = curation.edits
    there = [curation.may_have_cell(number) for number in range(5)]
    assert there == [False, True, True, False, False]

    # The dim spot, without the flow check or added; a cell grown where no
    # basin is, numbered above the basins
    curation.set_flow_check(False)
    assert curation.may_have_cell(3)
    curation.set_flow_check(True)
    curation.add(24, 38)
    spot = curation.grow(4, 40)
    assert curation.may_have_cell(3) and curation.may_have_cell(spot)

    # Both undone: neither may be there, and the grown one has no seed
    curation.edits = before
    assert not curation.may_have_cell(3) and not curation.may_have_cell(spot)

    for where in (curation.seed, curation.flood):
        with pytest.raises(ValueError, match="grown from the image is numbered 4"):
            where(spot)


def test_the_app_finds_the_files_of_a_group_where_the_group_writes_them(tmp_path):
    from generate_data import a_tested_experiment

    db, group_id = a_tested_experiment(tmp_path)
    group = db.groups[group_id]

    assert segmentation.group_files(tmp_path, "20260304/m1/e1", group_id) == (
        group._mean_image_paths()[0],
        group._candidates_path(),
    )
