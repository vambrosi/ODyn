"""
Cell candidates from Cellpose, for curating the ROIs of a group by hand.

Cellpose runs once, on the group mean image. Its network gives every pixel a
cell probability (a logit: 0 is even odds) and a flow toward the center of
the cell it belongs to. Cellpose then follows the flows from every pixel
above a probability threshold, and the pixels that end at the same spot form
one cell; cells whose shape disagrees with the flows are dropped (the flow
check), and so are cells under 15 px.

Following the flows once from every pixel above a low threshold (`FLOOR`)
gives each candidate cell its basin, every pixel that flows to it. A cell at
a higher threshold is then the pixels of its basin above that threshold,
connected to its brightest pixel, which matches Cellpose run at that
threshold. So cells can be resized, added and dropped later without the
network (`Curation`).

**USAGE**
```python
from odyn.segmentation import Curation, find_candidates, read_candidates

found = find_candidates(image, weights)  # the network, then the basins
found.write(path)

curation = Curation(read_candidates(path))  # the Cellpose defaults
curation.set_threshold(-1.0)                # every cell bigger
basin = curation.add(120, 85)               # the candidate under a point
curation.resize(basin, 150)                 # 150 px
curation.reject(curation.cell_at(40, 32))
labels = curation.labels()                  # the cells kept, numbered from 1
```

`find_candidates` needs Cellpose 4 (and its `torch`); the rest needs only
numpy, scipy, pandas and h5py.
"""

from __future__ import annotations

import copy
import heapq
import json
import os

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from scipy import ndimage as ndi

FLOOR = -3.0  # the lowest threshold kept: every basin pixel is above it
THRESHOLD = 0.0  # the Cellpose default
FLOW_THRESHOLD = 0.4  # the Cellpose default flow check
MIN_SIZE = 15  # px; Cellpose drops smaller cells
NITER = 200  # steps along the flows, the Cellpose default without a diameter
# The Cellpose 4 models, each a file named as the model in the
# `.odyn/segmentation/` folder of the main folder
MODELS = ("cpdino", "cpdino-vitb", "cpsam", "cpsam_v2")
WEIGHTS_FOLDER = "segmentation"
# The model for a group, by the depth goal of its experiments (`depth_goal`):
# cpdino, the pick of a survey of superficial fields, or cpsam for deep ones,
# where cpdino takes the haze for cells and finds few somas or none
MODEL = "cpdino"
DEEP_MODEL = "cpsam"


@dataclass
class Candidates:
    """
    One Cellpose run, on the grid of the image it ran on:
    - `prob`: the cell probability of every pixel (logits);
    - `flows`: the flows (2 x height x width, y then x);
    - `basins`: for every pixel, the number of its basin (`1` to `n`), or `0`
      where there is none;
    - per basin, index `i` for basin `i + 1`: `peak`, its brightest pixel (an
      index into the flattened image), and the `area` (px) and `flow_error`
      of its cell at the default threshold (`0` and NaN when it has none);
    - `settings`: how it was made.
    """

    prob: np.ndarray
    flows: np.ndarray
    basins: np.ndarray
    peak: np.ndarray
    area: np.ndarray
    flow_error: np.ndarray
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def cells(self) -> np.ndarray:
        """The basins that are cells at the Cellpose defaults."""
        passed = (self.area >= MIN_SIZE) & (self.flow_error <= FLOW_THRESHOLD)

        return np.flatnonzero(passed) + 1

    def write(self, path: str | Path) -> None:
        """Write it all to one HDF5 file (`read_candidates` reads it back)."""
        packed = {"compression": "gzip", "compression_opts": 4, "shuffle": True}

        with h5py.File(path, "w") as handle:
            handle.create_dataset("prob", data=self.prob.astype(np.float16), **packed)
            handle.create_dataset("flows", data=self.flows.astype(np.float16), **packed)
            handle.create_dataset(
                "basins", data=self.basins.astype(np.uint32), **packed
            )
            handle.create_dataset("peak", data=self.peak.astype(np.int64))
            handle.create_dataset("area", data=self.area.astype(np.int64))
            handle.create_dataset("flow_error", data=self.flow_error.astype(np.float32))
            handle.attrs["settings"] = json.dumps(self.settings)


def read_candidates(path: str | Path) -> Candidates:
    """The candidates `Candidates.write` wrote."""
    with h5py.File(path, "r") as handle:
        return Candidates(
            prob=np.asarray(handle["prob"], np.float32),
            flows=np.asarray(handle["flows"], np.float32),
            basins=np.asarray(handle["basins"]),
            peak=np.asarray(handle["peak"]),
            area=np.asarray(handle["area"]),
            flow_error=np.asarray(handle["flow_error"]),
            settings=json.loads(handle.attrs["settings"]),
        )


def find_candidates(
    image: np.ndarray, weights: str | Path, *, tile_px: None | int = None
) -> Candidates:
    """
    Run Cellpose 4 on `image` (one picture) with the model in the file
    `weights`, and find the candidate cells (see the module docstring).
    `tile_px` normalizes the brightness in tiles this wide, which lifts dark
    regions; `None` normalizes the whole image at once. Runs on a GPU when
    there is one.
    """
    flows, prob, settings = run_network(image, weights, tile_px=tile_px)
    found = candidates_from_maps(flows, prob)
    found.settings = {**settings, **found.settings}

    return found


def run_network(
    image: np.ndarray, weights: str | Path, *, tile_px: None | int = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    The Cellpose network output for `image`: the flows (2 x height x width),
    the cell probability (logits) and the settings used. In float32, on
    every CPU this process may use (or a GPU).
    """
    # Cellpose and torch are heavy, and only this needs them
    import cellpose
    import torch

    from cellpose import models

    # Every CPU, whatever the environment says: the caiman environment holds
    # torch to one thread (for CaImAn), and nothing else runs meanwhile
    threads = torch.get_num_threads()
    torch.set_num_threads(_usable_cpus())

    try:
        # Not bfloat16, the Cellpose default: a CPU without it in hardware
        # emulates it, three times slower on the cluster, for the same cells
        model = models.CellposeModel(
            gpu=True, pretrained_model=str(weights), use_bfloat16=False
        )
        normalize: Any = (
            True if tile_px is None else {"tile_norm_blocksize": int(tile_px)}
        )
        _, (_, flows, prob), _ = model.eval(
            np.asarray(image, np.float32), normalize=normalize, compute_masks=False
        )
        settings = {
            "model": Path(weights).name,
            "tile_px": tile_px,
            "cellpose": cellpose.version,
            "torch": torch.__version__,
            "device": str(model.device),
            "dtype": "float32",
            "threads": torch.get_num_threads(),
        }

    finally:
        torch.set_num_threads(threads)

    return np.asarray(flows, np.float32), np.asarray(prob, np.float32), settings


def _usable_cpus() -> int:
    """The CPUs this process may run on (those of the computer, if unknown)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def candidates_from_maps(
    flows: np.ndarray, prob: np.ndarray, *, niter: int = NITER
) -> Candidates:
    """
    The candidates from the network output: the basins (Cellpose followed
    from every pixel above `FLOOR`), and the brightest pixel, the area and
    the flow error of each one at the default threshold.
    """
    import torch

    from cellpose import dynamics

    device = torch.device("cpu")
    flows = np.asarray(flows, np.float32)
    prob = np.asarray(prob, np.float32)
    raw = dynamics.compute_masks(
        flows,
        prob,
        niter=niter,
        cellprob_threshold=FLOOR,
        flow_threshold=0,  # no flow check, no minimum size: every basin
        min_size=-1,
        device=device,
    )
    # Numbered 1 to n with no gaps, which `dynamics.flow_error` needs too
    ids, basins = np.unique(raw, return_inverse=True)
    basins = basins.reshape(raw.shape).astype(np.uint32)

    if ids[0] != 0:
        basins += 1

    n = int(basins.max())
    positions = ndi.maximum_position(prob, basins, np.arange(1, n + 1)) if n else []
    peak = np.array([y * prob.shape[1] + x for y, x in positions], np.int64)
    found = Candidates(
        prob=prob,
        flows=flows,
        basins=basins,
        peak=peak,
        area=np.zeros(n, np.int64),
        flow_error=np.full(n, np.nan, np.float32),
        settings={
            "floor": FLOOR,
            "threshold": THRESHOLD,
            "flow_threshold": FLOW_THRESHOLD,
            "min_size": MIN_SIZE,
            "niter": niter,
        },
    )

    # Each basin at the default threshold, checked as Cellpose checks it
    labels = cut(found, THRESHOLD)
    present = np.unique(labels)
    present = present[present > 0]

    if present.size:
        sequential = np.zeros(labels.shape, np.int32)
        inside = labels > 0
        sequential[inside] = np.searchsorted(present, labels[inside]) + 1
        errors, _ = dynamics.flow_error(sequential, flows, device=device)
        found.area[:] = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        found.flow_error[present - 1] = errors

    return found


def cut(
    found: Candidates, threshold: float, basins: None | Iterable[int] = None
) -> np.ndarray:
    """
    Each basin at `threshold`: its pixels above it, connected (by sides, not
    corners) to its brightest pixel. A label image of basin numbers, `0`
    elsewhere; `basins` limits it to some.
    """
    labels = np.zeros(found.basins.shape, np.uint32)
    boxes = ndi.find_objects(found.basins)
    width = found.basins.shape[1]
    wanted = range(1, len(boxes) + 1) if basins is None else basins

    for basin in wanted:
        box = boxes[basin - 1]
        y, x = divmod(int(found.peak[basin - 1]), width)

        if box is None or not found.prob[y, x] > threshold:
            continue

        inside = (found.basins[box] == basin) & (found.prob[box] > threshold)
        pieces, _ = ndi.label(inside)  # joined by sides, as Cellpose cells are
        piece = pieces[y - box[0].start, x - box[1].start]
        labels[box][pieces == piece] = basin

    return labels


def default_cells(found: Candidates) -> np.ndarray:
    """The cells Cellpose keeps at its defaults, as a label image of basins."""
    return cut(found, THRESHOLD, found.cells)


def group_files(
    main_folder: str | Path, experiment: str | Path, group_id: int
) -> tuple[Path, Path]:
    """
    Where the mean image of a group is (`Group.save_mean_images`), and its
    Cellpose candidates (`Group.save_cellpose_candidates`): in `processed/`
    of `experiment`, the folder of its first experiment inside
    `main_folder`. For the app, which finds them without the database.
    """
    processed = Path(main_folder) / experiment / "processed"

    return (
        processed / "mcor" / f"mean_group_{int(group_id)}.tif",
        processed / "segmentation" / f"cellpose_group_{int(group_id)}.h5",
    )


# --------------------------------------------------------------------------- #
# Curation
# --------------------------------------------------------------------------- #


def flood(
    found: Candidates,
    basin: int,
    seed: None | int = None,
    *,
    box: None | tuple[slice, slice] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The pixels of `basin` in the order a flood from `seed` takes them: always
    the most probable pixel touching (by a side) those taken, the lower index
    first among equals. `seed` is an index into the flattened image, the
    brightest pixel of the basin by default.

    Returns the pixels (indices into the flattened image) and, for each, the
    threshold it needs: the lowest probability on the way to it, which never
    rises. The first `n` pixels are the cell at the threshold where it has
    `n` pixels (`cut`), and the pixels in between fill the steps where a lower
    threshold adds several at once. `box` is the box of the basin, if known
    (`scipy.ndimage.find_objects`).
    """
    if box is None:
        box = ndi.find_objects(found.basins, max_label=basin)[basin - 1]

    width = found.basins.shape[1]
    seed = int(found.peak[basin - 1]) if seed is None else int(seed)
    y, x = divmod(seed, width)

    if box is None or found.basins[y, x] != basin:
        raise ValueError(f"Pixel {seed} is not in basin {basin}.")

    # Within the box of the basin, as Python lists: the loop below is the
    # whole cost, and a row-major index there sorts as the one in the image
    top, left = box[0].start, box[1].start
    rows, cols = found.basins[box].shape
    inside = (found.basins[box] == basin).ravel().tolist()
    prob = found.prob[box].ravel().tolist()
    start = (y - top) * cols + (x - left)
    taken = bytearray(rows * cols)
    taken[start] = 1
    heap = [(-prob[start], start)]
    order, level = [], []
    lowest = float("inf")

    while heap:
        _, p = heapq.heappop(heap)
        order.append(p)
        lowest = min(lowest, prob[p])
        level.append(lowest)
        column = p % cols

        for q in (
            p - 1 if column > 0 else -1,
            p + 1 if column < cols - 1 else -1,
            p - cols,
            p + cols if p + cols < rows * cols else -1,
        ):
            if q >= 0 and inside[q] and not taken[q]:
                taken[q] = 1
                heapq.heappush(heap, (-prob[q], q))

    local = np.asarray(order, np.int64)
    pixels = (local // cols + top) * width + local % cols + left

    return pixels, np.asarray(level, np.float32)


def _size_at(level: np.ndarray, threshold: float) -> int:
    """How many pixels of a flood are above `threshold`: its first ones."""
    return int(np.count_nonzero(level > threshold))


@dataclass
class Edits:
    """
    What a curator decided, on top of the candidates: small, so a copy can
    be kept for every step to undo, and saved as JSON (`as_json`,
    `from_json`).
    - `threshold`: the probability threshold of the cells that follow it
      (the Cellpose default, 0);
    - `flow_check`: keep the basins Cellpose keeps at its defaults (on), or
      every basin with a cell at the threshold (off);
    - `sizes`: cells with a size of their own (basin: pixels);
    - `seeds`: cells added by a click (basin: the pixel clicked, an index into
      the flattened image), grown from there;
    - `rejected`: cells taken out;
    - `min_um2`, `max_um2`: the area range; cells outside it are excluded.
    """

    threshold: float = THRESHOLD
    flow_check: bool = True
    sizes: dict[int, int] = field(default_factory=dict)
    seeds: dict[int, int] = field(default_factory=dict)
    rejected: set[int] = field(default_factory=set)
    min_um2: float = 0.0
    max_um2: None | float = None

    def as_json(self) -> dict[str, Any]:
        """The edits as JSON values (basins as text keys)."""
        return {
            "threshold": self.threshold,
            "flow_check": self.flow_check,
            "sizes": {str(b): int(n) for b, n in self.sizes.items()},
            "seeds": {str(b): int(p) for b, p in self.seeds.items()},
            "rejected": sorted(int(b) for b in self.rejected),
            "min_um2": self.min_um2,
            "max_um2": self.max_um2,
        }

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> Edits:
        """The edits `as_json` gave."""
        return cls(
            threshold=float(values["threshold"]),
            flow_check=bool(values["flow_check"]),
            sizes={int(b): int(n) for b, n in values["sizes"].items()},
            seeds={int(b): int(p) for b, p in values["seeds"].items()},
            rejected={int(b) for b in values["rejected"]},
            min_um2=float(values["min_um2"]),
            max_um2=None if values["max_um2"] is None else float(values["max_um2"]),
        )


@dataclass(frozen=True)
class Cell:
    """
    One cell of a `Curation`: its `basin`, its `pixels` (indices into the
    flattened image), its `kind` (`"cellpose"`, `"resized"` or `"added"`), the
    `threshold` it stands at (for a resized or added cell, the probability of
    its weakest pixel), and whether it is `rejected` or `excluded` (outside
    the area range). Only cells neither rejected nor excluded are kept.
    """

    basin: int
    pixels: np.ndarray
    kind: str
    threshold: float
    rejected: bool
    excluded: bool

    @property
    def kept(self) -> bool:
        return not (self.rejected or self.excluded)


class Curation:
    """
    The cells of one candidate set as a curator edits them. It starts from
    the cells Cellpose keeps at its defaults; every change goes through a
    method here (or a whole `edits` put back, to undo), and `cells`, `owner`,
    `labels` and `table` follow.

    **USAGE**
    ```python
    curation = Curation(read_candidates(path))
    before = curation.edits  # a copy
    curation.set_threshold(-1.0)
    curation.edits = before  # undo
    ```

    `cells` holds every cell, rejected and excluded ones too (flagged), and
    `owner` is the basin of every pixel in a cell (0 elsewhere), to find the
    cell under a point.
    """

    def __init__(self, found: Candidates, edits: None | Edits = None):
        self.found = found
        self.height, self.width = found.basins.shape
        um_per_px = found.settings.get("um_per_px") or [1.0, 1.0]
        self.um2_per_px = float(um_per_px[0]) * float(um_per_px[1])
        self._boxes = ndi.find_objects(found.basins)
        self._floods: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.cells: dict[int, Cell] = {}
        self.owner = np.zeros(found.basins.shape, np.uint32)
        self._edits = Edits() if edits is None else copy.deepcopy(edits)
        self._refresh()

    # ----------------------------------------------------------------- #
    # Reading
    # ----------------------------------------------------------------- #

    @property
    def edits(self) -> Edits:
        """A copy of the edits so far; put it back to undo what came after."""
        return copy.deepcopy(self._edits)

    @edits.setter
    def edits(self, edits: Edits) -> None:
        self._edits = copy.deepcopy(edits)
        self._refresh()

    def cell_at(self, y: int, x: int) -> None | int:
        """
        The basin of the cell at pixel `(y, x)`, rejected and excluded cells
        included, or `None`.
        """
        basin = int(self.owner[y, x])

        return basin or None

    def seed(self, basin: int) -> int:
        """Where the cell of `basin` grows from: the click, or the peak."""
        return int(self._edits.seeds.get(basin, self.found.peak[basin - 1]))

    def flood(self, basin: int) -> tuple[np.ndarray, np.ndarray]:
        """`flood` of `basin` from its seed (kept for later)."""
        key = (basin, self.seed(basin))

        if key not in self._floods:
            self._floods[key] = flood(self.found, *key, box=self._boxes[basin - 1])

        return self._floods[key]

    def kept(self) -> list[Cell]:
        """The cells kept (neither rejected nor excluded), by basin."""
        return [cell for _, cell in sorted(self.cells.items()) if cell.kept]

    def labels(self) -> np.ndarray:
        """The cells kept as a label image, numbered from 1 in basin order."""
        labels = np.zeros(self.found.basins.shape, np.uint32)

        for number, cell in enumerate(self.kept(), start=1):
            labels.flat[cell.pixels] = number

        return labels

    def table(self) -> pd.DataFrame:
        """
        One row per cell kept, by its number in `labels`: its basin, kind,
        size (px and um2), threshold and seed (row and column).
        """
        columns = ["label", "basin", "kind", "pixels", "area_um2", "threshold"]
        rows = []

        for number, cell in enumerate(self.kept(), start=1):
            size = len(cell.pixels)
            area = size * self.um2_per_px
            y, x = divmod(self.seed(cell.basin), self.width)
            rows.append(
                [number, cell.basin, cell.kind, size, area, cell.threshold, y, x]
            )

        return pd.DataFrame(rows, columns=[*columns, "seed_y", "seed_x"])

    # ----------------------------------------------------------------- #
    # Changing
    # ----------------------------------------------------------------- #

    def set_threshold(self, threshold: float) -> None:
        """The threshold of every cell without a size of its own."""
        self._edits.threshold = float(threshold)
        self._refresh()

    def set_flow_check(self, on: bool) -> None:
        """Keep only the basins Cellpose keeps at its defaults (on), or all."""
        self._edits.flow_check = bool(on)
        self._refresh()

    def set_area_range(
        self, min_um2: float = 0.0, max_um2: None | float = None
    ) -> None:
        """The area range: cells outside it are excluded (`None`: no max)."""
        self._edits.min_um2 = float(min_um2)
        self._edits.max_um2 = None if max_um2 is None else float(max_um2)
        self._refresh()

    def resize(self, basin: int, pixels: int) -> None:
        """Give the cell of `basin` a size of its own, in pixels."""
        total = len(self.flood(basin)[0])
        self._edits.sizes[basin] = max(1, min(total, int(pixels)))
        self._refresh([basin])

    def follow_threshold(self, basin: int) -> None:
        """Give the cell of `basin` the threshold of all again."""
        self._edits.sizes.pop(basin, None)
        self._refresh([basin])

    def reject(self, basin: int) -> None:
        """Take the cell of `basin` out; it stays, flagged as rejected."""
        self._edits.rejected.add(basin)
        self._refresh([basin])

    def restore(self, basin: int) -> None:
        """Put the cell of `basin` back, if it was rejected."""
        self._edits.rejected.discard(basin)
        self._refresh([basin])

    def add(self, y: int, x: int) -> None | int:
        """
        A click at pixel `(y, x)`: the cell of the basin there grows until it
        takes the pixel, or, when the basin has no cell, its candidate is
        added, grown from the pixel. Either way the cell is kept, even if it
        was rejected. Returns the basin, or `None` when no basin covers the
        pixel (or the cell there cannot reach it).

        An added cell starts at the threshold of all when the pixel is above
        it. Otherwise it starts halfway between `FLOOR` and the peak of the
        basin, since a threshold just under the pixel would leave a speck
        when the click is on the peak; never above the pixel itself.
        """
        pixel = y * self.width + x
        basin = int(self.found.basins[y, x])

        if not basin:
            return None

        if basin in self.cells:
            where = np.flatnonzero(self.flood(basin)[0] == pixel)

            if not where.size:
                return None

            if where[0] >= len(self.cells[basin].pixels):
                self._edits.sizes[basin] = int(where[0]) + 1
        else:
            prob = float(self.found.prob[y, x])
            peak = float(self.found.prob.flat[self.found.peak[basin - 1]])
            threshold = self._edits.threshold

            if not prob > threshold:
                threshold = max(FLOOR, min(prob - 0.01, (FLOOR + peak) / 2))

            self._edits.seeds[basin] = pixel
            level = self.flood(basin)[1]
            self._edits.sizes[basin] = max(1, _size_at(level, threshold))

        self._edits.rejected.discard(basin)
        self._refresh([basin])

        return basin

    # ----------------------------------------------------------------- #
    # Inside
    # ----------------------------------------------------------------- #

    def _candidate(self, basin: int) -> bool:
        """Whether `basin` may have a cell (see `Edits.flow_check`)."""
        if basin in self._edits.seeds or not self._edits.flow_check:
            return True

        return bool(
            self.found.area[basin - 1] >= MIN_SIZE
            and self.found.flow_error[basin - 1] <= FLOW_THRESHOLD
        )

    def _cell(self, basin: int) -> None | Cell:
        """The cell of `basin` under the edits, or `None` when it has none."""
        edits = self._edits

        if not self._candidate(basin):
            return None

        order, level = self.flood(basin)
        own = edits.sizes.get(basin)
        size = _size_at(level, edits.threshold) if own is None else own

        # Cellpose drops cells under MIN_SIZE; a size chosen by hand stays
        if size == 0 or (own is None and basin not in edits.seeds and size < MIN_SIZE):
            return None

        area = size * self.um2_per_px
        kind = (
            "added"
            if basin in edits.seeds
            else "cellpose" if own is None else "resized"
        )

        return Cell(
            basin=basin,
            pixels=order[:size],
            kind=kind,
            threshold=edits.threshold if own is None else float(level[size - 1]),
            rejected=basin in edits.rejected,
            excluded=area < edits.min_um2
            or (edits.max_um2 is not None and area > edits.max_um2),
        )

    def _refresh(self, basins: None | Iterable[int] = None) -> None:
        """Work out the cells again: all of them, or those of `basins`."""
        if basins is None:
            self.cells = {}
            self.owner[:] = 0
            basins = range(1, len(self.found.peak) + 1)

        for basin in basins:
            old = self.cells.pop(basin, None)

            if old is not None:
                self.owner.flat[old.pixels] = 0

            cell = self._cell(basin)

            if cell is not None:
                self.cells[basin] = cell
                self.owner.flat[cell.pixels] = basin
