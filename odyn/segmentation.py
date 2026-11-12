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

Where Cellpose has no basin for a cell, or a poor one, the cell can be grown
from the mean image instead: from a click, the brightest pixel next to those
taken, again and again, on the image smoothed a little and near the click
(`Curation.grow`). Such a cell is kept as its pixels, which only editing it
changes, and the cells of Cellpose go around it.

**USAGE**
```python
from odyn.segmentation import Curation, find_candidates, read_candidates

found = find_candidates(image, weights)  # the network, then the basins
found.write(path)

curation = Curation(read_candidates(path))  # the Cellpose defaults
curation.set_threshold(-1.0)                # every cell bigger
basin = curation.add(120, 85)               # the candidate under a point
curation.resize(basin, 150)                 # 150 px
curation = Curation(found, image=mean)      # with the mean image, to grow
cell = curation.grow(40, 210)               # a cell grown from the image
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

# Cells grown from the mean image: on the image smoothed this much (um, the
# sigma of a Gaussian), within this distance of their seed (um), starting at
# the edge where the image falls this far from the peak of the cell to the
# background around it (the lowest tenth within reach); a click moves to the
# brightest pixel this near it (px)
SMOOTH_UM = 1.5
REACH_UM = 15.0
EDGE = 0.5
CLIMB_PX = 3


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

    return _flood_box(found.prob[box], found.basins[box] == basin, box, seed, width)


def _flood_box(
    values: np.ndarray,
    inside: np.ndarray,
    box: tuple[slice, slice],
    seed: None | int,
    width: int,
    *,
    taken: None | np.ndarray = None,
    lowest: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    The flood of `flood` within `box`, over the pixels `inside` it, by
    `values` (both of the box only), from `seed` (an index into the flattened
    image, of width `width`), or on from the pixels already `taken` (indices
    too, all in the box), `lowest` being the lowest value on the way to them.
    Returns the pixels it takes (indices into the flattened image), in order,
    and the lowest value on the way to each.
    """
    # As Python lists: the loop below is the whole cost, and a row-major
    # index in the box sorts as the one in the image
    top, left = box[0].start, box[1].start
    rows, cols = values.shape
    inside = inside.ravel().tolist()
    value = values.ravel().tolist()
    done = bytearray(rows * cols)
    heap: list[tuple[float, int]] = []

    def spread(p: int) -> None:
        """Put the pixels touching `p` that may be taken on the heap."""
        column = p % cols

        for q in (
            p - 1 if column > 0 else -1,
            p + 1 if column < cols - 1 else -1,
            p - cols,
            p + cols if p + cols < rows * cols else -1,
        ):
            if q >= 0 and inside[q] and not done[q]:
                done[q] = 1
                heapq.heappush(heap, (-value[q], q))

    if taken is None:
        y, x = divmod(int(seed), width)
        start = (y - top) * cols + (x - left)
        done[start] = 1
        heap.append((-value[start], start))
    else:
        ys, xs = np.divmod(np.asarray(taken, np.int64), width)
        starts = ((ys - top) * cols + (xs - left)).tolist()

        for p in starts:
            done[p] = 1

        for p in starts:
            spread(p)

    order, level = [], []

    while heap:
        _, p = heapq.heappop(heap)
        order.append(p)
        lowest = min(lowest, value[p])
        level.append(lowest)
        spread(p)

    local = np.asarray(order, np.int64)
    pixels = (local // cols + top) * width + local % cols + left

    return pixels, np.asarray(level, np.float32)


def _size_at(level: np.ndarray, threshold: float) -> int:
    """How many pixels of a flood are above `threshold`: its first ones."""
    return int(np.count_nonzero(level > threshold))


def _nothing() -> tuple[np.ndarray, np.ndarray]:
    """A flood that takes no pixel."""
    return np.zeros(0, np.int64), np.zeros(0, np.float32)


def _runs(pixels: np.ndarray) -> list[list[int]]:
    """Sorted indices as runs of consecutive ones, `[first, count]` each."""
    if not len(pixels):
        return []

    breaks = np.flatnonzero(np.diff(pixels) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(pixels)]])

    return [[int(pixels[s]), int(e - s)] for s, e in zip(starts, ends)]


def _from_runs(runs: list[list[int]]) -> np.ndarray:
    """The indices `_runs` gave."""
    if not runs:
        return np.zeros(0, np.int64)

    return np.concatenate([np.arange(first, first + count) for first, count in runs])


@dataclass(frozen=True, eq=False)
class Drawn:
    """
    A cell drawn by a tool, kept as its pixels: they change only when the
    cell is edited, whatever changes around it. `kind` says which tool
    (`"image"`: grown from the mean image, `Curation.grow`), `seed` is where
    it grows from and `pixels` are its pixels (indices into the flattened
    image, sorted). It never changes in place, so copies of the edits share
    it (`Edits`).
    """

    kind: str
    seed: int
    pixels: np.ndarray

    def __post_init__(self) -> None:
        pixels = np.unique(np.asarray(self.pixels, np.int64))
        pixels.flags.writeable = False
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "pixels", pixels)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Drawn)
            and (self.kind, self.seed) == (other.kind, other.seed)
            and np.array_equal(self.pixels, other.pixels)
        )

    def __deepcopy__(self, memo: dict) -> Drawn:
        return self

    def as_json(self) -> dict[str, Any]:
        """The cell as JSON values, its pixels as runs (`[first, count]`)."""
        return {"kind": self.kind, "seed": self.seed, "runs": _runs(self.pixels)}

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> Drawn:
        """The cell `as_json` gave."""
        return cls(values["kind"], int(values["seed"]), _from_runs(values["runs"]))


@dataclass
class Edits:
    """
    What a curator decided, on top of the candidates: small, so a copy can
    be kept for every step to undo (the drawn cells shared between copies),
    and saved as JSON (`as_json`, `from_json`).
    - `threshold`: the probability threshold of the Cellpose cells that
      follow it (the Cellpose default, 0);
    - `flow_check`: keep the basins Cellpose keeps at its defaults (on), or
      every basin with a cell at the threshold (off);
    - `sizes`: Cellpose cells with a size of their own (basin: pixels);
    - `seeds`: Cellpose cells added by a click (basin: the pixel clicked, an
      index into the flattened image), grown from there;
    - `drawn`: cells drawn by a tool (`Drawn`), by number: the basin of the
      Cellpose cell one replaces, or a number above the basins;
    - `rejected`: Cellpose cells taken out (a drawn cell taken out is gone);
    - `min_um2`, `max_um2`: the area range; cells outside it are excluded.
    """

    threshold: float = THRESHOLD
    flow_check: bool = True
    sizes: dict[int, int] = field(default_factory=dict)
    seeds: dict[int, int] = field(default_factory=dict)
    drawn: dict[int, Drawn] = field(default_factory=dict)
    rejected: set[int] = field(default_factory=set)
    min_um2: float = 0.0
    max_um2: None | float = None

    def as_json(self) -> dict[str, Any]:
        """The edits as JSON values (cells as text keys)."""
        return {
            "threshold": self.threshold,
            "flow_check": self.flow_check,
            "sizes": {str(b): int(n) for b, n in self.sizes.items()},
            "seeds": {str(b): int(p) for b, p in self.seeds.items()},
            "drawn": {str(c): d.as_json() for c, d in self.drawn.items()},
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
            drawn={int(c): Drawn.from_json(d) for c, d in values["drawn"].items()},
            rejected={int(b) for b in values["rejected"]},
            min_um2=float(values["min_um2"]),
            max_um2=None if values["max_um2"] is None else float(values["max_um2"]),
        )


@dataclass(frozen=True)
class Cell:
    """
    One cell of a `Curation`: its number (`basin`: the basin of a Cellpose
    cell or of the one a drawn cell replaces, else a number above the
    basins), its `pixels` (indices into the flattened image), its `kind`
    (`"cellpose"`, `"resized"`, `"added"`, or `"image"` when drawn from the
    image), the `threshold` it stands at (for a resized or added cell, the
    probability of its weakest pixel; none for a drawn one), and whether it
    is `rejected` or `excluded` (outside the area range). Only cells neither
    rejected nor excluded are kept.
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
    `labels` and `table` follow, from the edits alone.

    **USAGE**
    ```python
    curation = Curation(read_candidates(path))
    before = curation.edits  # a copy
    curation.set_threshold(-1.0)
    curation.edits = before  # undo
    ```

    `cells` holds every cell, rejected and excluded ones too (flagged), and
    `owner` is the number of the cell at every pixel (0 where none is), to
    find the cell under a point.

    Cells are of two sorts, which never overlap:
    - Cellpose cells, one per basin and numbered as it: the pixels of the
      basin from its seed, as many as the threshold gives or a size of
      their own;
    - drawn cells (`Edits.drawn`), kept as their pixels, which change only
      when the cell is edited. A cell grown from the mean image (`grow`,
      which needs the `image`) takes the pixels free as it grows: in no kept
      cell and in no other drawn cell, whatever the basins.
    Drawn cells come first: Cellpose cells go around them, rejected ones
    too, and have no pixels when one covers their seed.
    """

    def __init__(
        self,
        found: Candidates,
        edits: None | Edits = None,
        *,
        image: None | np.ndarray = None,
    ):
        self.found = found
        self.height, self.width = found.basins.shape
        um_per_px = found.settings.get("um_per_px") or [1.0, 1.0]
        self.um2_per_px = float(um_per_px[0]) * float(um_per_px[1])
        self.basin_count = len(found.peak)
        self.image = None if image is None else np.asarray(image, np.float32)
        px_per_um = 2 / (float(um_per_px[0]) + float(um_per_px[1]))
        self._sigma_px = SMOOTH_UM * px_per_um
        self._reach_px = max(1, round(REACH_UM * px_per_um))
        self._smooth: None | np.ndarray = None
        self._boxes = ndi.find_objects(found.basins)
        # The latest flood of each Cellpose cell (by basin and seed) and of
        # each drawn cell, with what it depends on besides
        self._floods: dict[tuple[int, int], tuple[bytes, tuple]] = {}
        self._growths: dict[int, tuple[tuple[bytes, bytes], tuple]] = {}
        self.cells: dict[int, Cell] = {}
        self.owner = np.zeros(found.basins.shape, np.uint32)
        # The drawn cell at every pixel, and the pixels put there, by cell
        self._drawn_at = np.zeros(found.basins.shape, np.uint32)
        self._stamped: dict[int, np.ndarray] = {}
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
        The number of the cell at pixel `(y, x)`, rejected and excluded cells
        included, or `None`.
        """
        number = int(self.owner[y, x])

        return number or None

    @property
    def smooth(self) -> np.ndarray:
        """The mean image smoothed (`SMOOTH_UM`): what grown cells follow."""
        if self._smooth is None:
            if self.image is None:
                raise ValueError("Growing a cell from the image needs the image.")

            self._smooth = ndi.gaussian_filter(self.image, self._sigma_px)

        return self._smooth

    def from_image(self, cell: int) -> bool:
        """Whether `cell` is drawn, grown from the image (`grow`)."""
        drawn = self._edits.drawn.get(cell)

        return drawn is not None and drawn.kind == "image"

    def may_have_cell(self, number: int) -> bool:
        """
        Whether cell `number` may be there under the edits, with pixels at
        the threshold or not: a drawn one, an added one, or the Cellpose
        cell of a basin the flow check lets through (of any basin, with the
        flow check off; see `Edits.flow_check`).
        """
        edits = self._edits

        if number in edits.drawn or number in edits.seeds:
            return True

        if not 1 <= number <= self.basin_count:
            return False

        if not edits.flow_check:
            return True

        return bool(
            self.found.area[number - 1] >= MIN_SIZE
            and self.found.flow_error[number - 1] <= FLOW_THRESHOLD
        )

    def seed(self, cell: int) -> int:
        """
        Where `cell` grows from: for a Cellpose cell, the click that added it
        or the peak of its basin.
        """
        drawn = self._edits.drawn.get(cell)

        if drawn is not None:
            return drawn.seed

        if not 1 <= cell <= self.basin_count:
            raise ValueError(f"No basin or drawn cell is numbered {cell}.")

        return int(self._edits.seeds.get(cell, self.found.peak[cell - 1]))

    def flood(self, cell: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The pixels `cell` takes as it grows, in order, and for each the
        lowest value on the way to it: its first n pixels are the cell at n
        pixels. For a Cellpose cell, `flood` of its basin from its seed,
        around the drawn cells (nothing when one covers the seed); for a
        drawn cell, its own pixels, then those free now (`_growth`).
        """
        if cell in self._edits.drawn:
            return self._growth(cell)

        seed = self.seed(cell)
        box = self._boxes[cell - 1]
        drawn = self._drawn_at[box]
        covered = b""

        if drawn.any():
            inside = self.found.basins[box] == cell
            hidden = inside & (drawn > 0)

            if hidden.any():
                if self._drawn_at.flat[seed]:
                    return _nothing()

                covered = hidden.tobytes()

        kept = self._floods.get((cell, seed))

        if kept is None or kept[0] != covered:
            if covered:
                values = self.found.prob[box]
                found = _flood_box(values, inside & ~hidden, box, seed, self.width)
            else:
                found = flood(self.found, cell, seed, box=box)

            kept = (covered, found)
            self._floods[(cell, seed)] = kept

        return kept[1]

    def kept(self) -> list[Cell]:
        """The cells kept (neither rejected nor excluded), by number."""
        return [cell for _, cell in sorted(self.cells.items()) if cell.kept]

    def labels(self) -> np.ndarray:
        """The cells kept as a label image, numbered from 1 as in `kept`."""
        labels = np.zeros(self.found.basins.shape, np.uint32)

        for number, cell in enumerate(self.kept(), start=1):
            labels.flat[cell.pixels] = number

        return labels

    def table(self) -> pd.DataFrame:
        """
        One row per cell kept, by its number in `labels`: its basin (0 for a
        cell drawn where no basin is), kind, size (px and um2), threshold
        and seed (row and column).
        """
        columns = ["label", "basin", "kind", "pixels", "area_um2", "threshold"]
        rows = []

        for number, cell in enumerate(self.kept(), start=1):
            size = len(cell.pixels)
            area = size * self.um2_per_px
            y, x = divmod(self.seed(cell.basin), self.width)
            basin = cell.basin if cell.basin <= self.basin_count else 0
            rows.append([number, basin, cell.kind, size, area, cell.threshold, y, x])

        return pd.DataFrame(rows, columns=[*columns, "seed_y", "seed_x"])

    # ----------------------------------------------------------------- #
    # Changing
    # ----------------------------------------------------------------- #

    def set_threshold(self, threshold: float) -> None:
        """The threshold of every Cellpose cell without a size of its own."""
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

    def resize(self, cell: int, pixels: int) -> None:
        """
        Give `cell` a size of its own, in pixels: the first ones of its
        `flood`. So a drawn cell shrinks within its own pixels, and grows
        over those free now.
        """
        self._give_size(cell, self.flood(cell)[0], pixels)
        self._refresh([cell])

    def follow_threshold(self, cell: int) -> None:
        """Give Cellpose cell `cell` the threshold of all again."""
        if cell not in self._edits.drawn:  # which has no threshold
            self._edits.sizes.pop(cell, None)
            self._refresh([cell])

    def reject(self, cell: int) -> None:
        """
        Take `cell` out: a Cellpose cell stays, flagged as rejected; a drawn
        cell is gone (undo brings it back), and the Cellpose cell it
        replaced, if any, stays as rejected.
        """
        if self._edits.drawn.pop(cell, None) is None or cell <= self.basin_count:
            self._edits.rejected.add(cell)

        self._refresh([cell])

    def restore(self, cell: int) -> None:
        """Put Cellpose cell `cell` back, if it was rejected."""
        self._edits.rejected.discard(cell)
        self._refresh([cell])

    def add(self, y: int, x: int) -> None | int:
        """
        A click at pixel `(y, x)`: the cell of the basin there grows until it
        takes the pixel (its Cellpose cell, or the cell drawn in its place),
        or, when the basin has no cell, its candidate is added, grown from
        the pixel; where no basin is, a cell is drawn from the image
        (`grow`). Either way the cell is kept, even if it was rejected.
        Returns the cell, or `None` when the cell there cannot reach the
        pixel, or no basin covers it and there is no image.

        An added cell starts at the threshold of all when the pixel is above
        it. Otherwise it starts halfway between `FLOOR` and the peak of the
        basin, since a threshold just under the pixel would leave a speck
        when the click is on the peak; never above the pixel itself.
        """
        pixel = y * self.width + x
        basin = int(self.found.basins[y, x])

        if not basin:
            return self.grow(y, x)

        if basin in self.cells or basin in self._edits.drawn:
            order = self.flood(basin)[0]
            where = np.flatnonzero(order == pixel)

            if not where.size:
                return None

            cell = self.cells.get(basin)

            if cell is None or where[0] >= len(cell.pixels):
                self._give_size(basin, order, int(where[0]) + 1)
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

    def grow(self, y: int, x: int) -> None | int:
        """
        A cell drawn from the mean image at pixel `(y, x)`, numbered above
        the basins and the cells drawn so far (`Edits.drawn`). From the
        brightest free pixel near the click (`CLIMB_PX`), it takes the
        brightest free pixel next to those taken, again and again, on the
        smoothed image (`smooth`), within `REACH_UM`, and keeps those above
        its edge (`EDGE`). Free pixels are in no kept cell and in no other
        drawn cell, whatever the basins (`_free`). Returns the cell, or
        `None` when the pixel is not free or there is no image.
        """
        return self._draw(max([self.basin_count, *self._edits.drawn]) + 1, y, x)

    def use_basin(self, basin: int, on: bool) -> bool:
        """
        The cell of `basin` from the basin (on), its Cellpose cell, or from
        the image (off): a cell drawn in its place, grown from where it grows
        now (`grow`). Off and on again, the Cellpose cell is as it was. A
        cell drawn where no basin is has no basin to use. Returns whether
        anything changed.
        """
        if not 1 <= basin <= self.basin_count or on != (basin in self._edits.drawn):
            return False

        if not on:
            made = self._draw(basin, *divmod(self.seed(basin), self.width))

            return made is not None

        del self._edits.drawn[basin]
        self._refresh([basin])

        return True

    # ----------------------------------------------------------------- #
    # Inside
    # ----------------------------------------------------------------- #

    def _disk(
        self, y: int, x: int, radius: int
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        """The box around `(y, x)`, and in it the pixels within `radius`."""
        box = (
            slice(max(y - radius, 0), min(y + radius + 1, self.height)),
            slice(max(x - radius, 0), min(x + radius + 1, self.width)),
        )
        rows, columns = np.ogrid[box]

        return box, (rows - y) ** 2 + (columns - x) ** 2 <= radius * radius

    def _free(
        self, cell: int, y: int, x: int, radius: int
    ) -> tuple[tuple[slice, slice], np.ndarray]:
        """
        The box around pixel `(y, x)` and, in it, the pixels within `radius`
        drawn cell `cell` may take: in no kept cell and in no other drawn
        cell. Its own pixels are free to it, and so are those of the
        Cellpose cell it replaces.
        """
        box, free = self._disk(y, x, radius)
        owner = self.owner[box]

        for number in np.unique(owner).tolist():
            if number and number != cell:
                other = self.cells.get(number)

                if number in self._edits.drawn or (other is not None and other.kept):
                    free &= owner != number

        return box, free

    def _draw(self, cell: int, y: int, x: int) -> None | int:
        """
        Draw `cell` from the image at pixel `(y, x)` (`grow`); `None` when
        the pixel is not free or there is no image.
        """
        if self.image is None:
            return None

        box, free = self._free(cell, y, x, CLIMB_PX)

        if not free[y - box[0].start, x - box[1].start]:
            return None

        values = np.where(free, self.smooth[box], -np.inf)
        dy, dx = np.unravel_index(int(np.argmax(values)), values.shape)
        y, x = box[0].start + int(dy), box[1].start + int(dx)
        seed = y * self.width + x
        box, free = self._free(cell, y, x, self._reach_px)
        order, level = _flood_box(self.smooth[box], free, box, seed, self.width)

        # Its edge: `EDGE` of the way from the background around it to its peak
        disk = self._disk(y, x, self._reach_px)[1]
        background = float(np.percentile(self.smooth[box][disk], 10))
        edge = background + EDGE * (float(level[0]) - background)
        size = max(1, _size_at(level, edge))
        self._edits.drawn[cell] = Drawn("image", seed, order[:size])
        self._edits.rejected.discard(cell)  # of the Cellpose cell it replaces
        self._refresh([cell])

        return cell

    def _growth(self, cell: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The `flood` of drawn cell `cell`, on the smoothed image: its own
        pixels, in the order a flood from its seed takes them, then the free
        pixels (`_free`) a flood on from them takes, within `REACH_UM` of the
        seed.
        """
        drawn = self._edits.drawn[cell]
        y, x = divmod(drawn.seed, self.width)
        box, free = self._free(cell, y, x, self._reach_px)
        key = (drawn.pixels.tobytes(), free.tobytes())
        kept = self._growths.get(cell)

        if kept is None or kept[0] != key:
            own = np.zeros(free.shape, bool)
            rows, columns = np.divmod(drawn.pixels, self.width)
            own[rows - box[0].start, columns - box[1].start] = True
            values = self.smooth[box]
            order, level = _flood_box(values, own, box, drawn.seed, self.width)
            lowest = float(level[-1]) if level.size else float("inf")
            more, lower = _flood_box(
                values, free & ~own, box, None, self.width, taken=order, lowest=lowest
            )
            found = (np.concatenate([order, more]), np.concatenate([level, lower]))
            kept = (key, found)
            self._growths[cell] = kept

        return kept[1]

    def _give_size(self, cell: int, order: np.ndarray, size: int) -> None:
        """Make `cell` the first `size` pixels of `order`, its `flood`."""
        size = max(1, min(len(order), int(size)))
        drawn = self._edits.drawn.get(cell)

        if drawn is None:
            self._edits.sizes[cell] = size
        elif len(order):
            self._edits.drawn[cell] = Drawn(drawn.kind, drawn.seed, order[:size])

    def _outside(self, size: int) -> bool:
        """Whether a cell of `size` pixels is outside the area range."""
        area = size * self.um2_per_px
        edits = self._edits

        return area < edits.min_um2 or (
            edits.max_um2 is not None and area > edits.max_um2
        )

    def _cell(self, number: int) -> None | Cell:
        """The cell `number` under the edits, or `None` when it has none."""
        edits = self._edits
        drawn = edits.drawn.get(number)

        if drawn is not None:
            return Cell(
                basin=number,
                pixels=drawn.pixels,
                kind=drawn.kind,
                threshold=float("nan"),
                rejected=False,
                excluded=self._outside(len(drawn.pixels)),
            )

        if not self.may_have_cell(number):
            return None

        order, level = self.flood(number)
        own = edits.sizes.get(number)
        size = _size_at(level, edits.threshold) if own is None else min(own, len(order))

        # Cellpose drops cells under MIN_SIZE; a size chosen by hand stays
        if size == 0 or (own is None and number not in edits.seeds and size < MIN_SIZE):
            return None

        kind = (
            "added"
            if number in edits.seeds
            else "cellpose" if own is None else "resized"
        )

        return Cell(
            basin=number,
            pixels=order[:size],
            kind=kind,
            threshold=edits.threshold if own is None else float(level[size - 1]),
            rejected=number in edits.rejected,
            excluded=self._outside(size),
        )

    def _refresh(self, numbers: None | Iterable[int] = None) -> None:
        """
        Work out the cells again from the edits: all of them, or the cells
        `numbers` and the Cellpose cells whose basins the drawn ones among
        them leave or take. Drawn cells first, as their pixels are set; then
        Cellpose cells, around them.
        """
        drawn = self._edits.drawn

        if numbers is None:
            self.cells = {}
            self.owner[:] = 0
            self._drawn_at[:] = 0
            self._stamped = {}
            todo = {*range(1, self.basin_count + 1), *drawn}
        else:
            todo = set(numbers)

        # The drawn cells among them where they are now, minding the basins
        # they leave or take
        changed = todo & (set(self._stamped) | set(drawn))
        touched = set()

        for number in changed:
            old = self._stamped.pop(number, None)

            if old is not None:
                self._drawn_at.flat[old] = 0
                touched.update(np.unique(self.found.basins.flat[old]).tolist())

        for number in changed & set(drawn):
            pixels = drawn[number].pixels
            self._drawn_at.flat[pixels] = number
            self._stamped[number] = pixels
            touched.update(np.unique(self.found.basins.flat[pixels]).tolist())

        todo |= touched - {0}

        for number in todo:  # all out first: then each takes what it may
            old = self.cells.pop(number, None)

            if old is not None:
                self.owner.flat[old.pixels] = 0

        for number in sorted(todo, key=lambda number: (number not in drawn, number)):
            cell = self._cell(number)

            if cell is not None:
                self.cells[number] = cell
                self.owner.flat[cell.pixels] = number
