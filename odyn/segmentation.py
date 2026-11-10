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
network.

**USAGE**
```python
from odyn.segmentation import default_cells, find_candidates, read_candidates

found = find_candidates(image, weights)  # the network, then the basins
found.write(path)
found = read_candidates(path)
labels = default_cells(found)  # the cells Cellpose keeps at its defaults
```

`find_candidates` needs Cellpose 4 (and its `torch`); the rest needs only
numpy, scipy and h5py.
"""

from __future__ import annotations

import json

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

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


@dataclass
class Candidates:
    """
    One Cellpose run, on the grid of the image it ran on:
    - `prob`: the cell probability of every pixel (logits);
    - `flows`: the flows (2 x height x width, y then x);
    - `basins`: the basin of every pixel, `0` for none and `1` to `n`;
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
    the cell probability (logits) and the settings used.
    """
    # Cellpose and torch are heavy, and only this needs them
    import cellpose
    import torch

    from cellpose import models

    model = models.CellposeModel(gpu=True, pretrained_model=str(weights))
    normalize: Any = True if tile_px is None else {"tile_norm_blocksize": int(tile_px)}
    _, (_, flows, prob), _ = model.eval(
        np.asarray(image, np.float32), normalize=normalize, compute_masks=False
    )
    settings = {
        "model": Path(weights).name,
        "tile_px": tile_px,
        "cellpose": cellpose.version,
        "torch": torch.__version__,
        "device": str(model.device),
    }

    return np.asarray(flows, np.float32), np.asarray(prob, np.float32), settings


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
