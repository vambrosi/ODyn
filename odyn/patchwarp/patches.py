"""The patchwork: a grid of overlapping subfields, and the stitch back together.

The field is cut into `blocksize x blocksize` tiles which overlap by exactly
`2·overlap + 1` pixels on every interior seam:

    tile 0        : [0, c + v)
    tile j (0<j<B): [j·c − v − 1, (j+1)·c + v)
    tile B−1      : [(B−1)·c − v − 1, n)          c = ⌈n/B⌉,  v = overlap

Overlap is what makes the piecewise-affine field usable. Each tile gets its own
affine map, so the reconstruction is discontinuous across seams unless
neighbouring maps agree there; averaging over a band of `2v+1` pixels turns the
jump into a ramp of that width. The band is a plain (unweighted) mean, not a
feathered blend — a pixel covered by `k` tiles gets the mean of the `k` values,
`NaN`s (out-of-support samples) excluded.

`stitch` accumulates sums and counts on the canvas. The original merges seams
sequentially — rows first, then columns — and deletes the duplicated strip each
time; at a 4-tile corner that yields `mean(mean(a,b), mean(c,d)) = mean(a,b,c,d)`,
so the two are equal, and the accumulator form avoids the index bookkeeping.

`overlap` is derived from the *width* only (`⌈f · n_x / B⌉`) and then used on
both axes, as in the original; for non-square fields the vertical overlap is
therefore not `f` of the tile height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interp import warp_image
from .transforms import Transform


@dataclass(frozen=True)
class PatchGrid:
    """Tile ranges for a field of shape `(ny, nx)`."""

    ny: int
    nx: int
    blocksize: int
    overlap: int
    y_ranges: tuple[range, ...]
    x_ranges: tuple[range, ...]

    @staticmethod
    def _ranges(n: int, blocksize: int, overlap: int) -> tuple[range, ...]:
        if blocksize == 1:
            return (range(0, n),)
        c = int(np.ceil(n / blocksize))
        out = [range(0, min(c + overlap, n))]
        for j in range(1, blocksize - 1):
            out.append(
                range(max(j * c - overlap - 1, 0), min((j + 1) * c + overlap, n))
            )
        out.append(range(max((blocksize - 1) * c - overlap - 1, 0), n))
        return tuple(out)

    @classmethod
    def build(
        cls, shape: tuple[int, int], blocksize: int, overlap_frac: float
    ) -> "PatchGrid":
        ny, nx = shape
        overlap = 0 if blocksize == 1 else int(np.ceil(overlap_frac * (nx / blocksize)))
        return cls(
            ny=ny,
            nx=nx,
            blocksize=blocksize,
            overlap=overlap,
            y_ranges=cls._ranges(ny, blocksize, overlap),
            x_ranges=cls._ranges(nx, blocksize, overlap),
        )

    def slices(self, i: int, j: int) -> tuple[slice, slice]:
        ry, rx = self.y_ranges[i], self.x_ranges[j]
        return slice(ry.start, ry.stop), slice(rx.start, rx.stop)

    def tile(self, img: np.ndarray, i: int, j: int) -> np.ndarray:
        sy, sx = self.slices(i, j)
        return img[sy, sx]

    def indices(self):
        for i in range(self.blocksize):
            for j in range(self.blocksize):
                yield i, j

    def stitch(self, tiles: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
        """Average the tiles back onto one canvas; uncovered pixels are `NaN`."""
        total = np.zeros((self.ny, self.nx))
        count = np.zeros((self.ny, self.nx))
        for (i, j), tile in tiles.items():
            sy, sx = self.slices(i, j)
            valid = np.isfinite(tile)
            total[sy, sx] += np.where(valid, tile, 0.0)
            count[sy, sx] += valid
        with np.errstate(invalid="ignore"):
            return np.where(count > 0, total / count, np.nan)


def apply_patch_warps(
    frame: np.ndarray,
    grid: PatchGrid,
    warps: np.ndarray,
    model: Transform = Transform.AFFINE,
) -> np.ndarray:
    """Warp each tile of `frame` by its own map and stitch.

    `warps` has shape `(B, B, 3, 3)`. Each map is expressed in the tile's own
    local coordinates, which is how it was estimated.
    """
    tiles = {}
    for i, j in grid.indices():
        tile = np.asarray(grid.tile(frame, i, j), dtype=float)
        out = warp_image(tile, warps[i, j], model)
        out[out == 0] = np.nan  # 0 marks samples pulled from outside the tile
        tiles[(i, j)] = out
    return grid.stitch(tiles)
