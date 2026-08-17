"""Registering two imaging sessions to each other (`patchwarp_across_sessions.m`).

Same machinery, different problem: instead of a time series of blocks there are
just two summary images (or a few "image types" per session — mean, max, a
correlation image — which serve as independent trials of the same alignment).

Two passes:

1. **Global.** One warp for the whole field, estimated with a 3-level pyramid
   from identity. Estimated independently for each image type; the warp with
   the best `ρ` wins and is applied to all types. Taking the best rather than
   the average is deliberate — the types are the same geometry seen through
   different statistics, so the most informative one should not be diluted.
2. **Patchwise.** The globally-warped session 2 is cut into the usual grid and
   each tile gets its own warp against the corresponding tile of session 1,
   1 pyramid level, again best-of-types per tile, then stitched.

Composition, not iteration: the global warp handles the large rigid-ish offset
that would put a tile outside its counterpart's basin of attraction, and the
patch warps only need to explain the residual local distortion.

Empty regions (created by padding to a common size, and by the warps pulling
from outside the field) are filled with uniform noise between the 1st and 30th
percentile of the image, so that the correlation is not inflated by large
constant patches — this is cosmetic for the metric but matters for the ECC
support masks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ecc import ecc_align
from .interp import warp_image
from .normalize import imnormalize
from .patches import PatchGrid
from .transforms import Transform, identity


@dataclass
class AcrossSessionResult:
    global_warp: np.ndarray  # 3x3
    patch_warps: np.ndarray  # (B, B, 3, 3)
    rho_global: np.ndarray  # (n_types,)
    rho_patch: np.ndarray  # (B, B, n_types)
    image1: np.ndarray
    image2_global: np.ndarray
    image2_patched: np.ndarray
    grid: PatchGrid


def _fill_background(img: np.ndarray, mask: np.ndarray, rng) -> np.ndarray:
    """Replace `mask` with uniform noise in `[p1, p30]` of the valid values."""
    valid = img[~mask]
    if valid.size == 0:
        return img
    lo, hi = np.percentile(valid, [1, 30])
    out = img.copy()
    out[mask] = rng.uniform(lo, hi, size=int(mask.sum()))
    return out


def _pad_to(img: np.ndarray, shape: tuple[int, int], rng) -> np.ndarray:
    """Pad the bottom/right to `shape` with background noise."""
    h, w = img.shape[:2]
    out = np.empty(shape + img.shape[2:], dtype=float)
    mask = np.ones(shape, dtype=bool)
    mask[:h, :w] = False
    for c in range(img.shape[2]):
        canvas = np.zeros(shape)
        canvas[:h, :w] = img[:, :, c]
        out[:, :, c] = _fill_background(canvas, mask, rng)
    return out


def register_sessions(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    transform_global: Transform = Transform.AFFINE,
    transform_patch: Transform = Transform.AFFINE,
    blocksize: int = 4,
    overlap_frac: float = 0.15,
    norm_radius: float = 0,
    pyramid_levels_global: int = 3,
    pyramid_levels_patch: int = 1,
    n_iter: int = 100,
    learning_rate: float = 0.1,
    rng: np.random.Generator | int | None = None,
) -> AcrossSessionResult:
    """Register `image2` onto `image1`; both are `(ny, nx, n_image_types)`."""
    rng = np.random.default_rng(rng)
    image1 = np.atleast_3d(np.asarray(image1, dtype=float))
    image2 = np.atleast_3d(np.asarray(image2, dtype=float))
    n_types = image1.shape[2]

    if norm_radius:
        image1 = np.dstack(
            [imnormalize(image1[:, :, i], norm_radius) for i in range(n_types)]
        )
        image2 = np.dstack(
            [imnormalize(image2[:, :, i], norm_radius) for i in range(n_types)]
        )

    shape = (
        max(image1.shape[0], image2.shape[0]),
        max(image1.shape[1], image2.shape[1]),
    )
    image1 = _pad_to(image1, shape, rng)
    image2 = _pad_to(image2, shape, rng)

    # --- pass 1: whole field
    warps, rho = [], np.zeros(n_types)
    for i in range(n_types):
        res = ecc_align(
            image2[:, :, i],
            image1[:, :, i],
            model=transform_global,
            levels=pyramid_levels_global,
            n_iter=n_iter,
            warp_init=identity(),
            learning_rate=learning_rate,
            rng=rng,
        )
        warps.append(res.warp)
        rho[i] = res.rho
    global_warp = warps[int(np.nanargmax(rho))]

    image2_global = np.dstack(
        [
            _fill_background(
                w := warp_image(image2[:, :, i], global_warp, transform_global, shape),
                w == 0,
                rng,
            )
            for i in range(n_types)
        ]
    )

    # --- pass 2: per tile
    grid = PatchGrid.build(shape, blocksize, overlap_frac)
    patch_warps = np.empty((blocksize, blocksize, n_types, 3, 3))
    rho_patch = np.full((blocksize, blocksize, n_types), np.nan)
    for i, j in grid.indices():
        for t in range(n_types):
            res = ecc_align(
                grid.tile(image2_global[:, :, t], i, j),
                grid.tile(image1[:, :, t], i, j),
                model=transform_patch,
                levels=pyramid_levels_patch,
                n_iter=n_iter,
                warp_init=identity(),
                learning_rate=learning_rate,
                rng=rng,
            )
            patch_warps[i, j, t] = res.warp
            rho_patch[i, j, t] = res.rho

    best = np.nanargmax(rho_patch, axis=2)
    chosen = np.stack(
        [
            np.stack([patch_warps[i, j, best[i, j]] for j in range(blocksize)])
            for i in range(blocksize)
        ]
    )

    out = []
    for t in range(n_types):
        tiles = {}
        for i, j in grid.indices():
            tile = warp_image(
                grid.tile(image2_global[:, :, t], i, j), chosen[i, j], transform_patch
            )
            tile[tile == 0] = np.nan
            tiles[(i, j)] = tile
        stitched = grid.stitch(tiles)
        out.append(_fill_background(stitched, ~np.isfinite(stitched), rng))

    return AcrossSessionResult(
        global_warp=global_warp,
        patch_warps=chosen,
        rho_global=rho,
        rho_patch=rho_patch,
        image1=image1,
        image2_global=image2_global,
        image2_patched=np.dstack(out),
        grid=grid,
    )


def apply_session_registration(
    image: np.ndarray, result: AcrossSessionResult, model: Transform = Transform.AFFINE
) -> np.ndarray:
    """Reapply a fitted registration to another image of the same session
    (e.g. an ROI mask or a per-cell activity map)."""
    shape = result.image1.shape[:2]
    globally = warp_image(
        np.asarray(image, dtype=float), result.global_warp, model, shape
    )
    tiles = {}
    for i, j in result.grid.indices():
        tile = warp_image(
            result.grid.tile(globally, i, j), result.patch_warps[i, j], model
        )
        tile[tile == 0] = np.nan
        tiles[(i, j)] = tile
    return result.grid.stitch(tiles)
