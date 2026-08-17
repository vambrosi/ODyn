"""Bilinear resampling under a warp.

Two entry points, matching the original's `applyWarpOnPts` (sample at a sparse
set of points, used inside the ECC iteration) and `spatial_interp_patchwarp`
(resample a whole rectangular grid, used to render the corrected image).

Out-of-support samples are `NaN` from `bilinear_sample` and are turned into `0`
by the two warp helpers, following the original. Downstream code therefore
reads "exactly zero" as "no data"; `patches.apply_patch_warps` converts it back
to `NaN` before stitching.
"""

from __future__ import annotations

import numpy as np

from .transforms import Transform, warp_points


def bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of `img` at `(x, y)`, 0-based, `x` = column.

    Points whose 2x2 support is not fully inside the image return `NaN`.
    """
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    x0, y0 = np.floor(x), np.floor(y)
    x1, y1 = np.ceil(x), np.ceil(y)
    valid = (x0 >= 0) & (x1 <= w - 1) & (y0 >= 0) & (y1 <= h - 1)

    out = np.full(x.shape, np.nan)
    if not valid.any():
        return out

    xi0 = x0[valid].astype(np.intp)
    xi1 = x1[valid].astype(np.intp)
    yi0 = y0[valid].astype(np.intp)
    yi1 = y1[valid].astype(np.intp)
    fx = x[valid] - xi0
    fy = y[valid] - yi0

    f00 = img[yi0, xi0]
    f01 = img[yi1, xi0]
    f10 = img[yi0, xi1]
    f11 = img[yi1, xi1]
    out[valid] = (
        f00 * (1 - fx) * (1 - fy)
        + f10 * fx * (1 - fy)
        + f01 * (1 - fx) * fy
        + f11 * fx * fy
    )
    return out


def sample_warped(
    img: np.ndarray, a: np.ndarray, x: np.ndarray, y: np.ndarray, model: Transform
) -> np.ndarray:
    """`img(A(x, y))` at scattered points; out-of-support -> 0."""
    xp, yp = warp_points(a, x, y, model)
    v = bilinear_sample(img, xp, yp)
    return np.nan_to_num(v, nan=0.0)


def warp_image(
    img: np.ndarray,
    a: np.ndarray,
    model: Transform,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Render `img ∘ A` on the grid `[0, w) x [0, h)`; outside -> 0.

    `shape = (h, w)` defaults to the shape of `img`.
    """
    img = np.asarray(img, dtype=float)
    h, w = img.shape if shape is None else shape
    xx, yy = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    return sample_warped(img, a, xx.ravel(), yy.ravel(), model).reshape(h, w)


def central_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(∂x, ∂y)` by the centred difference `(f[i+1] - f[i-1])/2`, zero-padded.

    Matches `conv2(im, [.5 0 -.5], 'same')` in the original (which zero-pads at
    the border rather than switching to a one-sided rule as `np.gradient` does).
    """
    img = np.asarray(img, dtype=float)
    pad = np.pad(img, 1)
    gx = 0.5 * (pad[1:-1, 2:] - pad[1:-1, :-2])
    gy = 0.5 * (pad[2:, 1:-1] - pad[:-2, 1:-1])
    return gx, gy
