"""Contrast normalisation used before registration.

Two-photon frames have a slowly varying gain across the field and a
frame-to-frame brightness that tracks activity. ECC is invariant to a global
affine intensity change but not to a spatially varying one, and the discrete
correlation search in `rigid.py` is not invariant to either, so both stages
normalise first.

`imnormalize` divides by a local disk mean (a homomorphic / retinex-style flat
field), `rank_transform` replaces intensities by their dense rank, which is
invariant under any monotone intensity map at the cost of destroying the
gradient magnitudes ECC needs — hence: rank for the rigid stage, disk for the
warp stage.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def disk_kernel(radius: float, supersample: int = 8) -> np.ndarray:
    """Normalised disk averaging filter, `fspecial('disk', r)`.

    MATLAB computes the exact area of the disk inside each pixel analytically;
    here it is estimated by `supersample²` point samples per pixel, which
    differs from MATLAB only in the anti-aliased boundary ring (relative error
    `O(1/(r·supersample))`) and is irrelevant at the radii used (`r ≈ 32`).
    """
    r = float(radius)
    n = int(np.ceil(r))
    grid = (
        np.arange(-n, n + 1)[:, None]
        + (np.arange(supersample) + 0.5) / supersample
        - 0.5
    )
    coords = grid.ravel()
    inside = (coords[:, None] ** 2 + coords[None, :] ** 2) <= r**2
    k = inside.reshape(2 * n + 1, supersample, 2 * n + 1, supersample).mean(axis=(1, 3))
    return k / k.sum()


def imnormalize(im: np.ndarray, radius: float) -> np.ndarray:
    """`imnormalize2`: `mean(im) · im / localmean(im, radius)`.

    The local mean is boundary corrected by dividing the convolution of the
    image by the convolution of an all-ones image, i.e. weights are
    renormalised over the part of the disk that lies inside the frame.
    """
    im = np.asarray(im, dtype=float)
    f = disk_kernel(radius)
    if im.ndim == 2:
        im = im[:, :, None]
        squeeze = True
    else:
        squeeze = False
    weight = fftconvolve(np.ones(im.shape[:2]), f, mode="same")
    out = np.empty_like(im, dtype=float)
    for i in range(im.shape[2]):
        out[:, :, i] = fftconvolve(im[:, :, i], f, mode="same") / weight
    out = im.mean() * im / out
    return out[:, :, 0] if squeeze else out


def rank_transform(im: np.ndarray) -> np.ndarray:
    """Per-frame dense rank of the pixel values (1-based, as in MATLAB).

    Invariant under any strictly increasing intensity transform; used for the
    rigid stage where only the location of the correlation peak matters.
    """
    im = np.asarray(im)
    if im.ndim == 2:
        return (
            (np.unique(im, return_inverse=True)[1] + 1).reshape(im.shape).astype(float)
        )
    out = np.empty(im.shape, dtype=float)
    for i in range(im.shape[2]):
        frame = im[:, :, i]
        out[:, :, i] = (np.unique(frame, return_inverse=True)[1] + 1).reshape(
            frame.shape
        )
    return out
