"""Warp models: parameterisations, differentials, and pyramid rescaling.

A warp is always carried here as a `3x3` homogeneous matrix `A` acting on
*template* coordinates,

    (x', y', w)^T = A (x, y, 1)^T,        x' <- x'/w  (homography only),

so `A` maps a point of the template to the point of the moving image whose
intensity is sampled there. (Forward warping of coordinates = inverse warping
of intensities.)

A *model* is a smooth submanifold `M ⊂ GL(3)` together with a chart
`p ↦ A(p)`. ECC needs `∂A(p)x/∂p` at each sample point, which is
`warp_jacobian`. Parameter order follows the MATLAB original: the column-major
flattening of the free block of `A`, so that `param_update` is literally
`A[:2] += dp.reshape(2, 3, order="F")` in the affine case.

    translation  p = (tx, ty)                                        nop = 2
    euclidean    p = (θ, tx, ty)                                     nop = 3
    affine       p = (a11, a21, a12, a22, a13, a23)                  nop = 6
    homography   p = (h11, h21, h31, h12, h22, h32, h13, h23)        nop = 8

Coordinates are 0-based `(x = column, y = row)`. The MATLAB code is 1-based,
which conjugates every `A` by the translation `T: x ↦ x + (1,1)`. Identity is
fixed by that conjugation and estimation/application use one convention
consistently, so nothing observable changes; only the raw values of the
translation column are shifted by `(A[:2,:2] - I) @ (1,1)`.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np


class Transform(StrEnum):
    TRANSLATION = "translation"
    EUCLIDEAN = "euclidean"
    AFFINE = "affine"
    HOMOGRAPHY = "homography"


N_PARAMS: dict[Transform, int] = {
    Transform.TRANSLATION: 2,
    Transform.EUCLIDEAN: 3,
    Transform.AFFINE: 6,
    Transform.HOMOGRAPHY: 8,
}


def identity() -> np.ndarray:
    """The `3x3` identity, the natural warp initialisation for every model."""
    return np.eye(3)


def as_matrix(warp: np.ndarray, model: Transform) -> np.ndarray:
    """Promote a `2x1` / `2x3` / `3x3` warp (MATLAB conventions) to `3x3`."""
    warp = np.asarray(warp, dtype=float)
    a = np.eye(3)
    if model is Transform.TRANSLATION and warp.size == 2:
        a[:2, 2] = warp.ravel()
        return a
    if warp.shape == (2, 3):
        a[:2, :] = warp
        return a
    if warp.shape == (3, 3):
        return warp.copy()
    raise ValueError(f"cannot interpret warp of shape {warp.shape} as {model}")


def as_compact(a: np.ndarray, model: Transform) -> np.ndarray:
    """Inverse of `as_matrix`: the storage shape the original code uses."""
    if model is Transform.TRANSLATION:
        return a[:2, 2].copy()
    if model is Transform.HOMOGRAPHY:
        return a.copy()
    return a[:2, :].copy()


def warp_points(a: np.ndarray, x: np.ndarray, y: np.ndarray, model: Transform):
    """Apply `a` to the points `(x, y)`; returns `(x', y')`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xp = a[0, 0] * x + a[0, 1] * y + a[0, 2]
    yp = a[1, 0] * x + a[1, 1] * y + a[1, 2]
    if model is Transform.HOMOGRAPHY:
        w = a[2, 0] * x + a[2, 1] * y + a[2, 2]
        xp = xp / w
        yp = yp / w
    return xp, yp


def warp_jacobian(
    model: Transform, x: np.ndarray, y: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """`∂(x', y')/∂p` at the points `(x, y)`, shape `(n, 2, nop)`.

    Affine and translation are parameter-independent; euclidean and homography
    depend on the current `a`.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    zero, one = np.zeros(n), np.ones(n)
    j = np.zeros((n, 2, N_PARAMS[model]))

    match model:
        case Transform.TRANSLATION:
            j[:, 0, :] = np.stack([one, zero], axis=1)
            j[:, 1, :] = np.stack([zero, one], axis=1)
        case Transform.EUCLIDEAN:
            cos, sin = a[0, 0], a[1, 0]  # A[:2,:2] = [[c,-s],[s,c]]
            j[:, 0, :] = np.stack([-sin * x - cos * y, one, zero], axis=1)
            j[:, 1, :] = np.stack([cos * x - sin * y, zero, one], axis=1)
        case Transform.AFFINE:
            j[:, 0, :] = np.stack([x, zero, y, zero, one, zero], axis=1)
            j[:, 1, :] = np.stack([zero, x, zero, y, zero, one], axis=1)
        case Transform.HOMOGRAPHY:
            den = a[2, 0] * x + a[2, 1] * y + a[2, 2]
            xp, yp = warp_points(a, x, y, model)
            xd, yd, od = x / den, y / den, one / den
            j[:, 0, :] = np.stack(
                [xd, zero, -xd * xp, yd, zero, -yd * xp, od, zero], axis=1
            )
            j[:, 1, :] = np.stack(
                [zero, xd, -xd * yp, zero, yd, -yd * yp, zero, od], axis=1
            )
    return j


def image_jacobian(gx: np.ndarray, gy: np.ndarray, jac: np.ndarray) -> np.ndarray:
    """`G = ∇I · ∂W/∂p`, shape `(n, nop)` — the ECC design matrix."""
    return gx[:, None] * jac[:, 0, :] + gy[:, None] * jac[:, 1, :]


def param_update(model: Transform, a: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """`A(p + dp)` — additive in the chart, not in the matrix (euclidean)."""
    out = a.copy()
    match model:
        case Transform.TRANSLATION:
            out[:2, 2] += dp
        case Transform.EUCLIDEAN:
            theta = np.sign(a[1, 0]) * np.arccos(np.clip(a[0, 0], -1.0, 1.0)) + dp[0]
            c, s = np.cos(theta), np.sin(theta)
            out[:2, :2] = [[c, -s], [s, c]]
            out[0, 2] += dp[1]
            out[1, 2] += dp[2]
        case Transform.AFFINE:
            out[:2, :] += dp.reshape(2, 3, order="F")
        case Transform.HOMOGRAPHY:
            out += np.append(dp, 0.0).reshape(3, 3, order="F")
            out[2, 2] = 1.0
    return out


def rescale(a: np.ndarray, s: float) -> np.ndarray:
    """Conjugate `a` by the dilation `diag(s, s, 1)`: the warp for a grid
    scaled by `s` (`s = 2` moves one pyramid level towards finer resolution).

    `S A S⁻¹` multiplies the translation column by `s` and divides the
    projective row by `s`, which is exactly the original `next_level`.
    """
    scale = np.diag([s, s, 1.0])
    return scale @ a @ np.diag([1 / s, 1 / s, 1.0])
