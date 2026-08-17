"""Enhanced Correlation Coefficient alignment (Evangelidis & Psarakis, 2008),
in the stochastic forward-additive variant that PatchWarp ships.

Objective. For a template `t` and moving image `I`, with `W_p` the warp of
`transforms`, let `w(p) ∈ R^n` be `I ∘ W_p` sampled at `n` template points and
zero-meaned over the valid (in-support) subset, and `t̄` the template likewise.
Maximise

    ρ(p) = ⟨t̄, w(p)⟩ / (‖t̄‖ ‖w(p)‖).

Iteration. Linearise `w(p + Δp) ≈ w + G Δp` with `G = ∇I · ∂W/∂p ∈ R^{n×k}`
(`interp.central_gradients` and `transforms.image_jacobian`), put `C = GᵀG`.
Maximising the resulting quotient in `Δp` has the closed form

    λ  = (‖w‖² − wᵀG C⁻¹ Gᵀw) / (⟨t̄, w⟩ − t̄ᵀG C⁻¹ Gᵀw)
    Δp = C⁻¹ Gᵀ (λ t̄ − w).

`λ` is the projection scale that makes the linearised problem a least-squares
one; geometrically `λ t̄` is the point of the ray through `t̄` whose residual
against the current `w` is `G`-orthogonal. The update is damped by
`learning_rate` (default `0.75`), so this is a damped Gauss–Newton step on the
correlation, not the undamped step of the original paper.

Stochastic sampling. Each iteration draws `pts_per_iter` points *with
replacement* from the pixels whose gradient exceeds the mean gradient, rather
than using the whole patch. That is the 2017 modification carried in the
vendored `ecc_patchwarp.m`; it makes each step an unbiased-ish minibatch
Gauss–Newton step and is what keeps a 64-patch × 10^3-frame problem tractable.

Acceptance. Convergence of `ρ` is *not* the acceptance test. After the last
iteration the warp is rendered on the whole patch and `ρ` is recomputed there;
the result is accepted only if the warped patch still overlaps the template on
`> min_overlap` of its area, otherwise identity is returned with the
correlation of the *unwarped* pair. Downstream QC in `affine.py` uses that flag
and that `ρ`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interp import central_gradients, sample_warped, warp_image
from .transforms import (
    N_PARAMS,
    Transform,
    as_matrix,
    identity,
    image_jacobian,
    param_update,
    rescale,
    warp_jacobian,
)

_HESSIAN_COND_WARN = 1e15


@dataclass
class EccResult:
    """Outcome of one `ecc_align` call."""

    warp: np.ndarray  # 3x3, in the coordinates of the finest level
    rho: float  # correlation actually achieved (whole patch)
    success: bool  # False -> `warp` is identity, `rho` is the unwarped value
    rho_identity: float
    overlap: float
    diverged: bool = False
    history: list[float] = field(default_factory=list)  # per-iteration ρ (minibatch)


def _pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """`levels` images, index 0 = finest. Each level is a 2x2 box blur followed
    by decimation, as `conv2(x, ones(2)/4)` + `(1:2:end, 1:2:end)`."""
    out = [np.asarray(img, dtype=float)]
    for _ in range(levels - 1):
        prev = out[-1]
        full = np.zeros((prev.shape[0] + 1, prev.shape[1] + 1))
        full[:-1, :-1] += prev
        full[:-1, 1:] += prev
        full[1:, :-1] += prev
        full[1:, 1:] += prev
        out.append(full[::2, ::2] / 4.0)
    return out


def _masked_zero_mean(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the mean over `mask`, then zero out everything outside it."""
    n = int(mask.sum())
    out = v - (v * mask).sum() / max(n, 1)
    out[~mask] = 0.0
    return out


def _rho(t: np.ndarray, w: np.ndarray) -> float:
    nt, nw = np.linalg.norm(t), np.linalg.norm(w)
    if nt == 0 or nw == 0:
        return np.nan
    return float(t @ w / (nt * nw))


def ecc_align(
    image: np.ndarray,
    template: np.ndarray,
    *,
    model: Transform = Transform.AFFINE,
    levels: int = 1,
    n_iter: int = 50,
    warp_init: np.ndarray | None = None,
    learning_rate: float = 0.75,
    pts_per_iter: int = 8 * 15,
    min_overlap: float = 0.4,
    rng: np.random.Generator | int | None = None,
) -> EccResult:
    """Align `image` to `template` (same shape) and return the warp.

    `warp_init` is a warp in the coordinates of the finest level; it is pushed
    down to the coarsest pyramid level before the first iteration and pulled
    back up as the levels are refined.
    """
    image = np.asarray(image, dtype=float)
    template = np.asarray(template, dtype=float)
    if image.shape != template.shape:
        raise ValueError("image and template must have the same shape")
    rng = np.random.default_rng(rng)

    a = identity() if warp_init is None else as_matrix(warp_init, model)
    a = rescale(a, 0.5 ** (levels - 1))

    ims = _pyramid(image, levels)
    temps = _pyramid(template, levels)
    history: list[float] = []
    diverged = False
    level = 0

    for level in range(levels - 1, -1, -1):
        im, temp = ims[level], temps[level]
        gx, gy = central_gradients(im)
        # NB: signed comparison, as in the original — plausibly meant to be
        # |∇|, but it is the published behaviour, so it is kept.
        cand = np.flatnonzero((gx > gx.mean()) | (gy > gy.mean()))
        if cand.size == 0:
            continue
        ys, xs = np.unravel_index(cand, im.shape)
        flat_temp = temp.ravel()

        for it in range(n_iter):
            pick = rng.integers(0, cand.size, pts_per_iter)
            x, y = xs[pick].astype(float), ys[pick].astype(float)
            t_raw = flat_temp[cand[pick]]

            w = sample_warped(im, a, x, y, model)
            mask = w > 0  # 0 == outside the support of the warped image
            w = _masked_zero_mean(w, mask)
            t = _masked_zero_mean(t_raw, mask)
            history.append(_rho(t, w))

            if it == n_iter - 1:
                break

            wgx = sample_warped(gx, a, x, y, model)
            wgy = sample_warped(gy, a, x, y, model)
            g = image_jacobian(wgx, wgy, warp_jacobian(model, x, y, a))

            c = g.T @ g
            if not np.isfinite(c).all():
                diverged = True
                break
            with np.errstate(all="ignore"):
                gt, gw = g.T @ t, g.T @ w
                try:
                    ic_gw = np.linalg.solve(c, gw)
                except np.linalg.LinAlgError:
                    diverged = True
                    break
                num = w @ w - gw @ ic_gw
                den = t @ w - gt @ ic_gw
                lam = num / den
                try:
                    dp = np.linalg.solve(c, g.T @ (lam * t - w)) * learning_rate
                except np.linalg.LinAlgError:
                    diverged = True
                    break
            if not np.isfinite(dp).all():
                # Singular / near-singular Hessian: stop everywhere, do not
                # descend to finer levels.
                diverged = True
                break
            a = param_update(model, a, dp)

        if diverged:
            break
        if level > 0:
            a = rescale(a, 2.0)

    if diverged and level > 0:
        a = rescale(a, 2.0**level)

    # Acceptance: recompute ρ on the whole patch, warped vs. not.
    warped = warp_image(image, a, model)
    mask = warped > 0
    overlap = float(mask.mean())
    t_flat = template.ravel()
    m_flat = mask.ravel()
    rho_warp = _rho(
        _masked_zero_mean(t_flat.copy(), m_flat),
        _masked_zero_mean(warped.ravel(), m_flat),
    )
    rho_id = _rho(
        _masked_zero_mean(t_flat.copy(), m_flat),
        _masked_zero_mean(image.ravel().copy(), m_flat),
    )

    if overlap > min_overlap:
        return EccResult(a, rho_warp, True, rho_id, overlap, diverged, history)
    return EccResult(identity(), rho_id, False, rho_id, overlap, diverged, history)


def n_params(model: Transform) -> int:
    return N_PARAMS[model]
