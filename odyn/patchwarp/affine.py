"""Within-session distortion field: one affine map per (tile, time block).

This is the heart of PatchWarp (`patchwarp_affine.m`). The input is the
*per-stack downsampled* movie produced by the rigid stage — one frame per TIFF
stack, i.e. one frame per few thousand raw frames — so `nz` is the number of
time blocks over the session, typically 10²–10³. The output is a field

    A : {tiles} x {time blocks} -> Aff(2),

which `patches.apply_patch_warps` then applies to every raw frame of the
corresponding stack.

Why it works. The distortion drifts slowly and smoothly in time (thermal and
mechanical relaxation of the preparation), so:

* estimating one affine per tile per *block* rather than per frame is enough
  resolution in time, and buys the SNR of a block average;
* consecutive blocks have nearly equal warps, so the ECC problem for block
  `k+1` can be *warm-started* from the solution at block `k`. That continuation
  is what lets a non-convex correlation maximisation track a distortion far
  larger than its own basin of attraction.

The continuation runs outward from the middle of the session in both
directions, because the template is built from the middle blocks:

    blocks:  0 ....... S/2-1 | S/2 ....... S-1
    order:        <---- 4 3 2 1 | 1' 2' 3' 4' ---->

with each block seeded by the median of the first (resp. last) `n_seed = 7`
*accepted* warps of the block just processed. The seed falls back to the
predecessor's own seed when a block yields nothing usable, so the chain never
breaks.

Everything after estimation is robustification of a time series of matrices:
per-estimate acceptance tests, a temporal median filter, rejection of
implausible jumps, and linear interpolation across the gaps. The field handed
downstream is therefore always complete and reasonably smooth in time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ecc import ecc_align
from .normalize import imnormalize
from .patches import PatchGrid
from .transforms import Transform, identity


def _mround(x: float) -> int:
    """MATLAB `round`: half away from zero."""
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


@dataclass
class WarpOptions:
    """Parameters of `patchwarp_affine`, same names/defaults as `ops`."""

    norm_radius: float = 32
    template_stack_num: int = 5
    movave_stack_num: int = 1
    blocksize: int = 8
    overlap_frac: float = 0.15
    n_split: int = 6
    transform: Transform = Transform.AFFINE
    pyramid_levels: int = 1
    pyramid_iterations: int = 50
    learning_rate: float = 0.1
    abssum_threshold: float = 50.0
    abssum_jump_threshold: float = 10.0
    rho_threshold: float = 0.5
    medfilt_stack_num: int = 1
    scale_bounds: tuple[float, float] = (0.6, 1.4)
    n_seed: int = 7
    # The original nulls `rho` (not the warp) when an entry blows up, and a NaN
    # `rho` then fails every later comparison — so such a warp survives unless
    # it also violates `scale_bounds`. Set True for the presumably intended
    # behaviour.
    strict_abssum: bool = False


@dataclass
class WarpField:
    """Estimated field plus the diagnostics used to clean it."""

    warps: np.ndarray  # (B, B, nz, 3, 3)
    rho: np.ndarray  # (B, B, nz)
    success: np.ndarray  # (B, B, nz) bool
    grid: PatchGrid
    model: Transform
    raw: np.ndarray | None = field(default=None, repr=False)  # pre-cleanup copy


# --------------------------------------------------------------------------
# preprocessing


def crop_motion_border(stack: np.ndarray, edge_remove_pix: int = 0):
    """Drop the border the rigid stage left blank.

    Rigid correction shifts each frame, so a rim of every frame is undefined.
    The min-projection over the interior frames is zero exactly on the union of
    those rims; rows and columns that are entirely zero there are removed.
    Returns `(cropped, row_mask, col_mask)` so the crop can be undone.
    """
    stack = np.asarray(stack, dtype=float)
    if edge_remove_pix:
        stack = stack[:, edge_remove_pix:-edge_remove_pix, :]
    interior = stack[:, :, 1:-1] if stack.shape[2] > 2 else stack[:, :, 1:]
    zero_zone = interior.min(axis=2)
    row_mask = zero_zone.sum(axis=1) != 0
    col_mask = zero_zone.sum(axis=0) != 0
    return stack[np.ix_(row_mask, col_mask)], row_mask, col_mask


def build_template(stack: np.ndarray, n_frames: int, norm_radius: float) -> np.ndarray:
    """Normalised mean of `n_frames` blocks around the middle of the session."""
    nz = stack.shape[2]
    n_frames = min(n_frames, nz)
    start = _mround(nz / 2) - 1 - _mround((n_frames - 1) / 2)
    start = int(np.clip(start, 0, nz - n_frames))
    return imnormalize(stack[:, :, start : start + n_frames].mean(axis=2), norm_radius)


def temporal_smooth(stack: np.ndarray, k: int, norm_radius: float) -> np.ndarray:
    """Normalised running mean of `k` blocks (`k = 1` is a no-op mean).

    Near the ends the window is clamped to blocks `1..k-1` / `nz-k..nz-2`
    (0-based), i.e. the very first and very last blocks are deliberately never
    used — they are the partial stacks of the session.
    """
    stack = np.asarray(stack, dtype=float)
    ny, nx, nz = stack.shape
    h = _mround((k - 1) / 2)
    out = np.empty((ny, nx, nz))
    for i in range(nz):
        if i < h:
            lo, hi = 1, k
        elif i >= nz - h:
            lo, hi = nz - k - 1, nz - 1
        else:
            lo, hi = i - h, i - h + k
        lo, hi = max(lo, 0), min(hi, nz)
        frame = imnormalize(stack[:, :, lo:hi].mean(axis=2), norm_radius)
        # the original stores int16 here; kept because it slightly quantises
        # the ECC input and therefore the estimates
        out[:, :, i] = np.round(frame).astype(np.int16)
    return out


# --------------------------------------------------------------------------
# continuation schedule


def sanitize_n_split(n_split: int, nz: int) -> int:
    """Force `n_split` even and small enough for a non-degenerate last block."""
    if n_split % 2 == 1:
        n_split = 2 if n_split == 1 else n_split - 1
    if n_split > nz:
        n_split = nz if nz % 2 == 0 else nz - 1
    if (n_split - 1) * int(np.ceil(nz / max(n_split, 1))) + 1 >= nz:
        n_split -= 2 if n_split % 2 == 0 else 3
    return max(n_split, 2)


def chunk_ranges(nz: int, n_split: int) -> list[range]:
    c = int(np.ceil(nz / n_split))
    out = [range(i * c, min((i + 1) * c, nz)) for i in range(n_split - 1)]
    out.append(range((n_split - 1) * c, nz))
    return out


def chunk_schedule(n_split: int) -> list[tuple[int, int | None]]:
    """`(chunk, chunk_to_seed_next)` in centre-outward order.

    Left half is walked backwards from `S/2 − 1` to `0`, right half forwards
    from `S/2` to `S − 1`; the two starting blocks are seeded with identity.
    """
    half = n_split // 2
    order = [(k, k - 1 if k > 0 else None) for k in range(half - 1, -1, -1)]
    order += [(k, k + 1 if k < n_split - 1 else None) for k in range(half, n_split)]
    return order


# --------------------------------------------------------------------------
# estimation


def estimate_warp_field(
    stack: np.ndarray,
    opts: WarpOptions = WarpOptions(),
    *,
    rng: np.random.Generator | int | None = None,
    progress=None,
) -> WarpField:
    """Estimate and clean the `(tile, block)` warp field of a session.

    `stack` is the `(ny, nx, nz)` per-stack downsampled movie, already cropped
    by `crop_motion_border`.
    """
    stack = np.asarray(stack, dtype=float)
    ny, nx, nz = stack.shape
    rng = np.random.default_rng(rng)

    grid = PatchGrid.build((ny, nx), opts.blocksize, opts.overlap_frac)
    b = opts.blocksize
    n_split = sanitize_n_split(opts.n_split, nz)
    ranges = chunk_ranges(nz, n_split)

    template = build_template(stack, opts.template_stack_num, opts.norm_radius)
    smoothed = temporal_smooth(stack, opts.movave_stack_num, opts.norm_radius)

    warps = np.full((b, b, nz, 3, 3), np.nan)
    rho = np.full((b, b, nz), np.nan)
    success = np.zeros((b, b, nz), dtype=bool)

    # seeds[k][i, j] is the warp initialisation for tile (i,j) in chunk k
    seeds: dict[int, np.ndarray] = {
        n_split // 2 - 1: np.tile(identity(), (b, b, 1, 1)),
        n_split // 2: np.tile(identity(), (b, b, 1, 1)),
    }

    tiles_t = {(i, j): grid.tile(template, i, j) for i, j in grid.indices()}

    for chunk, seed_next in chunk_schedule(n_split):
        seed = seeds[chunk]
        for z in ranges[chunk]:
            frame = smoothed[:, :, z]
            for i, j in grid.indices():
                res = ecc_align(
                    grid.tile(frame, i, j),
                    tiles_t[(i, j)],
                    model=opts.transform,
                    levels=opts.pyramid_levels,
                    n_iter=opts.pyramid_iterations,
                    warp_init=seed[i, j],
                    learning_rate=opts.learning_rate,
                    rng=rng,
                )
                warps[i, j, z] = res.warp
                rho[i, j, z] = res.rho
                success[i, j, z] = res.success
            if progress is not None:
                progress(z, nz)

        _accept(warps, rho, success, ranges[chunk], opts)
        if seed_next is not None:
            seeds[seed_next] = _seed_from(
                warps, ranges[chunk], seed, opts.n_seed, first=seed_next < chunk
            )

    raw = warps.copy()
    warps = _median_filter(warps, rho, opts)
    warps = _reject_jumps(warps, opts.abssum_jump_threshold)
    warps = _fill_gaps(warps, opts.transform)
    return WarpField(warps, rho, success, grid, opts.transform, raw)


def _accept(warps, rho, success, zs: range, opts: WarpOptions) -> None:
    """Mark implausible estimates as `NaN`, in place.

    Tests, in the original's order:
      1. any `|A_kl| > abssum_threshold`, or ECC reported failure -> `ρ := NaN`;
      2. diagonal outside `scale_bounds`, or `ρ ≤ rho_threshold` -> reject.
    Note that (1) makes the `ρ` comparison in (2) vacuous (`NaN ≤ x` is false),
    so a blown-up warp is only actually removed if it also fails the scale test.
    """
    lo, hi = opts.scale_bounds
    for i in range(warps.shape[0]):
        for j in range(warps.shape[1]):
            for z in zs:
                a = warps[i, j, z]
                blown = np.nanmax(np.abs(a[:2, :])) > opts.abssum_threshold
                if blown or not success[i, j, z]:
                    rho[i, j, z] = np.nan
                if opts.strict_abssum and blown:
                    warps[i, j, z] = np.nan
                    continue
                bad_scale = opts.transform is not Transform.TRANSLATION and (
                    not (lo <= a[0, 0] <= hi) or not (lo <= a[1, 1] <= hi)
                )
                if bad_scale or rho[i, j, z] <= opts.rho_threshold:
                    warps[i, j, z] = np.nan


def _seed_from(warps, zs: range, fallback, n_seed: int, *, first: bool) -> np.ndarray:
    """Median of the `n_seed` accepted warps closest to the neighbouring chunk."""
    b = warps.shape[0]
    out = fallback.copy()
    zlist = list(zs) if first else list(zs)[::-1]
    for i in range(b):
        for j in range(b):
            valid = [z for z in zlist if np.isfinite(warps[i, j, z]).all()][:n_seed]
            if valid:
                out[i, j] = np.median(np.stack([warps[i, j, z] for z in valid]), axis=0)
    return out


def _median_filter(warps: np.ndarray, rho: np.ndarray, opts: WarpOptions) -> np.ndarray:
    """Elementwise temporal median over the blocks with `ρ > rho_threshold`.

    `half = round(n/2)`, so the default `medfilt_stack_num = 1` is *not* a
    no-op: it is a 3-tap median. Windows are clamped at the ends. Unlike the
    original this is out of place, so the filter does not cascade.
    """
    nz = warps.shape[2]
    half = _mround(opts.medfilt_stack_num / 2)
    out = warps.copy()
    good = rho > opts.rho_threshold
    for z in range(nz):
        if z < half:
            window = range(0, min(2 * half, nz))
        elif z >= nz - half:
            window = range(max(nz - 2 * half, 0), nz)
        else:
            window = range(z - half, z + half + 1)
        for i in range(warps.shape[0]):
            for j in range(warps.shape[1]):
                sel = [w for w in window if good[i, j, w]]
                if not sel:
                    out[i, j, z] = np.nan
                    continue
                with np.errstate(all="ignore"):
                    out[i, j, z] = np.nanmedian(
                        np.stack([warps[i, j, w] for w in sel]), axis=0
                    )
    return out


def _reject_jumps(warps: np.ndarray, threshold: float) -> np.ndarray:
    """Null both sides of any block-to-block step with `‖ΔA‖₁ > threshold`.

    A genuine distortion drift is slow; a large one-block step is an estimation
    failure, and the interpolation that follows bridges it.
    """
    out = warps.copy()
    d = np.abs(np.diff(out[:, :, :, :2, :], axis=2)).sum(axis=(3, 4))
    idx = np.argwhere(d > threshold)
    for i, j, z in idx:
        out[i, j, z] = np.nan
        out[i, j, z + 1] = np.nan
    return out


def _interp_extrap(y: np.ndarray) -> np.ndarray:
    """Linear interpolation over `NaN`s with linear (not clamped) extrapolation."""
    n = y.size
    x = np.arange(n, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return y
    if ok.sum() == 1:
        return np.full(n, y[ok][0])
    xs, ys = x[ok], y[ok]
    out = np.interp(x, xs, ys)
    lo = x < xs[0]
    hi = x > xs[-1]
    if lo.any():
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        out[lo] = ys[0] + s * (x[lo] - xs[0])
    if hi.any():
        s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        out[hi] = ys[-1] + s * (x[hi] - xs[-1])
    return out


def _fill_gaps(warps: np.ndarray, model: Transform) -> np.ndarray:
    """Fill rejected blocks by interpolating each matrix entry along time.

    Interpolating the entries of `A` independently leaves the matrix manifold —
    for affine that is harmless (`Aff(2)` is a linear space), and the original
    only ever does this for affine (it hardcodes a `2x3` loop and falls back to
    identity otherwise). Here it is done for whichever entries the model owns;
    for `homography` a proper treatment would interpolate in `sl(3)` via the
    matrix logarithm.
    """
    out = warps.copy()
    nrow = 3 if model is Transform.HOMOGRAPHY else 2
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if not np.isfinite(out[i, j, :, :nrow, :]).any():
                out[i, j] = identity()
                continue
            for r in range(nrow):
                for c in range(3):
                    out[i, j, :, r, c] = _interp_extrap(out[i, j, :, r, c])
            if model is not Transform.HOMOGRAPHY:
                out[i, j, :, 2, :] = [0.0, 0.0, 1.0]
    return out
