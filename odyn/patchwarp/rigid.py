"""Rigid (translation-only) registration by a coarse-to-fine correlation search.

Port of Aki Mitani's `ImageRegistrator` family (Mitani & Komiyama, *Front.
Neuroinform.* 2018). The model is an integer-plus-subpixel translation `d`
maximising the Pearson correlation between a fixed central crop of the target
and the correspondingly shifted window of the source:

    d* = argmax_d corr( T[B], S[B − d] ),   B = central `central_fraction` box.

Cropping the target is what makes the objective well defined: every candidate
`d` compares the same number of pixels, so there is no need for a normalisation
that trades overlap against similarity.

Three nested refinements:

1. **Greedy hill climb** over the 8-neighbourhood plus stay, from `d0`, at most
   `⌈((H−m)+(W−n))/2⌉` steps. Correlations are memoised per source frame, so
   the walk costs one evaluation per newly visited offset.
2. **Pyramid**: the same walk is run on box-decimated copies from depth
   `2^depth` upwards, each level seeding the next with `2·d`. This converts the
   `O(displacement)` walk length into `O(log)` and, more importantly, makes the
   objective's basin wide enough that the greedy walk does not stop at a
   local maximum induced by cell-scale texture.
3. **Subpixel**: fit `z = a(x²+y²) + bx + cy + e` — an *isotropic* paraboloid,
   4 parameters — by least squares to the 5-point cross of correlations around
   the integer optimum, and take its vertex `d = −(b, c)/2a`. Isotropy is the
   assumption that the correlation peak's curvature is direction independent;
   with 5 points and 4 parameters the fit is nearly interpolating, and
   `a = ¼Σ_cross z − z_0` is a discrete Laplacian. Rejected (fall back to the
   integer optimum) when `a ≥ 0`, i.e. when the stencil is not concave.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STEPS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def central_box(shape: tuple[int, int], fraction: float) -> tuple[slice, slice]:
    """The central `fraction` (per axis) sub-box of an array of `shape`."""
    edge = (1 - fraction) / 2
    starts = [int(np.floor(s * edge)) for s in shape]
    stops = [int(np.ceil(s * (1 - edge))) for s in shape]
    return slice(starts[0], stops[0]), slice(starts[1], stops[1])


def _decimate(img: np.ndarray) -> np.ndarray:
    """2x box downsample (`imresize(..., 1/2, 'box')`), dropping an odd tail."""
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    return img[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


class _Correlator:
    """Memoised `corr(target[box], source[box − d])` for one source frame."""

    def __init__(self, target: np.ndarray, box, source: np.ndarray):
        self.rows, self.cols = box
        self.tgt = np.asarray(target[box], dtype=float).ravel()
        self.tgt = self.tgt - self.tgt.mean()
        self.tgt_norm = np.linalg.norm(self.tgt)
        self.source = np.asarray(source, dtype=float)
        self.m = self.rows.stop - self.rows.start
        self.n = self.cols.stop - self.cols.start
        self.cache: dict[tuple[int, int], float] = {}

    def get(self, d: tuple[int, int]) -> float:
        dy, dx = int(d[0]), int(d[1])
        if (dy, dx) in self.cache:
            return self.cache[(dy, dx)]
        r0, c0 = self.rows.start - dy, self.cols.start - dx
        h, w = self.source.shape
        if r0 < 0 or c0 < 0 or r0 + self.m > h or c0 + self.n > w:
            return -2.0  # outside the search domain, as in the original
        win = self.source[r0 : r0 + self.m, c0 : c0 + self.n].ravel()
        win = win - win.mean()
        denom = self.tgt_norm * np.linalg.norm(win)
        cor = 0.0 if denom == 0 else float(np.clip(self.tgt @ win / denom, -1, 1))
        self.cache[(dy, dx)] = cor
        return cor


def _subpixel_peak(cross: dict[tuple[int, int], float]) -> tuple[np.ndarray, float]:
    """Vertex of the isotropic paraboloid through the 5-point cross.

    `cross` maps `(dy, dx) ∈ {(0,0), (±1,0), (0,±1)}` to correlation.
    """
    z0 = cross[(0, 0)]
    zm0, zp0 = cross[(-1, 0)], cross[(1, 0)]
    z0m, z0p = cross[(0, -1)], cross[(0, 1)]
    a = 0.25 * (zm0 + z0m + z0p + zp0) - z0
    b = 0.5 * (zp0 - zm0)  # ∂/∂y
    c = 0.5 * (z0p - z0m)  # ∂/∂x
    if a >= 0 or not np.isfinite([a, b, c]).all():
        return np.zeros(2), z0
    return np.array([-b / (2 * a), -c / (2 * a)]), z0 - (b**2 + c**2) / (4 * a)


@dataclass
class TranslationRegistrator:
    """Pyramid + subpixel translation registration against a fixed target."""

    target: np.ndarray
    central_fraction: float = 0.75
    pyramid_depth: int = 3

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=float)
        self.box = central_box(self.target.shape, self.central_fraction)
        self._targets = [self.target]
        for _ in range(self.pyramid_depth):
            self._targets.append(_decimate(self._targets[-1]))
        self._boxes = [
            central_box(t.shape, self.central_fraction) for t in self._targets
        ]

    def _hill_climb(self, corr: _Correlator, d0) -> tuple[np.ndarray, float]:
        d = np.asarray(d0, dtype=int).copy()
        span = (corr.source.shape[0] - corr.m) + (corr.source.shape[1] - corr.n)
        for _ in range(int(np.ceil(span / 2)) + 1):
            values = [corr.get(tuple(d + np.array(s))) for s in _STEPS]
            best = int(np.argmax(values))
            if best == 0:
                return d, values[0]
            d = d + np.array(_STEPS[best])
        return d, corr.get(tuple(d))

    def register(self, source: np.ndarray, d0=(0, 0)) -> tuple[np.ndarray, float]:
        """Return `(d, correlation)`; `d = (dy, dx)`, possibly fractional."""
        source = np.asarray(source, dtype=float)
        pyr = [source]
        for _ in range(self.pyramid_depth):
            pyr.append(_decimate(pyr[-1]))

        d = np.floor(np.asarray(d0, dtype=float) / 2**self.pyramid_depth).astype(int)
        for level in range(self.pyramid_depth, 0, -1):
            corr = _Correlator(self._targets[level], self._boxes[level], pyr[level])
            d, _ = self._hill_climb(corr, d)
            d = 2 * d

        corr = _Correlator(self.target, self.box, source)
        d_int, _ = self._hill_climb(corr, d)
        cross = {s: corr.get(tuple(d_int + np.array(s))) for s in _STEPS[:5]}
        frac, c = _subpixel_peak(cross)
        return d_int + frac, c


def integer_shift(frame: np.ndarray, d) -> np.ndarray:
    """Shift by integer `d = (dy, dx)`, filling the vacated border with `NaN`."""
    dy, dx = int(d[0]), int(d[1])
    h, w = frame.shape
    out = np.full((h, w), np.nan)
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = frame[
        -min(dy, 0) : h - max(dy, 0), -min(dx, 0) : w - max(dx, 0)
    ]
    return out


def shift(frame: np.ndarray, d) -> np.ndarray:
    """Shift by a possibly fractional `d = (dy, dx)`.

    Integer part by slicing, fractional part by convolution with the separable
    bilinear kernel `[1−t, t]` — identical to bilinear interpolation, but
    expressed as a filter so the whole frame is done in one pass.
    """
    frame = np.asarray(frame, dtype=float)
    dy, dx = float(d[0]), float(d[1])
    out = integer_shift(frame, (np.fix(dy), np.fix(dx)))
    fy, fx = dy - np.fix(dy), dx - np.fix(dx)
    if fy == 0 and fx == 0:
        return out

    ky = np.array([1 - abs(fy), abs(fy)]) if fy != 0 else np.array([1.0])
    kx = np.array([1 - abs(fx), abs(fx)]) if fx != 0 else np.array([1.0])
    if fy < 0:
        ky = ky[::-1]
    if fx < 0:
        kx = kx[::-1]

    h, w = out.shape
    padded = np.zeros((h + len(ky) - 1, w + len(kx) - 1))
    for i, a in enumerate(ky):
        for j, b in enumerate(kx):
            padded[i : i + h, j : j + w] += a * b * out
    r0 = 1 if fy < 0 else 0
    c0 = 1 if fx < 0 else 0
    return padded[r0 : r0 + h, c0 : c0 + w]


def register_stack(
    stack_align: np.ndarray,
    target: np.ndarray,
    *,
    norm_method: str = "rank",
    norm_radius: float = 32,
    central_fraction: float = 0.8,
    pyramid_depth: int = 3,
    stack_save: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly correct one TIFF stack against `target` (`pyramid_registration.m`).

    `stack_align` is the channel the shifts are estimated on, `stack_save` the
    channel they are applied to (defaults to the same). Normalisation is applied
    to the alignment channel and the target only — the saved data is never
    normalised, only shifted.

    Returns `(corrected, shifts)` with `shifts[i] = (dy, dx)`.
    """
    from .normalize import imnormalize, rank_transform

    stack_align = np.asarray(stack_align, dtype=float)
    if stack_save is None:
        stack_save = stack_align

    match norm_method:
        case "rank":
            target = rank_transform(np.asarray(target, dtype=float))
            stack_align = rank_transform(stack_align)
        case "local":
            target = imnormalize(np.asarray(target, dtype=float), norm_radius)
            stack_align = imnormalize(stack_align, norm_radius)
        case _:
            raise ValueError(f"unknown normalisation method {norm_method!r}")

    reg = TranslationRegistrator(target, central_fraction, pyramid_depth)
    n = stack_align.shape[2]
    shifts = np.zeros((n, 2))
    corrected = np.empty(stack_save.shape, dtype=float)
    for i in range(n):
        shifts[i], _ = reg.register(stack_align[:, :, i])
        corrected[:, :, i] = shift(stack_save[:, :, i], shifts[i])
    return corrected, shifts
