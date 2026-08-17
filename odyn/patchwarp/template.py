"""Template estimation by iterative trimmed averaging (`make_template*.m`).

The template is the average of the "cleanest" frames, where cleanliness is
measured against the current average — a fixed-point iteration on the selected
set:

    S_0 = {all frames};   m_k = mean of S_k (at 1/16 resolution)
    c_i = corr(frame_i, m_k)
    S_{k+1} = { i : c_i > Q_{1-θ}(c) }

i.e. keep the top `θ` fraction (default 0.2), five sweeps. Correlations are
computed on 16x-decimated frames, so this selects on coarse structure and
ignores single-cell activity. It is a hard-thresholded EM-flavoured robust
mean: no proof of convergence, but the set stabilises in practice after 2-3
sweeps because the retained frames dominate the mean.

`fft_denoise` additionally kills resonant-scanner stripe artefacts: in the
2D log-amplitude spectrum, take the max over rows (resp. columns) excluding the
DC cross, detrend with a 10-tap running median, and zero a ±3-bin band around
every frequency whose detrended power exceeds 0.5. The DC 5x5 block is spared.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, zoom


def _resize(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Stand-in for `imresize` (bicubic + antialias). Adequate for the
    correlation ranking, which only needs coarse structure."""
    h, w = frame.shape
    return zoom(np.asarray(frame, dtype=float), (shape[0] / h, shape[1] / w), order=1)


def make_template(
    stack: np.ndarray,
    *,
    reduction: tuple[int, int] = (16, 16),
    max_iter: int = 5,
    threshold: float = 0.2,
    fft_denoise: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(template, selected)` for a `(h, w, n)` stack."""
    stack = np.asarray(stack, dtype=float)
    h, w, n = stack.shape
    sh, sw = int(np.ceil(h / reduction[0])), int(np.ceil(w / reduction[1]))

    small = np.stack([_resize(stack[:, :, i], (sh, sw)).ravel() for i in range(n)], 1)
    selected = np.ones(n, dtype=bool)
    for _ in range(max_iter):
        m = small[:, selected].mean(axis=1)
        c = np.array([np.corrcoef(small[:, i], m)[0, 1] for i in range(n)])
        selected = c > np.quantile(c, 1 - threshold)
        if not selected.any():  # degenerate: all identical
            selected = np.ones(n, dtype=bool)
            break

    template = stack[:, :, selected].mean(axis=2)
    if fft_denoise:
        template = denoise_fft(template)
    return template, selected


def denoise_fft(
    template: np.ndarray, *, threshold: float = 0.5, remove_range: int = 3
) -> np.ndarray:
    """Notch out narrow-band stripe artefacts from a single image."""
    template = np.asarray(template, dtype=float)
    m, n = template.shape
    spec = np.fft.fftshift(np.fft.fft2(template))
    amp = np.log(np.abs(spec))
    mid_m, mid_n = round(m / 2), round(n / 2)

    def _peaks(a: np.ndarray, axis: int, block: slice, size: int) -> np.ndarray:
        masked = a.copy()
        if axis == 0:
            masked[block, :] = 0
        else:
            masked[:, block] = 0
        power = masked.max(axis=axis)
        detrended = power - median_filter(power, size=10, mode="nearest")
        return np.flatnonzero(detrended > threshold)

    cols = _peaks(amp, 0, slice(mid_m - 2, mid_m + 3), n)
    rows = _peaks(amp, 1, slice(mid_n - 2, mid_n + 3), m)

    mask = np.zeros_like(amp, dtype=bool)
    for c in cols:
        mask[:, max(c - remove_range, 0) : c + remove_range + 1] = True
    for r in rows:
        mask[max(r - remove_range, 0) : r + remove_range + 1, :] = True
    mask[mid_m - 2 : mid_m + 3, mid_n - 2 : mid_n + 3] = False  # keep DC

    spec[mask] = 0
    return np.abs(np.fft.ifft2(np.fft.ifftshift(spec)))
