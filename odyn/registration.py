"""
A reference image for motion correction: one image a whole loop is aligned to.

Left to itself, caiman aligns each TIFF to the template the previous TIFF
produced, starting from the first one. Slow drift is then followed rather than
corrected, and a bad first acquisition (common) seeds everything. A reference
built here from the middle of the loop is passed to caiman instead, so every
acquisition is aligned to the same image.

It is built the way suite2p builds its own: the most mutually correlated
frames seed it, then it is refined by aligning frames to it and averaging the
ones that match best. Only numpy, scipy and tifffile are needed.

**USAGE**
```python
positions = reference_acquisitions(len(acquisitions))   # which acquisitions
frames = sample_frames(paths_at(positions), per_file=50)
reference = reference_image(frames, max_shift_px=(40, 40))
```
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile

from scipy import ndimage

# How many frames seed the reference, and how many refinement passes follow
SEED_FRAMES = 20
PASSES = 3


def reference_acquisitions(count: int, wanted: int = 4) -> list[int]:
    """
    The positions (0-based, in time order) of the acquisitions a loop's
    reference is built from: `wanted` of them spread over its middle third.

    The first acquisitions are often bad, and so is the first one of each
    later block, which in a two-block loop is the exact middle; spreading the
    picks around the middle third avoids both.

    **EXAMPLE**
    ```python
    reference_acquisitions(160)   # [60, 73, 86, 100]
    ```
    """
    if count <= 0 or wanted <= 0:
        return []

    wanted = min(wanted, count)

    return sorted(
        {
            min(count - 1, int(count * (1 / 3 + (i + 0.5) / (3 * wanted))))
            for i in range(wanted)
        }
    )


def sample_frames(paths: Sequence[Path], per_file: int) -> np.ndarray:
    """
    Up to `per_file` frames from each TIFF in `paths`, evenly spread in time,
    as one `(frames, height, width)` float32 array.
    """
    stacks = []

    for path in paths:
        with tifffile.TiffFile(path) as tif:
            count = len(tif.pages)
            keys = np.unique(np.linspace(0, count - 1, min(count, per_file)).round())
            frames = tif.asarray(key=[int(k) for k in keys])

        stacks.append(
            np.asarray(frames, dtype=np.float32).reshape(-1, *frames.shape[-2:])
        )

    return np.concatenate(stacks)


def reference_image(
    frames: np.ndarray,
    max_shift_px: Sequence[int],
    *,
    seed: int = SEED_FRAMES,
    passes: int = PASSES,
) -> np.ndarray:
    """
    One image for `frames` (`(frames, height, width)`) to be aligned to.

    The `seed` frames most correlated with each other are averaged first;
    then, `passes` times, every frame is aligned to the current reference
    (by at most `max_shift_px`, as `(y, x)`) and the best-matching half is
    averaged into the next one. Frames from a bad moment (movement, a
    different plane) correlate poorly and drop out.
    """
    frames = np.asarray(frames, dtype=np.float32)

    if frames.ndim != 3 or len(frames) == 0:
        raise ValueError(f"Expected (frames, height, width), got {frames.shape}.")

    reference = frames[_most_correlated(frames, seed)].mean(axis=0)
    keep = max(1, min(len(frames), max(seed, len(frames) // 2)))

    for _ in range(passes):
        shifts, scores = zip(
            *(_align(frame, reference, max_shift_px) for frame in frames)
        )
        best = np.argsort(scores)[::-1][:keep]
        total = np.zeros_like(reference)

        for index in best:
            total += _shifted(frames[index], shifts[index])

        reference = total / len(best)

    return reference.astype(np.float32)


def _most_correlated(frames: np.ndarray, seed: int) -> np.ndarray:
    """The indices of the frame with the best `seed` matches, and of those."""
    # Correlations on a coarse copy: a seed only needs the gist of each frame
    step = max(1, max(frames.shape[1:]) // 128)
    small = frames[:, ::step, ::step].reshape(len(frames), -1)
    small = small - small.mean(axis=1, keepdims=True)
    small /= np.linalg.norm(small, axis=1, keepdims=True) + 1e-12
    correlation = small @ small.T
    np.fill_diagonal(correlation, -np.inf)

    seed = max(1, min(seed, len(frames)))
    partners = np.argsort(correlation, axis=1)[:, ::-1][:, : seed - 1]
    scores = np.take_along_axis(correlation, partners, axis=1).mean(axis=1)
    best = int(np.argmax(scores)) if seed > 1 else 0

    return np.concatenate([[best], partners[best]]).astype(int)


def _align(
    frame: np.ndarray, reference: np.ndarray, max_shift_px: Sequence[int]
) -> tuple[tuple[float, float], float]:
    """
    The `(dy, dx)` that moves `frame` onto `reference` (phase correlation,
    within `max_shift_px`, to a tenth of a pixel or so), and how well the
    moved frame then matches it (Pearson correlation, away from the edges).
    """
    # Tapered to zero at the edges, which would otherwise match best unshifted
    taper = _taper(frame.shape)
    product = np.fft.fft2((reference - reference.mean()) * taper) * np.conj(
        np.fft.fft2((frame - frame.mean()) * taper)
    )
    product /= np.abs(product) + 1e-12
    surface = np.fft.fftshift(np.fft.ifft2(product).real)

    centre = np.array(surface.shape) // 2
    limit = [max(0, int(m)) for m in max_shift_px]
    window = surface[
        centre[0] - limit[0] : centre[0] + limit[0] + 1,
        centre[1] - limit[1] : centre[1] + limit[1] + 1,
    ]
    peak = np.unravel_index(int(np.argmax(window)), window.shape)
    shift = [float(peak[axis] - limit[axis]) for axis in (0, 1)]

    # Parabola through the peak and its neighbours, for the sub-pixel part
    for axis in (0, 1):
        index = list(peak)
        if 0 < peak[axis] < window.shape[axis] - 1:
            index[axis] = peak[axis] - 1
            below = window[tuple(index)]
            index[axis] = peak[axis] + 1
            above = window[tuple(index)]
            middle = window[peak]
            bend = below - 2 * middle + above

            if bend < 0:
                shift[axis] += 0.5 * (below - above) / bend

    moved = _shifted(frame, (shift[0], shift[1]))
    margin = [int(np.ceil(abs(s))) + 1 for s in shift]
    inner = (
        slice(margin[0], moved.shape[0] - margin[0]),
        slice(margin[1], moved.shape[1] - margin[1]),
    )
    a = moved[inner].ravel() - moved[inner].mean()
    b = reference[inner].ravel() - reference[inner].mean()
    score = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    return (shift[0], shift[1]), score


def _taper(shape: tuple[int, ...]) -> np.ndarray:
    """A Hann window over `shape`, 1 in the middle and 0 at the edges."""
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1])).astype(np.float32)


def _shifted(frame: np.ndarray, shift: Sequence[float]) -> np.ndarray:
    return ndimage.shift(frame, shift, order=1, mode="nearest")
