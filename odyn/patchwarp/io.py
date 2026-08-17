"""I/O boundary — stubs.

Everything the algorithm needs from the filesystem is funnelled through this
module so the rest of the package is pure array code. The original reads
ScanImage multi-channel TIFF stacks and writes `*_corrected.tif` +
`*_summary.mat` pairs; wire these to `tifffile` / the ODyn database when the
port is put to work.

Layout produced by the original, kept here for reference:

    save_path/pre_warp/                <stack>_corrected.tif, <stack>_summary.mat
    save_path/pre_warp/target/         template_AVG<k>.tif
    save_path/pre_warp/downsampled/    downsampled_<n>.tif, downsampled_perstack.tif
    save_path/post_warp/               <stack>_corrected_warped.tif, *_summary_warped.mat
    save_path/post_warp/affine_transformation_matrix.mat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StackSummary:
    """The per-stack sidecar (`*_summary.mat`)."""

    shifts: np.ndarray | None = None  # (n_frames, 2) rigid (dy, dx)
    downsampled: np.ndarray | None = None  # (ny, nx, n_frames/n_downsampled)
    downsampled_perstack: np.ndarray | None = None  # (ny, nx, 1)
    target: np.ndarray | None = None
    method: str = ""
    n_downsampled: int = 0
    n_downsampled_perstack: int = 0
    info: dict = field(default_factory=dict)  # TIFF tags, passed through on write


def list_tiff_stacks(source: Path | str) -> list[Path]:
    """Sorted TIFF stacks of a session; each is one time block."""
    raise NotImplementedError


def read_tiff(path: Path | str, channel: int = 0, n_channels: int = 1):
    """Return `(stack (ny, nx, n_frames), info)`, de-interleaving channels."""
    raise NotImplementedError


def write_tiff(path: Path | str, stack: np.ndarray, info: dict | None = None) -> None:
    """Write an `int16` stack, carrying `info` through as TIFF metadata."""
    raise NotImplementedError


def save_summary(path: Path | str, summary: StackSummary) -> None:
    raise NotImplementedError


def load_summary(path: Path | str) -> StackSummary:
    raise NotImplementedError


def save_warp_field(path: Path | str, field_) -> None:
    """Persist an `affine.WarpField` plus the grid needed to reapply it."""
    raise NotImplementedError


def downsample_mean(stack: np.ndarray, n: int, axis: int = 2) -> np.ndarray:
    """Non-overlapping block mean along `axis` (`downsample_chunk(..., 'mean')`).

    `n <= 0` returns an empty stack, `n = inf` collapses the axis to one frame —
    which is how `downsampled_perstack` (one frame per TIFF stack) is made.
    """
    stack = np.asarray(stack)
    if not np.isfinite(n):
        return stack.mean(axis=axis, keepdims=True)
    if n <= 0:
        shape = list(stack.shape)
        shape[axis] = 0
        return np.empty(shape, dtype=stack.dtype)
    k = stack.shape[axis] // n
    trimmed = np.take(stack, np.arange(k * n), axis=axis)
    shape = list(trimmed.shape)
    shape[axis : axis + 1] = [k, n]
    return trimmed.reshape(shape).mean(axis=axis + 1)
