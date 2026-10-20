"""
The files a test motion correction is judged by, before the app shows it.

For the acquisitions a test ran on, written into one folder:
- `reference.tif`: the image every acquisition was aligned to;
- `side_by_side.tif` (when asked): one page per acquisition, its mean
image raw | test, each column scaled on its own. Scrubbed in Fiji, drift
shows as the picture sliding from page to page;
- `summary.csv`: per acquisition, how its images match the reference (see
below), and the shift caiman applied (for piecewise rigid, the median of
the patches' shifts);
- `metrics.json`: the numbers for the whole test, and the settings it ran
with when given;
- `flow_average.tif`: how far the corrected means still are from the
reference at each point, averaged over the acquisitions (in px, brighter is
more), which shows where a rigid correction falls short (one part of the
field moving apart from the rest, say);
- `flow_worst.tif`: the same for the acquisition with the most of it;
- `shifts.npz`, `shifts.png`: the shift of every frame (and, when asked, of
every patch, for piecewise rigid) and the measures below, as plots.

What is measured, and why this way rather than as NoRMCorre and caiman do:
- **Mean against the reference** (`*_corr_with_ref`): each acquisition's mean
image against one image, the same for raw and test. Averaging first removes
most noise and activity, so this follows drift between acquisitions. The
test mean is also compared with the average of all test means
(`test_corr_with_avg`): how well it fits the rest.
Comparing each movie with its own mean (NoRMCorre's `motion_metrics`) is
circular: a movie that drifted has a blurred mean that its frames still
match.
- **Frames against the reference** (`*_frame_corr_with_ref_median`, `_p10`,
`_p90`): the same per frame, summarized per acquisition. Single frames are
noisy, so they match less than the means do; the 10th percentile catches a
moment of movement that the mean smooths over. Noise and odor responses lower it too,
so it compares raw with test, or candidates, rather than reading as a grade.
- **Residual flow** (`test_flow_median_px`, `_p10_px`, `_p90_px`): the
optical flow (Farneback, as caiman's metrics) from the reference to each
corrected mean, the median and percentiles of its length over the field
(away from the edges). A 90th percentile far above the median means part of
the field moves on its own. What a rigid correction leaves is mostly
non-rigid motion, so this says whether piecewise rigid would help.
- **Patch jitter** (piecewise rigid only, `test_patch_jitter_median_px`,
`_p90_px`; `patch_jitter_median_px` in `metrics.json`): how far each patch's
own shift, apart from the frame's (the median of the patches'), moves from
one frame to the next, the median over the patches. Tissue bends slowly, so
a large jitter means the patches follow noise (too small, or too free to
deviate) rather than the tissue.
- **Crispness** (`metrics.json`): the mean squared gradient of the average of
all means, divided by its variance, so it compares across experiments and
brightness (NoRMCorre's is the unscaled norm of the gradient). Drift blurs
the average and lowers it.

Only numpy, tifffile, OpenCV and matplotlib: the corrected movies come in
through `load`, so caiman stays with the caller.

**USAGE**
```python
paths = write_report(
    folder, raw_paths, mmap_paths, load, shifts, reference,
    max_shifts=(60, 60), positions=[3, 12, ...], acq_ids=[...],
    settings={...}, patch_shifts=(dy_els, dx_els),
)
```
"""

from __future__ import annotations

import csv
import json

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import tifffile


def write_report(
    folder: Path,
    raw_paths: Sequence[Path],
    corrected_paths: Sequence[str | Path],
    load: Callable[[str | Path], np.ndarray],
    shifts: np.ndarray,
    reference: None | np.ndarray,
    *,
    max_shifts: Sequence[int],
    positions: Sequence[int],
    acq_ids: Sequence[int],
    settings: None | dict = None,
    patch_shifts: None | tuple[np.ndarray, np.ndarray] = None,
    keep_patch_shifts: bool = False,
    side_by_side: bool = False,
) -> list[Path]:
    """
    Write the report on one test into `folder` (see the module's description)
    and return the files. `raw_paths` and `corrected_paths` are the test's
    acquisitions in order, `load` reads a corrected one as `(frames, height,
    width)`, `shifts` holds every frame's `(dy, dx)` one acquisition after
    the other (for piecewise rigid, the median of the patches'), and
    `positions` and `acq_ids` say which acquisitions they were.
    Without a `reference` (caiman's own template), the average of the
    corrected means stands in for it, and the frames are not compared.
    `patch_shifts` are piecewise rigid's `(dy, dx)` per frame and patch, each
    `(frames, patches)`, which the patch jitter is measured from; with
    `keep_patch_shifts` they are saved in `shifts.npz` too (hundreds of
    megabytes at small patches). `settings` go into `metrics.json` as they
    are.
    """
    folder.mkdir(parents=True, exist_ok=True)
    margin = max(max_shifts) + 2
    given = None if reference is None else np.asarray(reference, np.float32)

    raw, raw_frames = [], []

    for path in raw_paths:
        mean, frames = _summarize(_tiff_frames(path), given, margin)
        raw.append(mean)
        raw_frames.append(frames)

    test, test_frames, frame_counts = [], [], []

    for path in corrected_paths:
        movie = np.asarray(load(path))
        frame_counts.append(len(movie))
        mean, frames = _summarize(movie, given, margin)
        test.append(mean)
        test_frames.append(frames)
        del movie

    average = np.mean(test, axis=0)
    reference = average if given is None else given
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 2)
    edges = np.concatenate([[0], np.cumsum(frame_counts)]).astype(int)
    flows = [residual_flow(reference, image, margin) for image in test]
    jitter = None if patch_shifts is None else _patch_jitter(shifts, patch_shifts)

    rows = []

    for i, (position, acq_id, path) in enumerate(zip(positions, acq_ids, raw_paths)):
        part = shifts[edges[i] : edges[i + 1]]
        magnitude = _inner(np.hypot(flows[i][..., 0], flows[i][..., 1]), margin)
        rows.append(
            {
                "position": int(position),
                "acq_id": int(acq_id),
                "file": Path(path).name,
                "raw_corr_with_ref": _round(correlation(raw[i], reference, margin)),
                "test_corr_with_ref": _round(correlation(test[i], reference, margin)),
                "test_corr_with_avg": _round(correlation(test[i], average, margin)),
                "raw_frame_corr_with_ref_median": _round(_median(raw_frames[i])),
                "raw_frame_corr_with_ref_p10": _round(_percentile(raw_frames[i], 10)),
                "raw_frame_corr_with_ref_p90": _round(_percentile(raw_frames[i], 90)),
                "test_frame_corr_with_ref_median": _round(_median(test_frames[i])),
                "test_frame_corr_with_ref_p10": _round(_percentile(test_frames[i], 10)),
                "test_frame_corr_with_ref_p90": _round(_percentile(test_frames[i], 90)),
                "test_flow_median_px": _round(float(np.median(magnitude)), 2),
                "test_flow_p10_px": _round(float(np.percentile(magnitude, 10)), 2),
                "test_flow_p90_px": _round(float(np.percentile(magnitude, 90)), 2),
                **(
                    {}
                    if jitter is None
                    else _jitter_columns(jitter[edges[i] + 1 : edges[i + 1]])
                ),
                "dy_median": _round(_median(part[:, 0]), 2),
                "dx_median": _round(_median(part[:, 1]), 2),
                "at_limit": (
                    _round(float(_at_limit(part, max_shifts).mean()), 3)
                    if len(part)
                    else None
                ),
            }
        )

    worst = int(np.argmax([row["test_flow_median_px"] for row in rows]))
    metrics = {
        "crispness_raw": _round(crispness(np.mean(raw, axis=0), margin), 5),
        "crispness_test": _round(crispness(average, margin), 5),
        "flow_worst_acq_id": rows[worst]["acq_id"],
        "flow_worst_median_px": rows[worst]["test_flow_median_px"],
        "frames_at_limit": _round(float(_at_limit(shifts, max_shifts).mean()), 4),
    }

    if jitter is not None:
        # Steps from one acquisition into the next are not motion
        steps = np.concatenate(
            [jitter[edges[i] + 1 : edges[i + 1]] for i in range(len(rows))]
        )
        metrics["patch_jitter_median_px"] = _round(_median(steps), 3)
        metrics["patch_jitter_p90_px"] = _round(_percentile(steps, 90), 3)

    if settings is not None:
        metrics["settings"] = settings

    written = {
        "reference": folder / "reference.tif",
        "side_by_side": folder / "side_by_side.tif",
        "summary": folder / "summary.csv",
        "metrics": folder / "metrics.json",
        "flow_average": folder / "flow_average.tif",
        "flow": folder / "flow_worst.tif",
        "npz": folder / "shifts.npz",
        "png": folder / "shifts.png",
    }
    tifffile.imwrite(written["reference"], reference.astype(np.float32))

    if side_by_side:
        # Pages, not colors: tifffile takes a stack of 3 or 4 for an RGB image
        tifffile.imwrite(
            written["side_by_side"],
            side_by_side_pages([raw, test]),
            photometric="minisblack",
        )
    else:
        del written["side_by_side"]

    _write_csv(written["summary"], rows)
    written["metrics"].write_text(json.dumps(metrics, indent=2))

    lengths = [np.hypot(flow[..., 0], flow[..., 1]) for flow in flows]
    tifffile.imwrite(
        written["flow_average"], np.mean(lengths, axis=0).astype(np.float32)
    )

    tifffile.imwrite(written["flow"], lengths[worst].astype(np.float32))
    np.savez(
        written["npz"],
        shifts=shifts,
        frame_counts=np.array(frame_counts),
        positions=np.array(positions),
        acq_ids=np.array(acq_ids),
        **(
            {}
            if patch_shifts is None or not keep_patch_shifts
            else {
                "patch_dy": np.asarray(patch_shifts[0], np.float32),
                "patch_dx": np.asarray(patch_shifts[1], np.float32),
            }
        ),
    )
    plot_shifts(written["png"], shifts, frame_counts, rows, max_shifts)

    return list(written.values())


def correlation(a: np.ndarray, b: np.ndarray, margin: int) -> float:
    """Pearson correlation of two images, away from a `margin` at the edges."""
    a = _inner(np.asarray(a, np.float64), margin).ravel()
    b = _inner(np.asarray(b, np.float64), margin).ravel()
    a = a - a.mean()
    b = b - b.mean()

    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def crispness(image: np.ndarray, margin: int) -> float:
    """Mean squared gradient over the variance: a blurred image scores lower."""
    inner = _inner(np.asarray(image, np.float64), margin)
    gy, gx = np.gradient(inner)

    return float(np.mean(gy**2 + gx**2) / (inner.var() + 1e-12))


def residual_flow(reference: np.ndarray, image: np.ndarray, margin: int) -> np.ndarray:
    """
    The optical flow from `reference` to `image`, `(height, width, 2)` as
    `(dx, dy)` in pixels, zero at the `margin`. Farneback's method with the
    settings caiman's metrics use, on both images scaled to 0-255 by the
    reference's 1st and 99.5th percentiles (the method is tuned to that range).
    """
    import cv2

    low, high = np.percentile(reference, [1, 99.5])

    def scaled(picture: np.ndarray) -> np.ndarray:
        values = (np.asarray(picture, np.float32) - low) / (high - low + 1e-12)
        return np.clip(values * 255, 0, 255).astype(np.float32)

    flow = cv2.calcOpticalFlowFarneback(
        scaled(reference), scaled(image), None, 0.5, 3, 100, 15, 5, 1.2 / 5, 0
    )
    edge = np.zeros(flow.shape[:2], bool)
    edge[margin : -margin or None, margin : -margin or None] = True
    flow[~edge] = 0

    return flow


def side_by_side_pages(
    columns: Sequence[Sequence[np.ndarray]], gap: int = 8
) -> np.ndarray:
    """
    One page per image of the columns, the columns next to each other, each
    scaled 0 to 1 by its own 1st and 99.5th percentiles (so they compare by
    position, not brightness).
    """
    pages = len(columns[0])
    height, width = np.asarray(columns[0][0]).shape
    canvas = np.zeros(
        (pages, height, len(columns) * width + (len(columns) - 1) * gap), np.float32
    )

    for column, images in enumerate(columns):
        stack = np.stack(images)
        low, high = np.percentile(stack, [1, 99.5])
        left = column * (width + gap)
        canvas[:, :, left : left + width] = np.clip(
            (stack - low) / (high - low + 1e-12), 0, 1
        )

    return canvas


def plot_shifts(
    path: Path,
    shifts: np.ndarray,
    frame_counts: Sequence[int],
    rows: list[dict],
    max_shifts: Sequence[int],
) -> None:
    """
    The shifts of every frame (dy, dx), how each acquisition's frames match the
    reference, and the motion left in its mean, as a PNG. The shift axes fit
    the shifts, and say the limit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 12, "legend.fontsize": 11}):
        _draw_shifts(plt, path, shifts, frame_counts, rows, max_shifts)


def _draw_shifts(plt, path, shifts, frame_counts, rows, max_shifts) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.25)
    fig.suptitle("Test motion correction", fontsize=16)
    edges = np.cumsum(frame_counts)[:-1]

    for axis, (index, name) in zip(axes[:2], ((0, "dy"), (1, "dx"))):
        values = shifts[:, index]
        axis.plot(values, lw=0.6, color="tab:green")

        for edge in edges:
            axis.axvline(edge, color="0.85", lw=0.5, zorder=0)

        reach = max(1.0, float(np.abs(values).max()) if len(values) else 1.0)
        axis.set_ylim(-1.2 * reach, 1.2 * reach)
        axis.set_ylabel(f"shift {name} (px)")
        axis.text(
            0.995,
            0.95,
            f"limit ±{max_shifts[index]} px",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="0.4",
        )

    axes[1].set_xlabel("frame (acquisitions one after the other, gray lines between)")

    positions = [row["position"] for row in rows]

    for who, color in (("raw", "tab:blue"), ("test", "tab:green")):
        _median_and_band(
            axes[2],
            positions,
            [row[f"{who}_frame_corr_with_ref_median"] for row in rows],
            [row[f"{who}_frame_corr_with_ref_p10"] for row in rows],
            [row[f"{who}_frame_corr_with_ref_p90"] for row in rows],
            f"{who} frames",
            color,
        )

    axes[2].set_ylabel("frame correlation\nwith the reference")
    axes[2].legend(loc="best")

    _median_and_band(
        axes[3],
        positions,
        [row["test_flow_median_px"] for row in rows],
        [row["test_flow_p10_px"] for row in rows],
        [row["test_flow_p90_px"] for row in rows],
        "over the field",
    )
    axes[3].set_ylim(bottom=0)
    axes[3].set_ylabel("motion left in the\ntest mean (px)")
    axes[3].legend(loc="best")
    axes[3].set_xlabel("acquisition position in the experiment")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def _median_and_band(
    axis, x, median, low, high, what: str, color: str = "tab:green"
) -> None:
    """A median line and the 10th to 90th percentile around it, shaded."""
    axis.plot(x, median, marker="o", color=color, label=f"{what}, median")

    if all(value is not None for value in low + high):
        axis.fill_between(
            x,
            low,
            high,
            color=color,
            alpha=0.15,
            lw=0,
            label=f"{what}, 10th to 90th percentile",
        )


def _patch_jitter(
    shifts: np.ndarray, patch_shifts: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Per frame, the median over the patches of how far each patch's shift,
    less the rigid one, moved since the frame before (the first is 0).
    """
    dy = np.asarray(patch_shifts[0], float) - shifts[:, :1]
    dx = np.asarray(patch_shifts[1], float) - shifts[:, 1:]
    steps = np.hypot(np.diff(dy, axis=0), np.diff(dx, axis=0))
    return np.concatenate([[0.0], np.median(steps, axis=1)])


def _jitter_columns(steps: np.ndarray) -> dict:
    return {
        "test_patch_jitter_median_px": _round(_median(steps), 3),
        "test_patch_jitter_p90_px": _round(_percentile(steps, 90), 3),
    }


def _summarize(
    frames: Iterable[np.ndarray], reference: None | np.ndarray, margin: int
) -> tuple[np.ndarray, list[float]]:
    """The mean of `frames`, and each frame's correlation with `reference`."""
    total, count, matches = None, 0, []

    if reference is not None:
        target = _inner(reference, margin).astype(np.float64).ravel()
        target = (target - target.mean()) / (
            np.linalg.norm(target - target.mean()) + 1e-12
        )

    for frame in frames:
        frame = np.asarray(frame, np.float64)
        total = frame.copy() if total is None else total + frame
        count += 1

        if reference is not None:
            values = _inner(frame, margin).ravel()
            values = values - values.mean()
            matches.append(float(values @ target / (np.linalg.norm(values) + 1e-12)))

    return (total / count).astype(np.float32), matches


def _tiff_frames(path: Path):
    """A TIFF's frames, one page at a time."""
    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            yield page.asarray()


def _inner(image: np.ndarray, margin: int) -> np.ndarray:
    return image[margin : -margin or None, margin : -margin or None]


def _median(values) -> None | float:
    return float(np.median(values)) if len(values) else None


def _percentile(values, q: float) -> None | float:
    return float(np.percentile(values, q)) if len(values) else None


def _round(value: None | float, digits: int = 4) -> None | float:
    return None if value is None else round(value, digits)


def _at_limit(shifts: np.ndarray, max_shifts: Sequence[int]) -> np.ndarray:
    """Which frames were pinned at the largest shift allowed."""
    return (np.abs(shifts) >= np.array(max_shifts) - 0.5).any(axis=1)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
