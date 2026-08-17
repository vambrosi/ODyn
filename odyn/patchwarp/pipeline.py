"""Session-level orchestration (`patchwarp.m`, `patchwarp_rigid.m`).

Two stages, run in order, each consuming the previous one's TIFFs:

    raw stacks --[rigid]--> pre_warp/ --[patchwise affine]--> post_warp/

The array work lives in `rigid.py` / `affine.py`; what is left here is the
schedule that decides *which template each stack is registered against*, plus
the file loop. All filesystem access goes through `io.py`, whose functions are
stubs — the functions below are written against that interface and will run
once it is implemented.

The rigid stage's template schedule is the same continuation idea as the warp
stage. A single session-wide template fails when the field drifts, so the
session is cut into `n_template_blocks` (odd) contiguous blocks and templates
are propagated outward from the middle:

    blocks  |  4   2  |  1  |  3   5        (template ids, odd = earlier half)
            <---------  centre  --------->

Template 1 is built from the middle stacks by `template.make_template`; after
block 1 is corrected, template 2 is rebuilt from the *corrected* tail of block
1 and template 3 from its corrected head, and so on outward. Each template is
thus only ever asked to match stacks recorded adjacent in time to the data it
was built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import io
from .affine import (
    WarpField,
    WarpOptions,
    crop_motion_border,
    estimate_warp_field,
)
from .patches import apply_patch_warps
from .rigid import register_stack
from .template import make_template
from .transforms import Transform


# --------------------------------------------------------------------------
# options


@dataclass
class RigidOptions:
    """`ops.rigid_*` / channel selection."""

    n_ch: int = 1
    align_ch: int = 0
    save_ch: int = 0
    norm_method: str = "rank"  # 'rank' | 'local'
    norm_radius: float = 32
    template_block_num: int = 3  # must be odd
    template_threshold: float = 0.2
    template_tiffstack_num: int = 3
    template_center_frac: float = 0.8
    template_fftdenoise: bool = False
    pyramid_depth: int = 3
    downsample_frame_num: int = 50


@dataclass
class PatchWarpOptions:
    """Top-level `ops`."""

    source_path: Path | str = ""
    save_path: Path | str = ""
    run_rigid: bool = True
    run_affine: bool = True
    rigid: RigidOptions = field(default_factory=RigidOptions)
    warp: WarpOptions = field(default_factory=WarpOptions)
    edge_remove_pix: int = 0
    seed: int | None = None

    def __post_init__(self):
        if self.rigid.template_fftdenoise:
            # rank normalisation would undo the FFT notch
            self.rigid.norm_method = "local"
        if self.rigid.template_block_num % 2 == 0:
            raise ValueError("rigid.template_block_num must be odd")


# --------------------------------------------------------------------------
# rigid template schedule (pure)


@dataclass(frozen=True)
class RigidStep:
    """Register `stacks` against template `template`, then build `builds`.

    `builds` lists `(new template id, 'head' | 'tail')`: which end of this
    block's corrected stacks to average into the next template. The centre
    block builds two — one for each direction of travel.
    """

    stacks: range
    template: int
    builds: tuple[tuple[int, str], ...] = ()


def template_block_schedule(n_stacks: int, n_blocks: int) -> list[RigidStep]:
    """Centre-outward block/template schedule; template ids are 1-based.

    Reproduces `block_range_list` in `patchwarp_rigid.m`: block 1 is the middle,
    even template ids run forward in time, odd ids (≥3) backward, and the two
    outermost blocks absorb the remainder of the division.
    """
    if n_blocks % 2 == 0:
        raise ValueError("n_blocks must be odd")
    if n_stacks < 3 or n_blocks == 1:
        return [RigidStep(range(0, n_stacks), 1)]

    b, n = n_blocks, n_stacks
    c = int(round(n / b))
    f = b // 2

    ranges: dict[int, range] = {
        1: range(f * c - 1, (f + 1) * c),
        b - 1: range((b - 1) * c, n),
        b: range(0, c - 1),
    }
    for i in range(1, (b - 3) // 2 + 1):
        ranges[2 * i] = range((f + i) * c, (f + 1 + i) * c)
        ranges[2 * i + 1] = range((f - i) * c - 1, (f - i + 1) * c - 1)

    last = (b - 1) // 2
    schedule = [RigidStep(ranges[1], 1, ((2, "tail"), (3, "head")))]
    for i in range(1, last + 1):
        forward = ((2 * (i + 1), "tail"),) if i < last else ()
        backward = ((2 * (i + 1) + 1, "head"),) if i < last else ()
        schedule.append(RigidStep(ranges[2 * i], 2 * i, forward))
        schedule.append(RigidStep(ranges[2 * i + 1], 2 * i + 1, backward))
    return schedule


# --------------------------------------------------------------------------
# stages


def run_rigid(source_path, save_path, opts: RigidOptions) -> None:
    """Rigid motion correction of a whole session. I/O bound; see `io`."""
    files = io.list_tiff_stacks(source_path)
    n = len(files)
    k = min(opts.template_tiffstack_num, n)

    # --- template 1: iterate the template over the middle stacks, register
    # them against it, then rebuild it from the corrected result.
    mid = int(round(n / 2)) - 1
    seed_range = range(max(mid - (k - 1) // 2, 0), min(mid - (k - 1) // 2 + k, n))
    stacks = [io.read_tiff(files[i], opts.align_ch, opts.n_ch)[0] for i in seed_range]
    target, _ = make_template(
        np.concatenate(stacks, axis=2),
        threshold=opts.template_threshold,
        fft_denoise=opts.template_fftdenoise,
    )
    corrected = [
        register_stack(
            s,
            target,
            norm_method=opts.norm_method,
            norm_radius=opts.norm_radius,
            central_fraction=opts.template_center_frac,
            pyramid_depth=opts.pyramid_depth,
        )[0]
        for s in stacks
    ]
    templates = {
        1: make_template(
            np.concatenate(corrected, axis=2),
            threshold=opts.template_threshold,
            fft_denoise=opts.template_fftdenoise,
        )[0]
    }

    # --- walk outward
    for step in template_block_schedule(n, opts.template_block_num):
        corrected_block: dict[int, np.ndarray] = {}
        for idx in step.stacks:
            stack, info = io.read_tiff(files[idx], opts.align_ch, opts.n_ch)
            out, shifts = register_stack(
                stack,
                templates[step.template],
                norm_method=opts.norm_method,
                norm_radius=opts.norm_radius,
                central_fraction=opts.template_center_frac,
                pyramid_depth=opts.pyramid_depth,
            )
            corrected_block[idx] = out
            _write_corrected(save_path, files[idx], out, shifts, info, opts)

        for new_id, which in step.builds:
            picked = _edge_slice(list(step.stacks), k, which)
            templates[new_id] = make_template(
                np.concatenate([corrected_block[i] for i in picked], axis=2),
                threshold=opts.template_threshold,
                fft_denoise=opts.template_fftdenoise,
            )[0]


def _edge_slice(indices: list[int], k: int, which: str) -> list[int]:
    """The `k` stacks nearest the block's head or tail (clamped to the block)."""
    return indices[:k] if which == "head" else indices[-k:]


def _write_corrected(save_path, src, corrected, shifts, info, opts: RigidOptions):
    """Persist one corrected stack plus its summary. See `io`."""
    name = Path(src).stem
    io.write_tiff(
        Path(save_path) / f"{name}_corrected.tif", corrected.astype(np.int16), info
    )
    io.save_summary(
        Path(save_path) / f"{name}_summary.mat",
        io.StackSummary(
            shifts=shifts,
            downsampled=io.downsample_mean(corrected, opts.downsample_frame_num),
            downsampled_perstack=io.downsample_mean(corrected, np.inf),
            method=f"interpolate {opts.norm_method}",
            n_downsampled=opts.downsample_frame_num,
            n_downsampled_perstack=-1,
            info=info,
        ),
    )


def run_affine(source_path, save_path, opts: PatchWarpOptions) -> WarpField:
    """Estimate the distortion field from the per-stack movie and apply it."""
    perstack, _ = io.read_tiff(
        Path(source_path) / "downsampled" / "downsampled_perstack.tif"
    )
    stack, row_mask, col_mask = crop_motion_border(perstack, opts.edge_remove_pix)

    warp_field = estimate_warp_field(stack, opts.warp, rng=opts.seed)
    io.save_warp_field(Path(save_path) / "affine_transformation_matrix.npz", warp_field)

    files = io.list_tiff_stacks(source_path)
    if len(files) != stack.shape[2]:
        raise ValueError("one warp per TIFF stack expected")
    for idx, path in enumerate(files):
        frames, info = io.read_tiff(path)
        out = apply_warps_to_stack(
            frames,
            warp_field,
            idx,
            row_mask,
            col_mask,
            opts.edge_remove_pix,
        )
        io.write_tiff(Path(save_path) / f"{Path(path).stem}_warped.tif", out, info)
    return warp_field


def apply_warps_to_stack(
    frames: np.ndarray,
    warp_field: WarpField,
    block: int,
    row_mask: np.ndarray,
    col_mask: np.ndarray,
    edge_remove_pix: int = 0,
) -> np.ndarray:
    """Apply block `block`'s field to every frame of one stack.

    All frames of a stack share one warp — that is the temporal resolution of
    the estimate. The output is placed back into the uncropped frame so the
    geometry matches the rigid stage's output.
    """
    frames = np.asarray(frames, dtype=float)
    if edge_remove_pix:
        frames = frames[:, edge_remove_pix:-edge_remove_pix, :]
    if frames.shape[0] == row_mask.size and frames.shape[1] == col_mask.size:
        frames = frames[np.ix_(row_mask, col_mask)]

    warps = warp_field.warps[:, :, block]
    out = np.full((row_mask.size, col_mask.size, frames.shape[2]), np.nan)
    for z in range(frames.shape[2]):
        out[np.ix_(row_mask, col_mask, [z])] = apply_patch_warps(
            frames[:, :, z], warp_field.grid, warps, warp_field.model
        )[:, :, None]
    return out


def patchwarp(opts: PatchWarpOptions):
    """Run the full pipeline: `run_rigid` then `run_affine`."""
    pre = Path(opts.save_path) / "pre_warp"
    post = Path(opts.save_path) / "post_warp"
    if opts.run_rigid:
        run_rigid(opts.source_path, pre, opts.rigid)
    if opts.run_affine:
        return run_affine(pre, post, opts)
    return None


__all__ = [
    "PatchWarpOptions",
    "RigidOptions",
    "RigidStep",
    "Transform",
    "WarpOptions",
    "apply_warps_to_stack",
    "patchwarp",
    "run_affine",
    "run_rigid",
    "template_block_schedule",
]
