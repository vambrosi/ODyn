"""PatchWarp — correction of non-uniform image distortion in 2-photon calcium
imaging, ported from Hattori & Komiyama, *Cell Reports Methods* 2 (2022) 100205.

Pipeline:

    raw stacks
      -> `pipeline.run_rigid`     translation-only correction against
                                  iteratively re-estimated templates (`rigid`)
      -> `pipeline.run_affine`    one affine map per (tile, time block),
                                  estimated by ECC (`affine`, `ecc`), applied
                                  and stitched (`patches`)

`across_sessions.register_sessions` is the same estimator used to align summary
images from different days.

See `README.md` in this directory for the mathematical overview and a list of
the deliberate departures from the MATLAB original.
"""

from .affine import WarpField, WarpOptions, estimate_warp_field
from .across_sessions import register_sessions
from .ecc import EccResult, ecc_align
from .patches import PatchGrid, apply_patch_warps
from .pipeline import PatchWarpOptions, RigidOptions, patchwarp, run_affine, run_rigid
from .rigid import TranslationRegistrator, register_stack
from .transforms import Transform

__all__ = [
    "EccResult",
    "PatchGrid",
    "PatchWarpOptions",
    "RigidOptions",
    "Transform",
    "TranslationRegistrator",
    "WarpField",
    "WarpOptions",
    "apply_patch_warps",
    "ecc_align",
    "estimate_warp_field",
    "patchwarp",
    "register_sessions",
    "register_stack",
    "run_affine",
    "run_rigid",
]
