# PatchWarp, ported to Python

A port of [PatchWarp](https://github.com/ryhattori/PatchWarp) (Hattori & Komiyama,
*Cell Reports Methods* **2** (2022) 100205) from MATLAB to NumPy. The algorithm is
complete and exercised end to end on synthetic data; everything touching the
filesystem is a stub in `io.py`.

## The problem

A two-photon session of a few hours produces a stack of frames whose geometry
drifts. Two distinct effects:

* **fast, rigid** — breathing/heartbeat/locomotion translate the field frame to
  frame by a few pixels;
* **slow, non-uniform** — thermal and mechanical relaxation of the preparation
  and of the objective *deform* the field over tens of minutes, differently in
  different parts of the FOV.

The second is the one PatchWarp exists for. Its model is *piecewise affine in
space, piecewise constant in time*: cut the field into a `B x B` grid of
overlapping tiles, cut the session into blocks of frames, and fit one affine map

  `A(tile, block) ∈ Aff(2)`

to each cell of that product. Fitting a global affine would not work (the
distortion is not uniform across the FOV); fitting a dense flow field would not
be identifiable from calcium data, where the "texture" is sparse cells whose
brightness changes for reasons unrelated to geometry. A grid of affines is the
compromise: enough spatial degrees of freedom to follow the distortion, few
enough parameters per tile (6) that a block average of frames determines them.

## Layout

| module | contents |
|---|---|
| `transforms.py` | the four warp models, their charts, differentials, pyramid rescaling |
| `interp.py` | bilinear resampling, forward warping of points and of grids |
| `ecc.py` | the ECC estimator — the numerical core |
| `rigid.py` | translation-only pyramid correlation search (Mitani's registrator) |
| `normalize.py` | local-disk normalisation, rank transform |
| `template.py` | iterative trimmed-mean template, FFT stripe notch |
| `patches.py` | the tile grid and the stitch |
| `affine.py` | the `(tile, block)` warp field: estimation + robustification |
| `across_sessions.py` | day-to-day registration (same estimator, two passes) |
| `pipeline.py` | session orchestration, rigid template schedule |
| `io.py` | stubs: TIFF/summary read & write, `downsample_mean` |

## Stage 1 — rigid correction (`rigid.py`, `pipeline.run_rigid`)

Per frame, estimate a translation `d` maximising the Pearson correlation between
a fixed central crop `B` of the target and the shifted window of the source:

  `d* = argmax_d corr( T[B], S[B − d] )`.

Cropping the target is what makes this well posed — every candidate `d` compares
the same number of pixels, so no overlap-versus-similarity trade-off appears and
the maximiser is not biased towards `d = 0`. The crop fraction also bounds the
maximum detectable displacement to `±(1−f)/2` of the frame.

Three nested refinements:

1. greedy hill climb over the 8-neighbourhood, memoised;
2. a `2^depth` pyramid of box-decimated copies, each level seeding the next with
   `2d`. This is not only a speedup: decimation widens the basin of the
   correlation peak, so the greedy walk does not stall on the local maximum that
   cell-scale texture creates one cell-diameter away;
3. subpixel by fitting the **isotropic** paraboloid `z = a(x²+y²) + bx + cy + e`
   to the 5-point cross around the integer optimum and taking its vertex.
   Five points, four parameters, so the fit is near-interpolating and
   `a = ¼Σ_cross z − z₀` is a discrete Laplacian; rejected when `a ≥ 0`.

Correlation is computed on either the *rank transform* (dense rank of the pixel
values — invariant under any monotone intensity map, which is what you want when
the only thing that matters is the location of the peak) or the local-disk
normalisation. Rank is the default.

**Templates.** A single session-wide template fails once the field has drifted,
so the session is cut into `n` (odd) blocks and templates are *propagated
outward from the middle*:

```
blocks:   |   4   |   2   |    1    |   3   |   5   |
                        centre
template 1  built from the middle stacks (trimmed mean of trimmed means)
template 2  built from the corrected tail of block 1        -> used for block 2
template 3  built from the corrected head of block 1        -> used for block 3
template 4  from the corrected tail of block 2, ...
```

Each template only ever has to match stacks recorded adjacent in time to the
data it was built from. `pipeline.template_block_schedule` is this schedule as a
pure function (it is the fiddliest index arithmetic in the original; having it
separately testable was worth the extra type).

## Stage 2 — the ECC estimator (`ecc.py`)

Given a template `t` and moving image `I`, let `w(p)` be `I ∘ W_p` sampled at
the template's points, zero-meaned over the in-support subset. Maximise

  `ρ(p) = ⟨t̄, w(p)⟩ / (‖t̄‖ ‖w(p)‖)`.

Zero-meaning and normalising make `ρ` invariant to `I ↦ αI + β`, which is
exactly the nuisance calcium imaging supplies. The criterion is not quadratic,
but linearising the *warp* keeps a closed form. With `G = ∇I · ∂W/∂p ∈ R^{n×k}`,
`C = GᵀG`, and `P = GC⁻¹Gᵀ` the orthogonal projector onto `range G`:

  `Δp = C⁻¹Gᵀ(λ t̄ − w)`,  `λ = ‖(I−P)w‖² / ⟨(I−P)t̄, (I−P)w⟩`.

*Derivation.* Write `y = w + GΔp`, so `y` ranges over the affine subspace
`w + range G`. Setting `∇_{Δp} ρ = 0` gives
`Gᵀt̄ ‖y‖² = ⟨t̄,y⟩ (Gᵀw + CΔp)`, i.e. `CΔp = Gᵀ(λ t̄ − w)` with
`λ = ‖y‖²/⟨t̄,y⟩`. Substituting back and using `P² = P` collapses the fixed
point to the expression above — the numerator is `‖w‖² − (Gᵀw)ᵀC⁻¹(Gᵀw)` and the
denominator `⟨t̄,w⟩ − (Gᵀt̄)ᵀC⁻¹(Gᵀw)`, which is what the code computes. So `λ` is
the scale at which the ray through `t̄` is `G`-orthogonally closest to the current
`w`: the step is a Gauss–Newton step towards the *direction* of `t̄`, with the
magnitude free.

Three things about this implementation specifically:

* **Damped.** `Δp` is multiplied by `learning_rate`, `0.1` throughout PatchWarp.
  This matters: the design matrix has columns of order `1` (translation) and
  order `image size` (linear part), so `cond C ~ 10⁴–10⁵` and the undamped step
  overshoots. Empirically (`levels=1`, a 9-pixel translation on a 160² synthetic
  field) `lr = 0.1` and `0.25` recover the warp to 3 decimals and `lr ≥ 0.5`
  diverges. Centring the tile coordinates would fix the conditioning properly;
  the original does not, and neither does this port.
* **Stochastic.** Each iteration draws `pts_per_iter = 120` points *with
  replacement* from the pixels whose gradient exceeds the mean gradient, instead
  of using the whole tile. A minibatch Gauss–Newton step; it is what makes
  `B² × n_blocks` ECC problems affordable. Consequence: the estimator is
  randomised, so `rng` is threaded explicitly through `ecc_align`,
  `estimate_warp_field` and `register_sessions`.
* **Acceptance is not convergence.** After the last iteration the warp is
  rendered on the whole tile and `ρ` recomputed there. If the warped tile
  overlaps the template on `≤ 40%` of its area the result is discarded and
  identity returned with the *unwarped* correlation, flagged `success = False`.

`Δp` containing a NaN (singular Hessian) aborts the whole pyramid, and the
current warp is mapped straight up to full resolution.

Pyramid rescaling is one line: moving a warp between levels is conjugation by
the dilation `diag(s, s, 1)`, which multiplies the translation column by `s` and
divides the projective row by `s` — exactly the original's `next_level`.

## Stage 2 — the warp field (`affine.py`)

Input: the **per-stack downsampled** movie the rigid stage emits — one frame per
TIFF stack, i.e. one frame per few thousand raw frames. So `nz` is the number of
time blocks (10²–10³), and the estimator sees block averages with good SNR.

```
crop_motion_border   drop the rim the rigid shifts left undefined
build_template       normalised mean of a few central blocks
temporal_smooth      running mean of k blocks, then local-disk normalisation
estimate ------------ per (tile, block) ECC, warm started (below)
_accept              per-estimate plausibility tests
_median_filter       temporal median over blocks with ρ > threshold
_reject_jumps        null both sides of any ‖ΔA‖₁ > threshold step
_fill_gaps           linear interp/extrap of each matrix entry over time
```

**The continuation.** This is the idea that makes the whole thing work. The
distortion drifts slowly, so consecutive blocks have nearly equal warps, and the
ECC problem for block `k+1` can be warm-started from the solution at block `k`.
The chain runs outward from the middle of the session in both directions,
because the template is built from the middle blocks:

```
blocks:   0 ......... S/2−1 | S/2 ......... S−1
order:         <---- 3 2 1  |  1' 2' 3' ---->
seed:     median of the first (resp. last) 7 accepted warps of the previous block
```

Non-convex correlation maximisation can only track a distortion inside its
basin of attraction; the continuation keeps every individual problem small even
when the total drift over the session is not. Seeds fall back to the
predecessor's own seed when a block yields nothing usable, so the chain never
breaks.

**Robustification.** Estimates are rejected if any matrix entry exceeds
`abssum_threshold`, if the diagonal leaves `[0.6, 1.4]` (a tile cannot plausibly
scale by more than ±40%), if `ρ ≤ rho_threshold`, or if ECC reported failure.
The surviving series is median filtered in time, then any block-to-block step
with `‖ΔA‖₁` above `abssum_jump_threshold` nulls both endpoints — a real drift
is slow, a large one-block step is an estimation failure. Finally every gap is
filled by linear interpolation of each matrix entry along time (legitimate for
affine, since `Aff(2)` is a linear space; see the note on homography below), so
the field handed to the applier is always complete.

**Application** (`patches.apply_patch_warps`). Each tile is warped in its own
local coordinates and the tiles are averaged back onto the canvas. Tiles overlap
by exactly `2v+1` pixels, which is what makes a piecewise-affine reconstruction
usable at all: neighbouring maps disagree at a seam, and averaging over a band
of that width turns the jump into a ramp. The blend is unweighted — a pixel
covered by `k` tiles gets the mean of the `k` values, NaNs excluded.

## Across sessions (`across_sessions.py`)

Two passes over a pair of summary images: one global warp (3 pyramid levels)
then one warp per tile (1 level) on the globally-warped result. Composition
rather than iteration — the global pass removes the large offset that would put
a tile outside its counterpart's basin, and the tiles explain only the residual.
Each session may carry several "image types" (mean, max, correlation image);
each is aligned independently and the one with the best `ρ` wins, per tile.

## Correspondence with the MATLAB source

| MATLAB | Python |
|---|---|
| `patchwarp.m` | `pipeline.patchwarp` |
| `patchwarp_rigid.m` | `pipeline.run_rigid`, `pipeline.template_block_schedule` |
| `pyramid_registration.m` | `rigid.register_stack` |
| `@ImageRegistrator`, `@PyramidImageRegistrator`, `@BilinearImageRegistrator`, `@CorrelationCalculator`, `@ImageWithMoment` | `rigid.TranslationRegistrator`, `rigid._Correlator` |
| `max2d_subpixel.m` | `rigid._subpixel_peak` |
| `patchwarp_affine.m` | `affine.estimate_warp_field` + `pipeline.run_affine` |
| `ecc_patchwarp.m` | `ecc.ecc_align` |
| `warp_jacobian_onPts.m`, `image_jacobian_onPts.m`, `param_update.m`, `next_level.m` | `transforms.*` |
| `spatial_interp_patchwarp.m`, `applyWarpOnPts.m`, `lininterp2_fast.m` | `interp.*` |
| `applywarp_Npatches*.m` | `patches.apply_patch_warps`, `pipeline.apply_warps_to_stack` |
| `imnormalize2.m`, `rank_transform.m` | `normalize.*` |
| `make_template*.m` | `template.make_template`, `template.denoise_fft` |
| `downsample_chunk.m` | `io.downsample_mean` |
| `patchwarp_across_sessions.m` | `across_sessions.register_sessions` |
| `@ImageBasis`, `read_tiff`, `write_tiff`, `Logger`, `fastdir`, ScanImage reader | not ported / `io` stubs |

## Deliberate differences

**Structural.**

1. Warps are always `3x3` homogeneous matrices with an explicit model tag. The
   original stores `2x1`, `2x3` or `3x3` depending on `transform` and repeatedly
   patches the third row (`warp(3,3) = 1`) at the point of use; several
   downstream branches exist only to handle those shape differences.
2. `parfor` becomes ordinary loops. The parallel structure of the original is
   over tiles and over stacks, both embarrassingly so; `multiprocessing` or
   `joblib` slots in at `estimate_warp_field`'s inner loop and at the file loops
   in `pipeline`. Note that MATLAB's `parfor` + `randi` makes the original's
   results irreproducible; here the generator is threaded explicitly.
3. All I/O is behind `io.py`; the array code never touches a path. The original
   interleaves reads, writes and `.mat` round-trips with the algorithm (a warp
   field is written to disk and read back before being applied).
4. Coordinates are 0-based. This conjugates every warp by a translation, which
   fixes identity and changes nothing observable; only the raw values of the
   translation column differ by `(A[:2,:2] − I)(1,1)ᵀ`.
5. `stitch` accumulates sums and counts instead of merging seams sequentially
   and deleting duplicated strips. Equal at 4-tile corners
   (`mean(mean(a,b), mean(c,d)) = mean(a,b,c,d)`), and it removes the original's
   need to track how index bookkeeping shifts after each deletion.

**Fixes to things that look like defects in the original.**

6. `patchwarp_affine.m` applies its temporal median filter **in place**, so a
   frame's filtered value feeds the window of the next frame — a cascade, not a
   median filter. Here it is out of place.
7. The NaN-filling interpolation hardcodes a `2x3` loop and is wrapped in a
   `try/catch` that resets the entire tile to identity; for `translation` and
   `homography` it therefore *always* falls back to identity. Generalised to
   whichever entries the model owns.
8. `lininterp2_fast.m` requires `ceil(x) < size(V,2)` strictly, dropping the last
   valid row and column. Relaxed to `≤`.
9. `ecc_patchwarp.m` computes a Gaussian-smoothed `TEMP{1}`/`IM{1}` and then
   immediately overwrites both with the unsmoothed images. Not ported — the
   effective behaviour is unsmoothed.
10. `_Correlator`'s bounds test is done directly on the window, rather than
    against a `max_size` derived from an assumption that the target crop is
    centred. Identical whenever the crop is centred, which it is.

**Faithfully kept, though arguably wrong.** Flagged in the code where they live.

11. The high-gradient mask is `∂x > mean(∂x) | ∂y > mean(∂y)` — *signed*, so it
    selects pixels on one side of edges only. `|∂|` was surely meant.
12. Exceeding `affinematrix_abssum_threshold` sets `ρ := NaN` rather than
    rejecting the warp, and `NaN ≤ threshold` is false in both MATLAB and NumPy,
    so the later `ρ` test becomes vacuous for exactly those estimates: a
    blown-up warp survives unless it *also* fails the scale test.
    `WarpOptions.strict_abssum = True` does the presumably intended thing.
13. `affinematrix_medfilt_tiffstack_num = 1` is documented as "no filtering" but
    `half = round(1/2) = 1`, so it is a 3-tap median.
14. `warp_overlap_pix` is computed from the field *width* only and then used on
    both axes; on a non-square field the vertical overlap is not the requested
    fraction of the tile height.
15. `temporal_smooth`'s clamped end windows deliberately exclude the very first
    and very last blocks (`2:k` and `nz−k:nz−1` in 1-based terms) — presumably
    because those stacks are partial.
16. The `int16` cast inside the moving average is kept, since it quantises the
    ECC input and hence the estimates.

**Approximations.**

17. `fspecial('disk', r)` — MATLAB integrates the disk area per pixel
    analytically; `normalize.disk_kernel` supersamples (`8²` points/pixel).
    Differs only in the anti-aliased boundary ring.
18. `imresize` (bicubic + antialias) — `template._resize` uses `scipy.ndimage.zoom`
    order 1, and `rigid._decimate` uses an exact 2x2 block mean for the `'box'`
    case. Only used for coarse-structure correlation ranking, where it does not
    matter; swap in a proper resampler if it ever does.
19. `medfilt1(..., 'truncate')` — `scipy.ndimage.median_filter(mode='nearest')`,
    which extends rather than truncates at the border.

## Not ported

ScanImage TIFF reading, `.mat` summaries, downsampled-movie generation and
writing, the parallel pool setup, `network_temp_copy` (copying to local disk
before reading over a network share), the demo scripts, and the z-stack
utilities in `utils/`. `io.py` documents the file layout the original produces.

## Validation so far

Only synthetic smoke tests, run during the port:

* warp Jacobians agree with finite differences for all four models;
* `rescale` is an involution up to sign of the exponent and matches `next_level`;
* ECC recovers a known affine to ~3 decimals (`lr = 0.1`), and recovers
  `A⁻¹` — confirming the direction convention: `ecc_align(I ∘ A, I) ≈ A⁻¹`;
* the rigid registrator recovers integer and fractional shifts exactly within
  its search range, and needs the pyramid to do so beyond ~1 tile of
  displacement;
* `PatchGrid.stitch ∘ tile = id` exactly;
* the full `estimate_warp_field` on a synthetic session with a drifting shear
  raises the correlation of the worst block against the template from 0.65 to
  0.94;
* `register_sessions` raises 0.918 to 0.998.

What is **not** validated: numerical agreement with the MATLAB implementation on
real data. Doing so needs a session plus MATLAB; the natural comparison points
are the per-stack rigid shifts `t`, and `warp_cell` before and after each
cleaning step.
