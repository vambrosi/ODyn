#!/usr/bin/env python3
"""
Make a small synthetic experiment to test the pipeline.

USAGE
    python generate_data.py                      # into ../tmp/generated_data
    python generate_data.py --out /some/folder --acquisitions 8 --seed 7
    python generate_data.py --h5-flat            # a defective add to repair

The result is a `main_folder`, so it can be used directly:

    from odyn import Database
    db = Database("tmp/generated_data", update=True)

OUTPUTS
    - Raw acquisition files that can be read be `tifffile`;
    - An olfactometer H5 with the imaging and odor TTLs (no Events.csv yet);
    - `ground_truth.json` to compare against function outputs.

BROKEN RECORDINGS
    The H5 switches make the states `add_experiment` has to survive, so the
    flags and the repair have something to run against:

    --no-h5             no H5 at all
    --h5-flat           an H5 whose triggers never rise (a session stopped
                        immediately); well formed, but all timing is missing
    --h5-high-start     recording began inside the first imaging window, so
                        that window's rise is not in the signal
    --skip-first-tiff   the first acquisition was never saved

WHAT IS MODELLED
    glomeruli       flat-topped discs with soft edges, random size and place
    response        some of them brighten during the odor, fast rise slow decay
    photobleaching  everything dims slowly over the acquisition
    motion          each vertex of a grid drifts, pulses and jitters on its own,
                    and the frame is warped by the affine of each triangle
    noise           Poisson, applied last

TO DO LIST
    - Adjust default parameters to match real data
    - Check motion steps
"""

from __future__ import annotations

import argparse
import io
import json
import struct

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import tifffile

from skimage.transform import PiecewiseAffineTransform, warp

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

OUTPUT_FOLDER = Path(__file__).resolve().parents[1] / "tmp" / "generated_data"

# Experiment metadata
EXP_DATE = "20260101"
MOUSE = "m001"
EXP = "e1"
EXP_START = datetime(2026, 1, 1, 10, 0, 0)

# Acquisition parameters
ACQUISITIONS = 4
FRAMES = 64
HEIGHT, WIDTH = 256, 320
UM_PER_PX = 4.0
FRAME_RATE = 14.0
LOOP_INTERVAL_S = 10.0

# Glomeruli shape/activation parameters
GLOMERULI = 22
RADIUS_UM = (28.0, 60.0)
EDGE_UM = 12.0

RESPONDING = 0.4
RESPONSE_DFF = (0.08, 0.30)
BRIGHTNESS_RANGE = (300.0, 900.0)

BACKGROUND_LEVEL = 40.0

# Response parameters
ODOR_START_S = 1.5
ODOR_DURATION_S = 1.0
RISE_S = 0.25
DECAY_S = 1.2
PHOTOBLEACH_FRACTION = 0.12

# Olfactometer H5 parameters
H5_SAMPLERATE = 1000.0  # Hz
H5_TTL_HIGH = 5.0  # volts, well over the rise `_get_h5_metadata` looks for
H5_LEAD_S = 2.0  # recorded before the first imaging window opens
H5_TAIL_S = 2.0  # recorded after the last one closes

# Motion is split in two to simulate real data:
#   - There a rigid jitter/drift motion that the whole field shares;
#   - There is warping that approximate the independent motion between regions.
# Those parameters are set to be within the pipeline's defaults, with the
# warping sitting close to DEFAULT_MAX_DEVIATION_UM to simulate the hard case.

MESH = (5, 6)
PULSE_HZ = 2.5

DRIFT_UM = 32.0  # slow wander of the whole field
PULSE_UM = 8.0  # breathing, whole field
WARP_UM = 8.0  # how far regions disagree with each other
JITTER_UM = 1.5  # frame to frame noise in the position


# --------------------------------------------------------------------------- #
# ScanImage TIFF
# --------------------------------------------------------------------------- #

SI_MAGIC = 117637889

# Bytes per TIFF tag type, so we can tell an inline value from an offset
TYPE_SIZE = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
    16: 8,
    17: 8,
    18: 8,
}


def _si_block(frame_data: str, roi_json: str = "{}") -> bytes:
    """The ScanImage header that `tifffile.scanimage_metadata` looks for."""
    text = frame_data.encode() + b"\0"
    roi = roi_json.encode()

    return struct.pack("<IIII", SI_MAGIC, 3, len(text), len(roi)) + text + roi


def _shift_offsets(data: bytearray, delta: int) -> bytearray:
    """
    Add `delta` to every file offset in a little-endian BigTIFF.

    ScanImage puts its metadata at byte 16, before the image directories, and
    tifffile will not write it there. So the file is written normally, the block
    is spliced in, and everything that points into the file moves along with it.
    """
    read = lambda at: struct.unpack_from("<Q", data, at)[0]
    write = lambda at, value: struct.pack_into("<Q", data, at, value)

    directory = read(8)
    write(8, directory + delta)

    while directory:
        count = read(directory)

        for i in range(count):
            tag_at = directory + 8 + i * 20
            tag, kind, values = struct.unpack_from("<HHQ", data, tag_at)
            inline = values * TYPE_SIZE.get(kind, 1) <= 8

            # These tags hold offsets, so their values move too
            if tag in (273, 324):
                base = tag_at + 12 if inline else read(tag_at + 12)
                for v in range(values):
                    write(base + v * 8, read(base + v * 8) + delta)

            if not inline:
                write(tag_at + 12, read(tag_at + 12) + delta)

        next_at = directory + 8 + count * 20
        directory = read(next_at)

        if directory:
            write(next_at, directory + delta)

    return data


def _write_tiff(
    path: Path, frames: np.ndarray, frame_data: str, description: str, um_per_px: float
) -> None:
    """Write `frames` as a ScanImage BigTIFF."""
    buffer = io.BytesIO()

    # One write call, not one per frame: separate calls make separate series and
    # caiman.load only reads the first one.
    with tifffile.TiffWriter(buffer, bigtiff=True) as writer:
        writer.write(
            frames,
            photometric="minisblack",
            software="SI.VERSION_MAJOR = 2023",
            description=description,
            resolution=(1e4 / um_per_px, 1e4 / um_per_px),
            resolutionunit=3,
        )

    block = _si_block(frame_data)
    data = _shift_offsets(bytearray(buffer.getvalue()), len(block))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(data[:16]) + block + bytes(data[16:]))


# --------------------------------------------------------------------------- #
# Olfactometer H5
# --------------------------------------------------------------------------- #


def _write_h5(
    path: Path,
    *,
    acquisitions: int,
    frames: int,
    frame_rate: float,
    flat: bool = False,
    high_start: bool = False,
) -> dict:
    """
    Write the olfactometer H5 the way `_get_h5_metadata` reads it back.

    Two square TTLs sampled at `H5_SAMPLERATE`: `ImagingWindow` is high for the
    length of each acquisition and `OdorDelivery` for the odor inside it. Window
    `i` opens `LOOP_INTERVAL_S` after window `i - 1`, which is the spacing
    `_match_acq_to_h5` compares against `frameTimestamps_sec`.

    `flat` writes both channels as zeros: a session stopped immediately gives a
    well formed file in which no trigger ever rises.

    `high_start` cuts the recording short at the front, between the first
    window's rise and its odor. The rise is then not in the signal at all, so
    no peak finder can recover it, and the odor pulse it belongs to is left
    with no trial start in front of it.
    """
    window_s = frames / frame_rate
    span_s = (acquisitions - 1) * LOOP_INTERVAL_S + window_s
    samples = int((H5_LEAD_S + span_s + H5_TAIL_S) * H5_SAMPLERATE)

    imaging = np.zeros(samples, np.float32)
    odor = np.zeros(samples, np.float32)

    def high(signal, start_s, duration_s):
        first = int(start_s * H5_SAMPLERATE)
        signal[first : first + int(duration_s * H5_SAMPLERATE)] = H5_TTL_HIGH

    if not flat:
        for index in range(acquisitions):
            opened = H5_LEAD_S + index * LOOP_INTERVAL_S

            high(imaging, opened, window_s)
            high(odor, opened + ODOR_START_S, ODOR_DURATION_S)

    cut = int((H5_LEAD_S + ODOR_START_S / 2) * H5_SAMPLERATE) if high_start else 0

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["samplerate"] = H5_SAMPLERATE
        f.create_dataset("ImagingWindow", data=imaging[cut:])
        f.create_dataset("OdorDelivery", data=odor[cut:])

    return {
        "path": path.name,
        "samplerate": H5_SAMPLERATE,
        "window_s": window_s,
        "lead_s": H5_LEAD_S - cut / H5_SAMPLERATE,
        "flat": flat,
        "high_start": high_start,
        "trials": 0 if flat else acquisitions - bool(high_start),
    }


# --------------------------------------------------------------------------- #
# The scene
# --------------------------------------------------------------------------- #


def _glomeruli(rng, height, width, um_per_px):
    """Random discs with soft edges, and which of them respond to the odor."""
    cells = []

    for _ in range(GLOMERULI):
        radius = rng.uniform(*RADIUS_UM) / um_per_px
        cells.append(
            {
                "y": float(rng.uniform(radius, height - radius)),
                "x": float(rng.uniform(radius, width - radius)),
                "radius_px": float(radius),
                "brightness": float(rng.uniform(*BRIGHTNESS_RANGE)),
                "responds": bool(rng.random() < RESPONDING),
                "response_dff": float(rng.uniform(*RESPONSE_DFF)),
            }
        )

    return cells


def _disc(height, width, cell, edge_px):
    """One glomerulus: flat in the middle, soft at the rim."""
    ys, xs = np.ogrid[:height, :width]
    distance = np.sqrt((ys - cell["y"]) ** 2 + (xs - cell["x"]) ** 2)

    return 1.0 / (1.0 + np.exp((distance - cell["radius_px"]) / edge_px))


def _response(frames, frame_rate):
    """Approximates calcium response: quick rise during the odor, slow decay after."""
    times = np.arange(frames) / frame_rate
    end = ODOR_START_S + ODOR_DURATION_S

    rising = np.clip(1.0 - np.exp(-(times - ODOR_START_S) / RISE_S), 0.0, None)
    peak = 1.0 - np.exp(-ODOR_DURATION_S / RISE_S)
    falling = peak * np.exp(-(times - end) / DECAY_S)

    course = np.where(times < end, rising, falling)

    return np.clip(course, 0.0, None)


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #


def _mesh(height, width):
    """Vertices of the grid whose triangles get their own affine."""
    rows, cols = MESH
    ys = np.linspace(0, height, rows + 1)
    xs = np.linspace(0, width, cols + 1)
    grid = np.meshgrid(xs, ys)

    return np.column_stack([grid[0].ravel(), grid[1].ravel()])


def _displacements(rng, vertices, frames, frame_rate, um_per_px):
    """
    How far each vertex moves on each frame, in pixels, as `(frames, n, 2)`.

    The whole field drifts and breathes together, and on top of that each vertex
    breathes with its own phase. Recording a whole olfactory bulb means separate
    regions pulse independently, so the field warps instead of just shifting --
    which is the part rigid correction cannot undo.
    """
    n = len(vertices)
    times = np.arange(frames) / frame_rate
    wave = lambda hz, phase: np.sin(2 * np.pi * hz * times[:, None, None] + phase)

    # --- Whole field: same displacement everywhere --- #

    speed = rng.uniform(0.05, 0.2, (1, 2))
    drift = (DRIFT_UM / um_per_px) * wave(speed, rng.uniform(0, 2 * np.pi, (1, 2)))
    pulse = (PULSE_UM / um_per_px) * wave(PULSE_HZ, rng.uniform(0, 2 * np.pi, (1, 2)))

    # --- Per region: same rhythm, its own phase, so regions disagree --- #

    warp_phase = rng.uniform(0, 2 * np.pi, (n, 2))
    regional = (WARP_UM / um_per_px) * wave(PULSE_HZ, warp_phase)

    jitter = (JITTER_UM / um_per_px) * rng.standard_normal((frames, n, 2))

    return drift + pulse + regional + jitter


def _warp(frame, vertices, offset):
    """Move the mesh by `offset` and warp the frame with it."""
    transform = PiecewiseAffineTransform()
    transform.estimate(vertices + offset, vertices)

    return warp(frame, transform, order=1, mode="edge", preserve_range=True)


# --------------------------------------------------------------------------- #
# Generating
# --------------------------------------------------------------------------- #


def generate(
    output_folder: Path = OUTPUT_FOLDER,
    *,
    acquisitions: int = ACQUISITIONS,
    frames: int = FRAMES,
    height: int = HEIGHT,
    width: int = WIDTH,
    um_per_px: float = UM_PER_PX,
    frame_rate: float = FRAME_RATE,
    motion: float = 1.0,
    seed: int = 0,
    h5: bool = True,
    h5_flat: bool = False,
    h5_high_start: bool = False,
    skip_first_tiff: bool = False,
) -> dict:
    """
    Write the recording and return what went into it.

    `out` becomes a `main_folder`, holding one experiment at
    `<date>/<mouse>/<exp>/raw/`.

    `motion` scales every displacement. Pass `0` for a still recording, which is
    lets you see the response by itself. (Can be used as mcor comparison.)

    The `h5*` and `skip_first_tiff` arguments make a broken recording instead of
    a good one; see the module docstring.
    """
    output_folder = Path(output_folder)
    rng = np.random.default_rng(seed)

    # Minimum amount of frames to show the full response
    needed = (ODOR_START_S + ODOR_DURATION_S + DECAY_S) * frame_rate

    if frames < needed:
        raise ValueError(
            f"Number of frames must be at least {int(needed) + 1} "
            f"to show a full response. That includes: \n"
            f"  - Odor start:     "
            f"  {ODOR_START_S} s = {ODOR_START_S * frame_rate:.2f} frames\n"
            f"  - Odor duration:  "
            f"  {ODOR_DURATION_S} s = {ODOR_DURATION_S * frame_rate:.2f} frames\n"
            f"  - Response decay: "
            f"  {DECAY_S} s = {DECAY_S * frame_rate:.2f} frames"
        )

    exp_name = f"{EXP_DATE}_{MOUSE}_{EXP}"
    exp_folder = output_folder / EXP_DATE / MOUSE / EXP
    raw_folder = exp_folder / "raw"

    cells = _glomeruli(rng, height, width, um_per_px)
    edge_px = EDGE_UM / um_per_px

    # The field of view, unmoving and noiseless
    resting = np.full((height, width), BACKGROUND_LEVEL, np.float32)
    responsive = np.zeros((height, width), np.float32)

    for cell in cells:
        disc = _disc(height, width, cell, edge_px).astype(np.float32)
        resting += cell["brightness"] * disc

        if cell["responds"]:
            responsive += cell["brightness"] * cell["response_dff"] * disc

    course = _response(frames, frame_rate)
    photobleach = np.exp(np.linspace(0, np.log(1 - PHOTOBLEACH_FRACTION), frames))
    vertices = _mesh(height, width)

    truth = {
        "exp_name": exp_name,
        "main_folder": str(output_folder),
        "height_px": height,
        "width_px": width,
        "um_per_px": um_per_px,
        "frame_rate": frame_rate,
        "frames": frames,
        "odor_start_s": ODOR_START_S,
        "odor_duration_s": ODOR_DURATION_S,
        "background": BACKGROUND_LEVEL,
        "photobleach_fraction": PHOTOBLEACH_FRACTION,
        "motion": motion,
        "seed": seed,
        "glomeruli": cells,
        "h5": None,
        "acquisitions": [],
    }

    if h5:
        truth["h5"] = _write_h5(
            exp_folder / f"{exp_name}.h5",
            acquisitions=acquisitions,
            frames=frames,
            frame_rate=frame_rate,
            flat=h5_flat,
            high_start=h5_high_start,
        )

    for index in range(acquisitions):
        # An acquisition that was never saved: the loop still counts it, so the
        # files that follow keep the frame timestamps they already had.
        if skip_first_tiff and index == 0:
            continue

        offsets = motion * _displacements(rng, vertices, frames, frame_rate, um_per_px)
        stack = np.empty((frames, height, width), np.float32)

        for t in range(frames):
            clean = (resting + course[t] * responsive) * photobleach[t]
            stack[t] = _warp(clean, vertices, offsets[t]) if motion else clean

        # Noise goes on after the movement
        noisy = rng.poisson(np.clip(stack, 0, None)).astype(np.int16)

        # `epoch` is the start of the loop and must be identical in every
        # raw file for that experiment, or add_experiment will reject it.
        # The acquisition's own start is epoch + frameTimestamps_sec.
        started = index * LOOP_INTERVAL_S
        name = f"{exp_name}_{index + 1:05d}.tif"

        _write_tiff(
            raw_folder / name,
            noisy,
            frame_data="\n".join(
                [
                    "SI.acqState = 'loop'",
                    "SI.hBeams.powers = [10 0]",
                    f"SI.hStackManager.framesPerSlice = {frames}",
                    f"SI.hRoiManager.scanFrameRate = {frame_rate}",
                    f"SI.loopAcqInterval = {LOOP_INTERVAL_S}",
                ]
            ),
            description="\n".join(
                [
                    "frameNumbers = 1",
                    f"frameTimestamps_sec = {started:.6f}",
                    f"epoch = [{EXP_START:%Y %m %d %H %M} {EXP_START.second:.3f}]",
                ]
            ),
            um_per_px=um_per_px,
        )

        truth["acquisitions"].append(
            {
                "raw_path": f"{EXP_DATE}/{MOUSE}/{EXP}/raw/{name}",
                "acq_start": (EXP_START + timedelta(seconds=started)).isoformat(),
                "mean_shift_px": np.abs(offsets).mean(axis=(1, 2)).round(3).tolist(),
            }
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT_FOLDER, type=Path)
    parser.add_argument("--acquisitions", default=ACQUISITIONS, type=int)
    parser.add_argument("--frames", default=FRAMES, type=int)
    parser.add_argument("--height", default=HEIGHT, type=int)
    parser.add_argument("--width", default=WIDTH, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--motion", default=1.0, type=float, help="scale displacements, 0 for still"
    )
    parser.add_argument(
        "--no-h5", dest="h5", action="store_false", help="write no H5 at all"
    )
    parser.add_argument(
        "--h5-flat", action="store_true", help="an H5 whose triggers never rise"
    )
    parser.add_argument(
        "--h5-high-start",
        action="store_true",
        help="start recording inside the first imaging window",
    )
    parser.add_argument(
        "--skip-first-tiff",
        action="store_true",
        help="the first acquisition was never saved",
    )

    args = parser.parse_args()
    truth = generate(
        args.out,
        acquisitions=args.acquisitions,
        frames=args.frames,
        height=args.height,
        width=args.width,
        motion=args.motion,
        seed=args.seed,
        h5=args.h5,
        h5_flat=args.h5_flat,
        h5_high_start=args.h5_high_start,
        skip_first_tiff=args.skip_first_tiff,
    )

    responding = sum(cell["responds"] for cell in truth["glomeruli"])
    size = sum(f.stat().st_size for f in Path(args.out).rglob("*.tif"))

    print(f"Saved {len(truth['acquisitions'])} acquisitions to {args.out}")
    print(f"    Frame Size       {args.height}x{args.width} (height x width)")
    print(f"    Frame Count      {args.frames}")
    print(f"    File Size        {size / 1e6:.1f} MB")
    print(f"    Glomeruli Count  {len(truth['glomeruli'])} ({responding} responding)")

    if truth["h5"] is None:
        print("    Olfactometer H5  none")
    else:
        print(f"    Olfactometer H5  {truth['h5']['trials']} trial triggers")

    print(f"Ground truth in {Path(args.out) / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
