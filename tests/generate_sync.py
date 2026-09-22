"""
Make a small synthetic behavior sync H5, laid out like the rig writer's.

USAGE
    from generate_sync import generate_sync
    truth = generate_sync(path, acquisitions=4, frames=64, frame_rate=14.0)

The file has the writer's root attributes and datasets at a low sample rate,
with one frame-clock burst per acquisition, one odor pulse inside each, a
camera strobe running through it all, and flat respiration and encoder
channels. `truth` says where everything is, in samples.

Options model what goes wrong on the rig: `fragments` adds short bursts,
`trailing_bursts` adds pulse-free bursts after the last acquisition (a
z-stack), `interrupted` leaves out what the writer only writes at Stop, and
`camera_until` stops the camera early.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

RATE_HZ = 1000
HIGH_V = 5.0
LOW_V = 0.0

BASELINE_S = 1.5  # burst start to valve open
ODOR_S = 1.0
INTERVAL_S = 3.0  # burst end to the next burst start
LEAD_S = 2.0  # recording start to the first burst
CAMERA_HZ = 20.0


def generate_sync(
    path: str | Path,
    *,
    acquisitions: int = 4,
    frames: int = 64,
    frame_rate: float = 14.0,
    start: datetime = datetime(2026, 1, 1, 10, 0, 0),
    mouse: str = "m001",
    experiment: str = "e1",
    fragments: int = 0,
    trailing_bursts: int = 0,
    interrupted: bool = False,
    camera_until: None | float = None,
    missing_pulse_in: tuple[int, ...] = (),
) -> dict:
    """Write the file and return where each burst and pulse landed, in samples."""
    path = Path(path)
    period = int(round(RATE_HZ / frame_rate))
    frame_high = period // 2

    bursts: list[dict] = []
    cursor = int(LEAD_S * RATE_HZ)

    def burst(count: int, with_pulse: bool) -> dict:
        nonlocal cursor
        onsets = cursor + np.arange(count) * period
        record = {
            "start": int(onsets[0]),
            "onsets": onsets.tolist(),
            "frames": count,
            "pulse": None,
        }

        if with_pulse:
            on = int(onsets[0] + BASELINE_S * RATE_HZ)
            record["pulse"] = (on, on + int(ODOR_S * RATE_HZ))

        cursor = int(onsets[-1] + period + INTERVAL_S * RATE_HZ)
        bursts.append(record)

        return record

    for index in range(acquisitions):
        burst(frames, with_pulse=index not in missing_pulse_in)

        # A fragment right after an acquisition: a stray trigger
        if index < fragments:
            burst(max(2, frames // 8), with_pulse=False)

    for _ in range(trailing_bursts):
        burst(max(2, frames // 6), with_pulse=False)

    n = cursor + int(LEAD_S * RATE_HZ)

    frame_clock = np.full(n, LOW_V, np.float32)
    odor = np.full(n, LOW_V, np.float32)
    camera = np.full(n, LOW_V, np.float32)

    for record in bursts:
        for onset in record["onsets"]:
            frame_clock[onset : onset + frame_high] = HIGH_V

        if record["pulse"] is not None:
            on, off = record["pulse"]
            odor[on:off] = HIGH_V

    camera_period = int(RATE_HZ / CAMERA_HZ)
    camera_end = n if camera_until is None else int(camera_until * RATE_HZ)

    for onset in range(camera_period, camera_end, camera_period):
        camera[onset : onset + camera_period // 2] = HIGH_V

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["mouse"] = mouse
        f.attrs["experiment"] = experiment
        f.attrs["start_time"] = start.isoformat()
        f.attrs["rate_hz"] = RATE_HZ

        if not interrupted:
            f.attrs["n_samples"] = n
            f.attrs["duration_s"] = n / RATE_HZ
            f.attrs["stop_time"] = (start + timedelta(seconds=n / RATE_HZ)).isoformat()

        encoder = f.create_dataset("encoder", data=np.zeros((n, 3), np.float32))
        encoder.attrs["column_names"] = ["EncoderA", "EncoderB", "Velocity"]
        encoder.attrs["wheel_diam_cm"] = 15.24

        respiration = f.create_dataset("respiration", data=np.zeros((n, 2), np.float32))
        respiration.attrs["resp_hp_hz"] = 0.5

        f.create_dataset("2pFrameSync", data=frame_clock)
        f.create_dataset("cameraFrameSync", data=camera)
        f.create_dataset("odorPulse", data=odor)

        if not interrupted:
            f.create_dataset("filter_changes", data=np.array([[0.0, 0.5, 15.0]]))

    return {
        "path": str(path),
        "rate_hz": RATE_HZ,
        "start": start,
        "n_samples": n,
        "period": period,
        "bursts": bursts,
    }
