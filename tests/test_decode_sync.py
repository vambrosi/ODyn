"""
Sync files: the pure decode in `odyn.sync`, then `Database.process_sync`
matching bursts to acquisitions and trials.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from odyn import Database
from odyn.database import DT_FORMAT, SyncFlag
from odyn.sync import decode, frame_onsets, pulses, read_header, split_bursts
from generate_sync import BASELINE_S, ODOR_S, RATE_HZ, generate_sync

START = datetime(2026, 1, 1, 10, 0, 0)
FRAMES = 64
FRAME_RATE = 14.0


# --------------------------------------------------------------------------- #
# The decode
# --------------------------------------------------------------------------- #


def test_edges_and_pulses(tmp_path):
    signal = np.array([0, 0, 5, 5, 0, 5, 0, 0, 5, 5], np.float32)

    assert pulses(signal).tolist() == [
        [2, 4],
        [5, 6],
    ]  # the open-ended last one is dropped
    assert pulses(np.array([5, 5, 0, 0, 5, 0], np.float32)).tolist() == [[4, 5]]

    onsets = frame_onsets(np.array([0, 5, 0, 5, 0, 0, 0, 0, 5, 0], np.float32), 1000)
    assert list(onsets) == [1, 8]  # 3 and 1 are 2 ms apart: a bounce

    assert [list(b) for b in split_bursts(np.array([0, 10, 20, 5000, 5010]), 1000)] == [
        [0, 10, 20],
        [5000, 5010],
    ]


def test_a_burst_decodes_to_frames_and_residuals(tmp_path):
    truth = generate_sync(tmp_path / "sync.h5", acquisitions=2, frames=FRAMES)
    decoded = decode(tmp_path / "sync.h5")

    assert not decoded.header.interrupted
    assert decoded.header.rate_hz == RATE_HZ
    assert decoded.header.start_time == START
    assert decoded.header.n_samples == truth["n_samples"]
    assert decoded.header.attributes["file"]["mouse"] == "m001"
    assert decoded.header.attributes["encoder"]["wheel_diam_cm"] == 15.24

    assert len(decoded.blocks) == 2 and not decoded.fragments

    block, record = decoded.blocks[0], truth["bursts"][0]
    period = truth["period"]

    assert block.start_sample == record["start"]
    assert block.frame_count == FRAMES
    assert block.frame_period_s == pytest.approx(period / RATE_HZ)
    assert block.odor_pulses == 1
    assert block.camera_frames > 0

    # The valve opened during frame 21 (1.5 s at 14 Hz), a bit after it began
    on, off = record["pulse"]
    assert block.odor_on_frame == (on - record["start"]) // period
    assert block.odor_on_residual_s == pytest.approx(
        ((on - record["start"]) % period) / RATE_HZ
    )
    assert 0 <= block.odor_on_residual_s < period / RATE_HZ

    # ...and the first frame with it closed comes ODOR_S later
    assert block.odor_off_frame == int(np.ceil((off - record["start"]) / period))
    assert decoded.time_at(block.odor_on_sample) == START + timedelta(
        seconds=on / RATE_HZ
    )


def test_short_bursts_are_fragments(tmp_path):
    generate_sync(tmp_path / "sync.h5", acquisitions=3, fragments=2, trailing_bursts=1)
    decoded = decode(tmp_path / "sync.h5")

    assert [b.frame_count for b in decoded.blocks] == [FRAMES] * 3
    assert [b.sync_block for b in decoded.blocks] == [0, 2, 4]
    assert len(decoded.fragments) == 3
    assert all(f.odor_pulses == 0 for f in decoded.fragments)


def test_an_interrupted_file_still_reads(tmp_path):
    truth = generate_sync(tmp_path / "sync.h5", acquisitions=2, interrupted=True)
    header = read_header(tmp_path / "sync.h5")

    assert header.interrupted
    assert header.stop_time is None
    assert header.n_samples == truth["n_samples"]  # from the data, not the attrs
    assert len(decode(tmp_path / "sync.h5").blocks) == 2


# --------------------------------------------------------------------------- #
# Matching into the database
# --------------------------------------------------------------------------- #


def build(
    tmp_path,
    truth,
    *,
    acquisitions=None,
    trials=None,
    frames_per_slice=FRAMES,
    exp_type="loop",
):
    """
    A database with one experiment whose acquisitions and trials line up with
    `truth`, as ScanImage and the olfactometer would have logged them (a grab
    has no trials).
    """
    db = Database(tmp_path, can_create=True, user="tester")
    db.set_mouse(mouse_id=1)
    db.add_session(mouse_id=1, session_date="2026-01-01")

    starts = [
        START + timedelta(seconds=b["start"] / RATE_HZ)
        for b in truth["bursts"]
        if b["frames"] == FRAMES
    ]
    pulses_at = [
        START + timedelta(seconds=b["pulse"][0] / RATE_HZ)
        for b in truth["bursts"]
        if b["pulse"] is not None
    ]

    if acquisitions is not None:
        starts = starts[:acquisitions]
    if trials is not None:
        pulses_at = pulses_at[:trials]

    with db._writing() as cur:
        cur.execute(
            """
            INSERT INTO experiments
                ( exp_id, session_id, exp_number, exp_path, exp_start
                , height_px, width_px, height_um, width_um, frames_per_slice
                , frame_rate, objective, exp_type
                ) VALUES (1, 1, 1, '20260101/m001/e1', ?
                         , 1, 1, 1, 1, ?, ?, 20, ?);
            """,
            [START.strftime(DT_FORMAT), frames_per_slice, FRAME_RATE, exp_type],
        )

        for index, at in enumerate(starts):
            # ScanImage's clock runs 100 ms ahead of the olfactometer's
            cur.execute(
                "INSERT INTO acquisitions (acq_id, exp_id, acq_start, raw_path) "
                "VALUES (?, 1, ?, ?);",
                [
                    index + 1,
                    (at + timedelta(milliseconds=100)).strftime(DT_FORMAT),
                    f"raw/{index}.tif",
                ],
            )

        if exp_type == "grab":
            return db

        cur.execute(
            "INSERT INTO programs (program_id, exp_id, program_name, program_type, "
            "program_start, program_path) VALUES (1, 1, 'p', 'passive', ?, 'events.csv');",
            [START.strftime(DT_FORMAT)],
        )

        for index, at in enumerate(pulses_at):
            cur.execute(
                """
                INSERT INTO trials
                    ( trial_id, trial_start, trial_odor_start, trial_odor_end
                    , odor_id, outcome, program_id, exp_id
                    ) VALUES (?, ?, ?, ?, 1, 'na', 1, 1);
                """,
                [
                    index + 1,
                    (at - timedelta(seconds=BASELINE_S)).strftime(DT_FORMAT),
                    at.strftime(DT_FORMAT),
                    (at + timedelta(seconds=ODOR_S)).strftime(DT_FORMAT),
                ],
            )

    return db


def test_bursts_are_matched_to_acquisitions_and_trials(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(sync, acquisitions=4)
    db = build(tmp_path, truth)

    db.process_sync(exp_id=1)

    stream = db.streams.iloc[0]
    assert stream["file_path"] == "20260101/m001/e1/sync/m001_e1.h5"
    assert stream["rate_hz"] == RATE_HZ
    assert stream["start_time"] == START
    assert stream["attributes"]["file"]["experiment"] == "e1"

    blocks = db.sync_blocks
    assert list(blocks.index) == [1, 2, 3, 4]
    assert list(blocks["sync_block"]) == [0, 1, 2, 3]
    assert (blocks["sync_frame_count"] == FRAMES).all()
    assert (blocks["sync_odor_pulses"] == 1).all()
    assert blocks["clock_offset_ms"].round().eq(100).all()

    # The join lands the odor frame on the acquisition
    acquisitions = db.acquisitions
    assert acquisitions["sync_odor_on_frame"].notna().all()
    assert acquisitions.loc[1, "sync_odor_start"] == START + timedelta(
        seconds=truth["bursts"][0]["pulse"][0] / RATE_HZ
    )

    assert list(db.trials["acq_id"]) == [1, 2, 3, 4]
    assert db.acquisition_trials["trial_id"].notna().all()

    call = db.latest_calls("process_sync").iloc[0]
    assert call["call_flag"] == 0
    assert call["output_blocks"] == 4 and call["output_matched_trials"] == 4


def test_a_rerun_replaces_what_the_first_wrote(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(sync, acquisitions=3)
    db = build(tmp_path, truth)

    db.process_sync(exp_id=1)
    db.process_sync(exp_id=1)

    assert len(db.streams) == 1
    assert len(db.sync_blocks) == 3
    assert list(db.trials["acq_id"]) == [1, 2, 3]

    (stream_id,) = db.streams.index
    assert (db.sync_blocks["stream_id"] == stream_id).all()


def test_the_file_is_found_or_named(tmp_path):
    truth = generate_sync(tmp_path / "elsewhere/rec.h5", acquisitions=2)
    db = build(tmp_path, truth)

    with pytest.raises(RuntimeError, match="No sync file in"):
        db.process_sync(exp_id=1)

    assert db.latest_calls("process_sync").iloc[0]["call_flag"] & SyncFlag.NO_SYNC_FILE

    db.process_sync(exp_id=1, sync_path="elsewhere/rec.h5")
    assert len(db.sync_blocks) == 2

    generate_sync(tmp_path / "20260101/m001/e1/sync/a.h5", acquisitions=2)
    generate_sync(tmp_path / "20260101/m001/e1/sync/b.h5", acquisitions=2)

    with pytest.raises(RuntimeError, match="2 H5 files"):
        db.process_sync(exp_id=1)


def test_mismatched_counts_are_refused_before_writing(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(sync, acquisitions=4)

    db = build(tmp_path, truth, acquisitions=3)
    with pytest.raises(RuntimeError, match="4 frame bursts.*3 acquisitions"):
        db.process_sync(exp_id=1)
    assert len(db.streams) == 0
    assert (
        db.latest_calls("process_sync").iloc[0]["call_flag"]
        & SyncFlag.BLOCK_COUNT_MISMATCH
    )

    db.path.unlink()
    db = build(tmp_path, truth, trials=3)
    with pytest.raises(RuntimeError, match="4 bursts have an odor pulse.*3 trials"):
        db.process_sync(exp_id=1)
    assert (
        db.latest_calls("process_sync").iloc[0]["call_flag"]
        & SyncFlag.TRIAL_COUNT_MISMATCH
    )


def test_what_is_off_is_flagged_but_stored(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(
        sync,
        acquisitions=4,
        fragments=1,
        trailing_bursts=2,
        interrupted=True,
        camera_until=8.0,
        missing_pulse_in=(3,),
    )
    db = build(tmp_path, truth, frames_per_slice=FRAMES + 1)

    db.process_sync(exp_id=1)

    flag = db.latest_calls("process_sync").iloc[0]["call_flag"]
    for expected in (
        SyncFlag.INTERRUPTED,
        SyncFlag.FRAGMENTS,
        SyncFlag.ODOR_PULSES,
        SyncFlag.NO_CAMERA,
        SyncFlag.FRAME_COUNT,
    ):
        assert flag & expected, expected

    assert not flag & SyncFlag.CLOCK_SKEW

    blocks = db.sync_blocks
    assert len(blocks) == 4
    assert blocks.loc[4, "sync_odor_pulses"] == 0
    assert (
        blocks.loc[4, "sync_odor_on_frame"] != blocks.loc[4, "sync_odor_on_frame"]
    )  # NaN
    assert blocks.loc[1, "sync_camera_frames"] > 0
    assert blocks.loc[4, "sync_camera_frames"] == 0
    assert pd.isna(db.streams.iloc[0]["stop_time"])

    # Three trials for the three bursts with a pulse
    assert list(db.trials["acq_id"]) == [1, 2, 3]


def test_clock_skew_is_flagged(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(sync, acquisitions=2, start=START + timedelta(minutes=1))
    db = build(tmp_path, truth)

    db.process_sync(exp_id=1)

    assert db.latest_calls("process_sync").iloc[0]["call_flag"] & SyncFlag.CLOCK_SKEW
    assert db.sync_blocks["clock_offset_ms"].round().eq(-59900).all()


# --------------------------------------------------------------------------- #
# Grabs
# --------------------------------------------------------------------------- #


def test_a_grabs_burst_is_its_acquisition(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(sync, acquisitions=1, missing_pulse_in=(0,))
    db = build(tmp_path, truth, exp_type="grab")

    db.process_sync(exp_id=1)

    assert db.streams.iloc[0]["exp_id"] == 1
    assert db.acquisitions.loc[1, "sync_block"] == 0
    assert db.acquisitions.loc[1, "sync_frame_count"] == FRAMES

    # No trials to match, and none expected
    call = db.latest_calls("process_sync").iloc[0]
    assert call["call_flag"] == 0
    assert call["output_matched_trials"] == 0


def test_an_odor_pulse_during_a_grab_is_flagged(tmp_path):
    sync = tmp_path / "20260101/m001/e1/sync/m001_e1.h5"
    truth = generate_sync(sync, acquisitions=1)  # a valve pulse in it
    db = build(tmp_path, truth, exp_type="grab")

    db.process_sync(exp_id=1)

    flag = db.latest_calls("process_sync").iloc[0]["call_flag"]
    assert flag & SyncFlag.ODOR_PULSES
    assert not flag & SyncFlag.NO_TRIALS
