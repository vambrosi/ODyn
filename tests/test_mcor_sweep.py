"""
`odyn.mcor_sweep`: a sweep's runs side by side, as a table and as one video,
on run folders as `test_motion_correction` leaves them.
"""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from odyn.mcor_sweep import collect, main, select, tile


def a_video(path, frames=5, width=64, height=48, value=0):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height)
    )

    for _ in range(frames):
        writer.write(np.full((height, width, 3), value, np.uint8))

    writer.release()


def a_run(folder, run, frames=5, **settings):
    run_folder = folder / run
    run_folder.mkdir(parents=True)
    pw_rigid = run != "rigid"
    metrics = {
        "crispness_test": 0.1,
        "frames_at_limit": 0.0,
        "settings": {"run": run, "pw_rigid": pw_rigid, "max_shift_um": [80, 80]}
        | settings,
    }

    if pw_rigid:
        metrics["patch_jitter_median_px"] = settings["max_deviation_um"] / 10

    (run_folder / "metrics.json").write_text(json.dumps(metrics))
    pd.DataFrame(
        {
            "position": [0, 1],
            "acq_id": [1, 2],
            "file": ["a.tif", "b.tif"],
            "test_corr_with_ref": [0.8, 0.9],
        }
    ).to_csv(run_folder / "summary.csv", index=False)
    a_video(run_folder / "Group_1_e1_test.avi", frames=frames, value=200)

    if not pw_rigid:
        a_video(run_folder / "Group_1_e1_raw.avi", frames=frames, value=50)
        a_video(run_folder / "Group_1_e1_raw_test.avi", frames=frames, width=128)


@pytest.fixture
def sweep(tmp_path):
    a_run(tmp_path, "rigid")

    for stride in (256, 128):
        for deviation in (12, 3):
            a_run(
                tmp_path,
                f"s{stride}_o{stride // 2}_d{deviation}",
                strides_um=[stride, stride],
                overlap_um=[stride // 2, stride // 2],
                max_deviation_um=deviation,
            )

    return tmp_path


def test_the_table_has_a_row_per_run_the_baseline_first(sweep):
    table = collect(sweep)

    assert table["run"].tolist() == [
        "rigid",
        "s128_o64_d3",
        "s128_o64_d12",
        "s256_o128_d3",
        "s256_o128_d12",
    ]
    assert table["test_corr_with_ref_median"].tolist() == [0.85] * 5
    assert table["patch_jitter_median_px"].iloc[1:].tolist() == [0.3, 1.2, 0.3, 1.2]
    assert (sweep / "sweep.csv").is_file()


def test_runs_with_one_setting_free_come_after_raw_and_rigid(sweep):
    table = collect(sweep)

    assert select(table, stride=128) == ["raw", "rigid", "s128_o64_d3", "s128_o64_d12"]
    assert select(table, deviation=3) == ["raw", "rigid", "s128_o64_d3", "s256_o128_d3"]

    with pytest.raises(ValueError, match="No run has stride 100"):
        select(table, stride=100)

    with pytest.raises(ValueError, match="Unknown settings size"):
        select(table, size=3)


def test_the_videos_are_tiled_in_a_grid(sweep):
    path = tile(sweep, ["raw", "rigid", "s128_o64_d3"], columns=2)

    video = cv2.VideoCapture(str(path))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    ok, frame = video.read()
    assert ok and frame.shape == (96, 128, 3)

    # raw (the baseline's raw video) top left, the tests next, black after
    assert frame[:48, :64].mean() == pytest.approx(50, abs=5)
    assert frame[:48, 64:].mean() == pytest.approx(200, abs=5)
    assert frame[48:, 64:].mean() < 5


def test_the_tiles_stop_with_the_shortest_video(sweep):
    a_run(
        sweep,
        "short",
        frames=3,
        strides_um=[64, 64],
        overlap_um=[32, 32],
        max_deviation_um=6,
    )

    path = tile(sweep, ["rigid", "short"], scale=0.5)

    video = cv2.VideoCapture(str(path))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    assert int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64


def test_the_command_line_tiles_runs_by_setting(sweep, capsys):
    assert main([str(sweep), "tile", "--where", "stride=256", "--columns", "4"]) == 0

    path = sweep / "tiles_raw_rigid_s256_o128_d3_s256_o128_d12.avi"
    assert path.is_file()
    assert str(path) in capsys.readouterr().out


def test_raw_is_the_experiments_raw_video_when_there_is_one(sweep):
    from odyn.mcor_sweep import _video

    # Folders from before raw/ have the baseline's
    assert _video(sweep, "raw").parent.name == "rigid"

    (sweep / "raw").mkdir()
    a_video(sweep / "raw" / "Group_1_e1_raw.avi", frames=4, value=50)
    assert _video(sweep, "raw") == sweep / "raw" / "Group_1_e1_raw.avi"
