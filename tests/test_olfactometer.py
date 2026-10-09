"""
The olfactometer's files copied into a loop folder at Submit, and found there
by ingestion: a stand-in for the olfactometer PC's three folders.
"""

import json

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from odyn.app.draft import Draft
from odyn.app.olfactometer import (
    DEFAULT_FOLDERS,
    OlfactometerFolders,
    copy_for_loop,
    recording_window,
    run_start,
)
from odyn.app.submit import copy_olfactometer
from generate_sync import generate_sync

DAY = datetime(2026, 8, 11)
RUN = "16odors passive 4s scope"


def olfactometer_pc(root: Path) -> OlfactometerFolders:
    """Its config, its log of the day, and three runs (13:00, 13:58, 16:30)."""
    folders = OlfactometerFolders(
        config=root / "Documents" / "Olfactometer",
        log=root / "RWD" / "Olfactometer" / "Log",
        data=root / "RWD" / "OM_Data",
    )
    folders.config.mkdir(parents=True)
    (folders.config / "Program.config").write_text("[]")
    (folders.config / "SystemConfig.config").write_text("{}")

    log = folders.program_log(DAY)
    log.parent.mkdir(parents=True)
    log.write_text("[13:00:00.100] Start program: 16odors passive 4s scope\n")

    for stamp in ("2026_08_11-13_00_00", "2026_08_11-13_58_59", "2026_08_11-16_30_00"):
        run = folders.data / stamp / f"{RUN}-Bilateral-CH1"
        run.mkdir(parents=True)
        (run / f"{RUN}-{stamp}-Events.csv").write_text("Events\n")

    return folders


def test_the_settings_name_the_folders_or_fall_back_to_the_pc(tmp_path):
    assert OlfactometerFolders.read(tmp_path).data == Path(
        DEFAULT_FOLDERS["olfactometer_data"]
    )

    (tmp_path / "settings.json").write_text(json.dumps({"olfactometer_data": "/x"}))
    folders = OlfactometerFolders.read(tmp_path)

    assert folders.data == Path("/x")
    assert folders.log == Path(DEFAULT_FOLDERS["olfactometer_log"])


def test_runs_are_picked_by_when_they_started(tmp_path):
    folders = olfactometer_pc(tmp_path)

    assert run_start("2026_08_11-13_58_59") == datetime(2026, 8, 11, 13, 58, 59)
    assert run_start("notes") is None

    # Started a minute before the recording still counts; after it, not
    runs = folders.runs_during(datetime(2026, 8, 11, 13, 1), datetime(2026, 8, 11, 15))
    assert [r.name for r in runs] == ["2026_08_11-13_00_00", "2026_08_11-13_58_59"]


def test_the_window_comes_from_the_sync_file_when_it_is_here(tmp_path):
    local = tmp_path / "m1_e1.h5"
    generate_sync(local, acquisitions=2)
    start, stop = recording_window({"local": str(local), "started": "2030-01-01"})

    assert start.year == 2026 and stop > start  # the file's clock, not the note's

    noted = {
        "local": "gone.h5",
        "started": "2026-08-11 13:00:00",
        "stopped": "2026-08-11 15:00:00",
    }
    assert recording_window(noted) == (DAY.replace(hour=13), DAY.replace(hour=15))
    assert recording_window({"local": "gone.h5", "started": "2026-08-11"}) is None


def test_a_loop_gets_its_runs_the_log_and_the_config(tmp_path):
    folders = olfactometer_pc(tmp_path / "pc")
    loop = tmp_path / "share" / "e1"
    window = (datetime(2026, 8, 11, 12, 59), datetime(2026, 8, 11, 15))

    copied = copy_for_loop(folders, loop, window)

    assert copied.runs == ["2026_08_11-13_00_00", "2026_08_11-13_58_59"]
    assert copied.log == "Program_20260811.txt"
    assert copied.config == ["Program.config", "SystemConfig.config"]
    target = loop / "olfactometer"
    assert len(list((target / "data").rglob("*Events.csv"))) == 2
    assert (target / "log" / "Program_20260811.txt").is_file()

    # Again (a second Submit): the log is refreshed, the rest left as it is
    folders.program_log(DAY).write_text("longer now\n")
    (folders.config / "Program.config").write_text("[changed]")
    again = copy_for_loop(folders, loop, window)

    assert again.runs == [] and again.config == []
    assert (target / "log" / "Program_20260811.txt").read_text() == "longer now\n"
    assert (target / "config" / "Program.config").read_text() == "[]"


def test_submit_copies_for_each_loop_with_a_recording(tmp_path):
    folders = olfactometer_pc(tmp_path / "pc")
    main = tmp_path / "share"
    draft = Draft.start(tmp_path / "drafts")
    draft.identify(mouse_id=1, date="2026-08-11")
    draft.set_part("e1", objective=20)
    draft.set_part("e2", objective=20)  # no recording: nothing to go by
    draft.add_recording(
        "e1",
        {
            "file": "m1_e1.h5",
            "local": str(tmp_path / "gone.h5"),
            "share": "x",
            "started": "2026-08-11 13:50:00",
            "stopped": "2026-08-11 15:00:00",
        },
    )

    for name in ("e1", "e2"):
        (main / draft.folder / name).mkdir(parents=True)

    assert copy_olfactometer(SimpleNamespace(main_folder=main), draft, folders) == {}

    e1 = main / draft.folder / "e1" / "olfactometer"
    assert [p.name for p in (e1 / "data").iterdir()] == ["2026_08_11-13_58_59"]
    assert not (main / draft.folder / "e2" / "olfactometer").exists()


def test_ingestion_reads_the_log_from_the_loop_first(tmp_path):
    from odyn.database import _parse_program_starts

    db = SimpleNamespace(main_folder=tmp_path)
    loop = tmp_path / "20260811" / "m1" / "e1"
    start = datetime(2026, 8, 11, 12, 59, 59)

    with pytest.raises(FileNotFoundError, match="No olfactometer log for 2026-08-11"):
        _parse_program_starts(db, start, loop)

    shared = tmp_path / ".odyn/olfactometer/Log/Info/202608/Program_20260811.txt"
    shared.parent.mkdir(parents=True)
    shared.write_text("[13:00:00.100] Start program: old place\n")
    assert _parse_program_starts(db, start, loop)[0][1] == "old place"

    copied = loop / "olfactometer" / "log" / "Program_20260811.txt"
    copied.parent.mkdir(parents=True)
    copied.write_text("[13:00:00.100] Start program: in the loop\n")
    assert _parse_program_starts(db, start, loop)[0][1] == "in the loop"
