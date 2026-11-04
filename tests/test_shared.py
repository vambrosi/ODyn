"""
The JSON files the programs share: written all or nothing, the same bytes
from every OS, and replaced even while a reader has the old one open.
"""

import json
import os
import time

from datetime import date
from pathlib import Path

import pytest

from odyn import shared
from odyn.shared import (
    DONE_FOLDER,
    FAILED_FOLDER,
    Inbox,
    read_json,
    server_now,
    write_json,
)


def test_a_file_reads_back_as_written(tmp_path):
    path = write_json(tmp_path / "a" / "b.json", {"mark": "✔", "on": date(2026, 11, 3)})

    assert read_json(path) == {"mark": "✔", "on": "2026-11-03"}  # a date as text
    assert path.read_bytes() == json.dumps(
        {"mark": "✔", "on": "2026-11-03"}, indent=2
    ).encode(
        "utf-8"
    )  # UTF-8, `\n` line ends, on every OS


def test_a_failed_write_leaves_the_old_file_and_no_partial(tmp_path):
    path = write_json(tmp_path / "b.json", {"n": 1})

    with pytest.raises(ValueError):
        write_json(path, {"n": float("nan")}, allow_nan=False)

    assert read_json(path) == {"n": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["b.json"]


def test_a_rename_is_tried_again_while_a_reader_holds_the_file(tmp_path, monkeypatch):
    path = write_json(tmp_path / "c.json", {"n": 1})
    tries = []
    real = shared.os.replace

    def busy_twice(source, target):
        tries.append(target)

        if len(tries) < 3:
            raise PermissionError("in use")

        real(source, target)

    monkeypatch.setattr(shared.os, "replace", busy_twice)
    monkeypatch.setattr(shared.time, "sleep", lambda _: None)
    write_json(path, {"n": 2})

    assert len(tries) == 3 and read_json(path) == {"n": 2}


def test_a_listing_never_takes_a_partial_file_for_a_json_one(tmp_path):
    hidden = tmp_path / ".d.json.0123abcd.partial"  # what a write in progress is
    hidden.write_text("{")

    assert list(Path(tmp_path).glob("*.json")) == []


def test_now_is_read_from_the_server_and_leaves_nothing(tmp_path):
    before = os.stat(tmp_path).st_mtime

    assert abs(server_now(tmp_path) - time.time()) < 5  # here, one machine
    assert list(tmp_path.iterdir()) == []
    assert before <= os.stat(tmp_path).st_mtime


def test_an_inbox_lists_its_messages_oldest_first(tmp_path):
    inbox = Inbox(tmp_path, sidecars=(".state.json",))
    later = inbox.put("20261103-120500-bob", {"n": 2})
    first = inbox.put("20261103-120000-alice", {"n": 1})
    (tmp_path / "20261103-120000-alice.state.json").write_text("{}")  # its own
    (tmp_path / ".20261103-121000-carol.json.ab12cd34.partial").write_text("{")

    assert inbox.items() == [first, later]
    assert Inbox(tmp_path / "none").items() == []


def test_a_message_is_filed_with_what_belongs_to_it(tmp_path):
    inbox = Inbox(tmp_path, sidecars=(".state.json", ".now"))
    path = inbox.put("20261103-120000-alice", {"n": 1})
    (tmp_path / "20261103-120000-alice.state.json").write_text("{}")
    (tmp_path / "20261103-120000-alice.now").write_text("")

    done = inbox.file(path, DONE_FOLDER, {"applied_at": "2026-11-03 12:05:00"})

    assert read_json(done) == {"n": 1, "applied_at": "2026-11-03 12:05:00"}
    assert sorted(p.name for p in done.parent.iterdir()) == [
        "20261103-120000-alice.json",
        "20261103-120000-alice.now",
        "20261103-120000-alice.state.json",
    ]
    assert inbox.items() == [] and inbox.items(DONE_FOLDER) == [done]

    # One that does not read is filed as it is
    broken = tmp_path / "20261103-130000-bob.json"
    broken.write_text("{")
    failed = inbox.file(broken, FAILED_FOLDER, {"reason": "unreadable"})

    assert failed.read_text() == "{"
