"""
ODyn's log: one format everywhere (the time, the level, the message), color
only where it is shown, caiman's lines among ours.
"""

import io
import logging
import os
import re
import subprocess
import sys
import threading

from odyn import Database, record_call
from odyn.utils import (
    CHECK,
    _CallLog,
    _ConsoleHandler,
    call_log_path,
    _Formatter,
    colored,
    logger,
    interactive,
    plain,
)

LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \[INFO\] ")


def a_record(message, name="odyn", level=logging.INFO):
    return logging.LogRecord(name, level, __file__, 1, message, None, None)


class Terminal(io.StringIO):
    def isatty(self):
        return True


def test_a_file_gets_plain_lines_and_a_terminal_colors():
    record = a_record(f"Added user 'alice'. {CHECK}")

    line = _Formatter().format(record)
    assert LINE.match(line) and line.endswith("Added user 'alice'. ✔")
    assert "\033" not in line

    console = _ConsoleHandler()
    console.setStream(io.StringIO())  # a job's output file, a redirect
    assert console.format(record) == line

    console.setStream(Terminal())
    shown = console.format(record)
    assert "\033[1;34mINFO\033[0m" in shown and "\033[1;32m✔\033[0m" in shown
    assert plain(shown) == line
    assert colored("ok ✔") == "ok \033[1;32m✔\033[0m"


def test_caimans_lines_are_ours_and_say_so():
    caught = io.StringIO()
    handler = logging.StreamHandler(caught)
    handler.setFormatter(_Formatter())
    logger.addHandler(handler)

    try:
        caiman = logging.getLogger("caiman")
        caiman.info("** Starting parallel motion correction **")
        caiman.debug("Frame 12")  # too detailed: left out
    finally:
        logger.removeHandler(handler)

    [line] = caught.getvalue().splitlines()
    assert line.endswith("[INFO] caiman: ** Starting parallel motion correction **")
    assert not caiman.propagate  # not twice, through the root's


def test_a_calls_stored_log_has_the_time_and_no_colors(tmp_path, capsys):
    db = Database(tmp_path, can_create=True, user="alice")
    db.set_mouse(mouse_id=1)
    db.add_session(mouse_id=1, session_date="2026-07-01")
    call_id = int(db.method_calls.index.max())
    path = db.call_log_file(call_id)
    stored = path.read_text(encoding="utf-8")

    assert path == tmp_path / ".odyn/logs/main/0000" / f"{call_id}.log"
    assert "call_log" not in db.method_calls.columns
    assert all(LINE.match(line) for line in stored.splitlines())
    assert "\033" not in stored and "✔" in stored

    # Read back a line each (plain here: not a terminal)
    db.call_log(call_id)
    assert capsys.readouterr().out == stored


@record_call
def shout(self, *, words: list[str], inner: bool = False) -> None:
    for word in words:
        logger.info(word)

    if inner:
        shout(self, words=["inner"])


def calls_named(db, method_name):
    calls = db.calls

    return list(calls.index[calls["method_name"] == method_name])


def test_a_log_is_in_its_file_while_the_call_runs(tmp_path, monkeypatch):
    db = Database(tmp_path, can_create=True, user="alice")
    seen = []

    @record_call
    def peek(self) -> None:
        logger.info("before")
        handler = next(h for h in logger.handlers if isinstance(h, _CallLog))
        handler.save()  # as every `CALL_LOG_FLUSH_S` seconds
        seen.append(handler.path.read_text(encoding="utf-8"))

    peek(db)

    assert "before" in seen[0]  # there before the call ended


def test_an_inner_calls_lines_are_in_the_outer_log_too(tmp_path):
    db = Database(tmp_path, can_create=True, user="alice")
    shout(db, words=["outer"], inner=True)
    outer, inner = calls_named(db, "Database.shout")

    outer_log = db.call_log_file(outer).read_text(encoding="utf-8")
    inner_log = db.call_log_file(inner).read_text(encoding="utf-8")

    assert "outer" in outer_log and "inner" in outer_log
    assert "inner" in inner_log and "outer" not in inner_log


def test_calls_on_two_threads_keep_their_own_lines(tmp_path):
    db = Database(tmp_path, can_create=True, user="alice")
    started, go = threading.Event(), threading.Event()

    @record_call
    def wait(self) -> None:
        started.set()
        go.wait(10)
        logger.info("from the waiting call")

    def other():
        started.wait(10)
        own = Database(tmp_path, user="bob")  # a connection is for one thread
        shout(own, words=["from the other call"])
        logger.info("from a thread with no call")
        go.set()

    helper = threading.Thread(target=other)
    helper.start()
    wait(db)
    helper.join()

    (waiting,) = calls_named(db, "Database.wait")
    (shouting,) = calls_named(db, "Database.shout")
    waiting_log = db.call_log_file(waiting).read_text(encoding="utf-8")
    shouting_log = db.call_log_file(shouting).read_text(encoding="utf-8")

    assert "from the waiting call" in waiting_log
    assert "from a thread with no call" in waiting_log  # a helper's
    assert "from the other call" not in waiting_log
    assert "from the waiting call" not in shouting_log


def test_another_process_writes_nothing_in_a_log(tmp_path):
    handler = _CallLog(tmp_path / "1.log")
    record = a_record("from a forked worker")
    record.process = os.getpid() + 1

    assert not handler.filter(record)
    handler.finish()


def test_a_log_that_cannot_be_saved_does_not_stop_the_call(tmp_path, odyn_log):
    db = Database(tmp_path, can_create=True, user="alice")
    (tmp_path / ".odyn/logs").write_text("not a folder")

    shout(db, words=["kept in memory"])

    assert "kept in memory until the call ends" in odyn_log.text
    assert "Could not save the log of this call" in odyn_log.text
    assert calls_named(db, "Database.shout")  # recorded all the same


def test_the_hints_are_for_a_person_typing():
    assert not interactive()  # pytest is not someone typing

    script = "import odyn"
    quiet = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert "Hovering over" not in quiet.stderr

    typed = subprocess.run(
        [sys.executable, "-i", "-c", script],
        capture_output=True,
        text=True,
        input="",
    )
    assert "Hovering over" in typed.stderr


def test_a_projects_logs_are_in_a_folder_of_its_own(tmp_path):
    path = call_log_path(tmp_path, "main-sync", 1234)

    assert path == tmp_path / ".odyn/logs/main-sync/0001/1234.log"
