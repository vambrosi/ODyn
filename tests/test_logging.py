"""
ODyn's log: one format everywhere (the time, the level, the message), color
only where it is shown, caiman's lines among ours.
"""

import io
import logging
import re
import subprocess
import sys

from odyn import Database
from odyn.utils import (
    CHECK,
    _ConsoleHandler,
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
    record = a_record(f"Added user 'ana'. {CHECK}")

    line = _Formatter().format(record)
    assert LINE.match(line) and line.endswith("Added user 'ana'. ✔")
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
    db = Database(tmp_path, can_create=True, user="ana")
    db.set_mouse(mouse_id=1)
    db.add_session(mouse_id=1, session_date="2026-07-01")
    call_id = int(db.method_calls.index.max())
    stored = db.method_calls.loc[call_id, "call_log"]

    assert all(LINE.match(line) for line in stored.splitlines())
    assert "\033" not in stored and "✔" in stored

    # Read back a line each (plain here: not a terminal)
    db.call_log(call_id)
    assert capsys.readouterr().out == stored


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
