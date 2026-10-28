from __future__ import annotations

import functools
import getpass
import inspect
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import time

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntFlag
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, Union

if TYPE_CHECKING:
    from .database import Database
    from datetime import datetime
    from sqlite3 import Connection

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Plain marks: a terminal shows them in color (see `colored`), what is stored
# keeps the plain character
CHECK = "✔"
CROSS = "✘"

ODYN_FOLDER = ".odyn"
INFO_FOLDER = ".odyn/olfactometer/Log/Info"

# Used twice for a project:
#   - its database lives in '<main_folder>/.odyn/projects/<name>.db';
#   - its own files (scripts, outputs, movies) in '<main_folder>/projects/<name>'
PROJECTS_FOLDER = "projects"

# Backups of every database, in '<main_folder>/.odyn/backups'
BACKUPS_FOLDER = "backups"

# Where saved results go, under the project folder or the main folder
OUTPUTS_FOLDER = "outputs"

# Default wait before "database locked"
DB_TIMEOUT_S = 30


def check_name(kind: str, name: str) -> None:
    """
    Refuse a name that is not safe as a file name on every machine.

    Used for anything that becomes a file or folder name, like project names.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(
            f"{kind} names are letters, digits, '_', '-' and '.', starting with "
            f"a letter or digit, but got {name!r}."
        )


def database_path(main_folder: str | Path, project: None | str = None) -> Path:
    """
    Where the database of `main_folder` (or of one of its projects) lives.

    `None` is the shared `.odyn/odyn.db`; a project is `.odyn/projects/<name>.db`.
    """
    odyn_folder = Path(main_folder) / ODYN_FOLDER

    if project is None:
        return odyn_folder / "odyn.db"

    # The project name becomes the file name
    check_name("Project", project)

    # Backups name the main database `main`, so a project cannot take that name
    if project.lower() == "main":
        raise ValueError(f"'main' is reserved for the main database, got {project!r}.")

    return odyn_folder / PROJECTS_FOLDER / f"{project}.db"


def backup_path(main_folder: str | Path, project: None | str, kind: str) -> Path:
    """
    Where a new backup of a database goes.

    Backups of the main database and of every project share one folder, as
    `.odyn/backups/<time>-<project or main>-<kind>.db`, so they sort by when
    they were made and never clash when copied together.
    """
    database_path(main_folder, project)  # checks the project name

    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = "main" if project is None else project

    return (
        Path(main_folder) / ODYN_FOLDER / BACKUPS_FOLDER / f"{stamp}-{label}-{kind}.db"
    )


# List is invariant     => list[float] is not a list[Value]
# Sequence is covariant => list[float] is a list[Value]
#
# `TypeAlias` rather than a `type` statement, which needs Python 3.12. These are
# evaluated where they are written, so names not defined yet are quoted.
BasicTypes: TypeAlias = Union[None, bool, int, float, str, "datetime"]
Value: TypeAlias = Union[BasicTypes, "Object", Sequence["Value"]]
Object: TypeAlias = dict[str, "Value"]

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

# `Database` exposes every table as a cached DataFrame, and `Group` exposes the
# same tables restricted to its experiments. Each pair differs only in the SQL,
# so both queries live side by side in one spec and one reader serves both
# classes: the cache check, the read, the date parsing, the index and the
# normalization below happen in exactly one place.


@dataclass(frozen=True)
class TableSpec:
    """
    How one table is read as a DataFrame.

    - `name` is the attribute (`db.acquisitions`) and the cache key;
    - `index` is the column that becomes the index;
    - `all_rows` is the query `Database` runs, `in_group` the one `Group` runs
      with its `group_id` as the single `?` parameter (`None` when the table
      has no per-group meaning);
    - `dates` are parsed as datetimes and `json` columns as Python objects.
    """

    name: str
    index: str
    all_rows: str
    in_group: None | str = None
    dates: tuple[str, ...] = ()
    json: tuple[str, ...] = ()


# Columns `sync_blocks` adds to an acquisition. It is one row per acquisition
# and was split out for storage reasons (it arrives when the sync file is
# processed, and processing it again replaces it whole), not because it describes
# something else, so it is joined back wherever acquisitions are read. LEFT,
# because those rows are missing until the sync file is processed.
_SYNC_COLUMNS = """
         , s.stream_id
         , s.sync_block
         , s.sync_start_sample
         , s.sync_frame_count
         , s.sync_frame_period_s
         , s.sync_odor_pulses
         , s.sync_odor_on_frame
         , s.sync_odor_off_frame
         , s.sync_odor_on_residual_s
         , s.sync_odor_start
         , s.sync_odor_end
         , s.sync_camera_frames
         , s.clock_offset_ms
"""

# Trial columns added to an acquisition. At most one trial per acquisition, so
# the join never splits a row; acquisitions without a trial keep NULLs.
_TRIAL_COLUMNS = """
         , t.trial_id
         , t.trial_start
         , t.trial_odor_start
         , t.trial_odor_end
         , t.odor_id
         , t.outcome
         , t.program_id
         , p.program_type
"""

# The join chain that restricts a table to a group's experiments
_GROUP_EXPERIMENTS = """
        FROM group_experiments AS g
        JOIN experiments AS e ON e.exp_id = g.exp_id
"""


def _simple(name: str, index: str, dates=(), json=(), via: None | str = None):
    """
    A spec for `SELECT * FROM name`, restricted to a group through `via`.

    `via` is the join from `experiments AS e` down to the table, aliased `x`,
    e.g. `"JOIN acquisitions AS x ON x.exp_id = e.exp_id"`. Tables with no
    per-group meaning leave it out.
    """
    in_group = None

    if via is not None:
        in_group = f"SELECT x.* {_GROUP_EXPERIMENTS} {via} WHERE g.group_id = ?;"

    return TableSpec(
        name,
        index,
        f"SELECT * FROM {name};",
        in_group,
        tuple(dates),
        tuple(json),
    )


USERS = _simple("users", "user_id")
MICE = _simple("mice", "mouse_id", dates=["added_to_db_at"])
LINES = TableSpec("lines", "line_name", "SELECT * FROM lines;")
MOUSE_LINES = TableSpec("mouse_lines", "mouse_id", "SELECT * FROM mouse_lines;")
VIRUSES = TableSpec("viruses", "virus_name", "SELECT * FROM viruses;")
MOUSE_VIRUSES = TableSpec("mouse_viruses", "mouse_id", "SELECT * FROM mouse_viruses;")
SESSIONS = _simple(
    "sessions",
    "session_id",
    dates=["added_to_db_at"],
    via="JOIN sessions AS x ON x.session_id = e.session_id",
)
EXPERIMENTS = TableSpec(
    "experiments",
    "exp_id",
    "SELECT * FROM experiments;",
    f"SELECT e.* {_GROUP_EXPERIMENTS} WHERE g.group_id = ?;",
    dates=("exp_start", "added_to_db_at"),
)
GROUP_EXPERIMENTS = TableSpec(
    "group_experiments",
    "group_id",
    # Group 0 is the database's own row and has no experiments
    f"SELECT g.group_id, e.* {_GROUP_EXPERIMENTS} WHERE g.group_id != 0;",
    dates=("exp_start", "added_to_db_at"),
)
ZSTACKS = _simple("zstacks", "zstack_id", dates=["zstack_start", "added_to_db_at"])
ACQUISITIONS = TableSpec(
    "acquisitions",
    "acq_id",
    f"""
    SELECT a.*
         {_SYNC_COLUMNS}
        FROM acquisitions AS a
        LEFT JOIN sync_blocks AS s ON s.acq_id = a.acq_id;
    """,
    f"""
    SELECT a.*
         {_SYNC_COLUMNS}
        {_GROUP_EXPERIMENTS}
        JOIN acquisitions AS a ON a.exp_id = e.exp_id
        LEFT JOIN sync_blocks AS s ON s.acq_id = a.acq_id
        WHERE g.group_id = ?;
    """,
    dates=("acq_start", "sync_odor_start", "sync_odor_end"),
)
ACQUISITION_TRIALS = TableSpec(
    "acquisition_trials",
    "acq_id",
    f"""
    SELECT a.*
         {_SYNC_COLUMNS}
         {_TRIAL_COLUMNS}
        FROM acquisitions AS a
        LEFT JOIN sync_blocks AS s ON s.acq_id = a.acq_id
        LEFT JOIN trials      AS t ON t.acq_id = a.acq_id
        LEFT JOIN programs    AS p ON p.program_id = t.program_id;
    """,
    f"""
    SELECT a.*
         {_SYNC_COLUMNS}
         {_TRIAL_COLUMNS}
        {_GROUP_EXPERIMENTS}
        JOIN acquisitions     AS a ON a.exp_id = e.exp_id
        LEFT JOIN sync_blocks AS s ON s.acq_id = a.acq_id
        LEFT JOIN trials      AS t ON t.acq_id = a.acq_id
        LEFT JOIN programs    AS p ON p.program_id = t.program_id
        WHERE g.group_id = ?;
    """,
    dates=(
        "acq_start",
        "sync_odor_start",
        "sync_odor_end",
        "trial_start",
        "trial_odor_start",
        "trial_odor_end",
    ),
)
STREAMS = _simple(
    "streams",
    "stream_id",
    dates=["start_time", "stop_time"],
    json=["attributes"],
    via="JOIN streams AS x ON x.exp_id = e.exp_id",
)
SYNC_BLOCKS = TableSpec(
    "sync_blocks",
    "acq_id",
    "SELECT * FROM sync_blocks;",
    f"""
    SELECT x.* {_GROUP_EXPERIMENTS}
        JOIN streams     AS s ON s.exp_id = e.exp_id
        JOIN sync_blocks AS x ON x.stream_id = s.stream_id
        WHERE g.group_id = ?;
    """,
    dates=("sync_odor_start", "sync_odor_end"),
)
MCOR_FILES = _simple(
    "mcor_files",
    "acq_id",
    via="JOIN acquisitions AS a ON a.exp_id = e.exp_id "
    "JOIN mcor_files AS x ON x.acq_id = a.acq_id",
)
PROGRAMS = _simple(
    "programs",
    "program_id",
    dates=["program_start"],
    json=["program_config"],
    via="JOIN programs AS x ON x.exp_id = e.exp_id",
)
TRIALS = _simple(
    "trials",
    "trial_id",
    dates=["trial_start", "trial_odor_start", "trial_odor_end"],
    via="JOIN programs AS p ON p.exp_id = e.exp_id "
    "JOIN trials AS x ON x.program_id = p.program_id",
)
EVENTS = _simple(
    "events",
    "event_id",
    dates=["event_time"],
    via="JOIN programs AS p ON p.exp_id = e.exp_id "
    "JOIN events AS x ON x.program_id = p.program_id",
)
ODORS = _simple("odors", "odor_id")
ODOR_COMPONENTS = TableSpec(
    "odor_components", "odor_id", "SELECT * FROM odor_components;"
)
PANELS = _simple("panels", "panel_id", dates=["added_to_db_at"])
PANEL_POSITIONS = TableSpec(
    "panel_positions", "panel_id", "SELECT * FROM panel_positions;"
)

# A vial's odor is its position's, joined back so a vial reads whole
VIALS = TableSpec(
    "vials",
    "vial_id",
    "SELECT v.*, p.odor_id FROM vials AS v "
    "JOIN panel_positions AS p USING (panel_id, vial_position);",
    dates=("made_on", "added_to_db_at"),
)
VIAL_ODORS = TableSpec("vial_odors", "vial_id", "SELECT * FROM vial_odors;")
EXPERIMENT_VIALS = TableSpec(
    "experiment_vials",
    "exp_id",
    "SELECT * FROM experiment_vials;",
    f"SELECT x.* {_GROUP_EXPERIMENTS} "
    "JOIN experiment_vials AS x ON x.exp_id = e.exp_id WHERE g.group_id = ?;",
)
ANNOTATION_KEYS = TableSpec(
    "annotation_keys",
    "applies_to",  # the reader sets a (applies_to, key) index, see `_read_table`
    "SELECT * FROM annotation_keys;",
    json=("allowed_values",),
)
ANNOTATIONS = _simple("annotations", "annotation_id")
METHOD_CALLS = TableSpec(
    "method_calls",
    "method_call_id",
    "SELECT * FROM method_calls;",
    "SELECT * FROM method_calls WHERE group_id = ?;",
    dates=("called_at", "ended_at"),
    json=(
        "parameter_inputs",
        "parameters_used",
        "call_output",
        "code",
        "environment",
        "consumed_calls",
    ),
)
OUTPUTS = TableSpec(
    "outputs",
    "output_id",
    "SELECT * FROM outputs;",
    """
    SELECT o.* FROM method_calls AS mc
        JOIN outputs AS o ON o.method_call_id = mc.method_call_id
        WHERE mc.group_id = ?;
    """,
)

# What an annotation can be attached to, and where that row lives. The target
# is polymorphic, so SQLite cannot check `annotations.target_id` with a foreign
# key and this stands in for one. Keep it in step with `applies_to` in
# `create.sql`.
ANNOTATION_TARGETS: dict[str, TableSpec] = {
    "mouse": MICE,
    "session": SESSIONS,
    "experiment": EXPERIMENTS,
    "zstack": ZSTACKS,
    "program": PROGRAMS,
    "acquisition": ACQUISITIONS,
    "group": TableSpec("groups", "group_id", "SELECT * FROM groups;"),
}


def _read_table(
    con: Connection, spec: TableSpec, query: str, params: list = []
) -> pd.DataFrame:
    """
    Read one table as the DataFrame the properties hand out.

    Besides the dates and the index, it settles what pandas would otherwise
    decide by version: a column with nothing but NULLs comes back as `object`
    of `None` on one pandas and as `str` of `NaN` on another, and a viewer
    written against one crashes on the other. Here every text column is
    `object` with `None` for NULL, JSON columns hold Python objects (`None`
    for NULL) and numbers keep NaN, on every machine.
    """
    frame = pd.read_sql_query(query, con, params=params, parse_dates=list(spec.dates))

    for column in frame.columns:
        if column in spec.dates:
            continue

        if column in spec.json:
            frame[column] = frame[column].map(
                lambda text: json.loads(text) if isinstance(text, str) else None
            )

        elif not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = (
                frame[column].astype(object).where(frame[column].notna(), None)
            )

    if spec is ANNOTATION_KEYS:
        return frame.set_index(["applies_to", "key"])

    return frame.set_index(spec.index)


def _method_calls_dataframe(con: Connection, query: str, params: list) -> pd.DataFrame:
    """
    Run `query` against method_calls and expand its JSON columns into columns.

    Shared body of Database.latest_calls / Group.latest_calls. `query` must
    select from method_calls and return the method_call_id column.

    The effective inputs of each call come back as `input_<name>` columns and
    its output as `output_<name>` columns, so an input and an output with the
    same name never land on each other.

    NOTE: json_normalize raises on None on macOS but yields no columns on
    Windows, so missing JSON is coerced to {} (no columns, every OS).
    """
    frame = _read_table(con, METHOD_CALLS, query, params)

    used = frame["parameters_used"].map(lambda x: x or {})
    output = frame["call_output"].map(lambda x: x or {})

    inputs = pd.json_normalize(list(used)).set_index(frame.index).add_prefix("input_")
    outputs = (
        pd.json_normalize(list(output)).set_index(frame.index).add_prefix("output_")
    )

    return pd.concat(
        [frame.drop(columns=["parameters_used", "call_output"]), inputs, outputs],
        axis=1,
    )


# --------------------------------------------------------------------------- #
# Call Recording
# --------------------------------------------------------------------------- #


class CallFlag(IntFlag):
    """
    Flags set automatically by @record_call.

    Bit 0 is reserved. Per-function enums should start at 1 << 1.
    """

    SUCCESS = 0
    RAISED = 1 << 0


@dataclass
class CallFrame:
    """Per-call scratch state pushed onto self._call_stack by @record_call."""

    call_id: int
    flag: int = 0
    output: Object | None = None
    used: Object = field(default_factory=dict)

    # Calls whose output this call read through `latest_output`
    consumed: list[int] = field(default_factory=list)


class CallRecorder:
    """
    `@record_call` helpers shared by `Database` and `Group`.

    Subclasses initialize `_call_stack` and `_tables`.
    """

    _call_stack: list[CallFrame]
    _tables: dict[str, pd.DataFrame]

    @property
    def _recording_db(self) -> Database:
        """The `Database` that owns the connection (self for `Database`)."""
        return getattr(self, "db", self)

    @property
    def current_call_id(self) -> int:
        assert (
            self._call_stack
        ), "'current_call_id' is only available inside a '@record_call'"
        return self._call_stack[-1].call_id

    def add_flag(self, flag) -> None:
        """Set bits on the current call's flag (bitwise OR). Use inside `@record_call`."""
        self._call_stack[-1].flag |= int(flag)

    def set_output(self, output: Object) -> None:
        """Record this call's output as JSON. Overwrite previous outputs."""
        self._call_stack[-1].output = output

    def update_parameters_used(self, params: Object) -> None:
        """Merge values into this call's parameters_used. Use inside `@record_call`."""
        self._call_stack[-1].used.update(params)

    def fail(self, flag, message: str = "") -> None:
        """Flag the current call and abort it by raising RuntimeError."""
        self.add_flag(flag)
        raise RuntimeError(message)

    def note_consumed(self, call_id: int) -> None:
        """Record that the current call read `call_id`'s output."""
        # Does nothing outside a recorded call
        if self._call_stack and call_id not in self._call_stack[-1].consumed:
            self._call_stack[-1].consumed.append(call_id)

    def add_output_file(self, path: str | Path) -> None:
        """Record a file in `outputs` (path relative to main_folder)."""
        db = self._recording_db
        rel_path = Path(path).relative_to(db.main_folder).as_posix()

        with db._locked() as con, con:
            con.execute(
                "INSERT INTO outputs (method_call_id, file_path, removed) VALUES (?, ?, FALSE);",
                [self.current_call_id, rel_path],
            )

        # The outputs table is shared, so drop both cached views.
        self._tables.pop("outputs", None)
        db._tables.pop("outputs", None)


def record_call(func):
    """
    Decorator for Database/Group methods that should be tracked in method_calls.

    Records the call, captures all log output during execution, and saves it
    to call_log when the method returns (even on exception).

    Pushes a CallFrame onto self._call_stack for the duration of the call, so
    the function body can read self.current_call_id (e.g. for a foreign key) and
    record results via self.add_flag(...) / self.set_output(...). On exception
    the CallFlag.RAISED bit is set. The flag and output are written to the
    method_calls row when the call returns (even on exception). Supports nesting.
    """

    @functools.wraps(func)
    def wrapper(self, **kwargs):
        # Support both Database (db = self) and Group (db = self.db)
        db = getattr(self, "db", self)

        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_plain_formatter)
        logger.addHandler(handler)

        # parameters_used starts as defaults plus whatever was passed.
        # methods may change them during the call.
        parameters_used = {**(func.__kwdefaults__ or {}), **kwargs}

        # Asked before taking the lock, since it runs git.
        code = get_code(func, _caller_file())

        with db._locked() as con, con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO method_calls
                    ( group_id
                    , user
                    , method_name
                    , module
                    , code
                    , environment
                    , parameter_inputs
                    , parameters_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    self.group_id,
                    db.user,
                    f"{type(self).__name__}.{func.__name__}",
                    func.__module__,
                    json.dumps(code),
                    json.dumps(get_environment()),
                    json.dumps(jsonable(kwargs)),
                    json.dumps(jsonable(parameters_used)),
                ],
            )
            call_id = cur.lastrowid

        logger.info(f"Recorded method call to db (method_call_id = {call_id}).")

        # A recorded call is a new row in method_calls, for both views
        self._tables.pop("method_calls", None)
        db._tables.pop("method_calls", None)

        frame = CallFrame(call_id, used=dict(parameters_used))
        self._call_stack.append(frame)

        try:
            return func(self, **kwargs)

        except Exception:
            frame.flag |= int(CallFlag.RAISED)
            raise

        finally:
            self._call_stack.pop()
            logger.removeHandler(handler)

            call_output = (
                json.dumps(jsonable(frame.output)) if frame.output is not None else None
            )

            with db._locked() as con, con:
                con.execute(
                    """
                    UPDATE method_calls
                        SET call_log = ?
                          , call_flag = ?
                          , call_output = ?
                          , parameters_used = ?
                          , consumed_calls = ?
                          , ended_at = datetime('now', 'localtime')
                        WHERE method_call_id = ?
                    """,
                    [
                        buf.getvalue(),
                        int(frame.flag),
                        call_output,
                        json.dumps(jsonable(frame.used)),
                        json.dumps(frame.consumed) if frame.consumed else None,
                        call_id,
                    ],
                )

    # NOTE: This is to make memorize_params work
    wrapper.__kwdefaults__ = func.__kwdefaults__
    return wrapper


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


# One format for every log line, wherever it goes: the terminal, a job's
# output file, the database (`record_call`). Color is added only where it is
# shown (a terminal or a notebook), never stored: escape codes in a file or a
# database get in the way of searching and of anything else that reads them.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Terminal colors: of each level's name, and of the marks
LEVEL_COLORS = {
    logging.DEBUG: "\033[0;37m",  # grey
    logging.INFO: "\033[1;34m",  # bold blue
    logging.WARNING: "\033[1;33m",  # bold yellow
    logging.ERROR: "\033[1;31m",  # bold red
    logging.CRITICAL: "\033[1;31m",
}
MARK_COLORS = {CHECK: "\033[1;32m", CROSS: "\033[1;31m"}
RESET = "\033[0m"
ESCAPE = re.compile(r"\033\[[0-9;]*m")

# caiman's own log, sent to ours (see `_ToOdyn`), at this level: its INFO
# lines are the steps of a motion correction (a few per acquisition); DEBUG
# is every frame and iteration
CAIMAN_LEVEL = logging.INFO


def colored(text: str) -> str:
    """`text` with its marks (`CHECK`, `CROSS`) in their terminal colors."""
    for mark, code in MARK_COLORS.items():
        text = text.replace(mark, f"{code}{mark}{RESET}")

    return text


def plain(text: str) -> str:
    """`text` without terminal colors (logs stored before they were left out)."""
    return ESCAPE.sub("", text)


def log_message(record: logging.LogRecord) -> str:
    """A record's message, said to be caiman's when it is."""
    message = record.getMessage()

    return f"caiman: {message}" if record.name.startswith("caiman") else message


class _Formatter(logging.Formatter):
    """`LOG_FORMAT`; with `color`, the level and the marks in color."""

    def __init__(self, color: bool = False):
        super().__init__(LOG_FORMAT, LOG_DATE_FORMAT)
        self.color = color

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        message = log_message(record)
        level = record.levelname

        if self.color:
            code = LEVEL_COLORS.get(record.levelno, "")
            level = f"{code}{level}{RESET}" if code else level
            message = colored(message)

        return f"{record.asctime} [{level}] {message}"


class _ConsoleHandler(logging.StreamHandler):
    """
    The console: in color on a terminal or in a notebook (which shows
    terminal colors), plain into a file (a job's output, a redirect).
    """

    def __init__(self):
        super().__init__()
        self.plain = _Formatter()
        self.color = _Formatter(color=True)

    def format(self, record: logging.LogRecord) -> str:
        return (self.color if _shows_color(self.stream) else self.plain).format(record)


def _ipython_shell():
    """
    The IPython shell Python runs in, or `None`. Not just whether IPython
    was imported: caiman imports it (with ipykernel) in any script.
    """
    ipython = sys.modules.get("IPython")

    try:
        return None if ipython is None else ipython.get_ipython()
    except Exception:  # an IPython too old, or half imported
        return None


def _shows_color(stream) -> bool:
    shell = _ipython_shell()

    if type(shell).__name__ == "ZMQInteractiveShell":  # a notebook
        return True

    try:
        return stream.isatty()
    except (AttributeError, ValueError):  # no stream, or a closed one
        return False


class _ToOdyn(logging.Handler):
    """Hands caiman's records to odyn's logger: the same lines, the same places."""

    def emit(self, record: logging.LogRecord) -> None:
        logger.handle(record)


def interactive() -> bool:
    """
    Whether Python runs for a person typing: a notebook, IPython, or the
    `python` prompt (not a script, a job or a command).
    """
    return (
        hasattr(sys, "ps1")
        or bool(sys.flags.interactive)
        or _ipython_shell() is not None
    )


_plain_formatter = _Formatter()

logger = logging.getLogger("odyn")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    logger.addHandler(_ConsoleHandler())

_caiman_logger = logging.getLogger("caiman")

if not any(isinstance(h, _ToOdyn) for h in _caiman_logger.handlers):
    _caiman_logger.addHandler(_ToOdyn())
    _caiman_logger.setLevel(CAIMAN_LEVEL)
    _caiman_logger.propagate = False  # once, not again by the root's


def jsonable(value):
    """
    Turn `value` into something `json.dumps` writes and SQLite will store.

    Everything recorded about a call goes through here, because what reaches a
    method is whatever the caller had to hand: an id out of a DataFrame index is
    a numpy integer, a file argument is a `Path`, a threshold that came from
    `np.percentile` is a numpy float, and one read out of a table can be NaN.
    None of those are JSON, and two of them fail in ways that are hard to read:

    - a numpy or `Path` argument raises `TypeError` at the *start* of the call,
      before the method has done anything, from a line about JSON;
    - a NaN does not raise at all. `json.dumps` writes a bare `NaN`, which is
      not valid JSON, so the `json_valid` CHECK rejects it -- and for
      `parameters_used` that happens in the UPDATE at the *end*, throwing away
      the work of a call that had otherwise finished.

    Non-finite floats become `None`, which reads back as NaN through pandas.
    Anything unrecognized becomes its `str`, because losing the exact form of an
    argument in the log is better than failing a call over it.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]

    # numpy scalars and arrays, pandas Series and Index. `tolist` gives Python
    # types, and recursing catches any NaN inside.
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())

    # datetime, date, time, pandas Timestamp.
    #
    # A space instead of 'T', like SQLite's `datetime('now')`. `sep` goes as a
    # keyword because `date` and `time` take none: they raise TypeError and fall
    # back, whereas `time.isoformat(" ")` would read it as a timespec and raise
    # ValueError instead.
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(sep=" ")

        except TypeError:
            return value.isoformat()

    if isinstance(value, os.PathLike):
        return os.fspath(value)

    return str(value)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

ODYN_ROOT = Path(__file__).resolve().parent.parent

# Packages whose version is recorded with every call:
#   `name: (distributions, module)`
# The module is the fallback for a conda package with no pip metadata
# (conda's OpenCV has none), and is only read if something already imported it.

ENVIRONMENT_PACKAGES = {
    "numpy": (("numpy",), "numpy"),
    "pandas": (("pandas",), "pandas"),
    "scipy": (("scipy",), "scipy"),
    "opencv": (("opencv-python", "opencv-python-headless", "opencv"), "cv2"),
    "caiman": (("caiman",), "caiman"),
    "tifffile": (("tifffile",), "tifffile"),
    "h5py": (("h5py",), "h5py"),
}


def get_user() -> str:
    """Who is running the call: `ODYN_USER` if set, else the computer login."""
    # Shared computers stay logged in as one user, so `ODYN_USER` is how
    # people tell themselves apart there.
    return os.environ.get("ODYN_USER") or getpass.getuser()


def _git(*args: str, cwd: str | Path) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    )


@functools.cache
def _repo_root(folder: str) -> None | Path:
    """The git work tree holding `folder`, or None."""
    try:
        return Path(_git("rev-parse", "--show-toplevel", cwd=folder).strip())

    except Exception:
        # No git, not a repository, or a folder that no longer exists
        return None


def _repo_state(root: Path) -> None | Object:
    """`{commit, dirty}` of the work tree at `root`."""
    # Not cached, unlike the root: a notebook left open while files are edited
    # or another commit is checked out must not keep recording the old state.
    try:
        commit = _git("rev-parse", "HEAD", cwd=root).strip()

        # Untracked files do not change the code that ran
        changes = _git("status", "--porcelain", "--untracked-files=no", cwd=root)

    except Exception:
        return None

    return {"commit": commit, "dirty": bool(changes.strip())}


def _caller_file() -> None | str:
    """The file that called into odyn, skipping the decorators in this file."""
    frame = sys._getframe(1)

    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back

    return frame.f_code.co_filename if frame is not None else None


def get_code(func, caller_file: None | str = None) -> Object:
    """
    Which code ran, per git repository: `{name: {"commit": ..., "dirty": ...}}`.

    Looks where `func` is defined, at odyn itself, and at the file that called
    it, so a script in a project repository is recorded too. odyn is always
    named `odyn`; other repositories go by their folder name. Folders outside
    any repository are left out.
    """
    folders = [ODYN_ROOT]

    try:
        folders.append(Path(inspect.getfile(func)).resolve().parent)

    except TypeError:
        pass  # no source file

    # Notebooks and the REPL report names like '<stdin>' that are not files
    if caller_file and Path(caller_file).is_file():
        folders.append(Path(caller_file).resolve().parent)

    code: Object = {}
    odyn_root = _repo_root(str(ODYN_ROOT))

    for folder in folders:
        root = _repo_root(str(folder))
        name = "odyn" if root == odyn_root else root.name if root else None

        if name is None or name in code:
            continue

        state = _repo_state(root)

        if state is not None:
            code[name] = state

    return code


@functools.cache
def get_environment() -> Object:
    """Python and the versions of `ENVIRONMENT_PACKAGES`, once per process."""
    from importlib.metadata import PackageNotFoundError, version

    found: Object = {"python": platform.python_version()}

    for name, (distributions, module) in ENVIRONMENT_PACKAGES.items():
        for distribution in distributions:
            try:
                found[name] = version(distribution)
                break

            except PackageNotFoundError:
                continue

        else:
            imported = sys.modules.get(module)

            if getattr(imported, "__version__", None):
                found[name] = str(imported.__version__)

    return found


# --------------------------------------------------------------------------- #
# Other Features
# --------------------------------------------------------------------------- #


def memorize_params(method):
    # NOTE: - Fails if there is no required keyword argument
    #       - Positional arguments will be ignored silently

    params = {}

    @functools.wraps(method)
    def wrapper(self, *, use_last_parameters=False, **kwargs):
        assert method.__kwdefaults__ is not None, "Must have a parameter default."

        # If user is passing invalid arguments, just pass
        # them to the method so it can report the error
        if not kwargs.keys() <= method.__kwdefaults__.keys():
            return method(self, **kwargs)

        # Clear chached parameters if caller is not using them
        if not use_last_parameters:
            params.clear()

        # Add kwargs (possibly overwriting) to last params
        params.update(kwargs)

        # INVARIANT: params are the kwargs of the last valid method call.
        return method(self, **params)

    return wrapper


# --------------------------------------------------------------------------- #
# Numerical Functions
# --------------------------------------------------------------------------- #


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]


def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))
