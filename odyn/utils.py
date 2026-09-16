from __future__ import annotations

import functools
import json
import logging
import math
import os
import subprocess

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from io import StringIO
from pathlib import Path
from tqdm.auto import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database
    from .groups import Group
    from datetime import datetime
    from sqlite3 import Connection

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CHECK = "\033[1;32m✔\033[0m"
CROSS = "\033[1;31m✘\033[0m"

ODYN_FOLDER = ".odyn"
INFO_FOLDER = ".odyn/olfactometer/Log/Info"

# Used twice for a project:
#   - its database lives in '<main_folder>/.odyn/projects/<name>.db';
#   - its own files (scripts, outputs, movies) in '<main_folder>/projects/<name>'
PROJECTS_FOLDER = "projects"

# Where saved results go, under the project folder or the main folder
OUTPUTS_FOLDER = "outputs"

# Default wait before "database locked"
DB_TIMEOUT_S = 30

# List is invariant     => list[float] is not a list[Value]
# Sequence is covariant => list[float] is a list[Value]
type BasicTypes = None | bool | int | float | str | datetime
type Value = BasicTypes | Object | Sequence[Value]
type Object = dict[str, Value]

# --------------------------------------------------------------------------- #
# Database queries
# --------------------------------------------------------------------------- #

# Two ways of accessing the same view, kept together so drifts are visible.
#
# ASSUMPTIONS:
# - Some acquisitions don't have trials (thus, we use LEFT JOIN);
# - At most one trial per acquisition, so no row-splitting.
#
# NOTES:
# - 'a.*' rather than a column list, to allow for new columns;
# - trial columns are named to leave out some and rename other collisions.

ACQUISITION_TRIALS = """
    SELECT a.*
         , t.trial_id
         , t.trial_start
         , t.odor_start AS trial_odor_start
         , t.odor_end   AS trial_odor_end
         , t.odor_id
         , t.outcome
         , t.h5_to_trial_ms
         , t.program_id
         , p.program_type
        FROM acquisitions AS a
        LEFT JOIN trials   AS t ON t.acq_id = a.acq_id
        LEFT JOIN programs AS p ON p.program_id = t.program_id;
"""

GROUP_ACQUISITION_TRIALS = """
    SELECT a.*
         , t.trial_id
         , t.trial_start
         , t.odor_start AS trial_odor_start
         , t.odor_end   AS trial_odor_end
         , t.odor_id
         , t.outcome
         , t.h5_to_trial_ms
         , t.program_id
         , p.program_type
        FROM group_experiments AS g
        JOIN experiments   AS e ON e.exp_id = g.exp_id
        JOIN acquisitions  AS a ON a.exp_id = e.exp_id
        LEFT JOIN trials   AS t ON t.acq_id = a.acq_id
        LEFT JOIN programs AS p ON p.program_id = t.program_id
        WHERE g.group_id = ?;
"""


def _acquisition_trials(
    con: Connection, query: str, params: list = []
) -> pd.DataFrame:
    """
    Shared body of `Database.acquisition_trials` / `Group.acquisition_trials`.

    When used with the queries above, it returns a left join between the
    acquisitions and trials tables. Both have odor window timings, which come
    from the H5 + TIFF metadata and the olfactometer events, respectively.
    Those can disagree significantly, so they are all kept in the join.
    """
    frame = pd.read_sql_query(
        query,
        con,
        params=params,
        # Names as the query returns them, before the rename below
        parse_dates=[
            "acq_start",
            "odor_start",
            "odor_end",
            "trial_start",
            "trial_odor_start",
            "trial_odor_end",
        ],
    )

    frame = frame.rename(
        columns={"odor_start": "acq_odor_start", "odor_end": "acq_odor_end"}
    )
    frame.set_index("acq_id", inplace=True)

    return frame


def _method_calls_dataframe(con: Connection, query: str, params: list) -> pd.DataFrame:
    """
    Run `query` against method_calls and expand its JSON columns into columns.

    Shared body of Database.latest_calls / Group.latest_calls. `query` must
    select from method_calls and return the method_call_id column.

    NOTE: json_normalize raises on None on macOS but yields no columns on
    Windows, so missing JSON is coerced to {} (no columns, every OS).
    """

    df = pd.read_sql_query(query, con, params=params)
    df.set_index("method_call_id", inplace=True)

    df["parameter_inputs"] = df["parameter_inputs"].apply(json.loads)
    df["parameters_used"] = df["parameters_used"].apply(json.loads)
    df["call_output"] = df["call_output"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else {}
    )

    df_parameters_used = pd.json_normalize(df.parameters_used).set_index(df.index)
    df_output = pd.json_normalize(df.call_output).set_index(df.index)

    return pd.concat(
        [
            df.drop(columns=["parameters_used", "call_output"]),
            df_parameters_used,
            df_output,
        ],
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


class CallRecorder:
    """
    `@record_call` helpers shared by `Database` and `Group`.

    Subclasses initialize `_call_stack` and `_outputs`.
    """

    _call_stack: list[CallFrame]

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
        self._outputs = None
        db._outputs = None


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

        with db._locked() as con, con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO method_calls
                    ( group_id
                    , method_name
                    , parameter_inputs
                    , git_commit
                    , parameters_used
                    ) VALUES (?, ?, ?, ?, ?);
                """,
                [
                    self.group_id,
                    f"{type(self).__name__}.{func.__name__}",
                    json.dumps(jsonable(kwargs)),
                    get_git_hash(),
                    json.dumps(jsonable(parameters_used)),
                ],
            )
            call_id = cur.lastrowid

        logger.info(f"Recorded method call to db (method_call_id = {call_id}).")

        # Reset method_calls caches
        self._method_calls = None
        if self.group_id != 0:
            db._method_calls = None

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

            call_output = json.dumps(jsonable(frame.output)) if frame.output is not None else None

            with db._locked() as con, con:
                con.execute(
                    """
                    UPDATE method_calls
                        SET call_log = ?
                          , call_flag = ?
                          , call_output = ?
                          , parameters_used = ?
                        WHERE method_call_id = ?
                    """,
                    [
                        buf.getvalue(),
                        int(frame.flag),
                        call_output,
                        json.dumps(jsonable(frame.used)),
                        call_id,
                    ],
                )

    # NOTE: This is to make memorize_params work
    wrapper.__kwdefaults__ = func.__kwdefaults__
    return wrapper


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class _ColorFormatter(logging.Formatter):
    _COLORS = {
        logging.DEBUG: "\033[0;37m",  # grey
        logging.INFO: "\033[1;34m",  # bold blue
        logging.WARNING: "\033[1;33m",  # bold yellow
        logging.ERROR: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        result = f"[{color}{record.levelname}{self._RESET}] {record.getMessage()}"

        # Add error stack trace to log
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            result += "\n" + record.exc_text

        return result


_plain_formatter = logging.Formatter("[%(levelname)s] %(message)s")

logger = logging.getLogger("odyn")
logger.setLevel(logging.DEBUG)
logger.propagate = False

if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_ColorFormatter())
    logger.addHandler(_console_handler)


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

    # datetime, date, time, pandas Timestamp
    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, os.PathLike):
        return os.fspath(value)

    return str(value)


# TODO: - Add failsafe in case git is not on the path
#       - Embed commit hash in pip installation
def get_git_hash():
    try:
        # Get odyn path
        package_root = Path(__file__).resolve().parent.parent

        # Get commit hash for that directory
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    except Exception:
        return "unknown-hash"


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
