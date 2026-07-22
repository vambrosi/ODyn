from __future__ import annotations

import functools
import json
import logging
import subprocess

from dataclasses import dataclass, field
from enum import Enum, IntEnum, IntFlag
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database
    from .groups import Group
    from datetime import datetime
    from sqlite3 import Connection

import pandas as pd

from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CHECK = "\033[1;32m✔\033[0m"
CROSS = "\033[1;31m✘\033[0m"

ODYN_FOLDER = ".odyn"
INFO_FOLDER = ".odyn/olfactometer/Log/Info"

type BasicTypes = None | bool | int | float | str | datetime
type Object = dict[str, BasicTypes | Object | list[Object]]


# --------------------------------------------------------------------------- #
# Logging
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

        with db.con as con:
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
                    json.dumps(kwargs),
                    get_git_hash(),
                    json.dumps(parameters_used),
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

            call_output = json.dumps(frame.output) if frame.output is not None else None

            with db.con:
                db.con.execute(
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
                        json.dumps(frame.used),
                        call_id,
                    ],
                )

    # NOTE: This is to make memorize_params work
    wrapper.__kwdefaults__ = func.__kwdefaults__
    return wrapper


def _method_calls_dataframe(
    con: Connection, query: str, params: list
) -> pd.DataFrame:
    """
    Run `query` against method_calls and expand its JSON columns into columns.

    Shared body of Database.latest_calls / Group.latest_calls. `query` must
    select from method_calls and return the method_call_id column.

    NOTE: json_normalize raises on None on macOS but yields no columns on
    Windows, so missing JSON is coerced to {} (no columns, every OS).
    """

    df = pd.read_sql_query(query, con, params=params)
    df.set_index("method_call_id", inplace=True)

    df.parameter_inputs = df.parameter_inputs.apply(json.loads)
    df.parameters_used = df.parameters_used.apply(json.loads)
    df.call_output = df.call_output.apply(
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
# Numerical Functions
# --------------------------------------------------------------------------- #


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]


def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))
