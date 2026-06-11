from __future__ import annotations

import functools
import json
import logging
import subprocess

from enum import Enum, IntEnum
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database
    from .groups import Group

from tqdm.auto import tqdm

CHECK = "\033[1;32m✔\033[0m"
CROSS = "\033[1;31m✘\033[0m"

ODYN_FOLDER = ".odyn"
INFO_FOLDER = ".odyn/olfactometer/Log/Info"


class TrialPhase(IntEnum):
    NOT_IN_TRIAL = 0
    TRIAL_START = 1
    ODOR_WINDOW = 2
    INTERVAL = 3
    RESPONSE_WINDOW = 4
    TRIAL_END = 5


class MovieType(Enum):
    RAW = "raw"
    MCOR = "mcor"
    TEST = "test"


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

    Pushes the call_id onto self._call_id_stack for the duration of the call,
    so the function body can read self.current_call_id to get its own
    method_call_id (e.g. to store in a foreign key). Supports nested calls.
    """

    @functools.wraps(func)
    def wrapper(self, **kwargs):
        # Support both Database (db = self) and Group (db = self.db)
        db = getattr(self, "db", self)

        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_plain_formatter)
        logger.addHandler(handler)

        with db.con as con:
            cur = con.cursor()
            cur.execute(
                """
                INSERT INTO method_calls
                    ( group_id
                    , method_name
                    , parameters
                    , git_commit
                    ) VALUES (?, ?, ?, ?);
                """,
                [
                    self.group_id,
                    f"{type(self).__name__}.{func.__name__}",
                    json.dumps(kwargs),
                    get_git_hash(),
                ],
            )
            call_id = cur.lastrowid

        logger.info(f"Recorded method call to db (method_call_id = {call_id}).")

        # Reset method_calls caches
        self._method_calls = None
        if self.group_id != 0:
            db._method_calls = None

        self._call_id_stack.append(call_id)

        try:
            return func(self, **kwargs)

        finally:
            self._call_id_stack.pop()
            logger.removeHandler(handler)

            with db.con:
                db.con.execute(
                    "UPDATE method_calls SET call_log = ? WHERE method_call_id = ?",
                    [buf.getvalue(), call_id],
                )

    return wrapper


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]


def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))
