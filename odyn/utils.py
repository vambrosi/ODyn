from __future__ import annotations

import contextlib
import functools
import json
import subprocess
import sys

from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database
    from .groups import Group

INFO = "[\033[1;34mINFO\033[0m]"
PASS = "[\033[1;32mTEST\033[0m]"
FAIL = "[\033[1;31mTEST\033[0m]"
WARNING = "[\033[1;33mWARNING\033[0m]"
CHECK = "\033[1;32m\u2714\033[0m"
CROSS = "\033[1;31m\u2718\033[0m"

ODYN_FOLDER = ".odyn"
INFO_FOLDER = ".odyn/olfactometer/Log/Info"


class TrialPhase(Enum):
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


class _Tee:
    """Writes to both the real stdout and a capture buffer simultaneously."""

    def __init__(self, real, buf: StringIO):
        self._real = real
        self._buf = buf

    def write(self, data: str) -> int:
        n = self._real.write(data)
        self._buf.write(data)
        return n

    def flush(self) -> None:
        self._real.flush()
        self._buf.flush()


def record_call(func):
    """
    Decorator for Database/Group methods that should be tracked in method_calls.

    Records the call, captures all print output during execution, and saves it
    to call_log when the method returns (even on exception).

    Sets self._current_call_id for the duration of the call so the function
    body can reference its own method_call_id (e.g. to store in a foreign key).
    """

    @functools.wraps(func)
    def wrapper(self, **kwargs):
        # Support both Database (db = self) and Group (db = self.db)
        db = getattr(self, "db", self)

        buf = StringIO()
        tee = _Tee(sys.stdout, buf)

        with contextlib.redirect_stdout(tee):
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

            print(f"{INFO} Recorded method call to db (method_call_id = {call_id}).")

            # Reset method_calls caches
            self._method_calls = None
            if self.group_id != 0:
                db._method_calls = None

            self._current_call_id = call_id
            try:
                return func(self, **kwargs)
            finally:
                del self._current_call_id
                with db.con:
                    db.con.execute(
                        "UPDATE method_calls SET call_log = ? WHERE method_call_id = ?",
                        [buf.getvalue(), call_id],
                    )

    return wrapper


# TODO: - Fix the logging interaction with this class
@dataclass
class ProgressBar:
    total: int
    current: int = 0

    def show(self) -> None:
        filled_squares = int(40 * self.current / self.total)

        progress_str = f"{INFO} [ \033[1;34m"
        progress_str += "\u2588" * filled_squares
        progress_str += "-" * (40 - filled_squares)
        progress_str += f"\033[0m ] {self.current:03d}/{self.total:03d} Files processed"

        print(progress_str, end="\r")

    def message(self, msg: str) -> None:
        print(f"{msg:<120}")
        self.show()

    def step(self) -> None:
        self.current = min(self.current + 1, self.total)
        self.show()

    def end(self, msg: Optional[str] = None) -> None:
        if msg is None:
            msg = f"{INFO} {self.total} files processed sucessfully."

        print(f"{msg:<120}")


def um_to_pixels(values_um, um_per_pixels):
    return [int(a / b) for (a, b) in zip(values_um, um_per_pixels)]


def clamp(x, min_x, max_x):
    return max(min_x, min(x, max_x))
