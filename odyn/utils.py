from __future__ import annotations

import json
import functools
import subprocess

from dataclasses import dataclass
from enum import Enum
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


def record_call(
    caller: Database | Group, db: Database, func_name: str, params: dict
) -> Optional[int]:
    # Makes sure it will not record self
    params.pop("self", None)

    with db.con as con:
        cur = con.cursor()

        git_commit = get_git_hash()

        query = """
                INSERT INTO method_calls
                    ( group_id
                    , method_name
                    , parameters
                    , git_commit
                    ) VALUES (?, ?, ?, ?);
            """
        cur.execute(
            query,
            [caller.group_id, func_name, json.dumps(params), git_commit],
        )

        call_id = cur.lastrowid
        print(f"{INFO} Recorded method call to db (method_call_id = {call_id}).")

        # Reset method_calls DataFrames
        caller._method_calls = None

        # If not a database, refresh parent database cache
        if caller.group_id != 0:
            db._method_calls = None

        return call_id


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
