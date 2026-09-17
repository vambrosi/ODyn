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

CHECK = "\033[1;32m✔\033[0m"
CROSS = "\033[1;31m✘\033[0m"

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


def _acquisition_trials(con: Connection, query: str, params: list = []) -> pd.DataFrame:
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

    # Calls whose output this call read through `latest_output`
    consumed: list[int] = field(default_factory=list)


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
                    get_user(),
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
