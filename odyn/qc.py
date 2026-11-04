"""
What the QC page shows, built where the share is quick to list (the cluster):
the recordings each user submitted, the required annotations still empty,
and each experiment's motion correction tests with their state, numbers and
files. `qc_summary` puts all of it, for every user, in one view file
(`qc.json`, see `write_qc_summary`), so the page reads a few KB instead of
listing hundreds of folders. The page then picks one user's rows and checks
its downloads against the file lists in it (`Listing(seen, scan=False)`).

**USAGE**
```python
write_qc_summary(db)                                   # the tick, a job end
summary = read_qc_summary(main_folder, "main-sync")    # the app
```

No Qt in here: the cluster builds it.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cluster import (
    CORRECTION_RUN,
    ClusterConfig,
    cluster_folder,
    heartbeat,
    latest_approvals,
    latest_picks,
    mcor_sweep_runs,
    mcor_tests,
    pending_requests,
)
from .database import SyncFlag
from .groups import FULL_FOLDER, RAW_PREVIEW_FOLDER, mcor_test_folder
from .locking import file_lock
from .mcor_sweep import run_numbers
from .shared import read_json, write_json
from .utils import jsonable

QC_FILE = "qc.json"

# Bumped when the view file changes shape: an app that finds another version
# builds the view itself instead of misreading it
QC_VERSION = 1

# The calls that add recordings, the kind each adds, and the output key listing
# the ids it added
INGEST_METHODS = {
    "add_experiment": ("experiment", "exp_ids"),
    "add_zstacks": ("zstack", "zstack_ids"),
}

# Where each kind's folder (an experiment) or file (the others) is
PATH_COLUMN = {"experiment": "exp_path", "zstack": "raw_path"}
COLUMNS = ["submitted_at", "kind", "path", "objective", "sync"]

# The missing annotations (`my_missing`)
MISSING_COLUMNS = ["waiting_since", "target_type", "target_id", "key", "label", "user"]
TEST_COLUMNS = [
    "submitted_at",
    "path",
    "run",
    "state",
    "local",
    "corr_with_ref",
    "flow_px",
    "flow_worst_px",
    "jitter_px",
    "at_limit",
    "finished_at",
]

TEST_HIDDEN = [
    "folder",
    "bytes",
    "group_id",
    "experiment_folder",
    "pick_call_id",
    "flow_worst_acq",
    "job",
    "whole",
    "user",
]
# The numbers shown for a test, from `mcor_sweep.run_numbers`: medians over
# the acquisitions, the whole test numbers, and the worst flow (the largest
# 90th percentile among the acquisitions, and which one)
TEST_NUMBERS = {
    "corr_with_ref": "test_corr_with_ref_median",
    "flow_px": "test_flow_median_px_median",
    "flow_worst_px": "test_flow_p90_px_max",
    "flow_worst_acq": "flow_p90_worst_acq_id",
    "jitter_px": "patch_jitter_median_px",
    "at_limit": "frames_at_limit",
}

# A job's state in words for the page ("unknown": no `squeue` here, so the
# job is either still in the cluster's queue or ended without finishing)
TEST_STATES = {"pending": "waiting", "unknown": "not finished"}

# The folders of an experiment's tests downloaded with it, beside the runs'
EXPERIMENT_FOLDERS = (RAW_PREVIEW_FOLDER, FULL_FOLDER)

# The files of a test folder worth having here (as `Path.match` patterns)
SMALL_FILES = (
    "*.avi",
    "*.mp4",
    "shifts.png",
    "reference.tif",
    "flow_average.tif",
    "flow_worst.tif",
    "summary.csv",
    "metrics.json",
    "shifts_summary.npz",
    "*.frames.csv",
)


@dataclass
class Listing:
    """
    One look at each test folder, kept for a while (one build of the QC view):
    on a network share, every question about a file is a round trip, so each
    folder is listed once, with its files' sizes and times, and every later
    question is answered from that. On Windows the listing brings the sizes
    and times along; elsewhere it costs one more call per file, once.

    With `scan=False` it only answers from `seen` (the lists `qc.json` holds),
    a folder not there having no files: the app never lists the share.

    Not for a download: what it copies is looked at then (`cache.pending`).
    """

    seen: dict[str, dict[str, tuple[int, float]]] = field(default_factory=dict)
    scan: bool = True

    def files(self, folder: str | Path) -> dict[str, tuple[int, float]]:
        """Test `folder`'s small files: `{name: (size, modified)}`."""
        key = str(folder)

        if key not in self.seen:
            if not self.scan:
                return {}

            self.seen[key] = _scan(folder)

        return self.seen[key]


def _scan(folder: str | Path) -> dict[str, tuple[int, float]]:
    """A folder's small files, `{name: (size, modified)}`, in one listing."""
    found = {}

    try:
        entries = list(os.scandir(folder))
    except OSError:  # not there (yet)
        return found

    for entry in entries:
        name = entry.name

        if name.startswith(".") or not any(
            Path(name).match(pattern) for pattern in SMALL_FILES
        ):
            continue

        try:
            if entry.is_file():
                info = entry.stat()
                found[name] = (info.st_size, info.st_mtime)
        except OSError:  # gone since the listing
            continue

    return found


def remote_size(folder: str | Path, listing: None | Listing = None) -> int:
    """How many bytes downloading test `folder` copies (all of it)."""
    return sum(size for size, _ in (listing or Listing()).files(folder).values())


def my_recordings(db, user: None | str, everyone: bool = False) -> pd.DataFrame:
    """
    One row per experiment or z-stack `user` submitted (`everyone`: anyone),
    newest first, with its sync flags once its sync file was processed.
    Indexed by `(kind, id)`, as annotations name their target.
    """
    calls = db.calls

    if calls.empty:
        return pd.DataFrame(columns=[*COLUMNS, "user"])

    method = calls["method_name"].map(lambda m: str(m).rsplit(".", 1)[-1])
    mine = calls[
        method.isin(list(INGEST_METHODS)) & (everyone or (calls["user"] == user))
    ]
    added: dict[tuple[str, int], pd.Timestamp] = {}

    for call_id, call in mine.iterrows():
        kind, key = INGEST_METHODS[method[call_id]]
        output = call["call_output"]
        ids = output.get(key, []) if isinstance(output, dict) else []

        for row_id in ids:
            if isinstance(row_id, int) and (kind, row_id) not in added:
                added[(kind, row_id)] = (call["called_at"], call["user"])

    tables = {"experiment": db.experiments, "zstack": db.zstacks}
    flags = _exp_sync_flags(db)
    rows = []

    for (kind, row_id), (submitted, by) in added.items():
        table = tables[kind]

        if row_id not in table.index:
            continue

        processed = flags.get((kind, row_id))
        rows.append(
            {
                "kind": kind,
                "id": row_id,
                "path": table.loc[row_id, PATH_COLUMN[kind]],
                "objective": table.loc[row_id, "objective"],
                "submitted_at": submitted,
                "sync": processed or ("" if kind == "zstack" else "not processed"),
                "user": by,
            }
        )

    frame = pd.DataFrame(rows, columns=["id", *COLUMNS, "user"])
    frame = frame.set_index(["kind", "id"], drop=False).drop(columns="id")

    # Newest first: the rows are in the order of the calls (and of the ids
    # within one), so reversed they break ties of `submitted_at` the same way
    return frame.iloc[::-1].sort_values("submitted_at", ascending=False, kind="stable")


def _exp_sync_flags(db) -> dict[tuple[str, int], str]:
    """Each experiment's latest `process_sync`, decoded or refused."""
    calls = db.calls  # cached, unlike `latest_calls`, which reads every column
    calls = calls[calls["method_name"].str.endswith("process_sync")]
    flags: dict[tuple[str, int], str] = {}

    for _, call in calls.sort_index(ascending=False).iterrows():  # newest first
        exp_id = (call["parameters_used"] or {}).get("exp_id")

        if exp_id is not None and ("experiment", int(exp_id)) not in flags:
            flags[("experiment", int(exp_id))] = sync_flags(int(call["call_flag"]))

    return flags


def sync_flags(call_flag: int) -> str:
    """The names of the bits set on a `process_sync` call, or `succeeded`."""
    names = [flag.name for flag in SyncFlag if call_flag & int(flag) and flag.name]

    if call_flag & 1:
        names.insert(0, "FAILED")

    return ", ".join(names) or "succeeded"


def my_tests(
    db,
    user: None | str,
    mine: pd.DataFrame,
    everyone: bool = False,
    listing: None | Listing = None,
) -> pd.DataFrame:
    """
    The test motion corrections of `user`'s experiments (`everyone`: of every
    user's), each with its `user`: who added the experiment (not who ran a
    job of it since), the newest submitted first (as in `mine`, from
    `my_recordings`): the cluster's test of each ("test") and the runs of
    each sweep (the baseline first), with the job's state and, once a report
    is there, a few of its numbers. `folder` is where the report is, `bytes`
    what downloading it copies, `experiment_folder` where the experiment's
    tests are; `local` is left for the app (`cache.state`). `listing` keeps
    what was seen of each folder (see `Listing`).
    """
    listing = listing or Listing()
    # Whose an experiment's tests are: whoever added it (`mine`), never who
    # ran a job since (a pick made for someone else is recorded as its
    # maker's, the experiment stays its owner's)
    owners = {
        int(row_id): who
        for (kind, row_id), who in mine["user"].items()
        if kind == "experiment"
    }
    jobs = [
        {**test, "run": None} for test in mcor_tests(db.main_folder, db.project)
    ] + mcor_sweep_runs(db.main_folder, db.project)
    jobs.sort(key=lambda job: job["submitted_at"], reverse=True)  # stable
    submitted = {
        int(row_id): when
        for (kind, row_id), when in mine["submitted_at"].items()
        if kind == "experiment"
    }
    links = db.group_experiments
    paths = db.experiments["exp_path"]
    rows = []
    acquisitions = None  # every group's, read once
    tops: dict[int, Path] = {}  # each group's test folder

    for job in jobs:
        group_id = int(job["group_id"])

        if group_id not in links.index:
            continue

        exp_ids = links.loc[[group_id], "exp_id"].astype(int)
        owner = next((owners[i] for i in exp_ids if i in owners), None)

        if owner is None and not everyone:  # not this user's experiment
            continue

        if group_id not in tops:
            acquisitions = acquisitions if acquisitions is not None else db.acquisitions
            tops[group_id] = mcor_test_folder(db, group_id, acquisitions)

        top = tops[group_id]
        folder = top / job["run"] if job["run"] else top
        times = [submitted[i] for i in exp_ids if i in submitted]
        row = {
            "submitted_at": max(times) if times else None,
            "path": ", ".join(str(paths.get(i, i)) for i in exp_ids),
            "run": job["run"]
            or job.get("label")
            or ("rigid (all)" if job.get("whole") else "test"),
            "state": TEST_STATES.get(job["state"], job["state"]),
            "local": "",
            **dict.fromkeys(TEST_NUMBERS),
            "finished_at": "",
            "folder": str(folder),
            "bytes": 0,
            "group_id": group_id,
            "experiment_folder": str(top),
            "pick_call_id": job.get("pick_call_id"),
            # What `Group.run_again` calls it
            "job": job["run"] or CORRECTION_RUN,
            "whole": bool(job.get("whole")),
            "user": owner or job["user"],  # (added before calls were recorded)
        }

        files = listing.files(folder) if job["state"] == "finished" else {}

        if "metrics.json" in files:
            numbers = _run_numbers(folder, files)
            row |= {key: numbers.get(name) for key, name in TEST_NUMBERS.items()}
            written = files["metrics.json"][1]
            row["finished_at"] = datetime.fromtimestamp(written).strftime(
                "%Y-%m-%d %H:%M"
            )
            row["bytes"] = remote_size(folder, listing)

        rows.append(row)

    # By experiment, newest first, keeping the jobs' order within one (stable)
    rows.sort(key=lambda row: str(row["submitted_at"] or ""), reverse=True)

    return pd.DataFrame(rows, columns=[*TEST_COLUMNS, *TEST_HIDDEN])


# A run's numbers, by folder and its files' sizes and times: read once
_NUMBERS: dict[tuple, dict] = {}


def _run_numbers(folder: Path, files: dict[str, tuple[int, float]]) -> dict:
    """`run_numbers` of `folder`, read again only when its files changed."""
    key = (
        str(folder),
        files.get("metrics.json"),
        files.get("summary.csv"),
    )

    if key not in _NUMBERS:
        try:
            _NUMBERS[key] = run_numbers(folder)
        except (OSError, ValueError):
            return {}

    return _NUMBERS[key]


def latest_raw_checks(db) -> dict[int, list[int]]:
    """Each group's latest `check_raw`: the acquisitions it excluded."""
    calls = db.calls

    if calls.empty:
        return {}

    checks = calls[calls["method_name"] == "Group.check_raw"].sort_index()
    latest: dict[int, list[int]] = {}

    for _, call in checks.iterrows():
        output = call["call_output"]

        if isinstance(output, dict):  # not one that failed
            latest[int(call["group_id"])] = list(output.get("excluded", []))

    return latest


def my_missing(db, mine: pd.DataFrame) -> pd.DataFrame:
    """
    The required annotations still empty on `mine`'s recordings, in the order
    of `mine` (newest first), each with `waiting_since`: when its recording
    was submitted, and `user`: who submitted it.
    """
    missing = db.missing_annotations()

    if missing.empty or mine.empty:
        return missing.iloc[0:0].assign(waiting_since=None, user=None)[
            ["waiting_since", *missing.columns, "user"]
        ]

    rank = {target: position for position, target in enumerate(mine.index)}
    order = [
        rank.get((kind, int(row_id)))
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]
    missing = missing.assign(_rank=order).dropna(subset="_rank")
    missing = missing.sort_values("_rank", kind="stable").drop(columns="_rank")
    since = [
        mine["submitted_at"].get((kind, int(row_id)))
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]

    whose = [
        mine["user"].get((kind, int(row_id))) if "user" in mine else None
        for kind, row_id in zip(missing["target_type"], missing["target_id"])
    ]

    return missing.reset_index(drop=True).assign(waiting_since=since, user=whose)[
        ["waiting_since", *missing.columns, "user"]
    ]


# --------------------------------------------------------------------------- #
# The view file
# --------------------------------------------------------------------------- #


def qc_summary(db) -> dict:
    """
    Everything the QC page shows, for every user, as JSON: the recordings and
    the missing annotations (each row with its `user`), the test rows (see
    `my_tests`), each group's latest pick, raw check and approval, the users'
    names, what the cluster did last, and the file list of every folder a
    download takes (`files`, a `Listing.seen`). Reads the database and lists
    the share: quick on the cluster, slow over a network.
    """
    listing = Listing()
    mine = my_recordings(db, None, everyone=True)
    tests = my_tests(db, None, mine, everyone=True, listing=listing)

    # Every folder an experiment's download can take, listed now, so the app
    # can tell what is there without listing the share itself
    for top in tests["experiment_folder"].unique():
        for folder in (Path(top), *(Path(top) / name for name in EXPERIMENT_FOLDERS)):
            listing.files(folder)

    users = db.users
    main = db.main_folder

    # Paths relative to the main folder, which each machine mounts elsewhere
    for column in ("folder", "experiment_folder"):
        tests[column] = [_relative(path, main) for path in tests[column]]

    return jsonable(
        {
            "version": QC_VERSION,
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recordings": mine.assign(id=[i for _, i in mine.index]).to_dict("records"),
            "missing": my_missing(db, mine).to_dict("records"),
            "tests": tests.to_dict("records"),
            "picks": latest_picks(db),
            "raw_checks": latest_raw_checks(db),
            "approvals": latest_approvals(db),
            "users": dict(zip(users["user_name"], users["display_name"])),
            "heartbeat": heartbeat(db.main_folder, db.project),
            "requests_waiting": len(pending_requests(db.main_folder, db.project)),
            "every_minutes": ClusterConfig.read(db.main_folder).every_minutes,
            "files": {
                _relative(key, main): files for key, files in listing.seen.items()
            },
        }
    )


def _relative(path: str | Path, main_folder: Path) -> str:
    return Path(path).relative_to(main_folder).as_posix()


def qc_path(main_folder: str | Path, project: None | str = None) -> Path:
    """Where the QC view of a database is."""
    return cluster_folder(main_folder, project) / QC_FILE


def write_qc_summary(db) -> Path:
    """Build the QC view (`qc_summary`) and put it in place, for the app."""
    path = qc_path(db.main_folder, db.project)

    # The tick and each job that ends write it: built under the lock too, so
    # an older build never lands after a newer one
    with file_lock(path):
        return write_json(path, qc_summary(db))


def read_qc_summary(main_folder: str | Path, project: None | str = None) -> None | dict:
    """
    The QC view, or `None` when there is none (no cluster run yet) or it is of
    another version (see `QC_VERSION`); the app then builds it itself.
    """
    try:
        summary = read_json(qc_path(main_folder, project))
    except (OSError, ValueError):
        return None

    return summary if summary.get("version") == QC_VERSION else None


def cluster_line(summary: dict, now: None | datetime = None) -> tuple[str, str]:
    """
    When the cluster last ran its tick, as `(text, kind)` for the QC page. The
    kind is `warning` when the cluster looks stopped (no tick for twice its
    interval), else `computed`.
    """
    beat = summary.get("heartbeat")
    waiting = int(summary.get("requests_waiting") or 0)
    plural = "" if waiting == 1 else "s"
    tail = "" if not waiting else f", {waiting} request{plural} waiting"

    if beat is None:
        return f"The cluster has not run yet{tail}", "computed"

    started = datetime.strptime(beat["started_at"], "%Y-%m-%d %H:%M:%S")
    minutes = ((now or datetime.now()) - started).total_seconds() / 60
    every = int(summary.get("every_minutes") or 30)
    ago = f"{minutes:.0f} min" if minutes < 90 else f"{minutes / 60:.0f} h"
    text = f"The cluster last ran at {beat['started_at'][:16]} ({ago} ago){tail}"

    if minutes > 2 * every:
        return f"{text}; it may have stopped", "warning"

    return text, "computed"
