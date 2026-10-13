"""
The cluster side: requests from other computers, and the `tick` that applies
them and does what the database is waiting for.

Workstations cannot start cluster jobs (logging in takes a password and Duo),
so anything that has to run there is left as a request file on the share,
and a short `tick` on the cluster applies it. A tick also adds the submitted
drafts whose recordings finished copying (`odyn.app.submissions`), submits a
test motion correction of each experiment it added (`submit_mcor_test`), processes
the sync files that arrived since their experiment or grabs were added, writes a
heartbeat, and (as a chain) submits the next tick. Everything a tick does is
safe to repeat, so a tick that ran twice, or not at all for a day, only
delays work.

Files, per database (`main` or the project's name), in
`<main_folder>/.odyn/cluster/<name>/`:
- `requests/` holds the requests waiting, `requests/done/` the applied ones
and `requests/rejected/` the refused ones, each with the reason;
- `submissions/` holds the submitted drafts not added yet, each with its
state, and `submissions/done/` the ones added;
- `heartbeat.json` says when the last tick ran and what it did, and
`tick.log` keeps a line per tick;
- `seen.json` is what each sync folder looked like when a tick last tried it;
- `jobs/` holds the job scripts and their output;
- `mcor/` has a file per group whose test motion correction was submitted
(`<group_id>.json`: the job, who it is for, when), so no group is tested twice
by the tick;
- `STOP` stops the chain (the running tick does not submit the next one).

The cluster's own settings (partition, conda environment, chain interval) are
in `<main_folder>/.odyn/cluster.json`, see `ClusterConfig`.

**COMMAND LINE**
```
python -m odyn.cluster <main_folder> [--project NAME] tick [--dry-run]
python -m odyn.cluster <main_folder> start [--every MINUTES] [--dry-run]
python -m odyn.cluster <main_folder> stop
python -m odyn.cluster <main_folder> status
python -m odyn.cluster <main_folder> request process_sync exp_id=12
python -m odyn.cluster <main_folder> mcor-test --group 7 [--dry-run]
```

**PYTHON**
```python
from odyn.cluster import send_request, tick

send_request(main_folder, "process_sync", project="main-sync", exp_id=12)
report = tick(main_folder, project="main-sync")
```
"""

from __future__ import annotations

import argparse
import getpass
import inspect
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import uuid

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .database import SYNC_FOLDER, Database, SyncFlag
from .utils import ODYN_FOLDER, database_path, get_user, logger

CLUSTER_FOLDER = "cluster"
CONFIG_FILE = "cluster.json"
REQUESTS_FOLDER = "requests"
DONE_FOLDER = "done"
REJECTED_FOLDER = "rejected"
JOBS_FOLDER = "jobs"
HEARTBEAT_FILE = "heartbeat.json"
LOG_FILE = "tick.log"
STOP_FILE = "STOP"
SEEN_FILE = "seen.json"
MCOR_FOLDER = "mcor"

# What a request may ask for. A request is executed by name, so this list is
# the whole of what another computer can make the cluster do.
REQUESTABLE = ("process_sync", "process_grabs_sync")

# A sync file this recent may still be arriving (one copied by hand has no
# temporary name), so it waits for the next tick
SETTLE_S = 300

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Where things are
# --------------------------------------------------------------------------- #


def cluster_folder(main_folder: str | Path, project: None | str = None) -> Path:
    """The folder of one database's requests, heartbeat and jobs."""
    database_path(main_folder, project)  # checks the project name

    return Path(main_folder) / ODYN_FOLDER / CLUSTER_FOLDER / (project or "main")


@dataclass
class ClusterConfig:
    """
    How jobs run on the cluster, read from `.odyn/cluster.json` (every key is
    optional; what is left out keeps the default below).

    **EXAMPLE** `cluster.json`
    ```json
    {"partition": "batch", "conda_env": "caiman", "every_minutes": 30}
    ```
    """

    partition: str = "batch"
    conda_sh: str = "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda_env: str = "caiman"
    every_minutes: int = 30
    # The tick submits a test motion correction of each experiment it adds
    test_new_experiments: bool = True
    tick_cpus: int = 1
    tick_memory: str = "8G"
    tick_time: str = "0:30:00"
    mcor_cpus: int = 12
    mcor_memory: str = "96G"
    mcor_time: str = "4:00:00"
    mcor_disk_gb: int = 128  # the node's scratch disk, for caiman's temporary files
    scratch: str = "/mnt/scratch"

    @classmethod
    def read(cls, main_folder: str | Path) -> ClusterConfig:
        path = Path(main_folder) / ODYN_FOLDER / CONFIG_FILE

        if not path.is_file():
            return cls()

        given = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(given) - known)

        if unknown:
            raise ValueError(
                f"Unknown settings in '{path}': {', '.join(unknown)}. The known "
                f"ones are {', '.join(sorted(known))}."
            )

        return cls(**given)


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


def send_request(
    main_folder: str | Path,
    method: str,
    *,
    project: None | str = None,
    user: None | str = None,
    **kwargs: Any,
) -> Path:
    """
    Ask the cluster to run `Database.<method>(**kwargs)` at its next tick, as
    `user` (the computer login if left out). Returns the request file.

    **EXAMPLE**
    ```python
    send_request(main_folder, "process_sync", project="main-sync", exp_id=12)
    ```

    Raises `ValueError` for a method that cannot be requested (see
    `REQUESTABLE`) or arguments it does not take, so a bad request is refused
    here rather than at the tick.
    """
    _check_request(method, kwargs)
    folder = cluster_folder(main_folder, project) / REQUESTS_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    request = {
        "method": method,
        "kwargs": kwargs,
        "user": user or get_user(),
        "requested_at": datetime.now().strftime(TIME_FORMAT),
        "requested_on": socket.gethostname(),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}-{method}-{uuid.uuid4().hex[:8]}.json"

    _write_json(path, request)  # a tick never reads half a file

    return path


def pending_requests(main_folder: str | Path, project: None | str = None) -> list[Path]:
    """The requests waiting for a tick, oldest first."""
    folder = cluster_folder(main_folder, project) / REQUESTS_FOLDER

    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.glob("*.json") if p.is_file() and not p.name.startswith(".")
    )


def _check_request(method: str, kwargs: dict[str, Any]) -> None:
    if method not in REQUESTABLE:
        raise ValueError(
            f"'{method}' cannot be requested. Requests can ask for "
            f"{', '.join(REQUESTABLE)}."
        )

    try:
        inspect.signature(getattr(Database, method)).bind(None, **kwargs)
    except TypeError as error:
        raise ValueError(f"Wrong arguments for '{method}' ({error}).") from None


def _apply(
    path: Path, main_folder: Path, project: None | str, report: TickReport
) -> None:
    """Run one request, then move it to `done/` or `rejected/`."""
    try:
        request = json.loads(path.read_text())
        method, kwargs = request["method"], dict(request.get("kwargs") or {})
        user = str(request["user"])
        _check_request(method, kwargs)

    except (OSError, ValueError, KeyError, TypeError) as error:
        _file_away(path, REJECTED_FOLDER, {"reason": str(error)})
        report.rejected.append(f"{path.name}: {error}")
        logger.warning(f"Rejected the request '{path.name}': {error}")
        return

    # Recorded as the person who asked, not as the account running the tick
    db = Database(main_folder, project=project, user=user)
    outcome: dict[str, Any] = {"applied_at": datetime.now().strftime(TIME_FORMAT)}

    try:
        getattr(db, method)(**kwargs)
        outcome["error"] = None

    except Exception as error:  # the call recorded it; the request is still done
        outcome["error"] = f"{type(error).__name__}: {error}"
        report.errors.append(f"{path.name}: {outcome['error']}")

    _file_away(path, DONE_FOLDER, outcome)
    report.applied.append(f"{method}({_arguments(kwargs)}) for {user}")


def _file_away(path: Path, folder: str, extra: dict[str, Any]) -> None:
    """Move a request into `folder`, with `extra` added to it when readable."""
    target = path.parent / folder
    target.mkdir(exist_ok=True)

    try:
        content = json.loads(path.read_text())
        content.update(extra)
        (target / path.name).write_text(json.dumps(content, indent=2, default=str))
        path.unlink()

    except (OSError, ValueError):
        os.replace(path, target / path.name)


def _arguments(kwargs: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in kwargs.items())


# --------------------------------------------------------------------------- #
# The tick
# --------------------------------------------------------------------------- #


@dataclass
class TickReport:
    """What one tick did (or, in a dry run, would do)."""

    started_at: str = ""
    dry_run: bool = False
    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    submissions: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    processed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_tick: None | str = None

    def summary(self) -> str:
        """One line for `tick.log`."""
        parts = [
            f"{len(self.applied)} requests applied",
            f"{len(self.rejected)} rejected",
            f"{len(self.submissions)} submissions looked at",
            f"{len(self.tests)} tests submitted",
            f"{len(self.processed)} sync files processed",
            f"{len(self.errors)} errors",
        ]

        if self.next_tick:
            parts.append(f"next tick {self.next_tick}")

        return f"{self.started_at}  " + ", ".join(parts)


def tick(
    main_folder: str | Path,
    project: None | str = None,
    *,
    dry_run: bool = False,
    chain: bool = False,
    settle_s: float = SETTLE_S,
) -> TickReport:
    """
    One round of the cluster's work: apply the waiting requests, add the
    submitted drafts whose recordings are complete (see
    `odyn.app.submissions`), submit a test motion correction of each experiment
    they added (unless `test_new_experiments` is off in `ClusterConfig`), process
    the sync files that arrived (or changed) since they were last processed,
    write the heartbeat and, with `chain`, submit the next tick.

    **PARAMETERS**
    - `dry_run` lists what would be done and changes nothing
    - `chain` submits the next tick (see `start_chain`) unless `STOP` exists
    - `settle_s`: sync files modified more recently than this wait for the
    next tick, since they may still be copying

    Returns a `TickReport`. A request, a test or a sync file that fails is
    reported and the tick carries on with the rest (a test that could not be
    submitted is tried again by the next tick).
    """
    main_folder = Path(main_folder).resolve()
    folder = cluster_folder(main_folder, project)
    config = ClusterConfig.read(main_folder)
    report = TickReport(
        started_at=datetime.now().strftime(TIME_FORMAT), dry_run=dry_run
    )

    for path in pending_requests(main_folder, project):
        if dry_run:
            report.applied.append(f"would apply {path.name}")
        else:
            _apply(path, main_folder, project, report)

    # Imported here: the queue belongs to the app (no Qt in it)
    from .app.submissions import process_submissions

    report.submissions = process_submissions(main_folder, project, dry_run=dry_run)

    db = Database(main_folder, project=project)

    if config.test_new_experiments:
        _submit_tests(db, project, report, dry_run)

    seen = _read_seen(folder)

    for work in _sync_work(db, settle_s, seen):
        if dry_run:
            report.processed.append(f"would process {work.what}")
            continue

        try:
            work.call()

        except Exception as error:  # recorded on the call; the next file goes on
            report.errors.append(f"{work.what}: {type(error).__name__}: {error}")

        report.processed.append(work.what)
        seen[work.key] = work.state
        _write_json(folder / SEEN_FILE, seen)

    if chain and not dry_run:
        if (folder / STOP_FILE).exists():
            logger.info(f"Not submitting the next tick: '{folder / STOP_FILE}' exists.")
        else:
            submit_tick(main_folder, project, delay_minutes=config.every_minutes)
            report.next_tick = (
                datetime.now() + timedelta(minutes=config.every_minutes)
            ).strftime(TIME_FORMAT)

    if not dry_run:
        _write_heartbeat(folder, report)

    logger.info(report.summary())

    return report


def _submit_tests(
    db: Database, project: None | str, report: TickReport, dry_run: bool
) -> None:
    """Submit a test motion correction of each experiment the queue added untested."""
    for group_id, where in _untested_groups(db, project):
        if dry_run:
            report.tests.append(f"{where}: would submit a test (group {group_id})")
            continue

        try:
            job_id = submit_mcor_test(db.main_folder, project, group_id)
            report.tests.append(
                f"{where}: test submitted (group {group_id}, job {job_id})"
            )

        except Exception as error:  # the next tick tries again
            report.errors.append(
                f"{where}: test of group {group_id}: {type(error).__name__}: {error}"
            )


def _untested_groups(db: Database, project: None | str) -> list[tuple[int, str]]:
    """
    The experiments the queue added (`odyn.app.submissions.added`) that no test was
    submitted for: the group each experiment started as, and the experiment's path.
    """
    # Imported here: the queue belongs to the app (no Qt in it)
    from .app.submissions import added

    exp_ids = {
        int(exp_id)
        for _, state in added(db.main_folder, project)
        for exp_id in state.written.get("experiments", {}).values()
    }

    if not exp_ids:
        return []

    sent = cluster_folder(db.main_folder, project) / MCOR_FOLDER
    links = db.group_experiments
    sizes = links.index.value_counts()
    paths = db.experiments["exp_path"]
    found = {}

    for group_id, exp_id in sorted(zip(links.index, links["exp_id"])):
        exp_id = int(exp_id)

        if exp_id in exp_ids and sizes[group_id] == 1 and exp_id not in found:
            found[exp_id] = int(group_id)

    return [
        (group_id, str(paths.loc[exp_id]))
        for exp_id, group_id in sorted(found.items())
        if not (sent / f"{group_id}.json").exists()
    ]


def _sync_work(db: Database, settle_s: float, seen: dict[str, str]) -> list[Work]:
    """
    The sync files waiting to be processed. An experiment's or a grabs file's turn
    comes when it was never processed, when it changed since its last
    processing, or when that processing found no file (or too many) and the
    folder has changed since. `seen` holds what each folder looked like when
    a tick last tried it, so one state is never tried twice. Files modified in
    the last `settle_s` seconds may still be copying and wait.
    """
    now = time.time()
    work: list[Work] = []
    retry = int(SyncFlag.NO_SYNC_FILE) | int(SyncFlag.MULTIPLE_SYNC_FILES)
    exp_calls = _last_calls(db, "process_sync", "input_exp_id")

    for exp_id, experiment in db.experiments.iterrows():
        files = _sync_files(db.main_folder / str(experiment["exp_path"]) / SYNC_FOLDER)

        if not files or any(now - p.stat().st_mtime < settle_s for p in files):
            continue

        key = f"experiment {int(exp_id)}"
        state = _state(files)
        last = exp_calls.get(int(exp_id))

        if seen.get(key) == state:
            continue

        if last is not None:
            called_at, flag = last
            changed = any(_modified_after(p, called_at) for p in files)

            if not changed and not flag & retry:
                continue

        what = files[0].relative_to(db.main_folder).as_posix()
        work.append(
            Work(what, key, state, lambda i=int(exp_id): db.process_sync(exp_id=i))
        )

    grab_calls = _last_calls(db, "process_grabs_sync", "input_sync_path")
    grabs = db.grabs
    folders = {
        (int(row["session_id"]), Path(str(row["raw_path"])).parent.parent)
        for _, row in grabs.iterrows()
    }

    for session_id, grabs_folder in sorted(folders):
        for path in _sync_files(db.main_folder / grabs_folder / SYNC_FOLDER):
            if now - path.stat().st_mtime < settle_s:
                continue

            rel = path.relative_to(db.main_folder).as_posix()
            key, state = f"grabs {rel}", _state([path])
            last = grab_calls.get(rel)

            if seen.get(key) == state:
                continue

            if last is not None and not _modified_after(path, last[0]):
                continue

            work.append(
                Work(
                    rel,
                    key,
                    state,
                    lambda s=session_id, r=rel: db.process_grabs_sync(
                        session_id=s, sync_path=r
                    ),
                )
            )

    return work


@dataclass
class Work:
    """One sync file (or folder) to process, and how to process it."""

    what: str  # the file, relative to the main folder
    key: str  # its entry in `seen.json`
    state: str  # what the folder looks like now
    call: Any


def _sync_files(folder: Path) -> list[Path]:
    # Hidden names are copies in progress or recordings replaced by a newer one
    if not folder.is_dir():
        return []

    return sorted(p for p in folder.glob("*.h5") if not p.name.startswith("."))


def _state(files: list[Path]) -> str:
    """The names, sizes and times of `files`, to tell when they change."""
    return "; ".join(
        f"{p.name} {p.stat().st_size} {p.stat().st_mtime_ns}" for p in files
    )


def _last_calls(db: Database, method: str, key: str) -> dict[Any, tuple]:
    """When `method` last ran for each value of its input `key`, and its flag."""
    calls = db.latest_calls(method)

    if calls.empty or key not in calls:
        return {}

    last: dict[Any, tuple] = {}

    for _, call in calls.iterrows():  # newest first
        value = call[key]

        if pd.isna(value):
            continue

        value = int(value) if key.endswith("_id") else str(value)
        flag = 0 if pd.isna(call["call_flag"]) else int(call["call_flag"])
        last.setdefault(value, (pd.Timestamp(call["called_at"]), flag))

    return last


def _modified_after(path: Path, when: pd.Timestamp) -> bool:
    return datetime.fromtimestamp(path.stat().st_mtime) > when.to_pydatetime()


def _read_seen(folder: Path) -> dict[str, str]:
    try:
        seen = json.loads((folder / SEEN_FILE).read_text())
    except (OSError, ValueError):
        return {}

    return seen if isinstance(seen, dict) else {}


def _write_json(path: Path, content: Any) -> None:
    """Write `content` under a hidden name, then rename (readers see all or none)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden = path.parent / f".{path.name}.partial"
    hidden.write_text(json.dumps(content, indent=2, default=str))
    os.replace(hidden, path)


def _write_heartbeat(folder: Path, report: TickReport) -> None:
    beat = {
        **asdict(report),
        "host": socket.gethostname(),
        "user": get_user(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _write_json(folder / HEARTBEAT_FILE, beat)

    with open(folder / LOG_FILE, "a") as log:
        log.write(report.summary() + "\n")


def heartbeat(main_folder: str | Path, project: None | str = None) -> None | dict:
    """What the last tick wrote, or `None` if no tick has run."""
    path = cluster_folder(main_folder, project) / HEARTBEAT_FILE

    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #


def job_name(project: None | str = None) -> str:
    """The Slurm job name of a database's tick (one at a time, by `singleton`)."""
    return f"odyn-tick-{project or 'main'}"


def tick_script(
    main_folder: str | Path, project: None | str, config: ClusterConfig
) -> str:
    """The job script of one chained tick."""
    main_folder = Path(main_folder).resolve()
    logs = cluster_folder(main_folder, project) / JOBS_FOLDER
    command = [
        "python",
        "-m",
        "odyn.cluster",
        str(main_folder),
        *(["--project", project] if project else []),
        "tick",
        "--chain",
    ]

    return f"""#!/bin/bash
#SBATCH --partition={config.partition}
#SBATCH --job-name={job_name(project)}
#SBATCH --output={logs}/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.tick_cpus}
#SBATCH --mem={config.tick_memory}
#SBATCH --time={config.tick_time}

# One odyn tick, which submits the next one (python -m odyn.cluster ... stop
# ends the chain).

set -euo pipefail

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

source "{config.conda_sh}"
set +u
conda activate {config.conda_env}
set -u

echo "tick     ${{SLURM_JOB_ID:-none}} on $(hostname) at $(date --iso-8601=seconds)"
{shlex.join(command)}
"""


def submit_tick(
    main_folder: str | Path,
    project: None | str = None,
    *,
    delay_minutes: int = 0,
    dry_run: bool = False,
) -> None | str:
    """
    Write the tick's job script to `jobs/` and submit it with `sbatch`, to
    start `delay_minutes` from now. `--dependency=singleton` keeps two ticks
    of one database from running at once. Returns the job id (`None` in a
    dry run, which only writes and shows the script).
    """
    config = ClusterConfig.read(main_folder)
    jobs = cluster_folder(main_folder, project) / JOBS_FOLDER
    jobs.mkdir(parents=True, exist_ok=True)
    script = jobs / "tick.sh"
    script.write_text(tick_script(main_folder, project, config))
    command = [
        "sbatch",
        "--parsable",
        "--dependency=singleton",
        *([f"--begin=now+{delay_minutes}minutes"] if delay_minutes else []),
        str(script),
    ]

    if dry_run:
        logger.info(f"Would run: {shlex.join(command)}")
        print(script.read_text())
        return None

    done = subprocess.run(command, capture_output=True, text=True)

    if done.returncode != 0:
        raise RuntimeError(
            f"sbatch failed: {done.stderr.strip() or done.stdout.strip()}"
        )

    job_id = done.stdout.strip().split(";")[0]
    logger.info(f"Submitted the tick as job {job_id}.")

    return job_id


def mcor_test_script(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    user: str,
    config: ClusterConfig,
) -> str:
    """The job script of one test motion correction (see `submit_mcor_test`)."""
    main_folder = Path(main_folder).resolve()
    logs = cluster_folder(main_folder, project) / JOBS_FOLDER
    command = [
        "python",
        "-m",
        "odyn.cluster",
        str(main_folder),
        *(["--project", project] if project else []),
        "run-mcor-test",
        "--group",
        str(group_id),
        "--user",
        user,
    ]

    return f"""#!/bin/bash
#SBATCH --partition={config.partition}
#SBATCH --job-name=odyn-mcor-{project or 'main'}-g{group_id}
#SBATCH --output={logs}/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.mcor_cpus}
#SBATCH --mem={config.mcor_memory}
#SBATCH --time={config.mcor_time}
#SBATCH --gres=disk:{config.mcor_disk_gb}

# A test motion correction of group {group_id}, for {user}. caiman's temporary
# files go to the node's scratch disk, removed however the job ends.

set -euo pipefail

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export CAIMAN_TEMP="{config.scratch}/${{SLURM_JOB_ID}}"
mkdir -p "$CAIMAN_TEMP"
trap 'rm -rf "$CAIMAN_TEMP"' EXIT

source "{config.conda_sh}"
set +u
conda activate {config.conda_env}
set -u

echo "mcor test of group {group_id} as job ${{SLURM_JOB_ID:-none}} on $(hostname)"
echo "cores    ${{SLURM_CPUS_PER_TASK:-?}}, temporary files in $CAIMAN_TEMP"
echo "started  $(date --iso-8601=seconds)"
{shlex.join(command)}
echo "finished $(date --iso-8601=seconds)"
"""


def submit_mcor_test(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    *,
    user: None | str = None,
    dry_run: bool = False,
) -> None | str:
    """
    Submit a test motion correction of `group_id` (`Group.test_motion_correction`)
    as its own job, recorded as `user`'s: by default whoever added the group's
    experiment. Writes the script to `jobs/` and what was submitted to
    `mcor/<group_id>.json`. Returns the job id (`None` in a dry run, which only
    writes and shows the script).
    """
    if user is None:
        user = experiment_owner(Database(main_folder, project=project), group_id)

    config = ClusterConfig.read(main_folder)
    folder = cluster_folder(main_folder, project)
    script = folder / JOBS_FOLDER / f"mcor-test-g{group_id}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(mcor_test_script(main_folder, project, group_id, user, config))
    command = ["sbatch", "--parsable", str(script)]

    if dry_run:
        logger.info(f"Would run: {shlex.join(command)}")
        print(script.read_text())
        return None

    done = subprocess.run(command, capture_output=True, text=True)

    if done.returncode != 0:
        raise RuntimeError(
            f"sbatch failed: {done.stderr.strip() or done.stdout.strip()}"
        )

    job_id = done.stdout.strip().split(";")[0]
    _write_json(
        folder / MCOR_FOLDER / f"{group_id}.json",
        {
            "group_id": group_id,
            "job_id": job_id,
            "user": user,
            "submitted_at": datetime.now().strftime(TIME_FORMAT),
        },
    )
    logger.info(
        f"Submitted the test motion correction of group {group_id} as job {job_id}."
    )

    return job_id


def run_mcor_test(
    main_folder: str | Path, project: None | str, group_id: int, user: str
) -> Path:
    """What the job runs: the test, recorded as `user`'s. Returns its folder."""
    db = Database(main_folder, project=project, user=user)

    return db.groups[group_id].test_motion_correction()


def experiment_owner(db: Database, group_id: int) -> str:
    """
    Who added the experiment of `group_id` (the user of its `add_experiment`
    call), which is who its tests and their QC belong to.
    """
    exp_ids = set(db.group_experiments.loc[[group_id], "exp_id"].astype(int))
    calls = db.latest_calls("add_experiment")

    for _, call in calls.iterrows():  # newest first
        added = call.get("output_exp_ids")

        if isinstance(added, list) and exp_ids & {int(i) for i in added}:
            return str(call["user"])

    raise ValueError(
        f"No add_experiment call added the experiment of group {group_id}."
    )


def mcor_tests(main_folder: str | Path, project: None | str = None) -> list[dict]:
    """
    The submitted tests (`mcor/<group_id>.json`), newest first, each with its
    `state` and `log` (the job's output file, or `None` before it starts):

    - "pending" or "running" (or another Slurm state) while in Slurm's queue
    - "finished" once its log ends with the `finished` line
    - "stopped" when the job ended without it (the log says why)
    - "unknown" without `squeue` (off the cluster)
    """
    folder = cluster_folder(main_folder, project)
    in_queue = _slurm_jobs()
    tests = []

    for path in (folder / MCOR_FOLDER).glob("*.json"):
        test = json.loads(path.read_text())
        name = f"odyn-mcor-{project or 'main'}-g{test['group_id']}"
        log = folder / JOBS_FOLDER / f"{name}-{test['job_id']}.out"
        test["log"] = log if log.is_file() else None

        if in_queue is not None and str(test["job_id"]) in in_queue:
            test["state"] = in_queue[str(test["job_id"])].lower()
        elif test["log"] is not None and _last_line(log).startswith("finished"):
            test["state"] = "finished"
        elif in_queue is None:
            test["state"] = "unknown"
        else:
            test["state"] = "stopped"

        tests.append(test)

    return sorted(tests, key=lambda t: t["submitted_at"], reverse=True)


def _slurm_jobs() -> None | dict[str, str]:
    """This user's jobs in Slurm's queue and their state; `None` without `squeue`."""
    try:
        done = subprocess.run(
            [
                "squeue",
                "--noheader",
                "--format=%i %T",
                f"--user={getpass.getuser()}",
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    return dict(
        line.split()[:2] for line in done.stdout.splitlines() if len(line.split()) > 1
    )


def _last_line(path: Path) -> str:
    lines = path.read_text(errors="replace").strip().splitlines()

    return lines[-1] if lines else ""


def queued_ticks(project: None | str = None) -> list[str]:
    """The ids of this database's ticks waiting or running in Slurm."""
    done = subprocess.run(
        ["squeue", "--noheader", "--format=%i %T %S", f"--name={job_name(project)}"],
        capture_output=True,
        text=True,
    )

    return [line for line in done.stdout.splitlines() if line.strip()]


def start_chain(
    main_folder: str | Path, project: None | str = None, *, dry_run: bool = False
) -> None | str:
    """
    Start the chain of ticks, each submitting the next `every_minutes` later.
    Removes `STOP` and submits a tick now, unless one is already queued.
    """
    folder = cluster_folder(main_folder, project)

    if not dry_run:
        queued = queued_ticks(project)

        if queued:
            logger.info(f"A tick is already queued: {'; '.join(queued)}")
            return None

        (folder / STOP_FILE).unlink(missing_ok=True)

    return submit_tick(main_folder, project, dry_run=dry_run)


def stop_chain(main_folder: str | Path, project: None | str = None) -> None:
    """Stop the chain by writing `STOP` and cancelling the tick waiting to start."""
    folder = cluster_folder(main_folder, project)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / STOP_FILE).write_text(datetime.now().strftime(TIME_FORMAT) + "\n")
    subprocess.run(
        ["scancel", f"--name={job_name(project)}", "--state=PENDING"],
        capture_output=True,
        text=True,
    )
    logger.info("Stopped the chain (a tick already running finishes first).")


def status(main_folder: str | Path, project: None | str = None) -> str:
    """
    The chain's state, as text: last tick, requests, submitted drafts, tests,
    `STOP` and the queue.
    """
    folder = cluster_folder(main_folder, project)
    beat = heartbeat(main_folder, project)
    lines = []

    if beat is None:
        lines.append("Last tick:  never")
    else:
        lines.append(
            f"Last tick:  {beat['started_at']} on {beat['host']} as {beat['user']}"
            + (f" (job {beat['job_id']})" if beat.get("job_id") else "")
        )
        lines.append(f"            {TickReport(**_report_fields(beat)).summary()}")

        for error in beat.get("errors", []):
            lines.append(f"  error     {error}")

    from .app.submissions import queued

    lines.append(f"Requests:   {len(pending_requests(main_folder, project))} waiting")
    states = [state.state for _, state in queued(main_folder, project)]
    lines.append(
        f"Submitted:  {states.count('waiting')} waiting, "
        f"{states.count('failed')} failed"
    )

    for path, state in queued(main_folder, project):
        if state.state == "failed":
            lines.append(f"  failed    {path.name}: {state.message}")

    tests = mcor_tests(main_folder, project)
    counts = {}

    for test in tests:
        counts[test["state"]] = counts.get(test["state"], 0) + 1

    lines.append(
        f"Tests:      {len(tests)} submitted"
        + "".join(f", {n} {state}" for state, n in sorted(counts.items()))
    )

    for test in tests:  # the ones to look at
        if test["state"] != "finished":
            lines.append(
                f"  {test['state']:<9} group {test['group_id']} for {test['user']}, "
                f"job {test['job_id']} ({test['submitted_at']})"
                + (
                    f": see {test['log'].name}"
                    if test["state"] == "stopped" and test["log"]
                    else ""
                )
            )

    rejected = sorted((folder / REQUESTS_FOLDER / REJECTED_FOLDER).glob("*.json"))
    lines.append(f"Rejected:   {len(rejected)} in total")
    lines.append(f"Stopped:    {'yes' if (folder / STOP_FILE).exists() else 'no'}")

    try:
        queued = queued_ticks(project)
        lines.append(f"Queued:     {'; '.join(queued) or 'none'}")
    except FileNotFoundError:  # not on the cluster
        lines.append("Queued:     unknown (no squeue here)")

    return "\n".join(lines)


def _report_fields(beat: dict) -> dict:
    known = {f.name for f in fields(TickReport)}

    return {key: value for key, value in beat.items() if key in known}


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _value(text: str) -> Any:
    """A `key=value` value typed on the command line, read as JSON when it is."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv: None | list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odyn.cluster",
        description="Requests and the tick that applies them on the cluster.",
    )
    parser.add_argument("main_folder")
    parser.add_argument("--project", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "tick",
        help="apply requests, add submitted drafts, test them, process sync files",
    )
    run.add_argument("--dry-run", action="store_true", help="only say what it would do")
    run.add_argument("--chain", action="store_true", help="then submit the next tick")

    start = commands.add_parser("start", help="start the chain of ticks")
    start.add_argument("--dry-run", action="store_true", help="only show the job")

    commands.add_parser("stop", help="stop the chain of ticks")
    commands.add_parser("status", help="last tick, queue and requests")

    test = commands.add_parser(
        "mcor-test", help="submit a test motion correction of a group as a job"
    )
    test.add_argument("--group", type=int, required=True)
    test.add_argument("--user", help="whose it is (default: who added the experiment)")
    test.add_argument("--dry-run", action="store_true", help="only show the job")

    job = commands.add_parser("run-mcor-test", help="what the mcor-test job runs")
    job.add_argument("--group", type=int, required=True)
    job.add_argument("--user", required=True)

    ask = commands.add_parser("request", help="leave a request for the next tick")
    ask.add_argument("method", choices=REQUESTABLE)
    ask.add_argument("arguments", nargs="*", help="key=value")

    args = parser.parse_args(argv)

    if args.command == "tick":
        report = tick(
            args.main_folder, args.project, dry_run=args.dry_run, chain=args.chain
        )

        for line in [
            *report.applied,
            *report.submissions,
            *report.tests,
            *report.processed,
        ]:
            print(f"  {line}")

        for line in report.errors:
            print(f"  error: {line}")

        return 1 if report.errors else 0

    if args.command == "start":
        start_chain(args.main_folder, args.project, dry_run=args.dry_run)

    elif args.command == "stop":
        stop_chain(args.main_folder, args.project)

    elif args.command == "status":
        print(status(args.main_folder, args.project))

    elif args.command == "mcor-test":
        submit_mcor_test(
            args.main_folder,
            args.project,
            args.group,
            user=args.user,
            dry_run=args.dry_run,
        )

    elif args.command == "run-mcor-test":
        folder = run_mcor_test(args.main_folder, args.project, args.group, args.user)
        print(f"Wrote {folder}")

    elif args.command == "request":
        kwargs = {}

        for argument in args.arguments:
            key, equals, value = argument.partition("=")

            if not equals:
                parser.error(f"Arguments are key=value, got {argument!r}.")

            kwargs[key] = _value(value)

        path = send_request(
            args.main_folder, args.method, project=args.project, **kwargs
        )
        print(f"Left {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
