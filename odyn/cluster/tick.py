"""
The tick: one round of the cluster's work, the chain of ticks, the
heartbeat, and `status`.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ..database import SYNC_FOLDER, Database, SyncFlag
from ..shared import FAILED_FOLDER, read_json, write_json
from ..utils import get_user, logger
from .config import (
    ClusterConfig,
    HEARTBEAT_FILE,
    JOBS_FOLDER,
    LOG_FILE,
    MCOR_FOLDER,
    SEEN_FILE,
    SETTLE_S,
    SHARE_UMASK,
    STOP_FILE,
    TIME_FORMAT,
    TickReport,
    cluster_folder,
)
from .mcor import (
    _sweep_sent,
    submit_mcor_sweep,
    submit_mcor_test,
    submit_picks,
    submit_reruns,
    sweep_grid_for,
)
from .messages import _apply, apply_claims, pending_requests, requests
from .records import (
    archive_approved,
    group_objective,
    mcor_sweep_runs,
    mcor_tests,
    record_job_states,
    test_name,
)
from .slurm import _sbatch, job_name, queued_ticks, stop_reason


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
    `odyn.drafts.submissions`), submit a test motion correction of each
    experiment they added (unless `test_new_experiments` is off in `ClusterConfig`; a
    whole one, which saves every acquisition, with `whole_new_experiments`)
    and a sweep of it (with `sweep_grid`), submit a whole correction with each
    test picked in the app (`submit_picks`), keep the numbers of the tests
    of experiments approved a while ago and delete the rest
    (`archive_approved`), process the sync files that
    arrived (or changed) since they were last processed, write the heartbeat
    and, with `chain`, submit the next tick.

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

    if not dry_run:
        apply_claims(main_folder, project, report)

    # Imported here: it imports this module
    from ..drafts.submissions import process_submissions

    report.submissions = process_submissions(main_folder, project, dry_run=dry_run)

    db = Database(main_folder, project=project)

    _submit_tests(db, project, report, dry_run, config)

    submit_picks(db, project, report, dry_run)
    submit_reruns(db, project, report, dry_run)
    archive_approved(db, report, dry_run, config.keep_tests_days)

    if not dry_run:
        record_job_states(main_folder, project)

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
        write_json(folder / SEEN_FILE, seen)

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
        write_qc_view(db)

    logger.info(report.summary())

    return report


def write_qc_view(db: Database) -> None:
    """
    Rebuild the QC page's view (`odyn.qc`). A failure is logged, never raised:
    the page then shows the view before, and the next tick tries again.
    """
    from ..qc import write_qc_summary  # it imports this module

    try:
        write_qc_summary(db)
    except Exception as error:
        logger.warning(f"Could not write the QC view: {type(error).__name__}: {error}")


def _submit_tests(
    db: Database,
    project: None | str,
    report: TickReport,
    dry_run: bool,
    config: ClusterConfig,
) -> None:
    """
    Submit a test motion correction of each experiment the queue added untested
    (a whole one, which corrects and saves every acquisition, with
    `whole_new_experiments`), and a sweep of it with `sweep_grid`, once each.
    """
    whole = config.whole_new_experiments
    what = "whole test" if whole else "test"
    folder = cluster_folder(db.main_folder, project)

    for group_id, where in _added_groups(db, project):
        sent = folder / MCOR_FOLDER / f"{group_id}.json"

        if config.test_new_experiments and not sent.exists():
            if dry_run:
                report.tests.append(
                    f"{where}: would submit a {what} (group {group_id})"
                )
            else:
                try:
                    job_id = submit_mcor_test(
                        db.main_folder, project, group_id, whole=whole
                    )
                    report.tests.append(
                        f"{where}: {what} submitted (group {group_id}, job {job_id})"
                    )

                except Exception as error:  # the next tick tries again
                    report.errors.append(
                        f"{where}: test of group {group_id}: "
                        f"{type(error).__name__}: {error}"
                    )

        grid = sweep_grid_for(config, group_objective(db, group_id))

        if grid and not _sweep_sent(folder, group_id):
            if dry_run:
                report.tests.append(f"{where}: would submit a sweep (group {group_id})")
                continue

            try:
                jobs = submit_mcor_sweep(
                    db.main_folder,
                    project,
                    group_id,
                    grid,
                    time=config.sweep_time,
                )
                report.tests.append(
                    f"{where}: sweep submitted (group {group_id}, jobs "
                    f"{' and '.join(str(job) for job in jobs or ())})"
                )

            except Exception as error:  # the next tick tries again
                report.errors.append(
                    f"{where}: sweep of group {group_id}: "
                    f"{type(error).__name__}: {error}"
                )


def _added_groups(db: Database, project: None | str) -> list[tuple[int, str]]:
    """
    The experiments the queue added (`odyn.drafts.submissions.added`): the
    group each experiment started as, and the experiment's path.
    """
    # Imported here: the queue belongs to the app (no Qt in it)
    from ..drafts.submissions import added

    exp_ids = {
        int(exp_id)
        for _, state in added(db.main_folder, project)
        for exp_id in state.written.get("experiments", {}).values()
    }

    if not exp_ids:
        return []

    links = db.group_experiments
    sizes = links.index.value_counts()
    paths = db.experiments["exp_path"]
    found = {}

    for group_id, exp_id in sorted(zip(links.index, links["exp_id"])):
        exp_id = int(exp_id)

        if exp_id in exp_ids and sizes[group_id] == 1 and exp_id not in found:
            found[exp_id] = int(group_id)

    return [
        (group_id, str(paths.loc[exp_id])) for exp_id, group_id in sorted(found.items())
    ]


def _sync_work(db: Database, settle_s: float, seen: dict[str, str]) -> list[Work]:
    """
    The sync files waiting to be processed. An experiment's file's turn comes
    when it was never processed, when it changed since its last
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
        seen = read_json(folder / SEEN_FILE)
    except (OSError, ValueError):
        return {}

    return seen if isinstance(seen, dict) else {}


def _write_heartbeat(folder: Path, report: TickReport) -> None:
    beat = {
        **asdict(report),
        "host": socket.gethostname(),
        "user": get_user(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(folder / HEARTBEAT_FILE, beat)

    with open(folder / LOG_FILE, "a") as log:
        log.write(report.summary() + "\n")


def heartbeat(main_folder: str | Path, project: None | str = None) -> None | dict:
    """What the last tick wrote, or `None` if no tick has run."""
    path = cluster_folder(main_folder, project) / HEARTBEAT_FILE

    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


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

# What the job writes on the share is the lab's: writable by the group (the
# lab's SMB access goes through it), not by others
umask {SHARE_UMASK:04o}

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
    job_id = _sbatch(
        [
            "--dependency=singleton",
            *([f"--begin=now+{delay_minutes}minutes"] if delay_minutes else []),
            str(script),
        ],
        dry_run,
    )

    if job_id is None:
        print(script.read_text())
        return None

    logger.info(f"Submitted the tick as job {job_id}.")

    return job_id


def _job_lines(
    title: str, jobs: list[dict], name: Callable[[dict], str], what: str = "submitted"
) -> list[str]:
    """
    For `status`: how many `jobs` (tests or sweep runs) in each state, then a
    line for each not finished, by group and name, with why it stopped.
    """
    counts: dict[str, int] = {}

    for job in jobs:
        counts[job["state"]] = counts.get(job["state"], 0) + 1

    lines = [
        f"{title:<12}{len(jobs)} {what}"
        + "".join(f", {n} {state}" for state, n in sorted(counts.items()))
    ]

    for job in jobs:  # the ones to look at
        if job["state"] == "finished":
            continue

        line = (
            f"  {job['state']:<9} group {job['group_id']} {name(job)} for "
            f"{job['user']}, job {job['job_id']} ({job['submitted_at']})"
        )

        if job["state"] == "stopped" and job["log"]:
            reason = stop_reason(job["log"])
            line += f": {reason + ', ' if reason else ''}see {job['log'].name}"

        lines.append(line)

    return lines


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

    from ..drafts.submissions import queued

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
    lines += _job_lines("Tests:", tests, lambda test: test_name(test))
    runs = mcor_sweep_runs(main_folder, project)
    lines += _job_lines("Sweeps:", runs, lambda run: run["run"], "runs")

    failed = requests(main_folder, project).items(FAILED_FOLDER)
    lines.append(f"Failed:     {len(failed)} requests in total")
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
