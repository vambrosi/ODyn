"""
Slurm: submitting jobs, what is in its queue, a job's state from its
output file, and the job scripts' common part.
"""

from __future__ import annotations

import getpass
import os
import shlex
import subprocess

from pathlib import Path

from ..utils import logger
from .config import ClusterConfig, ENDED, JOBS_FOLDER, SHARE_UMASK, cluster_folder


def _sbatch(arguments: list[str], dry_run: bool) -> None | str:
    """Run `sbatch` with `arguments`; the job id (`None` in a dry run)."""
    command = ["sbatch", "--parsable", *arguments]

    if dry_run:
        logger.info(f"Would run: {shlex.join(command)}")
        return None

    done = subprocess.run(command, capture_output=True, text=True)

    if done.returncode != 0:
        raise RuntimeError(
            f"sbatch failed: {done.stderr.strip() or done.stdout.strip()}"
        )

    return done.stdout.strip().split(";")[0]


def job_name(project: None | str = None) -> str:
    """The Slurm job name of a database's tick (one at a time, by `singleton`)."""
    return f"odyn-tick-{project or 'main'}"


def _odyn_command(main_folder: str | Path, project: None | str) -> list[str]:
    return [
        "python",
        "-m",
        "odyn.cluster",
        str(Path(main_folder).resolve()),
        *(["--project", project] if project else []),
    ]


def _mcor_script(
    main_folder: str | Path,
    project: None | str,
    config: ClusterConfig,
    *,
    name: str,
    about: str,
    command: str,
    time: str,
    disk_gb: None | int = None,
) -> str:
    """
    A job script that runs `command` with caiman's files on scratch
    (`disk_gb` of it, `mcor_disk_gb` by default).
    """
    logs = cluster_folder(Path(main_folder).resolve(), project) / JOBS_FOLDER

    return f"""#!/bin/bash
#SBATCH --partition={config.partition}
#SBATCH --job-name={name}
#SBATCH --output={logs}/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.mcor_cpus}
#SBATCH --mem={config.mcor_memory}
#SBATCH --time={time}
#SBATCH --gres=disk:{disk_gb or config.mcor_disk_gb}

# {about}
# caiman's temporary files go to the node's scratch disk, removed however the
# job ends.

set -euo pipefail

# What the job writes on the share is the lab's: writable by the group (the
# lab's SMB access goes through it), not by others
umask {SHARE_UMASK:04o}

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export CAIMAN_TEMP="{config.scratch}/${{SLURM_JOB_ID}}"
mkdir -p "$CAIMAN_TEMP"
trap 'rm -rf "$CAIMAN_TEMP"' EXIT

source "{config.conda_sh}"
set +u
conda activate {config.conda_env}
set -u

echo {shlex.quote(about)}
echo "{name} as job ${{SLURM_JOB_ID:-none}} on $(hostname)"
echo "cores    ${{SLURM_CPUS_PER_TASK:-?}}, temporary files in $CAIMAN_TEMP"
echo "started  $(date --iso-8601=seconds)"
{command}
echo "finished $(date --iso-8601=seconds)"
{shlex.join(_odyn_command(main_folder, project))} qc-view >/dev/null 2>&1 || true
"""


def _job_state(
    job_id: str,
    log: None | Path,
    in_queue: None | dict[str, str],
    recorded: None | str = None,
) -> str:
    """
    A job's state for `mcor_tests` (`<array>_<task>` for an array's task).
    Off the cluster, `recorded` (by the last tick) stands in for `squeue`.
    """
    if in_queue is None and recorded in ENDED:  # over: its log says no more
        return recorded

    # Before `squeue`: a job still in it can be past its work (updating the
    # QC view at its end)
    if log is not None and _last_line(log).startswith("finished"):
        return "finished"

    queued = None if in_queue is None else _queued(job_id, in_queue)

    if queued is not None:
        return queued.lower()

    if in_queue is None:
        return recorded or "unknown"

    return "stopped"


def _queued(job_id: str, in_queue: dict[str, str]) -> None | str:
    """
    The state of `job_id` in `squeue`'s list, which shows an array's waiting
    tasks together (`123_[4-17]`, `123_[4,6-9%4]`).
    """
    if job_id in in_queue:
        return in_queue[job_id]

    array, _, task = job_id.partition("_")

    for key, state in in_queue.items():
        head, _, spans = key.partition("_[")

        if not task or head != array or not spans:
            continue

        for span in spans.rstrip("]").split("%")[0].split(","):
            first, _, last = span.partition("-")

            if int(first) <= int(task) <= int(last or first):
                return state

    return None


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


def _job_logs(folder: Path) -> set[str]:
    """
    The names of the job logs in `folder`'s `jobs/`, in one listing (on a
    network share, asking about each file is a round trip).
    """
    try:
        return {entry.name for entry in os.scandir(folder / JOBS_FOLDER)}
    except OSError:
        return set()


# How much of the end of a log `_last_line` reads: logs grow long
TAIL_BYTES = 8192


def _last_line(path: Path) -> str:
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - TAIL_BYTES))
        tail = handle.read().decode(errors="replace")

    lines = tail.strip().splitlines()

    return lines[-1] if lines else ""


def stop_reason(log: Path) -> str:
    """Why a job stopped, from the end of its log, when Slurm says it there."""
    last = _last_line(log)

    if "TIME LIMIT" in last:
        return "time limit"

    if "oom" in last.lower() or "OUT_OF_MEMORY" in last:
        return "out of memory"

    if "CANCELLED" in last:
        return "cancelled"

    return ""


def queued_ticks(project: None | str = None) -> list[str]:
    """The ids of this database's ticks waiting or running in Slurm."""
    done = subprocess.run(
        ["squeue", "--noheader", "--format=%i %T %S", f"--name={job_name(project)}"],
        capture_output=True,
        text=True,
    )

    return [line for line in done.stdout.splitlines() if line.strip()]
