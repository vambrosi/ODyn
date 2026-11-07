"""
Where the cluster keeps its files, its settings (`ClusterConfig`), and
what a tick reports (`TickReport`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from ..shared import read_json
from ..utils import ODYN_FOLDER, database_path

CLUSTER_FOLDER = "cluster"
CONFIG_FILE = "cluster.json"
REQUESTS_FOLDER = "requests"
JOBS_FOLDER = "jobs"
HEARTBEAT_FILE = "heartbeat.json"
LOG_FILE = "tick.log"
STOP_FILE = "STOP"
SEEN_FILE = "seen.json"
MCOR_FOLDER = "mcor"
SWEEPS_FOLDER = "sweeps"
# The run of a sweep every other run is compared with, and takes the
# reference and the movie levels from
BASELINE_RUN = "rigid"

# The name `Group.run_again` gives the experiment's own job (the cluster's
# test, or the full correction), as opposed to a sweep's runs
CORRECTION_RUN = "correction"

# A sync file this recent may still be arriving (one copied by hand has no
# temporary name), so it waits for the next tick
SETTLE_S = 300

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Files written on the share from the cluster must stay writable by the lab's
# group: the computers in the lab reach the share through it (SMB), and a
# file the group cannot write (a database, a request) breaks the app there.
# A umask of 0027 (a common default) leaves the group read-only
SHARE_UMASK = 0o007


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
    # The tick submits a test motion correction of each experiment it adds;
    # with `whole_new_experiments`, one that corrects and saves every
    # acquisition (the final rigid correction), reporting on the test's
    whole_new_experiments: bool = True
    test_new_experiments: bool = True
    # With a grid (as `submit_mcor_sweep` takes it), the tick also submits a
    # sweep of each experiment it adds, each run with `sweep_time` (default:
    # `mcor_time`). The sizes depend on the objective, so it can be one grid
    # per objective instead: {"20": grid, "10": grid} (see `sweep_grid_for`)
    sweep_grid: None | dict = None
    sweep_time: None | str = None
    # Days after an approval its tests are kept on the share, before the tick
    # keeps their numbers in `processed/mcor/report/` and deletes the rest
    keep_tests_days: float = 7.0
    tick_cpus: int = 1
    tick_memory: str = "8G"
    tick_time: str = "0:30:00"
    mcor_cpus: int = 12
    mcor_memory: str = "96G"
    mcor_time: str = "4:00:00"
    mcor_disk_gb: int = 128  # the node's scratch disk, for caiman's temporary files
    # A whole correction: longer, and more scratch (a batch of 40 acquisitions
    # and the 24 the report reads, about 2.5 GB each at 1200 x 1200 px)
    whole_time: str = "6:00:00"
    whole_pw_time: str = "12:00:00"  # piecewise rigid, slower
    whole_disk_gb: int = 200
    scratch: str = "/mnt/scratch"
    # Nodes motion correction jobs are not sent to, as Slurm writes them
    # ("cnode-07-[21-29],cnode-08-[24-27,29-35]"): caiman ran about 4 times
    # slower on those (2026-11-06), and whole corrections ran out of time.
    # Every name must exist, or Slurm refuses the job
    exclude_nodes: None | str = None

    @classmethod
    def read(cls, main_folder: str | Path) -> ClusterConfig:
        path = Path(main_folder) / ODYN_FOLDER / CONFIG_FILE

        if not path.is_file():
            return cls()

        given = read_json(path)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(given) - known)

        if unknown:
            raise ValueError(
                f"Unknown settings in '{path}': {', '.join(unknown)}. The known "
                f"ones are {', '.join(sorted(known))}."
            )

        return cls(**given)


@dataclass
class TickReport:
    """What one tick did (or, in a dry run, would do)."""

    started_at: str = ""
    dry_run: bool = False
    applied: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
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
            f"{len(self.claims)} QC decisions recorded",
            f"{len(self.submissions)} submissions looked at",
            f"{len(self.tests)} tests submitted",
            f"{len(self.processed)} sync files processed",
            f"{len(self.errors)} errors",
        ]

        if self.next_tick:
            parts.append(f"next tick {self.next_tick}")

        return f"{self.started_at}  " + ", ".join(parts)


# A job in these states is over (see `mcor_tests`)
ENDED = ("finished", "stopped", "failed", "cancelled", "timeout")
