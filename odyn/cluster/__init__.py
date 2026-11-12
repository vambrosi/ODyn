"""
The cluster side: requests from other computers, and the `tick` that applies
them and does what the database is waiting for.

Workstations cannot start cluster jobs (logging in takes a password and Duo),
so anything that has to run there is left as a request file on the share,
and a short `tick` on the cluster applies it. A tick also adds the submitted
drafts whose recordings finished copying (`odyn.drafts.submissions`),
submits a test motion correction of each experiment it added
(`submit_mcor_test`), records the QC decisions made in the app (after an
approval, it submits the mean images and Cellpose candidates of the group as
one job: `submit_cellpose`), processes the sync files that arrived since their
experiment was added, writes a heartbeat, and (as a chain) submits the next
tick. Everything a tick does is safe to repeat, so a tick that ran twice, or
not at all for a day, only delays work.

Files, per database (`main` or the project's name), in
`<main_folder>/.odyn/cluster/<name>/`:
- `requests/` holds the requests waiting, `requests/done/` the applied ones
and `requests/failed/` the refused ones, each with the reason;
- `submissions/` holds the submitted drafts not added yet, each with its
state, and `submissions/done/` the ones added;
- `heartbeat.json` says when the last tick ran and what it did (and, after
a crash, the error: `record_crash`), and `tick.log` keeps a line per tick;
- `claims/` holds the QC decisions made in the app, waiting for the tick to
record them (`odyn.claims`), and `claims/done/` the recorded ones;
- `qc.json` is what the QC page shows (`odyn.qc`), rebuilt by each tick and
as each test or sweep job ends;
- `seen.json` is what each sync folder looked like when a tick last tried it;
- `jobs/` holds the job scripts and their output;
- `mcor/` has a file per group whose test motion correction was submitted
(`<group_id>.json`: the job, who it is for, when), so no group is tested twice
by the tick;
- `sweeps/` has a file per group whose sweep of test motion corrections was
submitted (`g<group_id>.json`: its runs and jobs, see `submit_mcor_sweep`);
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
python -m odyn.cluster <main_folder> mcor-test --group 7 [--whole] [--dry-run]
python -m odyn.cluster <main_folder> mcor-sweep --group 7 --grid grid.json
python -m odyn.cluster <main_folder> mcor-backfill --batch 4 [--objective 20]
python -m odyn.cluster <main_folder> full-previews --group 7 --user ana
python -m odyn.cluster <main_folder> full-report --group 7 [--group 8 ...]
python -m odyn.cluster <main_folder> full-report --missing
python -m odyn.cluster <main_folder> mean-images --group 7 [--group 8 ...] [--all-saved]
python -m odyn.cluster <main_folder> mean-images --missing
python -m odyn.cluster <main_folder> cellpose --group 7 [--model cpsam] [--tile-um 200]
python -m odyn.cluster <main_folder> cellpose --missing
```
`full-previews` writes the full run's videos of a group corrected before
they were written (`Group.save_full_previews`), and `full-report` the
report on every acquisition (`Group.full_report`; `--missing` lists the
groups without one), and `mean-images` the mean image of each corrected
acquisition and their average (`Group.save_mean_images`; `--missing` lists
the groups without them, `--all-saved` takes the acquisitions not approved
yet too); run them on a node (`srun`), as they read every acquisition.
`cellpose` finds the cell candidates in the mean image of a group
(`Group.save_cellpose_candidates`; `--missing` lists the groups with a mean
image and none), with the model weights in `.odyn/segmentation/` of the main
folder: `cpsam` for a group of experiments annotated deep, `cpdino` for the
others, unless `--model` says. Run it on a node too (a GPU is not needed).
After an approval in the app, the tick runs both for the group, as one job.

The code, by part: `config` (folders and settings), `slurm` (jobs and
their states), `messages` (requests and claims), `records` (the test and
sweep records, and the decisions in the database), `mcor` (submitting test
motion corrections, sweeps and whole corrections), `tick` (the tick, its
chain and `status`) and `cli` (the command line). The names below come
from here, whichever part holds them.

**PYTHON**
```python
from odyn.cluster import send_request, tick

send_request(main_folder, "process_sync", project="main-sync", exp_id=12)
report = tick(main_folder, project="main-sync")
```
"""

from .config import (
    BASELINE_RUN,
    CORRECTION_RUN,
    ClusterConfig,
    ENDED,
    SHARE_UMASK,
    TIME_FORMAT,
    TickReport,
    cluster_folder,
)
from .messages import apply_claims, pending_requests, send_request
from .records import (
    archive_approved,
    experiment_owner,
    groups_without_cellpose_candidates,
    groups_without_full_report,
    groups_without_mean_images,
    latest_approvals,
    latest_picks,
    mcor_sweep_runs,
    mcor_tests,
    pick_candidate,
    pick_repeats,
    record_job_states,
)
from .mcor import (
    mcor_backfill,
    run_mcor_sweep,
    run_mcor_test,
    submit_cellpose,
    submit_mcor_sweep,
    submit_mcor_test,
    submit_picks,
    submit_reruns,
    sweep_grid_for,
    sweep_runs,
)
from .tick import (
    crashed,
    heartbeat,
    record_crash,
    start_chain,
    status,
    stop_chain,
    tick,
    tick_script,
)
from .cli import main

__all__ = [
    "BASELINE_RUN",
    "CORRECTION_RUN",
    "ClusterConfig",
    "ENDED",
    "SHARE_UMASK",
    "TIME_FORMAT",
    "TickReport",
    "apply_claims",
    "archive_approved",
    "cluster_folder",
    "crashed",
    "experiment_owner",
    "groups_without_cellpose_candidates",
    "groups_without_full_report",
    "groups_without_mean_images",
    "heartbeat",
    "latest_approvals",
    "latest_picks",
    "main",
    "mcor_backfill",
    "mcor_sweep_runs",
    "mcor_tests",
    "pending_requests",
    "pick_candidate",
    "pick_repeats",
    "record_crash",
    "record_job_states",
    "run_mcor_sweep",
    "run_mcor_test",
    "send_request",
    "start_chain",
    "status",
    "stop_chain",
    "submit_cellpose",
    "submit_mcor_sweep",
    "submit_mcor_test",
    "submit_picks",
    "submit_reruns",
    "sweep_grid_for",
    "sweep_runs",
    "tick",
    "tick_script",
]
