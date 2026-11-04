"""
The records of the test motion corrections (`mcor/`, `sweeps/`) and of
the QC decisions in the database: what was submitted, how each job
stands, the latest pick, approval and rerun of each group, and what an
approved experiment keeps of its tests.
"""

from __future__ import annotations

import shutil

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..database import Database
from ..groups import candidate_settings
from ..shared import read_json, write_json
from .config import (
    BASELINE_RUN,
    ENDED,
    JOBS_FOLDER,
    MCOR_FOLDER,
    SWEEPS_FOLDER,
    TIME_FORMAT,
    TickReport,
    cluster_folder,
)
from .slurm import _job_logs, _job_state, _slurm_jobs


def group_paths(db: Database, group_id: int) -> str:
    """The experiment folders of `group_id`, as one line (for logs)."""
    links = db.group_experiments

    if group_id not in links.index:
        return ""

    paths = db.experiments["exp_path"]
    exp_ids = links.loc[[group_id], "exp_id"].astype(int)

    return ", ".join(str(paths.get(exp_id, exp_id)) for exp_id in exp_ids)


def group_objective(db: Database, group_id: int) -> None | int:
    """The objective of the experiments of `group_id`, if one and known."""
    links = db.group_experiments

    if group_id not in links.index:
        return None

    exp_ids = links.loc[[group_id], "exp_id"].astype(int)
    objectives = set(db.experiments.loc[exp_ids, "objective"].dropna().astype(int))

    return objectives.pop() if len(objectives) == 1 else None


def latest_picks(db: Database) -> dict[int, dict]:
    """
    The latest `pick_mcor_candidate` call of each group: `call_id`, `user`,
    `run`, `settings` and `correct` (whether it needs a whole correction).
    """
    calls = db.calls

    if calls.empty:
        return {}

    method = calls["method_name"].map(lambda m: str(m).rsplit(".", 1)[-1])
    picks: dict[int, dict] = {}

    for call_id, call in calls[method == "pick_mcor_candidate"].iterrows():
        output = call["call_output"]
        inputs = call["parameter_inputs"]

        if not isinstance(output, dict) or not isinstance(inputs, dict):
            continue  # it failed

        group_id = int(call["group_id"])

        if group_id not in picks or int(call_id) > picks[group_id]["call_id"]:
            picks[group_id] = {
                "call_id": int(call_id),
                "user": str(call["user"]),
                "run": str(output.get("run")),
                "settings": dict(inputs.get("settings") or {}),
                "correct": bool(output.get("correct")),
            }

    return picks


# What an approved experiment keeps of its tests, in `processed/mcor/report/`:
# the full run's numbers, and each test's (the videos and the rest go)
FULL_REPORT_FILES = (
    "summary.csv",
    "metrics.json",
    "flow_average.tif",
    "flow_worst.tif",
    "shifts_summary.npz",
)
TEST_REPORT_FILES = ("summary.csv", "metrics.json")
REPORT_FOLDER = "report"


def latest_approvals(db: Database) -> dict[int, dict]:
    """
    Each group's latest `approve_mcor_files` that worked: `call_id`, `user`,
    `called_at` and the acquisitions `excluded`.
    """
    calls = db.calls

    if calls.empty:
        return {}

    approvals: dict[int, dict] = {}

    approved = calls[calls["method_name"] == "Group.approve_mcor_files"]

    for call_id, call in approved.sort_index().iterrows():
        output = call["call_output"]

        if isinstance(output, dict) and output.get("approved"):
            approvals[int(call["group_id"])] = {
                "call_id": int(call_id),
                "user": str(call["user"]),
                "called_at": call["called_at"],
                "excluded": list(output.get("excluded") or []),
            }

    return approvals


def archive_approved(
    db: Database, report: TickReport, dry_run: bool, keep_days: float
) -> None:
    """
    For each group approved more than `keep_days` ago (and not picked again
    since), keep its tests' numbers in `processed/mcor/report/` of its first
    experiment (`archive_tests`) and delete its test folder from the share.
    Nothing is deleted unless every copy checks out; the next tick tries
    again.
    """
    picks = latest_picks(db)
    now = datetime.now()

    for group_id, approval in sorted(latest_approvals(db).items()):
        pick = picks.get(group_id)

        if pick is not None and pick["call_id"] > approval["call_id"]:
            continue  # picked again since: being checked again

        approved_at = pd.Timestamp(approval["called_at"]).to_pydatetime()

        if (now - approved_at).total_seconds() < keep_days * 86400:
            continue

        group = db.groups[group_id]
        tests = group.mcor_test_folder()

        if not tests.is_dir():
            continue

        what = f"group {group_id} ({', '.join(group.experiments['exp_path'])})"

        if dry_run:
            report.tests.append(f"{what}: would keep its tests' numbers, delete them")
            continue

        try:
            kept = archive_tests(group, approval, pick)
            shutil.rmtree(tests)
            report.tests.append(
                f"{what}: tests' numbers kept in {kept.relative_to(db.main_folder)}"
                ", tests deleted"
            )
        except Exception as error:  # the next tick tries again
            report.errors.append(f"{what}: archive: {type(error).__name__}: {error}")


def archive_tests(group, approval: dict, pick: None | dict = None) -> Path:
    """
    Copy what is worth keeping of `group`'s tests into `report/` beside its
    reference (`processed/mcor/` of its first experiment): the full run's
    report (`FULL_REPORT_FILES`), each test's numbers (`TEST_REPORT_FILES`,
    in `tests/<run>/`; the whole correction's in `tests/final/`) and
    `approval.json` (who approved what, when, and from which pick). Each
    copy is checked by size. Returns the folder.
    """
    from ..groups import FULL_FOLDER

    tests = group.mcor_test_folder()
    reference = group._reference_path()
    folder = reference.parent / REPORT_FOLDER
    copies: list[tuple[Path, Path]] = [
        (tests / FULL_FOLDER / name, folder / name) for name in FULL_REPORT_FILES
    ]
    runs = [("final", tests)] + [
        (run.name, run)
        for run in sorted(tests.iterdir())
        if run.is_dir()
        and run.name != FULL_FOLDER  # copied above, as the full run's
        and (run / "metrics.json").is_file()
    ]

    for name, run in runs:
        copies += [(run / f, folder / "tests" / name / f) for f in TEST_REPORT_FILES]

    for source, target in copies:
        if not source.is_file():
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

        if target.stat().st_size != source.stat().st_size:
            raise OSError(f"{target} was not copied whole")

    record = {
        "group_id": group.group_id,
        "experiments": list(group.experiments["exp_path"]),
        "approved_by": approval["user"],
        "approved_at": str(approval["called_at"]),
        "approval_call_id": approval["call_id"],
        "excluded_acq_ids": approval["excluded"],
        "picked_run": None if pick is None else pick["run"],
        "picked_settings": None if pick is None else pick["settings"],
        "reference": reference.name,
        "archived_at": datetime.now().strftime(TIME_FORMAT),
    }

    try:
        from ..groups import _sha1

        record["reference_sha1"] = _sha1(reference)
    except OSError:
        pass

    write_json(folder / "approval.json", record)

    return folder


def pick_candidate(
    db: Database,
    group_id: int,
    run: str,
    folder: str | Path,
    set_aside: Sequence[str] = (),
) -> bool:
    """
    Pick test `run` (report in `folder`) to correct every acquisition of
    `group_id`, recorded as the `db` user (`Group.pick_mcor_candidate`). The
    next tick submits the whole correction (`submit_picks`). Returns whether
    one is needed. Refused while the last whole correction of the group is
    still running, since both would write the same files. `set_aside` are
    the runs set aside before it, first to last (`Group.pick_mcor_candidate`).
    """
    sent = cluster_folder(db.main_folder, db.project) / MCOR_FOLDER / f"{group_id}.json"

    if sent.is_file():
        last = next(
            (
                test
                for test in mcor_tests(db.main_folder, db.project)
                if int(test["group_id"]) == group_id
            ),
            None,
        )

        if last is not None and last.get("whole") and last["state"] not in ENDED:
            raise ValueError(
                f"The last motion correction of every acquisition of group {group_id} "
                f"(job {last['job_id']}) has not finished yet; pick again once it has."
            )

    metrics = Path(folder) / "metrics.json"

    try:
        settings = read_json(metrics).get("settings") or {}
    except (OSError, ValueError) as error:
        raise ValueError(f"No settings of {run} to pick ({error}).") from None

    return db.groups[group_id].pick_mcor_candidate(
        run=run, settings=candidate_settings(settings), set_aside=list(set_aside)
    )


def pick_repeats(folder: str | Path, top: str | Path) -> tuple[bool, bool]:
    """
    Whether picking the test in `folder` repeats the settings of the last
    whole correction, whose report is in `top` (the experiment's test folder,
    on the share or its copy here), and whether that last one was rigid (or
    none was run). Its files are kept then, not corrected again.
    """

    def settings(where: Path) -> dict:
        try:
            return read_json(Path(where) / "metrics.json").get("settings") or {}
        except (OSError, ValueError):
            return {}

    picked = candidate_settings(settings(folder))
    whole = settings(top)
    last = candidate_settings(whole) if whole.get("whole") else None

    return last == picked, last is None or not last.get("pw_rigid")


def latest_reruns(db: Database) -> dict[int, dict]:
    """Each group's latest `Group.run_again` call: `call_id`, `user`, `runs`."""
    calls = db.calls

    if calls.empty:
        return {}

    latest: dict[int, dict] = {}
    asked = calls[calls["method_name"] == "Group.run_again"].sort_index()

    for call_id, call in asked.iterrows():
        inputs, output = call["parameter_inputs"], call["call_output"]

        if isinstance(inputs, dict) and isinstance(output, dict):
            latest[int(call["group_id"])] = {
                "call_id": int(call_id),
                "user": str(call["user"]),
                "runs": list(inputs.get("runs") or []),
            }

    return latest


def groups_without_full_report(db: Database) -> list[tuple[int, str]]:
    """
    The groups whose correction is saved, aligned to a saved reference, but
    with no report on every acquisition (`full/summary.csv`, see
    `Group.full_report`): `(group_id, experiment folders)`.
    """
    from ..groups import FULL_FOLDER

    corrected = set(db.mcor_files.index)
    links = db.group_experiments
    missing = []

    for group_id in sorted(set(links.index)):
        group = db.groups[int(group_id)]
        acq_ids = set(group.acquisitions.index)

        if not acq_ids or not acq_ids <= corrected:
            continue

        if not group._reference_path().is_file():
            continue

        if (group.mcor_test_folder() / FULL_FOLDER / "summary.csv").is_file():
            continue

        where = ", ".join(str(p) for p in group.experiments["exp_path"])
        missing.append((int(group_id), where))

    return missing


def experiment_owner(db: Database, group_id: int) -> str:
    """
    Who added the experiment of `group_id` (the user of its `add_experiment`
    call), which is who its tests and their QC belong to.
    """
    links = db.group_experiments

    if group_id not in links.index:
        raise ValueError(f"There is no group {group_id} with experiments.")

    exp_ids = set(links.loc[[group_id], "exp_id"].astype(int))
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

    Off the cluster (no `squeue`), the state the last tick recorded
    (`record_job_states`), or "unknown" without one.
    """
    folder = cluster_folder(main_folder, project)
    in_queue = _slurm_jobs()
    logs = _job_logs(folder)
    tests = []

    for path in (folder / MCOR_FOLDER).glob("*.json"):
        test = read_json(path)
        recorded = test.get("state")
        name = f"odyn-mcor-{project or 'main'}-g{test['group_id']}"
        log = f"{name}-{test['job_id']}.out"
        test["log"] = folder / JOBS_FOLDER / log if log in logs else None
        test["state"] = _job_state(str(test["job_id"]), test["log"], in_queue, recorded)
        tests.append(test)

    return sorted(tests, key=lambda t: t["submitted_at"], reverse=True)


def mcor_sweep_runs(main_folder: str | Path, project: None | str = None) -> list[dict]:
    """
    Every run of the submitted sweeps (`sweeps/g<group_id>.json`), newest
    sweep first and its baseline first, each with the sweep's `group_id`,
    `user` and `submitted_at`, its `run`, `job_id` (`<array>_<task>` for
    the piecewise runs), and `log` and `state` as `mcor_tests` has them.
    """
    folder = cluster_folder(main_folder, project)
    in_queue = _slurm_jobs()
    logs = _job_logs(folder)
    sweeps = []

    for path in (folder / SWEEPS_FOLDER).glob("g*.json"):
        sweep = read_json(path)

        if "baseline_job" in sweep:  # not a dry run
            sweeps.append(sweep)

    runs = []

    for sweep in sorted(sweeps, key=lambda s: s["submitted_at"], reverse=True):
        name = f"odyn-sweep-{project or 'main'}-g{sweep['group_id']}"
        recorded = sweep.get("states") or {}

        for run, job_id in sweep_jobs(sweep):
            log = f"{name}-{job_id}.out"
            log = folder / JOBS_FOLDER / log if log in logs else None
            runs.append(
                {
                    "group_id": int(sweep["group_id"]),
                    "user": sweep["user"],
                    "submitted_at": sweep["submitted_at"],
                    "run": run,
                    "job_id": job_id,
                    "log": log,
                    "state": _job_state(job_id, log, in_queue, recorded.get(job_id)),
                }
            )

    return runs


def sweep_jobs(sweep: dict) -> list[tuple[str, str]]:
    """
    A submitted sweep's runs and their jobs, the baseline first: the job that
    ran it last (`reruns`, see `submit_reruns`), else the first.
    """
    reruns = sweep.get("reruns") or {}
    jobs = [(BASELINE_RUN, str(sweep["baseline_job"]))] + [
        (run["run"], f"{sweep['array_job']}_{task}")
        for task, run in enumerate(sweep["runs"], start=1)
    ]

    return [(run, str(reruns.get(run, job_id))) for run, job_id in jobs]


def record_job_states(main_folder: str | Path, project: None | str) -> int:
    """
    Write each test job's state (as `mcor_tests` sees it here, with `squeue`)
    into its record, so the app, which has no `squeue`, can tell a job that
    stopped from one still running. Returns how many records changed.
    """
    in_queue = _slurm_jobs()

    if in_queue is None:
        return 0

    folder = cluster_folder(main_folder, project)
    logs = folder / JOBS_FOLDER
    changed = 0

    for path in (folder / MCOR_FOLDER).glob("*.json"):
        test = read_json(path)
        name = f"odyn-mcor-{project or 'main'}-g{test['group_id']}"
        log = logs / f"{name}-{test['job_id']}.out"
        state = _job_state(
            str(test["job_id"]), log if log.is_file() else None, in_queue
        )

        if test.get("state") != state:
            write_json(path, test | {"state": state})
            changed += 1

    for path in (folder / SWEEPS_FOLDER).glob("g*.json"):
        sweep = read_json(path)

        if "baseline_job" not in sweep:
            continue

        name = f"odyn-sweep-{project or 'main'}-g{sweep['group_id']}"
        states = {}

        for _, job_id in sweep_jobs(sweep):
            log = logs / f"{name}-{job_id}.out"
            states[job_id] = _job_state(
                job_id, log if log.is_file() else None, in_queue
            )

        if sweep.get("states") != states:
            write_json(path, sweep | {"states": states})
            changed += 1

    return changed


def test_name(test: dict) -> str:
    """
    A test's name, as the app shows it: its label (a picked correction's
    `<run> (all)`), `rigid (all)` for a whole one, else `test`.
    """
    return test.get("label") or ("rigid (all)" if test.get("whole") else "test")
