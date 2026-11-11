"""
Submitting test motion corrections and sweeps (and what their jobs run),
whole corrections after a pick, runs asked again, and the mean images and
Cellpose candidates after an approval.
"""

from __future__ import annotations

import json
import shlex
import subprocess

from datetime import datetime
from pathlib import Path
from typing import Any

from ..database import Database
from ..groups import candidate_settings
from ..shared import read_json, write_json
from ..utils import logger
from .config import (
    BASELINE_RUN,
    CORRECTION_RUN,
    ClusterConfig,
    JOBS_FOLDER,
    MCOR_FOLDER,
    SWEEPS_FOLDER,
    TIME_FORMAT,
    TickReport,
    cluster_folder,
)
from .records import (
    experiment_owner,
    group_objective,
    group_paths,
    latest_picks,
    latest_reruns,
    mcor_sweep_runs,
)
from .slurm import _mcor_script, _odyn_command, _sbatch


def sweep_grid_for(config: ClusterConfig, objective: None | int) -> None | dict:
    """
    The sweep grid for an experiment through `objective` (10, 20, ...): the
    one grid for all, or that objective's (`ClusterConfig.sweep_grid`), or
    `None` without one.
    """
    grid = config.sweep_grid

    if not grid:
        return None

    if "runs" in grid:  # one grid for every objective
        return grid

    return None if objective is None else grid.get(str(int(objective)))


def mcor_backfill(
    main_folder: str | Path,
    project: None | str,
    *,
    batch: int,
    objective: None | int = None,
    dry_run: bool = False,
) -> list[str]:
    """
    Correct the next `batch` experiments of the database that were never
    tested (no test submitted, by the tick or by hand), oldest first, as the
    tick does a new one: the whole rigid correction (which saves every
    acquisition, over the files there) and a sweep with the grid of the
    experiment's objective (`sweep_grid_for`). `objective` keeps those
    through it. Each is recorded as its owner's (`experiment_owner`), whose
    QC page lists it. Returns what was done (or would be), a line each.

    Run it again for the next batch: what was submitted is not again.
    """
    db = Database(main_folder, project=project)
    config = ClusterConfig.read(main_folder)
    folder = cluster_folder(main_folder, project)
    links = db.group_experiments
    sizes = links.index.value_counts()
    paths = db.experiments["exp_path"]
    done: list[str] = []
    sent = waiting = 0

    for group_id, exp_id in sorted(
        zip(links.index, links["exp_id"].astype(int)),
        key=lambda pair: str(paths[pair[1]]),
    ):
        group_id, where = int(group_id), str(paths[exp_id])

        if sizes[group_id] != 1 or (folder / MCOR_FOLDER / f"{group_id}.json").exists():
            continue

        seen = group_objective(db, group_id)

        if objective is not None and seen != objective:
            continue

        if sent >= batch:
            waiting += 1
            continue

        grid = sweep_grid_for(config, seen)
        what = f"{where} ({seen}x, group {group_id})"

        if grid is None:
            done.append(f"{what}: no sweep grid for {seen}x in cluster.json, skipped")
            continue

        if dry_run:
            done.append(f"{what}: would submit the whole correction and a sweep")
            sent += 1
            continue

        try:
            whole = submit_mcor_test(main_folder, project, group_id, whole=True)
            sweep = (
                submit_mcor_sweep(
                    main_folder, project, group_id, grid, time=config.sweep_time
                )
                if not _sweep_sent(folder, group_id)
                else None
            )
        except Exception as error:  # said, and the next one is tried
            done.append(f"{what}: {type(error).__name__}: {error}")
            continue

        sent += 1
        done.append(
            f"{what}: whole correction job {whole}"
            + (f", sweep jobs {sweep[0]} and {sweep[1]}" if sweep else "")
        )

    if waiting:
        done.append(f"{waiting} more waiting for the next batch")

    return done


def _sweep_sent(folder: Path, group_id: int) -> bool:
    """Whether a sweep of `group_id` was submitted (its jobs recorded)."""
    try:
        sweep = read_json(folder / SWEEPS_FOLDER / f"g{group_id}.json")
    except (OSError, ValueError):
        return False

    return "baseline_job" in sweep


def submit_reruns(
    db: Database, project: None | str, report: TickReport, dry_run: bool
) -> None:
    """
    Submit again the test jobs a `Group.run_again` call asked for, once
    each: the experiment's own (`CORRECTION_RUN`, with its settings, label
    and pick), and a sweep's runs (the stopped baseline with every run not
    finished after it, which waited on it). Each record keeps the call it
    ran (`rerun_call_id`), so a call is never run twice.
    """
    folder = cluster_folder(db.main_folder, project)

    for group_id, rerun in sorted(latest_reruns(db).items()):
        call_id, names = rerun["call_id"], set(rerun["runs"])
        sent_path = folder / MCOR_FOLDER / f"{group_id}.json"
        sweep_path = folder / SWEEPS_FOLDER / f"g{group_id}.json"

        try:
            if CORRECTION_RUN in names and sent_path.is_file():
                sent = read_json(sent_path)

                if sent.get("rerun_call_id") != call_id:
                    what = f"group {group_id}: {sent.get('label') or 'test'} again"

                    if dry_run:
                        report.tests.append(f"would submit {what}")
                    else:
                        job_id = submit_mcor_test(
                            db.main_folder,
                            project,
                            group_id,
                            user=sent["user"],
                            whole=bool(sent.get("whole")),
                            settings=sent.get("settings") or None,
                            label=sent.get("label"),
                            pick_call_id=sent.get("pick_call_id"),
                        )
                        again = read_json(sent_path)
                        write_json(sent_path, again | {"rerun_call_id": call_id})
                        report.tests.append(f"{what} submitted (job {job_id})")

            if names - {CORRECTION_RUN} and sweep_path.is_file():
                sweep = read_json(sweep_path)

                if sweep.get("rerun_call_id") != call_id:
                    what = f"group {group_id}: sweep runs again"

                    if dry_run:
                        report.tests.append(f"would submit {what}")
                    else:
                        jobs = rerun_sweep(db.main_folder, project, sweep, names)
                        sweep = read_json(sweep_path)
                        write_json(
                            sweep_path,
                            sweep
                            | {
                                "reruns": (sweep.get("reruns") or {}) | jobs,
                                "rerun_call_id": call_id,
                            },
                        )
                        report.tests.append(
                            f"{what} submitted ({', '.join(sorted(jobs))})"
                        )

        except Exception as error:  # the next tick tries again
            report.errors.append(
                f"group {group_id}: run again: {type(error).__name__}: {error}"
            )


def rerun_sweep(
    main_folder: str | Path, project: None | str, sweep: dict, names: set[str]
) -> dict[str, str]:
    """
    Submit again the runs `names` of `sweep` with its own job script: the
    baseline as one job, the piecewise runs as an array of their tasks (after
    the baseline, when it runs again too, with every run not finished). The
    first sweep's tasks still waiting on a failed baseline are cancelled.
    Returns each run's new job id.
    """
    folder = cluster_folder(main_folder, project)
    script = folder / JOBS_FOLDER / f"mcor-sweep-g{sweep['group_id']}.sh"
    logs = folder.resolve() / JOBS_FOLDER
    states = {run: state for run, state in _sweep_states(main_folder, project, sweep)}
    baseline = BASELINE_RUN in names
    tasks = [
        (task, run["run"])
        for task, run in enumerate(sweep["runs"], start=1)
        if run["run"] in names or (baseline and states.get(run["run"]) != "finished")
    ]
    jobs: dict[str, str] = {}
    base_job = None

    if baseline:
        _cancel_waiting(sweep)
        base_job = _sbatch([str(script)], dry_run=False)
        jobs[BASELINE_RUN] = str(base_job)

    if tasks:
        arguments = [
            f"--array={','.join(str(task) for task, _ in tasks)}",
            f"--output={logs}/%x-%A_%a.out",
        ]

        if base_job is not None:
            arguments.append(f"--dependency=afterok:{base_job}")

        array = _sbatch([*arguments, str(script)], dry_run=False)
        jobs |= {run: f"{array}_{task}" for task, run in tasks}

    return jobs


def _sweep_states(main_folder, project, sweep: dict) -> list[tuple[str, str]]:
    """Each run of `sweep` with its state, as `mcor_sweep_runs` has it."""
    states = {
        (run["run"], run["state"])
        for run in mcor_sweep_runs(main_folder, project)
        if run["group_id"] == int(sweep["group_id"])
    }

    return sorted(states)


def _cancel_waiting(sweep: dict) -> None:
    """Cancel the sweep's first array, whose tasks wait on its baseline."""
    try:
        subprocess.run(["scancel", str(sweep["array_job"])], capture_output=True)
    except FileNotFoundError:
        pass


def submit_picks(
    db: Database, project: None | str, report: TickReport, dry_run: bool
) -> None:
    """
    Submit a whole correction for the latest pick of each group
    (`latest_picks`) that needs one and was not submitted yet, with the
    picked settings, recorded as the user who picked them.
    """
    folder = cluster_folder(db.main_folder, project) / MCOR_FOLDER

    for group_id, pick in sorted(latest_picks(db).items()):
        if not pick["correct"]:
            continue

        try:
            sent = read_json(folder / f"{group_id}.json")
        except (OSError, ValueError):
            sent = {}

        if sent.get("pick_call_id") == pick["call_id"]:
            continue

        what = f"{pick['run']} (all) of group {group_id}"

        if dry_run:
            report.tests.append(f"would submit {what}")
            continue

        try:
            job_id = submit_mcor_test(
                db.main_folder,
                project,
                group_id,
                user=pick["user"],
                whole=True,
                settings=pick["settings"],
                label=f"{pick['run']} (all)",
                pick_call_id=pick["call_id"],
            )
            report.tests.append(f"{what} submitted (job {job_id})")

        except Exception as error:  # the next tick tries again
            report.errors.append(f"{what}: {type(error).__name__}: {error}")


def mcor_test_script(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    user: str,
    config: ClusterConfig,
    whole: bool = False,
    settings: None | dict = None,
    label: None | str = None,
    path: str = "",
) -> str:
    """
    The job script of one test motion correction (see `submit_mcor_test`);
    `path` is the experiment folder, said in its log.
    """
    settings = settings or {}
    command = [
        *_odyn_command(main_folder, project),
        "run-mcor-test",
        "--group",
        str(group_id),
        "--user",
        user,
        *(["--whole"] if whole else []),
        *(["--settings", json.dumps(settings)] if settings else []),
        *(["--label", label] if label else []),
    ]
    kind = "piecewise rigid" if settings.get("pw_rigid") else "rigid"
    where = f"group {group_id}" + (f" ({path})" if path else "")
    about = (
        f"Whole {kind} motion correction of {where} for {user}"
        if whole
        else f"Test motion correction of {where} for {user}"
    )

    if whole:
        time = config.whole_pw_time if settings.get("pw_rigid") else config.whole_time
    else:
        time = config.mcor_time

    return _mcor_script(
        main_folder,
        project,
        config,
        name=f"odyn-mcor-{project or 'main'}-g{group_id}",
        about=about,
        command=shlex.join(command),
        time=time,
        disk_gb=config.whole_disk_gb if whole else config.mcor_disk_gb,
    )


def submit_mcor_test(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    *,
    user: None | str = None,
    whole: bool = False,
    settings: None | dict = None,
    label: None | str = None,
    pick_call_id: None | int = None,
    dry_run: bool = False,
) -> None | str:
    """
    Submit a test motion correction of `group_id` (`Group.test_motion_correction`)
    as its own job, recorded as `user`'s: by default whoever added the group's
    experiment. With `whole`, the job corrects and saves every acquisition (the
    final correction, with `whole_time`, or `whole_pw_time` for piecewise
    rigid, and `whole_disk_gb`) and reports on the test acquisitions.
    `settings` are the test settings (`CANDIDATE_SETTINGS`; rigid by
    default), `label` the name on the report, and `pick_call_id` the
    `pick_mcor_candidate` call it runs.
    Writes the script to `jobs/` and what was submitted to
    `mcor/<group_id>.json`. Returns the job id (`None` in a dry run, which only
    writes and shows the script).
    """
    db = Database(main_folder, project=project)

    if user is None:
        user = experiment_owner(db, group_id)

    path = group_paths(db, group_id)
    config = ClusterConfig.read(main_folder)
    folder = cluster_folder(main_folder, project)
    script = folder / JOBS_FOLDER / f"mcor-test-g{group_id}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        mcor_test_script(
            main_folder,
            project,
            group_id,
            user,
            config,
            whole,
            settings,
            label,
            path=path,
        )
    )
    job_id = _sbatch([str(script)], dry_run)

    if job_id is None:
        print(script.read_text())
        return None

    write_json(
        folder / MCOR_FOLDER / f"{group_id}.json",
        {
            "group_id": group_id,
            "path": path,
            "job_id": job_id,
            "user": user,
            "whole": whole,
            "settings": settings or {},
            "label": label,
            "pick_call_id": pick_call_id,
            "submitted_at": datetime.now().strftime(TIME_FORMAT),
        },
    )
    what = "whole test" if whole else "test"
    logger.info(
        f"Submitted the {what} motion correction of group {group_id} as job {job_id}."
    )

    return job_id


def run_mcor_test(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    user: str,
    whole: bool = False,
    settings: None | dict = None,
    label: None | str = None,
) -> Path:
    """
    What the job runs: the test (`whole`: saving every acquisition), with
    `settings` (`CANDIDATE_SETTINGS`) and `label`, recorded as `user`.
    Returns the test folder.
    """
    db = Database(main_folder, project=project, user=user)
    given = candidate_settings(settings) if settings else {}

    if label:
        given["label"] = label

    return db.groups[group_id].test_motion_correction(whole=whole, **given)


def submit_cellpose(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    *,
    user: str,
    dry_run: bool = False,
) -> None | str:
    """
    Submit what follows an approval of group `group_id` as one job, its calls
    recorded as made by `user`: the mean images of its approved acquisitions
    (`Group.save_mean_images`), then the Cellpose candidates in them
    (`Group.save_cellpose_candidates`, with the default model). Writes the
    script to `jobs/` and returns the job id (`None` in a dry run, which only
    writes and shows the script).
    """
    db = Database(main_folder, project=project)
    config = ClusterConfig.read(main_folder)
    path = group_paths(db, group_id)
    where = f"group {group_id}" + (f" ({path})" if path else "")
    steps = [
        [*_odyn_command(main_folder, project), step, "--group", str(group_id)]
        + ["--user", user]
        for step in ("mean-images", "cellpose")
    ]
    script = (
        cluster_folder(main_folder, project) / JOBS_FOLDER / f"cellpose-g{group_id}.sh"
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        _mcor_script(
            main_folder,
            project,
            config,
            name=f"odyn-cellpose-{project or 'main'}-g{group_id}",
            about=f"Mean images and Cellpose candidates of {where} for {user}",
            command="\n".join(shlex.join(step) for step in steps),
            # It reads every acquisition, as a whole correction does
            time=config.whole_time,
            disk_gb=1,  # nothing goes to scratch
        )
    )
    job_id = _sbatch([str(script)], dry_run)

    if job_id is None:
        print(script.read_text())
        return None

    logger.info(
        f"Submitted the mean images and Cellpose candidates of group {group_id} "
        f"as job {job_id}."
    )

    return job_id


def submit_mcor_sweep(
    main_folder: str | Path,
    project: None | str,
    group_id: int,
    grid: dict[str, Any],
    *,
    user: None | str = None,
    time: None | str = None,
    dry_run: bool = False,
) -> None | tuple[str, str]:
    """
    Submit a sweep of test motion corrections of `group_id`, to compare
    piecewise rigid settings: a rigid baseline run (named "rigid") as one job,
    then every piecewise run of `grid` at once as an array job that starts
    when the baseline ends well. Slurm runs as many as it allows and keeps the
    rest waiting.

    The baseline builds and saves the reference, which the piecewise runs
    reuse. Every run writes its test video, of the same frames and with the
    same black and white as the experiment's raw video (in `raw/` of its test
    folder, written by the first), so all the videos compare. No run writes
    `side_by_side.tif`, and the patches' shifts are kept in `shifts.npz` only
    with `"keep_patch_shifts": true` (for looking into the patches afterwards;
    large at small patches). Each run has its folder in the experiment's test
    folder (`Group.mcor_test_folder`).

    `grid` is (as a JSON file for the command line):
    ```json
    {"test_acqs": 24, "max_shift_um": [80, 80],
     "runs": [{"strides_um": [128, 128], "overlap_um": [64, 64],
               "max_deviation_um": 6}, ...]}
    ```
    A run is named from its settings (`s128_o64_d6`), or by its own `"run"`.

    Writes the job script to `jobs/` and the sweep to `sweeps/g<group>.json`,
    which the jobs read. Returns the two job ids (`None` in a dry run).
    """
    runs = sweep_runs(grid)
    db = Database(main_folder, project=project)

    if user is None:
        user = experiment_owner(db, group_id)

    config = ClusterConfig.read(main_folder)
    folder = cluster_folder(main_folder, project)
    sweep = folder / SWEEPS_FOLDER / f"g{group_id}.json"
    content = {
        "main_folder": str(Path(main_folder).resolve()),
        "project": project,
        "group_id": group_id,
        "user": user,
        "test_acqs": int(grid.get("test_acqs", 24)),
        "max_shift_um": list(grid.get("max_shift_um", [80, 80])),
        "keep_patch_shifts": bool(grid.get("keep_patch_shifts", False)),
        "runs": runs,
    }
    write_json(sweep, content)

    name = f"odyn-sweep-{project or 'main'}-g{group_id}"
    command = [
        *_odyn_command(main_folder, project),
        "run-mcor-sweep",
        "--sweep",
        str(sweep.resolve()),
    ]
    script = folder / JOBS_FOLDER / f"mcor-sweep-g{group_id}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        _mcor_script(
            main_folder,
            project,
            config,
            name=name,
            about=(
                f"One run of the sweep of group {group_id} "
                f"({group_paths(db, group_id)}), for {user}"
            ),
            # Task 0 is the baseline, the array's tasks the piecewise runs
            command=shlex.join(command) + ' --task "${SLURM_ARRAY_TASK_ID:-0}"',
            time=time or config.mcor_time,
        )
    )

    logs = folder.resolve() / JOBS_FOLDER
    baseline = _sbatch([str(script)], dry_run)
    array = _sbatch(
        [
            f"--array=1-{len(runs)}",
            f"--dependency=afterok:{baseline or '<baseline>'}",
            f"--output={logs}/%x-%A_%a.out",
            str(script),
        ],
        dry_run,
    )

    if baseline is None or array is None:
        print(script.read_text())
        return None

    content |= {
        "baseline_job": baseline,
        "array_job": array,
        "submitted_at": datetime.now().strftime(TIME_FORMAT),
    }
    write_json(sweep, content)
    logger.info(
        f"Submitted the sweep of group {group_id}: the baseline as job "
        f"{baseline}, then {len(runs)} piecewise runs as job {array}."
    )

    return baseline, array


def sweep_runs(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """
    The piecewise runs of a sweep's `grid`, each with its name, checked. A
    run needs its patches (`strides_um`, `overlap_um`, `max_deviation_um`)
    and may add `gsig_filt_um` (named `_g<um>`) or `shifts_opencv` (`_cv`).
    """
    required = {"strides_um", "overlap_um", "max_deviation_um"}
    known = required | {"run", "gsig_filt_um", "shifts_opencv"}
    runs = []

    for given in grid.get("runs", []):
        unknown = sorted(set(given) - known)

        if unknown:
            raise ValueError(
                f"Unknown run settings {', '.join(unknown)}. The known ones are "
                f"{', '.join(sorted(known))}."
            )

        missing = sorted(required - set(given))

        if missing:
            raise ValueError(f"A run has no {', '.join(missing)}: {given}.")

        run = {
            "strides_um": list(given["strides_um"]),
            "overlap_um": list(given["overlap_um"]),
            "max_deviation_um": given["max_deviation_um"],
        }
        name = (
            f"s{_short(run['strides_um'])}_o{_short(run['overlap_um'])}"
            f"_d{_short([run['max_deviation_um']])}"
        )

        if given.get("gsig_filt_um") is not None:
            run["gsig_filt_um"] = given["gsig_filt_um"]
            name += f"_g{_short([run['gsig_filt_um']])}"
        elif given.get("shifts_opencv"):
            run["shifts_opencv"] = True
            name += "_cv"

        run["run"] = given.get("run") or name
        runs.append(run)

    if not runs:
        raise ValueError("The grid has no runs.")

    names = [run["run"] for run in runs]
    repeated = sorted(
        {n for n in names if names.count(n) > 1} | {BASELINE_RUN} & set(names)
    )

    if repeated:
        raise ValueError(
            f"Run names must differ, and not be {BASELINE_RUN!r}: "
            f"{', '.join(repeated)}."
        )

    return runs


def _short(values: list[float]) -> str:
    """`[128, 128]` as "128", `[128, 96]` as "128x96", `[2.5]` as "2.5"."""
    text = [f"{v:g}" for v in values]

    return text[0] if len(set(text)) == 1 else "x".join(text)


def run_mcor_sweep(sweep: str | Path, task: int) -> Path:
    """
    What a sweep's job runs: run `task` of the sweep in `sweep` (0 is the
    baseline), recorded as the sweep's user. Returns its folder.
    """
    content = read_json(sweep)
    db = Database(
        content["main_folder"], project=content["project"], user=content["user"]
    )
    group = db.groups[int(content["group_id"])]
    common = {
        "test_acqs": content["test_acqs"],
        "max_shift_um": content["max_shift_um"],
        "side_by_side": False,
    }

    if task == 0:
        return group.test_motion_correction(**common, run=BASELINE_RUN)

    return group.test_motion_correction(
        **common,
        **content["runs"][task - 1],
        keep_patch_shifts=bool(content.get("keep_patch_shifts", False)),
        pw_rigid=True,
        reuse_reference=True,
    )
