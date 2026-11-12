"""
`python -m odyn.cluster`: the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path
from typing import Any

from ..database import Database
from ..shared import read_json
from .config import SHARE_UMASK
from .mcor import (
    mcor_backfill,
    run_mcor_sweep,
    run_mcor_test,
    submit_mcor_sweep,
    submit_mcor_test,
)
from .messages import REQUESTABLE, send_request
from .records import (
    experiment_owner,
    groups_without_cellpose_candidates,
    groups_without_full_report,
    groups_without_mean_images,
)
from .tick import (
    record_crash,
    start_chain,
    status,
    stop_chain,
    tick,
    write_qc_view,
)


def _value(text: str) -> Any:
    """A `key=value` value typed on the command line, read as JSON when it is."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def main(argv: None | list[str] = None) -> int:
    os.umask(SHARE_UMASK)  # see `SHARE_UMASK`
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
    commands.add_parser("qc-view", help="rebuild the QC page's view (qc.json)")

    test = commands.add_parser(
        "mcor-test", help="submit a test motion correction of a group as a job"
    )
    test.add_argument("--group", type=int, required=True)
    test.add_argument("--user", help="whose it is (default: who added the experiment)")
    test.add_argument(
        "--whole",
        action="store_true",
        help="correct and save every acquisition (the final rigid correction)",
    )
    test.add_argument("--dry-run", action="store_true", help="only show the job")

    job = commands.add_parser("run-mcor-test", help="what the mcor-test job runs")
    job.add_argument("--group", type=int, required=True)
    job.add_argument("--user", required=True)
    job.add_argument("--whole", action="store_true")
    job.add_argument("--settings", type=json.loads, help="test settings, as JSON")
    job.add_argument("--label", help="name on the report")

    sweep = commands.add_parser(
        "mcor-sweep", help="submit a sweep of test motion corrections of a group"
    )
    sweep.add_argument("--group", type=int, required=True)
    sweep.add_argument("--grid", type=Path, required=True, help="the runs (JSON)")
    sweep.add_argument("--user", help="whose it is (default: who added the experiment)")
    sweep.add_argument("--time", help="the time limit of each run (default: mcor_time)")
    sweep.add_argument("--dry-run", action="store_true", help="only show the jobs")

    one = commands.add_parser("run-mcor-sweep", help="what a sweep's jobs run")
    one.add_argument("--sweep", type=Path, required=True)
    one.add_argument("--task", type=int, required=True)

    backfill = commands.add_parser(
        "mcor-backfill",
        help="correct the next experiments never tested: whole rigid, and a sweep",
    )
    backfill.add_argument("--batch", type=int, required=True, help="how many")
    backfill.add_argument("--objective", type=int, help="only those through it")
    backfill.add_argument("--dry-run", action="store_true", help="only say which")

    previews = commands.add_parser(
        "full-previews", help="write the full run's videos of a corrected group"
    )
    previews.add_argument("--group", type=int, required=True)
    previews.add_argument("--user", help="whose (default: who added the experiment)")

    full = commands.add_parser(
        "full-report", help="write the report on every acquisition of a correction"
    )
    full.add_argument("--group", type=int, action="append", default=[])
    full.add_argument(
        "--missing", action="store_true", help="list the groups without one"
    )
    full.add_argument("--user", help="whose (default: who added the experiment)")

    means = commands.add_parser(
        "mean-images",
        help="write the mean image of each corrected acquisition, and their average",
    )
    means.add_argument("--group", type=int, action="append", default=[])
    means.add_argument(
        "--missing", action="store_true", help="list the groups without them"
    )
    means.add_argument(
        "--all-saved",
        action="store_true",
        help="every saved acquisition, not only the approved ones",
    )
    means.add_argument("--user", help="whose (default: who added the experiment)")

    cells = commands.add_parser(
        "cellpose", help="find the cell candidates of a group with Cellpose"
    )
    cells.add_argument("--group", type=int, action="append", default=[])
    cells.add_argument(
        "--missing",
        action="store_true",
        help="list the groups with a mean image but no candidates",
    )
    cells.add_argument(
        "--model",
        help="the Cellpose 4 model (default: cpsam for deep experiments, else cpdino)",
    )
    cells.add_argument(
        "--tile-um", type=float, help="normalize the brightness in tiles this wide"
    )
    cells.add_argument("--user", help="whose (default: who added the experiment)")

    ask = commands.add_parser("request", help="leave a request for the next tick")
    ask.add_argument("method", choices=REQUESTABLE)
    ask.add_argument("arguments", nargs="*", help="key=value")

    args = parser.parse_args(argv)

    if args.command == "tick":
        try:
            report = tick(
                args.main_folder, args.project, dry_run=args.dry_run, chain=args.chain
            )
        except Exception as error:
            if not args.dry_run:  # for `status` and the QC page
                record_crash(args.main_folder, args.project, error)
            raise

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
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        print(status(args.main_folder, args.project, color=color))

    elif args.command == "qc-view":
        write_qc_view(Database(args.main_folder, project=args.project))

    elif args.command == "mcor-test":
        submit_mcor_test(
            args.main_folder,
            args.project,
            args.group,
            user=args.user,
            whole=args.whole,
            dry_run=args.dry_run,
        )

    elif args.command == "mcor-sweep":
        submit_mcor_sweep(
            args.main_folder,
            args.project,
            args.group,
            read_json(args.grid),
            user=args.user,
            time=args.time,
            dry_run=args.dry_run,
        )

    elif args.command == "mcor-backfill":
        for line in mcor_backfill(
            args.main_folder,
            args.project,
            batch=args.batch,
            objective=args.objective,
            dry_run=args.dry_run,
        ):
            print(f"  {line}")

    elif args.command == "full-previews":
        db = Database(args.main_folder, project=args.project)
        user = args.user or experiment_owner(db, args.group)
        db = Database(args.main_folder, project=args.project, user=user)
        paths = db.groups[args.group].save_full_previews()
        print("Wrote " + ", ".join(str(path) for path in paths))

    elif args.command == "full-report":
        db = Database(args.main_folder, project=args.project)

        if args.missing:
            for group_id, where in groups_without_full_report(db):
                print(f"  group {group_id}: {where}")

        for group_id in args.group:
            user = args.user or experiment_owner(db, group_id)
            mine = Database(args.main_folder, project=args.project, user=user)
            print(f"Wrote {mine.groups[group_id].full_report()}")

    elif args.command == "mean-images":
        db = Database(args.main_folder, project=args.project)

        if args.missing:
            for group_id, where, approved in groups_without_mean_images(db):
                note = "" if approved else " (nothing approved yet)"
                print(f"  group {group_id}: {where}{note}")

        for group_id in args.group:
            user = args.user or experiment_owner(db, group_id)
            mine = Database(args.main_folder, project=args.project, user=user)
            group = mine.groups[group_id]
            paths = group.save_mean_images(only_approved=not args.all_saved)
            print("Wrote " + ", ".join(str(path) for path in paths))

    elif args.command == "cellpose":
        db = Database(args.main_folder, project=args.project)

        if args.missing:
            for group_id, where in groups_without_cellpose_candidates(db):
                print(f"  group {group_id}: {where}")

        for group_id in args.group:
            user = args.user or experiment_owner(db, group_id)
            mine = Database(args.main_folder, project=args.project, user=user)
            group = mine.groups[group_id]
            path = group.save_cellpose_candidates(
                model=args.model, tile_um=args.tile_um
            )
            print(f"Wrote {path}")

    elif args.command == "run-mcor-sweep":
        folder = run_mcor_sweep(args.sweep, args.task)
        print(f"Wrote {folder}")

    elif args.command == "run-mcor-test":
        folder = run_mcor_test(
            args.main_folder,
            args.project,
            args.group,
            args.user,
            args.whole,
            args.settings,
            args.label,
        )
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
