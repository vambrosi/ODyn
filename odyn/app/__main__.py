"""
`python -m odyn.app <main_folder> [--project NAME] [--user NAME] [--create]
[--no-update]`

Run as a command, the app first updates its own git clone and restarts if
that brought anything new (see `update`); `--no-update` skips it.

For trying the app without the rig, the recorder can take a stand-in for the
DAQ: `--simulate` (invented signals) or `--replay FILE [--speed X]` (a sync
file recorded on the rig, played back X times faster than it was recorded).
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


def main(argv: None | list[str] = None) -> int:
    if argv is None:  # launched, not called (a test, say)
        from .update import update_and_restart

        update_and_restart(sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m odyn.app",
        description=(
            "The ODyn app (use it to update the DB, to record/submit experiments, "
            "and for quality checks)."
        ),
    )
    parser.add_argument(
        "main_folder", type=Path, help="the folder that contains .odyn/ and the data"
    )
    parser.add_argument(
        "--project",
        default="main-sync",
        help='the database project (default: "main-sync"; "" for the project-less one)',
    )
    parser.add_argument(
        "--user",
        default=None,
        help="same as the network username (default: the current login name)",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="make the database if it does not exist yet",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="do not update ODyn before starting (also: ODYN_NO_UPDATE=1)",
    )
    stand_in = parser.add_mutually_exclusive_group()
    stand_in.add_argument(
        "--simulate",
        action="store_true",
        help="record from a simulated source, not the DAQ",
    )
    stand_in.add_argument(
        "--replay",
        type=Path,
        metavar="FILE",
        help="record by playing back a sync file from the rig, not the DAQ",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="how many times faster than recorded --replay plays (default 1)",
    )
    args = parser.parse_args(argv)

    if args.speed != 1.0 and args.replay is None:
        parser.error("--speed only applies to --replay")

    source = None

    if args.simulate:
        from .recorder import SimulatedSource

        source = SimulatedSource()

    elif args.replay is not None:
        from .recorder import ReplaySource

        try:
            source = ReplaySource(args.replay, speed=args.speed)

        except (OSError, KeyError, ValueError) as error:
            parser.error(f"cannot replay {args.replay}: {error}")

    from .window import open_window

    app, worker, _ = open_window(
        args.main_folder,
        project=args.project or None,
        user=args.user,
        can_create=args.create,
        recorder_source=source,
    )
    code = app.exec()
    worker.stop()

    return code


if __name__ == "__main__":
    sys.exit(main())
