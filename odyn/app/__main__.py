"""
`python -m odyn.app <main_folder> [--project NAME] [--user NAME] [--create]`
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


def main(argv: None | list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odyn.app",
        description="The ODyn entry app: register, record, annotate, submit, check.",
    )
    parser.add_argument(
        "main_folder", type=Path, help="the folder holding .odyn/ and the data"
    )
    parser.add_argument(
        "--project",
        default=None,
        help="the database project (default: the project-less one)",
    )
    parser.add_argument(
        "--user", default=None, help="who is using the app (default: the login name)"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="make the database if it does not exist yet",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="record from a simulated source, not the DAQ",
    )
    args = parser.parse_args(argv)

    from .window import open_window

    source = None

    if args.simulate:
        from .recorder import SimulatedSource

        source = SimulatedSource()

    app, worker, _ = open_window(
        args.main_folder,
        project=args.project,
        user=args.user,
        can_create=args.create,
        recorder_source=source,
    )
    code = app.exec()
    worker.stop()

    return code


if __name__ == "__main__":
    sys.exit(main())
