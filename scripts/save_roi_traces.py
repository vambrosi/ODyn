#!/usr/bin/env python3
"""
Save the ROI traces of several groups, one after the other.

examples:
    python save_roi_traces.py --project PROJECT_NAME --groups 48 49 53
    python save_roi_traces.py --project PROJECT_NAME --groups 48 49 --import-masks

Each group needs a mask: either imported already, or found at its default
path with `--import-masks`.

A group that fails is reported and the rest still run, and the exit status is
non-zero if any of them did, for easy reference.
"""

import argparse

from logsetup import setup_logging

MAIN_FOLDER = "/home/groups/MossLab/ImagingData"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--main-folder", default=MAIN_FOLDER, help="where the DB lives")
    parser.add_argument("--project", help="pick ODyn project DB (optional)")
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        required=True,
        help="run these groups in turn",
    )
    parser.add_argument(
        "--photobleach-s",
        default=1.0,
        type=float,
        help="how many seconds to remove from the start",
    )
    parser.add_argument(
        "--import-masks",
        action="store_true",
        help="import each group's mask from the default path before saving",
    )
    parser.add_argument(
        "--all-mcors",
        action="store_true",
        help="pass to use all mcors (instead of only approved ones)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Delay import so --help doesn't have a long pause
    logger = setup_logging()

    from odyn import Database

    db = Database(args.main_folder, project=args.project)

    failed = []

    for group_id in args.groups:
        logger.info(f"=== group {group_id} {'=' * 40}")

        try:
            group = db.groups[group_id]

            if args.import_masks:
                group.import_mask()

            group.save_roi_traces(
                photobleach_window_s=args.photobleach_s,
                only_approved=not args.all_mcors,
            )

        # One unreadable movie or missing mask should not cost the whole batch,
        # and the call itself is already recorded with the exception in its log
        except Exception as error:
            logger.error(f"Group {group_id} failed: {error}")
            failed.append(group_id)

    if failed:
        logger.error(f"{len(failed)} of {len(args.groups)} groups failed: {failed}")
        raise SystemExit(1)

    logger.info(f"Saved traces for {len(args.groups)} groups.")


if __name__ == "__main__":
    main()
