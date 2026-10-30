"""ODyn's log on stdout, for a SLURM output file."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Point odyn's console log at stdout, at `level`.

    ODyn's log already has the time on every line, and no color codes when
    it goes to a file (see `odyn.utils`). stdout, because logging defaults
    to stderr and SLURM merges both into one file, where different buffering
    interleaves them out of order. One stream keeps them in sequence (with
    PYTHONUNBUFFERED=1 set by the job script).

    `record_call` adds and removes its own handler around each call, so each
    call's log file (`db.call_log(id)`) is unaffected by any of this.
    """
    from odyn.utils import logger

    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)

    logger.setLevel(level)

    return logger
