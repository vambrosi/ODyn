"""Plain, timestamped logging for a SLURM output file."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Point odyn's logger at stdout, with timestamps and no color.

    A terminal renders ANSI escapes but a .out file only stores them, so INFO
    lines end up looking like '[[1;34mINFO[0m] ...' in the default formatting.

    The handler is replaced because odyn's logger does not propagate and
    installs its own, so adding one gives every line twice.

    stdout, because logging defaults to stderr and SLURM merges both into one
    file, where different buffering interleaves them out of order. One stream
    keeps them in sequence (with PYTHONUNBUFFERED=1 set by the job script).

    `record_call` adds and removes its own handler around each call, so what
    the database stores in `call_log` is unaffected by any of this.
    """
    from odyn.utils import logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
