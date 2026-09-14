"""
Launch the entry app.

    python -m odyn.app MAIN_FOLDER [--project NAME]

Qt is an optional dependency, so a missing PySide6 is reported as something to
install rather than as a traceback.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("--project", default=None)

    arguments = parser.parse_args()

    try:
        from .window import run
    except ImportError as missing:
        print(
            f"The entry app needs Qt, which is not installed ({missing}).\n"
            f"Install it with:  pip install -e '.[gui]'",
            file=sys.stderr,
        )

        return 1

    return run(arguments.main_folder, project=arguments.project)


if __name__ == "__main__":
    raise SystemExit(main())
