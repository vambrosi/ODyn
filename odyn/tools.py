"""
Development tools that do not use a database.

**COMMAND LINE**
```
python -m odyn.tools diagram     # regenerate odyn/schema.svg from create.sql
```
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys

from pathlib import Path

from .utils import logger

CREATE_SCRIPT = Path(__file__).parent / "create.sql"
DIAGRAM_SCRIPT = Path(__file__).parent / "diagram.sql"
SCHEMA_DIAGRAM = Path(__file__).parent / "schema.svg"


def generate_diagram() -> None:
    """
    Regenerate `schema.svg` from `create.sql` (needs GraphViz `dot`).

    Builds a in-memory DB from the current schema, renders it to GraphViz
    DOT via `diagram.sql`, and pipes that through `dot`. Skips
    with a warning if `dot` is not installed.
    """

    con = sqlite3.connect(":memory:")

    try:
        con.executescript(CREATE_SCRIPT.read_text())
        rows = con.execute(DIAGRAM_SCRIPT.read_text()).fetchall()
    finally:
        con.close()

    # SQLite3 CLI joins result rows with newlines.
    # We do the same here so the DOT statements stay on separate lines.
    dot = "\n".join((row[0] or "") for row in rows)

    try:
        result = subprocess.run(
            ["dot", "-Tsvg"], input=dot, capture_output=True, text=True, check=True
        )

    except FileNotFoundError:
        logger.warning("GraphViz 'dot' not found; skipping schema.svg regeneration.")
        return

    except subprocess.CalledProcessError as error:
        logger.warning(f"Could not generate schema diagram: {error.stderr.strip()}")
        return

    SCHEMA_DIAGRAM.write_text(result.stdout)
    logger.info(f"Regenerated schema diagram at '{SCHEMA_DIAGRAM}'.")


def main(argv: None | list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odyn.tools",
        description="Development tools that do not use a database.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("diagram", help="regenerate schema.svg (needs GraphViz)")

    args = parser.parse_args(argv)

    if args.command == "diagram":
        generate_diagram()

    return 0


if __name__ == "__main__":
    sys.exit(main())
