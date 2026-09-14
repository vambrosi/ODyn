#!/usr/bin/env python3
"""
Copy odor panel recipes out of the lab odor workbook into the database.

USAGE
    python import_panels.py MAIN_FOLDER ODORS_XLSX [--sheet NAME]... [--project P]

Each named sheet becomes one panel, named after the sheet in lower case with
spaces turned into underscores, so `Print v3` registers as `print_v3`. Rerunning
it over an unchanged sheet does nothing, and over a corrected one replaces that
panel's vials -- unless a session has already run the panel, in which case the
recipe is what that session's trials mean and the run stops rather than
rewriting it. Rename the sheet to register a changed recipe alongside the old.

The workbook is the place the recipes are written and edited; this only takes a
snapshot, so that a vial position can be resolved to an odor from the database
alone. It never writes back.

SHEET LAYOUT
    A header row, then one row per vial, then a blank row and a footer. The
    columns this reads are the vial position, the delivered odor's number, the
    flows and volumes, and an `Odor A`/`Odor B` pair of number, ppm and ul.

Every odor number must already be in the `odors` table; the script lists the
ones that are not and stops without writing.

REQUIRES
    openpyxl to read the workbook.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odyn import Database  # noqa: E402

DEFAULT_SHEETS = ["Print v3"]

# Where each value sits in the two-row header, by the label of its second row.
# Matching on the label rather than the position means an inserted column moves
# the reader with it.
VIAL_COLUMNS = {
    "vial_position": "Vial pos",
    "odor_id": "Odor #",
    "odor_sccm": "Odor sccm",
    "total_sccm": "Total sccm",
    "total_volume_ml": "Total V (ml)",
    "solvent_volume_ml": "Mineral oil volume (ml)",
}

COMPONENT_COLUMNS = {
    "A": {"odor_id": "A #", "target_ppm": "A ppm", "liquid_ul": "A ul",
          "percent_vv": "Odor A % v/v"},
    "B": {"odor_id": "B #", "target_ppm": "B ppm", "liquid_ul": "B ul",
          "percent_vv": "Odor B % v/v"},
}


def panel_name(sheet: str) -> str:
    """`'Print v3'` -> `'print_v3'`, so the name survives being typed."""
    return "_".join(sheet.lower().split())


def header_index(rows: list[tuple]) -> None | tuple[int, dict[str, int]]:
    """
    Which row holds the labels, and which column each one is in.

    The labels wrap onto several lines inside their cells, so the line breaks
    are collapsed to single spaces: `'Vial\\npos'` is looked up as `'Vial pos'`.
    """
    for number, row in enumerate(rows[:4]):
        index = {}

        for column, cell in enumerate(row):
            if cell is None:
                continue

            label = " ".join(str(cell).split())

            if label:
                index.setdefault(label, column)

        if VIAL_COLUMNS["vial_position"] in index:
            return number, index

    return None


def cell(row: tuple, index: dict[str, int], label: str):
    """The value under `label`, or `None` if the sheet has no such column."""
    column = index.get(label)

    if column is None or column >= len(row):
        return None

    value = row[column]

    return None if isinstance(value, str) and not value.strip() else value


def read_panel(path: Path, sheet: str) -> list[dict]:
    """Every vial of one sheet, in the shape `Database.add_panel` takes."""
    book = openpyxl.load_workbook(path, data_only=True, read_only=True)

    try:
        rows = list(book[sheet].iter_rows(values_only=True))
    finally:
        book.close()

    found = header_index(rows)

    if found is None:
        raise ValueError(
            f"Sheet {sheet!r} of {path.name} has no "
            f"{VIAL_COLUMNS['vial_position']!r} column, so it is not a panel."
        )

    header_row, index = found
    vials = []

    for row in rows[header_row + 1:]:
        position = cell(row, index, VIAL_COLUMNS["vial_position"])

        # The vial rows run until the blank line before the footer, which holds
        # who mixed the rack and when -- per-session facts, kept elsewhere.
        if position is None:
            break

        # A position may carry footnote marks, as in '4**'.
        position = int("".join(c for c in str(position) if c.isdigit()))
        odor_id = cell(row, index, VIAL_COLUMNS["odor_id"])

        # A position left empty is a vial the rack does not use, not a vial
        # holding nothing: it has no odor to resolve to, so it gets no row.
        if odor_id is None:
            continue

        vial = {
            name: cell(row, index, label)
            for name, label in VIAL_COLUMNS.items()
        }

        vial["vial_position"] = position
        vial["odor_id"] = int(odor_id)

        vial["components"] = [
            component
            for labels in COMPONENT_COLUMNS.values()
            if (component := read_component(row, index, labels)) is not None
        ]

        vials.append(vial)

    return vials


def read_component(row: tuple, index: dict[str, int], labels: dict[str, str]):
    """One side of a mix, or `None` where the sheet leaves it blank."""
    odor_id = cell(row, index, labels["odor_id"])

    if odor_id is None:
        return None

    return {
        "odor_id": int(odor_id),
        **{
            name: cell(row, index, label)
            for name, label in labels.items()
            if name != "odor_id"
        },
    }


def unknown_odors(db: Database, vials: list[dict]) -> set[int]:
    """Odor numbers the sheet uses that the database has no name for."""
    known = set(db.odors.index)
    used = set()

    for vial in vials:
        used.add(int(vial["odor_id"]))
        used.update(int(part["odor_id"]) for part in vial["components"])

    return used - known


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("odors_xlsx", type=Path)
    parser.add_argument("--sheet", action="append", dest="sheets")
    parser.add_argument("--project", default=None)

    args = parser.parse_args()
    sheets = args.sheets or DEFAULT_SHEETS

    db = (
        Database(args.main_folder, project=args.project)
        if args.project
        else Database(args.main_folder)
    )

    for sheet in sheets:
        name = panel_name(sheet)
        vials = read_panel(args.odors_xlsx, sheet)

        if not vials:
            print(f"{sheet}: no vial rows found, skipping.")
            continue

        missing = unknown_odors(db, vials)

        if missing:
            print(
                f"{sheet}: odors {sorted(missing)} are not in the database. "
                f"Add them to `odors` first; nothing was written."
            )
            return 1

        try:
            db.add_panel(
                panel_name=name,
                vials=vials,
                description=f"Imported from {args.odors_xlsx.name}, sheet {sheet!r}.",
            )
        except ValueError as refused:
            print(f"{sheet}: {refused}")
            return 1

        mixes = sum(1 for vial in vials if len(vial["components"]) > 1)
        print(f"{sheet} -> {name}: {len(vials)} vials, {mixes} of them mixes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
