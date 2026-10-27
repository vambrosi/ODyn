"""
How much room each experiment takes on the share, by part, to compare before
and after the motion corrections are run again.

`python scripts/space_report.py <main_folder> [--only paths.txt] [--out FILE]`

Every experiment folder (`YYYYMMDD/<mouse>/<exp>`) is measured by part: the
files at its top (the raw TIFFs), each of its folders (`sync`, `raw`, ...)
and each folder of `processed/` on its own (`processed/mcor`,
`processed/patchwarp`, ...). For each part: how many files and their bytes,
the room they take on the disk, and the largest kinds of file. `--only`
keeps the experiments listed in a file, one `YYYYMMDD/<mouse>/<exp>` a line.

Writes one line per experiment and part to a CSV (by default
`space-<date>.csv` here) and prints the totals by part. Reading only: nothing
on the share changes. Run it on a login node or with `srun`, as it walks
every file.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

DAY = re.compile(r"^\d{8}$")


def experiments(main: Path, only: None | set[str]):
    """The experiment folders: `YYYYMMDD/<mouse>/<exp>`."""
    for day in sorted(main.iterdir()):
        if not (day.is_dir() and DAY.match(day.name)):
            continue

        for mouse in sorted(p for p in day.iterdir() if p.is_dir()):
            for exp in sorted(p for p in mouse.iterdir() if p.is_dir()):
                key = f"{day.name}/{mouse.name}/{exp.name}"

                if only is None or key in only:
                    yield key, exp


def walk(folder: Path):
    """Every file under `folder`, as `(path, size, allocated)`, links skipped."""
    stack = [folder]

    while stack:
        here = stack.pop()

        try:
            entries = list(os.scandir(here))
        except OSError as error:
            print(f"  cannot read {here}: {error}", file=sys.stderr)
            continue

        for entry in entries:
            try:
                if entry.is_symlink():
                    continue

                if entry.is_dir():
                    stack.append(Path(entry.path))
                elif entry.is_file():
                    info = entry.stat()
                    yield entry.name, info.st_size, info.st_blocks * 512
            except OSError:
                continue


def kind(name: str) -> str:
    """A file's kind, by its extension (`.tif`, `.h5`; `.gzip.h5` as one)."""
    suffixes = Path(name).suffixes[-2:]

    return "".join(suffixes).lower() if suffixes else "(none)"


def parts(exp: Path):
    """An experiment's parts, as `(name, files)`: see the module description."""
    top = []

    for entry in sorted(os.scandir(exp), key=lambda e: e.name):
        if entry.is_symlink():
            continue

        if entry.is_file():
            info = entry.stat()
            top.append((entry.name, info.st_size, info.st_blocks * 512))
        elif entry.is_dir() and entry.name == "processed":
            loose = []

            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_dir() and not sub.is_symlink():
                    yield f"processed/{sub.name}", walk(Path(sub.path))
                elif sub.is_file():
                    info = sub.stat()
                    loose.append((sub.name, info.st_size, info.st_blocks * 512))

            if loose:
                yield "processed/(files)", iter(loose)
        elif entry.is_dir():
            yield entry.name, walk(Path(entry.path))

    if top:
        yield "(top)", iter(top)


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"

        size /= 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("main_folder", type=Path)
    parser.add_argument("--only", type=Path, help="experiments to measure, a line each")
    parser.add_argument("--out", type=Path, default=Path(f"space-{date.today()}.csv"))
    args = parser.parse_args()

    only = None

    if args.only:
        only = {
            line.strip().strip("/").removeprefix("./").removesuffix("/sync")
            for line in args.only.read_text().splitlines()
            if line.strip()
        }

    totals = defaultdict(lambda: [0, 0, 0, 0])  # experiments, files, bytes, disk
    count = 0

    with open(args.out, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(
            ["experiment", "part", "files", "bytes", "disk_bytes", "largest_kinds"]
        )

        for key, exp in experiments(args.main_folder, only):
            count += 1
            print(f"  {key}", file=sys.stderr)

            for part, files in parts(exp):
                n = size = disk = 0
                kinds = Counter()

                for name, b, d in files:
                    n += 1
                    size += b
                    disk += d
                    kinds[kind(name)] += b

                listed = "; ".join(f"{k} {human(b)}" for k, b in kinds.most_common(3))
                out.writerow([key, part, n, size, disk, listed])
                total = totals[part]
                total[0] += 1
                total[1] += n
                total[2] += size
                total[3] += disk

        if only:
            missing = sorted(
                only - {key for key, _ in experiments(args.main_folder, only)}
            )

            for key in missing:
                out.writerow([key, "(missing)", 0, 0, 0, ""])

    print(
        f"\n{count} experiments in {args.main_folder}; per experiment in {args.out}\n"
    )
    print(f"{'part':<26}{'experiments':>12}{'files':>10}{'size':>12}{'on disk':>12}")

    for part, (exps, n, size, disk) in sorted(
        totals.items(), key=lambda item: -item[1][3]
    ):
        print(f"{part:<26}{exps:>12}{n:>10}{human(size):>12}{human(disk):>12}")

    all_size = sum(t[2] for t in totals.values())
    all_disk = sum(t[3] for t in totals.values())
    print(
        f"{'all':<26}{'':>12}{sum(t[1] for t in totals.values()):>10}"
        f"{human(all_size):>12}{human(all_disk):>12}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
