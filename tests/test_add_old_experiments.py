"""`scripts/add_old_experiments.py`: experiments from before the app."""

import runpy
import sys

from datetime import datetime
from pathlib import Path

from generate_data import generate

from odyn import Database
from odyn.cluster import experiment_owner

SCRIPT = Path(__file__).parents[1] / "scripts" / "add_old_experiments.py"


def run(*args, monkeypatch) -> int:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *map(str, args)])

    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as done:
        return done.code

    return 0


def test_old_experiments_are_added_as_who_recorded_them(tmp_path, monkeypatch, capsys):
    for seed, (mouse, exp) in enumerate((("m1", "e1"), ("m1", "e2"), ("m2", "e1"))):
        generate(
            tmp_path,
            acquisitions=2,
            frames=56,
            height=32,
            width=40,
            mouse=mouse,
            exp=exp,
            seed=seed,
            start=datetime(2026, 7, 7, 9 + seed),
        )

    Database(tmp_path, project="old", can_create=True, user="admin")
    table = tmp_path / "old.csv"
    table.write_text(
        "exp_path,user,objective,log\n"
        "20260707/m1/e1,ana,20,x.xlsx\n"
        "20260707/m1/e2,ana,10,x.xlsx\n"
        "20260707/m2/e1,bo,,x.xlsx\n"
        "20260707/m3/e1,bo,20,x.xlsx\n"
    )

    # Said first, a missing folder too
    assert (
        run(tmp_path, table, "--project", "old", "--dry-run", monkeypatch=monkeypatch)
        == 1
    )
    said = capsys.readouterr().out
    assert "would add 20260707/m1/e1 as ana, 20x" in said
    assert "20260707/m2/e1: no objective" in said
    assert "20260707/m3/e1: no such folder" in said
    assert Database(tmp_path, project="old").experiments.empty

    # Added as who recorded them; the rest go on past a failure
    assert run(tmp_path, table, "--project", "old", monkeypatch=monkeypatch) == 1
    said = capsys.readouterr().out
    assert "Added (2)" in said and "Failed (1)" in said
    db = Database(tmp_path, project="old")
    assert sorted(db.experiments["exp_path"]) == ["20260707/m1/e1", "20260707/m1/e2"]
    assert sorted(db.experiments["objective"].tolist()) == [10, 20]
    assert len(db.sessions) == 1  # one mouse, one day
    links = db.group_experiments
    assert {experiment_owner(db, int(g)) for g in links.index} == {"ana"}

    # Again: nothing twice
    run(tmp_path, table, "--project", "old", monkeypatch=monkeypatch)
    assert "20260707/m1/e1: already added" in capsys.readouterr().out
    assert len(Database(tmp_path, project="old").experiments) == 2
