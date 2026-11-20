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
        "20260707/m1/e1,alice,20,x.xlsx\n"
        "20260707/m1/e2,alice,10,x.xlsx\n"
        "20260707/m2/e1,bob,,x.xlsx\n"
        "20260707/m3/e1,bob,20,x.xlsx\n"
    )

    # Said first, a missing folder too
    assert (
        run(tmp_path, table, "--project", "old", "--dry-run", monkeypatch=monkeypatch)
        == 1
    )
    said = capsys.readouterr().out
    assert "would add 20260707/m1/e1 as alice, 20x" in said
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
    assert {experiment_owner(db, int(g)) for g in links.index} == {"alice"}

    # Again: nothing twice
    run(tmp_path, table, "--project", "old", monkeypatch=monkeypatch)
    assert "20260707/m1/e1: already added" in capsys.readouterr().out
    assert len(Database(tmp_path, project="old").experiments) == 2


def test_grabs_are_added_with_their_conditions(tmp_path, monkeypatch, capsys):
    # Three long grabs in their own folders, named for their condition (as
    # another lab records them), and a loop
    names = {"e1": "Baseline", "e2": "50Hz_2ms", "e3": "50Hz_6ms"}

    for seed, (exp, name) in enumerate(names.items()):
        generate(
            tmp_path,
            acquisitions=1,
            frames=112,
            height=32,
            width=40,
            exp=exp,
            seed=seed,
            acq_state="grab",
            start=datetime(2026, 7, 7, 9 + seed),
        )
        raw = tmp_path / f"20260707/m1/{exp}/raw"
        [tiff] = raw.glob("*.tif")
        tiff.rename(raw / f"{name}_00001.tif")

    generate(
        tmp_path,
        acquisitions=2,
        frames=56,
        height=32,
        width=40,
        exp="e4",
        start=datetime(2026, 7, 7, 13),
    )
    Database(tmp_path, project="old", can_create=True, user="admin")
    table = tmp_path / "old.csv"
    table.write_text(
        "exp_path,user,objective,exp_type,condition\n"
        "20260707/m1/e1,alice,20,grab,\n"
        "20260707/m1/e2,alice,20,grab,\n"
        "20260707/m1/e3,alice,20,grab,6 ms\n"
        "20260707/m1/e4,alice,20,,ket/xyl\n"
        "20260707/m1/e5,alice,20,stack,\n"
    )

    # Said first; the unknown type fails
    assert (
        run(tmp_path, table, "--project", "old", "--dry-run", monkeypatch=monkeypatch)
        == 1
    )
    said = capsys.readouterr().out
    assert "would add 20260707/m1/e1 as alice, 20x, a grab, condition Baseline" in said
    assert "would add 20260707/m1/e3 as alice, 20x, a grab, condition 6 ms" in said
    assert "would add 20260707/m1/e4 as alice, 20x, condition ket/xyl" in said
    assert "20260707/m1/e5: exp_type is one of loop, grab" in said

    run(tmp_path, table, "--project", "old", monkeypatch=monkeypatch)
    assert "Added (4)" in capsys.readouterr().out
    db = Database(tmp_path, project="old")
    experiments = db.experiments.set_index("exp_path")
    assert experiments["exp_type"].to_dict() == {
        "20260707/m1/e1": "grab",
        "20260707/m1/e2": "grab",
        "20260707/m1/e3": "grab",
        "20260707/m1/e4": "loop",
    }

    # Every acquisition with its condition; the new ones are options now
    conditions = db.annotations_for("acquisition")["condition"]
    by_experiment = (
        db.acquisitions.join(conditions)
        .join(db.experiments["exp_path"], on="exp_id")
        .groupby("exp_path")["condition"]
        .agg(lambda values: sorted(set(values)))
    )
    assert by_experiment.to_dict() == {
        "20260707/m1/e1": ["baseline"],  # an option already, whatever the case
        "20260707/m1/e2": ["50Hz_2ms"],
        "20260707/m1/e3": ["6 ms"],
        "20260707/m1/e4": ["ket/xyl"],
    }
    options = db.annotation_keys.loc[("acquisition", "condition"), "allowed_values"]
    assert list(options) == ["baseline", "ket/xyl", "50Hz_2ms", "6 ms"]
