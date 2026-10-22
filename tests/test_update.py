"""
`odyn.app.update`: the app pulls its own clone at launch, only when safe.
A bare repository stands in for GitHub.
"""

import subprocess

from pathlib import Path

import pytest

from odyn.app.update import notices, update


def git(folder: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(folder), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit(folder: Path, name: str, text: str) -> None:
    (folder / name).write_text(text)
    git(folder, "add", name)
    git(folder, "commit", "-q", "-m", f"Change {name}")


@pytest.fixture
def clones(tmp_path):
    """GitHub (bare), the developer's clone and a user's, both on main."""
    hub = tmp_path / "hub.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(hub)], check=True)

    def clone(name):
        folder = tmp_path / name
        subprocess.run(["git", "clone", "-q", str(hub), str(folder)], check=True)
        git(folder, "config", "user.email", f"{name}@lab")
        git(folder, "config", "user.name", name)
        git(folder, "checkout", "-q", "-B", "main")
        return folder

    dev = clone("dev")
    commit(dev, "odyn.py", "1")
    git(dev, "push", "-q", "origin", "main")
    user = clone("user")

    return dev, user


def test_a_new_main_is_pulled(clones):
    dev, user = clones
    assert update(user).reason == "already up to date"

    commit(dev, "odyn.py", "2")
    git(dev, "push", "-q", "origin", "main")
    result = update(user)

    assert result.updated and not result.dependencies
    assert (user / "odyn.py").read_text() == "2"
    assert result.commit == git(dev, "log", "-1", "--format=%h")

    # New packages: said, not installed
    commit(dev, "pyproject.toml", "[project]")
    git(dev, "push", "-q", "origin", "main")
    assert update(user).dependencies


def test_nothing_is_pulled_when_it_is_not_safe(clones, tmp_path):
    dev, user = clones
    commit(dev, "odyn.py", "2")
    git(dev, "push", "-q", "origin", "main")

    # Local changes (untracked files do not count)
    (user / "odyn.py").write_text("mine")
    (user / "notes.txt").write_text("fine")
    assert update(user).reason == "the clone has local changes"
    git(user, "checkout", "-q", "--", "odyn.py")

    # Another branch: a developer's clone
    git(user, "checkout", "-q", "-b", "qt-gui")
    assert update(user).reason == "the clone is on qt-gui, not main"
    git(user, "checkout", "-q", "main")

    # Commits of its own: not a fast-forward
    commit(user, "local.py", "x")
    assert update(user).reason == "the clone has commits not on origin/main"
    git(user, "reset", "-q", "--hard", "HEAD~1")

    # No network (GitHub gone): skipped, not stuck
    (tmp_path / "hub.git").rename(tmp_path / "gone.git")
    result = update(user)
    assert not result.updated and result.reason.startswith("git fetch failed")


def test_the_restarted_app_says_what_was_updated(monkeypatch):
    monkeypatch.delenv("ODYN_UPDATED", raising=False)
    monkeypatch.delenv("ODYN_UPDATE_NOTE", raising=False)
    assert notices() == ("", "")

    monkeypatch.setenv("ODYN_UPDATED", "abc1234 (2026-10-21)")
    monkeypatch.setenv("ODYN_UPDATE_NOTE", "run pip install")
    assert notices() == ("ODyn updated to abc1234 (2026-10-21)", "run pip install")


def test_the_app_restarts_once_after_an_update(monkeypatch, tmp_path):
    import odyn.app.update as module

    restarted = []
    monkeypatch.delenv("ODYN_UPDATED", raising=False)
    monkeypatch.delenv("ODYN_NO_UPDATE", raising=False)
    monkeypatch.setattr(module, "clone_folder", lambda: tmp_path)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(
        module, "update", lambda root: module.Update(True, "", "abc1234", "2026-10-21")
    )
    monkeypatch.setattr(module.os, "execv", lambda exe, args: restarted.append(args))

    module.update_and_restart(["/data", "--project", "p"])
    assert restarted[0][1:] == ["-m", "odyn.app", "/data", "--project", "p"]
    assert module.os.environ["ODYN_UPDATED"] == "abc1234 (2026-10-21)"

    # The restarted app does not update again, nor one told not to
    module.update_and_restart(["/data"])
    monkeypatch.delenv("ODYN_UPDATED")
    module.update_and_restart(["/data", "--no-update"])
    assert len(restarted) == 1
