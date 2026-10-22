"""
Keeping the app up to date: at every launch, before the window opens, the
app pulls its own git clone (`pip install -e` installs) and restarts once if
that brought anything new, so every module is the new version.

It only updates when that is safe, and otherwise says why in the log:
- the clone is on `main` (the branch the users follow; a developer's clone on
  another branch is left alone);
- nothing tracked was changed locally;
- `main` only moved ahead (a fast-forward).

Git never asks for a password here (`GIT_TERMINAL_PROMPT=0`) and is stopped
after `TIMEOUT_S`, so no network, or a slow one, never keeps the app from
opening. When an update changes `pyproject.toml` (new packages, say), the
app says to run `pip install` rather than installing anything itself.

Off with `--no-update` or `ODYN_NO_UPDATE=1`.

No Qt in here: it runs before the app is imported.
"""

from __future__ import annotations

import os
import subprocess
import sys

from dataclasses import dataclass
from pathlib import Path

from ..utils import logger

BRANCH = "main"
TIMEOUT_S = 15

# Set for the restarted app: what was updated, and what to do about it
UPDATED_ENV = "ODYN_UPDATED"
NOTE_ENV = "ODYN_UPDATE_NOTE"
OFF_ENV = "ODYN_NO_UPDATE"


@dataclass
class Update:
    """What `update` did: `updated`, or why not (`reason`)."""

    updated: bool = False
    reason: str = ""
    commit: str = ""  # the short commit now checked out, once updated
    date: str = ""  # and its date
    dependencies: bool = False  # `pyproject.toml` changed


def clone_folder() -> None | Path:
    """The git clone this ODyn runs from, or `None` (not a clone)."""
    root = Path(__file__).resolve().parents[2]

    return root if (root / ".git").exists() else None


def _git(root: Path, *args: str, timeout: float = TIMEOUT_S) -> str:
    """Run git in `root`; its output, or `RuntimeError` saying why not."""
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except FileNotFoundError:
        raise RuntimeError("git is not installed") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {args[0]} took more than {timeout:.0f} s") from None

    if done.returncode:
        problem = (done.stderr or done.stdout).strip().splitlines()
        raise RuntimeError(f"git {args[0]} failed: {problem[-1] if problem else ''}")

    return done.stdout.strip()


def update(root: Path, branch: str = BRANCH, timeout: float = TIMEOUT_S) -> Update:
    """Bring the clone in `root` up to `origin/<branch>`, if safe (see above)."""
    try:
        current = _git(root, "rev-parse", "--abbrev-ref", "HEAD")

        if current != branch:
            return Update(reason=f"the clone is on {current}, not {branch}")

        if _git(root, "status", "--porcelain", "--untracked-files=no"):
            return Update(reason="the clone has local changes")

        _git(root, "fetch", "--quiet", "origin", branch, timeout=timeout)
        old = _git(root, "rev-parse", "HEAD")
        new = _git(root, "rev-parse", f"origin/{branch}")

        if old == new:
            return Update(reason="already up to date")

        try:
            _git(root, "merge-base", "--is-ancestor", old, new)
        except RuntimeError:
            return Update(reason=f"the clone has commits not on origin/{branch}")

        changed = _git(root, "diff", "--name-only", old, new).splitlines()
        _git(root, "merge", "--ff-only", "--quiet", new)
        commit, date = _git(root, "log", "-1", "--format=%h %cs").split()

        return Update(
            updated=True,
            commit=commit,
            date=date,
            dependencies="pyproject.toml" in changed,
        )

    except RuntimeError as error:
        return Update(reason=str(error))


def update_and_restart(argv: list[str]) -> None:
    """
    Update before the app starts (unless turned off, or this is the restart
    after one), and restart it with the same `argv` if anything changed.
    """
    off = "--no-update" in argv or os.environ.get(OFF_ENV) == "1"
    root = clone_folder()

    if off or root is None or UPDATED_ENV in os.environ:
        return

    result = update(root)

    if not result.updated:
        logger.info(f"ODyn was not updated: {result.reason}.")
        return

    logger.info(f"ODyn updated to {result.commit} ({result.date}); restarting.")
    os.environ[UPDATED_ENV] = f"{result.commit} ({result.date})"

    if result.dependencies:
        os.environ[NOTE_ENV] = (
            "This update changed what ODyn needs installed. Close the app and run "
            f'pip install -e "{root}[gui]" in the caiman environment'
        )

    command = [sys.executable, "-m", "odyn.app", *argv]

    if sys.platform == "win32":
        # Windows has no real exec: run the new app and pass on how it ended
        sys.exit(subprocess.call(command))

    os.execv(sys.executable, command)


def notices() -> tuple[str, str]:
    """For the restarted app: what to show in the status bar, and a warning."""
    updated = os.environ.get(UPDATED_ENV, "")

    return (f"ODyn updated to {updated}" if updated else ""), os.environ.get(
        NOTE_ENV, ""
    )
