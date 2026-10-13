import os

import pytest

from odyn.utils import logger


@pytest.fixture
def odyn_log(caplog):
    """`caplog` for odyn's logger, which does not pass records up to the root."""
    logger.addHandler(caplog.handler)

    yield caplog
    logger.removeHandler(caplog.handler)


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """
    Stand-ins for `sbatch`, `squeue` and `scancel` that log their arguments;
    `squeue` prints what `queue.txt` holds.
    """
    bin_folder = tmp_path / "bin"
    bin_folder.mkdir()
    calls = tmp_path / "slurm-calls.txt"

    for name, output in [
        ("sbatch", "echo 4242"),
        ("squeue", f'cat "{tmp_path}/queue.txt" 2>/dev/null || true'),
        ("scancel", "true"),
    ]:
        script = bin_folder / name
        script.write_text(f'#!/bin/sh\necho "{name} $*" >> "{calls}"\n{output}\n')
        script.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_folder}{os.pathsep}{os.environ['PATH']}")

    return calls
