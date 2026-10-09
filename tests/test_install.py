"""
The launchers `python -m odyn.app.install` makes, built here without the
system they are for: the macOS bundle in a stand-in home folder (and its
script run on this system), the Windows batch file as text.
"""

import plistlib
import subprocess
import sys

from pathlib import Path, PureWindowsPath

from odyn.app.install import install, windows_batch


def test_the_macos_bundle_runs_the_app_with_this_python(tmp_path):
    (bundle,) = install(
        tmp_path / "share", "main-sync", "ODyn test", tmp_path, platform="darwin"
    )

    assert bundle == tmp_path / "Applications" / "ODyn test.app"
    contents = bundle / "Contents"
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    assert info["CFBundleName"] == "ODyn test"
    assert info["CFBundleExecutable"] == "launch"
    assert (contents / "MacOS" / "python").resolve() == Path(sys.executable).resolve()

    # The script, with the app swapped for a line that says what it got
    script = (contents / "MacOS" / "launch").read_text()
    assert f"'-m' 'odyn.app' '{tmp_path / 'share'}' '--project' 'main-sync'" in script
    probe = contents / "MacOS" / "probe"
    probe.write_text(
        script.replace("'-m' 'odyn.app'", "-c 'import sys; print(sys.argv[1:])'")
    )
    probe.chmod(0o755)
    subprocess.run([str(probe)], check=True)

    log = tmp_path / ".odyn" / "ODyn test.log"
    assert log.read_text().strip() == str(
        [str(tmp_path / "share"), "--project", "main-sync"]
    )

    # Again: the bundle is made anew, not stacked on the old one
    install(tmp_path / "other", "p", "ODyn test", tmp_path, platform="darwin")
    assert "other" in (contents / "MacOS" / "launch").read_text()


def test_the_macos_script_falls_back_to_the_python_itself(tmp_path):
    (bundle,) = install(tmp_path / "share", home=tmp_path, platform="darwin")
    macos = bundle / "Contents" / "MacOS"
    (macos / "python").unlink()
    (macos / "python").write_text("#!/bin/sh\nexit 1\n")  # a link that cannot start
    (macos / "python").chmod(0o755)
    script = (macos / "launch").read_text()
    (macos / "launch").write_text(
        script.replace("'-m' 'odyn.app'", "-c 'import sys; print(sys.executable)'")
    )
    subprocess.run([str(macos / "launch")], check=True)

    last = (tmp_path / ".odyn" / "ODyn.log").read_text().strip().splitlines()[-1]
    assert Path(last).resolve() == Path(sys.executable).resolve()


def test_the_windows_batch_runs_the_app_once_the_folder_is_there(tmp_path):
    # A Python outside conda is run as it is
    text = windows_batch(
        PureWindowsPath(r"Z:\ImagingData"),
        "main-sync",
        tmp_path,
        PureWindowsPath(r"C:\Users\MossLab\miniforge3\envs\caiman\python.exe"),
    )
    lines = text.split("\r\n")

    assert not any("activate.bat" in line for line in lines)
    assert r'if not exist "Z:\ImagingData\.odyn" (' in lines
    assert (
        r'"C:\Users\MossLab\miniforge3\envs\caiman\python.exe" -m odyn.app '
        r'"Z:\ImagingData" --project main-sync'
    ) in lines


def test_the_windows_batch_finds_the_conda_root(tmp_path):
    from odyn.app.install import _conda_root

    env = tmp_path / "miniforge3" / "envs" / "caiman"
    (env / "conda-meta").mkdir(parents=True)
    (tmp_path / "miniforge3" / "conda-meta").mkdir()

    assert _conda_root(env) == tmp_path / "miniforge3"
    assert _conda_root(tmp_path / "miniforge3") == tmp_path / "miniforge3"
    assert _conda_root(tmp_path) is None

    text = windows_batch(Path("Z:/x"), "p", env, env / "python.exe")
    assert f'call "{tmp_path / "miniforge3"}\\Scripts\\activate.bat" "{env}"' in text
