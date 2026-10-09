"""
`python -m odyn.app.install <main_folder> [--project NAME] [--name NAME]`

Makes a launcher that opens the app on `main_folder` with the Python running
this command (the `caiman` environment, say), with the ODyn name and icon:
- **Windows**: `<name>.bat` in `~/.odyn/`, which activates the conda
  environment (Qt and h5py need its DLLs on the PATH) and stays open after an
  error, and a desktop shortcut to it with the ODyn icon (its console starts
  minimized).
- **macOS**: `~/Applications/<name>.app`, which the Dock and the app switcher
  show by its name and icon. The bundle runs the environment's Python through
  a link inside it, so macOS takes the bundle for the application (see
  `macos_script`); its output goes to `~/.odyn/<name>.log`.

Run it again to change the folder or the project; a second `--name` makes a
second launcher (one on a test folder, say):
```
python -m odyn.app.install Z:\\ImagingData
python -m odyn.app.install Z:\\ImagingData\\tests --name "ODyn test"
```
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

from .draft import FOLDER_NAME

ASSETS = Path(__file__).parent / "assets"
ICON_PNG = ASSETS / "odyn.png"
ICON_ICO = ASSETS / "odyn.ico"

DEFAULT_PROJECT = "main-sync"


def install(
    main_folder: Path,
    project: str = DEFAULT_PROJECT,
    name: str = "ODyn",
    home: None | Path = None,
    platform: str = sys.platform,
) -> list[Path]:
    """
    Make the launcher for this system (see the module's description) and
    return what was written. `home` stands in for the user's home folder.
    """
    home = home or Path.home()

    if platform == "win32":
        return _windows(main_folder, project, name, home)

    if platform == "darwin":
        return _macos(main_folder, project, name, home)

    raise SystemExit(f"No launcher for {platform} (run python -m odyn.app instead)")


def _command(main_folder: Path, project: str) -> list[str]:
    """The arguments after the Python executable."""
    return ["-m", "odyn.app", str(main_folder), "--project", project]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #


def _conda_root(prefix: Path) -> None | Path:
    """The conda installation an environment belongs to, or `None`."""
    if not (prefix / "conda-meta").is_dir():
        return None

    return prefix.parent.parent if prefix.parent.name == "envs" else prefix


def windows_batch(main_folder: Path, project: str, prefix: Path, python: Path) -> str:
    """The `.bat` that runs the app from the environment at `prefix`."""
    root = _conda_root(prefix)
    lines = [
        "@echo off",
        "rem Made by python -m odyn.app.install (run it again to change this).",
        "rem The console behind the app shows its log, and stays open after an",
        "rem error.",
        "",
    ]

    if root is not None:
        # Activating (not just running the env's python) puts the env's DLLs
        # on the PATH, which Qt and h5py need
        lines += [
            f'call "{root}\\Scripts\\activate.bat" "{prefix}"',
            "if errorlevel 1 (",
            f"    echo Could not activate the environment in {prefix}.",
            "    pause",
            "    exit /b 1",
            ")",
            "",
        ]

    lines += [
        f'if not exist "{main_folder}\\{FOLDER_NAME}" (',
        f"    echo {main_folder} is not reachable, or holds no {FOLDER_NAME} "
        "folder. Is the drive mapped?",
        "    pause",
        "    exit /b 1",
        ")",
        "",
        f'"{python}" -m odyn.app "{main_folder}" --project {project}',
        "if errorlevel 1 pause",
        "",
    ]

    return "\r\n".join(lines)


def _windows(main_folder: Path, project: str, name: str, home: Path) -> list[Path]:
    folder = home / FOLDER_NAME
    folder.mkdir(parents=True, exist_ok=True)
    batch = folder / f"{name}.bat"
    batch.write_text(
        windows_batch(main_folder, project, Path(sys.prefix), Path(sys.executable)),
        newline="",
    )
    icon = folder / ICON_ICO.name
    shutil.copy2(ICON_ICO, icon)

    # WScript.Shell makes .lnk files; the console starts minimized (7) and
    # the app opens normally
    desktop = Path(
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[Environment]::GetFolderPath('Desktop')",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    link = desktop / f"{name}.lnk"
    script = "; ".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f"$link = $shell.CreateShortcut('{link}')",
            f"$link.TargetPath = '{batch}'",
            f"$link.WorkingDirectory = '{folder}'",
            f"$link.IconLocation = '{icon},0'",
            "$link.WindowStyle = 7",
            f"$link.Description = 'ODyn on {main_folder} ({project})'",
            "$link.Save()",
        ]
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)

    return [batch, icon, link]


# --------------------------------------------------------------------------- #
# macOS
# --------------------------------------------------------------------------- #


def macos_script(main_folder: Path, project: str, python: Path, log: Path) -> str:
    """
    The bundle's executable: the app through the link next to it, which makes
    macOS take this bundle for the running application. Should the link not
    start (a Python that cannot find its libraries from there), the
    environment's own Python runs the app instead, under its own name.
    """
    arguments = " ".join(_quote(a) for a in _command(main_folder, project))

    return (
        "#!/bin/sh\n"
        "# Made by python -m odyn.app.install (run it again to change this).\n"
        'here="$(cd "$(dirname "$0")" && pwd)"\n'
        f"mkdir -p {_quote(str(log.parent))}\n"
        f"exec >> {_quote(str(log))} 2>&1\n"
        'python="$here/python"\n'
        '"$python" -c "import encodings" || python=' + _quote(str(python)) + "\n"
        f'exec "$python" {arguments}\n'
    )


def macos_info(name: str) -> dict:
    """The bundle's `Info.plist`."""
    return {
        "CFBundleName": name,
        "CFBundleDisplayName": name,
        "CFBundleIdentifier": "edu.odyn." + "".join(c for c in name if c.isalnum()),
        "CFBundleExecutable": "launch",
        "CFBundleIconFile": "odyn.icns",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
    }


def _macos(main_folder: Path, project: str, name: str, home: Path) -> list[Path]:
    bundle = home / "Applications" / f"{name}.app"
    contents = bundle / "Contents"

    if bundle.exists():
        shutil.rmtree(bundle)

    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()

    with open(contents / "Info.plist", "wb") as file:
        plistlib.dump(macos_info(name), file)

    log = home / FOLDER_NAME / f"{name}.log"
    script = contents / "MacOS" / "launch"
    script.write_text(macos_script(main_folder, project, Path(sys.executable), log))
    script.chmod(0o755)
    os.symlink(sys.executable, contents / "MacOS" / "python")
    _macos_icon(contents / "Resources" / "odyn.icns")

    # Launch Services caches a bundle's name and icon; this refreshes them
    subprocess.run(["touch", str(bundle)], check=False)

    return [bundle]


def _macos_icon(target: Path) -> None:
    """`odyn.png` as an `.icns`, with the tools every Mac has (else none)."""
    with tempfile.TemporaryDirectory() as temporary:
        iconset = Path(temporary) / "odyn.iconset"
        iconset.mkdir()

        try:
            for size in (16, 32, 128, 256, 512):
                for scale in (1, 2):
                    suffix = "" if scale == 1 else "@2x"
                    subprocess.run(
                        [
                            "sips",
                            "-z",
                            str(size * scale),
                            str(size * scale),
                            str(ICON_PNG),
                            "--out",
                            str(iconset / f"icon_{size}x{size}{suffix}.png"),
                        ],
                        check=True,
                        capture_output=True,
                    )

            subprocess.run(
                ["iconutil", "-c", "icns", str(iconset), "-o", str(target)],
                check=True,
                capture_output=True,
            )

        except (OSError, subprocess.CalledProcessError) as error:
            print(f"No icon for the app (could not make odyn.icns: {error})")


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def main(argv: None | list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m odyn.app.install",
        description=(
            "Make a launcher for the ODyn app (a desktop shortcut on Windows, "
            "an app in ~/Applications on macOS) that runs it with this Python."
        ),
    )
    parser.add_argument(
        "main_folder", type=Path, help="the folder that contains .odyn/ and the data"
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f'the database project (default: "{DEFAULT_PROJECT}")',
    )
    parser.add_argument(
        "--name",
        default="ODyn",
        help='the launcher\'s name (default: "ODyn"; another name makes a second one)',
    )
    args = parser.parse_args(argv)

    if not (args.main_folder / FOLDER_NAME).is_dir():
        print(
            f"Note: {args.main_folder} holds no {FOLDER_NAME} folder from here "
            "(the launcher checks again when it starts)"
        )

    for path in install(args.main_folder, args.project, args.name):
        print(f"Made {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
