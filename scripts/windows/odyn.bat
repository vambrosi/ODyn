@echo off
rem The ODyn entry app on the main-sync project, run from the caiman env.
rem The shortcut made by make_shortcut.ps1 opens this (window minimized);
rem the console behind the app shows its log, and stays open after an error.

set CONDA_ROOT=C:\Users\MossLab\miniforge3
set CONDA_ENV=caiman
set MAIN_FOLDER=Z:\ImagingData
set PROJECT=main-sync

rem Activating (not just running the env's python) puts the env's DLLs on the
rem PATH, which Qt and h5py need on Windows
call "%CONDA_ROOT%\Scripts\activate.bat" %CONDA_ENV%
if errorlevel 1 (
    echo Could not activate the %CONDA_ENV% environment in %CONDA_ROOT%.
    pause
    exit /b 1
)

if not exist "%MAIN_FOLDER%\.odyn" (
    echo %MAIN_FOLDER% is not reachable, or holds no .odyn folder. Is Z: mapped?
    pause
    exit /b 1
)

python -m odyn.app "%MAIN_FOLDER%" --project %PROJECT%
if errorlevel 1 pause
