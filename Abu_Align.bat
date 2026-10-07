@echo off
chcp 65001 >nul
title Abu Align & Train Version 0.1.3.1
cd /d "%~dp0"

set "CONDA_PYTHON="
if exist "%USERPROFILE%\miniconda3\envs\abu_align\python.exe" set "CONDA_PYTHON=%USERPROFILE%\miniconda3\envs\abu_align\python.exe"
if not defined CONDA_PYTHON if exist "%USERPROFILE%\miniconda3\envs\vggt\python.exe" set "CONDA_PYTHON=%USERPROFILE%\miniconda3\envs\vggt\python.exe"
if not defined CONDA_PYTHON if exist "%~dp0env\python.exe" set "CONDA_PYTHON=%~dp0env\python.exe"

if not defined CONDA_PYTHON (
    echo [ERROR] Conda environment python not found!
    echo Please run install_abu_align.bat or set up the conda environment.
    pause
    exit /b 1
)

start "" "%CONDA_PYTHON%" "abu_align_gui.py"
exit
