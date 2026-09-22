@echo off
chcp 65001 >nul
title Abu Align & Train Version 0.0.1.2
cd /d "%~dp0"
set "CONDA_PYTHON=C:\Users\USer\miniconda3\envs\vggt\python.exe"

if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Python environment not found at: %CONDA_PYTHON%
    pause
    exit /b 1
)

start "" "%CONDA_PYTHON%" "abu_align_gui.py"
exit
