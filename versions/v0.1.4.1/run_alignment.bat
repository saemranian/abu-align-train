@echo off
chcp 65001 >nul
title 3D Photoscanning Alignment & Reconstruction Launcher
color 0A

echo ======================================================================
echo       3D Photoscanning Alignment & Reconstruction Launcher
echo ======================================================================
echo.

set "CONDA_PYTHON=C:\Users\USer\miniconda3\envs\vggt\python.exe"
set "SCRIPT_PATH=C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\vggt\align_cli.py"

if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Conda environment python not found at: %CONDA_PYTHON%
    pause
    exit /b 1
)

echo [Step 1/6] Select Reconstruction Engine:
echo   [1] VGGT       - Standard 1B (Tested, Ready & Cached on PC) [Default]
echo   [2] VGGT-Omega - Next-Gen Register Attention (Low VRAM, Auto-fallback if pending)
echo.
set "ENGINE_CHOICE=2"
set /p "ENGINE_CHOICE=Enter choice [1 or 2, default=2]: "

if "%ENGINE_CHOICE%"=="1" (
    set "ENGINE=vggt"
    echo Selected: VGGT (Standard 1B)
) else (
    set "ENGINE=vggt_omega"
    echo Selected: VGGT-Omega (Low VRAM)
)
echo.

echo [Step 2/6] Dataset Folder:
if "%~1"=="" (
    echo Default dataset:
    echo   C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Sony\101MSDCF\CC\Interior\RedBox\SEL Copy
    set "DATASET_DIR="
    set /p "DATASET_DIR=Enter folder path to photos (or press Enter for default): "
) else (
    set "DATASET_DIR=%~1"
)

if "%DATASET_DIR%"=="" (
    set "DATASET_DIR=C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Sony\101MSDCF\CC\Interior\RedBox\SEL Copy"
)
set DATASET_DIR=%DATASET_DIR:"=%
echo.

echo [Step 3/6] Image Stride / Keyframe Sampling:
set "STRIDE_VAL=1"
set /p "STRIDE_VAL=Enter stride (1 = all photos, 2 or 3 = skip frames, default=1): "
if "%STRIDE_VAL%"=="" (
    set "STRIDE_VAL=1"
)
echo.

echo [Step 4/6] Image Processing Resolution:
echo   [1] 1024 x 1024  (Standard - Optimal balance of VRAM and speed) [Default]
echo   [2] 1536 x 1536  (High Detail - Sharper colors & fine features)
echo   [3] 2048 x 2048  (Max Clarity - Best for 60MP Sony photos on RTX 4090)
echo.
set "RES_CHOICE=1"
set /p "RES_CHOICE=Enter choice [1-3, default=1]: "

set "LOAD_RES=1024"
if "%RES_CHOICE%"=="2" (
    set "LOAD_RES=1536"
    echo Selected: 1536 x 1536
)
if "%RES_CHOICE%"=="3" (
    set "LOAD_RES=2048"
    echo Selected: 2048 x 2048
)
if "%RES_CHOICE%"=="1" (
    echo Selected: 1024 x 1024
)
echo.

echo [Step 5/6] Point Cloud Quality & Alignment Mode:
echo   [1] Fast Preview      - 100K points, Conf 2.2 (~20 sec)
echo   [2] High Quality      - 300K points, Conf 1.8 [Default / Recommended]
echo   [3] Ultra Dense       - 800K points, Conf 1.5 (Photogrammetry scan grade)
echo   [4] Bundle Adjustment - ALIKED+SP tracking + Full Lens Distortion (k1-k4, t1-t2)
echo.
set "QUALITY_CHOICE=2"
set /p "QUALITY_CHOICE=Enter choice [1-4, default=2]: "

set "EXTRA_FLAGS=--conf_thresh 1.8 --max_points 300000"
if "%QUALITY_CHOICE%"=="1" (
    set "EXTRA_FLAGS=--conf_thresh 2.2 --max_points 100000"
    echo Selected: Fast Preview (100K points)
)
if "%QUALITY_CHOICE%"=="2" (
    set "EXTRA_FLAGS=--conf_thresh 1.8 --max_points 300000"
    echo Selected: High Quality (300K points)
)
if "%QUALITY_CHOICE%"=="3" (
    set "EXTRA_FLAGS=--conf_thresh 1.5 --max_points 800000"
    echo Selected: Ultra Dense (800K points)
)
if "%QUALITY_CHOICE%"=="4" (
    set "EXTRA_FLAGS=--conf_thresh 1.8 --max_points 300000 --use_ba"
    echo Selected: Deep Bundle Adjustment + Full Lens Distortion (k1-k4, t1-t2)
)
echo.

echo [Step 6/6] Output Folder Naming (Non-Overwriting):
echo   * Press Enter to automatically create the next unique folder (e.g. sparse_03_omega)
echo   * Previous results (sparse, sparse 1, sparse 2) will be 100%% preserved!
echo.
set "CUSTOM_OUT="
set /p "CUSTOM_OUT=Enter custom folder name (or press Enter for auto-naming): "

set "OUT_FLAG="
if not "%CUSTOM_OUT%"=="" (
    set "OUT_FLAG=--output_dir "%CUSTOM_OUT%""
    echo Selected custom output folder: %CUSTOM_OUT%
) else (
    echo Auto-generating next unique sparse folder...
)
echo.

echo ======================================================================
echo Engine:     %ENGINE%
echo Dataset:    %DATASET_DIR%
echo Stride:     %STRIDE_VAL%
echo Resolution: %LOAD_RES% x %LOAD_RES%
echo Settings:   %EXTRA_FLAGS%
echo Outputs:    COLMAP (.bin + .txt), RealityCapture (.csv), PLY (.ply)
echo ======================================================================
echo.

"%CONDA_PYTHON%" -u "%SCRIPT_PATH%" --input "%DATASET_DIR%" --engine "%ENGINE%" --stride %STRIDE_VAL% --load_res %LOAD_RES% %EXTRA_FLAGS% %OUT_FLAG%

echo.
echo ======================================================================
echo Reconstruction run complete!
echo Output saved in: %DATASET_DIR%\results\
echo.
echo Application Folders:
echo   [1] COLMAP_Lichtfield\       -^> For Lichtfield / NeRFstudio / Gaussian Splatting
echo                                   (cameras.txt, images.txt, points3D.txt, .bin)
echo   [2] Postshot_RealityCapture\ -^> For Postshot training ^& RealityCapture
echo                                   (cameras.csv, points.ply)
echo   * No redundant '0' subfolder created.
echo Press any key to exit...
echo ======================================================================
pause >nul
