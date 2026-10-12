@echo off
chcp 65001 >nul
title Abu Align - 360DVO Setup & Environment Builder

echo ======================================================================
echo    Abu Align & Train - 360DVO Deep Visual Odometry Setup Helper
echo    (IEEE RA-L 2026: Monocular 360-Degree Deep Visual Odometry)
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python
set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\abu_align\python.exe"
if not exist "%ENV_PYTHON%" set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\vggt\python.exe"
if not exist "%ENV_PYTHON%" (
    echo [ERROR] Conda environment Python not found.
    pause
    exit /b 1
)

:: 2. Install evo and basic requirements
echo [*] Installing evo and visualization dependencies...
"%ENV_PYTHON%" -m pip install evo viser pyyaml scikit-image || true

:: 3. Clone 360DVO
if not exist "360DVO" (
    echo [*] Cloning 360DVO repository...
    git clone https://github.com/chris1004336379/360DVO.git
)

:: 4. Download Eigen 3.4.0 if not present
if exist "360DVO" (
    cd 360DVO
    if not exist "thirdparty\eigen-3.4.0" (
        echo [*] Downloading Eigen 3.4.0 for ODBA bundle adjustment solver...
        powershell -Command "Invoke-WebRequest -Uri 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip' -OutFile 'eigen.zip'; Expand-Archive -Path 'eigen.zip' -DestinationPath 'thirdparty'; Remove-Item 'eigen.zip'"
    )
    
    echo [*] Building 360DVO CUDA & C++ extensions...
    "%ENV_PYTHON%" -m pip install -e . --no-build-isolation || (
        echo [NOTICE] If CUDA extension build requires MSVC build tools, 360DVO can run in pure PyTorch/fallback mode.
    )
    cd ..
)

:: 5. Create Checkpoints Directory
if not exist "checkpoints" mkdir checkpoints
echo.
echo ======================================================================
echo    360DVO Setup Completed!
echo    Place '360dvo.pth' weights into the 'checkpoints\' folder.
echo ======================================================================
echo.
pause
