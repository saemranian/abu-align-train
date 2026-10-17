@echo off
chcp 65001 >nul
title Abu Align & Train - PFGS360 Setup & Environment Builder

echo ======================================================================
echo    Abu Align & Train - PFGS360 Setup Helper
echo    (CVPR 2026: Pose-Free Omnidirectional Gaussian Splatting for 360 Videos)
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

:: 2. Install core Python dependencies
echo [*] Installing nerfstudio / PFGS360 foundational dependencies...
"%ENV_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || true
"%ENV_PYTHON%" -m pip install ninja jaxtyping typeguard rich open3d timm scikit-image || true

:: 3. Clone PFGS360
if not exist "PFGS360" (
    echo [*] Cloning PFGS360 repository...
    git clone https://github.com/zcq15/PFGS360.git
)

:: 4. Build gsplat360 / nerfstudio360 extensions if present
if exist "PFGS360" (
    cd PFGS360
    if exist "setup.py" (
        echo [*] Installing PFGS360 in editable mode...
        "%ENV_PYTHON%" -m pip install -e . --no-build-isolation || (
            echo [NOTICE] If CUDA compilation fails, ensure MSVC and CUDA Toolkit are on PATH.
        )
    )
    cd ..
)

:: 5. Create Checkpoints Directory for Depth Priors
if not exist "checkpoints" mkdir checkpoints
if not exist "checkpoints\pfgs360" mkdir checkpoints\pfgs360

echo.
echo ======================================================================
echo    PFGS360 Setup Completed!
echo    Ready for Pose-Free 360° GS training and OpenSfM / 360DVO handoff.
echo    Place UniK3D model weights into 'checkpoints\pfgs360\' if using local weights.
echo ======================================================================
echo.
pause
