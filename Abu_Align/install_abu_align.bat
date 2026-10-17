@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Abu Align & Train - Automated One-Click Installer for Windows

echo ======================================================================
echo    Abu Align & Train - Automated One-Click Installer (Windows)
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Locate Conda executable
echo [*] Checking Conda installation...
set "CONDA_EXE="
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "CONDA_EXE=conda"
) else (
    if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_EXE=%USERPROFILE%\miniconda3\condabin\conda.bat"
    if not defined CONDA_EXE if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_EXE=%USERPROFILE%\anaconda3\condabin\conda.bat"
    if not defined CONDA_EXE if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_EXE=%ProgramData%\miniconda3\condabin\conda.bat"
    if not defined CONDA_EXE if exist "C:\ProgramData\anaconda3\condabin\conda.bat" set "CONDA_EXE=C:\ProgramData\anaconda3\condabin\conda.bat"
)

if not defined CONDA_EXE (
    echo [!] Conda was not found in PATH or standard installation locations.
    echo [*] Downloading and installing Miniconda silently for you...
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile '$env:TEMP\miniconda_installer.exe'"
    if exist "%TEMP%\miniconda_installer.exe" (
        echo [*] Running Miniconda installation to %USERPROFILE%\miniconda3 ...
        start /wait "" "%TEMP%\miniconda_installer.exe" /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\miniconda3
        del /f /q "%TEMP%\miniconda_installer.exe" >nul 2>&1
        set "CONDA_EXE=%USERPROFILE%\miniconda3\condabin\conda.bat"
    ) else (
        echo [ERROR] Failed to download Miniconda installer. Please install Miniconda manually from https://docs.conda.io/
        pause
        exit /b 1
    )
)

echo [OK] Conda detected: %CONDA_EXE%
echo.

:: 2. Create or verify abu_align conda environment
echo [*] Checking / Creating Conda environment 'abu_align' (Python 3.11)...
call %CONDA_EXE% create -y -n abu_align python=3.11
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Conda create returned an error, checking if 'abu_align' already exists...
)

:: Locate Python in the environment
set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\abu_align\python.exe"
if not exist "%ENV_PYTHON%" (
    if exist "%USERPROFILE%\anaconda3\envs\abu_align\python.exe" (
        set "ENV_PYTHON=%USERPROFILE%\anaconda3\envs\abu_align\python.exe"
    )
)

if not exist "%ENV_PYTHON%" (
    echo [ERROR] Failed to verify Python in 'abu_align' environment at: %ENV_PYTHON%
    pause
    exit /b 1
)

echo [OK] Python located: %ENV_PYTHON%
echo.

:: 3. Install PyTorch with CUDA 12.1
echo [*] Installing PyTorch 2.3.1 + CUDA 12.1...
"%ENV_PYTHON%" -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] PyTorch install returned error code, trying pip install with retry...
    "%ENV_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
echo.

:: 4. Install requirements
echo [*] Installing Abu Align dependencies from requirements.txt...
if exist "requirements.txt" (
    "%ENV_PYTHON%" -m pip install -r requirements.txt
) else (
    "%ENV_PYTHON%" -m pip install numpy==1.26.4 Pillow huggingface_hub einops safetensors scipy trimesh opencv-python pycolmap roma timm lightglue psutil rich tqdm matplotlib plyfile viser
)
echo.

:: 5. Install editable repositories if present
echo [*] Installing local package dependencies...
if exist "setup.py" (
    "%ENV_PYTHON%" -m pip install -e . --no-deps
)
if exist "..\vggt-omega\setup.py" (
    "%ENV_PYTHON%" -m pip install -e "..\vggt-omega" --no-deps
)

:: 6. Compile Native Launcher Executable
echo [*] Compiling native Windows C# Launchers...
set "CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if exist "%CSC%" if exist "Launcher.cs" (
    "%CSC%" /target:winexe /out:Abu_Align.exe Launcher.cs >nul 2>&1
    copy /y Abu_Align.exe Abu_Align_and_Train.exe >nul 2>&1
    echo [OK] Successfully compiled Abu_Align.exe and Abu_Align_and_Train.exe
)

:: 7. Create Desktop Shortcut
echo [*] Creating Desktop Shortcut...
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'Abu Align and Train.lnk')); $s.TargetPath = [System.IO.Path]::Combine('%~dp0', 'Abu_Align.exe'); $s.WorkingDirectory = '%~dp0'; $s.Description = 'Abu Align and Train - Neural Photogrammetry Suite'; $s.Save()"

echo.
echo ======================================================================
echo    Installation Completed Successfully!
echo    You can now launch Abu Align & Train via:
echo      1. Desktop Shortcut: 'Abu Align and Train'
echo      2. Double-clicking: Abu_Align.exe or Abu_Align.bat
echo ======================================================================
echo.
pause
