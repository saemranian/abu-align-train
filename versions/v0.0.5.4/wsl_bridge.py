"""
Abu Align & Train - WSL2 Ubuntu Linux Bridge
============================================
Seamlessly connects the Windows desktop GUI and CLI with the native Linux
environments inside WSL2 Ubuntu for high-performance GPU-accelerated research engines:
1. Mapillary OpenSfM (C++ Ceres Solver with S^2 Equirectangular SfM)
2. 360DVO (IEEE RA-L 2026 DPVO Omnidirectional Differentiable Bundle Adjustment on RTX 4090)
3. NVIDIA PyCuSFM (CUDA-accelerated Structure-from-Motion with TensorRT)
4. PFGS360 (CVPR 2026 Pose-Free Omnidirectional Gaussian Splatting + Consistent Depth Priors)

Eliminates Windows MSVC / C++ compilation errors and cp1252 character crashes
while allowing the user to stay 100% inside their preferred Windows desktop UI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict

WSL_DISTRO = "Ubuntu"
WSL_CONDA_ENV = "/home/ahmad/miniconda3/envs/abu_linux"
WSL_PYTHON = f"{WSL_CONDA_ENV}/bin/python"
WSL_OPENSFM_BIN = "/home/ahmad/OpenSfM/bin/opensfm"
WSL_360DVO_DIR = "/home/ahmad/360DVO"
WSL_PYCUSFM_CLI = f"{WSL_CONDA_ENV}/bin/cusfm_cli"
WSL_NS_TRAIN = f"{WSL_CONDA_ENV}/bin/ns-train"
WSL_NS_EXPORT = f"{WSL_CONDA_ENV}/bin/ns-export"


def is_wsl_available() -> bool:
    """Checks if WSL2 is installed, running, and has the Ubuntu distribution."""
    if sys.platform != "win32":
        return False
    wsl_exe = shutil.which("wsl")
    if not wsl_exe:
        return False
    try:
        res = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "echo", "WSL_OK"],
            capture_output=True,
            text=True,
            timeout=8
        )
        return "WSL_OK" in res.stdout
    except Exception:
        return False


def is_abu_linux_env_ready() -> bool:
    """Checks if the abu_linux conda environment with PyTorch and CUDA is ready in WSL."""
    if not is_wsl_available():
        return False
    try:
        res = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, WSL_PYTHON, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "True" in res.stdout
    except Exception:
        return False


def ensure_wsl_packages(packages: List[str], log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """
    Checks if specified Python packages are importable in WSL abu_linux,
    and automatically installs any missing ones via pip.
    """
    if not is_abu_linux_env_ready():
        return False
    missing = []
    for pkg in packages:
        # Get import module name (e.g. 'roma==1.4.1' -> 'roma')
        import_name = pkg.split("==")[0].split(">=")[0].split("<=")[0].strip()
        try:
            res = subprocess.run(
                ["wsl", "-d", WSL_DISTRO, WSL_PYTHON, "-c", f"import {import_name}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if res.returncode != 0:
                missing.append(pkg)
        except Exception:
            missing.append(pkg)

    if not missing:
        return True

    for pkg in missing:
        msg = f"      [*] WSL Bridge: Installing missing environment dependency: {pkg}..."
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
        try:
            res = subprocess.run(
                ["wsl", "-d", WSL_DISTRO, f"{WSL_CONDA_ENV}/bin/pip", "install", pkg],
                capture_output=True,
                text=True,
                timeout=180
            )
            if res.returncode == 0:
                succ = f"      [+] WSL Bridge: Successfully installed {pkg}."
                if log_callback:
                    log_callback(succ)
                else:
                    print(succ)
            else:
                err = f"      [!] WSL Bridge: Warning: pip install {pkg} exited with code {res.returncode}"
                if log_callback:
                    log_callback(err)
                else:
                    print(err)
        except Exception as e:
            err = f"      [!] WSL Bridge: Failed to install {pkg}: {e}"
            if log_callback:
                log_callback(err)
            else:
                print(err)
    return True


def win_to_wsl_path(win_path: str) -> str:
    """
    Converts a Windows path to a WSL path.
    Example: 'C:\\Users\\USer\\data' -> '/mnt/c/Users/USer/data'
    """
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return f"/mnt/{drive}{p[2:]}"
    return p


def wsl_to_win_path(wsl_path: str) -> str:
    """
    Converts a WSL path back to a Windows path.
    Example: '/mnt/c/Users/USer/data' -> 'C:\\Users\\USer\\data'
    """
    p = wsl_path.strip()
    if p.startswith("/mnt/") and len(p) >= 7 and p[6] == "/":
        drive = p[5].upper()
        rest = p[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    return p.replace("/", "\\")


def run_command_in_wsl(
    bash_cmd: str,
    log_callback: Optional[Callable[[str], None]] = None,
    cwd: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    terminate_check_callback: Optional[Callable[[], bool]] = None
) -> Tuple[int, str]:
    """
    Executes a bash command string inside WSL2 Ubuntu within the abu_linux environment.
    Streams stdout/stderr in real-time with UTF-8 safety to avoid Windows encoding issues.
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            try:
                print(msg, flush=True)
            except Exception:
                pass

    # Build full bash invocation that activates abu_linux PATH and library path
    path_prep = (
        f"export PATH='{WSL_CONDA_ENV}/bin:/usr/local/cuda/bin:/usr/bin:/bin:$PATH' && "
        f"export LD_LIBRARY_PATH='{WSL_CONDA_ENV}/lib/python3.10/site-packages/torch/lib:{WSL_CONDA_ENV}/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' && "
        f"export PYTHONIOENCODING=utf-8 && export LC_ALL=C.UTF-8 && export LANG=C.UTF-8"
    )

    if extra_env:
        for k, v in extra_env.items():
            path_prep += f" && export {k}='{v}'"

    if cwd:
        wsl_cwd = win_to_wsl_path(cwd)
        full_script = f"cd '{wsl_cwd}' && {path_prep} && {bash_cmd}"
    else:
        full_script = f"{path_prep} && {bash_cmd}"

    cmd = ["wsl", "-d", WSL_DISTRO, "bash", "-c", full_script]

    output_lines = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )

        for line in proc.stdout:
            cleaned = line.rstrip()
            output_lines.append(cleaned)
            log(cleaned)
            if terminate_check_callback and terminate_check_callback():
                log("      [*] [WSL Bridge] Early termination requested. Stopping WSL command...")
                try:
                    proc.terminate()
                except Exception:
                    pass
                break

        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
        return proc.returncode, "\n".join(output_lines)
    except Exception as e:
        err_msg = f"[WSL Bridge Error] Failed to execute command: {e}"
        log(err_msg)
        return 1, err_msg


_OPENSFM_AVAIL: Optional[bool] = None
_360DVO_AVAIL: Optional[bool] = None
_PYCUSFM_AVAIL: Optional[bool] = None
_PFGS360_AVAIL: Optional[bool] = None


def is_opensfm_available() -> bool:
    """Verifies that Mapillary OpenSfM with Ceres Solver is available in WSL."""
    global _OPENSFM_AVAIL
    if _OPENSFM_AVAIL is not None:
        return _OPENSFM_AVAIL
    if not is_wsl_available():
        _OPENSFM_AVAIL = False
        return False
    ret, out = run_command_in_wsl(f"test -f {WSL_OPENSFM_BIN} && echo OPENSFM_EXISTS", log_callback=lambda _: None)
    _OPENSFM_AVAIL = ("OPENSFM_EXISTS" in out)
    return _OPENSFM_AVAIL


def is_360dvo_available() -> bool:
    """Verifies that 360DVO with CUDA extensions and weights is available in WSL."""
    global _360DVO_AVAIL
    if _360DVO_AVAIL is not None:
        return _360DVO_AVAIL
    if not is_wsl_available():
        _360DVO_AVAIL = False
        return False
    cmd = f"test -f {WSL_360DVO_DIR}/360dvo.pth && {WSL_PYTHON} -c 'import dpvo, cuda_corr, cuda_ba; print(\"DVO_READY\")'"
    ret, out = run_command_in_wsl(cmd, log_callback=lambda _: None)
    _360DVO_AVAIL = ("DVO_READY" in out)
    return _360DVO_AVAIL


def is_pycusfm_available() -> bool:
    """Verifies that NVIDIA PyCuSFM is available in WSL."""
    global _PYCUSFM_AVAIL
    if _PYCUSFM_AVAIL is not None:
        return _PYCUSFM_AVAIL
    if not is_wsl_available():
        _PYCUSFM_AVAIL = False
        return False
    ret, out = run_command_in_wsl(f"{WSL_PYCUSFM_CLI} --help", log_callback=lambda _: None)
    _PYCUSFM_AVAIL = (ret == 0 and "CUSFM" in out)
    return _PYCUSFM_AVAIL


def is_pfgs360_available() -> bool:
    """Verifies that PFGS360 is available in WSL."""
    global _PFGS360_AVAIL
    if _PFGS360_AVAIL is not None:
        return _PFGS360_AVAIL
    if not is_wsl_available():
        _PFGS360_AVAIL = False
        return False
    ret, out = run_command_in_wsl(f"test -f {WSL_NS_TRAIN} && {WSL_PYTHON} -c 'import nerfstudio, gsplat360, spherical_distortion; print(\"PFGS_OK\")'", log_callback=lambda _: None)
    _PFGS360_AVAIL = (ret == 0 and "PFGS_OK" in out)
    return _PFGS360_AVAIL

