# Abu Align & Train v0.0.5.3 — ACE0 WSL2 Dependency Self-Healing, PyTorch Orthonormalization Fallback & Argparse Fix

## Overview
Version v0.0.5.3 resolves a critical regression in the Niantic ACE0 pipeline where mapping failed inside WSL2 Ubuntu due to a missing `roma` rotation matrix package. This release introduces automatic dependency self-healing for the WSL2 environment, a pure PyTorch fallback for camera pose orthonormalization, and fixes an `argparse` format-string crash.

---

## Key Changes & Implementations

### 1. ACE0 WSL2 Environment & Dependency Self-Healing
- **WSL Package Installation**: Installed missing `roma` (1.6.1) and `pycolmap` (4.2.0) into the WSL conda environment `/home/ahmad/miniconda3/envs/abu_linux`.
- **Automated Verification (`ensure_wsl_packages`)**: Added `ensure_wsl_packages()` in `wsl_bridge.py` to inspect required Python packages in the Ubuntu subsystem and auto-install any missing dependencies via pip before launching compute workloads.
- **Engine Hook**: Integrated dependency verification into `acezero_engine.py` prior to initiating the ACE0 mapping loop.

### 2. Pure PyTorch Pose Orthonormalization Fallback
- **Guarded Import & Safe Execution**: In `refine_poses.py`, wrapped `import roma` in a safe try-except block.
- **Native PyTorch Implementation**: Added native implementations for Gram-Schmidt (`_special_gramschmidt`) and SVD-based Procrustes (`_special_procrustes`) orthonormalization with full autograd gradient support, ensuring pose refinement functions reliably even in stripped or offline environments.
- **Synchronized Across Copies**: Applied identical safeguards to both `abu align/acezero/refine_poses.py` and root `acezero/refine_poses.py`.

### 3. Argparse Format-String Fix in `train_ace.py`
- **Escaped `%` Literals**: Fixed unescaped percent signs (`70%%`, `10%%`) in help strings of `train_ace.py` which previously triggered a `TypeError: %o format: an integer is required, not dict` when inspecting CLI options under Python 3.10.

---

## Verification
- Ran test imports of all core ACE0 modules (`train_ace`, `ace_zero`, `dsacstar`, `ace_trainer`, `refine_poses`) inside WSL2 with return code 0.
- Verified `train_ace.py --help` and `ace_zero.py --help` execute without formatting errors.
- Validated PyTorch fallback accuracy against `roma` (error diff: 0.0, autograd backward pass verified).
- Executed complete unit test suite across desktop UI and presets: 25/25 tests passed.

---

## Modified & Created Files
- `acezero_engine.py`
- `wsl_bridge.py`
- `acezero/refine_poses.py`
- `acezero/train_ace.py`
- `../acezero/refine_poses.py`
- `../acezero/train_ace.py`
- `docs/WALKTHROUGH_v0.0.5.3.md`
- `walkthrough.md`
