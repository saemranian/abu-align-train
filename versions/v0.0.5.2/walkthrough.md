# Abu Align & Train v0.0.5.2 — ACE0 WSL2 Native Acceleration, Cubemap FOV & Distortion Telemetry

## Overview
Version v0.0.5.2 brings native Ubuntu WSL2 integration for the Niantic ACE0 visual localization & mapping engine, unlocks dynamic Field of View (FOV) adjustment and true overlap calculations in the 360° Cubemap Rig Designer, adds real-time lens distortion optics telemetry in the viewport, and consolidates the automated checkpoint workflow.

---

## Key Changes & Implementations

### 1. Niantic ACE0 Native WSL2 Ubuntu Execution & C++ Extension
- **Native Extension Compilation**: Built and linked `dsacstar.cpython-310-x86_64-linux-gnu.so` on Ubuntu WSL2 using GCC 13.3 with OpenMP and system OpenCV 4 (`/usr/include/opencv4`).
- **Path & Environment Bridge**: Updated `acezero_engine.py` to auto-detect the Ubuntu WSL2 environment (`abu_linux`), translate Windows paths to WSL `/mnt/c/...`, and launch training seamlessly with native Linux GPU acceleration.
- **Cache Pre-seeding**: Synced ZoeDepth and Torch Hub checkpoints into `/home/ahmad/.cache/torch/hub/` to eliminate cold-start download stalls during mapper initialization.
- **Argument Parser Fix**: Resolved `argparse` formatting bug in `acezero/ace_zero.py`.

### 2. 360° Rig Designer Optics & Cubemap Flexibility
- **Unlocked Cubemap FOV**: Removed the legacy hardcoded 90° FOV clamp in `abu_align_gui.py`, allowing users to customize cubemap lens FOV (e.g. 90° to 120°+).
- **Geometric Overlap Calculation**: Implemented accurate cubemap overlap formula:
  $$\text{Overlap} \% = \max\left(0, \frac{\text{FOV} - 90^\circ}{\text{FOV}} \times 100\%\right)$$
- **Viewport Optics Telemetry**: In `rig_viewport.py`, added dynamic optical indicator reflecting radial distortion coefficient ($k_1$) and lens projection type alongside camera frustums.

### 3. Engine & Parameter Enhancements
- **360DVO / OpenSfM Controls**: Refined density budget handling and camera parameter propagation across both visual odometry and structure-from-motion pipelines.

### 4. Base-10 Versioning & Artifact Packaging
- **Base-10 Increment**: Bumped version from `0.0.5.1` to `0.0.5.2` across `VERSION.txt`, `Abu_Align.bat`, `Abu_Align_and_Train.bat`, `abu_align_gui.py`, `align_cli.py`, `desktop_ui.py`, and `train_engine.py`.
- **Launcher Binaries**: Recompiled `Abu_Align.exe` and `Abu_Align_and_Train.exe` using Microsoft Visual C# compiler (`csc.exe`).
- **Snapshot Backup**: Archived full snapshot in `versions/v0.0.5.2/`.

---

## Verification
- All 25 unit tests executed and passed (`tests/test_desktop_ui.py`, `tests/test_rig_designer.py`, etc.).
- C# launcher binaries built cleanly with 0 warnings/errors.
- Verified WSL2 bridge path translation between Windows root and Ubuntu subsystem.

---

## Modified & Created Files
- `acezero_engine.py`
- `acezero/ace_zero.py`
- `acezero/dsacstar/dsacstar.cpython-310-x86_64-linux-gnu.so`
- `abu_align_gui.py`
- `rig_viewport.py`
- `desktop_ui.py`
- `align_cli.py`
- `train_engine.py`
- `Abu_Align.bat`, `Abu_Align_and_Train.bat`
- `Abu_Align.exe`, `Abu_Align_and_Train.exe`
- `VERSION.txt`
- `version_manager.py`
- `versions/v0.0.5.2/`
- `docs/WALKTHROUGH_v0.0.5.2.md`
- `walkthrough.md`
