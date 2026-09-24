# Abu Align & Train v0.0.5.4 — Fix 'Save & Finish' Early Exit & Checkpoint Auto-Conversion

## Overview
Version v0.0.5.4 resolves an issue where clicking the **Save & Finish** button during 3D Gaussian Splatting training logged the early stop request but failed to halt the active optimizer (such as LichtFeld Studio), causing training to run continuously. This update introduces live sentinel polling across all training engines, graceful process termination, and automatic conversion of intermediate project checkpoints (`project.licht`) into final `.ply` / `.spz` splats.

---

## Log Analysis & Root Cause (`abu_align_2026-09-25_00-06-34.log`)

1. **Log Observation**:
   - The user clicked "Save & Finish" while training with LichtFeld Studio at ~iteration 17,432.
   - The GUI logged:
     ```
     [SAVE & FINISH] Early stop requested by user!
     Preserving current Gaussians, concluding optimization, and exporting 3DGS model...
     ```
   - However, training did not halt. The log showed LichtFeld Studio continuing to train through iterations 17,433, 20,450+, and maintaining 97% GPU load until the user manually terminated the task.
2. **Root Causes**:
   - **Missing Sentinel Check in `train_lichtfeld`**: `train_engine.py` contained helper functions (`check_save_and_finish_sentinel`) for sentinel files (`.finish_and_save`), but `train_lichtfeld` never checked this sentinel inside its stdout streaming loop.
   - **Post-Wait Blocking in Other Engines**: In `train_nerfstudio_splatfacto` and `train_3dgrut`, the check was only placed after `proc.wait()`, meaning early termination could never be signaled to a running process.
   - **Missing Checkpoint Conversion**: LichtFeld Studio continuously maintains an adopted snapshot project file (`project.licht`). If training is stopped early, no standalone `.ply` was exported because the final iteration export step had not been reached.
   - **GUI Output Directory String Mismatch**: `abu_align_gui.py` parsed `self.latest_output_dir` searching strictly for `"Output Directory:"`, whereas the CLI outputs `"Target Output:"` and `"Output Dir:"`.

---

## Key Changes & Implementations

### 1. Live Sentinel Polling in `train_lichtfeld` (`train_engine.py`)
- Inside `for line in proc.stdout:`, added active checks for `check_save_and_finish_sentinel(output_dir, dataset_dir)`.
- Upon detection:
  - Logs: `* [LichtFeld] Early 'Save & finish' detected! Concluding optimization and exporting 3DGS model...`
  - Gracefully terminates the running `LichtFeld-Studio.exe` process (using `psutil` tree termination or `proc.terminate()`).
  - Breaks the read loop cleanly.

### 2. Automatic Checkpoint Snapshot Conversion to `.ply` / `.spz`
- If no `.ply` or `.spz` was created prior to the early stop, `train_lichtfeld` discovers the latest snapshot (`project.licht` or `.resume`) in the model directory.
- Invokes `LichtFeld-Studio convert <latest.licht> <output.ply> -y` to immediately serialize the intermediate Gaussians into a standard 3DGS file (validated: converted a 267 MB snapshot into a 159 MB `.ply` in ~3s).
- Safely consumes the sentinel flag upon successful export.

### 3. Early Exit Hardening Across All Engines
- **Nerfstudio Splatfacto**: Added active sentinel checking within the stdout reading loop in `train_nerfstudio_splatfacto`.
- **NVIDIA 3DGRUT**: Added active sentinel checking within the loop in `train_3dgrut`.
- **PFGS360 (WSL2)**: Added `terminate_check_callback` to `wsl_bridge.run_command_in_wsl()` to terminate background WSL commands when `check_save_and_finish_sentinel` returns True.

### 4. Robust Output Directory Detection & Sentinel Propagation (`abu_align_gui.py`)
- Expanded log parsing to match `"Target Output:"` and `"Output Dir:"` in addition to `"Output Directory:"`.
- Enhanced `_save_and_finish_training()` to deploy `.finish_and_save` flags into `results/` and active subdirectories, guaranteeing detection across all nested paths.

---

## Verification
- Validated `LichtFeld-Studio convert` command directly on the actual 267 MB `project.licht` snapshot generated during the user's run; conversion produced a 159.7 MB valid 3DGS `.ply` model in 3.1s.
- Executed the full automated test suite: 25/25 unit tests passed cleanly.
- Bumped version to `v0.0.5.4` and synchronized across all versioned files.

---

## Modified Files
- `train_engine.py`
- `wsl_bridge.py`
- `abu_align_gui.py`
- `VERSION.txt`
- `docs/WALKTHROUGH_v0.0.5.4.md`
- `walkthrough.md`
