# Abu Align & Train v0.0.3.7 — PFGS360 Live Training & Telemetry Fix

## 1. Issue Diagnosis & Root Causes

In the previous run reported by the user (`media_1790110129516.png`), clicking **TRAIN 3DGS ONLY** with PFGS360 appeared stuck with `Status: Initializing 3DGS Training...`, `0.0%`, low GPU/CPU utilization, and an empty black console.

Investigation revealed three distinct root causes:

1. **Dataparser Selection Mismatch (CPU Matching vs Instant GPU Handoff)**:
   - The user selected `◉ Pose Free Direct [Unposed Video - CPU Matching]`.
   - The dataset already had 169 frames with pre-aligned OpenSfM poses in `results/sparse_16_opensfm/3dgs_dataset`.
   - In `Pose-Free Direct` mode, PFGS360 intentionally discards prior poses and calculates relative poses from scratch on CPU using pairwise SIFT matching (`compute relative camera pose for views [0, 1, 2, 3] and 4: 10635 pairs`). This is CPU-bound and takes ~15–30 minutes before 3DGS optimization begins.
   - In contrast, when `Auto-Detect` or `OpenSfM / 360DVO Handoff` is chosen, PFGS360 instantly detects `reconstruction.json` or `transforms.json`, bypasses CPU matching completely, engages 100% RTX 4090 GPU, and finishes training + PLY export in under ~70 seconds.

2. **Console Output Buffering & Tkinter Event Loop**:
   - In earlier versions, subprocess stdout buffering delayed log output on Windows until buffers flushed.
   - Text logged via `_log` lacked an immediate `self.root.update_idletasks()` flush, causing the GUI console to remain blank while the process began.

3. **Telemetry & Progress Bar Gap for Training Engines**:
   - `_check_queue()` in `abu_align_gui.py` did not include `[PFGS360]` in its stage badge and telemetry handlers.
   - While stage 3 (Neural Inference) and stage 5 (Tracking & BA) had regex percentage parsers, stage 6 (3DGS Training) lacked parsing for iteration steps (`Step (% Done)`), depth prior generation (`XX%|...| X/Y`), and ETAs (`1 m, 0 s`).
   - Consequently, the progress bar remained pinned at `0.0%` and the status label remained on `Status: Initializing 3DGS Training...` for the entire duration of the training session.

---

## 2. Changes Made in v0.0.3.7

1. **Smart Pre-Aligned Poses Guard for PFGS360**:
   - When the user selects `Pose-Free Direct [CPU Matching]` but the dataset already contains pre-aligned poses (`transforms.json`, `reconstruction.json`, or in `3dgs_dataset/` / `results/`), the GUI displays an interactive confirmation dialog asking if the user would like to switch to `Auto-Detect` to leverage existing poses and train immediately at 100% GPU power.
   - Clicking **Yes** automatically switches the dataparser mode to `auto` and launches 100% GPU training.

2. **Live Training Telemetry & Real-Time Progress Bar**:
   - Upgraded `_check_queue()` in `abu_align_gui.py` to parse:
     - **Depth Prior Generation**: `[PFGS360] 25%|██▌ | 43/169` -> updates progress bar and shows `Stage 6/6: Precomputing Depth Priors [43/169] (25%)`.
     - **Training Iterations**: `[PFGS360] 1000 (3.33%)` -> updates progress bar to 3.3% and shows `Stage 6/6: PFGS360 Active [Step 1,000] (3.3%)`.
     - **Accurate ETA**: extracts elapsed iteration time and ETA (`⏱ ETA: 1 m, 0 s`).
     - **LichtFeld Studio**: `Iteration 500/7000 (7.1%) - ETA: 45s`.
     - **Pose-Free CPU Matching**: `compute relative camera pose for views ... and X:` -> displays `Stage 6/6: CPU Relative Pose Matching (View #X)`.
     - **Export & Completion**: updates telemetry to `Exporting 3D Gaussian Splat (.ply)...` and sets progress to 100% when finished.

3. **Instant GUI Console Rendering**:
   - Added `self.root.update_idletasks()` to `self._log()` to force Tkinter to immediately repaint text in the console widget.

4. **Launchers & Version Snapshot**:
   - Recompiled `Abu_Align.exe` and `Abu_Align_and_Train.exe` with C# compiler `csc.exe`.
   - Created `versions/v0.0.3.7/` snapshot via `version_manager.py --bump`.
   - Added unit test `test_train_only_pfgs360_workflow_and_telemetry_parsing` in `tests/test_desktop_ui.py`.

---

## 3. Verification & Benchmark

1. **Live PFGS360 Training Verification**:
   - Executed `train_engine.py` on the user's test dataset (`Container_03\Insta360\Tests\Extract 1\CC\Equirectangular`).
   - `Auto-Detect` mode detected pre-aligned OpenSfM poses, bypassed CPU matching, engaged 100% RTX 4090 GPU, trained 1,042,222 Gaussians, and exported `pfgs360_splat.ply` (258,472,613 bytes) in **77.1 seconds**.
   - `Pose-Free Direct` mode executed successfully in **70.1 seconds** for 5 test steps and exported `pfgs360_splat.ply` (260,610,869 bytes).

2. **Automated Test Suite**:
   - Ran `python -m unittest discover -s tests -v`:
   - All 11 tests passed in 15.55s with 0 errors and 0 failures.
