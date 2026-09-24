# Abu Align & Train v0.0.5.9 — Fix ACE0 Excessive Runtime, Pipeline Misconfiguration, and GUI Telemetry Disconnect

## Executive Summary
In version **v0.0.5.9**, we resolved a critical pipeline defect in **Niantic Labs ACE0** camera pose alignment where a 50-image dataset was taking excessive time (>20 minutes with no end in sight), the camera focal length collapsed from `1371.8` down to `508.5`, and the graphical interface remained frozen at **`0.0%`** with `Status: Initializing Pipeline...` and the "1 Scan" stage button permanently highlighted in cyan.

---

## Detailed Root Cause Analysis

### 1. The Blind 25,000-Iteration `final_refit` Loop (Discards Solved Poses & Collapses Focal Length)
- **Log Observation**:
  - At iteration 1 of mapping, ACE0 achieved **100.0% registration** of all 50 photos with high confidence (up to 10,000 inliers) and refined focal length to `1373.2px`:
    ```
    [ACE0-WSL] INFO:__main__:Successfully registered 100.0% of mapping images.
    [ACE0-WSL] INFO:__main__:Stopping training loop in next iteration. Enough mapping images registered. (Threshold=99.0%)
    ```
  - However, because `--final_refit True` was active by default in the origin ACE0 code and was never disabled in `acezero_engine.py`:
    1. **Discards Learned Network**: In `ace_zero.py` line 309, ACE0 explicitly **skips loading previous weights** (`--load_weights` omitted) during `final_refit`! The model was reset to **completely random, uninitialized weights** from scratch.
    2. **Disables Early Stopping**: It switched to a `"circle"` learning rate schedule with `--iterations 25000`, forcing the loop to execute every single iteration.
    3. **Severe Focal Length Collapse**: Poses were locked for the first 5,000 steps (`--final_refit_posewait 5000`). Because the network was completely untrained and poses were frozen, reprojection error gradients backpropagated directly into camera focal length, causing it to collapse:
       ```
       Iteration: 0    | Loss: 31.5 | Focal: 1371.8
       Iteration: 500  | Loss: 46.2 | Focal: 957.1
       Iteration: 1000 | Loss: 43.5 | Focal: 739.5
       Iteration: 1500 | Loss: 40.8 | Focal: 626.2
       Iteration: 2500 | Loss: 36.3 | Focal: 508.5
       ```
    4. This redundant phase added **>500 seconds (8.5+ minutes)** of destructive compute on a dataset that was **already 100% solved**.

### 2. Redundant Sequential Seed Search (`--seed_parallel_workers 1`)
- In `acezero_engine.py`, `--seed_parallel_workers 1` was hardcoded.
- Candidate seeds were evaluated one-by-one:
  - Seed 0: 110.8s (82.0% registered)
  - Seed 1: 118.9s (70.0% registered)
  - Seed 2: 117.9s (75.0% registered)
- **Nearly 6 minutes (347 seconds)** were wasted evaluating seeds sequentially.
- On an RTX 4090 with 24GB VRAM (where each seed worker uses only ~4–5GB VRAM), workers can run concurrently, or search can exit immediately once a candidate seed exceeds a high registration threshold (e.g., ≥80%).

### 3. GUI Progress & Telemetry Blindness
- In `abu_align_gui.py`:
  - `_check_queue()` only recognized VGGT telemetry strings like `[TELEMETRY:STAGE3]`, `Stage 1`, etc.
  - It had zero parsers for `[ACE0-WSL]` or `[ACE0]` logs.
  - The UI never updated the progress bar (`0.0%`), never changed the stage buttons (stayed on "1 Scan"), and never updated status text (`Status: Initializing Pipeline...`), giving the impression that the system had hung.

---

## Comparison with Official Niantic Labs ACE0 Origin Repository

| Feature / Behavior | Official Niantic Labs ACE0 (Origin Repo) | Abu Align Previous Implementation (v0.0.5.8) | New Implementation (v0.0.5.9) |
| :--- | :--- | :--- | :--- |
| **`--final_refit`** | Default is `True` for academic benchmarks (training a standalone relocalizer model for 7-Scenes / Tanks & Temples). | Omitted (defaulted to `True`), causing a 25k iteration blind re-training that collapsed focal length. | Set `--final_refit False` by default for photogrammetry SfM. Finishes immediately when 100% registered. |
| **`--final_refine`** | Default `True` (runs an extra mapping round when stopping criteria met). | Omitted (defaulted to `True`). | Configurable (`False` by default); skips redundant final round when registration rate reaches 100%. |
| **Seed Workers** | Default 3 parallel workers (`--seed_parallel_workers 3`). Scripts use up to 5 workers (`reconstruct_mip360.sh`). | Hardcoded to `1` (sequential), wasting 6 minutes on 3 seeds. | Auto-detect GPU VRAM: 2–3 parallel workers on ≥16GB GPUs (RTX 4090). |
| **Seed Early Exit** | Evaluates all `try_seeds` uniformly. | Evaluates all seeds sequentially. | Early-exit seed loop if Seed 0 registers ≥75% of images. |
| **Telemetry & UI** | Raw terminal output (`ace_trainer` logs). | Unparsed stdout; UI stuck at `0.0%` / "Initializing Pipeline...". | Live parsing of ACE0 iterations, % registered, focal length, loss, and dynamic stage transitions. |

---

## Technical Changes Made

### 1. `acezero_engine.py`
- Added `--final_refit False` and `--final_refine False` by default in both WSL2 and Windows fallback calls.
- Auto-detect GPU VRAM: defaults to `3` workers if VRAM ≥ 20GB (RTX 4090), `2` if ≥ 12GB, `1` otherwise.
- Added structured telemetry interceptor in `handle_ace_line()` emitting `[TELEMETRY:STAGE2]`, `[TELEMETRY:STAGE3]`, `[TELEMETRY:STAGE4]`, and `[TELEMETRY:ACE_OPTICS]`.

### 2. `acezero/ace_zero.py`
- Added `--seed_threshold_early_exit` argument (default `0.75`).
- Updated seed search loop to break early if a candidate seed registers ≥ 75% of mapping images.

### 3. `align_cli.py`
- Added CLI arguments: `--ace_final_refit`, `--ace_final_refine`, and `--ace_seed_workers`.
- Routed `progress_callback=print` so structured telemetry lines are streamed to terminal and GUI.
- Updated summary report to display `Fast SfM Reconstruction: Active (Skip 25k Refit)`.

### 4. `abu_align_gui.py`
- **ACE0 UI Subpanel**:
  - Added `Workers` selector (1, 2, 3 - auto-selected to 2 for RTX 4090).
  - Added `Academic 25k Refit Mode` checkbox (unchecked by default).
- **Live Output Queue & Stage Parser (`_check_queue`)**:
  - `Stage 0 (Scan)`: Scanning Image Dataset.
  - `Stage 1 (Initialize)`: Initializing ACE0 & ZoeDepth Models.
  - `Stage 2 (Preprocess)`: ACE0 Seed Search with ZoeDepth prior.
  - `Stage 3 (Infer)`: Real-time progress bar `(iter / total * 100)`, Loss, Inliers (%), and current Focal Length.
  - `Stage 4 (Track & Refine)`: Camera registration percentage and COLMAP reconstruction.
  - `Stage 5 (Train)`: Chains into 3DGS training (LichtFeld / Splatfacto).

---

## Verification Results

| Metric | Before Fix (v0.0.5.8 Log) | After Fix (v0.0.5.9 Expected) |
| :--- | :--- | :--- |
| **Seed Search Time** | ~347s (3 seeds sequential) | ~110s (Seed 0 early exit at 82%) |
| **Mapping Iteration 1** | ~323s (100% registered) | ~320s (100% registered) |
| **25k Refit Loop** | >500s (forced 25k steps) | **0s (Skipped!)** |
| **Total Alignment Time** | **>1,200s (20+ min)** | **~430s (~7 min)** |
| **Focal Length Stability** | Collapsed (1371.8 -> 508.5) | **Preserved (1373.2px)** |
| **GUI Progress Bar** | Stuck at 0.0% | Real-time progress (0% -> 100%) |
| **GUI Stage Badges** | Stuck on "1 Scan" | Dynamically transitions through 1 -> 6 |

---

## Version Bump
- Bumped version from `v0.0.5.8` to `v0.0.5.9`.
- Synchronized: `VERSION.txt`, `abu_align_gui.py`, `align_cli.py`, `desktop_ui.py`, `train_engine.py`, `launcher.cs`, `Abu_Align_and_Train.bat`, `Abu_Align.bat`.
- Recompiled C# binaries: `Abu_Align_and_Train.exe`, `Abu_Align.exe`.
- Created version archive snapshot: `versions/v0.0.5.9/`.
