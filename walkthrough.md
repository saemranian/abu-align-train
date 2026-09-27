# v0.0.8.9 — Dynamic Anchor & Chunk Coupling, Cache Key Differentiation, and Drift-Resistant Reconstruction

## Outcome
Abu Align & Train v0.0.8.9 resolves the issue where changing the **Overlap (Anchor)** setting (e.g. 24, 48, 256) appeared to have no effect on reconstruction accuracy or point clouds. It uncovers and fixes two underlying root causes: (1) multi-tier cache fingerprinting omitting windowing hyperparameters, causing stale cache hits when altering overlap, and (2) an arbitrary half-chunk clamp (`chunk_size // 2`) that silently forced large overlaps (such as 256) back to 48 frames. In addition, it adds dynamic auto-expansion of chunk sizes, GUI coupling with live tooltips, and provides clear architectural guidance for large datasets (2,500+ photos) without modifying upstream external repositories.

## Root Cause Analysis from Diagnostic Logs

1. **Log 1 (`abu_align_2026-09-27_02-30-34.log`) — Baseline Run (Anchor 48)**:
   - Chunk: 96, Anchor: 48, Alignment Mode: `ultra` (Ultra Sub-Pixel BA).
   - Ran 53 windows of 96 frames with 48-frame anchors. Successfully generated baseline reconstruction and saved results to `.abu_cache`.

2. **Log 2 (`abu_align_2026-09-27_03-01-38.log`) — Anchor 24 Run (Stale Cache Hit)**:
   - Command requested `--chunk_size 96 --anchor_size 24 --align_mode none`.
   - Result: `[SMART CACHE HIT] Loaded neural poses & 3D points in 3.36s! Skipping Stage 3 & 4 neural inference`.
   - **Bug**: `compute_dataset_hash` in `cache_manager.py` only factored in image paths, engine, resolution, and stride. Because `chunk_size` and `anchor_size` were omitted, Run 2 produced the exact same hash as Run 1 and reloaded Run 1's poses in 3 seconds without executing Stage 3 & 4. Additionally, `--align_mode none` disabled Ceres bundle adjustment.

3. **Log 3 (`abu_align_2026-09-27_03-06-14.log`) — Anchor 256 Run (Silently Clamped to 48)**:
   - Command requested `--chunk_size 96 --anchor_size 256 --align_mode none`.
   - Result in log line 32: `Auto-chunking into 53 windows of 96 frames (48-frame anchors)`.
   - **Bug**: Line 801 in `align_cli.py` contained `overlap = min(overlap, max(chunk_size // 2, 1))`. When `chunk_size = 96`, `chunk_size // 2` is 48. The user's anchor setting of 256 was completely clamped down to 48! Run 3 ran with the exact same 48-frame overlap as Run 1, with Ceres BA disabled (`--align_mode none`).

## Key Improvements in v0.0.8.9

### 1. Multi-Tier Cache Key Fingerprinting (`cache_manager.py` & `align_cli.py`)
- Updated `compute_dataset_hash` to accept optional `chunk_size` and `anchor_size` parameters.
- Incorporates `chunk_{chunk_size}` and `anchor_{anchor_size}` tokens directly into the deterministic SHA-256 fingerprint.
- Switching between Anchor 24, 48, 128, or 256 now reliably generates distinct cache keys, ensuring fresh neural inference and feature tracking for each unique configuration.

### 2. Auto-Expanding Chunking & Decoupled Overlap Clamping (`align_cli.py`)
- Removed the arbitrary `chunk_size // 2` clamp. Overlap is now permitted up to `chunk_size - 1`.
- **Dynamic Auto-Expansion**: When a user specifies an overlap greater than or equal to the current chunk size (e.g. `--anchor_size 256` with default `--chunk_size 96`), the pipeline automatically expands the chunk size (e.g. `256 + 64 = 320` or `384` frames) to maintain positive sliding steps (`step = chunk_size - overlap > 0`) without dropping the user's requested anchor capacity.
- Displays an explicit notice: `[Auto-Expansion] Requested Overlap (256) >= Chunk Size (96). Auto-expanding Chunk Size to 320 frames to maintain forward progress.`

### 3. Dynamic GUI Coupling & Informative Tooltips (`desktop_ui.py`)
- Added synchronized trace handlers between `app.anchor_var` and `app.chunk_size_var`:
  - When the user selects a large Overlap (e.g. 128 or 256), the Chunk dropdown automatically advances to a compatible size (e.g. 192, 256, 384, or 512).
  - When the user selects a smaller Chunk, Overlap safely adjusts to maintain a valid window step.
- Added comprehensive tooltips to both **Chunk** and **Overlap** dropdowns explaining their exact roles in VRAM allocation, window transitions, and drift prevention on large-scale datasets.

### 4. Architectural Guidance for Massive (2,500+ Photo) Datasets
- **Root Cause of "Bunches of Photos" Disjoint Clusters**:
  VGGT-Omega is a feed-forward transformer trained on windows of 32–96 frames. Chaining 53 windows sequentially via pairwise Umeyama alignments is an open-loop visual odometry process that accumulates rotational and translational drift over time. Because CoTracker only tracks within 32-frame sliding windows, there are no global loop closure constraints between window 53 and window 1 when the camera circles back into a previously visited room.
- **Hardware Optimization (RTX 4090 / 24 GB VRAM)**:
  Logs confirmed that Chunk 96 used only 8.1 GB out of 24.0 GB VRAM. An RTX 4090 can easily handle `chunk_size = 256` or `384` with `overlap = 128` or `192`, reducing the total window count from 53 down to 10–15 windows and drastically stiffening pairwise registration.
- **Global SfM & Prior Trajectory Integration**:
  For millimeter-level precision across 2,500+ photos in complex indoor layouts, Abu Align's built-in **RealityScan Engine** (`--engine realityscan`) or **OpenSfM** provides vocabulary-tree global loop closure. Abu Align can feed the VGGT-Omega camera trajectory directly as a **Prior Trajectory** (`--rs_prior_run auto`), giving the best of both worlds: neural initialization plus rigorous global multi-ray bundle adjustment.

## Upstream Repository Integrity
- **Zero Modifications to External Upstream Repositories**:
  All changes are strictly confined to the Abu Align orchestration and UI layer (`align_cli.py`, `cache_manager.py`, `desktop_ui.py`, and test suites).
  External submodules and dependencies (`vggt-omega/`, `vggt/`) remain completely unmodified and pristine.

## Verification
- **Automated Test Suite**: Ran all 108 unit tests in the dedicated conda `vggt` environment (`C:\Users\USer\miniconda3\envs\vggt\python.exe`). All 108 tests passed in 45.7s with 0 errors and 0 failures.
- **New Regression Tests**: Added `test_chunk_anchor_cache_fingerprint` and `test_chunk_overlap_auto_expansion` to `tests/test_vggt_omega_optimization.py`.
- **C# Launcher Compilation**: Recompiled native `Abu_Align.exe` and `Abu_Align_and_Train.exe` binaries using .NET `csc.exe`.
- **Release Snapshot**: Verified code snapshot and documentation in `versions/v0.0.8.9/`.

## Files Modified
- `cache_manager.py`: Added `chunk_size` and `anchor_size` support to `compute_dataset_hash`.
- `align_cli.py`: Wired effective chunk/anchor parameters into dataset cache hashing, eliminated half-chunk clamp, and enabled chunk auto-expansion.
- `desktop_ui.py`: Added dynamic trace coupling and detailed tooltips for Chunk and Overlap controls.
- `tests/test_vggt_omega_optimization.py`: Added automated regression tests for cache fingerprinting and chunk auto-expansion.
- `walkthrough.md`: Comprehensive technical and diagnostic documentation.
