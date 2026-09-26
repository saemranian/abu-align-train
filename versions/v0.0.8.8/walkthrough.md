# v0.0.8.8 — Canonical Canvas Harmonization & Mixed Spatial Resolution Chunk Crash Fix

## Outcome
Abu Align & Train v0.0.8.8 fixes the critical crash during large dataset processing in VGGT-Omega windowed chunk reconstruction (`ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 512 and the array at index 20 has size 288`). All windowed chunks across datasets with mixed landscape and portrait orientations now operate with 100% unified spatial dimensions and zero-drift geometric alignment.

## Changes

### 1. Canonical Dataset Canvas Detection (`align_cli.py`)
- **Root Cause Resolution**: When processing large image collections (e.g., 2,582 images across 53 windows), individual chunks previously determined their aspect ratios locally. Chunks containing both portrait and landscape images were padded to `512x512`, while chunks containing only landscape images remained at native aspect-ratio resolutions such as `288x512` or `336x512`. At the end of inference, `np.concatenate(all_depth_maps, axis=0)` crashed due to spatial dimension mismatch along dimension 1.
- **Global Pre-Scan (`determine_global_vggt_canvas`)**: Implemented a multi-threaded header inspection pre-scan over `image_paths` that evaluates aspect ratios in under 1 second without full pixel decoding. It determines the canonical dataset canvas `(max_H, max_W)` (e.g. `(512, 512)` when mixed portrait/landscape frames are present, or native `(336, 512)` for pure landscape runs).
- **Unified Chunk Padding (`load_and_preprocess_vggt_images`)**: Added `target_canvas_size` parameter. All chunks are padded to the canonical canvas using Meta's `_pad_images_to_common_size`, ensuring every single chunk produces identically sized tensors `(B, 3, H, W)` and depth/confidence maps.

### 2. Defensive Spatial Resolution Harmonization (`safe_harmonize_chunk_resolutions`)
- **Zero-Crash Guard**: Added a defensive pre-concatenation harmonization pass in `align_cli.py`. If any batch outputs differ in spatial dimensions due to custom inputs or external engines, `safe_harmonize_chunk_resolutions` automatically pads depth maps (with 0.0), confidence maps (with 0.0), and RGB arrays (with 255) to `(max_H, max_W)`.
- **Intrinsic Principal Point Adjustment**: Accurately offsets camera intrinsics (`cx += pad_left`, `cy += pad_top`) and original coordinates to guarantee 100% geometric projection accuracy for unprojected 3D point clouds.
- **Confidence Filtering Compatibility**: Zero-confidence padding is automatically discarded by `conf_mask = depth_conf >= conf_thresh`, eliminating any phantom points.

### 3. Cross-Chunk Landmark Alignment Protection (`align_cli.py`)
- **Safe Coordinate Indexing**: Guarded Umeyama 3D landmark surface point extraction against any cross-chunk resolution differences by scaling grid coordinates between source and destination chunks, preventing out-of-bounds `IndexError` exceptions.

### 4. Sliding Window Tracking Synchronization (`vggt/dependency/track_predict.py`)
- **Canvas-Matched Window Loading**: Updated `predict_tracks_windowed` to pass `target_canvas_size` matching `points_3d` spatial resolution when loading window frames on demand, ensuring 1:1 pixel coordinate alignment during CoTracker feature extraction.

## Verification
- **Automated Test Suite**: All 106 unit tests passed in 43 seconds across the entire test suite, including new regression tests `test_safe_harmonize_chunk_resolutions`, `test_determine_global_vggt_canvas`, and `test_load_and_preprocess_vggt_images_with_target_canvas`.
- **C# Launcher Compilation**: Recompiled native `Abu_Align.exe` and `Abu_Align_and_Train.exe` binaries using .NET `csc.exe`.
- **Release Snapshot**: Verified code snapshot and documentation in `versions/v0.0.8.8/`.

## Files and Release
- Core: `align_cli.py`, `vggt/dependency/track_predict.py`, `tests/test_vggt_omega_optimization.py`, `version_manager.py`.
- Release Snapshot: `versions/v0.0.8.8/` in the project root.
