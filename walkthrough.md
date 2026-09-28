# v0.1.0.6 — 360 Splitter: Equirectangular Mask Layer Auto-Detection & RealityScan Alignment

## Overview & Outcome
Abu Align & Train **v0.1.0.6** introduces native **Equirectangular Mask Projection & Layer Auto-Detection** to the **360° Panoramic Splitter & Rig Projector**:

1. **Equirectangular Mask Auto-Detection & Fast Indexing**:
   * **Intelligent Discovery**: Automatically scans the dataset for equirectangular mask folders following common naming conventions (`mask`, `masks`, `Mask`, `Masks`, `Mask Pass`, `mask_pass`, `Mask_Pass`, `mask pass`, `Masks Pass`, `masks_pass`, `Mask 1`, `Mask 2`, or any subdirectory containing `mask`), while strictly ignoring output folders (`pinhole_rig`, `perspectives`, `results`, `images`, `cameras_csv`).
   * **Flexible Matching**: Indexes masks by filename, stem, lowercase stem, and index sequence. Handles suffix variations (e.g. `Container1_Mask000.jpg`, `Container1_Mask000_mask.jpg`, `frame_00001.png`).
   * **GUI & CLI Controls**: Added dedicated UI entry, Browse button, `🔍 Auto-detect` button, live mask telemetry badge in GUI Card 1, and CLI flags (`--mask-dir`, `--no-auto-mask`, `--masks-only`).

2. **RealityScan / RealityCapture Layer Naming Convention**:
   * Equirectangular masks are projected using the exact same virtual multi-camera rig parameters (FOV, resolution, camera orientations, distortion compensation) as the main RGB panoramas.
   * Saved into each perspective folder alongside the main images named with `.mask` inserted before the file extension:
     * Main view: `perspectives/down/Container1_Mask000_down.jpg`
     * Mask view: `perspectives/down/Container1_Mask000_down.mask.jpg`
   * When imported into RealityScan / RealityCapture, RealityCapture automatically detects and pairs `<image>.mask.<ext>` as the mask layer for `<image>.<ext>`.
   * **Critical Constraints Satisfied**:
     * **Zero XMP Sidecars**: Absolutely no `.xmp` sidecars are generated for the `.mask` files, preventing duplicate camera definitions in RealityScan.
     * **Clean CSV Manifests**: Neither `cameras_<cam>.csv` nor `all_cameras.csv` include `.mask` files, preserving single-camera-per-view tracking.

3. **Dedicated Masks-Only Projection Workflow**:
   * Added `split_masks_for_dataset()` helper function and GUI action button `🎭 Export Masks Only`.
   * Reads existing rig parameters directly from `rig_manifest.json` in `pinhole_rig/` and generates all perspective mask layers without needing to re-render or overwrite RGB images.

4. **Executed on User's Active Dataset**:
   * Dataset: `C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Insta Exports\VID_20260902_094505_00_003_frames\CC\Masked`
   * Equirectangular Masks: `CC\Masked\Mask Pass` (682 panorama masks, 7680x3840).
   * Processed all 682 masks across 6 cubemap perspectives (`back`, `down`, `front`, `left`, `right`, `up`) at 2400x2400 resolution.
   * **Generated 4,092 `.mask.jpg` files** across the 6 perspective folders in 152.5 seconds (0 mask XMP files, 0 CSV changes).

---

## Verification & Automated Test Results

* **Dedicated Mask Test Suite**: [`tests/test_splitter_masks.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/tests/test_splitter_masks.py) (4 tests, 0.145s)
  * Verified auto-detection of `Mask Pass`, convention matching, and output directory exclusion.
  * Verified exact filename, stem, and index matching in `build_mask_lookup()`.
  * Verified end-to-end projection of main images and `.mask.jpg` images with identical pixel dimensions.
  * Verified that `<image>.jpg.xmp` exists and `<image>.mask.jpg.xmp` does NOT exist.
  * Verified that `all_cameras.csv` contains zero `.mask` rows.
  * Verified standalone `split_masks_for_dataset()` workflow.
* **Full Regression Test Suite**:
  * **139 tests passed in 34.586s** (100% pass rate, 0 failures).
* **Upstream Cleanliness & SHA256 Parity**:
  * Upstream submodules (`vgg-ttt/`, `vggt-omega/`) and WSL `/home/ahmad/360DVO/` remain clean and untouched.
  * Byte-for-byte SHA256 identity between `360dvo_engine.py` and `dvo360_engine.py` (`60cf1f1a0e51...`).
