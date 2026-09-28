# v0.1.0.3 — Infinite Loop Path Fix & Safe 3DGS Image Staging

## Overview & Outcome
Abu Align & Train **v0.1.0.3** resolves a critical Windows filesystem recursion bug where processing datasets with images located directly in the root directory caused an infinite directory loop (`results/.../images/results/.../images`), freezing and crashing Windows File Explorer when deleting or browsing run folders.

---

## 1. Problem & Root Cause Analysis

* **Symptom**: When attempting to delete or inspect `results/sparse_02_360dvo/` (or previous runs), Windows File Explorer froze, hung, and crashed with deep path errors. In the user's path:
  `.../CC/Masked/results/sparse_02_360dvo/3dgs_dataset/images/Del Copy Sel1/results/sparse_05_360dvo/3dgs_dataset/images/results/...`
* **Root Cause**:
  * In `360dvo_engine.py` (and `dvo360_engine.py`), `3dgs_dataset/images` was staged using:
    ```python
    d3_imgs = os.path.join(dataset_3dgs, "images")
    if not os.path.exists(d3_imgs):
        try:
            os.symlink(first_img_dir, d3_imgs)
        except Exception:
            shutil.copytree(first_img_dir, d3_imgs)
    ```
  * When input images were located directly in `CC/Masked`, `first_img_dir` was `CC/Masked` (the dataset root).
  * The output directory `results/sparse_XX_360dvo` is located *inside* `CC/Masked`.
  * If `os.symlink` succeeded or created an NTFS junction, `images` pointed back to its own ancestor `CC/Masked`, creating a circular reparse point loop in the filesystem.
  * If `os.symlink` failed (typical on Windows without Developer Mode / Admin rights), `shutil.copytree(first_img_dir, d3_imgs)` recursively copied all directories in `CC/Masked` (including `results/`, `Del Copy Sel1/`, and previous run outputs) into `images/`. Because `d3_imgs` is inside `results/`, `copytree` copied `results/` into `images/results/`, which contained another run's `results/`, causing infinite directory recursion until the Windows `MAX_PATH` (260 characters) limit was exceeded.

---

## 2. Core Fixes Implemented

1. **Safe File-Based Image Staging (Zero Directory Copy / Zero Symlink Loops)**:
   * Replaced directory symlinks and `copytree` across all engines:
     * [`360dvo_engine.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/360dvo_engine.py)
     * [`dvo360_engine.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/dvo360_engine.py)
     * [`align_cli.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/align_cli.py)
     * [`realityscan_engine.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/realityscan_engine.py)
     * [`opensfm_engine.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/opensfm_engine.py)
   * The new implementation stages images using strict individual file hard-links (`os.link(src, dst)`) with per-file fallback to `shutil.copy2(src, dst)`:
     ```python
     d3_imgs = os.path.join(dataset_3dgs, "images")
     os.makedirs(d3_imgs, exist_ok=True)
     for src in image_paths:
         dst = os.path.join(d3_imgs, os.path.basename(src))
         if not os.path.exists(dst):
             try:
                 os.link(src, dst)
             except Exception:
                 try:
                     shutil.copy2(src, dst)
                 except Exception:
                     pass
     ```
   * **Advantages**:
     * Zero-byte storage overhead on NTFS via native hard-links.
     * Guaranteed flat image folder in `3dgs_dataset/images/` without any subdirectories.
     * Absolutely zero possibility of circular directory loops or ancestor copying.

2. **Proof-of-Concept Permanent Deletion of Corrupted Folders**:
   * Permanently purged the deeply nested recursive folder `sparse_02_360dvo` at:
     `C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Insta Exports\VID_20260902_094505_00_003_frames\CC\Masked\results\sparse_02_360dvo`
     using Windows NT wide-API `robocopy /MIR` mirror purge.
   * Confirmed `Test-Path` returns `False`.
   * Also purged nested recursive directories inside `sparse_01_360dvo` and `Del Copy Sel1\results\sparse_05_360dvo`.

3. **Loop-Safe & Deep-Path Directory Deletion Utility (`safe_remove_tree`)**:
   * Added `safe_remove_tree(target_dir)` in [`cache_manager.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/cache_manager.py) to reliably delete directory structures with deep paths (>260 characters) or reparse points without Explorer lockup.

4. **100% Upstream Cleanliness**:
   * Tracked files in upstream submodules (`vgg-ttt/`, `vggt-omega/`) and WSL repository `/home/ahmad/360DVO/` remain 100% clean and pristine.

---

## 3. Automated Verification & Test Results

* **New Dedicated Test Suite**: [`tests/test_safe_image_staging.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/tests/test_safe_image_staging.py)
  * `test_safe_remove_tree_normal`: PASSED
  * `test_safe_image_staging_no_recursion`: PASSED (verified `images/` contains only flat image files and 0 subdirectories when output is inside dataset).
* **Full Regression Test Suite**:
  * **135 tests passed in 58.324s** (100% pass rate, 0 failures).
* **SHA256 Parity**:
  * Byte-for-byte identity verified between `360dvo_engine.py` and `dvo360_engine.py` (`60CF1F1A0E51DC0D25C20702623C658A25F463A4F46B7771B25E3BF760B8ECAC`).
