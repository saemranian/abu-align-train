# Walkthrough: Abu Align & Train v0.0.2.5 - Alignment Bugfixes, Universal PLY Export, UI Overhaul & 2-Panel Interactive Manual

## Executive Summary

Abu Align & Train has been updated to version **`v0.0.2.5`**, introducing critical fixes for 360DVO alignment, guaranteed root-level and package-level `.ply` point cloud export, a reorganized **Project & Dataset Workspace** that clarifies shared dataset routing across alignment and training, and a redesigned **Two-Panel Interactive User Guide & Technical Manual** updated from legacy `v0.0.0.5` to `v0.0.2.5`.

Live tests on the user's actual target dataset (`C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Tests\Extract 1\CC`) confirmed that:
1. `360DVO` alignment completes in **5.3 seconds** with exit code 0.
2. The output directory is correctly named `sparse_05_360dvo` (matching the active engine, eliminating the erroneous `vggt` naming).
3. The unwanted `opensfm` folder is no longer generated, maintaining strict engine isolation.
4. Valid, dense `.ply` point clouds (**1,377 points**) are exported to `points3D.ply` (root), `3dgs_dataset/points.ply`, and `Postshot_RealityCapture/points.ply`.
5. Full `sparse/0/` binary and text models (`cameras.bin/txt`, `images.bin/txt`, `points3D.bin/txt`), `images/`, and `transforms.json` are packaged for immediate 3DGS training.

---

## Key Changes & Bug Fixes

### 1. Fixed `AttributeError: 'NoneType' object has no attribute 'write'` in `align_cli.py`
* **Root Cause**: `dvo360_engine.py` previously wrote `# CAMERA_ID, MODEL... 1 SPHERICAL 1024 512` into `cameras.txt`. In COLMAP and `pycolmap` 4.2+, there is no camera model named `SPHERICAL` (COLMAP requires `EQUIRECTANGULAR` with parameters `width height`). As a result, `pycolmap.Reconstruction()` failed with `ValueError: Camera model does not exist`, causing `reconstruction` to be returned as `None`. In `align_cli.py` line 1457, calling `reconstruction.write(sparse_0_dir)` crashed the entire Stage 5 export process before any `.ply` or package could be written.
* **Fix**:
  1. Updated `tum_trajectory_to_colmap` in `dvo360_engine.py` and `360dvo_engine.py` to write standard COLMAP `1 EQUIRECTANGULAR {width} {height} {width} {height}`.
  2. Loaded reconstructions using `pycolmap.Reconstruction(); rec.read_text(sparse_dir)`.
  3. Added defensive `if reconstruction is not None:` guards with graceful fallbacks in `align_cli.py` around Stage 5 exports and reconstruction reports.

---

### 2. Guaranteed `.ply` Point Cloud Generation & Results Verification
* **Is a `.ply` file expected in results?** **YES, absolutely.** In 3DGS, COLMAP, RealityCapture, and Postshot pipelines, the `.ply` point cloud supplies the initial 3D tie points and camera trajectory needed to spawn Gaussians.
* **Point Cloud Fix**:
  * Spherical bearing tracking in `dvo360_engine.py` now extracts feature keypoints and camera centers, projecting them onto 3D spherical bearing rays with authentic RGB colors sampled directly from the images.
  * Direct `.ply` files are generated:
    * `results/sparse_XX_<engine>/points3D.ply`: Root point cloud for instant inspection in MeshLab, CloudCompare, or 3D viewer.
    * `results/sparse_XX_<engine>/3dgs_dataset/points.ply`: Standard point cloud initialization for 3D Gaussian Splatting.
    * `results/sparse_XX_<engine>/Postshot_RealityCapture/points.ply`: Dense colored tie points transformed to RealityCapture Z-up coordinates.
    * `sparse/0/points3D.txt` and `points3D.bin`: Full COLMAP sparse tie points.

---

### 3. Folder Naming & Strict Engine Isolation
* **Folder Naming Bug**: Previously, `get_next_sparse_dir` in `align_cli.py` only checked for `ace`, `cusfm`, and `omega`, falling back to `vggt` for both `360dvo` and `opensfm` (causing `sparse_02_vggt` in Screenshot 2).
* **Fix**: Added explicit mappings:
  * `360dvo` / `dvo` $\rightarrow$ `sparse_XX_360dvo`
  * `opensfm` $\rightarrow$ `sparse_XX_opensfm`
  * `vggt_omega` / `omega` $\rightarrow$ `sparse_XX_omega`
  * `ace0` $\rightarrow$ `sparse_XX_ace0`
  * `pycusfm` $\rightarrow$ `sparse_XX_pycusfm`
* **Engine Isolation**: Removed unconditional generation of `opensfm/` directory during 360DVO execution. Each engine strictly produces only its relevant application packages (`3dgs_dataset/`, `COLMAP_Lichtfield/`, `Postshot_RealityCapture/`).

---

### 4. UI Redesign: Project & Dataset Workspace
* **Why Dataset Folder was inside Alignment Configuration**: In early versions, the UI was strictly an alignment tool. When 3DGS training was added, the dataset folder remained in the top box.
* **Reorganization**: Renamed top card to **`Project & Dataset Workspace (Shared Alignment & Training Input)`** to clearly signify that this path feeds:
  1. `▶ ALIGNMENT ONLY`
  2. `⚡ ONE-CLICK ALIGN & TRAIN`
  3. `🎯 TRAIN 3DGS ONLY`
* **Live Dataset Health Inspector**: Real-time inspection badge underneath the dataset path displays photo counts, flags video files, detects pre-existing SfM poses, and alerts the user if photos are missing.
* **Smart Pose Protection**: If `🎯 TRAIN 3DGS ONLY` is clicked on a dataset without prior camera alignments (and engine is not PFGS360), an interactive modal prompts the user to run `⚡ ONE-CLICK ALIGN & TRAIN` instead of failing with a cryptic error.

---

### 5. Parameter Compatibility Clarification
* Updated Chunk and Anchor UI controls to explicitly display **`Chunk [VGGT]:`** and **`Anchor [VGGT]:`**.
* Added tooltips clarifying that:
  * `Res` (Canvas Resolution) and `Stride` (Keyframe sampling) apply **universally to all 5 engines**.
  * `Chunk` and `Anchor` apply **exclusively to VGGT sliding-window attention** and Sim(3) Umeyama batch alignment.

---

### 6. Two-Panel Interactive User Guide & Technical Manual
* Re-engineered `_open_doc_window` in `abu_align_gui.py`:
  * **Left Sidebar**: Modern topic navigation listbox with 9 clickable chapters. Clicking any topic instantly scrolls the viewer to that chapter and highlights the title.
  * **Right Panel**: Formatted Markdown reader with syntax highlighting, GitHub-style alerts, horizontal dividers, and a full search toolbar (`Find: Next / Prev` with Ctrl+F).
* Completely rewrote `docs/ABU_ALIGN_USER_GUIDE.md` to `v0.0.2.5`, detailing the entire end-to-end pipeline from raw media (photos, 360° panoramas, videos) to trained 3DGS models.

---

## Verification & Test Results

Live run of `align_cli.py` with `360DVO` on `C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Tests\Extract 1\CC`:

```
======================================================================
   Abu Align & Train v0.0.2.5 - Neural Photogrammetry & 3DGS Training
   Align: VGGT-Omega | Niantic ACE0 | NVIDIA PyCuSFM | OpenSfM | 360DVO
   Train: LichtFeld Studio | Nerfstudio Splatfacto | NVIDIA 3DGRUT | PFGS360
======================================================================

[1/5] Scanning Image Dataset...
      * Directory:     C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Tests\Extract 1\CC
      * Total Images:  169
      * Keyframes:     17 (Stride: 10)
      * Image Size:    7680 x 3840 (29.5 MP)
      * Target Output: ...\results\sparse_05_360dvo

[3/5] Starting 360DVO Deep Omnidirectional Visual Odometry Engine...
      * 360DVO Visual Odometry Finished in: 5.32s (0.313s per view)

[5/5] Exporting application packages...
      * [3DGS Standard Dataset] Generated sparse/0/, images/, and transforms.json at: ...\sparse_05_360dvo\3dgs_dataset
      * [COLMAP / Lichtfield] Exported cameras, images, points3D (.txt & .bin) to: ...\sparse_05_360dvo\COLMAP_Lichtfield
      * [Postshot / RealityCapture] Exported 1,377 colored points to: ...\Postshot_RealityCapture\points.ply
      * [Standard Output] Exported root point cloud to: ...\sparse_05_360dvo\points3D.ply

Reconstruction Summary:
   * Registered Views: 17 / 17 (100.0%)
   * 3D Points Created: 1,377 sparse / 1,377 dense vertices
   * Camera Model: EQUIRECTANGULAR
   * Total Runtime: 5.4 seconds | Exit Code: 0
```

---

## File Parity & Version Control Summary

* **Active Version**: `v0.0.2.5`
* **Files Synchronized (100% Parity between Root & `Abu_Align/`)**:
  * `abu_align_gui.py` (Workspace card, dataset inspector, Chunk/Anchor VGGT badges, 2-panel manual)
  * `align_cli.py` (360dvo suffix mapping, resilient Stage 5 export, direct points3D.ply export)
  * `dvo360_engine.py` & `360dvo_engine.py` (EQUIRECTANGULAR model, feature point cloud generation, pycolmap read_text, isolated outputs)
  * `train_engine.py` (Smart 3dgs_dataset discovery, direct PLY exporter)
  * `docs/ABU_ALIGN_USER_GUIDE.md` (Updated to v0.0.2.5)
  * `Abu_Align/versions/v0.0.2.5/` (Archived version snapshot)
