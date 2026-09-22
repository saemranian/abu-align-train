# Walkthrough: Abu Align & Train v0.0.3.0

## Overview

**Abu Align & Train v0.0.3.0** successfully implements **Phase 1** of our repository alignment plan: **Mapillary OpenSfM Engine & UI Alignment** (aligned with `https://github.com/mapillary/OpenSfM`).

This release provides comprehensive exposure of Mapillary OpenSfM's official configuration parameters directly inside the Abu Align GUI and CLI, backed by rich mouse-hover tooltips (`ToolTip`), robust dual-execution fallback (OpenSfM binary/Docker CLI $\leftrightarrow$ Native Python Spherical Structure-from-Motion on $S^2$ manifold), and seamless multi-format exports for 3D Gaussian Splatting (`3dgs_dataset/`, `reconstruction.json`, `metadata.json`, `transforms.json`, `sparse/0/`, `points3D.ply`).

---

## Key Changes & Implementations

### 1. Mapillary OpenSfM Engine Alignment (`opensfm_engine.py`)
- **Official OpenSfM Parameters Exposed**:
  - `feature_type`: Support for `SIFT` (recommended sub-pixel accuracy), `HAHOG` (Hessian-Affine + Histogram of Gradients), `ORB` (rapid binary features), and `AKAZE` (anisotropic diffusion).
  - `feature_min_frames`: Feature budget per equirectangular frame ($1,500$ to $5,000$, default $3,000$).
  - `matching_order_neighbors`: Sequential matching window ($3$ to $12$ frames, default $5$).
  - `matcher_type`: `WORDS` (Bag-of-Words vocabulary tree), `FLANN` (Fast Approximate Nearest Neighbors with reciprocal verification), and `EXHAUSTIVE` (Brute-Force pairwise matching).
  - `bundle_interval`: Bundle adjustment frequency (incremental BA runs every $1, 2, 5, 10$ frames).
  - `optimize_camera_parameters`: Refines spherical camera parameters (vertical aspect scale $s_y$) during bundle adjustment to compensate for non-ideal equirectangular stitches.
- **Improved Manifold Triangulation**:
  - Refined parallax threshold ($\alpha_{\min} = 0.5^\circ$) and chordal error threshold ($\varepsilon_{\max} = 0.08$) in `triangulate_multiview_rays`.
  - Ensures confident multi-view 3D tie points are preserved even across small baselines.
- **Dual-Execution Fallback**:
  - Automatically searches for external OpenSfM binary CLI in system PATH / Conda environments.
  - If external binary is not detected, automatically engages Native Python Spherical SfM on the $S^2$ manifold using Kronecker product epipolar estimation $(\mathbf{b}_2 \otimes \mathbf{b}_1) \text{vec}(E) = 0$, positive cheirality checking, closed-form multi-view ray intersection, and Ceres/SciPy Levenberg-Marquardt spherical bundle adjustment.

---

### 2. Command-Line Interface (`align_cli.py`)
- **New CLI Arguments Added**:
  - `--opensfm_min_features`: Feature budget per frame (default: 3000).
  - `--opensfm_neighbors`: Sequential matching window in frames (default: 5).
  - `--opensfm_bundle_interval`: Bundle adjustment interval (default: 1).
  - `--opensfm_optimize_camera`: Optimizes spherical camera calibration during BA.
- **Diagnostic Telemetry & Reconstruction Report**:
  - All OpenSfM parameters are prominently displayed in the CLI startup banner and logged in `reconstruction_report.txt`.

---

### 3. GUI Layout & Informative Mouse-Hover Tooltips (`abu_align_gui.py`)
- **Modern 3-Column OpenSfM Subpanel**:
  - **Column 1 (Camera & Projection Model)**:
    - Radiobuttons: `Equirectangular 360° (Spherical S²)`, `Perspective Pinhole (Rectilinear)`, `Fisheye (Ultra-Wide 180°+)`.
    - Checkbox: `Optimize Spherical Camera`.
  - **Column 2 (Features & Matcher Strategy)**:
    - Feature Extractor: `sift`, `hahog`, `orb`, `akaze`.
    - Feature Budget per View: `1500`, `2500`, `3000`, `4000`, `5000`.
    - Matcher Strategy: `words`, `flann`, `exhaustive`.
    - Matching Window (Neighbors): `3`, `4`, `5`, `8`, `12`.
    - BA Interval: `1`, `2`, `5`, `10` view(s).
  - **Column 3 (360° Multi-Camera Rig Projector & 3DGS Bridge)**:
    - `⚡ Auto-Project Multi-Camera Rig` checkbox.
    - Rig type (`ring`, `cubemap`, `dome`) and resolution selector.
    - `🌐 Open Rig Projector Dialog...` button for the interactive 3D virtual rig visualizer.
- **Comprehensive Tooltip Guidance on Mouse Hover**:
  - Every single control features a detailed `ToolTip` explaining what the parameter does, its role in the SfM pipeline, and best-practice recommended values.

---

### 4. Version Management & Parity Protocol (`v0.0.3.0`)
- Cascaded base-10 odometer increment from `v0.0.2.9` $\to$ `v0.0.3.0` via `version_manager.py`.
- Recompiled C# launcher executables using Microsoft .NET `csc.exe`:
  - `Abu_Align_and_Train.exe`
  - `Abu_Align_and_Train_v0.0.3.0.exe`
  - `Abu_Align/Abu_Align_and_Train.exe`
  - `Abu_Align/Abu_Align_and_Train_v0.0.3.0.exe`
- 100% file parity verified between workspace root and `Abu_Align/`.
- Full version snapshot stored at `Abu_Align/versions/v0.0.3.0/`.

---

## Verification & Test Results

### 1. OpenSfM Engine Unit Test
- Ran `run_opensfm_spherical_sfm` on equirectangular test sequence with custom parameters (`budget=3000, window=5, bundle_interval=2, optimize_camera=True`).
- **Telemetry Output**:
  - `* Feature Extractor: SIFT (Budget: 3000/frame)`
  - `* Matcher: FLANN (Neighbor Window: 5)`
  - `* Bundle Interval: Every 2 view(s)`
  - `* Optimize Cam: Enabled`
  - `* Spherical BA converged with final cost 0.0018 in 0.45s`
  - `* Refined spherical camera calibration: scale_y=1.00436`
  - `* Formats Exported: reconstruction.json, metadata.json, transforms.json, COLMAP sparse/0/, points3D.ply`
- **Result: PASSED (100%)**.

### 2. End-to-End CLI Pipeline Test
- Executed `align_cli.py -i scratch/test_frames -o scratch/test_cli_opensfm_phase1 -e opensfm --projection equirectangular --opensfm_feature sift --opensfm_matcher flann --opensfm_min_features 3000 --opensfm_neighbors 5 --opensfm_bundle_interval 2 --opensfm_optimize_camera`.
- **Result**: Registered 10/10 views in 2.90s, produced all universal dataset packages in `3dgs_dataset/`, `COLMAP_Lichtfield/`, and `Postshot_RealityCapture/`, and logged the diagnostic summary to `reconstruction_report.txt`.
- **Result: PASSED (100%)**.

### 3. File Parity & Snapshot Verification
- Verified 100% byte-for-byte parity across `opensfm_engine.py`, `align_cli.py`, `abu_align_gui.py`, `train_engine.py`, `360dvo_engine.py`, `dvo360_engine.py`, `version_manager.py`, `VERSION.txt`, `Abu_Align_and_Train.bat`, `Abu_Align.bat`, and `launcher.cs`.
- Snapshot confirmed in `Abu_Align/versions/v0.0.3.0/`.
- **Result: PASSED (100%)**.
