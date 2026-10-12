# Abu Align & Train v0.0.7.1 — Multi-Engine Alignment Run Discovery, Point Cloud & Camera Trajectory 3D Browser Viewer

Version **v0.0.7.1** completely overhauls the **Scene Inspector 3D Browser Viewer** (`fast_viewer.py`), resolving a critical dropdown isolation bug where only `lichtfeld_models` was shown, and enabling full interactive 3D inspection of tie point clouds and camera trajectories across **all alignment runs** (`sparse_*`, `dvo_*`, etc.) produced by any Abu Align engine (VGGT-Omega, VGGT, OpenSfM, 360DVO, ACE0, and NVIDIA PyCuSFM).

---

## 1. Problem Description & Root Causes

When launching the 3D Browser Viewer (`http://localhost:8080/`) from the GUI after training:
1. **Viewport Black Screen & Single Run in Dropdown**:
   - The Reconstruction dropdown in the viewer showed only a single entry: `lichtfeld_models`.
   - The 3D viewport was completely pitch black with 0 points and 0 cameras rendered.
2. **Root Cause 1 — Training Output Override in GUI**:
   - In `abu_align_gui.py`, `self.latest_output_dir` was originally set to the alignment run (e.g. `results/sparse_10_omega`). However, when 3DGS training (LichtFeld Studio) was launched and completed, `self.latest_output_dir` was overwritten with the training subfolder: `.../results/sparse_10_omega/3dgs_models/lichtfeld_models`.
   - `_launch_viewer` passed this deep subfolder via `--input` to `fast_viewer.py`.
3. **Root Cause 2 — Rigid 2-Level Traversal in Project Resolution**:
   - In `fast_viewer.py`, `resolve_project_structure` only checked whether `parent` or `grandparent` was named `"results"`.
   - Because `target_dir` was `.../results/sparse_10_omega/3dgs_models/lichtfeld_models`, its parent was `3dgs_models` and grandparent was `sparse_10_omega`—neither was `"results"`.
   - Consequently, `fast_viewer.py` fell into `runs = {"lichtfeld_models": ".../lichtfeld_models"}`.
   - `lichtfeld_models` contained only `project.licht` (0 point clouds, 0 camera files), yielding an empty screen.
4. **Root Cause 3 — Camera Frustums Disabled & Viewport Unframed**:
   - `show_cams_check` had `initial_value=False` (cameras hidden by default).
   - There was no auto-framing on connection, so even when points were present, user viewports could start looking into empty space.
5. **Root Cause 4 — Engine Format Disconnects**:
   - `fast_viewer.py` only supported text COLMAP `images.txt`, TUM, and `transforms.json`. Other standard outputs—such as COLMAP binary `images.bin` (pycolmap/COLMAP), OpenSfM `reconstruction.json`, ACE0 `poses_final.txt`, and RealityCapture `cameras.csv`—were not supported.

---

## 2. Technical Implementation

### A. Upward Directory Traversal & Multi-Run Discovery (`fast_viewer.py`)
- Rewrote `resolve_project_structure(target_dir)` with an upward loop that traverses ancestor directories until finding `results/`.
- Once `results/` is discovered:
  - `dataset_dir = os.path.dirname(results_dir)`
  - Discovers **all** alignment runs inside `results/` (`sparse_10_omega` down to `sparse_01_omega`), ignoring internal caches like `.abu_cache` and bare `3dgs_models` folders.
  - Sorts runs in modification-time descending order (latest run first).
  - Automatically identifies which child run led to `target_dir` and assigns it as `initial_selected`.

### B. Unified Multi-Engine Camera Parsers (`fast_viewer.py`)
Implemented native, dependency-free parsers for all Abu Align photogrammetry formats:
1. `load_cameras_from_colmap_bin`: High-performance binary struct parser for `images.bin` extracting `qvec` + `tvec` and converting OpenCV world-to-cam to camera-to-world (c2w).
2. `load_cameras_from_opensfm`: Reads OpenSfM `reconstruction.json`, applying Rodrigues rotation formula on angle-axis vectors and inverting poses to world frame.
3. `load_cameras_from_ace0`: Parses ACE0 `poses_final.txt` / `poses_subpixel.txt` (`filename qw qx qy qz x y z focal confidence`).
4. `load_cameras_from_cameras_csv`: Parses RealityCapture / Postshot CSV position records.
5. `load_run_cameras`: Central dispatch engine prioritizing TUM → COLMAP txt → COLMAP bin → transforms.json → OpenSfM → ACE0 → CSV.

### C. Asset Scoring & 3DGS Splat Isolation (`fast_viewer.py`)
- Updated `find_run_assets` with point cloud priority scoring:
  - `points3D.ply` (COLMAP / VGGT-Omega standard) = 100 pts
  - `points.ply` (Postshot / RealityCapture) = 95 pts
  - `360dvo_points.ply` = 90 pts
  - `reconstruction.ply` = 85 pts
  - `pc_final.ply` / `point_cloud.ply` = 80 pts
- **Strict Isolation**: Explicitly filters out and assigns score `-1000.0` to any `.ply` inside `3dgs_models/`, `checkpoints/`, `weights/`, or 0-byte files, preventing Gaussian Splat checkpoints from corrupting point cloud rendering.

### D. Viewport Auto-Framing & Default Display Settings (`fast_viewer.py`)
- Set `show_cams_check = server.gui.add_checkbox("Camera frustums", initial_value=True)` (Default ON).
- Set `show_traj_check = server.gui.add_checkbox("Trajectory", initial_value=True)` (Default ON).
- Added consecutive duplicate filtering before Catmull-Rom spline construction, with a robust fallback to `add_line_segments` if spline generation encounters numerical singularities.
- Added dynamic camera frustum scaling matching scene bounds (`np.ptp(cam_positions)`).
- Implemented automatic scene bounding box calculation (2%–98% percentiles of points + cameras) that sets client camera `look_at` and `position` on run load and via `@server.on_client_connect` for newly opened browser tabs.
- Handled empty or incomplete runs gracefully without throwing unhandled exceptions.

### E. GUI Output Directory Separation (`abu_align_gui.py`)
- Split output tracking into:
  - `self.latest_alignment_dir`: Preserves the photogrammetry alignment output directory (`results/sparse_XX_*`).
  - `self.latest_train_dir`: Tracks 3DGS training output (`3dgs_models/lichtfeld_models`).
- In `_launch_viewer`: passes `self.latest_alignment_dir` (falling back to `self.latest_output_dir` or `dataset`), preventing 3DGS training output from masking alignment runs.

---

## 3. Verification & Testing

1. **Automated Unit Tests** (`tests/test_fast_viewer.py`):
   - `test_resolve_project_structure_upward_traversal`: Verified passing a deep path (`results/sparse_02/3dgs_models/lichtfeld_models`) properly discovers all sibling runs and sets `initial_selected = sparse_02`.
   - `test_find_run_assets_scoring_and_rejection`: Verified rejection of splats inside `3dgs_models` and prioritization of `points3D.ply`.
   - `test_colmap_bin_parser`: Verified binary struct unpacking and w2c-to-c2w transformation.
   - `test_opensfm_parser`: Verified Rodrigues angle-axis to quaternion/position conversion.
   - `test_ace0_parser`: Verified ACE0 pose line parsing and coordinate frame inversion.
2. **Full Regression Suite**:
   - Ran `python -m unittest discover tests`: All 57 unit tests passed in 31.06s.
3. **Live Verification on User Dataset**:
   - Tested target path: `.../Orbiting_object... frames/results/sparse_10_omega/3dgs_models/lichtfeld_models`.
   - Result: Discovered all 10 runs (`sparse_10_omega` through `sparse_01_omega`), correctly selected `sparse_10_omega`, loaded 300,000 points from `points3D.ply`, and loaded 443 cameras from `COLMAP_Lichtfield/images.txt`.

---

## 4. Version Bump & Compilation
- Cascaded base-10 odometer increment from `v0.0.7.0` to `v0.0.7.1` in `version_manager.py`.
- Recompiled native C# launchers:
  - `Abu_Align.exe`
  - `Abu_Align_and_Train.exe`
  - `versions/historical_binaries/Abu_Align_v0.0.7.1.exe`
  - `versions/historical_binaries/Abu_Align_and_Train_v0.0.7.1.exe`
- Created full snapshot archive at `versions/v0.0.7.1/`.
