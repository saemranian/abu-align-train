# v0.0.8.6 — RealityScan Engine, Smooth 3D Navigation, Frustum Thumbnails & Photo Windows

## Outcome
Abu Align & Train v0.0.8.6 integrates Epic Games / Capturing Reality RealityScan into the multi-engine photogrammetry suite with prior trajectory ingestion, brings buttery-smooth 60 FPS navigation to the native 3D workspace, displays image thumbnails directly inside camera frustums, adds dedicated separate camera photo viewer windows to both 3D viewers, and restores the browser trajectory fitting and run folder navigation controls.

## Changes

### 1. RealityScan Photogrammetry & Prior Trajectory Alignment Engine
- **Engine Core (`realityscan_engine.py`)**: Full adapter supporting auto-detection of RealityScan 2.2, 2.1, 2.0, and RealityCapture installations.
- **Prior Trajectory Ingestion**: Ingests camera trajectories calculated by earlier runs (VGGT-Omega, VGG-TTT, ACE0, PyCuSFM, OpenSfM, 360DVO) or `transforms.json` into RealityScan as imported trajectory priors (`cameras.csv`).
- **XML Import Definition**: Automatically generates the format definition XML matching GUID `{35CD8B84-6573-417D-8FEE-BE8BBEEC00D3}` (`Image X/Lon Y/Lat Z/Alt Yaw Pitch Roll`), comma-separated, local Euclidean coordinate space, and global camera prior accuracy.
- **Alignment & Calibration Settings (Screenshots 2 & 3)**:
  - Prior pose: Position and orientation, local:1 - Euclidean, Omega/Phi/Kappa, Accuracy: Global camera prior settings.
  - Prior calibration: Approximate, 35mm equivalent focal length, principal point (0, 0), skew 0.0, aspect 1.0.
  - Feature detection: High quality, 80,000 max features per Mpx, 320,000 max features per image, Low overlap, Downscale 1, 40,000 preselector features, High detector sensitivity.
  - Alignment: 2.0 px max reprojection error, Force component rematch True, Brown3 distortion model, GPU acceleration enabled.
- **Dual Execution Modes**: Headless background CLI with real-time stdout streaming directly into Abu Align's activity console, or Interactive RealityScan GUI window.
- **Export Harmonization**: Automatically produces COLMAP (`sparse/0/`), Postshot (`cameras.csv`, `points.ply`, `transforms.json`), and 3DGS dataset directories for immediate training in LichtFeld, Splatfacto, 3DGRUT, or PFGS360.
- **CLI & GUI Integration**: Added `--engine realityscan` (and aliases `realitycapture`, `rc`) with all `--rs_*` options to `align_cli.py`, added RealityScan cards and parameter panel to `desktop_ui.py` and `abu_align_gui.py`.

### 2. Smooth 3D Workspace Navigation
- **60 FPS Responsive Viewport (`workspace_3d.py`)**: Resolved navigation stutter and lag caused by rasterizing and depth-sorting (`np.argsort`) 300,000 points on every mouse drag event.
- **Dynamic Drag Decimation**: Tracks mouse dragging state (`_is_dragging=True`). During active orbit and pan, points are decimated to ~25,000 points, depth sorting is bypassed, and point radius is locked to 1. Full point resolution and depth sorting are restored instantaneously on mouse release.

### 3. Camera Frustum Thumbnails & Separate Photo Viewer Windows
- **Native 3D Workspace (`workspace_3d.py` & `native_inspection.py`)**:
  - Image thumbnails are rendered directly onto camera frustums in the 3D viewport.
  - Added `CameraPhotoViewer(tk.Toplevel)` window with photo preview, Zoom/Pan, Next/Previous camera navigation, Focus in 3D, Look Through, and "Open in Windows Viewer" (`os.startfile`).
  - Added `🖼️ View Photo` button and `Auto-open Photo` checkbox to inspect toolbar.
- **Browser 3D Viewer (`viewer_workspace.py`)**:
  - Frustums now display image thumbnails via Viser's native `image=` frustum plane texture mapping.
  - Added `🖼️ Open Image in Window` button and `Open image on select` toggle.
  - Selecting any camera opens the photo in the native Windows image viewer and pops up an in-browser photo modal with full details.

### 4. Restored Browser Navigation Controls (`viewer_workspace.py`)
- Restored `🎯 Fit the trajectory` button in the Viser browser sidebar, centering and framing the camera trajectory bounding box.
- Restored `📂 Open Run Folder` button in the Viser browser sidebar to launch Windows File Explorer directly into the active run directory.

## Verification
- **Automated Test Suite**: 103 unit tests passed in 56 seconds across all modules (`test_realityscan_engine`, `test_native_inspection`, `test_viewer_workspace`, `test_gui_presets`, `test_desktop_ui`, `test_vggt_omega_optimization`, `test_vggttt_engine`, etc.).
- **Code Compilation**: Verified all 8 core modified/new Python modules compile with 0 errors via `py_compile`.
- **C# Launcher Compilation**: Recompiled native `Abu_Align.exe` and `Abu_Align_and_Train.exe` binaries using .NET `csc.exe`.

## Files and Release
- Core: `realityscan_engine.py`, `align_cli.py`, `desktop_ui.py`, `abu_align_gui.py`, `workspace_3d.py`, `native_inspection.py`, `viewer_workspace.py`, `version_manager.py`.
- Tests: `tests/test_realityscan_engine.py`, `tests/test_native_inspection.py`, `tests/test_viewer_workspace.py`, `tests/test_gui_presets.py`.
- Release Snapshot: `versions/v0.0.8.6/` in the project root.
