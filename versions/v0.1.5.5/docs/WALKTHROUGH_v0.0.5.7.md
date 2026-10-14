# Abu Align & Train v0.0.5.7 — Fix OpenSfM Camera Coordinate Axis Gizmos in 3D Point Clouds

## Overview
Version **v0.0.5.7** resolves an issue in the **Mapillary OpenSfM** alignment pipeline where synthetic X (Red), Y (Green), and Z (Blue) camera coordinate axis gizmos were embedded into the exported 3D point clouds (`reconstruction.ply`, `points3D.ply`, `points.ply`), polluting the scene with floating RGB crosses and degrading subsequent 3D Gaussian Splatting (3DGS) training.

---

## Origin OpenSfM Repository Investigation

### 1. Code Inspection (`opensfm/io.py` & `opensfm/commands/export_ply.py`)
Inspecting the official Mapillary OpenSfM repository source code:
- In `opensfm/commands/export_ply.py`:
  ```python
  parser.add_argument(
      "--no-cameras",
      action="store_true",
      default=False,
      help="Do not save camera positions",
  )
  ```
- In `opensfm/io.py` (`reconstruction_to_ply()`):
  ```python
  if not no_cameras:
      for shot in reconstruction.shots.values():
          o = shot.pose.get_origin()
          R = shot.pose.get_rotation_matrix()
          for axis in range(3):
              c = np.eye(3)[axis] * 255  # Axis 0: [255,0,0], Axis 1: [0,255,0], Axis 2: [0,0,255]
              for depth in np.linspace(0, 2, 10):
                  p = o + depth * R[axis]
                  s = "{} {} {} {} {} {}".format(
                      p[0], p[1], p[2], int(c[0]), int(c[1]), int(c[2])
                  )
                  vertices.append(s)
  ```

### 2. Root Cause
- When OpenSfM runs `export_ply`, `--no-cameras` is `False` by default.
- OpenSfM draws 30 synthetic points (10 along +X in pure red, 10 along +Y in pure green, 10 along +Z in pure blue) for **every single camera pose** directly into the vertex buffer of `reconstruction.ply`.
- In the user's run with 50 registered cameras:
  - **Authentic 3D Tie Points**: 19,706
  - **Injected Camera Gizmo Points**: 50 cameras × 30 points = **1,500**
  - **Total in Point Cloud**: 19,706 + 1,500 = **21,206 vertices** (matching the 3D viewer exactly).
- When opened in a 3D viewer or used as initial Gaussians in LichtFeld Studio / Splatfacto, these coordinate axes appeared as red, green, and blue rays along the camera trajectory.

---

## Fixes Implemented in v0.0.5.7

### 1. Enforced `--no-cameras` Flag in OpenSfM Execution (`opensfm_engine.py`)
Updated the OpenSfM execution stages for both native Windows and WSL2:
- WSL2: `wsl_bridge.run_command_in_wsl(f"{WSL_OPENSFM_BIN} {cmd_name} --no-cameras '{wsl_out}'")` when `cmd_name == "export_ply"`.
- Windows: `cmd = [self.opensfm_bin, "export_ply", "--no-cameras", self.output_dir]`.
This instructs OpenSfM itself never to generate synthetic camera gizmos.

### 2. Prioritized Pure Tie Points from `reconstruction.json` (`align_cli.py` & `opensfm_engine.py`)
`reconstruction.json` contains only authentic Ceres bundle-adjusted 3D tie points and zero synthetic camera points.
- In `align_cli.py`, added `load_opensfm_points_from_json()` to load the exact, clean point cloud directly from `reconstruction.json`.
- In `opensfm_to_transforms_json()`, prioritized writing `points3D.ply` and `points.ply` directly from `reconstruction.json["points"]`.

### 3. Added Automated Camera Gizmo Stripper (`strip_camera_gizmos_from_ply`)
Implemented a robust filter in `opensfm_engine.py` that strips any pure RGB coordinate axes from PLY files and updates the element vertex header count:
```python
def strip_camera_gizmos_from_ply(ply_path: str, output_path: Optional[str] = None) -> bool:
    # Filters out pure Red (255,0,0), Green (0,255,0), and Blue (0,0,255) axis vertices
```
This is automatically run whenever existing PLY files are transferred to `3dgs_dataset/` or root `points3D.ply`.

### 4. Sanitized Existing Run Point Clouds
Executed `strip_camera_gizmos_from_ply()` on the user's `results/sparse_06_opensfm` dataset:
- `points3D.ply`: 21,206 -> **19,706 vertices** (1,500 gizmo points stripped)
- `3dgs_dataset/points.ply`: 21,206 -> **19,706 vertices**
- `3dgs_dataset/points3D.ply`: 21,206 -> **19,706 vertices**
- `opensfm_workspace/reconstruction.ply`: 21,206 -> **19,706 vertices**

---

## Verification
- Verified that `reconstruction.json` contains exactly 19,706 authentic points with 0 synthetic axis points.
- Verified that all sanitized PLY files contain 19,706 vertices with zero pure-RGB camera gizmos remaining.
- Bumped version to `v0.0.5.7` with synchronized GUI, CLI, batch scripts, and recompiled C# launchers.

---

## Modified Files
- `opensfm_engine.py`
- `align_cli.py`
- `VERSION.txt`
- `abu_align_gui.py`
- `desktop_ui.py`
- `train_engine.py`
- `launcher.cs`
- `Abu_Align.bat`
- `Abu_Align_and_Train.bat`
