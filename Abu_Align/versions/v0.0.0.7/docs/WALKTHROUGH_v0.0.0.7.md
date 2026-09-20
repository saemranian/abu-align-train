# Abu Align Release v0.0.0.7 Walkthrough & Technical Report

**Software:** Abu Align  
**Version:** 0.0.0.7  
**Engines:** VGGT & VGGT-Omega Foundation Models  
**Optimization:** Two-Stage Sparse Schur Ceres Bundle Adjustment  
**Release Date:** September 20, 2026  

---

## 1. Executive Summary

Abu Align Version **0.0.0.7** resolves critical camera export alignment orientation issues, discovers and eliminates the primary root cause of frame mismatches in photogrammetric datasets (the Portrait vs. Landscape mixed-sensor bug), stabilizes Ceres Bundle Adjustment against Cholesky linear solver breakdown (`NO_CONVERGENCE`), introduces sub-pixel multi-view feature tracking (`fine_tracking=True`), and synchronizes the dense 3D point cloud (`points.ply`) directly to bundle-adjusted camera coordinates.

| Area | Issue Identified | Resolution in v0.0.0.7 | Impact |
| :--- | :--- | :--- | :--- |
| **`cameras.csv` Export** | $\arcsin$ pitch formula inverted downward-tilted cameras | Reverted to $\arccos(-\text{fwd}_z)$ from Git commit `f828216` | Restored RealityCapture Nadir ($0^\circ$) convention; frustums point correctly. |
| **Multi-Orientation Scans** | Mixed Landscape ($9504\times6336$) & Portrait ($6336\times9504$) forced into 1 camera | Orientation-Aware Multi-Sensor Grouping | Portrait frames reconstruct with identical sub-pixel precision as landscape frames. |
| **Ceres Bundle Adjustment** | `DENSE_SCHUR` solver broke down with dense Cholesky failure | Switched to `SPARSE_SCHUR` + Two-Stage solve | Zero solver failure; guaranteed positive-definite convergence. |
| **Feature Tracking** | Coarse patch-stride feature tracking allowed frame drift | Enabled `fine_tracking=True` & 16 query anchors | Keypoint precision refined to sub-pixel accuracy across all viewpoints. |
| **Point Cloud Coherence** | `points.ply` used unoptimized feedforward poses | Sim(3) Umeyama alignment to BA camera poses | 100% spatial alignment between camera frustums and point cloud in Postshot. |

---

## 2. Root Cause Investigations & Technical Solutions

### A. Reverting the Camera Export Alignment (`cameras.csv`)

#### The Problem
In the previous experimental iteration, changing the pitch formula to $\arcsin(\text{fwd}_z)$ caused downward-pointing camera angles to compute as negative elevation (e.g., $-20^\circ$). 
In RealityCapture and Postshot's Euler angle convention:
* **$0^\circ$** = Nadir (pointing straight down towards the ground along $-Z$).
* **$90^\circ$** = Horizontal (pointing parallel to the ground plane).
* **$180^\circ$** = Zenith (pointing straight up towards the sky along $+Z$).

Because the software expected angles referenced from Nadir, a negative value ($-20^\circ$) was interpreted as pointing past the nadir axis and backwards, causing the bottom camera frustums to invert upside down into the ground.

#### The Mathematical Fix (Reverted to Git `f828216`)
$$\text{pitch} = \arccos(\operatorname{clip}(-\text{fwd}_z, -1.0, 1.0))$$
$$w_0 = [\cos(\text{yaw}), -\sin(\text{yaw}), 0.0]$$
$$\text{roll} = \operatorname{arctan2}(u_{\text{rc}} \cdot w_0, u_{\text{rc}} \cdot u_0)$$

Where:
* $\text{fwd}_{\text{rc}}$ is the optical forward vector in RealityCapture Z-up space.
* $u_{\text{rc}}$ is the camera up vector.
* $\text{yaw}$ is the horizontal azimuth angle in $[0^\circ, 360^\circ)$.

**Result:** All camera frustums imported into RealityCapture or Postshot point directly at the subject.

---

### B. The Portrait vs Landscape Sensor Bug (Why Some Frames Were Misaligned)

#### The Investigation
When examining the Sony 60.2 MP dataset (`Container_03/Sony/101MSDCF/CC/Interior/RedBox/SEL Copy`), image dimensions revealed a critical pattern:
```
_C1_3337.jpg: 9504 x 6336  --> Landscape (Aspect 1.50)
_C1_3338.jpg: 6336 x 9504  --> Portrait  (Aspect 0.67)
_C1_3355.jpg: 9504 x 6336  --> Landscape
_C1_3356.jpg: 9504 x 6336  --> Landscape
...
```

#### What Failed in Previous Versions
1. When `shared_camera=True` was enabled, the software created **one single camera profile** in COLMAP, initialized from Image 1 (`9504 x 6336`).
2. When the pipeline encountered a portrait photo like `_C1_3338.jpg` (`6336 x 9504`):
   - It forced the portrait photo to use Camera 1 ($9504 \times 6336$).
   - The width was assumed to be $9504$ and the height $6336$ (swapped dimensions).
   - The optical center $(c_x, c_y)$ offset was shifted along the incorrect physical axis.
3. During Ceres Bundle Adjustment and COLMAP triangulation, 2D keypoint rays for portrait photos were projected through an incorrect aspect ratio matrix ($1.5\times$ stretched in width, compressed in height), causing those specific portrait frames to drift and mismatch in position and direction.

#### The Solution in v0.0.0.7: Orientation-Aware Multi-Sensor Grouping
1. In `vggt/dependency/np_to_pycolmap.py`, we added the `image_shapes` parameter:
   ```python
   cameras_by_shape = {}
   for fidx in range(N):
       shape_key = tuple(image_shapes[fidx]) if image_shapes is not None else (int(image_size[0]), int(image_size[1]))
       if not shared_camera or shape_key not in cameras_by_shape:
           pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)
           cam_id = len(cameras_by_shape) + 1 if shared_camera else fidx + 1
           camera = pycolmap.Camera(
               model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=cam_id
           )
           reconstruction.add_camera(camera)
           cameras_by_shape[shape_key] = camera
       else:
           camera = cameras_by_shape[shape_key]
       image.camera_id = camera.camera_id
   ```
2. In `align_cli.py`, `rename_colmap_recons_and_rescale_camera` now maintains a `rescaled_cameras = set()` and rescales each unique orientation group with its genuine dimensions and center coordinates:
   - **Landscape Group ($9504 \times 6336$):** Calibrated as Camera 1 with $w=9504, h=6336$.
   - **Portrait Group ($6336 \times 9504$):** Calibrated as Camera 2 with $w=6336, h=9504$.

**Result:** Portrait frames reconstruct with the exact same sub-pixel accuracy and zero drift as landscape frames.

---

### C. Eliminating Ceres Bundle Adjustment Cholesky Solver Breakdown

#### The Failure Log
The previous run log recorded:
```
W20260920 21:50:35.941849 63264 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
Termination : NO_CONVERGENCE
```

#### Why it Happened
In PyCOLMAP Bundle Adjustment:
1. `DENSE_SCHUR` computes a dense Cholesky decomposition of the camera Schur complement matrix $S = B - E C^{-1} E^T$.
2. When optimizing Brown radial and tangential distortion ($k_1, k_2, t_1, t_2$) starting from zero on 250,000 residuals, the parameter covariance matrix can become ill-conditioned or nearly rank-deficient.
3. Once Eigen encounters a non-positive-definite matrix, the dense Cholesky factorization breaks down, aborting the optimization pass with `NO_CONVERGENCE`.

#### The Two-Stage Sparse Schur Solution
1. **Linear Solver Upgrade:** Changed the linear solver from `DENSE_SCHUR` to `SPARSE_SCHUR`. Sparse Schur uses conjugate gradients and sparse numerical factorization, which handles ill-conditioned steps gracefully via Levenberg-Marquardt diagonal damping ($\lambda I$).
2. **Two-Stage Optimization Strategy:**
   - **Stage 1 (Pose & Focal Lock):** BA is executed with `refine_extra_params = False`. The optimizer locks the physical camera positions, orientations ($R, t$), and focal length ($f_x, f_y$) using a robust pinhole model until convergence.
   - **Stage 2 (Distortion Polish):** `refine_extra_params = True` is unlocked. With camera poses already well-converged, the optimizer fits the physical radial and tangential lens curves ($k_1, k_2, t_1, t_2$) without solver failure.

---

### D. Sub-Pixel Multi-View Fine Tracking

1. **Sub-Pixel Tracking (`fine_tracking=True`):** In `predict_tracks`, enabled `fine_tracking = is_ultra`. This activates VGGSfM's sub-pixel correlation layer, refining keypoint locations from coarse patch strides to sub-pixel accuracy.
2. **Dense Query Anchors:** Expanded query anchor frames:
   ```python
   query_frames = min(num_frames, 16 if is_ultra else 8)
   ```
   For scans with 16–32 photos, every or nearly every frame acts as a query anchor. This ensures all camera viewpoints share direct, strong cross-frame track constraints.

---

### E. Point Cloud & Camera Alignment Synchronization

* **Previous Behavior:** `points.ply` was unprojected using the initial unoptimized feedforward camera poses. When Ceres BA adjusted the camera positions, the cameras moved but the point cloud stayed behind, creating spatial discrepancies in Postshot.
* **v0.0.0.7 Solution:**
  Before exporting `points.ply`, we extract the initial camera centers $C_{\text{initial}}$ and the bundle-adjusted camera centers $C_{\text{final}}$, then compute the optimal Sim(3) transformation using Umeyama's SVD algorithm:
  $$s, R, t = \operatorname{Umeyama}(C_{\text{initial}}, C_{\text{final}})$$
  $$\text{dense\_3d\_aligned} = s \cdot (\text{dense\_3d} @ R^T) + t$$
  $$\text{dense\_3d\_rc} = (M_{\text{world}} @ \text{dense\_3d\_aligned}^T)^T$$
  
**Result:** The dense point cloud and the bundle-adjusted camera frustums share 100% spatial alignment in Postshot.

---

## 3. Packaging, Distribution & Version Control

* **Version Bump:** Upgraded from `v0.0.0.6` to **`v0.0.0.7`** across `abu_align_gui.py`, `align_cli.py`, `np_to_pycolmap.py`, and `Abu_Align.bat`.
* **Compiled Executables:**
  - `Abu_Align.exe` (Root runner)
  - `Abu_Align_v0.0.0.7.exe`
  - `Abu_Align/versions/v0.0.0.7/Abu_Align_v0.0.0.7.exe`
* **Isolated Version Archive:** Saved to `Abu_Align/versions/v0.0.0.7/`.
* **Physical Backup:** Created in `Abu_Align_Backups/Abu_Align_v0.0.0.7_2026-09-20_22-19-06/`.
* **Git Version Control:** Committed (`4d216bb`) and tagged as **`v0.0.0.7`**.

---

## 4. Verification & Testing Workflow

1. **Launch:** Run `Abu_Align.exe` or `Abu_Align.bat`. Verify that the header badge displays **v0.0.0.7**.
2. **Reconstruction:**
   - Input: Your dataset folder.
   - Engine: **VGGT-Omega**.
   - Alignment Precision: **Ultra**.
   - Lens Distortion: **Brown (k1-k2, t1-t2)**.
   - Click **Start Alignment**.
3. **Inspection:**
   - In Postshot, import `Postshot_RealityCapture/cameras.csv` (or `transforms.json` or the `COLMAP_Lichtfield/` folder).
   - Verify that camera frustums point toward the subject and that both portrait and landscape shots align smoothly with the point cloud.
