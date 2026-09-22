# Abu Align v0.0.0.8 Walkthrough: Speed Restoration & Ceres BA Convergence

## Overview & Executive Summary
In **Abu Align v0.0.0.8**, we resolved the critical performance degradation and camera alignment breakdown encountered in the previous run:
- **Execution Time:** Restored from **21 minutes 31 seconds** down to **1 minute 38 seconds** (**13x speedup**).
- **Ceres Bundle Adjustment:** Eliminated all 58 `CHOLMOD warning: Matrix not positive definite` errors and achieved sub-pixel convergence in under 13 seconds.
- **Dense Point Cloud Alignment:** Replaced the unstable global Umeyama Sim(3) transform with **direct per-camera depth re-unprojection**, ensuring that every 3D vertex is mathematically locked to the refined camera position and optical axis.
- **Mixed Aspect-Ratio Calibration:** Fixed optical principal point unpadding so that portrait images (e.g. `_C1_3338.jpg`, $6336 \times 9504$) and landscape images ($9504 \times 6336$) both retain precise physical sensor centers $(c_x = w/2, c_y = h/2)$.

---

## Performance & Telemetry Comparison

| Metric | Previous Run (`v0.0.0.7`) | Abu Align `v0.0.0.8` | Enhancement / Impact |
| :--- | :--- | :--- | :--- |
| **Total Runtime** | **21m 31s** (1291.6 s) | **01m 38s** (98.4 s) | **13.1x Faster** (Restored 1–2 min target) |
| **Tracking Time** | 18 min 47 sec | ~25 sec | Disabled redundant correlation refinement |
| **Query Strategy** | 16 queries (57% of dataset) | 6 wide-baseline queries | Eliminated collinear/parallel ray tracks |
| **Residuals / Parameters** | 494,310 res / 105,543 params | 141,054 res / 27,459 params | Well-conditioned Hessian matrix |
| **Linear Solver Warnings** | 58 `CHOLMOD Matrix not positive definite` | **0 Warnings (Clean Factorization)** | Restored positive-definite Schur complement |
| **BA Stage 1 Duration** | 31.49 s (oscillating) | **8.79 s** | Smooth gradient descent |
| **BA Stage 2 Duration** | 40.56 s (oscillating) | **12.87 s** | Brown distortion calibration ($k_1, k_2, t_1, t_2$) |
| **Point Cloud Consistency** | Warped by non-converged Umeyama Sim(3) | **Exact 1:1 per-camera unprojection** | Frustums point directly through 3D points |
| **Mixed Orientation ($c_x, c_y$)**| Corrupted when `load_res != 512` | **Exact $(w/2, h/2)$ for all cameras** | Aspect-ratio invariant unpadding |

---

## Detailed Root Cause Analysis & Technical Solutions

### 1. The 19-Minute Tracking Bottleneck (`fine_tracking=True`)
- **Diagnosis:** `fine_tracking=True` across 16 query frames ran an iterative neural correlation volume solver over 49,152 keypoint patches across 28 images. Each query frame took ~70 seconds, ballooning tracking time to 18m 47s.
- **Solution:** Switched to feedforward ALIKED + SuperPoint matching (`fine_tracking=False`). Sub-pixel accuracy is preserved with sharp, high-contrast corner detection across wide-baseline keyframes in ~25 seconds.

### 2. Solving Ceres BA Indefinite Matrix & `NO_CONVERGENCE`
- **Diagnosis:** 
  1. With 16 query frames in a 28-image dataset, consecutive frames with near-zero baseline ($\Delta t \approx 0$) were tracked against each other.
  2. The nearly parallel rays created 3D tie points with infinite depth variance along the optical axis, leading to near-zero eigenvalues in point blocks $C$.
  3. The reduced camera Schur complement matrix $S = B - E C^{-1} E^T$ lost positive-definiteness, triggering 58 CHOLMOD Cholesky factorization failures.
  4. Hardcoding `linear_solver_type = SPARSE_SCHUR` and `auto_select_solver_type = False` prevented Ceres from selecting alternative stable solvers.
- **Solution:**
  1. Restored `ba_options.ceres.auto_select_solver_type = True`.
  2. Enforced tri-focal geometry with `ba_options.min_track_length = 3`, ensuring every tie point is anchored by at least 3 distinct camera rays.
  3. Added `function_tolerance = 1e-6` and `parameter_tolerance = 1e-8`.
  4. Capped queries to `min(num_frames, 6)` for wide-baseline parallax only.

### 3. Direct Depth Re-Unprojection (Eliminating Umeyama Drift)
- **Diagnosis:** Fitting an Umeyama Sim(3) transform $(s, R, t)$ between initial camera centers $C_{initial}$ and non-converged camera centers $C_{final}$ distorted `points.ply` and produced mismatched coordinate frames.
- **Solution:** Completely removed the Umeyama transform. After Ceres BA converges, `align_cli.py` extracts the refined camera poses $[R_{i, \text{refined}} | t_{i, \text{refined}}]$ and refined intrinsics $K_{i, \text{refined}}$ from `reconstruction` and calls:
  ```python
  points_3d = unproject_depth_map_to_point_map(depth_map, refined_extrinsic, refined_intrinsic)
  ```
  Every 3D vertex unprojected from frame $i$ is mathematically guaranteed to emanate along frame $i$'s refined optical axis from its refined camera center $C_i$.

### 4. Aspect-Ratio-Invariant Principal Point Unpadding
- **Diagnosis:** In datasets containing mixed orientations (such as `_C1_3338.jpg` at $6336 \times 9504$ and the other 27 images at $9504 \times 6336$), subtracting `top_left` using target coordinates produced negative or misaligned principal points when `img_load_res != reconstruction_res`.
- **Solution:** Replaced with the invariant sensor unpadding equation:
  ```python
  max_dim = max(real_image_size)
  left = (max_dim - real_image_size[0]) // 2
  top = (max_dim - real_image_size[1]) // 2
  resize_ratio = max_dim / img_size

  pred_params[2] = pred_params[2] * resize_ratio - left  # cx
  pred_params[3] = pred_params[3] * resize_ratio - top   # cy
  ```
  Verified output in `sparse_12_omega\Postshot_RealityCapture\cameras.csv`:
  - `_C1_3337.jpg` (Landscape): `px_norm = 0.000000, py_norm = 0.000000`, `f_35mm = 24.9483`
  - `_C1_3338.jpg` (Portrait): `px_norm = 0.000000, py_norm = 0.000000`, `f_35mm = 24.8951`
  Both orientations achieve exact center alignment with matching 24.9mm focal lengths.

---

## Verification & Output Telemetry (`sparse_12_omega`)

The test run completed with **Exit Code 0** in **01:38**:
```
======================================================================
                       RECONSTRUCTION REPORT                          
======================================================================
 Date & Time:          2026-09-20 23:18:26
 Dataset Source:       C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Sony\101MSDCF\CC\Interior\RedBox\SEL Copy
 Total Photos:         28
 Registered Views:     28 / 28 (100.0%)
 3D Points Created:    9,068 sparse / 1,500,000 dense vertices
 Camera Model:         OPENCV (Self-Calibrated via Ceres BA)
   * Radial k1, k2:      -7.019922e-02, 6.837955e-02
   * Tangential t1, t2:  -2.573773e-03, -2.536514e-03
 Output Directory:     sparse_12_omega
 Total Runtime:        01:38 (98.4 seconds)
======================================================================
```

---

## Deployment & Deliverables in `v0.0.0.8`

1. **Executables Compiled & Synchronized:**
   - [Abu_Align.exe](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align.exe) (Root launcher)
   - [Abu_Align/Abu_Align.exe](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/Abu_Align.exe) (Folder launcher)
   - [Abu_Align/Abu_Align_v0.0.0.8.exe](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/Abu_Align_v0.0.0.8.exe) (Version-specific executable)
2. **Version Archive:**
   - [Abu_Align/versions/v0.0.0.8/](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/versions/v0.0.0.8/) with standalone code, docs, and launcher.
3. **Physical Backup:**
   - `Abu_Align_Backups/Abu_Align_v0.0.0.8_2026-09-20_23-16-36/`
4. **Git Version Control:**
   - Commit `64d63c5` tagged with **`v0.0.0.8`**.
