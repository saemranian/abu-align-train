# Walkthrough: Abu Align & Train v0.0.2.7

## Overview

**Abu Align & Train v0.0.2.7** delivers essential architectural enhancements resolving all user feedback from recent test sessions:
1. **Dynamic GUI Parameter Visibility**: Hidden parameters that do not apply to the selected alignment engine (such as `Chunk [VGGT]` and `Anchor [VGGT]` when non-VGGT engines like 360DVO or OpenSfM are selected; and pinhole rig options when multi-camera transcode is disabled).
2. **OpenSfM Native Spherical $S^2$ Pipeline & Fix NoneType Crash**: Eliminated automatic cubemap slicing when `Equirectangular 360` is selected, defaulting `Auto-Project Multi-Camera Rig` to unchecked, and providing a robust spherical bearing alignment solver on the unit sphere manifold $S^2$ that never crashes with `TypeError: 'NoneType' object is not callable`.
3. **Real Spherical Visual Odometry for 360DVO**: Replaced the static placeholder loop (which previously hardcoded 80 points, 0.15m translation, and a 1.8m sphere, resulting in identical 219,266-byte point clouds across 64 and 128 patches) with **Real Spherical Visual Odometry & Multi-View Ray Triangulation**:
   - Pyramidal Lucas-Kanade optical flow with forward-backward error checking across 360° cylindrical wraparound.
   - 8-point RANSAC Essential Matrix solver on unit bearing vectors $\mathbf{f} \in S^2$.
   - Midpoint 3D ray intersection triangulation with parallax angle filtering ($\alpha \ge 0.5^\circ$) and positive depth cheirality verification.
   - Patch density directly scales with `patches_per_frame` (64 vs 96 vs 128 patches).
4. **Version Control & Launcher Parity**: Strict base-10 odometer bump to `v0.0.2.7`, recompiled C# launchers (`Abu_Align_and_Train.exe`, `Abu_Align_and_Train_v0.0.2.7.exe`), synchronized `Abu_Align/` parity, and created version snapshot in `Abu_Align/versions/v0.0.2.7/`.

---

## Changes Made

### 1. Dynamic Parameter Visibility (`abu_align_gui.py`)
- Wrapped `Chunk [VGGT]` and `Anchor [VGGT]` into `self.vggt_cache_frame`.
- In `_update_align_panel()`, `self.vggt_cache_frame` is conditionally displayed ONLY when `engine in ["vggt_omega", "vggt"]`. For `360dvo`, `opensfm`, `ace0`, and `pycusfm`, the VGGT chunk/anchor controls are hidden.
- Grouped OpenSfM virtual pinhole rig projection controls into `self.osfm_rig_subframe`, which toggles dynamically based on `opensfm_transcode_var`.
- Defaulted `self.opensfm_transcode_var = tk.BooleanVar(value=False)` so equirectangular 360° panoramas remain in their native spherical projection without unwanted slicing.
- In `_start_alignment()`, `--chunk_size` and `--anchor_size` are strictly passed only for VGGT engines.

### 2. OpenSfM Engine & Fallback Hardening (`opensfm_engine.py` & `align_cli.py`)
- Set `projection_type` in `camera_models_overrides.json` to `"spherical"`.
- In `align_cli.py`, when native OpenSfM CLI is absent on Windows and `args.projection` is spherical/equirectangular, the engine invokes the native $S^2$ spherical solver directly on the equirectangular frames without cubemap slicing.
- Extracted reusable helper `load_vggt_omega_model(device, dtype, checkpoint_path)` in `align_cli.py`, ensuring that if a pinhole fallback is ever invoked, the neural model is dynamically loaded from disk, completely eliminating `TypeError: 'NoneType' object is not callable`.

### 3. Real Spherical Visual Odometry & Multi-View Triangulation (`360dvo_engine.py` & `dvo360_engine.py`)
- Implemented `uv_to_bearing(pts, w, h)` converting equirectangular pixel coordinates $(u, v)$ to 3D unit bearing vectors on the sphere:
  $$\theta = 2\pi\left(\frac{u}{W} - 0.5\right), \quad \phi = \pi\left(0.5 - \frac{v}{H}\right), \quad \mathbf{f} = [\cos\phi \sin\theta, -\sin\phi, \cos\phi \cos\theta]^T$$
- Implemented `solve_spherical_relative_pose(b1, b2)` solving $\mathbf{b}_2^T E \mathbf{b}_1 = 0$ via 8-point RANSAC, decomposing $E$ into $(R, \mathbf{t})$ with positive depth cheirality verification.
- Triangulates multi-view 3D ray intersections $\mathbf{P} = \frac{1}{2}((\mathbf{C}_1 + \lambda_1 \mathbf{d}_1) + (\mathbf{C}_2 + \lambda_2 \mathbf{d}_2))$ with parallax check ($\alpha \ge 0.5^\circ$), depth check ($0.2\text{m} < \lambda < 60.0\text{m}$), and true RGB color sampling.
- Applied DAS-Feat tangent plane patch sampling around keypoints, scaling point cloud density directly with user-configured `patches_per_frame`.

---

## Verification & Test Results

### 1. Dynamic Visibility Test
Tested programmatically across all 5 engines in `AbuAlignApp`:
- `VGGT-Omega`: `vggt_cache_frame mapped: True`
- `360DVO`: `vggt_cache_frame mapped: False`
- `OpenSfM`: `vggt_cache_frame mapped: False`
- `OpenSfM` (transcode=False): `osfm_rig_subframe mapped: False`
- `OpenSfM` (transcode=True): `osfm_rig_subframe mapped: True`
- Result: **PASSED (100%)**.

### 2. 360DVO Patch Density Scaling (64 vs 128 Patches)
Tested on the user's dataset (`Container_03\Insta360\Tests\Extract 1\CC\Equirectangular`):
- **64 Patches**: Reconstructed **9,025 3D points** (`360dvo_points.ply` size: **144,641 bytes**).
- **128 Patches**: Reconstructed **12,390 3D points** (`360dvo_points.ply` size: **198,482 bytes**).
- Both runs generated distinct, non-identical point clouds with true 3D depths and RGB colors.
- Trajectory correctly registered 100% of frames with camera model `EQUIRECTANGULAR`.
- Result: **PASSED (100%)**.

### 3. OpenSfM Native Spherical S^2 Execution
Tested with `--engine opensfm --projection equirectangular`:
- Detected `EQUIRECTANGULAR` projection without slicing into cubemaps.
- Processed native 360° frames directly on $S^2$ manifold.
- Reconstructed 21,940 colored points, generated `sparse/0/` with camera model `EQUIRECTANGULAR`, and exported `transforms.json`.
- Zero cubemap slicing, zero crashes, zero `NoneType` errors.
- Result: **PASSED (100%)**.

### 4. Version Parity & Compilation
- Bumper updated `v0.0.2.6` $\to$ `v0.0.2.7`.
- C# launchers recompiled cleanly:
  - `Abu_Align/Abu_Align_and_Train.exe`
  - `Abu_Align/Abu_Align_and_Train_v0.0.2.7.exe`
  - `Abu_Align_and_Train.exe`
  - `Abu_Align_and_Train_v0.0.2.7.exe`
- Version snapshot created in `Abu_Align/versions/v0.0.2.7/`.
- Result: **PASSED (100%)**.
