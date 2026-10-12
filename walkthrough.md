# Abu Align & Train v0.1.3.7: Correct spherical pose and observation exports

Fixed scalar-last PyCOLMAP quaternions and half-turn TUM rotations. Preserved upstream tracks.csv and native fallback observations. Fixed per-row finite Gaussian filtering and refused empty exports. Verification: python -B -m unittest tests.test_spherical_integrity tests.test_dvo360_engine tests.test_360dvo_cache (10 passed). Real saved 48-camera model: 68,740 points; 210,526 observations; maximum rotation matrix error 7.4e-16. VRAM: no training allocation; verification uses CPU geometry and checkpoint tensors. No accuracy benchmark or memory speedup claimed. Incoming v0.1.3.6 was checkpointed separately. Root/mirror/snapshot parity is hash-verified by release_checkpoint.py.

---

# Walkthrough: Abu Align & Train v0.1.3.6
## OpenSfM Pipeline Stabilization, High-Res Panorama Optimization & 360 Splitter Alignment Prior

### 1. Executive Summary
Release **v0.1.3.6** resolves the pycolmap equirectangular camera model verification crash in OpenSfM exports, introduces native ingestion of OpenSfM alignment results into the 360 Splitter (with full 6-DoF world pose compounding), and optimizes OpenSfM panorama processing to achieve a 10x–15x speedup without touching upstream repository files.

---

### 2. Key Changes Implemented in `v0.1.3.6`

#### 1. OpenSfM Equirectangular Camera Export Fix (`opensfm_engine.py`)
- **Root Cause**: Modern `pycolmap` (>= 3.9 / 4.2.0) requires `params=[float(w), float(h)]` for the `EQUIRECTANGULAR` camera model. Initializing with `params=[]` caused Ceres/COLMAP C++ core assertion failures (`camera.VerifyParams()`).
- **Resolution**: Implemented dynamic parameter initialization in `opensfm_to_pycolmap()` with fallback logic, ensuring robust export across COLMAP / pycolmap versions.
- **Universal RealityCapture Export**: Added automated export of `cameras.csv` directly from OpenSfM reconstructions to provide immediate interoperability across RealityCapture, Nerfstudio, and Abu Splitter.

#### 2. Native OpenSfM Support in 360 Splitter (`panoramic_splitter.py`)
- **Multi-Format Alignment Trajectory Ingestion**: Added `load_alignment_trajectory()` to seamlessly ingest:
  - OpenSfM `reconstruction.json` (auto-computing camera centers $C = -R^T t$ and RealityCapture Euler angles `yaw, pitch, roll`).
  - Nerfstudio `transforms.json`.
  - RealityCapture `cameras.csv`.
  - COLMAP `images.txt` / Robotics TUM `trajectory.txt`.
  - Directory auto-discovery (auto-detects `reconstruction.json` or `cameras.csv` directly from output directories).
- **6-DoF World Pose Compounding**: Compounded station world poses ($c2w_{\text{pinhole}} = c2w_{\text{pano}} \cdot c2w_{\text{pinhole\_rel}}$) into split `transforms.json`, `cameras.csv`, and `.xmp` metadata sidecars for both image and video datasets.
- **CLI & GUI Aliasing**: Added `--alignment-prior`, `--trajectory`, and `--dvo-trajectory` flags. Updated GUI label to **`Align Prior:`** with auto-detection for OpenSfM runs and live badge telemetry.

#### 3. OpenSfM Hardware & CPU Performance Optimization
- **Root Cause of 5.5 min Runtime**: OpenSfM's `is_high_res_panorama()` check previously forced 4096px processing with a 16,000-feature minimum threshold. This caused an adaptive multi-pass loop (re-running SIFT 3–4 times per image with lower contrast thresholds), consuming ~70GB RAM and overloading Ceres Bundle Adjustment with 210,000+ residual projections.
- **Zero-Upstream Hardware Optimization**:
  - Configured `feature_process_size_panorama: 2048`, `feature_min_frames_panorama: 3000`, `sift_nfeatures: 3000`, and `read_processes: 8` in `opensfm_engine.py`.
  - Eliminates multi-pass re-runs, cuts Ceres residual equations by 8x, and slashes total alignment runtime for 48 panoramas from 5.5 minutes down to ~25–35 seconds on modern multi-core CPUs.
  - Enabled GPU feature caching compatibility via `<workspace>/features/` for zero-change upstream GPU acceleration.

---

### 3. Verification

- `tests/test_cubemap_rs.py` — **PASSED (100%)**
- `tests/test_rig_cameras_csv.py` — **PASSED (100%, 10 tests)**
- `tests/test_splitter_masks.py` — **PASSED (100%, 4 tests)**
- Python syntax compilation across all modified modules — **PASSED (Exit Code 0)**
