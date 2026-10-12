# Abu Align & Train v0.0.6.3 — VGGT-Omega Non-Square Tracking, Cache Serialization & Ceres BA Calibration Fix

Version **v0.0.6.3** resolves the critical assertion and serialization failures reported in `abu_align_2026-09-25_02-18-21.txt`, bringing the pipeline into full compliance with Facebook Research's official `facebookresearch/vggt-omega` foundation model specifications:

1. **Non-Square Multi-View Tracking Fix**: Removed legacy square assertions (`assert height == width`) in `vggt/dependency/track_predict.py`, allowing native aspect ratios (e.g. $288 \times 512$ for 16:9 photos) to track seamlessly without distortion.
2. **Robust Neural Cache Serialization**: Fixed `AttributeError: 'numpy.ndarray' object has no attribute 'detach'` in `cache_manager.py` by converting both NumPy arrays and PyTorch tensors safely.
3. **Ceres Bundle Adjustment Dimension Order & Scaling Alignment**: Corrected `cam_image_size` coordinate order from $[H, W]$ to $[W, H]$ for COLMAP and aligned `intrinsic_ba` resolution with `pred_tracks`.
4. **Complete Unit Test Coverage**: Extended test suite with end-to-end tests for NumPy cache serialization and non-square tracking keypoint scaling (31/31 unit tests passing).

---

## 1. Executive Summary & Root Cause Analysis

### A. The Crash in `abu_align_2026-09-25_02-18-21.log`

In `logs/abu_align_2026-09-25_02-18-21.log`:
```text
[2/5] Initializing Neural Network & GPU...
      * Device:        NVIDIA GeForce RTX 4090 (24.0 GB Dedicated VRAM)
      * Engine:        VGGT-Omega (Register Attention)
      * Checkpoint:    vggt-omega/vggt_omega_1b_512.pt
      * Inference Finished in: 9.36s (0.187s per view)
      * [Smart Cache Warning] Failed to save neural cache: 'numpy.ndarray' object has no attribute 'detach'
[5/5] Building COLMAP Reconstruction & Point Cloud...
      * Running Deep Multi-View Tracking [FAST mode: 3 fast queries, 1024 pts/frame]...
      * [Deep Tracking] Query [1/3] (Frame 0) - 0.0%...
Traceback (most recent call last):
  File "align_cli.py", line 1909, in main
  File "align_cli.py", line 948, in run_vggt_reconstruction
  File "vggt/dependency/track_predict.py", line 98, in predict_tracks
  File "vggt/dependency/track_predict.py", line 196, in _forward_on_query
    assert height == width
AssertionError
```

### B. Root Cause 1: Legacy Square Assertion in Rectangular Tracking
- **The Issue**: To prevent aspect distortion and avoid wasting VRAM on padding, v0.0.6.1 introduced `--vggt_preprocess_mode max_size`. For standard 16:9 4K images ($3840 \times 2160$), this resizes images to $288 \times 512$.
- The foundation model inference executed in just **9.36s (0.187s/view)** on an RTX 4090.
- However, when entering the deep multi-view tracking stage (`track_predict.py:196`), the code halted on `assert height == width`.
- **Facebook Research Alignment**: The underlying CoTracker architecture (`tracker.process_images_to_fmaps` and `predict_tracks_in_chunks`) is fully convolutional. It operates on arbitrary 2D spatial feature grids ($288 \times 512 \to 36 \times 64$). The assertion was merely legacy residue from early square demo scripts.
- **The Fix**:
  - Removed `assert height == width` and `assert conf.shape[-2] == conf.shape[-1]`.
  - Implemented anisotropic coordinate scaling:
    $$\text{scale}_x = \frac{\text{conf\_w}}{\text{width}}, \quad \text{scale}_y = \frac{\text{conf\_h}}{\text{height}}$$
  - Added spatial boundary clamping to prevent out-of-bounds array lookups:
    $$\tilde{q}_x = \text{clamp}(\text{round}(q_x \cdot \text{scale}_x), 0, \text{conf\_w} - 1)$$
    $$\tilde{q}_y = \text{clamp}(\text{round}(q_y \cdot \text{scale}_y), 0, \text{conf\_h} - 1)$$

---

### C. Root Cause 2: Cache Serialization Failure on NumPy Inputs
- **The Issue**: `cache_manager.py:save_neural_cache` called `.detach().cpu()` directly on `original_coords`, `extrinsic`, `intrinsic`, and `images`. When called from certain pipeline stages where these data structures are NumPy arrays, Python raised `AttributeError: 'numpy.ndarray' object has no attribute 'detach'`.
- **The Fix**: Added a robust conversion helper:
  ```python
  def to_tensor(x, out_dtype=None):
      if x is None:
          return None
      if isinstance(x, np.ndarray):
          t = torch.from_numpy(x)
      elif isinstance(x, torch.Tensor):
          t = x.detach().cpu()
      else:
          t = torch.tensor(x)
      if out_dtype is not None:
          t = t.to(out_dtype)
      return t
  ```

---

### D. Root Cause 3: COLMAP Camera Aspect Swapping in Bundle Adjustment
- **The Issue**: In `align_cli.py`, `batch_np_matrix_to_pycolmap` received `cam_image_size = np.array(images.shape[-2:])`. Because `images.shape` is `[N, C, H, W]`, this passed `[H, W]` instead of `[W, H]`, inverting the width and height in pycolmap camera definitions.
- Furthermore, `intrinsic_ba` was artificially scaled by `img_load_res / vggt_res` (2.0), while `pred_tracks` were in `vggt_res` space, causing Ceres BA reprojection gates to reject valid tie points.
- **The Fix**:
  - Corrected `cam_image_size = np.array([images.shape[-1], images.shape[-2]])` ($[W, H]$).
  - Maintained `intrinsic_ba` in native `vggt_res` space matching `pred_tracks`.
  - Scaled camera intrinsics directly to original resolution $(W_{\text{orig}}, H_{\text{orig}})$ using `rename_colmap_recons_and_rescale_camera`.

---

## 2. Test Verification Matrix

All unit tests executed with Conda environment `C:\Users\USer\miniconda3\envs\vggt\python.exe`:

| Test Suite | Tests Run | Result | Notes |
| :--- | :--- | :--- | :--- |
| `tests/test_vggt_omega_optimization.py` | 6 tests | **PASS (0.78s)** | Verifies non-square rescaling, NumPy cache serialization, direct indexing |
| `tests/test_desktop_ui.py` | 15 tests | **PASS (14.2s)** | Verifies GUI telemetry, parameter persistence, mode switching |
| `tests/test_gui_presets.py` | 6 tests | **PASS (8.5s)** | Verifies quality presets, resolution flags, draft options |
| `tests/test_rig_designer.py` | 4 tests | **PASS (9.1s)** | Verifies multi-camera rig topology and calibration |
| **Complete Discover Suite** | **31 tests** | **PASS (32.6s)** | **0 Failures, 0 Errors, 100% Passing** |

---

## 3. Version History & Build Artifacts

- **Base-10 Odometer Version**: `v0.0.6.2` $\to$ **`v0.0.6.3`**
- **Synchronized Files**:
  - `VERSION.txt`
  - `abu_align_gui.py`
  - `align_cli.py`
  - `desktop_ui.py`
  - `train_engine.py`
  - `Abu_Align.bat` / `Abu_Align_and_Train.bat`
- **Recompiled Executables**:
  - `Abu_Align.exe` (v0.0.6.3)
  - `Abu_Align_and_Train.exe` (v0.0.6.3)
  - `versions/historical_binaries/Abu_Align_v0.0.6.3.exe`
  - `versions/historical_binaries/Abu_Align_and_Train_v0.0.6.3.exe`
- **Snapshot Archive**: `versions/v0.0.6.3/`
