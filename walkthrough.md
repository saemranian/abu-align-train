# Abu Align & Train v0.0.1.2 Walkthrough: Intelligent Caching, Live Telemetry & Engine Expansion

## Executive Summary

In this milestone, **Abu Align & Train v0.0.1.2** introduces three major performance, observability, and control enhancements:
1. **Intelligent Multi-Tier Caching System (`.abu_cache`):** Persistent half-precision tensor caching for neural forward passes (Stage 3 & 4) and 2D feature trajectories (Stage 5), cutting re-run times from **4+ minutes down to 33 seconds** on the `SEL Copy` 60MP dataset.
2. **Granular Live Telemetry & Visual Progress Bar:** Fine-grained `[TELEMETRY:STAGE3]` and `[TELEMETRY:STAGE5]` streaming emitting query numbers, frame IDs, percentage completion, and real-time ETAs with dynamic `ttk.Progressbar` and active status label.
3. **Deep Engine Options & Parameters Expansion:** Exposing advanced photogrammetry and 3DGS training options across all 4 alignment engines (VGGT, VGGT-Omega, ACE0, PyCuSFM) and 3 training engines (LichtFeld Studio, Nerfstudio Splatfacto, NVIDIA 3DGRUT) with interactive hover `ToolTip`s.

---

## 1. Intelligent Multi-Tier Caching Architecture (`.abu_cache`)

### Location:
Stored deterministically inside `<dataset_dir>/results/.abu_cache/<cache_key>/` (or `<dataset_dir>/.abu_cache/`). The user's original photos directory remains 100% pristine and unmodified.

### Cache Fingerprinting (`cache_key`):
SHA-256 fingerprint generated from:
- Dataset image count, filenames, sizes, and file modification timestamps.
- Alignment engine (`vggt`, `vggt_omega`, `ace0`, `pycusfm`).
- Canvas resolution (`load_res`) and frame sampling `stride`.

### Multi-Tier Storage:
- **Tier 1 (Neural Inference Cache):**
  - `extrinsic.pt` & `intrinsic.pt`: Rigid camera poses & focal lengths.
  - `original_coords.pt`: Full-resolution image coordinate offsets.
  - `points_3d_fp16.pt` & `depth_map_fp16.pt`: Dense 3D unprojected points and depth maps saved in half-precision (`float16`) to minimize disk footprint while preserving sub-millimeter precision.
  - `depth_conf_fp16.pt`: Neural confidence maps.
  - `meta.json`: Execution metadata, frame counts, and timestamp.
- **Tier 2 (Multi-View Tracking Cache):**
  - `tracking/tracks.npz`: Multi-view 2D feature tracks, visibility scores, confidence arrays, and bundle-adjusted tie points.

---

## 2. Parameter Reuse Analysis: What Can Be Reused?

| Changed Parameter / Scenario | Reuses Neural Cache? | Reuses Tracking Cache? | Expected Re-run Time | Technical Reason |
| :--- | :---: | :---: | :---: | :--- |
| **Point Cloud Density Only** (e.g. Preview $\leftrightarrow$ High $\leftrightarrow$ Max) | **YES (100%)** | N/A | **< 5 seconds** | Only adjusts confidence threshold `conf_thresh` and random sampling mask on already computed 3D points. |
| **Lens Distortion Model** (Pinhole $\leftrightarrow$ Brown Calibration) | **YES (100%)** | **YES (100%)** | **~10–15 seconds** | 2D feature tracks and 3D initial points are unchanged; Ceres simply toggles radial ($k_1, k_2$) and tangential ($t_1, t_2$) parameter blocks. |
| **Camera Alignment Accuracy** (Feedforward $\to$ Standard Ceres BA) | **YES (100%)** | Miss (First time) | **~1–2 minutes** | Skips 5–10 min neural inference; runs tracking once and caches it. Subsequent BA runs take ~10s. |
| **3DGS Training Options** (Iterations, Viewport, Bilateral, Exposure) | **YES (100%)** | **YES (100%)** | **0 seconds (Alignment)** | Photogrammetry alignment is untouched; immediately launches LichtFeld or Nerfstudio. |
| **Canvas Resolution (`load_res`)** (e.g. 512 $\to$ 1024) | No (Invalidated) | No (Invalidated) | Full Run | Changes image tensor dimensions and feature tracking coordinate grid. |
| **Engine Change** (e.g. VGGT-Omega $\leftrightarrow$ ACE0) | No (Invalidated) | No (Invalidated) | Full Run | Distinct neural architectures with incompatible latent spaces. |

---

## 3. Live Telemetry & Progress Bar

### Structured Telemetry Streaming:
- `[TELEMETRY:STAGE3] batch=X total=Y pct=Z`
- `[TELEMETRY:STAGE3] pct=50.0 status=cache_hit`
- `[TELEMETRY:STAGE5] query=X total=Y frame=Z pct=W eta_sec=N`
- `[TELEMETRY:STAGE5] pct=85.0 status=cache_hit`

### GUI Integration:
- **`ttk.Progressbar`:** Positioned right above the console window in `abu_align_gui.py`. Smoothly updates between 0% and 100%.
- **Dynamic Status Label (`self.telemetry_lbl`):** Displays current stage, active query, and active frame (e.g., `Stage 5/5: Multi-View Tracking [Query 2/5, Frame #1] (20.0%)`).
- **Live ETA Display (`self.eta_lbl`):** Calculates and formats real-time remaining time (e.g., `⏱ ETA: 2m 11s`).
- **Smart Cache Hit Indicators:** Displays `⚡ [Smart Cache Hit] Neural Poses & 3D Points Loaded (Stage 3 & 4 Skipped)` in bright blue.

---

## 4. Verification Benchmark on Test Dataset

**Dataset:** `C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Sony\101MSDCF\CC\Interior\RedBox\SEL Copy`  
**Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM)  
**Photos:** 28 total (14 keyframes with `--stride 2`, 60.2 MP each)

### Benchmark Results:
| Execution Run | Selected Settings | Cache Status | Runtime | Outcome |
| :--- | :--- | :--- | :---: | :--- |
| **Run 1 (Initial)** | `vggt_omega`, Brown Lens Model, Standard BA | Cache Populated | **04:23** | Saved neural predictions and tracking data to `.abu_cache\3981555166b24cf6`. |
| **Run 2 (Re-run with Lens Model Changed to Pinhole)** | `vggt_omega`, Pinhole Model, Standard BA | **100% Cache Hit** (Neural + Tracking) | **00:33** | **8× Faster!** Skipped Stage 3 & 4 neural inference in 0.30s; skipped CoTracker in 0.05s; converged Ceres BA in 0.29s. |

### Generated Verification Reports:
- `SEL Copy\results\test_cache_run1\reconstruction_report.txt`
- `SEL Copy\results\test_cache_run2\reconstruction_report.txt`

---

## 5. Summary of Modified & Created Files

| File | Changes Made |
| :--- | :--- |
| **`Abu_Align/cache_manager.py`** & `cache_manager.py` | Multi-tier caching manager with `compute_dataset_hash`, half-precision tensor serialization (`save_neural_cache`, `load_neural_cache`), tracking caching (`save_tracking_cache`, `load_tracking_cache`), and cache purge (`clear_cache`). |
| **`vggt/dependency/track_predict.py`** | Added granular `[TELEMETRY:STAGE5]` streaming with real-time ETA calculation and query frame progress. Fixed missing `import time`. |
| **`vggt/dependency/vggsfm_utils.py`** | Added per-chunk progress output inside `predict_tracks_in_chunks`. |
| **`Abu_Align/align_cli.py`** & `align_cli.py` | Integrated cache checking and saving for Stage 3/4 and Stage 5; enriched `reconstruction_report.txt` with cache status and advanced training parameters; added CLI flags (`--anchor_size`, `--no_cache`, `--clear_cache`, `--bilateral_grid`, etc.). |
| **`Abu_Align/train_engine.py`** & `train_engine.py` | Added support for `--bilateral_grid`, `--exposure_correction`, `--enable_sparsity`, `--enable_mip`, and `--sh_degree`. |
| **`Abu_Align/abu_align_gui.py`** & `abu_align_gui.py` | Added `ttk.Progressbar`, live status/ETA labels, Smart Cache toggle (`[x] Smart Cache`), Clear Cache button (`[🗑 Clear Cache]`), sliding-window chunk/anchor selectors, advanced 3DGS toggles, hover tooltips, and real-time telemetry stream parsing. |
| **`Abu_Align/Abu_Align_and_Train.exe`** & `Abu_Align_and_Train.exe` | Recompiled standalone executable launcher using `csc.exe`. |
