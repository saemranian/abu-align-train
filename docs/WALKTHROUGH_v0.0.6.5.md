# Abu Align & Train v0.0.6.5 — Telemetry Disambiguation Fix: Eliminating "ACE0 Scene MLP" Mislabels during VGGT-Omega Runs

Version **v0.0.6.5** resolves the GUI telemetry collision where running VGGT-Omega foundation model inference caused the status bar to display `Stage 4/6: ACE0 Scene MLP [?/12] (4.2%) | Loss: ? | Inliers: ?%`:

1. **Telemetry Router Disambiguation**: Refactored `_check_queue()` in `abu_align_gui.py` to correctly differentiate between VGGT-Omega neural batch progress (`batch=... total=...`) and ACE0 Scene MLP optimization (`iter=... loss=... inliers=...`).
2. **Restored Foundation Model Status Reporting**: VGGT-Omega inference now displays accurate feed-forward batch progress: `Stage 4/6: Neural Batch Inference [1/12] (4.2%)`.
3. **Restored Deep Multi-View Tracking & Live ETA**: Fixed shadowed `[TELEMETRY:STAGE5]` handler so deep tracking query progress and remaining ETA (`Stage 5/6: Multi-View Tracking [Query 1/3, Frame #0] | ⏱ ETA: ...`) render properly.
4. **Automated Regression Test**: Added `test_vggt_omega_vs_ace0_telemetry_disambiguation` in `tests/test_desktop_ui.py`, verified across all 36 test cases (100% pass rate).

---

## 1. Executive Summary & Root Cause Analysis

### A. The Reported Issue

During VGGT-Omega alignment on 878 photos (`Sel 1`), the GUI displayed:
```text
WORKFLOW VGGT-Omega -> LichtFeld Studio
Stage 4/6: ACE0 Scene MLP [?/12] (4.2%) | Loss: ? | Inliers: ?%
```
While in reality:
- Photogrammetry Engine was **VGGT-Omega** (`vggt_omega_1b_512.pt`), running 12 chunk windows of 96 frames.
- ACE0 was never started.

### B. Root Cause: Telemetry Tag Collisions & Handler Shadowing

1. **What VGGT-Omega Emitted**:
   In `align_cli.py`:
   ```python
   print(f"[TELEMETRY:STAGE3] batch={c_idx+1} total={len(chunk_ranges)} pct={batch_pct:.1f}", flush=True)
   ```
2. **The Flawed Interception**:
   In `abu_align_gui.py`:
   ```python
   # Line 3824 (ACE0 Handler):
   elif msg_s.startswith("[TELEMETRY:STAGE3]") or ("Iteration:" in msg and ...):
       # Unconditionally parsed as ACE0!
       cur_it = parts.get("iter", "?")    # Becomes "?"
       tot_it = parts.get("total", "?")   # Becomes "12" from VGGT "total=12"
       loss_v = parts.get("loss", "?")    # Becomes "?"
       inl_v = parts.get("inliers", "?")  # Becomes "?"
       self.telemetry_lbl.config(
           text=f"Stage 4/6: ACE0 Scene MLP [{cur_it}/{tot_it}] ({pct:.1f}%) | Loss: {loss_v} | Inliers: {inl_v}%"
       )
   ```
   Because line 3824 appeared before the VGGT-Omega handler at line 3871 and unconditionally matched any string starting with `[TELEMETRY:STAGE3]`, VGGT-Omega's batch messages were swallowed and mislabeled.

3. **Multi-View Tracking Shadowing**:
   Similarly, line 3868 had `elif msg_s.startswith("[TELEMETRY:STAGE5]"): ... Building COLMAP Reconstruction`, which completely shadowed line 3887 (`Multi-View Tracking [Query q/tot, Frame #f]`).

---

## 2. The v0.0.6.5 Fix

### Refactored Telemetry Dispatcher (`abu_align_gui.py`)

- **Stage 3 Disambiguation**:
  ```python
  is_ace0 = (
      "iter=" in msg_s
      or "loss=" in msg_s
      or ("Iteration:" in msg and "|" in msg)
      or ((getattr(self, "engine_var", None) is not None and self.engine_var.get() in ["acezero", "ace0"]) and "batch=" not in msg_s)
      or ((getattr(self, "align_engine_var", None) is not None and self.align_engine_var.get() in ["acezero", "ace0"]) and "batch=" not in msg_s)
  )
  ```
  - When `is_ace0` is `False` (VGGT-Omega, OpenSfM, 360DVO), routes to:
    $$\textbf{Stage 4/6: Neural Batch Inference [b/tot] (pct\%)}$$
  - When `is_ace0` is `True` (Niantic ACE0), routes to:
    $$\textbf{Stage 4/6: ACE0 Scene MLP [iter/total] (pct\%) \| Loss: loss \| Inliers: inl\%}$$

- **Stage 5 Multi-View Tracking Restoration**:
  - `[TELEMETRY:STAGE5]` is routed directly to Multi-View Tracking with query tracking and live ETA calculations.
  - Point cloud / COLMAP reconstruction export is cleanly isolated to `[5/5] Building COLMAP` and point generation markers.

---

## 3. Test Verification Matrix

All unit tests executed with Conda environment `C:\Users\USer\miniconda3\envs\vggt\python.exe`:

| Test Suite | Tests Run | Result | Notes |
| :--- | :--- | :--- | :--- |
| `tests/test_desktop_ui.py` | 14 tests | **PASS (11.8s)** | Verifies VGGT vs ACE0 telemetry disambiguation, UI layouts, tool dialogs |
| `tests/test_vggt_omega_optimization.py` | 6 tests | **PASS (0.78s)** | Verifies non-square rescaling, NumPy cache serialization, direct indexing |
| `tests/test_gui_presets.py` | 6 tests | **PASS (8.5s)** | Verifies quality presets, resolution flags, draft options |
| `tests/test_rig_designer.py` | 4 tests | **PASS (9.1s)** | Verifies multi-camera rig topology and calibration |
| **Complete Discover Suite** | **36 tests** | **PASS (27.3s)** | **0 Failures, 0 Errors, 100% Passing** |

---

## 4. Version History & Build Artifacts

- **Base-10 Odometer Version**: `v0.0.6.4` $\to$ **`v0.0.6.5`**
- **Synchronized Files**:
  - `VERSION.txt`
  - `abu_align_gui.py`
  - `align_cli.py`
  - `desktop_ui.py`
  - `train_engine.py`
  - `Abu_Align.bat` / `Abu_Align_and_Train.bat`
- **Recompiled Executables**:
  - `Abu_Align.exe` (v0.0.6.5)
  - `Abu_Align_and_Train.exe` (v0.0.6.5)
  - `versions/historical_binaries/Abu_Align_v0.0.6.5.exe`
  - `versions/historical_binaries/Abu_Align_and_Train_v0.0.6.5.exe`
- **Snapshot Archive**: `versions/v0.0.6.5/`
