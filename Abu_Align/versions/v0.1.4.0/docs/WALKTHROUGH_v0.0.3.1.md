# Walkthrough: Abu Align & Train v0.0.3.1

## Overview

**Abu Align & Train v0.0.3.1** successfully implements **Phase 2** of our repository alignment plan: **360DVO Deep Omnidirectional Visual Odometry Engine & UI Alignment** (aligned with the IEEE RA-L 2026 paper and official repository *Deep Visual Odometry for Monocular 360-Degree Camera*).

This release introduces comprehensive configuration of 360DVO's core spherical odometry and factor-graph optimization features directly in the GUI and CLI, backed by rich mouse-hover tooltips (`ToolTip`), dynamic polar mask filtering, sliding-window Omnidirectional Direct Bundle Adjustment (ODBA), metric scale estimation, and universal dataset bridge exports (`3dgs_dataset/`, `sparse/0/`, `transforms.json`, `opensfm/`, `reconstruction_report.txt`).

---

## Key Changes & Implementations

### 1. 360DVO Engine Alignment (`360dvo_engine.py` & `dvo360_engine.py`)
- **Official 360DVO (IEEE RA-L 2026) Capabilities**:
  - **DAS-Feat Patch Density & Sampling**: Support for `64`, `96`, `128`, and `256` patches per frame distributed on the spherical manifold $S^2$.
  - **Dynamic Polar Mask Cutoff (`polar_mask_pct`)**: Excludes high-distortion zenith ($y < h \cdot \text{pct}$) and nadir ($y > h(1 - \text{pct})$) regions, cleanly eliminating tripod heads, selfie sticks, and featureless sky glare.
  - **Dynamic Metric Scale Estimation (`metric_scale`)**: Calibrates monocular bearing disparities $\Delta \theta$ into real-world metric translations, featuring stationary clamp ($0.0\text{m}$ translation when angular disparity $< 0.002\text{ rad} \approx 0.12^\circ$) and proportional scaling up to $0.35\text{m}$ per frame during transit.
  - **Sliding-Window ODBA (Omnidirectional Direct Bundle Adjustment)**: Joint local optimization over the recent $K$ keyframes (default $K=8$) using Levenberg-Marquardt on the spherical manifold bearing coplanarity constraint.
  - **Universal 3DGS Handoff Bridge**:
    - Converts TUM trajectory format (`timestamp tx ty tz qx qy qz qw`) to standard COLMAP `sparse/0/` format (`cameras.txt`, `images.txt`, `points3D.txt`).
    - Generates NeRF/3DGS `transforms.json` with `camera_model: EQUIRECTANGULAR`.
    - Generates OpenSfM-compatible `opensfm/reconstruction.json` and `metadata.json` for instant pipeline interoperability.
- **Strict Engine Synchronization**: `360dvo_engine.py` and `dvo360_engine.py` maintain 100% byte-for-byte SHA256 parity.

---

### 2. Command-Line Interface (`align_cli.py`)
- **New 360DVO CLI Arguments**:
  - `--dvo_patches`: Number of active tracked patches per frame (64, 96, 128, 256; default: 96).
  - `--dvo_polar_mask`: Polar zenith/nadir cutoff fraction (0.0 to 0.20; default: 0.08 = 8%).
  - `--dvo_ba_window`: Local ODBA sliding-window bundle adjustment size (default: 8 keyframes).
  - `--dvo_scale`: Metric scene depth scale multiplier (default: 3.2m).
  - `--dvo_viz` / `--no_dvo_viz`: Launch or disable real-time Viser browser visualizer.
  - `--dvo_weights`: Custom checkpoint path (`.pth`).
- **Engine Dispatch & Telemetry**:
  - Fully wired into `run_360dvo_reconstruction` with all arguments passed through.
  - Reconstruction report accurately logs active patch density, polar cutoff %, ODBA window, and metric scale multiplier.

---

### 3. GUI Layout & Informative Mouse-Hover Tooltips (`abu_align_gui.py`)
- **Redesigned 3-Column 360DVO Subpanel**:
  - **Column 1 (Patch Graph & Density)**:
    - Radiobuttons: `Fast Realtime (64 Patches)`, `Standard Odometry (96 Patches)`, `Dense High-Precision (128 Patches)`, `Ultra Dense (256 Patches)`.
  - **Column 2 (Factor Graph & ODBA Optimization)**:
    - Polar Mask Combobox (`0%`, `5%`, `8%`, `12%`, `15%`).
    - ODBA Window Combobox (`4`, `8`, `12`, `16` keyframes).
    - Metric Scale Combobox (`1.0`, `2.0`, `3.2`, `5.0` meters).
    - Checkbox: `Real-Time 3D Visualizer`.
  - **Column 3 (Universal 3DGS Bridge)**:
    - Clear specifications for TUM Odometry, COLMAP `sparse/0/`, `transforms.json` (EQUIRECTANGULAR), and OpenSfM bridge for immediate 100% GPU training in PFGS360 and LichtFeld Studio.
- **Comprehensive Tooltip Guidance on Mouse Hover**:
  - Every single radio button, label, combobox, and checkbox features a descriptive `ToolTip` clarifying mathematical roles, best practices, and defaults.

---

## Verification & Parity Results

1. **Automated Test Run (`scratch/test_frames`)**:
   - Processed 10 equirectangular 360° frames with 128 active patches, 8% polar mask, 8-frame ODBA window, and 3.2m metric scale.
   - Odometry tracking finished in 0.69s (0.069s/view).
   - Produced `360dvo_trajectory.txt`, `transforms.json`, `opensfm/reconstruction.json`, `metadata.json`, and COLMAP `sparse/0/` dataset.
2. **File Parity**:
   - `360dvo_engine.py` and `dvo360_engine.py` SHA256: `98F83ABBF148045476DDDB03C2D7F2C455A65B057052C9D80A87ECAD3B42F7A0` (100% match).
   - Root workspace and `Abu_Align/` mirrored files match with 100% SHA256 integrity.
3. **C# Launchers**:
   - Recompiled `Abu_Align_and_Train.exe` and `Abu_Align_and_Train_v0.0.3.1.exe` with base-10 odometer versioning.
