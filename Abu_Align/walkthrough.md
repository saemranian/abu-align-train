# Abu Align & Train v0.1.4.3: Redesign the manual 360 control-point workspace

Replaced the basic toolbar/two-canvas prototype with a dedicated professional workspace: searchable/filterable point list and distance panel, independently navigable panorama panes, editable point/survey properties, per-image measurement lists and uncertainties, configurable refinement budgets, and a visible convergence/check-point result report. Added a full Help tab and delayed hover guides for settings, tools and navigation. The compatibility entry point keeps the existing Prepare input menu working. Styles are scoped to this tool so the main application's theme is preserved.

Interaction improvements: background image decode with generation checks and cancellation of queued stale requests; original image dimensions verified before accepting measurements; cursor-anchored wheel zoom, middle/Space drag panning, Fit and 1:1, focus measurement, optional linked scale, panorama seam wrap, draggable markers, 1/0.1/10 pixel keyboard nudges, and readable selected/hover point labels. Viewport-sized mip-level crops replace full enlarged bitmaps; render requests are coalesced. Dashed projections are unconfirmed guidance and never become automatic measurements. Disabled controls and their attached distances remain on disk but are excluded from refinement. Undo/redo covers point, measurement, property and distance edits; renames update distance endpoints. Dirty-file guards, finite-value validation, atomic JSON writes, source/output collision checks and immutable refinement snapshots reduce accidental data loss. Scrollable properties/settings and pinned status/count controls remain reachable at the minimum window size.

Verification: 20 focused tests passed in 7.790 s. The suite includes the main application's actual Prepare-menu callback, parent-theme preservation, asynchronous stale-image handling, tile-size bounds, coordinate invariants, marker nudging, undo/redo, property/distance updates, dirty close cancellation, Help/hover lifecycle and disabled-control solver equivalence. New/modified Python sources compile. Visual review used procedural synthetic images at 1580x940 and 1100x720; no private panorama screenshots were transmitted. Local runtime check used two authorized 7680x3840 panoramas: both loaded in 0.692 s, image/mip memory 235699200 bytes, measured process RSS 395419648 bytes (about 377 MiB). Sixty 800x600 viewport renders had median 8.362 ms and measured P95 sample 17.624 ms; this excludes display/event-loop latency and does not establish guaranteed FPS. The editor uses CPU image memory and invokes no CUDA allocation or training run. User images were not uploaded or changed.

Files: control_points_ui.py compatibility entry; new control_points_workspace.py, control_points_model.py and panorama_viewport.py; spherical_refinement.py disabled-control handling; new tests/test_control_points_workspace.py and updates to desktop/refinement tests; docs/360_CONTROL_POINTS_USER_GUIDE.md, docs/SPHERICAL_CONTROLS.md and docs/NATIVE_360_IMPLEMENTATION.md. Version headers, both C# launchers, walkthroughs, active-file mirrors and snapshots are synchronized by the release helper. This is an improved native spherical UI, not a claim of full RealityScan functionality; geographic conversion, component merging and automatic matching remain separate features.

---

# Abu Align & Train v0.1.4.2: Retain neural spherical DVO tracks for manual and joint refinement

Root cause: 360DVO exported poses and a display cloud but discarded neural image measurements, leaving global spherical BA unable to constrain that cloud. The tracked runner now preserves measured targets before sliding-window BA, bounded to 500000 observations. Stable frame/patch identities survive keyframe removal. Confidence/polar/finite/latitude filtering, longitude wrap and final-trajectory triangulation provide original-resolution observations in the native OpenSfM and COLMAP handoffs. Loose display points remain in their PLY, while the BA handoff contains tracked landmarks. Cache identity includes the track helper. The main CLI and desktop fail when neural tracking cannot initialize unless the experimental fallback is explicitly requested.

Verification: 14 focused memory/input/geometry/manifest/track tests pass, including strict-backend rejection, stable identity and bounded storage, seam wrapping and exact final-pose triangulation. Local neural GPU smoke on 16 authorized panoramas at width 1024 and keyframe threshold 1 completed in 4.83 s, peak allocated VRAM 727599104 bytes (0.68 GiB). It captured 8173 observations with no truncation, accepted 1006 landmarks/8026 observations, and exported both tracked native handoffs. Joint spherical BA on 1000 points/7965 observations reduced fitted RMS from 0.45712 to 0.44230 degrees in 19.77 s. Its 20-evaluation budget did not converge; this result is not an independent accuracy benchmark. The separate width-512 test did not initialize neural tracking and is explicitly a failed neural check, not a successful GPU result.

No images were uploaded. GPU allocation belongs to neural DVO; joint refinement remains sparse CPU SciPy. User originals are untouched. Georeferenced accuracy, component merging, direct GCP-constrained training and long-run rendering benchmarks remain outside the validated claims. Launchers, active mirrors, walkthroughs and version snapshots are synchronized by the release helper.

---

# Abu Align & Train v0.1.4.1: Bound spherical training memory and integrate native COLMAP candidate

Root causes: eager full-dataset image/depth tensors, unused parser downscale flags, overlapping evaluation views, unbounded initialization/densification, stale output discovery, and export that reloaded the GPU pipeline. The adapter streams per view with a bounded 256 MiB image cache and disk depth cache. Derived images apply resolution to pixels; camera dimensions and stored landmark pixels follow that resize. Originals remain unchanged. Every run has a unique output folder, with validated Gaussian export on CPU. Fixed, refine and pose-free modes are explicit; fixed camera parameters are verified unchanged after each iteration. Native COLMAP preserves spherical cameras and observations; the GUI and main CLI expose it as a candidate. Its installed backend is CPU-only and spherical guided SIFT matching is deliberately disabled following a reproduced native abort. OpenSfM retrieval handles small datasets without requesting an extra neighbor.

Verification: 28 focused tests passed; all root Python sources compile. Integrated native COLMAP CLI registered all 48 authorized panoramas in one component and exported standard packages in 41.4 s (feature width 1024, budget 3000). It retained 5866 points and 30600 observations, median fitted angular residual 0.05261 degrees. Integrated PFGS360 wrapper completed five iterations on 43 training/five held-out native ERP views at 480x240, capped at 8192 Gaussians, with depth/sky masking and verified PLY export. Fixed camera drift was exactly zero. Worker runtime 47.62 s, peak RAM 4.16 GiB, peak allocated/reserved VRAM 4.42/4.51 GiB. CPU export avoids a second GPU/dataset load. The local UniK3D runtime required timm 1.0.22 because 0.6.7 lacks timm.layers; the focused dependency is recorded in docs/PFGS_RUNTIME_REQUIREMENTS.txt.

These are runtime/integrity checks, not a long-run image-quality or absolute-accuracy benchmark. No user images were uploaded. Full 8K training, published benchmark reproduction, GPU spherical BA, direct GCP-constrained training, component merging and persistent DVO observations are not yet claimed. See docs/NATIVE_360_IMPLEMENTATION.md for the current gates.

---

# Abu Align & Train v0.1.4.0: Native spherical manual controls, joint refinement and evaluation

Added a two-pane panorama control editor to Prepare, immutable refinement outputs, sparse joint camera/landmark BA, local XYZ GCP gauge initialization, distance constraints, independent withheld check points, and seam-aware bearing residuals. Preserved/retriangulated tracks outside the automatic optimization budget. Added CLI refinement and independent named-camera ATE/RPE/registration evaluation plus solid-angle PSNR. OpenSfM worker count is bounded and fallback retrieval proposes interior revisits; geometric verification still rejects bad candidates. Verification: 23 integrity/UI tests passed; synthetic joint BA reduced ATE >5x and angular RMS >10x; local GCP/check test below 1cm. Refinement is sparse CPU; GPU acceleration and real accuracy comparisons are not claimed. UI zoom renders viewport tiles to bound preview memory. No source photos or original reconstructions are overwritten.

---

# Abu Align & Train v0.1.3.9: Reproducible panorama inputs and neural polar masking

Added ordered SHA256 image manifests and cache identity for all selected images, keyframe thresholds, polar masks, runner code and explicit weights. 360DVO now executes the version-controlled runner directly, consumes exact selected frames, rejects corrupt/incomplete/uninitialized sequences, masks polar patch weights in neural BA, and records engine/GPU provenance. Fixed WSL PATH variable expansion and unbuffered output. Verification: 11 targeted tests passed; new WSL runner completed 16/16 frames with polar masking and trajectory/cloud export. Retained safe <=1024 width until larger CUDA indexing is benchmarked. No whole-route accuracy claim. GPU smoke allocation fits RTX4090; exact per-run VRAM now recorded by runner. CPU fallback is explicitly labeled experimental.

---

# Walkthrough: Abu Align & Train v0.1.3.8
## Resolution of OpenSfM Camera Trajectory & Point Cloud Axis Misalignment

### 1. Executive Summary
Release **v0.1.3.8** fixes the 90-degree coordinate misalignment between OpenSfM camera trajectories and point clouds in the 3D Workspace and across all downstream exporters (COLMAP `images.txt`, Nerfstudio `transforms.json`, and RealityCapture `cameras.csv`).

---

### 2. Root Cause Analysis & Technical Resolution

#### The Problem
In the 3D Workspace for OpenSfM runs, the camera ribbon (green wireframe tunnel) was standing upright/perpendicular (oriented along the Y axis), while the sparse point cloud tie points were spread horizontally (oriented along the X axis).

#### Root Cause
1. PyCOLMAP's C++ bindings (`pycolmap.Rotation3d`) wrap Eigen's `Quaterniond`, which expects quaternion coefficients in scalar-last `[x, y, z, w]` memory order.
2. In `opensfm_engine.py`, `acezero_engine.py`, and `pycusfm_engine.py`, `pycolmap.Rotation3d` was being constructed with `[qw, qx, qy, qz]`. Eigen interpreted `qw` as `x`, `qx` as `y`, `qy` as `z`, and `qz` as `w`, creating an incorrect rotation matrix rotated 90 degrees around an oblique axis.
3. When downstream exporters (`reconstruction.write_text()`, `export_transforms_json()`, `export_realitycapture_csv()`) calculated camera world positions via $C = -R_{w2c}^T t$, the scrambled rotation caused camera X positions ($X \in [-27.8, +21.6]$) to map into Y ($Y \in [-27.7, +21.6]$). Meanwhile, `points3D.ply` (loaded directly from OpenSfM coordinates) remained oriented along X ($X \in [-93.6, +331.4]$).
4. When `fast_viewer.py` / `workspace_3d.py` loaded the scene, the cameras moved along Y while the points spread along X, causing the perpendicular misalignment visible in the screenshot.

#### The Fix
1. **Direct SO(3) Matrix Construction**: Updated `opensfm_to_pycolmap()` and `pycusfm_engine.py` to construct `pycolmap.Rotation3d(R_w2c)` directly from the $3 \times 3$ rotation matrix, bypassing quaternion order conventions completely.
2. **Robust Quaternion Conversion**: Updated `acezero_engine.py` and manual text exporters to strictly use `quaternion_xyzw()` / `SciPy.as_matrix()`.
3. **Automatic Dataset Re-Export**: Re-exported all camera poses and transforms in `results/sparse_04_opensfm` and `results/sparse_03_opensfm`. Both cameras and point clouds now strictly share the exact same coordinate system and orientation.

---

### 3. Verification

- Verified camera center bounds in `sparse_04_opensfm`:
  - Points X range: `[-93.65, 331.42]`, Y range: `[-66.64, 96.32]`, Z range: `[-411.35, 98.17]`
  - Cameras X range: `[-27.78, 21.62]`, Y range: `[-2.88, 2.29]`, Z range: `[-3.65, -0.02]`
  - Trajectory vector: $\Delta X = -49.4\text{m}$, aligned with point cloud corridor.
- Python syntax compilation across all modified modules — **PASSED (Exit Code 0)**
