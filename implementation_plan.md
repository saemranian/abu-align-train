# Implementation Plan: Dynamic Engine-Specific Parameter Panels & Repository Alignment (v0.0.1.4)

Make the configuration options in **Abu Align** and **Abu Train** accurately match their respective official repositories. When the user changes an engine, the options must dynamically switch to show only the precise parameters native to that engine, complete with interactive hover tooltips and backend CLI wiring. The version will be updated to **`v0.0.1.4`** with automated Git commit logging.

---

## User Review Required

> [!IMPORTANT]
> **Dynamic Parameter Panels Behavior:**
> Currently, the GUI presents static parameters (e.g. VGGT's point cloud density and bundle adjustment) regardless of whether ACE0, PyCuSFM, or VGGT is active. With this update:
> - Switching **Alignment Engine** will dynamically morph the options area into the exact controls native to that engine (ACE0 mapping rounds/refinement/seeds; PyCuSFM GPU features/matchers/cuDSS iterations; VGGT confidence/density/Ceres BA/distortion).
> - Switching **Training Engine** will dynamically morph the training options into the exact controls native to that engine (LichtFeld Vulkan GUI/GUT/SPZ; Nerfstudio Splatfacto model variant/camera optimizer/bilateral grid/mip-filtering; NVIDIA 3DGRUT 3DGUT vs 3DGRUT/OptiX BVH/ADMM sparsity).

---

## Proposed Changes

### 1. Research & Parameter Mapping

#### A. Alignment Engines (Abu Align)

| Engine | Official Repository | Precise Native Parameters | GUI Controls & Tooltip Guidance |
| :--- | :--- | :--- | :--- |
| **VGGT & VGGT-Omega** | `facebookresearch/vggt` & VGGT-Omega | • Canvas Res: `512, 1024, 1536, 2048`<br>• Stride: `1, 2, 3`<br>• Alignment Acc.: `Feedforward, Standard Ceres BA, Ultra Sub-Pixel BA`<br>• Lens Model: `Pinhole, Brown (k1-k2, t1-t2)`<br>• Point Density: `Preview (100k), High (300k), Ultra (800k), Max (1.5M)`<br>• VRAM Guard: `Chunk size & Anchor size` | Radio buttons & dropdowns with tooltips explaining neural transformer token limits, Cauchy loss, and self-calibrating radial/tangential distortion. |
| **ACE0** | `nianticlabs/acezero` (ECCV 2024 Oral) | • Mapping Rounds: `15 (Quick), 30 (Standard), 50 (Deep)`<br>• Pose Refinement: `MLP ('mlp'), Naive PnP ('naive'), None ('none')`<br>• Initialization Seeds: `1, 3, 5`<br>• Calibration: `Refine Focal Length vs Fixed 70% heuristic`<br>• Point Cloud Filter: `Filtered Reprojection vs Dense Unfiltered` | Radio buttons & checkbuttons tailored to Scene Coordinate Regression, ZoeDepth seeds, and differentiable MLP backprop. |
| **NVIDIA PyCuSFM** | `nvidia-isaac/pyCuSFM` (Isaac GPU SfM) | • GPU Feature Extractor: `ALIKED, SuperPoint, SIFT`<br>• Matcher: `LightGlue (GPU Transformer), SuperGlue`<br>• Pose Initializer: `VGGT-Omega Coarse Prior (Hybrid) vs Sequential`<br>• cuDSS Bundle Adjustment: `50, 100, 200 Iterations`<br>• Cloud Colorization: `RGB Colorized Tie Points vs Geometric` | Radio buttons & dropdowns for GPU-accelerated sparse linear solver (cuDSS), neural descriptor extraction, and coarse-to-fine hybrid initialization. |

#### B. Training Engines (Abu Train)

| Engine | Official Repository | Precise Native Parameters | GUI Controls & Tooltip Guidance |
| :--- | :--- | :--- | :--- |
| **LichtFeld Studio** | `LichtFeld-Studio` | • Viewport Mode: `Interactive Vulkan Live 3D GUI vs Headless CLI`<br>• Optics Model: `Standard Rectilinear 3DGS vs GUT (Gaussian Unscented Transform)`<br>• SH Degree: `SH 0 (Diffuse), SH 1, SH 2, SH 3 (Full SOTA Specularity)`<br>• Export Format: `Standard .ply vs Compressed .spz (10x smaller)`<br>• Iterations: `7,000, 15,000, 30,000` | Real-time Vulkan window toggle, Unscented Transform optics, spherical harmonics specularity, and Scaniverse `.spz` web compression. |
| **Nerfstudio Splatfacto** | `nerfstudio-project/nerfstudio` | • Model Variant: `splatfacto (Standard) vs splatfacto-big (High Detail)`<br>• Camera Optimizer: `None, SO3xR3 (Pose Refinement), SE3 (Lie Group)`<br>• Bilateral Grid: `Learned 3D color grid for exposure/WB differences`<br>• Anti-Aliased Mip-Filtering: `Mip-Splatting for multi-scale view`<br>• SH Degree: `0, 1, 2, 3, 4`<br>• Iterations: `7,000, 15,000, 30,000` | Model scale dropdown, camera optimization mode, Bilateral appearance grid, and anti-aliasing mip-filtering. |
| **NVIDIA 3DGRUT / 3DGUT** | `nv-tlabs/3dgrut` | • Representation: `3DGUT (Unscented Transform) vs 3DGRUT (Ray Tracing)`<br>• BVH Backend: `OptiX Hardware RT Cores vs CUDA Linear BVH`<br>• Appearance: `Affine Exposure / Lighting Correction`<br>• ADMM Sparsity: `Active (λ=0.001) for ~60% file compression`<br>• SH Degree: `0, 1, 2, 3`<br>• Iterations: `7,000, 15,000, 30,000` | Ray-tracing architecture selection, hardware OptiX BVH traversal, per-photo affine appearance, and ADMM sparsity pruning. |

---

### 2. GUI Dynamic Panel Architecture (`abu_align_gui.py`)

#### [MODIFY] [abu_align_gui.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/abu_align_gui.py)
- Refactor the middle/right portions of `Alignment Configuration` into a dynamic switcher frame `self.align_dynamic_container`:
  - Hook `self.engine_var.trace_add("write", self._on_align_engine_changed)`
  - Methods:
    - `_build_vggt_align_panel(parent)`
    - `_build_ace0_align_panel(parent)`
    - `_build_pycusfm_align_panel(parent)`
- Refactor the right portion of `3D Gaussian Splatting Training Suite` into a dynamic switcher frame `self.train_dynamic_container`:
  - Hook `self.train_engine_var.trace_add("write", self._on_train_engine_changed)`
  - Methods:
    - `_build_lichtfeld_train_panel(parent)`
    - `_build_nerfstudio_train_panel(parent)`
    - `_build_3dgrut_train_panel(parent)`
- Attach detailed `ToolTip` instances to every single radio button, checkbutton, and combobox.
- Update parameter gathering in `_start_alignment` and `_start_train_only` to pass the active engine's specific flags to `align_cli.py` and `train_engine.py`.

---

### 3. CLI & Engine Adapter Synchronization

#### [MODIFY] [align_cli.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/align_cli.py)
- Ensure all arguments for ACE0 (`--ace_iterations`, `--ace_refinement`, `--ace_try_seeds`, `--ace_refine_calib`, `--ace_dense_pc`) and PyCuSFM (`--cusfm_features`, `--cusfm_matcher`, `--cusfm_init`, `--cusfm_ba_iterations`, `--cusfm_rgb`) are parsed and routed cleanly to `acezero_engine.py` and `pycusfm_engine.py`.
- Update final `reconstruction_report.txt` generation to format engine-specific parameter sections accurately.

#### [MODIFY] [train_engine.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/train_engine.py)
- Ensure all engine-specific flags are accepted and passed:
  - LichtFeld: `--headless`, `--gut`, `--sh_degree`, `--export_spz`
  - Nerfstudio: `--ns_model` (`splatfacto`/`splatfacto-big`), `--camera_optimizer`, `--bilateral_grid`, `--enable_mip`, `--sh_degree`
  - 3DGRUT: `--representation` (`3dgut`/`3dgrut`), `--bvh_backend` (`optix`/`cuda`), `--exposure_correction`, `--enable_sparsity`, `--sh_degree`

---

### 4. Automated Version Management (`v0.0.1.4`) & Git Commit

#### [EXECUTE] [version_manager.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/version_manager.py)
- Execute `version_manager.py --set 0.0.1.4` (or `--bump`) to synchronize version `v0.0.1.4` across:
  - `VERSION.txt`
  - GUI headers, window title, badges, About box
  - CLI banner & headers
  - Training engine banner
  - Batch files
  - Recompiles `Abu_Align_and_Train_v0.0.1.4.exe` and `Abu_Align_and_Train.exe` with `csc.exe`
- Git commit on `main` logging the complete release:
  `feat(v0.0.1.4): add dynamic engine-specific parameter panels matching official repositories with interactive tooltips`

---

## Verification Plan

### Automated Tests
1. **Python Compilation & Syntax Test:**
   ```powershell
   C:\Users\USer\miniconda3\envs\vggt\python.exe -c "import py_compile; py_compile.compile('Abu_Align/abu_align_gui.py', doraise=True); py_compile.compile('Abu_Align/align_cli.py', doraise=True); py_compile.compile('Abu_Align/train_engine.py', doraise=True); print('Syntax Verified')"
   ```
2. **GUI Dynamic Panel Switching Verification:**
   Script tests instantiating Tkinter, switching `engine_var` between `vggt_omega`, `ace0`, and `pycusfm`, and verifying that the dynamic alignment panel displays the correct subwidgets. Then switches `train_engine_var` between `lichtfeld`, `nerfstudio`, and `3dgrut` and verifies that the dynamic training panel displays the correct subwidgets.
3. **Version Manager & Standalone Executables Check:**
   Verify `version_manager.py` increments to `v0.0.1.4` and builds `Abu_Align_and_Train_v0.0.1.4.exe`.
4. **Git Commit Hygiene:**
   Verify `git status` is clean with all changes committed.
