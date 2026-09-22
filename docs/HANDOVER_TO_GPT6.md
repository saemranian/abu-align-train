# Abu Align & Train — Comprehensive Project Guide & GPT-6 Handover Documentation

> **Target Audience**: OpenAI GPT-6 (Astra / Canvas / Work Mode) & Google Antigravity Co-Development Team  
> **Current Version**: `v0.0.3.4` (Strict Base-10 Odometer Versioning)  
> **Canonical Project Root**: `C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\abu align`  
> **Primary Interpreters**:  
> - Photogrammetry, Solvers, GUI & CLI: `C:\Users\USer\miniconda3\envs\vggt\python.exe`  
> - 3D Gaussian Splatting & PFGS360: `C:\Users\USer\miniconda3\envs\gs_clean\python.exe`

---

## 1. Executive Summary & Purpose

**Abu Align & Train** is an open-source, production-grade **Neural Photogrammetry and 3D Gaussian Splatting (3DGS) Training Suite**. It provides a unified graphical interface (GUI), command-line interface (CLI), and standalone C# executables designed to solve 3D camera poses and train photorealistic 3DGS models from:
1. **Standard Perspective Photos & Drone Imagery** (Pinhole, Brown distortion).
2. **360° Omnidirectional Equirectangular Panoramas & Videos** (Insta360, Ricoh Theta, GoPro Max) on the $S^2$ spherical manifold.

The software bridges 5 photogrammetry/SfM alignment engines with 4 real-time 3D Gaussian Splatting training engines through an automated one-click pipeline.

```
+-----------------------------------------------------------------------------------------------+
|                                      ABU ALIGN & TRAIN GUI                                    |
|                                     (abu_align_gui.py / .exe)                                 |
+---------------------------------------------------------------+-------------------------------+
                                                                |
                     [STAGE 1: ALIGNMENT & SFM]                 | [STAGE 2: 3DGS TRAINING]
                                                                |
+-------------------------------------------------------------+ | +---------------------------+
| 1. VGGT-Omega (Feedforward Deep Transformer SfM)             | | | 1. LichtFeld Studio (LFS) |
| 2. Niantic ACE0 (Visual Relocalization & Pose Optimization) | | |    (Vulkan Real-Time GUI) |
| 3. NVIDIA PyCuSFM (GPU-Accelerated Aliked + LightGlue)      | | | 2. Nerfstudio Splatfacto  |
| 4. Mapillary OpenSfM (Native Spherical S² Structure-from-M) | | |    (PyTorch SOTA 3DGS)    |
| 5. 360DVO (Direct Visual Odometry on S² Manifold)           | | | 3. NVIDIA 3DGRUT / 3DGUT  |
+-------------------------------------------------------------+ | |    (Ray-Traced Optics)    |
                                |                               | | 4. PFGS360 (CVPR 2026)    |
                                v                               | |    (Pose-Free 360° GS)    |
               +----------------------------------+             | +---------------------------+
               | Standardized 3DGS Dataset Output |             |               ^
               | - transforms.json (NeRF/3DGS)    |-------------+---------------+
               | - sparse/0/ (COLMAP format)      |      (Instant 100% GPU Handoff)
               | - opensfm/reconstruction.json    |
               +----------------------------------+
```

---

## 2. Directory Structure & Single Source of Truth

The repository has been consolidated into **ONE single canonical directory**. All nested duplicate folders (such as older `Abu_Align/` subdirectories) have been removed.

```
C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\abu align\
|-- abu_align_gui.py               # Main Tkinter Desktop Application (dark theme, hardware telemetry)
|-- align_cli.py                   # Unified CLI entry point for all 5 photogrammetry engines
|-- train_engine.py                # Unified 3DGS training runner (LichtFeld, Splatfacto, 3DGRUT, PFGS360)
|-- opensfm_engine.py              # Mapillary OpenSfM Spherical S² SfM Engine
|-- 360dvo_engine.py               # 360DVO Spherical Direct Visual Odometry Engine
|-- dvo360_engine.py               # Compatible alias module for 360DVO
|-- acezero_engine.py              # Niantic ACE0 Relocalization & Pose Engine adapter
|-- pycusfm_engine.py              # NVIDIA PyCuSFM (Aliked/LightGlue GPU SfM) adapter
|-- cache_manager.py               # Smart multi-tier caching (.abu_cache) for features and tracks
|-- video_extractor.py             # Automatic equirectangular video to frame extractor (ffmpeg/decord)
|-- panoramic_splitter.py          # Equirectangular to 8-view virtual pinhole cube rig splitter
|-- equirectangular_transcoder.py  # Equirectangular image slicing and optimization
|-- version_manager.py             # Automated base-10 version bumper, snapshotter, and C# compiler
|-- VERSION.txt                    # Single-line plain version string (e.g., 0.0.3.4)
|-- launcher.cs                    # C# launcher source code
|-- Abu_Align_and_Train.bat        # Windows Batch script launcher
|-- Abu_Align_and_Train.exe        # Compiled standalone Windows executable launcher (C# / .NET)
|
|-- versions/                      # Immutable historical version archive
|   |-- v0.0.0.2 ... v0.0.3.4/     # Complete historical snapshots of code at every release
|   \-- historical_binaries/       # Archived compiled executables from past versions
|
|-- docs/                          # Technical documentation & release walkthroughs
|   |-- WALKTHROUGH_v0.0.3.4.md    # Latest release notes and mathematical fixes
|   |-- WALKTHROUGH_v0.0.3.3.md    # OpenSfM Stage 3 optimization walkthrough
|   \-- ...
|
\-- logs/                          # Run diagnostic and telemetry logs (abu_align_*.log, abu_train_*.log)
```

> [!IMPORTANT]
> **NOTE ON SIBLING FOLDERS**:
> In `General Stuff 01`, there are sibling reference folders (`acezero`, `3dgrut`, `vggt-omega`). These house external neural network weights and submodules. The sibling folder `vggt/` is an old, retired clone frozen at `v0.0.1.8` and must **never** be edited. All active code lives strictly in `General Stuff 01/abu align`.

---

## 3. Environment & Runtime Specifications

### Conda Environment 1: `vggt` (Photogrammetry, GUI, CLI, OpenSfM, 360DVO)
- **Python Path**: `C:\Users\USer\miniconda3\envs\vggt\python.exe`
- **Key Libraries**:
  - Python 3.10
  - PyTorch 2.5.1 + CUDA 12.4
  - OpenCV 4.10 (`cv2`)
  - SciPy 1.13 (`scipy.optimize`)
  - PyCeres / Ceres Solver (Bundle adjustment)
  - Tkinter & PIL (Pillow)
  - Viser (Interactive 3D web viewer)

### Conda Environment 2: `gs_clean` (Nerfstudio, PFGS360, CUDA Splatting)
- **Python Path**: `C:\Users\USer\miniconda3\envs\gs_clean\python.exe`
- **Executables**:
  - `ns-train.exe`: `C:\Users\USer\miniconda3\envs\gs_clean\Scripts\ns-train.exe`
- **Key Libraries**:
  - Nerfstudio 1.1.4
  - `nerfstudio360` / `gsplat360` (PFGS360 CVPR 2026 model implementation)
  - `spherical_distortion` custom CUDA extensions (SCA-PE equirectangular splatting)
  - UniK3D / ZoeDepth / MiDaS monocular depth backends

---

## 4. Architecture of the 5 Alignment Engines

### 1. Mapillary OpenSfM Spherical Engine (`opensfm_engine.py`)
- **Target**: 360° Equirectangular video frames and panoramic tours.
- **Geometry**: Direct spherical projection on the unit sphere $S^2$:
  $$\mathbf{b} = \begin{pmatrix} \cos(\phi)\sin(\theta) \\ -\sin(\phi) \\ \cos(\phi)\cos(\theta) \end{pmatrix}$$
  where $\theta = (u/W - 0.5) \cdot 2\pi$ and $\phi = (0.5 - v/H) \cdot \pi$.
- **5-Stage Pipeline**:
  - **Stage 1**: Spherical SIFT feature extraction with equirectangular solid-angle area compensation.
  - **Stage 2**: Essential Matrix epipolar matching on $S^2$ with full 3D Sampson tangent error denominator:
    $$\text{den} = \|E \mathbf{b}_1\|_2^2 + \|E^T \mathbf{b}_2\|_2^2 + 10^{-9}$$
  - **Stage 3**: $O(M)$ multi-view track synthesis via Disjoint-Set Union-Find.
  - **Stage 4**: Incremental camera resection using Cauchy M-estimator (`f_scale=0.03`) with a physical motion continuity gate ($\le 1.2\text{m}$ max jump) to prevent camera teleportation into the ceiling.
  - **Stage 5**: Windowed Spherical Bundle Adjustment optimizing 16-camera overlapping blocks.

### 2. 360DVO Engine (`360dvo_engine.py` / `dvo360_engine.py`)
- **Target**: High-speed equirectangular direct visual odometry (IEEE RA-L 2026).
- **Core Mechanism**: Spherical patch-based direct photometric alignment using pyramid Lucas-Kanade on $S^2$.
- **Stabilization**:
  - Velocity vector sign consistency check: ensures translation direction $\mathbf{t}_{\text{dir}}$ cannot flip $180^\circ$ backwards during low-parallax motion.
  - 3-frame trajectory moving-average smoother to eliminate high-frequency camera jitter.
  - Tight triangulation bounds: $1.5^\circ \le \theta \le 65^\circ$, depth $0.2\text{m} \le d \le 35\text{m}$, chordal error $\le 0.08 \times \min(l_1, l_2)$.

### 3. VGGT-Omega Engine
- **Target**: Feedforward deep transformer-based photogrammetry for uncalibrated perspective photos and drones.
- **Mechanism**: Large vision geometry foundation model predicting camera intrinsic matrix $K$, extrinsic poses $(R, t)$, and dense 3D point cloud depths in a single GPU pass.

### 4. Niantic ACE0 Engine (`acezero_engine.py`)
- **Target**: Visual relocalization and camera pose regression.
- **Mechanism**: Trains a lightweight scene-specific coordinate regression MLP, optimizing 6-DoF camera poses via DSAC* (Differentiable Sample Consensus).

### 5. NVIDIA PyCuSFM Engine (`pycusfm_engine.py`)
- **Target**: High-precision perspective SfM.
- **Mechanism**: GPU-accelerated ALIKED local feature extraction combined with LightGlue neural matching.

---

## 5. Architecture of the 4 Training Engines (`train_engine.py`)

1. **LichtFeld Studio (LFS)**:
   - Vulkan-based real-time 3D Gaussian Splatting engine with live visualizer GUI.
   - Supports Gaussian Unscented Transform (GUT) for non-linear optical distortion and `.spz` compression.
2. **Nerfstudio Splatfacto**:
   - PyTorch-based state-of-the-art 3DGS implementation with bilateral grid color correction and anti-aliased mip-filtering.
3. **NVIDIA 3DGRUT / 3DGUT**:
   - Ray-traced Gaussian representation with OptiX/CUDA BVH backends.
4. **PFGS360 (CVPR 2026)**:
   - Pose-Free Omnidirectional Gaussian Splatting on 360° videos.
   - Supports **two operational modes**:
     - **Mode A (OpenSfM / 360DVO Handoff - 100% GPU)**: Reads pre-aligned poses from `opensfm/reconstruction.json` or `transforms.json` for instant RTX 4090 training.
     - **Mode B (Pose-Free Direct)**: Computes camera trajectories from scratch on CPU/GPU using UniK3D monocular depth priors with a sliding match window (`match_views=8`).

---

## 6. Co-Development Protocol: Best Practices for GPT-6 & Antigravity

When **GPT-6** and **Antigravity** are pair-programming on this repository, both agents must adhere to the following rules to guarantee stability, prevent merge conflicts, and avoid code regressions.

### Rule 1: Single Source of Truth
- **NEVER create duplicate copies** in subdirectories (do not create `Abu_Align/` or `vggt/` folders).
- All edits must happen directly on the canonical files in:
  `C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\abu align\`

### Rule 2: Recommended Division of Labor
To avoid editing the same lines of code at the same time, divide responsibilities by subsystem:
- **GPT-6 Astra Lanes**:
  - **GUI Enhancements (`abu_align_gui.py`)**: UI layouts, ttk style enhancements, tooltips, dialogs, presets manager, user input validation, telemetry graphs.
  - **CLI Features (`align_cli.py`)**: Argument parsing, formatting of reports, dataset validation scripts.
  - **Post-Processing & Exporters**: RealityCapture CSV export, COLMAP text/bin converters, WebGL/Viser export adapters.
- **Antigravity Lanes**:
  - **Math & Photogrammetry Kernels (`opensfm_engine.py`, `360dvo_engine.py`)**: Epipolar geometry, Sampson error on $S^2$, Bundle Adjustment, Lie algebra rotations.
  - **CUDA & Machine Learning Integration (`train_engine.py`)**: Nerfstudio dataparser plumbing, PyTorch CUDA extensions, memory management, process pipe streaming.
- **If Working Sequentially on the Same File**:
  - Always verify existing lines before proposing changes.
  - Check `git diff` to ensure no unintended overwrites occur.

### Rule 3: Strict Base-10 Odometer Versioning
Version format is strictly `vMAJOR.MINOR.PATCH.BUILD` (e.g., `v0.0.3.4`).
- Each digit is strictly `[0..9]`.
- When incrementing, if the rightmost digit hits 9, the next increment rolls over to 0 and carries +1 to the left:
  $$\text{v0.0.3.4} \to \text{v0.0.3.5} \dots \to \text{v0.0.3.9} \to \text{v0.0.4.0}$$
- **To bump version**: Run the version manager tool:
  ```powershell
  python version_manager.py --bump
  ```
  This automatically:
  1. Increments the version string according to odometer rules.
  2. Updates `VERSION.txt`, GUI badges, CLI headers, batch files, and launcher code.
  3. Recompiles `Abu_Align_and_Train.exe` using `csc.exe`.
  4. Creates an immutable snapshot in `versions/vX.Y.Z.W/`.
  5. Archives the versioned binary in `versions/historical_binaries/`.

### Rule 4: Mandatory Syntax & Lint Verification
Before marking any task complete, both agents must run:
```powershell
# In Conda vggt environment:
C:\Users\USer\miniconda3\envs\vggt\python.exe -c "import py_compile; py_compile.compile('abu_align_gui.py', doraise=True); py_compile.compile('align_cli.py', doraise=True); py_compile.compile('train_engine.py', doraise=True); py_compile.compile('opensfm_engine.py', doraise=True); py_compile.compile('360dvo_engine.py', doraise=True); print('All core modules compiled with zero syntax errors!')"
```

### Rule 5: Windows Path & Process Hygiene
- Always use `PYTHONUNBUFFERED=1` and `PYTHONIOENCODING=utf-8` when spawning subprocesses.
- Wrap Windows paths in quotes or `raw strings` (`r"C:\Users\..."`) to avoid backslash escaping issues.
- Preserve all existing docstrings, comments, and telemetry callbacks.

---

## 7. Immediate Task Queue (Ready for GPT-6)

Here are the highest-priority tasks that GPT-6 can immediately undertake:

1. **Preset System in GUI (`abu_align_gui.py`)**:
   - Add a dropdown to save and load user presets (e.g., "360 Video Tour - Fast", "Drone Mapping - High Accuracy", "Indoor Container Scan - OpenSfM + Splatfacto").
   - Store presets as clean JSON files in a `presets/` directory.
2. **Export to RealityCapture & Postshot**:
   - Enhance the output bundle in `results/sparse_XX/` with a standard `cameras.csv` matching RealityCapture's exact column headers (`#Image name, X, Y, Z, Heading, Pitch, Roll`).
3. **Automated End-to-End Test Harness**:
   - Create a lightweight test script `tests/test_pipeline_quick.py` that runs 5 frames through OpenSfM and verifies that `transforms.json` and `points3D.txt` are created properly.
4. **Enhanced Telemetry Progress Chart**:
   - In `abu_align_gui.py`, add a mini canvas visualizer showing the live estimated camera path (2D top-down trajectory scatter plot) as alignment progresses.

---

## 8. Quick Command Cheat Sheet for GPT-6

```powershell
# 1. Check current version:
python version_manager.py --current

# 2. Test CLI alignment:
python align_cli.py --input "path/to/equirectangular/images" --engine opensfm --stride 1 --opensfm_feature sift --opensfm_matcher words

# 3. Test Standalone 3DGS Training:
python train_engine.py --dataset "path/to/results/sparse_14_opensfm/3dgs_dataset" --engine pfgs360 --iterations 7000

# 4. Bump version and build executables:
python version_manager.py --bump

# 5. Check Git Status:
git status
```
