# Abu Align & Train Suite — Master Architecture & Co-Development Handover Protocol

> **Target Audience**: OpenAI GPT-6 (Astra / Canvas / Work Mode) & Google Antigravity Co-Development Team  
> **Current Version**: `v0.1.4.9` (Strict Base-10 Odometer Versioning; VERSION.txt is authoritative)
> **Canonical Project Root**: `C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\abu align`  
> **Documentation Location**: `C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\abu align\docs\HANDOVER_TO_GPT6.md`  
> **Primary Environments**:  
> - Photogrammetry, CV, SAM2, GUI & CLI: `C:\Users\USer\miniconda3\envs\vggt\python.exe`  
> - 3D Gaussian Splatting & PFGS360: `C:\Users\USer\miniconda3\envs\gs_clean\python.exe`  
> - Native Linux GPU Visual Odometry: WSL2 Ubuntu (`wsl -d Ubuntu`, RTX 4090 GPU accelerated)

---

## 1. Executive Summary & Suite Vision

### Shared viewer update — Abu 3D Viewer M54

The shared viewer is named **Abu 3D Viewer M54**, shortened to **Abu M54** while it
is developed toward future Abu 3D tool combinations. Main Align/Train and manual
360 controls use `abu_m54_native.M54NativeViewer`; `abu_m54_browser` supplies shared
browser navigation, picking, grid and transformation controls. The standalone
`Abu_M54.exe` / `abu_m54_app.py` opens reconstruction JSON, PLY, camera CSV, COLMAP
models, trajectories and related supported result formats, including Windows Open
with registration. Scene/file/transform math lives in `abu_m54.py` with float64
world coordinates and normalized GPU displays. Do not create another viewer fork.
Read [the M54 architecture and user guide](ABU_M54.md) and
[its verification record](ABU_M54_VERIFICATION.md) before changing viewer behavior.
Sources remain immutable; model transformations are preview operations until an
explicit full-source copy export. No client images/data are uploaded.

**Abu Align & Train** is an end-to-end, production-grade **Neural Photogrammetry, Preprocessing, and 3D Gaussian Splatting (3DGS) Training Suite**. Originally started as an alignment bridge, it has evolved into a comprehensive suite of high-performance tools engineered for:

1. **Standard Perspective Imagery & Drone Surveys** (Pinhole, Brown-Conrady, Fisheye).
2. **360° Omnidirectional Equirectangular Video & Panoramas** (Insta360, Ricoh Theta, GoPro Max) on the $S^2$ spherical Riemannian manifold.
3. **Rigid Multi-Camera Rig Splitting & Photogrammetry Export** (Cubemap virtual pinhole cameras with locked $90^\circ$ rotations and exact RealityCapture / RealityScan XMP metadata).
4. **Preprocessing & Quality Control** (AI zero-shot segmentation masking, SOTA sharpness/blur filtering, high-speed frame picking, and batch renaming).
5. **Real-Time Interactive 3D Inspection** (Viser WebGL viewer with robust percentile bounding, trajectory-proportional camera frustums, and station nodes).

```
+-------------------------------------------------------------------------------------------------------+
|                                          ABU DESKTOP SUITE                                            |
|                       (desktop_ui.py / abu_align_gui.py / Abu_Align_and_Train.exe)                     |
+-------------------------------------------------------------------------------------------------------+
                                                     |
    +------------------------------------------------+-----------------------------------------------+
    | PREPROCESSING & UTILITY SUITE ("ABU TOOLS")    | CORE PHOTOGRAMMETRY & TRAINING PIPELINE       |
    +------------------------------------------------+-----------------------------------------------+
    | 1. Frame Picker (Light-speed stride selector)  | [STAGE 1: ALIGNMENT & SFM]                    |
    | 2. Blur Drop (SOTA sharpness filter & reject)  | - 360DVO (Native WSL2 GPU Direct Odometry)    |
    | 3. SAM 2 Masker (Zero-shot AI nadir/rig mask)  | - Mapillary OpenSfM (Spherical S² SfM)        |
    | 4. 360 Splitter (Cubemap pinhole rig + XMP)    | - VGGT-Omega / VGG-T3 (Feedforward Deep SfM)  |
    | 5. Video Extractor (GPU frame extraction)      | - Niantic ACE0 (Coordinate Regression MLP)    |
    | 6. Batch Renamer & File Sift (Regex triage)    | - NVIDIA PyCuSFM (ALIKED + LightGlue SfM)     |
    | 7. Rig Designer & Viewport (Multi-cam rigs)    | - RealityScan Engine (Epic RS Rig bridge)     |
    +------------------------------------------------+-----------------------------------------------+
                                                     |
                                                     v
                                    +----------------------------------+
                                    | Standardized Results Output      |
                                    | - COLMAP_Lichtfield/             |
                                    | - Postshot_RealityCapture/       |
                                    | - 3dgs_dataset/ (transforms.json)|
                                    | - points3D.ply (Dense tie cloud) |
                                    +----------------------------------+
                                                     |
                     +-------------------------------+-------------------------------+
                     |                                                               |
                     v                                                               v
      [STAGE 2: 3DGS TRAINING]                                        [STAGE 3: 3D VISUALIZATION]
      - LichtFeld Studio (Vulkan Real-Time GUI + GUT)                 - fast_viewer.py / viewer_workspace.py
      - Nerfstudio Splatfacto (PyTorch SOTA 3DGS)                     - Interactive Viser Browser View (8080)
      - NVIDIA 3DGRUT / 3DGUT (Ray-Traced Optics)                     - Trajectory-Scaled Frustums & Nodes
      - PFGS360 (CVPR 2026 Pose-Free 360° Splatting)                  - Multi-Run A/B Registration Inspection
```

---

## 2. Directory Structure & Single Source of Truth

All active code lives strictly in **ONE canonical directory**. Nested duplicate copies (`Abu_Align/`, legacy `vggt/`) have been removed.

```
C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\abu align\
|-- desktop_ui.py                  # Main Abu Suite Launcher (integrates all sub-tools)
|-- abu_align_gui.py               # Main Tkinter Desktop Application (dark theme, live hardware telemetry)
|-- align_cli.py                   # Unified CLI entry point for all alignment engines
|-- train_engine.py                # Unified 3DGS training runner (LichtFeld, Splatfacto, 3DGRUT, PFGS360)
|-- fast_viewer.py                 # Standalone interactive 3D Viser browser visualizer launcher
|-- viewer_workspace.py            # Viser workspace engine (bounds, frustums, node markers, A/B comparison)
|-- version_manager.py             # Base-10 odometer version manager, snapshotter, and C# compiler
|-- VERSION.txt                    # Single-line plain version string (current: 0.1.3.5)
|-- launcher.cs                    # C# launcher source code
|-- Abu_Align_and_Train.bat        # Windows Batch script launcher
|-- Abu_Align_and_Train.exe        # Compiled standalone Windows executable launcher (C# / .NET)
|
|-- [PHOTOGRAMMETRY & ALIGNMENT ENGINES]
|   |-- 360dvo_engine.py           # 360DVO Native GPU Odometry runner (WSL2 bridge integration)
|   |-- dvo360_engine.py           # Compatibility alias for 360DVO engine
|   |-- wsl_bridge.py              # Windows <-> WSL2 Ubuntu process, GPU, and path translator
|   |-- opensfm_engine.py          # Mapillary OpenSfM Spherical S² Structure-from-Motion
|   |-- vggttt_engine.py           # NVIDIA VGG-T3 / VGGT-Omega foundation feedforward alignment
|   |-- acezero_engine.py          # Niantic ACE0 visual relocalization & DSAC* regression
|   |-- pycusfm_engine.py          # NVIDIA PyCuSFM (ALIKED + LightGlue GPU SfM)
|   |-- realityscan_engine.py      # RealityScan camera alignment exporter and rig bridge
|
|-- [ABU UTILITY SUITE ("ABU TOOLS")]
|   |-- frame_picker_engine.py     # Frame Picker core engine (stride/interval selector, high speed)
|   |-- frame_picker_ui.py         # Frame Picker interactive Tkinter GUI with preview slider
|   |-- frame_picker_cli.py        # Frame Picker command-line interface
|   |-- blur_drop_engine.py        # Blur Drop core engine (Laplacian, Tenengrad, FFT, CPBD scoring)
|   |-- blur_drop_ui.py            # Blur Drop interactive GUI with zoomable/scrollable image grid
|   |-- blur_drop_cli.py           # Blur Drop command-line interface
|   |-- panoramic_splitter.py      # Equirectangular cubemap pinhole splitter with RealityScan rig XMP
|   |-- equirectangular_transcoder.py # Panorama slicing, scaling, and format transcoding
|   |-- sam2_engine.py             # Segment Anything 2 (SAM 2) zero-shot masking backend
|   |-- sam2_masker_ui.py          # SAM 2 interactive visual masking GUI (tripod/nadir removal)
|   |-- renamer_engine.py          # High-speed batch renamer (regex, padding, prefix/suffix)
|   |-- renamer_ui.py              # Batch renamer GUI with live preview
|   |-- renamer_cli.py             # Batch renamer command-line interface
|   |-- file_sift_engine.py        # File Sift rapid asset categorization engine
|   |-- file_sift_ui.py            # File Sift triage GUI
|   |-- file_sift_cli.py           # File Sift command-line interface
|   |-- rig_designer_ui.py         # Multi-camera rig calibration and visual modeling
|   |-- rig_viewport.py            # Interactive 3D viewport for camera rig orientation
|   |-- video_extractor.py         # High-speed GPU video frame extraction (ffmpeg / decord)
|   |-- cache_manager.py           # Multi-tier caching manager (.abu_cache)
|
|-- docs/                          # Technical guides, release walkthroughs, schemas
|   |-- HANDOVER_TO_GPT6.md        # THIS DOCUMENT (Master architecture & co-development protocol)
|   |-- True_Schema_360_Camera_Rig.xmp # Canonical RealityScan 360 cubemap rig XMP sidecar schema
|   |-- RealityScan Rig Guide 01..06.txt # Technical references for RealityScan / RC camera rigs
|   |-- WALKTHROUGH_*.md           # Historical release walkthroughs
|
|-- presets/                       # Saved pipeline and engine configuration presets (JSON)
|-- versions/                      # Immutable historical snapshots of every version
|   \-- historical_binaries/       # Archived compiled executables from past releases
\-- logs/                          # Run diagnostic and telemetry logs (abu_align_*.log)
```

---

## 3. Environment & Runtime Specifications

### Conda Environment 1: `vggt` (Photogrammetry, CV, SAM2, GUI, CLI, Viser)
- **Executable**: `C:\Users\USer\miniconda3\envs\vggt\python.exe`
- **Key Packages**:
  - Python 3.10
  - PyTorch 2.5.1 + CUDA 12.4 (`bfloat16` & `float32`)
  - Torchvision, Torchaudio
  - OpenCV 4.10 (`cv2`) with GUI support
  - SciPy 1.13 (`scipy.optimize`, `scipy.spatial`)
  - PyCeres / Ceres Solver (Spherical Bundle Adjustment)
  - SAM 2 (`segment-anything-2`)
  - Viser 0.2.23+ (WebGL 3D interactive viewer)
  - Open3D, Trimesh, Plyfile
  - Tkinter, CustomTkinter, Pillow (PIL)

### Conda Environment 2: `gs_clean` (Nerfstudio, PFGS360, CUDA Splatting)
- **Executable**: `C:\Users\USer\miniconda3\envs\gs_clean\python.exe`
- **Executables**: `C:\Users\USer\miniconda3\envs\gs_clean\Scripts\ns-train.exe`
- **Key Packages**:
  - Nerfstudio 1.1.4
  - `nerfstudio360` / `gsplat360` (PFGS360 CVPR 2026 pose-free model)
  - `spherical_distortion` custom CUDA extensions (SCA-PE equirectangular splatting)
  - UniK3D / ZoeDepth monocular depth backends

### Linux Subsystem: WSL2 Ubuntu (`Ubuntu`, RTX 4090 Native Acceleration)
- **Command**: `wsl -d Ubuntu -- bash -c "..."`
- **Execution Script**: Embedded in `/mnt/c/Users/USer/.../scratch/abu_360dvo_runner.py`
- **Virtual Environment**: `/home/ahmad/360DVO` (Native PyTorch, CUDA 12.x, distortion-aware spherical correlation)
- **Used by**: `360dvo_engine.py` via `wsl_bridge.py`.

---

## 4. Deep-Dive: Core Subsystems & Key Recent Upgrades

### 1. 360DVO Engine (`360dvo_engine.py` / `wsl_bridge.py`)
- **Underlying Technology**: Direct visual odometry on the $S^2$ spherical manifold (IEEE RA-L 2026).
- **Recent Upgrade**: Fully operational native Linux execution via WSL2 on the RTX 4090.
- **Key Parameters**:
  - `Canvas Resolution`: 1024px safe integer DPVO patch correlation.
  - `Active Patches`: 96 patches per frame with gradient-biased edge sampling.
  - `Polar Zenith/Nadir Mask`: 15% cutoff (removes operator helmet and nadir tripod).
  - `ODBA Sliding Window`: 16 keyframe bundle adjustment window.
  - `Metric Scale`: 3.2m default physical scene scale multiplier.
  - `Dense Micro-Cloud (9x)`: 3×3 sub-patch extraction producing 220k+ dense colored tie points.
  - `Refinement`: 24 final global bundle adjustment iterations.
- **Exports**: Automatically generates `360dvo_trajectory.txt` (TUM format), `COLMAP_Lichtfield/` (cameras, images, points3D in both text and binary formats), `Postshot_RealityCapture/cameras.csv`, `3dgs_dataset/transforms.json`, and `points3D.ply`.

### 2. Interactive 3D Visualizer (`fast_viewer.py` & `viewer_workspace.py`)
- **Recent Critical Overhaul (v0.1.3.5)**:
  1. **Percentile-Based Scene Bounds**: In equirectangular reconstructions, spherical rays near the horizon create outlier tie points at hundreds of meters. The viewer previously used raw min/max, which inflated the scene span to 1,370m and backed the camera up 1.5 km away! `scene_bounds()` now computes robust **1.5% to 98.5% percentiles**, tightening scene bounds to ~35m and placing the camera directly in front of the model.
  2. **Trajectory-Proportional Frustum Sizing**: Frustum scale was previously tied to the uncleaned scene span, blowing up to 24.6 meters per camera! It is now computed dynamically from the camera trajectory and inter-camera step distance:
     $$\text{cam\_base\_scale} = \min(\max(\text{med\_step} \times 1.6, \text{traj\_span} \times 0.02), \text{traj\_span} \times 0.08, 0.45)$$
     yielding realistic ~18 cm frustums along a 10 cm step trajectory.
  3. **Camera Station Nodes**: Added a new layer `Camera stations (nodes)` (`self.stations`). Every registered camera position is rendered as a distinct node dot along the polyline path.
  4. **All Frustums Display**: Added a `Frustum display` control (`All frames` vs `Sampled (max 250)`), defaulting to all frames.
  5. **Real Filename Prioritization**: `load_run_cameras()` now prioritizes `COLMAP_Lichtfield/images.txt` over raw timestamp TUM files, enabling photo thumbnails, photo planes, and the **View photo** button to display original images (`Container1_Mask000.jpg`).
  6. **Auto Trajectory Framing**: When opening `http://localhost:8080`, Viser automatically calls `fit_trajectory()`, framing the camera path immediately.

### 3. Panoramic Splitter & RealityScan XMP Rig Integration (`panoramic_splitter.py`)
- **Purpose**: Deconstructs equirectangular 360° panoramas into 6 cubemap perspective images (Front, Back, Left, Right, Up, Down).
- **RealityScan 2.2+ Camera Rig Schema**: Photogrammetry engines like RealityCapture and RealityScan fail if cubemap images are treated as independent uncalibrated cameras. They must be registered as a **fixed multi-camera rig**:
  - All 6 cameras in a station share the **exact same optical center** $(X, Y, Z)$.
  - Exact $90^\circ$ relative rotation matrix:
    - `Front`: Yaw $0^\circ$, Pitch $0^\circ$, Roll $0^\circ$
    - `Right`: Yaw $+90^\circ$, Pitch $0^\circ$, Roll $0^\circ$
    - `Back`: Yaw $180^\circ$, Pitch $0^\circ$, Roll $0^\circ$
    - `Left`: Yaw $-90^\circ$ ($270^\circ$), Pitch $0^\circ$, Roll $0^\circ$
    - `Up`: Pitch $+90^\circ$
    - `Down`: Pitch $-90^\circ$
  - Sidecar `.xmp` files are written next to each exported JPG using the official schema in `docs/True_Schema_360_Camera_Rig.xmp`.

### 4. Abu Preprocessing Tools
- **Frame Picker (`frame_picker_*.py`)**: High-speed, multi-threaded image picker that samples every $N$-th image ("frame stride / interval"). Features a live preview slider before copying or cutting files to the target destination.
- **Blur Drop (`blur_drop_*.py`)**: SOTA image sharpness scoring engine combining Modified Laplacian variance, Tenengrad gradient energy, Fast Fourier Transform (FFT) high-frequency spectral power, and CPBD (Cumulative Probability of Blur Detection). Includes a multi-threaded batch worker and an interactive Tkinter preview grid with mouse-wheel zooming and panning.
- **SAM 2 Masker (`sam2_*.py`)**: Zero-shot AI segmentation tool for removing tripods, operator reflections, moving pedestrians, and cars before running photogrammetry.
- **Renamer (`renamer_*.py`)**: High-speed batch renamer supporting regular expressions, sequential number padding, search-and-replace, and live previews.

---

## 5. Co-Development Protocol: Antigravity & GPT-6 Astra

To ensure that **Antigravity** and **GPT-6** collaborate harmoniously on this codebase without conflicting or regressing each other's work, both agents must follow these rules:

### Rule 1: Subsystem Division of Labor

```
+-------------------------------------------------------------------------------------------------+
|                                    CO-DEVELOPMENT LANES                                         |
+----------------------------------------------------+--------------------------------------------+
| GPT-6 ASTRA LANES                                  | ANTIGRAVITY LANES                          |
+----------------------------------------------------+--------------------------------------------+
| * GUI Layout & UX:                                 | * Core Math & Solvers:                     |
|   - abu_align_gui.py (Widgets, dialogs, styling)   |   - opensfm_engine.py (Spherical math, BA)  |
|   - blur_drop_ui.py, frame_picker_ui.py            |   - 360dvo_engine.py (Odometry, ODBA)      |
|   - renamer_ui.py, file_sift_ui.py                 |   - vggttt_engine.py (Foundation SfM)      |
|   - Mouse event handling & canvas viewports        |                                            |
|                                                    | * GPU & Low-Level Systems:                 |
| * Tooling, Scripts & Utilities:                    |   - wsl_bridge.py (Linux / CUDA bridge)    |
|   - frame_picker_cli.py, blur_drop_cli.py          |   - train_engine.py (Nerfstudio / CUDA)    |
|   - renamer_engine.py, file_sift_engine.py         |   - PyCeres / SciPy optimization kernels  |
|                                                    |                                            |
| * Exporters & Schemas:                             | * 3D Engine & Inspection:                  |
|   - RealityScan XMP sidecars & rig templates       |   - viewer_workspace.py (Viser scene graph)|
|   - cameras.csv / RealityCapture converters        |   - fast_viewer.py (Spatial KD-trees)      |
|   - Presets manager & JSON validation              |   - Memory buffers & percentile bounds     |
|                                                    |                                            |
| * Documentation & Test Suites:                     | * Hardware Telemetry & Architecture:       |
|   - docs/*.md guides & walkthroughs                |   - Process streaming, pipe handlers       |
|   - tests/test_*.py integration tests              |   - VRAM / CUDA allocation safety          |
+----------------------------------------------------+--------------------------------------------+
```

### Rule 2: Non-Destructive Editing & File Verification
1. **Never Assume Memory**: Before modifying any file, always view the latest contents using read/view tools to inspect the active code lines.
2. **Preserve Comments & Docstrings**: Do not delete or strip existing comments, docstrings, mathematical explanations, or telemetry hooks.
3. **No Duplicate Files**: Never create duplicate folders (such as `Abu_Align/` or `temp/`). All code belongs in the canonical root.
4. **Clean File Handover**: When finishing a task, record a brief note of what changed so the other agent can pick up context seamlessly.

### Rule 3: Strict Base-10 Odometer Versioning
Version format is strictly `vMAJOR.MINOR.PATCH.BUILD` (e.g. `v0.1.3.5`).
- Every digit is strictly an integer between `0` and `9`.
- When incrementing, if the rightmost digit hits 9, it rolls over to 0 and carries +1 to the left:
  $$\text{v0.1.3.5} \to \dots \to \text{v0.1.3.9} \to \text{v0.1.4.0} \dots \to \text{v0.1.9.9} \to \text{v0.2.0.0}$$
- **To bump version**: Run the version manager tool:
  ```powershell
  python version_manager.py --bump
  ```
  This automatically:
  1. Increments `VERSION.txt` following odometer rules.
  2. Updates version badges across GUI files, CLI headers, and batch scripts.
  3. Recompiles `Abu_Align_and_Train.exe` and `Abu_Align.exe` using `csc.exe` (Roslyn C# compiler).
  4. Creates an immutable snapshot in `versions/vX.Y.Z.W/`.
  5. Archives the versioned binary in `versions/historical_binaries/`.

### Rule 4: Mandatory Pre-Handoff Compilation Check
Before declaring any task complete or handing over to the user or the other agent, run the syntax compilation test across all core modules:
```powershell
& "C:\Users\USer\miniconda3\envs\vggt\python.exe" -c "
import py_compile, glob
modules = [
    'abu_align_gui.py', 'desktop_ui.py', 'align_cli.py', 'train_engine.py',
    'fast_viewer.py', 'viewer_workspace.py', '360dvo_engine.py', 'opensfm_engine.py',
    'panoramic_splitter.py', 'frame_picker_engine.py', 'blur_drop_engine.py'
]
for m in modules:
    py_compile.compile(m, doraise=True)
print('[SUCCESS] All critical Abu Align modules compiled with 0 syntax errors!')
"
```

### Rule 5: Windows Path & Subprocess Hygiene
- Always wrap Windows filesystem paths in raw strings (`r"C:\Users\..."`) or forward slashes to prevent escape sequences.
- Spawning subprocesses must specify `PYTHONUNBUFFERED=1` and `PYTHONIOENCODING=utf-8`.
- In PowerShell, run commands sequentially with `;` (do **not** use `&&`).

---

## 6. Standard Results Output Format (`results/sparse_XX_<engine>/`)

Every alignment run produces a standardized bundle ready for 3DGS training, photogrammetry inspection, and RealityCapture import:

```
results/sparse_02_360dvo/
|-- 3dgs_dataset/                  # Universal 3DGS dataset format
|   |-- images/                    # Direct links / staged photos
|   |-- sparse/0/                  # COLMAP format (cameras.bin, images.bin, points3D.bin)
|   \-- transforms.json            # Standard NeRF / 3DGS camera poses
|-- COLMAP_Lichtfield/             # Specialized COLMAP format for LichtFeld Studio & NeRFstudio
|   |-- cameras.txt / .bin
|   |-- images.txt / .bin          # Exact 6-DoF poses with image filenames
|   \-- points3D.txt / .bin
|-- Postshot_RealityCapture/       # RealityCapture and Postshot direct export
|   |-- cameras.csv                # Exact CSV format with #Image name, X, Y, Alt, Pitch, Roll, Yaw
|   |-- transforms.json
|   \-- points.ply                 # Dense colored point cloud tie points
|-- 360dvo_trajectory.txt          # Robotics TUM format (timestamp, tx, ty, tz, qx, qy, qz, qw)
|-- points3D.ply                   # Root dense colored point cloud (200k+ vertices)
|-- reconstruction_report.txt      # Formatted summary of photos, registration %, runtime, engine settings
\-- abu_align_YYYY-MM-DD_*.log     # Complete execution and hardware telemetry log
```

---

## 7. Quick Command Cheat Sheet for Agents

```powershell
# 1. Check current version
python version_manager.py --current

# 2. Run Headless 360DVO Alignment (WSL2 GPU accelerated)
python align_cli.py --input "path/to/360_images" --engine 360dvo --stride 1 --load_res 1024 --dvo_patches 96

# 3. Launch Interactive 3D Viser Viewer on a run
python fast_viewer.py --input "path/to/results/sparse_02_360dvo" --port 8080

# 4. Run Headless Frame Picker (sample every 5th image)
python frame_picker_cli.py --input "path/to/images" --output "path/to/picked" --interval 5 --mode copy

# 5. Run Headless Blur Drop (filter out blurry photos)
python blur_drop_cli.py --input "path/to/images" --output "path/to/sharp" --threshold 45.0 --algorithm hybrid

# 6. Run Equirectangular Cubemap Splitter with RealityScan Rig XMP sidecars
python panoramic_splitter.py --input "path/to/panoramas" --output "path/to/pinhole_rig" --layout cubemap --fov 90 --export_xmp

# 7. Bump Version & Recompile Launcher Executables
python version_manager.py --bump

# 8. Run Full Automated Test Suite
python -m unittest discover -s tests -p "test_*.py"
```

---

> *This documentation is maintained jointly by the Google Antigravity and OpenAI GPT-6 co-development team. Always update this file when introducing new engines, tools, or major architectural changes.*
