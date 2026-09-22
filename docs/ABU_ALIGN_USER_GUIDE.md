# Abu Align & Train: Complete User Guide & Technical Manual

**Software:** Abu Align & Train (Unified Neural Photogrammetry & 3D Gaussian Splatting Suite)  
**Version:** v0.0.3.6
**Maintainer:** saemranian  
**Target Environments:** Windows 10/11 (CUDA 12.x+), Linux / RunPod Cloud GPU  
**Integrated Align Engines:** VGGT-Omega, Niantic ACE0, NVIDIA PyCuSFM, Mapillary OpenSfM, 360DVO  
**Integrated Train Engines:** LichtFeld Studio, Nerfstudio Splatfacto, NVIDIA 3DGRUT / 3DGUT, PFGS360  
**Target Applications:** Postshot, LichtFeld Studio, RealityCapture, MeshLab, Nerfstudio, COLMAP  

---

## Desktop workspace (v0.0.3.6)

The desktop UI uses three focused tabs: **Alignment**, **Training**, and
**Activity**. Dataset selection, presets, progress, and run controls remain
visible across tabs. Choose a dataset with **Browse folder** or Ctrl+O.

- **Alignment:** select an engine, image resolution, frame stride, and its
  detailed parameters. Settings groups wrap as the window narrows. Scroll or
  use Tab to reach remaining controls; focused controls scroll into view.
- **Training:** choose the training engine and iterations. **Train after
  alignment** determines whether the main workflow button includes training.
- **Activity:** inspect the read-only session log and open Results, Models,
  Logs, or the 3D viewer. Turn off **Follow log** to inspect earlier messages.
  The visible console retains about 5,000 lines; the full run log remains on
  disk. **Monitor** opens live hardware readings. Larger windows also show
  stages and telemetry inline; compact windows prioritize the log.

**Align & train / Start alignment** follows the automatic training option.
**Align only** always skips training. **Train only** uses an existing
reconstruction or a supported pose-free workflow. Starting a run opens Activity.
Pause and Stop become available during a run.

Use **Save as…** (Ctrl+Shift+S) to save a profile, then select it and choose
**Load**. Profiles keep all engine settings, but preserve the current dataset.
Use Alt+1/2/3 or Ctrl+Tab to switch tabs. Buttons support Enter and Space;
parameter tooltips appear on hover or keyboard focus and dismiss with Escape.

The Video frames and 360° splitter tools remain available beside the presets.
The workflow guidance below describes the same engines; older screenshots and
all-caps action names refer to the previous layout.

---

## 1. System Overview & Core Architecture

### 1.1 What is Abu Align & Train?
**Abu Align & Train** is an end-to-end professional neural reconstruction workstation that bridges **Deep Learning Photogrammetry** with **3D Gaussian Splatting (3DGS)**. It eliminates the fragile setup, slow computation (15–45 minutes), and textureless failure modes of classical Structure-from-Motion (SfM) by replacing them with modern foundation transformers, neural coordinate regression, high-throughput GPU matching, and spherical visual odometry.

```
RAW MEDIA INGESTION
[ Planar Photos | 360° Panoramas | Planar Videos | 360° Equirectangular Videos ]
                           │
                           ▼
           PROJECT & DATASET WORKSPACE
   [ Video to Frame Extractor | 360° Virtual Multi-Camera Rig Splitter ]
                           │
                           ▼
        NEURAL PHOTOGRAMMETRY & ALIGNMENT ENGINES
   ┌──────────────────────────────────────────────────────────────┐
   │ • VGGT-Omega:     Compressed Token Attention (~60% less VRAM) │
   │ • Niantic ACE0:   Scene Coordinate Regression (Ultra-Low VRAM)│
   │ • NVIDIA PyCuSFM: ALIKED/SuperPoint + LightGlue GPU SfM      │
   │ • OpenSfM:        Mapillary S² Manifold Spherical Bundle Adj.│
   │ • 360DVO:         Deep Visual Odometry for 360° Video Frames │
   └──────────────────────────────────────────────────────────────┘
                           │
                           ▼
            UNIVERSAL 3DGS DATASET FACTORY
   [ 3dgs_dataset/ : sparse/0/ + images/ + transforms.json + points.ply ]
                           │
                           ▼
         3D GAUSSIAN SPLATTING TRAINING SUITE
   ┌──────────────────────────────────────────────────────────────┐
   │ • LichtFeld Studio: Native Vulkan Real-Time GUI (120+ FPS)   │
   │ • Splatfacto:       PyTorch SOTA Optimizer + Direct PLY      │
   │ • NVIDIA 3DGRUT:    OptiX Ray Traced Gaussians + GUT Optics  │
   │ • PFGS360:          Pose-Free 360° GS + UniK3D Depth Priors  │
   └──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  FINAL 3D ASSETS
   [ .ply Gaussian Splat | .spz Compressed Splat | COLMAP | RealityCapture ]
```

---

## 2. Project & Dataset Workspace

### 2.1 Why the Dataset Folder is Positioned at the Top Workspace
In earlier versions, the `Dataset Folder` selector was placed inside an "Alignment Configuration" box. In **v0.0.2.5**, this has been redesigned into the **Project & Dataset Workspace** because the active dataset path serves as the **shared source of truth** for all three pipeline workflows:
1. **`▶ ALIGNMENT ONLY`**: Reads raw photos or video frames from this folder to compute camera alignments.
2. **`⚡ ONE-CLICK ALIGN & TRAIN`**: Reads input media, generates camera alignments, and automatically streams the generated `3dgs_dataset/` into the selected 3DGS training engine.
3. **`🎯 TRAIN 3DGS ONLY`**: Reads either pre-existing camera alignments (`sparse/0/`, `transforms.json`, or the latest `results/sparse_XX/3dgs_dataset/`) or feeds unposed frames directly into PFGS360's Pose-Free mode.

### 2.2 Live Dataset Health Inspector
Whenever you select or edit a dataset path, the built-in Health Inspector validates the directory structure in real time:
* **✓ Ready (Photos Detected):** Displays the total photo count and confirms whether camera alignments already exist.
* **🎬 Video Detected:** Flags video files (`.mp4`, `.mov`, `.insv`) and suggests extracting frames or running 360DVO.
* **⚠️ Path Not Found / Empty:** Prevents accidental execution on empty directories.
* **Smart Pose Verification:** If you click `🎯 TRAIN 3DGS ONLY` on a dataset that has no camera alignments (and you are not using PFGS360 Pose-Free), the software immediately alerts you and offers to run `⚡ ONE-CLICK ALIGN & TRAIN` with a single click.

### 2.3 Integrated Media Tools
* **🎬 Video Extractor Dialog (`Tools -> Video Extractor...`):** Extracts any high-resolution video into PNG/JPEG image sequences with configurable target FPS, temporal stride, native resolution downsampling, and Laplacian blur filtering.
* **🌐 360° Multi-Camera Rig Splitter (`Tools -> 360° Rig Splitter...`):** Converts 360° equirectangular panoramas into virtual multi-camera pinhole rigs:
  * **Ring Mode (Recommended):** 8 virtual pinhole cameras at 90° FOV providing an optimal **50.0% overlap** for classical and neural SfM engines.
  * **Cubemap Mode:** 6 orthogonal faces (Front, Right, Back, Left, Top, Bottom) at 90° FOV.
  * **Dome Mode:** 14 views arranged across equatorial and pole tiers.

---

## 3. Photogrammetry Alignment Engines

Abu Align provides 5 specialized alignment engines tailored for different camera optics and hardware profiles:

### 3.1 VGGT-Omega *(Recommended for Planar Photos)*
* **Architecture:** Foundation Visual Geometry Grounded Transformer with compressed register tokens.
* **Key Advantages:** Processes wide-baseline views without feature matching; ~60% lower VRAM consumption than standard models; continuous dense depth prediction.
* **Best Used For:** DSLR, mirrorless, smartphone, and drone photo captures with challenging textureless surfaces (white walls, smooth concrete, ceilings).

### 3.2 Niantic ACE0 *(ECCV 2024)*
* **Architecture:** Implicit Scene Coordinate Regression network.
* **Key Advantages:** Extremely low VRAM footprint (~2.0 GB); trains an MLP directly on unposed images; robust against repetitive patterns and lighting variations.
* **Best Used For:** Large multi-room indoor environments, low-VRAM laptops (6GB–8GB GPUs), and quick spatial maps.

### 3.3 NVIDIA PyCuSFM *(cuSFM GPU)*
* **Architecture:** High-throughput CUDA SfM by NVIDIA Isaac. Uses learned ALIKED / SuperPoint keypoints, LightGlue GPU feature matching, and NVIDIA cuDSS sparse Cholesky solver.
* **Key Advantages:** Blazing fast pairwise matching on RTX GPUs; high sub-pixel precision.
* **Best Used For:** High-resolution drone flights, large aerial datasets, and precision photogrammetry.

### 3.4 Mapillary OpenSfM *(360° Spherical SfM)*
* **Architecture:** Native spherical bundle adjustment operating on the unit bearing $S^2$ manifold.
* **Key Advantages:** Eliminates equirectangular polar distortion by projecting bearing vectors across the unit sphere; seamlessly handles continuous street-level spherical trajectories.
* **Best Used For:** Insta360, Ricoh Theta, and GoPro Max equirectangular photo captures.

### 3.5 360DVO *(IEEE RA-L 2026 - Deep 360° Visual Odometry)*
* **Architecture:** Deep visual odometry directly on continuous 360° equirectangular video and sequential spherical frames.
* **Key Advantages:** Uses distortion-aware spherical feature tracking (DAS-Feat) and tangent-plane patch sampling; operates at real-time speeds (15–30 FPS).
* **Best Used For:** Walking tours, drone 360° video sweeps, continuous robotic surveying, and unposed equirectangular video streams.

---

## 4. Parameter Compatibility Matrix

Not all parameters apply to all alignment engines. The following matrix details parameter scope:

| Parameter | UI Control | Applicable Engines | Description |
| :--- | :--- | :--- | :--- |
| **Resolution (`--load_res`)** | `Res: [1024]` | **All Engines (Universal)** | Longest-edge canvas scaling before feature extraction (512, 1024, 1536, 2048). |
| **Temporal Stride (`--stride`)** | `Stride: [1]` | **All Engines (Universal)** | Samples every $N$-th photo or video frame. |
| **Smart Cache (`.abu_cache`)** | `⚡ Smart Cache` | **All Engines (Universal)** | Caches neural predictions, tracks, and poses for instant re-runs (< 5s). |
| **Chunk Size (`--chunk_size`)** | `Chunk [VGGT]:` | **VGGT / VGGT-Omega Only** | Sliding-window attention chunk length (16 to 1024) for large image sequences. |
| **Anchor Overlap (`--anchor_size`)**| `Anchor [VGGT]:` | **VGGT / VGGT-Omega Only** | Overlap frames connecting consecutive sliding windows via Sim(3) Umeyama alignment. |
| **Patch Density (`--dvo_patches`)** | `Patch Graph` | **360DVO Only** | Number of active spherical feature patches tracked per frame (64, 96, 128). |
| **ACE Iterations / Seeds** | `ACE Iterations` | **ACE0 Only** | Coordinate network regression training steps and ensemble initialization seeds. |
| **Matching Algorithm** | `Matcher` | **PyCuSFM & OpenSfM** | LightGlue GPU matcher, FLANN, or Brute-Force feature matcher. |

---

## 5. Result Directory Structure & Expected Output Files

Each alignment run creates a dedicated, engine-isolated folder in `results/`:
```
results/
└── sparse_XX_<engine>/               (e.g. sparse_03_360dvo, sparse_01_vggt)
    ├── points3D.ply                  ★ Direct colored tie-point cloud (MeshLab / Postshot)
    ├── transforms.json               ★ Standard NeRF / 3DGS camera poses & intrinsics
    ├── reconstruction_report.txt     ★ Detailed telemetry, timings, and camera calibration
    ├── 3dgs_dataset/                 ★ Universal 3DGS Training Package
    │   ├── images/                   ★ Direct links / staged photos
    │   ├── sparse/0/                 ★ Standard COLMAP files (cameras, images, points3D)
    │   │   ├── cameras.bin / .txt
    │   │   ├── images.bin / .txt
    │   │   └── points3D.bin / .txt
    │   ├── transforms.json
    │   └── points.ply                ★ Gaussian initialization point cloud
    ├── COLMAP_Lichtfield/            ★ COLMAP format for LichtFeld Studio & classical tools
    │   ├── cameras.bin / .txt
    │   ├── images.bin / .txt
    │   └── points3D.bin / .txt
    └── Postshot_RealityCapture/      ★ Formatted for Postshot and RealityCapture
        ├── cameras.csv               ★ RealityCapture camera orientation table
        ├── transforms.json
        └── points.ply                ★ RealityCapture Z-up coordinate point cloud
```

### Is it expected to have a `.ply` file in the results?
**YES, ABSOLUTELY.**  
A `.ply` point cloud is a mandatory output of every successful photogrammetry run:
1. **`results/sparse_XX_<engine>/points3D.ply`:** The root-level point cloud containing reconstructed 3D tie points and the camera trajectory. Can be opened immediately in MeshLab, CloudCompare, or 3D viewer.
2. **`3dgs_dataset/points.ply`:** The seed point cloud used by 3D Gaussian Splatting algorithms to spawn initial 3D Gaussians.
3. **`Postshot_RealityCapture/points.ply`:** Point cloud transformed into the standard RealityCapture Z-up reference frame.

---

## 6. 3D Gaussian Splatting Training Suite (Abu Train)

Abu Align & Train includes 4 world-class 3DGS training engines:

### 6.1 LichtFeld Studio *(Vulkan Real-Time GUI + GUT)*
* **Technology:** Native C++ / Vulkan engine delivering real-time interactive splatting at 120+ FPS.
* **GUT Optics:** Gaussian Unscented Transform lens correction propagates ray bundles through non-linear camera models.
* **Export Formats:** Standard `.ply` and compressed `.spz` (FP16 packed quaternions, ~10x smaller).

### 6.2 Nerfstudio Splatfacto *(PyTorch SOTA)*
* **Technology:** Gold-standard PyTorch 3DGS optimizer with bilateral grid color correction, anti-aliased mip-filtering, and automatic camera pose optimization (`so3_x_r3`).
* **Export:** Automatically exports standalone `.ply` models.

### 6.3 NVIDIA 3DGRUT / 3DGUT *(Gaussian Ray Tracing)*
* **Technology:** NVIDIA Research framework combining OptiX ray tracing with Gaussian Unscented Transform.
* **Optimization:** Supports ADMM sparsity pruning (reducing VRAM and file sizes by 50–60%) and auto-exposure compensation.

### 6.4 PFGS360 *(CVPR 2026 - Pose-Free 360° GS)*
* **Technology:** Pose-Free Omnidirectional Gaussian Splatting designed specifically for equirectangular 360° media.
* **Routing Modes:**
  * **Auto-Detect [Default]:** If camera poses exist (OpenSfM or 360DVO), routes to SfM handoff. If unposed raw video is supplied, activates Pose-Free Direct joint pose and splat estimation.
  * **Spherical Priors:** UniK3D, midas, or zoedepth spherical depth models with Spherical Coordinate-Aware Positional Encoding (SCA-PE) to eliminate pole-pinch distortion.

---

## 7. Execution Workflows & Automated Routing

### 7.1 `⚡ ONE-CLICK ALIGN & TRAIN`
* **What happens:**
  1. Executes photogrammetry alignment using the selected Alignment Engine.
  2. Generates the universal `3dgs_dataset/` containing COLMAP sparse models, transforms, and point clouds.
  3. Seamlessly bridges to the selected Training Engine (LichtFeld Studio, Splatfacto, 3DGRUT, or PFGS360).
  4. Automatically trains 3D Gaussians and exports final `.ply` / `.spz` models into `3dgs_models/`.
* **Intelligence:** You never need to manually copy files or convert formats. The bridge verifies input compatibility automatically.

### 7.2 `▶ ALIGNMENT ONLY`
* **What happens:** Runs photogrammetry alignment and packages the results into `COLMAP_Lichtfield/`, `Postshot_RealityCapture/`, and `3dgs_dataset/`. Ideal when you wish to inspect camera poses or export to external tools (RealityCapture, Postshot).

### 7.3 `🎯 TRAIN 3DGS ONLY`
* **What happens:** Bypasses alignment and immediately starts 3DGS training.
* **Pose Resolution:** Checks the selected dataset folder for:
  1. `3dgs_dataset/`
  2. `sparse/0/`
  3. `transforms.json`
  4. The newest `results/sparse_XX/3dgs_dataset/`
* If no prior camera poses are found and the engine is not PFGS360, a prompt guides you to run `⚡ ONE-CLICK ALIGN & TRAIN`.

---

## 8. Hardware, VRAM Shield & Stability Best Practices

### 8.1 VRAM Shield & Downsampling
* When processing ultra-high-resolution imagery (e.g. 8K equirectangular frames or 60MP medium format photos), GPU VRAM can easily become exhausted.
* Abu Align & Train enforces a **VRAM Shield**:
  * On GPUs with 8GB–12GB VRAM, images are automatically capped at 2K resolution during attention tokenization.
  * For long sequences, set `Stride` to 2 or 3 to avoid GPU out-of-memory errors and excessive memory paging.

### 8.2 Avoiding Paging Loops
* If your GPU VRAM is maxed out, Windows begins paging GPU memory into system RAM, causing severe slowdowns.
* **Best Settings for 8GB–12GB GPUs:**
  * Canvas Res: `1024`
  * Stride: `2` or `3`
  * Chunk: `64` | Anchor: `16`
  * Engine: `VGGT-Omega` or `ACE0`
* **Best Settings for 24GB GPUs (RTX 3090 / 4090):**
  * Canvas Res: `1536` or `2048`
  * Stride: `1`
  * Chunk: `96`–`160` | Anchor: `24`–`32`
  * Engine: `VGGT-Omega`, `PyCuSFM`, or `360DVO`

---

## 9. Troubleshooting & FAQ

#### Q1: Why did 360DVO previously output `sparse_02_vggt`?
**Fixed in v0.0.2.5.** Previously, `get_next_sparse_dir` did not have an explicit case for `360dvo`, falling back to `vggt`. Output folders now strictly match the active engine (`sparse_XX_360dvo`, `sparse_XX_opensfm`, `sparse_XX_ace0`, etc.).

#### Q2: Why was there an unwanted `opensfm/` folder inside 360DVO results?
**Fixed in v0.0.2.5.** 360DVO now strictly isolates its outputs. It generates standard COLMAP `sparse/0/`, `transforms.json`, and `.ply` point clouds without generating redundant OpenSfM folders.

#### Q3: Where can I view my trained 3DGS models?
Trained models are saved to `results/sparse_XX_<engine>/3dgs_models/` as `.ply` or `.spz` files. They can be opened in:
* **LichtFeld Studio:** Drag-and-drop the `.ply` or `.spz` directly into the viewport.
* **Postshot:** Open `Postshot_RealityCapture/points.ply` or import the trained splat.
* **Web Viewers:** Open in supersplat, playcanvas, or nerfstudio viewer.
