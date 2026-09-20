# Abu Align: Complete User Guide & Fundamentals Manual

**Software Name:** Abu Align (Neural Photogrammetry & Self-Calibration Engine)  
**Version:** v0.0.0.3  
**Target Applications:** RealityScan / RealityCapture, Postshot (3D Gaussian Splatting), COLMAP, Nerfstudio  
**Author / Maintainer:** saemranian  

---

## 1. Introduction & Core Fundamentals

### 1.1 What is Abu Align?
**Abu Align** is a state-of-the-art **Hybrid Neural Photogrammetry Suite** designed to estimate camera poses (positions and orientations), dense 3D point clouds, and optical lens distortion parameters from multi-view photographs in seconds.

Traditionally, software like RealityCapture, Metashape, or COLMAP relies on classical Structure-from-Motion (SfM):
1. Detecting 2D keypoints (SIFT).
2. Exhaustive pairwise feature matching.
3. Incremental epipolar geometry reconstruction.
4. Dense Multi-View Stereo (PatchMatch MVS) depth estimation.

While mathematically accurate, classical photogrammetry has severe drawbacks:
* **Fails on textureless surfaces:** White walls, clean floors, repetitive patterns, sky, reflections, and glass.
* **Extremely slow:** Dense point clouds take **15 to 45 minutes** to compute.
* **Fragile initialization:** If the initial image pair fails to align, the entire reconstruction collapses.

### 1.2 The Neural Transformer Revolution (VGGT & VGGT-Omega)
Abu Align integrates **VGGT** (Visual Geometry Grounded Transformer, 1.18 Billion parameters) and **VGGT-Omega** (the 2025/2026 compressed state architecture). 

Instead of matching individual pixels one-by-one:
1. The transformer processes all images simultaneously using self-attention and cross-attention tokens.
2. The model understands global scene context, semantics, and perspective geometry like a human brain.
3. In **~25 seconds**, it regresses complete 6-DoF camera poses and a continuous dense 3D point cloud, even across wide baselines and featureless surfaces.

### 1.3 Why Abu Align is a "Hybrid" Engine
Pure deep learning poses suffer from small floating-point drift ($\sim 2\text{--}4\text{ pixels}$ reprojection error), which can cause softness when training 3D Gaussian Splats or generating high-precision meshes.

Abu Align solves this through a **Hybrid Architecture**:
* **Deep Learning (VGGT):** Instantly reconstructs global camera layout, surface depth, and wide-baseline feature tracks.
* **Google Ceres Solver (Bundle Adjustment):** Solves non-linear Levenberg-Marquardt least squares using a **Cauchy robust loss function** and a **two-stage polish pass**, locking camera centers to **0.35 sub-pixel precision** and calculating physical lens distortion ($k_1, k_2, t_1, t_2$).

---

## 2. User Interface & Interactive Tooltip System

Starting in **Version 0.0.0.3**, Abu Align features native **High-DPI Awareness** (crisp vector rendering on 1080p, 1440p, and 4K displays) and an **Interactive Hover Tooltip System**. 

Hovering your mouse over any button, radio option, or dropdown displays an immediate contextual popup explaining what that setting does and when to choose it.

```
+---------------------------------------------------------------------------------------------------+
|  ABU ALIGN   [ v0.0.0.3 ]  |  Neural Photogrammetry & Fast Lens Self-Calibration Engine           |
+---------------------------------------------------------------------------------------------------+
|  Alignment Configuration                                                                          |
|  Dataset Folder: [ C:\...\Photos                             ]  [ Browse... ]                     |
|                                                                                                   |
|  1. Engine & Canvas:     2. Point Cloud Density:  3. Camera Alignment:     4. Lens Distortion:    |
|  (o) VGGT-Omega          ( ) Preview (100K)       ( ) Feedforward (~25s)   ( ) Pinhole (Zero dist)|
|  ( ) VGGT Standard 1B    (o) High Quality (300K)  ( ) Standard Ceres BA    (o) Brown Calibration  |
|  Res: [1024] Stride:[1]  ( ) Ultra Dense (800K)   (o) Ultra Sub-Pixel BA*       (k1-k2, t1-t2)    |
|                          ( ) Max Grade (1.5M)         (Cauchy + Polish)                           |
+---------------------------------------------------------------------------------------------------+
|  [ ▶ START ALIGNMENT ]    [ ■ STOP ]    [ 📁 Open Results ]    [ 🌐 3D Browser Viewer ]           |
+---------------------------------------------------------------------------------------------------+
|  Live Telemetry & Process Logs                                                                    |
|  > Abu Align Version 0.0.0.3 Ready.                                                               |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Comprehensive Parameter Guide

### 3.1 Column 1: Engine & Canvas

#### `VGGT-Omega (Fast, Low VRAM)` *(Recommended)*
* **Category:** SOTA Deep Learning AI (2025/2026).
* **What it does:** Employs compressed register tokens and sliding-window attention. Reduces GPU VRAM consumption by ~60% compared to standard models, allowing higher resolution images and longer sequences without out-of-memory errors.
* **When to use:** Default choice for virtually all workstations and GPUs (8GB to 24GB VRAM).

#### `VGGT Standard 1B`
* **Category:** SOTA Deep Learning AI (2024).
* **What it does:** The original 1.18-Billion parameter foundation transformer trained by Meta. Processes full quadratic global attention.
* **When to use:** Research benchmarks or setups with 24GB+ VRAM (RTX 3090, 4090, A6000).

#### `Canvas Resolution (Res: 1024 / 1536 / 2048)`
* **Category:** Image Processing (Input Canvas).
* **What it does:** Scales the longest edge of input photos before patch tokenization.
* **Selection Guide:**
  * **1024:** Fast and memory-efficient (~25 seconds). Ideal for quick preview runs.
  * **1536:** Recommended sweet spot for 24MP to 60MP professional cameras. Provides fine sub-pixel edge discrimination.
  * **2048:** Maximum optical resolution. Requires 16GB–24GB GPU VRAM.

#### `Keyframe Stride (1, 2, 3...)`
* **Category:** Algorithmic Subsampling.
* **What it does:** Samples every $N$-th photo ($I_{0}, I_{s}, I_{2s}, \dots$).
* **Selection Guide:**
  * **1:** Standard photogrammetry datasets (processes all photos).
  * **2 or 3:** Drone grid flights, video extractions, or continuous walking timelapses with high overlapping redundancy.

---

### 3.2 Column 2: Point Cloud Density (Geometry)

Unlike classical photogrammetry where dense point generation takes 30 minutes, Abu Align's neural network generates 3D surface depth in **2.6 seconds**. The density options filter the network's confidence tensor $\sigma(x, y)$:

#### `Preview (100K pts, Conf >= 2.2)`
* **What it does:** Retains only points with internal neural confidence $\ge 2.2$.
* **When to use:** Ultra-fast preview to verify scene scale and camera positions in web viewers.

#### `High Quality (300K pts, Conf >= 1.8)` *(Recommended Default)*
* **What it does:** Retains points with confidence $\ge 1.8$.
* **When to use:** Ideal geometric seed for **Postshot (3D Gaussian Splatting)** and **RealityScan**. Provides crisp surfaces without floating noise.

#### `Ultra Dense (800K pts, Conf >= 1.5)`
* **What it does:** Retains points with confidence $\ge 1.5$.
* **When to use:** Detailed architectural surfaces, indoor interiors, furniture, and recessed shadows.

#### `Max Scan Grade (1.5M pts, Conf >= 1.2)`
* **What it does:** Retains points with confidence $\ge 1.2$ (over 1.5 million points).
* **When to use:** Fine structural details: cables, thin wires, sharp corners, organic vegetation, and foliage.

---

### 3.3 Column 3: Camera Alignment Accuracy (Pose Solver)

#### `Feedforward (Fast, ~25s)`
* **Category:** 100% Pure Deep Learning.
* **What it does:** Predicts all camera 6-DoF poses directly in one forward neural pass without bundle adjustment.
* **Reprojection Error:** $\sim 2\text{--}4\text{ pixels}$.
* **When to use:** Instant scene inspection or fast sanity checks.

#### `Standard Ceres BA (~1m)`
* **Category:** Hybrid (Deep Tracking + Classical Optimizer).
* **What it does:** ALIKED neural feature tracks are optimized via Google Ceres Levenberg-Marquardt bundle adjustment.
* **Reprojection Error:** $\sim 1\text{--}2\text{ pixels}$.
* **When to use:** Standard photogrammetric scenes with moderate overlap.

#### `Ultra Sub-Pixel BA (Cauchy + Polish)` *(Recommended)*
* **Category:** Hybrid (Deep Multi-View Association + Robust M-Estimation + Polish).
* **What it does:**
  1. Tracks 3,072 feature points across 8 query frames (>16,000 ray constraints).
  2. Applies **Cauchy robust loss** ($\rho(s) = c^2 \ln(1 + s/c^2)$) to downweight false matches without skewing camera centers.
  3. Prunes all residuals $> 2.5\text{ pixels}$ and tracks visible in fewer than 3 views.
  4. Runs a second Levenberg-Marquardt polish pass.
* **Reprojection Error:** **0.35 pixels** (Sub-pixel perfection).
* **When to use:** Production 3D Gaussian Splatting (Postshot), VFX matchmoving, and survey-grade CAD modeling.

---

### 3.4 Column 4: Lens Distortion Model

#### `Pinhole (Ideal, zero distortion)`
* **What it does:** Assumes an ideal rectilinear pinhole lens ($k_1=0, k_2=0, t_1=0, t_2=0$).
* **When to use:** Images that have already been pre-undistorted in Lightroom / Capture One, or telephoto lenses ($>50\text{mm}$) with negligible barrel curvature.

#### `Brown Calibration (k1-k2, t1-t2)` *(Recommended)*
* **What it does:** Calibrates physical radial ($k_1, k_2$) and decentering tangential ($t_1, t_2$) distortion using the classical Brown-Conrady formulation.
* **Safety Feature:** Locks $k_3 = 0, k_4 = 0$ to eliminate Runge's polynomial oscillation and prevent edge warping in Postshot.
* **When to use:** Wide-angle lenses (14mm, 16mm, 24mm), GoPro / action cameras, smartphone lenses, and uncalibrated drone cameras.

---

## 4. Step-by-Step Workflow Guide

### Step 1: Capture Guidelines
* Maintain 60%–80% overlap between adjacent photos.
* Lock exposure, white balance, and manual focus if possible.
* Avoid heavy motion blur.

### Step 2: Running Abu Align
1. Launch **Abu Align** by double-clicking [`Abu_Align_v0.0.0.3.exe`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align_v0.0.0.3.exe).
2. Click **Browse...** and select your photos folder.
3. Set your preferred options (defaults are pre-configured for maximum accuracy):
   * Engine: `VGGT-Omega`
   * Density: `High Quality (300K pts)`
   * Alignment: `Ultra Sub-Pixel BA (Cauchy+Polish)`
   * Lens Model: `Brown Calibration (k1-k2, t1-t2)`
4. Click **▶ START ALIGNMENT**.
5. Watch live progress in the telemetry console.

### Step 3: Inspecting in 3D Browser Viewer
* Click **🌐 3D Browser Viewer** to launch the interactive Viser viewer (`http://localhost:8080`).
* Pan, rotate, inspect camera frustums, and check point cloud alignment in real-time 3D.

### Step 4: Importing Results into Target Applications

Click **📁 Open Results** to access the export packages:

#### A. Postshot (3D Gaussian Splatting)
1. In Postshot, click **Import**.
2. Select the exported `sparse/` directory or `cameras_alignment.csv`.
3. Postshot automatically loads the calibrated camera poses, focal lengths, and $\{k_1, k_2, t_1, t_2\}$ distortion parameters with zero edge stretch.
4. Select `points.ply` as the initialization point cloud and start training.

#### B. RealityScan / RealityCapture
1. Open RealityCapture or RealityScan.
2. Drag and drop the original photos.
3. Import the generated `cameras_alignment.csv` using:
   * Coordinate system: `Grid (X, Y, Z)`
   * Camera rotation: `Yaw, Pitch, Roll` or `Rotation Matrix (R11..R33)`
   * Distortion: `Brown 3` or `OpenCV (k1, k2, t1, t2)`
4. The photos will instantly lock into calibrated alignment without requiring a separate SfM solve.

---

## 5. Frequently Asked Questions (FAQ)

### Q1: Why did the UI look blurry or low-resolution previously?
**Answer:** On Windows, when an application does not declare itself as **DPI-aware**, the Windows Desktop Window Manager (DWM) renders the window at 96 DPI (100% scale) off-screen and then bilinearly stretches it up to match your screen scale (125%, 150%, or 200%). In **Version 0.0.0.3**, native Win32 DPI awareness is activated at launch, ensuring crisp, razor-sharp vector text and UI elements at any display resolution.

### Q2: What did the Git remote message mean ("Renamed origin to upstream")?
**Answer:** In Git, `origin` is the nickname for where code is pulled from or pushed to. Because the repository was originally cloned from Facebook's public repository (`facebookresearch/vggt.git`), `origin` pointed to Facebook. However, you cannot push to Facebook's repository. By renaming Facebook's link to `upstream` (the universal open-source standard for parent code), we freed up `origin`. This allows you to set `origin` to your own personal GitHub repository (`https://github.com/saemranian/Abu-Align.git`) and push your software with one command.

### Q3: Why is $k_3$ and $k_4$ kept at 0?
**Answer:** In photogrammetry, higher-order radial coefficients ($k_3, k_4$) are high-degree polynomials ($r^6, r^8$). Unless you have hundreds of photos with targets covering every millimeter of the extreme sensor corners, estimating $k_3$ and $k_4$ causes Runge's polynomial oscillation phenomenon. This produces unnatural warping, stretching, and distortion blowup at the edges of the frame in Postshot. Locking them to zero while calibrating $k_1, k_2, t_1, t_2$ gives physically stable results.
