# Abu Align & Train v0.0.2.6 Walkthrough: Professional 360° Multi-Camera Rig Projector & Real-Time Canvas Visualizer

## Executive Summary

**Abu Align & Train v0.0.2.6** delivers an advanced, professional overhaul of the **360° Multi-Camera Rig Projector & Panoramic Splitter** application. This update bridges 360° spherical imaging and high-fidelity 3D Gaussian Splatting (3DGS) / Structure-from-Motion (SfM) by providing rigorous mathematical rig synthesis, real-time visual canvas feedback, direct video intake, and native resolution matching.

In adherence to the strict base-10 odometer rule, the version has been incremented from `v0.0.2.5` to **`v0.0.2.6`**.

---

## 1. Key New Features & Professional Capabilities

```mermaid
flowchart TD
    subgraph Intake ["360° Spherical Input Sources"]
        PanoFolder["📁 Equirectangular Panoramas (.jpg, .png, .webp)"]
        VideoFile["🎬 360° Panoramic Video (.mp4, .mov, .insv, .mkv)"]
    end

    subgraph RigArchitectures ["Virtual Multi-Camera Rig Architectures"]
        Ring["Equatorial Ring (N cams, 0-360° yaw)"]
        Cube["6-Face Cubemap (Front, Right, Back, Left, Up, Down)"]
        Dome["Multi-Tier Spherical Dome (+/- Pitch Tiers + Poles)"]
        Fibonacci["Fibonacci Sphere (Golden Spiral Uniform 4π)"]
    end

    subgraph AutoMathSolver ["Auto-Math Coupling Engine"]
        Solver1["Lock Overlap: Auto-calculate FOV from N & Overlap %"]
        Solver2["Lock FOV: Auto-calculate Camera Count N from FOV & Overlap %"]
        AutoRes["⚡ Auto-Res: Match Native Equirectangular Angular PPD"]
        Distortion["Brown-Conrady Radial Distortion (k1, k2)"]
    end

    subgraph RealTimeVisualizer ["High-DPI Interactive Live Canvas"]
        PolarView["🧭 Polar Radar: Cardinal bearings, FOV cones, color-coded overlap arcs"]
        EquirectView["🗺️ 2D ERP Footprint: 2:1 map unroll with pitch/yaw bounding boxes"]
        Telemetry["Telemetry Dashboard: Δθ, Overlap %, fx/fy, Total views estimate"]
    end

    subgraph Outputs ["Standardized Pinhole Dataset Output"]
        ImagesDir["images/ (Rectilinear pinhole views)"]
        TransformsJSON["transforms.json (COLMAP / Nerfstudio / PFGS360)"]
        RigManifest["rig_manifest.json (Complete mathematical metadata)"]
    end

    Intake --> AutoMathSolver
    RigArchitectures --> AutoMathSolver
    AutoMathSolver --> RealTimeVisualizer
    AutoMathSolver --> Outputs
```

---

## 2. Mathematical Rig Coupling & Optics Formulas

### A. Horizontal Ring Overlap & FOV Coupling
For an equatorial ring with $N$ virtual cameras:
- **Angular separation (step)**:
  $$\Delta \theta = \frac{360^\circ}{N}$$
- **Horizontal overlap percentage**:
  $$\text{Overlap} = \max\left(0, \frac{\text{FOV}_h - \Delta \theta}{\text{FOV}_h}\right) \times 100\%$$
- **Auto-Math Mode: Lock Overlap (Auto-FOV)**:
  $$\text{FOV}_h = \frac{\Delta \theta}{1 - \frac{\text{Overlap}}{100}} = \frac{360^\circ}{N \times (1 - \text{Overlap}/100)}$$
- **Auto-Math Mode: Lock FOV (Auto-Camera Count)**:
  $$N = \left\lceil \frac{360^\circ}{\text{FOV}_h \times (1 - \text{Overlap}/100)} \right\rceil$$

### B. Spherical Fibonacci / Golden Spiral Distribution
To achieve uniform, mathematically optimal camera distribution over the entire $4\pi$ steradian sphere without polar clustering:
For $i = 0, \dots, N-1$:
$$y_i = 1 - \frac{2i + 1}{N}, \quad r_i = \sqrt{\max(0, 1 - y_i^2)}$$
$$\theta_i = i \times \phi, \quad \text{where } \phi = 2\pi \left(1 - \frac{1}{\frac{1 + \sqrt{5}}{2}}\right) \approx 137.507764^\circ$$
$$\text{pitch}_i = \arcsin(y_i), \quad \text{yaw}_i = \text{atan2}(\cos\theta_i \cdot r_i, \sin\theta_i \cdot r_i)$$

### C. ⚡ Auto-Res: Native Angular Pixel Density Matching
To avoid oversampling or loss of detail at the center of the rectilinear pinhole crop:
- Equirectangular angular resolution:
  $$\text{PPD}_\text{ERP} = \frac{W_\text{ERP}}{2\pi \text{ radians}}$$
- Center pinhole pixel density matches $\text{PPD}_\text{ERP}$:
  $$W_\text{opt} = 2 f \tan\left(\frac{\text{FOV}_h}{2}\right) \approx \frac{W_\text{ERP}}{\pi} \tan\left(\frac{\text{FOV}_h}{2}\right)$$
Rounded to the nearest multiple of 16 for GPU texture and video encoder alignment.

### D. Aspect Ratio & Vertical FOV
For output dimensions $W \times H$:
$$\text{FOV}_v = 2 \arctan\left( \frac{H}{W} \tan\left(\frac{\text{FOV}_h}{2}\right) \right)$$

### E. Radial Distortion Model ($k_1, k_2$)
Normalized camera plane coordinates $(u, v)$ with radius $r^2 = u^2 + v^2$:
$$u_d = u(1 + k_1 r^2 + k_2 r^4), \quad v_d = v(1 + k_1 r^2 + k_2 r^4)$$
Enables barrel or pincushion distortion modeling or correction, exported directly to `transforms.json` with camera model `OPENCV`.

---

## 3. Real-Time Interactive Canvas Visualizer

The UI features a dedicated, high-DPI canvas (`520x370 px`) that updates instantaneously as the user moves any slider or modifies any parameters:

1. **🧭 Polar Radar View (Top-Down)**:
   - Concentric range rings ($30^\circ, 60^\circ, 90^\circ$) with cardinal bearings: $N (0^\circ)$, $E (+90^\circ)$, $S (180^\circ)$, $W (-90^\circ)$.
   - Center camera rig node with directional ray arrows.
   - Translucent stippled field-of-view cones.
   - **Dynamic Overlap Color Coding**:
     - **Optimal (40% - 65%)**: `#10b981` (Vibrant Emerald Green - recommended for SfM tie point matching).
     - **Dense (>65%)**: `#3b82f6` (Sky Blue - dense multi-view redundancy).
     - **Moderate (20% - 40%)**: `#f59e0b` (Amber warning).
     - **Gap (<20%)**: `#ef4444` (Red alert - danger of disjoint reconstructions).
2. **🗺️ 2D Equirectangular Footprint View**:
   - 2:1 aspect ratio spherical projection map.
   - Latitude lines ($\pm 60^\circ, \pm 30^\circ, 0^\circ \text{ Equator}$) and Longitude lines ($\pm 180^\circ, \pm 90^\circ, 0^\circ$).
   - Rectangular bounding footprints for all virtual cameras with automatic horizontal $360^\circ$ boundary wrapping and center crosshairs.
3. **Live Mathematical Telemetry Dashboard**:
   - Displays real-time angular separation $\Delta \theta$, overlap %, focal lengths $f_x, f_y$, vertical FOV, estimated total output views, and spherical coverage area.

---

## 4. Direct 360° Video Intake Pipeline

Users can now select 360° video files (`.mp4`, `.mov`, `.insv`, `.mkv`, `.avi`) directly:
- Automatic metadata inspection: dimensions, native FPS, total frame count, and video duration.
- Flexible frame sampling:
  - **Target FPS**: Extract at a desired frequency (e.g. 1.0, 2.0, or 5.0 fps).
  - **Keyframe Stride**: Extract every $K$-th frame.
  - **Max Frames Cap**: Safeguard to limit output view count.
- Extracts and projects video frames on the fly without intermediate disk bloat.

---

## 5. Verification Results

| Test Scenario | Test Input | Settings | Result | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| **Video Intake & Projection** | 8K 360° Video (7680x3840 @ 50fps) | 4-cam Ring, 90° FOV, 512px, 1 frame | ✅ Success (4 views + transforms.json) | **0.78s** |
| **Fibonacci Sphere CLI** | 5.7K Panorama (5760x2880) | 12-cam Fibonacci, 90° FOV, 512px | ✅ Success (12 views + transforms.json) | **0.52s** |
| **Auto-Res Solver** | 5.7K ERP (5760px) @ 90° FOV | Auto-Res button | ✅ Solved: **1840 x 1840 px** | Instant |
| **Auto-Math Solver Coupling** | 8 cams @ 50% overlap | Lock Overlap mode | ✅ Adjusting $N=6 \to \text{FOV}=120.0^\circ$ | Instant |
| **Interactive GUI & Canvas** | AbuAlignApp GUI Dialog | Polar & Equirect views, 2-column layout | ✅ Rendered & passed headless test | Clean exit |
