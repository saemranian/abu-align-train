# Walkthrough: Abu Align & Train v0.0.2.8

## Overview

**Abu Align & Train v0.0.2.8** introduces an interactive **3D Virtual Camera Rig Visualizer & Orbit Viewport** directly inside the **360° Multi-Camera Rig Projector & Splitter** dialog (`media_1790086520218.png`).

Users can now examine, rotate, zoom, and inspect their synthesized virtual camera rigs in full 3D space in real-time as they tune rig parameters (camera count, FOV, overlap %, aspect ratio, and pitch tiers).

---

## Key Features & Changes

### 1. Interactive 3D Camera Rig Orbit Viewport (`abu_align_gui.py`)
- **Three Unified View Modes**:
  1. `🌐 3D Orbit Rig` (default): True 3D perspective viewport with orbit rotation, pan, zoom, wireframe reference sphere, optical axis rays, and depth-sorted frustums.
  2. `🧭 Polar Radar`: Top-down circular compass showing angular distribution and FOV wedges on the equatorial plane.
  3. `🗺️ 2D Footprint`: Equirectangular $(360^\circ \times 180^\circ)$ panoramic unroll displaying rectilinear projection rectangles.
- **Controls & Interaction**:
  - **Left-Click Drag**: Orbit rotates the camera around the rig ($\text{Yaw} \in [0^\circ, 360^\circ]$, $\text{Elevation/Pitch} \in [-88^\circ, +88^\circ]$).
  - **Shift + Left-Click Drag / Right-Click Drag**: Screen-space camera pan ($\Delta X, \Delta Y$).
  - **Mouse Wheel / Scroll**: Smooth continuous zoom ($\text{distance multiplier} \in [1.2, 9.0]$).
  - **Double-Click**: Instantly resets 3D camera to default azimuth ($35^\circ$), elevation ($22^\circ$), and zoom ($3.4\times$).
  - **`🔄 Reset 3D` Button**: One-click reset in the top header bar.
  - **`⚡ Auto-Spin` Button**: Automated turntable spin animation ($30\text{ FPS}$) with start/stop toggle and clean thread cancellation on dialog exit.
  - **Active Cursor**: Switches dynamically to `"fleur"` when hovering over the canvas in 3D mode.

### 2. 3D Perspective Rendering & Frustum Geometry
- **$S^2$ Unit Sphere Reference Wireframe**:
  - Equator circle in vivid indigo (`#4f46e5`, width=1.5).
  - Latitude parallels at $\pm 30^\circ$ and $\pm 60^\circ$ (dashed `#252538`).
  - Meridian circles at $0^\circ/180^\circ$ and $90^\circ/270^\circ$ (dashed `#232338`).
  - Polar axis with directional arrow from Nadir ($-Y$) to Zenith ($+Y$, `#a855f7`).
  - Cardinal ticks and labels at the Equator: North ($0^\circ$), East ($+90^\circ$), South ($180^\circ$), West ($-90^\circ$) in sky blue (`#38bdf8`).
- **Depth-Sorted Frustums (Painter's Algorithm)**:
  - Each camera's 3D forward vector $\mathbf{fwd}$, right vector $\mathbf{right}$, and up vector $\mathbf{up}$ are derived from $(\text{yaw}, \text{pitch}, \text{roll})$.
  - Frustum plane calculated at distance $D=1.12$ with aspect-ratio-corrected dimensions:
    $$w = D \cdot \tan(\alpha_h), \quad h = \frac{w}{\text{aspect}}$$
  - Computes 4 sensor plane corners ($TL, TR, BR, BL$), pyramid edges from origin $(0,0,0)$, and center optical axis ray with arrow tip.
  - **Translucent Sensor Face**: Shaded with Tkinter `stipple="gray25"` and color-coded by rig overlap:
    - **Optimal ($40\% \le \text{overlap} \le 65\%$)**: Emerald green (`#10b981`).
    - **High ($>65\%$)**: Sky blue (`#38bdf8`).
    - **Medium ($20\% \le \text{overlap} < 40\%$)**: Amber (`#f59e0b`).
    - **Low ($<20\%$)**: Crimson red (`#ef4444`).
  - Sorted back-to-front by camera view depth ($Z_{\text{depth}}$) prior to drawing, ensuring foreground frustums occlude background elements without Z-fighting.
  - Billboard labels (`C0`, `C1`, etc.) floating just outside sensor planes.
- **Corner Orientation Triad (Orthographic XYZ Gizmo)**:
  - Mini 3D axes widget in the bottom-right corner showing $+X$ (Red), $+Y$ (Green), $+Z$ (Blue) dynamically rotating with the orbit camera.
- **Onscreen HUD Diagnostics**:
  - Live readout of camera count $N$, FOV, overlap %, orbit azimuth, elevation, zoom level, and mouse navigation hints.

### 3. Version Control & Launcher Synchronization
- Strict base-10 odometer increment from `v0.0.2.7` $\to$ `v0.0.2.8`.
- C# launchers recompiled with Microsoft .NET `csc.exe`:
  - `Abu_Align/Abu_Align_and_Train.exe`
  - `Abu_Align/Abu_Align_and_Train_v0.0.2.8.exe`
  - `Abu_Align_and_Train.exe`
  - `Abu_Align_and_Train_v0.0.2.8.exe`
- 100% parity verified between workspace root and `Abu_Align/`.
- Full version snapshot preserved in `Abu_Align/versions/v0.0.2.8/`.

---

## Verification & Test Results

### 1. Interactive 3D Visualizer Test Suite
Executed comprehensive automated headless GUI tests:
- **Button Existence**: Verified `🌐 3D Orbit Rig`, `🧭 Polar Radar`, `🗺️ 2D Footprint`, `🔄 Reset 3D`, and `⚡ Auto-Spin` buttons are properly configured.
- **Mouse Drag Orbit**: Simulated left-drag from $(200, 200) \to (240, 180)$, confirming continuous angle updates and redraws without error.
- **Mouse Wheel Zoom**: Simulated zoom in ($\Delta = +120$) and zoom out ($\Delta = -120$), verifying smooth distance scaling.
- **Mouse Pan**: Simulated Shift-drag and Right-drag, verifying screen-space offset translation.
- **Reset 3D**: Invoked reset button, confirming return to $(35^\circ, 22^\circ, 3.4\times)$.
- **Auto-Spin Turntable**: Toggled auto-spin on and off, verifying state changes and clean timer loop cancellation.
- **Result: PASSED (100%)**.

### 2. View Mode & Rig Architecture Coverage
- Tested switching between all 3 canvas view modes (`3d`, `polar`, `equirect`) on live canvas.
- Tested all 4 virtual rig architectures in 3D orbit mode:
  - `ring`: Horizontal equatorial ring with $N$ cameras.
  - `cubemap`: 6-face cubemap ($90^\circ$ FOV, Zenith + Nadir + 4 horizontal faces).
  - `dome`: Multi-tier spherical dome with zenith/nadir and customizable pitch tiers.
  - `fibonacci`: Golden spiral spherical distribution on $S^2$.
- All 4 architectures rendered completely with zero gimbal-lock errors, zero mathematical singularity issues, and full depth-sorted canvas elements.
- **Result: PASSED (100%)**.

### 3. Parity & Build Verification
- Verified version numbers across `VERSION.txt`, `abu_align_gui.py`, `align_cli.py`, `train_engine.py`, `Abu_Align.bat`, and `Abu_Align_and_Train.bat`.
- Verified binary presence and execution of `Abu_Align_and_Train_v0.0.2.8.exe`.
- Created clean version snapshot in `Abu_Align/versions/v0.0.2.8/`.
- **Result: PASSED (100%)**.
