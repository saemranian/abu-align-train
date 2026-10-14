# Walkthrough: Abu Align & Train v0.1.4.8
## 360 Splitter UI Overhaul: Expansive Sliders, Direct Keyboard Entry, and Unclipped Ergonomics

### Overview
In `v0.1.4.8`, the **360 Splitter (Virtual Rig Designer)** received a complete UI and ergonomics overhaul addressing tight slider tracks, cramped spacing, label clipping, and keyboard precision:

1. **Expansive 360px+ Slider Tracks**:
   - Reorganized the parameter row layout. The slider track previously competed with fixed labels and checkboxes, squishing it to an unusable ~100px stub.
   - Sliders now expand freely across **360px–385px** of track length, providing smooth, high-precision drag adjustments for Cameras, Horizontal FOV, Overlap %, and Lens Distortion ($k_1$).

2. **Direct Numeric Keyboard Input with Interactive Stepping**:
   - Every parameter row now features an integrated numeric Entry box alongside the slider:
     - **Cameras**: Exact camera count (3–32), with disabled state locking to 6 for Cubemap mode.
     - **Field of View**: Exact horizontal FOV in degrees (e.g. `114.0`, `90.0`, `65.0`).
     - **Overlap**: Exact target overlap percentage (0.0%–85.0%).
     - **Distortion ($k_1$)**: Exact radial distortion coefficient (-0.20 to +0.20, with sign and 2 decimal places).
     - **Dome Pitch & Cams/Tier**: Exact pitch angle (+5° to +85°) and cams per tier.
     - **Output Dimensions & FPS**: Resolution stepping (64px) and video frame rate stepping.
   - All entries support `<Return>`, `<KP_Enter>`, and `<FocusOut>` commit, plus `<Up>`/`<Down>` arrow stepping (with Shift 2x acceleration) and mouse wheel scrolling.

3. **Zero Text & Label Truncation**:
   - Eliminated the truncated `"Cube Faces (Fi"` bug. Labels now maintain a fixed, clean `"Cameras"` title (width 11), while the right-hand unit badge cleanly renders `"faces (cube)"` or `"cams"`.
   - All parameter row labels (`Cameras`, `Field of view`, `Overlap`, `Distortion`) have uniform width and alignment.

4. **Interactive 35mm Focal Equivalent Badge**:
   - The 35mm equivalent focal length (e.g., `(18.0mm)`) is now rendered as a compact, styled pill/badge right beside the degree symbol.
   - Clicking the badge toggles telemetry visibility without altering slider dimensions or layout.

5. **Wider Left Control Column & Improved Padding**:
   - Left settings column expanded from 560px to 620px (`left.configure(width=620, bg=BG)`).
   - Inner `ScrollPage` body padding reduced to `12px` (from `22px`), freeing up 20px of horizontal room for controls.

---

### Verification
- Tested all 4 rig architectures (Cubemap, Ring, Dome, Fibonacci).
- Verified bidirectional sync: dragging slider updates keyboard entry; typing into entry updates slider and 3D visualizer.
- Tested keyboard arrow stepping and mouse wheel scrolling on all numeric inputs.
- Verified absence of label clipping in Cubemap mode.
- Launcher executables recompiled and verified.
