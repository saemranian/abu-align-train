# Main 3D workspace — design review and implementation

## Problems addressed

The original native workspace put nearly every command in one horizontal toolbar.
Controls could disappear at ordinary window sizes. It offered a single run, no
camera selection, and no comparison reference. Its browser counterpart reloaded
files and reset the viewpoint after display changes, making inspection frustrating.
Some imported half-turn camera rotations were silently replaced with identity;
NeRF camera axes were also incompatible with the viewer's camera convention.
Those errors made camera look-through unreliable before any visual polish.

## Phase 1 — Navigation and hierarchy

- Keep the native scene central. Reconstruction selection and navigation stay
  visible; Compare opens a scrollable side panel, and Inspect reveals secondary
  display/camera controls. Toolbar controls wrap at narrower widths.
- The main app collapses its setup header while the 3D tab is selected. Pipeline
  progress and run controls remain available. Returning to another tab restores
  the setup layout. Native Fit calculates distance from the projected bounds and
  current viewport dimensions, rather than using a fixed zoom value.
- Orbit or pan with left drag; right/Shift-drag pans, scroll zooms. F fits the
  scene; 1/3/7 select Front/Right/Top; brackets step through cameras.
- Save and restore viewpoints, including camera look-through. Click a projected
  camera center to select it, then Focus or Look through. Selection uses a yellow
  outline. Camera frustums are illustrative, not calibrated lens models.
- Browser standard views include Z-up/Y-up and fit according to viewport aspect
  and field of view. Its collapsible sidebar gives more room to the scene.

## Phase 2 — Trustworthy comparison

- A is the reference reconstruction; B is independently selected. A only, B only,
  and Overlay share the same viewpoint. Overlay uses mint/coral layer colors.
- Optional similarity registration fits rotation, positive scale, and translation
  from uniquely matching camera image names. Duplicate names are excluded. At
  least three noncollinear camera centers are required. Raw coordinates remain
  the default. Neither mode writes to source files.
- Camera-center residuals use scene units and measure inter-run agreement, not
  accuracy against ground truth. Registration residuals use the fitted cameras.
- Optional B-to-A point distance colors use directed nearest-neighbor distances
  against up to 50,000 sampled A points. The legend reports median, color range,
  and sampling limits. These are not exact surface distances or correspondences.
- Separate camera sequences avoid drawing a false path between runs. Sequence
  lines follow source file order, which need not be capture order.

## Phase 3 — Loading and rendering

- Native loading runs off the UI thread; only the main thread updates Tk. A new
  selection during loading is queued. Two cached source runs avoid repeat reads.
- Native preview caps each cloud at 150,000 sampled points. Browser budgets are
  100,000 / 300,000 / 1,000,000 points per run. Frustum drawing is limited while
  the camera selector retains the full camera list.
- Native rasterization paints far to near, handles offscreen depth/elevation
  views, and uses the requested point footprint size.
- Invalid registration leaves the previous scene intact. Empty or malformed PLY
  headers return instead of looping forever in the shared viewer reader.
- COLMAP/OpenSfM/ACE0 rotation conversion now handles all rotations. NeRF OpenGL
  camera axes are converted to the viewer's OpenCV convention. Relative image
  paths are preserved where available to avoid matching same-named files in
  different folders.
- Viser 0.2.23 has a conditional React-hook bug when hiding its share button.
  Retain that built-in button to avoid a blank frontend; the viewer binds to
  localhost and does not start public sharing automatically.

## Phase 4 — Verification and release

Tests cover registration recovery and rejection, duplicate matching, camera axes,
sample distances, view preservation, save/restore, real Viser API integration,
native minimum-size layout with both panels, asynchronous loading, empty/offscreen
rendering, and existing desktop/preset behavior. Synthetic browser and native
previews were used; user reconstruction files were not modified.

## Practical limits

This is a CPU raster preview inside Tk, with a browser WebGL alternative. It is
not a GPU-native CAD renderer. Performance on very large production projects,
all monitor/DPI combinations, and camera calibration image overlays still need
dedicated validation. Comparison is an overlay/A-B workflow, not synchronized
split-screen. No ML workloads or reconstruction benchmarks were run in this pass.
