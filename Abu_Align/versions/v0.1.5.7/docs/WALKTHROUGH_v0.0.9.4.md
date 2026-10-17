# v0.0.9.4 — Purposeful 3D inspection presets

## Critique and phases
The same dense scene served every inspection task. Looking at surface detail
required several layer toggles; examining trajectories required removing points.
Fit included invisible content, undermining otherwise simple navigation.

Phase 1: consistent Overview, Geometry and Cameras presets in both viewers.
Overview shows points, cameras and trajectories. Geometry isolates points.
Cameras isolates frustums and trajectories. Presets preserve the current viewpoint
and source reconstruction data; they do not change A/B comparison selection.

Phase 2: Fit now uses visible-layer bounds. Native rendering supports a separate
Points toggle, skips hidden point rasterization, and does not pick hidden cameras
or draw their selection markers. Browser Geometry mode also hides camera offset
links. Native manual layer changes report Custom rather than a stale preset name.

Phase 3: moved browser Open run folder to Run details, keeping navigation focused
on the scene. Native profile selector is integrated into the wrapping toolbar.

## Verification
42 tests passed in 36.673 seconds across test_viewer_workspace,
test_native_inspection and test_desktop_ui. Following final refinements, all 26
viewer/native tests passed in 6.549 seconds. Tests instantiate real Tk and Viser
interfaces using synthetic data and cover source/view preservation, layer presets,
visible-camera fitting, hidden-camera picking, and minimum-size layout.
Run: python -m unittest discover -s tests -p test_viewer_workspace.py
and repeat for test_native_inspection.py and test_desktop_ui.py.
No manual visual review or production dataset benchmark was performed this pass.

## Memory and release
No training/alignment was run. VRAM and production FPS are unmeasured. Native
camera-only inspection skips point rasterization; source preview caches are retained
for quick switching. Existing sampling budgets remain unchanged.
Modified implementation: native_inspection.py, viewer_workspace.py, workspace_3d.py.
Tests: tests/test_native_inspection.py and tests/test_viewer_workspace.py.
Release headers, launchers, VERSION.txt and documentation synchronized.
Backup: versions/v0.0.9.4/. Concurrent vggttt_engine.py and its test modifications
are excluded from the UI commit; backup uses the committed engine baseline.
