# v0.0.8.5 — Main 3D workspace and comparison

## Outcome
The embedded 04 3D Workspace now provides a dedicated inspection layout with
stable navigation and two-run comparison. The browser viewer shares registration
and distance math and offers the same A/B inspection workflow.

## Changes
- Focused main-tab layout; setup header collapses while inspecting, and progress,
  Pause, and Stop remain available. Returning to setup restores the usual layout.
- Native Compare sidebar, optional Inspect controls, wrapping toolbar, adaptive
  Fit, standard views, orbit/pan modes, wheel zoom, keyboard shortcuts, camera
  picking, camera focus/look-through, and saved viewpoints.
- A-only/B-only/overlay modes, mint/coral colors, camera-center residuals,
  optional similarity registration, and sampled B-to-A point-distance colors.
- Background native loading, two-run caches, finite-point filtering, bounded
  preview sampling, safer offscreen rendering, and far-to-near point drawing.
- Correct half-turn rotations and NeRF OpenGL camera directions for inspection.
- Browser viewpoint stays fixed during display changes; standard views support
  Z-up/Y-up. Compatible Viser theme avoids its hidden-share-button React crash.
- Source reconstruction files are never rewritten by comparison or navigation.

## Verification
Synthetic browser and full native-app previews were inspected. Browser Top,
overlay, and registered camera residuals were verified visually. Native minimum
layout, background loading, fit margins, camera actions, view restoration,
registration failure retention, and distance legends are covered by tests.
Final automated result: 44 tests passed in 63.570 seconds across
 test_fast_viewer, test_viewer_workspace, test_native_inspection,
 test_desktop_ui, and test_gui_presets.
Command: python -m unittest discover -s tests -p <test_file.py> -v for each
listed test module (the final run combined these into one unittest suite).

Backup verification: all 23 Python modules compiled; active-source, launcher,
icon, and documentation parity passed. Native and browser workspace imports were
verified from inside the standalone snapshot.

## Memory and limits
Native previews retain up to 150,000 points per run; browser budgets range from
100,000 to 1,000,000 per run. Nearest-sample cloud comparison uses up to 50,000 A
points. Two source runs are cached. Large PLY files are still fully parsed before
native sampling, so peak load memory can exceed retained preview memory.
No alignment or training was launched, and no VRAM or production-scene FPS
benchmark was performed. Native rendering is a CPU preview, not a GPU CAD engine.
Camera frustums are illustrative. Residuals measure sampled inter-run agreement
in scene units, not ground-truth reconstruction accuracy. Registration requires
three or more uniquely matched, noncollinear camera centers.

## Files and release
Core: native_inspection.py, workspace_3d.py, desktop_ui.py, viewer_workspace.py,
fast_viewer.py, and version_manager.py. Tests: test_native_inspection.py,
test_viewer_workspace.py, and test_desktop_ui.py. Release includes synchronized
version headers, launchers, icons, and documentation.
Backup: versions/v0.0.8.5/ in the consolidated project root.
See docs/WORKSPACE_3D_DESIGN.md for critique, phases, and usage details.
Concurrent Antigravity releases advanced the baseline to v0.0.8.4 and captured
some browser-workspace edits. This release preserves that committed engine work.
