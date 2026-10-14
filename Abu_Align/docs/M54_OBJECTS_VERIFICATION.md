# Abu M54 v0.1.5.2 — exact poses, nulls, gizmos and desktop display

Root causes: the standalone created Tk without DPI awareness and never set the
window icon. Windows could bitmap-stretch it; the child Python/Tk titlebar kept
the feather despite the launcher having an Abu resource. The installed Viser
0.2.23 gizmo supplies a TransformControlsHandle to on_update, not a click event.
The old client-event check discarded real drags and recreated the gizmo during
each redraw. Closing that server with a browser attached also left producer
waiters; the app adapter now closes actual owned connections, drains that private
loop and joins its exact worker. No installed package or upstream repo is changed.

Implementation: DPI setup before Tk, explicit multi-resolution Abu ICO on each
M54 top-level, modeless object browser with adjustable panes, exact camera poses
and point coordinates, independent float64 oriented nulls with bounded history
and atomic save/load. Native RGB shafts/rings support world/local axes, one Undo
per drag, Esc cancellation and model A/B targets. Full fitted B matrices are
preserved at drag start. Browser handles persist through updates; local fields,
null markers and history work alongside orbit/pan. Nulls remain world objects,
separate from GCP measurements and reconstruction/training exports. Cloud/control
rotations are undefined; model-frame rotation is displayed separately. A browser
opened from the native viewer receives an independent null snapshot.

Control JSON sidecars triangulate in the selected reconstruction frame and survive
the pickle-free background archive. No source geometry is altered. The standalone
prompts to save unsaved nulls on close; embedded/browser hosts require explicit Save.
Gizmo scale is numeric only. Browser history groups updates separated by <0.5s,
because this installed API has no drag-end callback. Native history has exact
gesture boundaries. The default browser sidebar now shows all null history buttons.

## Verification

**129 tests passed in 86.450 seconds**, including camera
calibration/refinement, Manual control points, CSV/grid/help, native/main adapters
and new object/gesture cases. Clean process exit; only the existing PyTorch sparse
CSR beta warning appears. Tests use synthetic fixtures; no client dataset opened,
uploaded or committed. Fresh Windows child tests verify process DPI awareness=2
and the real titlebar WM_GETICON. Measured native DPI: 144.071
(this monitor is approximately 150%). Browser tests deliver actual installed
TransformControlsUpdateMessage packets, not guessed callback event shapes.
Connected shutdown is required to finish <2.5s without pending/unretrieved tasks.
Active gesture close releases the GPU context and pending timer/grab.

Visible loopback browser checks created a null, changed XYZ/rotation fields, dragged
an arrow (X 2.5 to approximately 5.3964, along its oriented axis), dragged a ring
(Y rotation to approximately 47.762 degrees) and orbited normally. The final layout
was checked again with Add, Delete, Undo and Redo accessible. The screenshots showed
only synthetic geometry and were returned to this cloud assistant. Test tabs/server
were closed afterward. No private panorama thumbnails were captured.

Test command (from the isolated checkout, with root and tests on Python sys.path):

```powershell
$env:PYTHONPATH = "$PWD;$PWD\tests"
& C:\Users\USer\miniconda3\envs\vggt\python.exe -B -m unittest test_manual_cameras test_control_points_workspace test_control_import_navigation test_control_comparison_help test_control_acceleration test_spherical_refinement test_spherical_integrity test_desktop_ui test_abu_m54 test_m54_objects test_native_inspection test_viewer_workspace
```

## Performance and memory

Synthetic scene: 50,000 points, 593 cameras,
100 nulls, inspector open, grid and gizmo visible. Renderer:
NVIDIA GeForce RTX 4090/PCIe/SSE2; OpenGL 4.6.0 NVIDIA 610.62. GPU completion is synchronized
with glFinish. 130 measured frames after 20 warmup frames:
median 16.612 ms, P95 16.825 ms.
Whole-model transform + render: median 31.009
ms, P95 38.572 ms over 60 updates.
Observed RSS max 170.62 MiB; last-half rendering RSS
growth 12.0 KiB. Null history is capped at
32 snapshots / 1,000 nulls; inspector topology is cached. Frame buffers/display
points remain bounded. Contexts, worker threads, socket connections and timers are
closed by the owning host. The viewer uses OpenGL and does not allocate PyTorch/CUDA
tensors. OpenGL per-process VRAM is **not instrumented under WDDM**; no zero-VRAM or
universal-FPS claim is made. This is a synthetic single-PC timing, not a survey or
alignment accuracy benchmark.

Local metric script and JSON are in `C:\Users\USer\.codex\visualizations\2026\10\12\01a1309e-ce5f-7812-885a-2fe829e2d27d` as
`m54_objects_probe.py` and `m54_objects_benchmark.json`. Run the script with
`benchmark` for the same fixture. The full test log is
`m54_objects_verified_tests.log` in that directory.

## Release and concurrent work

Release checkout: `C:\Users\USer\.codex\worktrees\m54-viewer-release\abu align`; branch `codex/m54-viewer-release`.
The other agent owns the canonical v0.1.5.0 counter and ongoing GUI/splitter/rig
edits. This release bumps the already committed M54 v0.1.5.1 to v0.1.5.2, compiles
Suite/versioned/M54 launchers, and verifies root/Abu_Align/both snapshots for the
isolated checkout. Only M54-owned files and the two scene archive functions are
synchronized to the shared folder and its mirror; unrelated edits and its counter
are preserved. Use the isolated Abu_M54.exe for the exact verified v0.1.5.2 build.
Git clean status applies to this isolated release, not the other agent's checkout.
