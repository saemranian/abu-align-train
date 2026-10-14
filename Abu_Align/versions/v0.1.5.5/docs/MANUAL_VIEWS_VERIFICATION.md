# Manual views v0.1.5.3 — verification

## Root causes and design

The earlier opaque splitters still let stretched Treeview/child requested widths
feed back into parent geometry. Repacking the central hint and crowded toolbar
made that feedback visible as clipped controls and unused central space. Pane
frames now own their allocated rectangles (geometry propagation disabled), image
selectors use weighted grids, and the toolbar has stable rows. Sash motion uses
one fixed 16 ms frame consuming the latest position; mouse release commits the
final position. It does not debounce until movement stops or recursively enter
Tk's event loop for each mouse movement. The normal geometry loop paints each
accepted frame. Vertical measurement sizing uses the same scoped drag handler.

The old grid sent even warm cache hits through decoding workers and repeatedly
created/destroyed child widgets. It now pools lightweight Canvas items, reads warm
Tk images immediately, deduplicates local decode work and prefetches small catalogs
or nearby rows. Two CPU workers use JPEG draft scaling. PIL, Tk-image and failure
caches each have a 128-entry cap; only visible/overscan tiles are displayed. An
uncached first visit has a neutral image icon; failures remain explicit/logged.
No Loading banner is used. All images remain on one virtualized scroll region.

Left and Right independently select Image, Grid or 3D. Embedded grids default to
opening the other pane. Mode switches retain existing zoom/view state and suspend
hidden image rendering. All M54 instances share one background scene result;
navigation/transform/file parsing remain in the shared M54 engine. Selecting a
camera opens the other pane only when it is already Image. UI previews do not
change source geometry, original pixels, refinement controls or training files.

The angular-error numeric cells alone are coloured green/yellow/red. Defaults
are <=0.10°, <=0.50° and above0.50°; unknowns remain dashes. Limits change live and
are saved separately as display_settings, with atomic validation before loading.
They are excluded from solver settings. The high-DPI minimum-size check exposed
a wrapped legend consuming the table height; it now lives in the scrolling form.

The broad run also exposed Tk finalizers during threaded image loading. New
thumbnail workers now own only plain state, locks, queues and PIL buffers. Original
image/native scene workers similarly capture immutable input snapshots and result
queues instead of the GUI owner. Stale generations are rejected on the main thread.
Contexts, thumbnail workers, grabs, listeners and scheduled callbacks close with
their owner. No upstream repositories or installed packages were modified.

## Verification

Final relevant suite: **152 tests passed in 102.454 s**,
exit code0, without Tk main-thread finalizer warnings. It covers the Manual views,
CSV/navigation/help, calibration, spherical/CUDA refinement, main desktop handoff,
M54 scene/native/browser/object controls, and legacy loaders. Synthetic tests verify
layout before mouse release, coalescing100 motion events without restarting the
deadline, vertical table resizing, shared warm grids with no redecodes, preserved
mode state, two GPU views sharing source data, exact error-cell colours, settings
roundtrip and invalid-load atomicity, source/measurement integrity and cleanup.
The existing PyTorch sparse CSR beta warning is not a test failure.

```powershell
$env:PYTHONPATH = "$PWD;$PWD\tests"
& C:\Users\USer\miniconda3\envs\vggt\python.exe -B -m unittest test_manual_views test_control_points_workspace test_control_import_navigation test_control_comparison_help test_control_acceleration test_manual_cameras test_spherical_refinement test_spherical_integrity test_desktop_ui test_abu_m54 test_m54_objects test_native_inspection test_viewer_workspace test_fast_loader test_fast_viewer
```

## Local timings and memory

Synthetic48-image grid: cold thumbnails 0.225s;
80 warm scroll jumps median 8.922ms, P95
12.899ms. Additional decodes during scrolling:0.
Tile pool remained 24 to 24.
Monitor density 144.071dpi. The fixed sash handlers measured median
0.162ms, P95 0.203ms;
the complete scheduled frame including normal Tk paint measured median
170.539ms, P95 226.071ms.
Maximum central parent/child width mismatch after each frame before release:0px;
actual sash range 319px. Full-window Tk painting
at2500x1400 remains a limit; these measurements do not establish60fps or claim
that every large-window paint is instantaneous. A minimal same-size Tk baseline
with36 buttons also took about47.5ms per full paint on this PC. Coalescing prevents
motion-event repaint backlog and geometry feedback; it does not replace Tk/GDI.

Observed synthetic scrolling RSS max 122.91MiB;
last-half growth 8.0KiB. RGB thumbnail
buffers 2.62MiB, excluding Tk/GUI overhead.
The authorized48 real panoramas were decoded locally in a withdrawn Tk root for
numeric-only cache checks: 1.200s cold, warm lookup median
0.001100ms, P95 0.001545ms;
48 decoded, 0 failed, process RSS
86.26MiB. A warm lookup alone excludes painting.
M54 uses the existing OpenGL GPU renderer; these UI changes introduce no CUDA
training tensors. Per-process OpenGL VRAM is not instrumented under WDDM.

Logs/metric scripts are local to `C:\Users\USer\.codex\visualizations\2026\10\12\01a1309e-ce5f-7812-885a-2fe829e2d27d`:
manual_views_final_tests.log, manual_views_probe.py, manual_views_benchmark.json,
manual_views_private_metrics.json. Private photo bytes are not in these artifacts
or in Git. No accuracy improvement is inferred from a UI/cache benchmark.

## Screenshot privacy incident

During a requested synthetic UI check, the computer-use screenshot tool returned
the existing client photo window instead of the selected synthetic window. One
screenshot containing private image pixels was returned to this cloud conversation.
The user was informed immediately; capture stopped and subsequent verification
used numerical/local tests only. Original dataset files were not uploaded or
modified. This incident is separate from the hidden local thumbnail benchmark;
do not claim that no private pixels were transmitted in this turn.

## Release isolation

Verified checkout: `C:\Users\USer\.codex\worktrees\m54-viewer-release\abu align`, branch codex/m54-viewer-release. The canonical
v0.1.5.0 counter and other agent's mainGUI/splitter/rig work are preserved. This
release bumps the isolated counter to v0.1.5.3, recompiles the Suite/versioned C#
launchers, mirrors every active file and creates both complete version snapshots.
Only owned Manual UI files/guides are synchronized to the canonical checkout and
its Abu_Align mirror. Clean Git status and full parity refer to the isolated release.
