# Manual control points — responsive panels and grid, v0.1.5.5

## Use the corrected interface

Close existing Abu windows, then launch the same `Abu_Align.bat` in the main
project folder. Its title and badge now read VERSION.txt, as do About, document
windows and new log headers. Both batch launchers and the Abu_Align mirror use
that same version. The earlier source changes reached the main folder while its
counter remained0.1.5.0; v0.1.5.4 was an isolated release to preserve concurrent
work. After confirmation that the other agent finished, v0.1.5.5 consolidates the
completed current shared source and prior Manual/M54 work into the main release.

In Manual control points, drag the visible handles between Points, the central
views and Properties/Settings. The middle divider changes Left/Right widths.
The Properties divider changes measurement-table height. The cursor indicates
the divider's direction; the enlarged handles scale with desktop density.
Dragging now continues when the pointer crosses a child image, table or notebook,
and release there commits the final size and releases the mouse grab.

**Settings → Pane resizing** offers:

- **Auto** (default): live pane sizing in small windows; above2 million physical
  window pixels, show a responsive green divider guide and commit on release.
- **Fast**: use the guide at every window size. Content stays stable during the
  gesture and redraws after release; this avoids repeated full-window Tk paints.
- **Live**: resize content during dragging. Large high-DPI windows can repaint
  more slowly; this is an explicit choice, not an assumption of60fps.

The choice saves in control-file layout_settings, separately from solver settings.
Older files default to Auto. Invalid choices are rejected before changing the
loaded document. **Project → Reset pane sizes**, or the Settings button, restores
the point list, image widths, properties width and measurement-table height.
If you shrink the whole window, side panels reduce to fit usable centre/minimum
sizes. Enlarging restores the preferred side widths instead of leaving clipped
or unreachable panes. Measurements and reconstruction data do not change.

The grid handles full and fractional wheel deltas, batches pending input into
one normal GUI-idle frame and keeps every image in the same scroll region. A warm
scroll does not trigger another decode or thumbnail build. The item pool updates
when the visible row range changes; moving within that range reuses its items.
The scrollbar coalesces pointer input too. Nearby/prefetched uncached images still
decode on two CPU workers, using a neutral placeholder for an initial miss.
Hidden grids do not create Tk photos on background completion, and thumbnail
width variants are quantized to avoid a new photo for every pixel of resizing.
There is no Loading banner, and invalid originals remain explicitly logged.

## Root causes fixed

The earlier grid handled each wheel event immediately, including repeated layout
and thumbnail requests after reaching a boundary. Every small delta was treated
as a full notch, causing excessive movement on high-resolution wheels/trackpads.
The corrected path accumulates the real delta and applies the latest target once.
Viewport signatures skip unchanged layouts and redundant scroll notifications.

The earlier drag listener existed only on the PanedWindow widget and used local
event coordinates. Tk's local grab can deliver events to children inside the
grabbed subtree, so dragging/releasing over images or tables was lost. The scoped
tool-toplevel listener and screen-to-split coordinate conversion fix that gap.
Sashes are clamped to adjacent minima. Tk only refreshes sash coordinates during
its geometry pass; consecutive sash_place calls during window resizing used stale
coordinates, shrinking the final panel unexpectedly. The controller now assigns
desired pane widths together and tracks preferred widths from the requested
final position, rather than stale coordinates. Both horizontal and vertical
paths, Fast/Live, shrink/expand, source preservation and cleanup are tested.

While consolidating the completed shared GUI, the broader tests caught an
existing splitter-dialog startup error: Tk Text tags interpret fg as fgstipple,
not the widget foreground alias. Five tags now use the explicit foreground
option; colours and the other agent's rig/splitter work remain intact.

## Verification — 2026-10-14

Final main-folder suite: **208 tests passed in 143.768s**,
exit0, without Tcl finalizer/fatal warnings. It includes Manual image policy,
local predictions, CSV/main handoff, perspective/spherical/CUDA refinement,
image/grid/M54 layouts, desktop version/presets, native/browser/null/gizmo
behaviour, rig CSV/designer and splitter masks. PyTorch's sparse CSR beta warning
is informational. Synthetic splitter fixtures are separate compatibility tests;
no client panoramas were converted. A restricted-sandbox run stalled while
starting the localhost Viser server; the owned process was stopped and the
browser/full suite rerun with localhost access. No remote image service was used.

```powershell
$env:PYTHONPATH = "$PWD;$PWD\tests"
& C:\Users\USer\miniconda3\envs\vggt\python.exe -B -m unittest test_manual_responsiveness test_manual_evidence test_manual_views test_control_points_workspace test_control_import_navigation test_control_comparison_help test_control_acceleration test_manual_cameras test_spherical_refinement test_spherical_integrity test_desktop_ui test_gui_presets test_abu_m54 test_m54_objects test_native_inspection test_viewer_workspace test_fast_loader test_fast_viewer test_run_manifest test_360dvo_cache test_dvo_tracks test_rig_cameras_csv test_rig_designer test_splitter_masks
```

Numeric Windows144.07dpi test with48 synthetic original panoramas and a mapped
1500x900 grid,2500x1400 tool window; no screenshots:

| Measurement | Before | After |
|---|---:|---:|
| 80 fractional-wheel events: handler work | 63.65ms | 3.33ms |
| Burst complete, including GUI processing | 69.32ms | 19.00ms |
| Burst arrange calls | 80 | 1 |
| Separate full-notch frame median / P95 | 6.29/11.57ms | 8.09/12.38ms |
| Left child-target drag | 0px | +120px |
| Settings/right child-target drag | 0px | -150px |
| Measurement-height child-target drag | 0px | -65px |
| Additional decodes while scrolling | 0 | 0 |

The old small-delta path over-scrolled; this comparison measures input/layout cost,
not identical visual scroll distance. Single-notch latency did not materially
improve; the important gains are burst/backlog removal and correct fine movement.
Continuous Fast-guide frame median/P95
28.12/36.49ms,
including its fixed16ms schedule. Repeated press/move gestures measured
53.39/73.72ms, compared with
143.57/168.20ms for old live
full-pane updates. These are different resize modes. The final content repaint
still costs 111.01ms median,
144.68ms P95 at this large size. Auto avoids doing
that paint at every movement; it does not eliminate Tk/GDI or promise60fps.
Maximum central geometry mismatch: 0px.
Synthetic process RSS 88.23MiB.

Authorized48 real panoramas were decoded read-only in a withdrawn window, with
numerical output only: 1.169s cold,48 decoded,0 failures,
warm photo lookup median/P95 0.001100/
0.001200ms,0 additional photo builds over480 lookups.
RGB cache 2.62MiB; process RSS
79.72MiB. A warm lookup excludes painting. Source photos,
filenames and pixel buffers were neither emitted nor copied into the release.
No screenshot capture or dataset uploads were used in this update.

Cache limits remain128 RGB thumbnails,128 Tk photo variants and128 failures, with
two decode workers. This is CPU/Pillow/Tk presentation work; no new CUDA training
or prediction tensors were introduced. Shared M54 still uses its existing OpenGL
renderer. Per-process OpenGL VRAM is not instrumented under WDDM. Solver/matching
GPU behaviour and source geometry are unchanged by these interface corrections.

## Main release and backup

The main VERSION.txt is reconciled with the last published isolated release
0.1.5.4, then version_manager.py --bump publishes v0.1.5.5. C# Suite/versioned
launchers are recompiled. All active source, headers, docs/tests, executables,
walkthrough and Abu_Align copies match both full v0.1.5.5 snapshots. The completed
shared work is committed on codex/manual-ui-responsiveness, with clean main
Git status. Earlier isolated snapshots are preserved in their attached worktree.
No upstream repository source or user dataset was changed. Read the complete
file inventory in [MANUAL_UI_RELEASE_FILES.md](MANUAL_UI_RELEASE_FILES.md).
