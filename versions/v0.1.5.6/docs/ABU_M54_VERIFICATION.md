# Abu M54 verification — 2026-10-13

## UI root causes and corrections

The horizontal splitters used deferred resize behavior. Configure callbacks also
repacked an unchanged hint, and image scaling used normal interpolation throughout
dragging. Splitters now update live, layout changes occur only when visibility
changes, render requests coalesce and resizing uses a lighter mip/nearest preview.
Normal interpolation returns after 100 ms without resizing. This changes display
quality during interaction, not saved image coordinates or measurements.

The measurement table was inside the same scrolling property form with a small
fixed requested height. Properties and Measurements now occupy separate vertical
panes. The form scrolls independently and its divider gives the table more height.
Measurement actions are in a compact menu, keeping space available at minimum size
and under combined scaling/theme settings. M54 navigation bars were similarly
condensed; additional controls live in the modeless Scene tools window.

## Shared engine and integrity checks

Both embedded viewers now instantiate `M54NativeViewer`. Main/browser/control
adapters share float64 scene math, camera conventions, positive uniform transforms,
selection/pivot rules, grid generation and full-source copy export. Independent
synthetic tests verify:

- camera-relative point vectors change only by uniform scale when the whole model
  is translated/rotated/scaled, and source arrays/controls remain unchanged;
- native JSON, exported double-precision PLY and exact camera CSV agree;
- camera-bearing/orientation round-trips, including fractional transforms from an
  integer-valued transforms JSON;
- preservation of small coordinate differences near 1e9 in PLY/COLMAP data;
- explicit RealityScan-style yaw/pitch/roll parsing instead of identity orientations;
- rejection of measurement CSVs, ambiguous components, invalid scale, truncated
  binary tracks and source/output collisions;
- depth-aware cloud picking, grid planes, control/camera/cloud orbit pivots, native
  history, grid settings, actual GPU context, standalone isolated loading and
  browser selection/transform/gizmo callbacks with actual installed Viser APIs.

Source arrays stay float64 until normalized display buffers are made. Browser A/B
layers use the same A display frame. Pose CSV orientation conventions are explicit;
unrecognized conventions are not inferred. Full exports read source data rather
than writing a truncated display sample. Native exported PLY copies also retain
double precision.

Background loader/comparison/export jobs capture data, queues and plain process
state rather than retaining Tk window owners. Generation/cancellation checks
discard stale work. The installed Viser 0.2.23 stop method only signals its event
loop; its daemon prints a final message afterward. A combined run passed every
assertion but exposed a buffered-stdout shutdown race at Python exit. The M54
server adapter now joins that exact owned server thread for up to three seconds;
repeated stops are safe. No installed Viser code is changed. Rechecked viewer
tests passed with a clean process exit.

The complete acceptance suite passed **99 tests in 71.041 seconds with exit code 0**
across ten modules, after the final shutdown correction. The final focused viewer/export/lifecycle
suite passed **39 tests in 14.826 seconds with exit code 0**. The earlier complete
run passed all 99 assertions in 77.345 seconds before the Viser shutdown correction.

## Reproduce

Use the local `vggt` Python environment from the workspace root. A writable local
temporary directory may be set via `tempfile.tempdir` when running in a restricted
test environment. Loopback socket tests need permission to bind local sockets.
The native renderer tests require a Windows OpenGL driver.

```powershell
python -B -m unittest discover -s tests -p test_abu_m54.py
python -B -m unittest discover -s tests -p test_native_inspection.py
python -B -m unittest discover -s tests -p test_viewer_workspace.py
python -B benchmark_m54.py --output C:\local_test_artifacts\m54_benchmark.json
```

Full acceptance modules: `test_control_points_workspace`,
`test_control_import_navigation`, `test_control_comparison_help`,
`test_control_acceleration`, `test_spherical_refinement`, `test_spherical_integrity`,
`test_desktop_ui`, `test_abu_m54`, `test_native_inspection`, `test_viewer_workspace`.
Run them with `unittest.defaultTestLoader.loadTestsFromNames` after adding the tests
directory to `sys.path`, or use equivalent module-specific discovery commands.

## Measured local performance and memory

Synthetic benchmark on the local **NVIDIA RTX 4090**, OpenGL 4.6, driver 610.62:

| Workload | Median | P95 |
|---|---:|---:|
| 50,000 points + 593 cameras/frustums + trajectory + grid, 60 native GPU frames | 16.58 ms | 18.26 ms |
| 5760 × 2880 synthetic panorama, 40 normal-quality viewport draws | 17.89 ms | 31.51 ms |
| Same panorama, 40 resize-preview draws | 7.01 ms | 12.65 ms |

GPU frames explicitly include `glFinish` to wait for completed rendering. Swap/
display synchronization can influence their timing. Panorama timings exclude
event dispatch. The preview's measured median was about 2.55 times faster on this
synthetic image; these are not guaranteed frame rates or client-scene benchmarks.

Observed process RSS maximum during that benchmark: **269,500,416 bytes**, about
257 MiB. The panorama's RGB mip assets used **66,096,000 bytes**. A 50,000-point
float64 XYZ + uint8 RGB pair uses 1,350,000 bytes, before normalized buffers,
preview copies, cameras and search structures. Camera wirefrustums are vectorized
and cached. Retained clouds, thumbnail caches and native transform history are
bounded; GPU contexts, caches, local loader processes and browser servers close
with their host. Peak additional OpenGL VRAM was not instrumented; no CUDA training
tensors are introduced by this viewer change. Parsing and full-copy exports can
temporarily require more RAM than a display sample.

## Privacy, scope and release

All checks use synthetic image/geometry fixtures. No client dataset is uploaded,
modified or included in Git. Browser servers bind loopback and deny public share
requests before serving private geometry. Windows Open with registration points
to the locally compiled `Abu_M54.exe` and preserves existing default applications.

Research references and the precise supported formats/limits are in
[ABU_M54.md](ABU_M54.md). This validates the implemented viewer features; it is not
an absolute reconstruction-accuracy benchmark or a claim to implement every CAD,
mesh, Gaussian renderer or point-cloud interchange format. Completed v0.1.4.8
splitter/UI changes from the other agent are preserved in this release.
