# Manual control points / M54 verification — 2026-10-13

The final combined acceptance suite passed **113 tests in 71.901
seconds, exit code 0**. It covers the existing control/CSV/navigation/comparison,
spherical CPU/CUDA refinement, desktop UI, shared native/browser M54 and main
inspection modules, plus `test_manual_cameras`. A final solver-focused run passed
25 tests in 14.455 seconds after preserving disabled legacy controls. A 51-test
focused integration run also passed before that validation adjustment.

Independent new checks cover distorted pinhole and fisheye ray/projection
round-trips, pixel-noise whitening, analytic Jacobians against finite differences,
camera/track/calibration COLMAP round-trips, perspective refinement from an already
refined result, CPU/CUDA held-out checks, original immutability, unsupported lenses,
wrong control projections/dimensions, untracked point exclusion, insufficient
constraints, live numeric/drag/Undo/invalid values and perspective seam/predictions.
The renamed menu/title tests follow the requested UI name. The responsive-worker
fixture now has valid multi-view tracks; the old one-point fixture was underconstrained
and must not be reported as a meaningful camera refinement.

Browser UI checks on a loopback-only synthetic server verified actual left-drag
camera orbit, live scale updates without Apply and automatic exit from one-click
picking. Initial position/up/target ordering was also corrected and the connected
camera target verified. The installed Viser 0.2.23 client explicitly disables
camera drag when scene pointer mode is enabled. The app now attaches that callback
only during picking. No installed Viser or upstream source was modified.

## Small synthetic perspective probe

Four calibrated distorted cameras, 24 source landmarks, three ground controls and
one held-out check; the solve uses 23 landmarks/92 image observations. Camera
intrinsics and measurements are exact synthetic values, with perturbed initial
poses. This is a numerical integrity fixture, **not realistic survey accuracy or a
benchmark on the client's panoramas**.

| Path | Whole refinement + exports | Independent check error | Angular RMS after | Solver peak allocated VRAM |
|---|---:|---:|---:|---:|
| CPU | 0.230 s | 7.7e-11 scene units | 1.35e-10° | 0 MiB |
| RTX 4090 CUDA | 2.637 s | 1.47e-08 scene units | 1.19e-08° | 8.256 MiB |

Before check error: 0.00854941 scene units; initial
angular RMS: 0.137855°. Both paths converged.
The GPU run includes 2.122 s setup and
0.502 s solver time. Small jobs need not be faster
on CUDA; Auto keeps small solves on analytic CPU and selects CUDA for larger jobs.
This probe does not measure a fresh VGGT inference run or end-to-end client accuracy.

## Memory and exports

The CUDA figure is the solver's additional peak allocated PyTorch VRAM, not total
driver/OpenGL/device memory. Sparse normal-equation operations avoid a dense Hessian.
Each precomputed 3×3 float64 observation whitening matrix occupies 72 bytes, before
residual/Jacobian/index structures. Total process RSS and new OpenGL VRAM were not
instrumented in this release; prior M54 synthetic viewer measurements remain in
`ABU_M54_VERIFICATION.md`. No image tensor or upstream neural training state was added.
Image caches, display clouds, histories and live render requests retain their bounds.

Full calibrated inputs retain original intrinsics and genuine observations. Changed
poses cannot correct untracked cloud centres: they are excluded from the corrected
sparse cloud, while their original file stays immutable. Unobserved cameras stay
at initialized poses. Necessary constraint/connectivity checks are not a proof
against geometric degeneracy; distributed controls and independent checks matter.
Refinement PLY uses float64 XYZ. Perspective COLMAP exports preserve calibration,
exact names/poses and valid sparse tracks. Saved controls include settings and
projection/dimension metadata; each worker has an explicit settings snapshot.

## Reproduce and release

Use the existing local vggt Python environment. Run unittest with the tests folder
on sys.path, covering `test_manual_cameras`, `test_control_points_workspace`,
`test_control_import_navigation`, `test_control_comparison_help`,
`test_control_acceleration`, `test_spherical_refinement`, `test_spherical_integrity`,
`test_desktop_ui`, `test_abu_m54`, `test_native_inspection`, `test_viewer_workspace`.
Native checks require the local Windows OpenGL driver; browser tests bind loopback.
All fixtures are synthetic. No client images were uploaded or added to Git.

The v0.1.5.1 release uses the attached isolated checkout, building on the M54 v0.1.4.9
commit. The observed canonical v0.1.5.0 release is reserved for the active other
agent, so this release advances from that reserved number without importing their
unfinished GUI/splitter/rig edits. Clean Git and byte-identical root/mirror/snapshot
verification apply to the isolated checkout. The canonical feature files are
available during co-development; its active release copies are not this verified
backup. No public deployment or remote data processing was performed.
