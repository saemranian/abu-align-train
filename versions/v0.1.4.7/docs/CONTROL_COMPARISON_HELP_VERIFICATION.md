# Persistent image grid, reconstruction comparison and illustrated Help

## Changes and rationale

Double-clicking a thumbnail previously raised the large editor above the grid. The
grid still existed, but appeared to have closed. It is now an owned utility window:
opening an image keeps the grid above the editor and preserves its destination
pane, selection, geometry, filter and scroll position. The grid closes through its
own close button or when its parent editor closes.

The embedded 3D navigator now has a collapsible **Compare reconstructions** sidebar.
The alignment open in the control editor is **A** (mint); a separately loaded
reconstruction is **B** (coral). B is read-only. The sidebar occupies width instead
of taking navigation height, with scrolling settings and permanently reachable
load, switch, fit and report actions. It supports:

- A only, B only and Overlay, with a stable viewpoint when switching layers.
- Single-component OpenSfM/native spherical reconstruction JSON, NeRF/OpenGL
  transforms JSON, sparse PLY and completed run folders using the main workspace's
  existing camera/asset readers, including COLMAP packages.
- Original RGB, layer colours and directed sampled B-to-A point-distance colours,
  with a zero-to-P95 legend. Empty clouds give an explicit unavailable message.
- Camera-centre offset lines, position RMS, orientation RMS and individual camera
  differences. Selecting a B camera opens an A image only when the names match
  uniquely. Cloud-only PLY files have no cameras to match.
- Optional camera-based B-to-A frame fitting. Raw coordinates are the default.
  Allow scale selects similarity fitting; disabling it preserves scale with rigid
  fitting. At least three non-collinear uniquely matched centres are required.
  Invalid/ambiguous fitting explicitly falls back to raw B. The transform affects
  the preview only and is included in the saved JSON comparison report.

Camera names match exactly or by a unique basename using the same matching logic
as the main 3D workspace. No index-based image correspondence is invented. A
multi-component OpenSfM JSON is rejected rather than overlaying disconnected
coordinate frames. Direct OpenSfM run folders use this same safeguard. Source
reconstructions, control measurements and training packages are not modified.

The Help tab now opens a searchable **Illustrated guide** with seven task topics:
Start here, Images/predictions, Navigation, Point types/CSV, Refinement/exports,
3D/comparison and Troubleshooting. Each task has locally drawn diagrams, icons and
buttons opening the relevant tool/settings. The **Full reference** tab retains the
detailed reference. Cards adapt to one or two columns and scroll at smaller sizes.

## Interpretation and limits

Camera and cloud differences measure agreement between the selected reconstructions,
not surveyed accuracy. Fitting can hide coordinate/scale differences; keep it off
when those differences are the subject of inspection. Camera centres used to fit
the preview are also used in its agreement report. Cloud distances are directed
nearest-point distances to a bounded sample, not point-to-surface accuracy, and
depend on density and frame fitting. P95 is the measured value; a small numerical
floor is used only when dividing for colour mapping.

A controls remain editable and selectable; B controls are not imported or merged.
The optional browser viewer shows the active A alignment. This release does not
add ICP, component merging or an independent survey benchmark.

## Responsiveness and memory

B parsing runs in an isolated child process. A single background comparison worker
handles numeric preparation; generation checks discard stale results. Clear/close
terminates this tool's loader and cancels queued work, and temporary files are
cleaned. A control-only refresh reuses the comparison when A geometry is unchanged.
All retained/displayed clouds are bounded to 50,000 points per model. At that bound,
one float64 XYZ plus uint8 RGB array pair uses 1,350,000 bytes (about 1.29 MiB),
excluding cameras, preview copies, nearest-neighbour structures and renderer arrays.
The parser can use more memory temporarily while reading a complete JSON or PLY.
Peak host RAM and additional viewer VRAM were not instrumented in this release.

The existing native WGL/OpenGL navigator continues to render on the local GPU.
Loading and agreement calculation use the CPU outside the GUI event handler;
this release adds no GPU solver or training tensors. Help diagrams are generated
locally with a small bounded cache, released when Help is destroyed. No upstream
repositories or installed packages are changed.

## Verification

The regression suite covers persistent grid stacking and state, independent known
similarity/rigid transforms, immutable source arrays/files/control documents,
ambiguous fits, bounded loading, archive round-trips, actual child-process loading,
overlay/switching, camera selection, report saving, clearing and Help search/actions.
Format checks independently verify transforms/OpenGL conversion, COLMAP camera
centres/orientation, PLY loading, disconnected-folder rejection and absent-cloud
distance reporting. Sidebar layout checks require the viewport to retain the full
available 3D body height, rather than relying on a particular monitor's DPI.

Reproduction in the local `vggt` environment from the project root:

```python
import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
# If required by a restricted environment, set tempfile.tempdir to its writable
# local test-artifact directory before running the suite.
modules = ['test_control_points_workspace', 'test_control_import_navigation',
           'test_control_comparison_help', 'test_control_acceleration',
           'test_spherical_refinement', 'test_spherical_integrity', 'test_desktop_ui']
result = unittest.TextTestRunner().run(
    unittest.defaultTestLoader.loadTestsFromNames(modules))
assert result.wasSuccessful()
```

Final regression: **60 tests passed in 53.760 seconds**, including the independent
multi-format loader check. An earlier full run passed 59 tests in 56.867 seconds.
Locally generated Help illustrations were visually
reviewed. Automated Tk tests instantiate and exercise the native 3D interface;
a full desktop screenshot was not captured during this change.

Read-only local model check: two existing 593-camera reconstructions, each sampled
to 50,000 cloud points, loaded in approximately 6.8 seconds combined; fitted
comparison including sampled cloud distances took approximately 0.17 seconds.
All 593 camera names matched and reported values were finite. This is one local
runtime check, not a guaranteed latency or independent accuracy benchmark. No
panorama pixels were decoded or uploaded for this check. UI tests use synthetic
images/geometry, and all private data and temporary outputs remain outside Git.

The completed concurrent GUI, rig-designer and splitter changes released by the
other agent in v0.1.4.6 are preserved in the new release snapshot. They were not
authored or performance-benchmarked as part of this control-point change.

## Design references

RealityScan's [3D view/navigation guide](https://rshelp.capturingreality.com/en-US/tutorials/quickstart_3.htm)
and [control-point workflow](https://rshelp.capturingreality.com/en-US/tools/createcontrolpoints.htm)
informed the image/3D selection and task-oriented help design. Its
[component documentation](https://rshelp.capturingreality.com/en-US/appbasics/components.htm)
reinforces the need to distinguish disconnected reconstruction frames.
CloudCompare's [registration documentation](https://www.cloudcompare.org/doc/wiki/index.php/ICP)
is a useful reference for interpreting fitted cloud comparisons. This implementation
uses matched camera centres rather than ICP.
