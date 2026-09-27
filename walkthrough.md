# v0.0.9.2 — Viewer comparison and photo inspection fixes

## Findings and changes
Comparison appeared inactive because selecting a comparison run left A-only visibility
unchanged; opening the native Compare panel also left B hidden. Selecting B now
activates Overlay and fits both reconstructions. The browser has an explicit
Compare A and B action. Refresh avoids assigning the same run to A and B when
another result exists. Display-only changes retain the existing viewpoint.

Camera matching now prioritizes exact relative paths and then uniquely occurring
basenames across export prefixes. Ambiguous basenames are never paired. This allows
camera-center residuals and optional similarity registration across common engine
exports. Both point clouds and camera trajectories retain A mint / B coral colors.
Trajectories connect file-order camera centers, not inferred capture chronology.

The browser point-size slider imposed a 0.001 minimum and step. It is now a numeric
field with 0.000001 minimum/step, preserving 0.0001 and smaller supported values.
The native raster viewer still uses pixel-sized points; its units differ.

Browser photos use one replaceable, concise in-browser preview without also opening
Windows Photos. Automatic preview is opt-in. Frustum thumbnail visibility now respects
the toggle even for cached images. Explicitly selected photos are still available.
Native photo inspection fixes the missing BORDER import that prevented construction,
resolves B images from the B run, uses concise filenames, a separate metadata line,
simplified actions, EXIF orientation, and padded fit without enlarging small images.

## Verification
48 tests passed in 37.716 seconds across test_viewer_workspace, test_native_inspection,
test_fast_viewer, test_desktop_ui, and test_gui_presets. After final control refinements,
22 viewer/native regression tests passed in 5.509 seconds.
Run: python -m unittest discover -s tests -p test_viewer_workspace.py
and python -m unittest discover -s tests -p test_native_inspection.py.
Tests instantiate real Viser APIs and Tk windows with synthetic reconstructions.
They cover unique-name fallback, ambiguity rejection, sub-0.001 size retention,
comparison activation, registration, photo dialog construction, and no external
photo application launch. User production datasets were not available for validation.
No claim of a manual browser visual review in this pass.

## Memory and limits
No alignment/training or GPU allocation was requested. VRAM and production FPS were
not benchmarked. Existing two-run preview caches and sample limits remain in effect.
Photo handles close after loading. Comparisons measure sampled agreement in scene
units, not ground-truth accuracy. Registration needs three noncollinear matched
cameras; ambiguous or insufficient matches produce an explanatory error.

## Release
Modified implementation: viewer_workspace.py, native_inspection.py.
Regression tests: tests/test_viewer_workspace.py, tests/test_native_inspection.py.
Version headers, VERSION.txt, launchers, and release documentation are synchronized.
Canonical consolidated backup: versions/v0.0.9.2/.
