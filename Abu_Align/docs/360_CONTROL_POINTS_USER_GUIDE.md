# 360° Control Points — user guide

Open **Prepare input ▾ → 360° manual control points** in Abu Align & Train.

360° CONTROL POINTS — WORKFLOW

1. OPEN AN ALIGNMENT
Choose Open alignment and select an OpenSfM, neural 360DVO or native spherical COLMAP reconstruction.json. Choose the matching original panorama folder. Work on one reconstruction component; image dimensions must match its cameras.

2. CREATE AND MEASURE
Add point (Ctrl+N) creates a named tie point immediately. In Place points mode, click the same physical feature in both images. Use the image selectors or Previous/Next buttons to add more views. Aim for 3–6 well-separated views with translation and clear parallax. Distribute control points throughout the scene.

3. NAVIGATE PRECISELY
Wheel: zoom around the cursor. Middle-drag or Space + left-drag: pan. Fit / F: entire panorama. 1:1 / 1: original pixel size. Drag an existing marker to adjust it. Arrows nudge the selected measurement by 1 px; Ctrl+Arrow by 0.1 px; Shift+Arrow by 10 px. Right-click a marker removes its measurement. Ctrl+Z / Ctrl+Y undo and redo.

Wrap panorama seam allows close-up panning across the left/right boundary. Latitude stays bounded. Pixel display uses nearest-pixel rendering for precise inspection. Marker coordinates always refer to original image pixels regardless of display zoom. Both image panes navigate independently; linked zoom synchronizes pixel scale only.

4. REVIEW POINTS AND MEASUREMENTS
Search or filter the Points list. Yellow rows need at least two measurements. Select a point to see its type, properties, every measured image, pixel coordinates, uncertainty and angular residual. Double-click a measurement to open and focus it in the active image pane, or choose View left/right. Disabled points retain their measurements but are excluded from refinement. Distances attached to disabled points are also inactive and are labelled in the distance list.

5. SET POINT PROPERTIES
Tie point: the same feature in different images; no surveyed coordinates required.
Ground control: surveyed local X/Y/Z in metres. At least three non-collinear ground controls establish the world frame.
Check point: surveyed local coordinates for independent evaluation only; never optimized. Checks still require two or more image measurements to triangulate.

Apply properties validates the name, type, coordinates, enabled state and survey uncertainty. Smaller uncertainty gives a stronger constraint: enter realistic precision. Image uncertainty is in original pixels, survey uncertainty is in metres. Each measurement can have its own image uncertainty.

6. ADD A KNOWN DISTANCE (OPTIONAL)
In Distances, select two different enabled, measured tie/ground-control points. Enter the independently measured distance and uncertainty in metres, then Add distance. A distance establishes scale. Check points cannot be distance endpoints because they are not optimized.

7. SAVE AND REFINE
Ctrl+S saves portable control_points.json. Set the automatic-landmark and solver-evaluation budgets in Settings, then Refine alignment. Manual optimization points are always retained. Larger budgets need more CPU time and RAM. Refinement runs in the background with edits locked for a consistent snapshot. It writes a new reconstruction, the exact controls used and a report. Source photos and reconstruction remain unchanged.

8. INSPECT RESULTS
Results reports angular RMS before/after, solver convergence, independent check errors and output paths. Reaching the evaluation limit is not convergence. Reduced fitted residual is not proof of absolute accuracy. Open result inspects the new alignment with the same controls; it does not automatically replace or export the application's training package.

PREDICTIONS ARE GUIDANCE, NOT MEASUREMENTS
A dashed blue marker predicts the selected point using at least two consistent measurements and the current alignment. It is never saved automatically. Find the actual feature and click it yourself. Predictions are suppressed when triangulation is unreliable or residuals are high. Avoid repeated patterns, reflections, moving objects, sky and stitching artefacts.

PRIVACY AND LIMITS
All images and controls remain on this PC. No uploads or cloud services are used. Image navigation uses CPU memory; spherical joint refinement uses the CPU solver. Geographic latitude/longitude conversion, disconnected-component merging and automatic matching are not supplied by this editor.

SHORTCUTS
Ctrl+O  Open alignment     Ctrl+N  Add point     Ctrl+S  Save controls
Ctrl+Z  Undo               Ctrl+Y  Redo
F       Fit image         1       Original pixel scale
Space + drag / middle drag  Pan
Arrows  1 px    Ctrl+Arrows 0.1 px    Shift+Arrows 10 px


## Local verification

20 targeted workflow, geometry, manifest and main-app menu tests passed. These cover marker nudging, undo/redo, point renaming with distance endpoints, asynchronous navigation, viewport memory bounds, settings forwarding, immutable source files, dirty-close cancellation, minimum-size scrolling, hover-popup cleanup and exclusion of disabled controls/distances from the solver.

The authorized local 7680×3840 panoramas stayed on this PC. Two original images plus their mip levels used 235,699,200 bytes; measured process RSS was 395,419,648 bytes (about 377 MiB). Background loading of both images took 0.692 s. For 60 renders of an 800×600 viewport, median CPU render time was 8.362 ms and the measured 95th-percentile sample was 17.624 ms. This microbenchmark excludes event-loop/display latency; it is not a guaranteed FPS measurement. No CUDA allocations or training jobs are introduced by the editor. Synthetic-only screenshots were used for visual review at 1580×940 and 1100×720; no private panorama screenshots were transmitted.

## Design reference

The point/measurement/inspector workflow was informed by the official [RealityScan control-point guide](https://rshelp.capturingreality.com/en-US/tools/controlpoints.htm) and [measurement-placement guide](https://rshelp.capturingreality.com/en-US/tools/createcontrolpoints.htm). This editor provides its own native spherical workflow; it does not claim full RealityScan feature parity.
