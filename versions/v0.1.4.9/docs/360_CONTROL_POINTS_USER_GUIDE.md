# Manual 360 Control Points

The Help tab includes a searchable illustrated guide and a full reference. M54 is the shared 3D viewer; see ABU_M54.md for its standalone and browser workflow.

360° CONTROL POINTS — WORKFLOW

1. CHOOSE YOUR STARTING POINT
Before alignment: Open panoramas loads original full 2:1 equirectangular images. You can create, import, measure and save points without camera poses. Predictions, residuals, 3D and Refine become available after opening an alignment for those same images; prepared measurements are retained.
After alignment: Open alignment selects an OpenSfM, neural 360DVO or native spherical COLMAP reconstruction.json and its original panorama folder. Work on one reconstruction component; image dimensions must match its cameras.

2. CREATE AND MEASURE
Add point (Ctrl+N) creates a named tie point immediately. In Place points mode, click the same physical feature in both images. Use the image selectors or Previous/Next buttons to add more views. Aim for 3–6 well-separated views with translation and clear parallax. Distribute control points throughout the scene.

3. NAVIGATE PRECISELY
Wheel: zoom around the cursor. Middle-drag or Space + left-drag: pan. Fit / F: entire panorama. 1:1 / 1: original pixel size. Drag an existing marker to adjust it. Left/Right browse images in the active pane; Alt+Left/Right also browse. Ctrl+Arrow nudges 0.1 px; Shift+Arrow nudges 10 px. Disable arrow browsing in Settings for plain-arrow 1 px nudging. Text fields retain normal typing keys. Right-click a marker removes its measurement. Ctrl+Z / Ctrl+Y undo and redo.

Image grid shows all images on one continuous scrolling page; thumbnails load as you scroll with a bounded cache. Search by filename and filter images with or without a measurement of the selected point. Choose the Left or Right destination, then double-click a thumbnail or Open selected. The grid stays open above the editor, retaining its filter, selection and scroll position. Measured markers are yellow; predicted guidance is blue. A Predicted filter shows unmeasured candidate images.

Wrap panorama seam allows close-up panning across the left/right boundary. Latitude stays bounded. Pixel display uses nearest-pixel rendering for precise inspection. Marker coordinates always refer to original image pixels regardless of display zoom. Both image panes navigate independently; linked zoom synchronizes pixel scale only.

4. REVIEW POINTS AND MEASUREMENTS
Drag the divider above Measurements upward to make the table taller. The form above it scrolls independently. Measurement actions opens left/right views, predictions, uncertainty and removal; double-click still opens a measured image. Dividers resize live, and normal image quality returns after a short lightweight resize preview.
Search or filter the Points list. Yellow rows need at least two measurements. Select a point to see its type, properties, every measured image, pixel coordinates, uncertainty and angular residual. Double-click a measurement to open and focus it in the active image pane, or choose View left/right. Disabled points retain their measurements but are excluded from refinement. Distances attached to disabled points are also inactive and are labelled in the distance list.

5. SET POINT PROPERTIES
Tie point: the same feature in different images; no surveyed coordinates required.
Ground control: surveyed local X/Y/Z in metres. At least three non-collinear ground controls establish the world frame.
Check point: surveyed local coordinates for independent evaluation only; never optimized. Checks still require two or more image measurements to triangulate.

Apply properties validates the name, type, coordinates, enabled state and survey uncertainty. Smaller uncertainty gives a stronger constraint: enter realistic precision. Image uncertainty is in original pixels, survey uncertainty is in metres. Each measurement can have its own image uncertainty.

6. ADD A KNOWN DISTANCE (OPTIONAL)
In Distances, select two different enabled, measured tie/ground-control points. Enter the independently measured distance and uncertainty in metres, then Add distance. A distance establishes scale. Check points cannot be distance endpoints because they are not optimized.

7. IMPORT REALITYSCAN CSV (OPTIONAL)
Import CSV opens a preview with column mapping before any edits are applied. Image measurements use image path, point name, X pixel, Y pixel, optionally image uncertainty. CSV, semicolon, tab and quoted space-separated files are supported. Basenames must uniquely match the original panoramas, with the same projection and pixel dimensions. Never transfer perspective, cropped or resized image pixels onto a panorama. Use Point names only for such exports and re-mark their features on the panoramas.
Local survey XYZ uses point name, X, Y, Z and optionally survey uncertainty. Choose Ground control or Check point and the source units. Geographic latitude/longitude requires conversion outside this tool. Keep existing values is the default; Replace existing values is explicit. One import is one undo step; review unmatched rows and errors before applying.

8. SAVE AND CHOOSE AN ALIGNMENT ACTION
Ctrl+S saves portable control_points.json. Set the automatic-landmark and solver-evaluation budgets in Settings, then Refine alignment. Manual optimization points are always retained. Larger budgets need more CPU time and RAM. Refinement runs in the background with edits locked for a consistent snapshot. It writes a new reconstruction, the exact controls used and a report. Source photos and reconstruction remain unchanged.
Refine alignment operates on the alignment currently open in this editor. To run the full pipeline, Use with main Align saves and attaches the controls to the main window. There, keep Apply manual 360 controls after initial alignment checked, choose the desired supported engine, and press the main Align button. Initial registration runs first, followed by spherical control refinement and exports using the refined poses. The CLI validates selected images and captures an immutable copy of the controls before inference. Controls are not initialization seeds and do not merge disconnected components. Attaching controls does not start alignment automatically.

PROCESSING HARDWARE AND EXPORTS
Auto uses CUDA for larger problems when available; CUDA forces GPU processing; CPU uses vectorized analytic derivatives. Default solver budget is 100. A separate worker keeps the editor responsive. Progress shows loading, optimization, retriangulation and export. Cancel refinement stops the worker and preserves input files. An iteration limit is not convergence.
Every successful refinement writes <name>.points.ply, <name>.cameras.csv and <name>.exports.json beside the JSON. The PLY contains refined sparse landmarks. The CSV stores exact native-frame centres, camera-to-world quaternions, world-to-camera rotations/translations and spherical dimensions; it is not a RealityScan measurement CSV. Scale remains arbitrary unless established by metric constraints. To export an older result, open its refined JSON, then Results > Export cloud & camera CSV.

9. INSPECT RESULTS AND 3D
Results reports angular RMS before/after, solver convergence, independent check errors and output paths. Reaching the evaluation limit is not convergence. Reduced fitted residual is not proof of absolute accuracy. Open result inspects the new alignment with the same controls; it does not automatically replace or export the application's training package.
3D view opens Abu 3D Viewer M54, the shared GPU viewer also used by the main app and standalone Abu_M54.exe. Drag to orbit, right/middle drag to pan, wheel to zoom, F to fit. Click a control, camera or cloud sample; Orbit selected, O or double-click sets its orbit pivot. M54 tools includes world up, ground grid/plane/elevation, shading, standard views, bookmarks and reversible move/rotate/uniform-scale previews. Export transformed copy reads the full source and saves a new model/cloud/camera CSV; separate surveyed control coordinates are not transformed. The Workspace tab returns to the images. Compare reconstructions opens a collapsible side panel. A is the editable alignment; Load B or Run folder loads a read-only comparison in a separate process. Show A only, B only or Overlay; Switch A/B preserves the viewpoint. Camera offsets and per-camera differences show agreement. Fit B to A optionally fits matching non-collinear cameras; Allow scale selects similarity versus rigid fitting. Source files and controls remain unchanged. B distance to A colours directed distances to sampled A points; density affects the result. These metrics are not independent accuracy. Save comparison report preserves the chosen frame transform and differences.
Browser view opens the optional local WebGL viewer for A. Public sharing is disabled. Raw survey XYZ is not guessed into an unregistered frame. The display samples at most 50,000 cloud points; optimization data remains complete. This inspects the open alignment, not a live preview of an unfinished solve.

PREDICTIONS ARE GUIDANCE, NOT MEASUREMENTS
A dashed blue marker predicts the selected point using at least two consistent measurements and the current alignment. It is never saved automatically. Find the actual feature and click it yourself. Predictions are suppressed when triangulation is unreliable or residuals are high. Avoid repeated patterns, reflections, moving objects, sky and stitching artefacts.

PREDICTED IMAGE LIST
Properties > Measurement actions > Predicted images lists unmeasured registered images and predicted pixels, ranked near the views supporting triangulation. Select an image and Open & focus prediction, or show candidates in the grid. Occlusion and actual visibility are unknown. After measuring an image it leaves the unmeasured list. Predictions never become saved measurements automatically.

TROUBLESHOOTING
Results > Open troubleshooting log opens the local session JSONL file. It records point/measurement/property/distance edits, image and CSV selections, settings, saved paths, solver progress, device/timing/memory, outputs, cancellation and full error traces. Image contents are not logged. Logs normally live in %LOCALAPPDATA%/Abu_Align/logs, with a temporary-directory fallback if needed.

PRIVACY AND LIMITS
All images and controls remain on this PC. No uploads or cloud services are used. Image navigation uses CPU memory; joint refinement uses CUDA sparse linear solves or the analytic CPU path. Native panorama measurements require no cubemaps. Geographic latitude/longitude conversion, disconnected-component merging and automatic matching are not supplied by this editor.

SHORTCUTS
Ctrl+O  Open alignment     Ctrl+N  Add point     Ctrl+S  Save controls
Ctrl+Z  Undo               Ctrl+Y  Redo
F       Fit image         1       Original pixel scale
Space + drag / middle drag  Pan
Left/Right  Browse images    Alt+Left/Right  Browse images
Ctrl+Arrows 0.1 px nudge     Shift+Arrows 10 px nudge
Plain arrows 1 px nudge when arrow browsing is disabled

[RealityScan CSV format](https://rshelp.capturingreality.com/en-US/tools/cpmeasurementsexport.htm)
