# Abu 3D Viewer M54 (Abu M54)

Abu M54 is the shared point-cloud and camera-reconstruction viewer in Abu Align &
Train. M54 is its development name pending the future combination with other Abu
3D tools. The embedded main workspace, manual 360-control workspace and standalone
application use the same native GPU renderer. Browser surfaces share the scene,
file, transformation and navigation rules through a Viser adapter.

## Open and navigate

- Main app: **04 3D Workspace · Abu M54**, then **M54 tools**.
- Manual controls: **3D view**, then **M54 tools**. The Workspace tab returns to images.
- Standalone: run `Abu_M54.exe`, then Open file or Run folder. Files also open through
  Windows **Open with → Abu 3D Viewer M54**. The registration is per-user and does
  not replace default applications. The standalone Windows Open with button can
  repair registration after moving the installation.
- Browser: use Browser in the control/standalone viewer, the main app's existing
  browser action, or `python abu_m54_app.py --browser path/to/model.json`.

Drag to orbit, middle/right-drag to pan and wheel to zoom. Click a control, camera
or displayed cloud sample to select it. **Orbit selected**, **O** or double-click
makes the selection the orbit pivot. **Fit** returns to a scene-centred view. Native
M54 tools optionally enables orbit-on-selection; browsers offer the same option
and an explicit Pick cloud sample mode. Picking a cloud sample never creates a
manual control measurement. It selects an existing display sample, not an arbitrary
surface intersection. Camera optics/frustums are illustrative, not calibrated ERP
perspective lenses.

M54 tools includes world-up selection, ground grid, XY/XZ/YZ plane, elevation in source scene units,
native point shading and size, camera size/trajectory, standard views and view
bookmarks. Existing main-viewer profiles, camera inspector, look-through, photo
inspection and A/B comparison remain available. In the control workspace,
**Compare reconstructions** retains its read-only B workflow and comparison report.

## Transform and export

Select A or B in scene tools, enter move XYZ in source units, rotate XYZ in degrees
and positive uniform scale, then **Apply preview**. Choose scene centre, a selected
item or world origin as the rotation/scale pivot. The browser also provides a move
and rotate gizmo; scale remains explicit numerically. Transformations move cloud
points, camera centres/orientations and triangulated control markers together.
Native history survives closing/reopening the tools window. Undo, Redo and Reset
affect the preview, and opening a different A model clears the previous model's
history. A manual B transformation disables automatic frame fitting so it is not
immediately fitted away.

**Export transformed copy** reads full source geometry, independently of the display
sample. It writes a new reconstruction JSON when appropriate, cloud PLY, exact
camera CSV and a `.m54.json` transform/provenance record. Generic point sources
export PLY/CSV. A new basename is required; existing targets/companions and sources
are protected. Transforms JSON camera rotations retain orthonormal optics; scale
changes camera positions and points, not the rotation matrix. Original image files
are not copied or altered. Source-relative image paths are rebased where they can
be resolved when exporting transforms JSON elsewhere.

Transforming a reconstruction is a coordinate/model operation, not bundle adjustment
or proof of improved accuracy. It does not transform surveyed coordinates in a
separate control file or regenerate existing training packages. Keep survey/control
frames consistent before refining an exported copy. Camera/cloud comparison values
measure agreement; fitting can remove an origin/rotation/scale difference you might
want to inspect. Keep fitting off when evaluating that difference.

## File support

| Input | Displayed data |
|---|---|
| OpenSfM/native spherical reconstruction JSON | Single component's points and exact poses |
| NeRF/Nerfstudio transforms JSON | Camera-to-world poses; OpenGL-to-OpenCV camera-axis conversion; companion cloud |
| PLY, binary or ASCII | XYZ and RGB when available; double coordinates preserved; companion camera CSV |
| Abu camera CSV | Named centres and explicit scalar-first camera-to-world quaternions |
| Abu/RealityScan-style camera CSV | `#name,x,y,alt,yaw,pitch,roll` convention: yaw from North, pitch from nadir |
| COLMAP `images.txt` / `images.bin` | Exact image names and world-to-camera poses converted to centres/orientations |
| COLMAP `points3D.txt` / `points3D.bin` | Points/RGB and adjacent image poses if present |
| COLMAP cameras/rigs/frames files | Opens their surrounding model folder |
| TUM trajectories / ACE0 poses files | Existing engine-specific pose readers |
| XYZ / PTS | XYZ, optional RGB; PTS begins with a point count |
| M54 scene NPZ | Pickle-free scene archive |
| Completed engine run folder | Shared discovery of COLMAP, transforms, native reconstruction, cloud and poses |

Camera-measurement CSVs are not pose CSVs. Unspecified quaternion/Euler conventions
are rejected rather than guessed. A PLY containing Gaussian splat properties is
shown as point centres; this is not a Gaussian-splat renderer. Intrinsics alone,
training weights, diagnostics and reports are not a camera reconstruction. Multiple
disconnected OpenSfM components require a separately selected component; they are
not silently merged into one frame. LAS/LAZ, E57, full mesh editing and CAD operations
are future adapters, not supported-format claims in this release.

## Responsiveness and precision

The control workspace's horizontal and vertical dividers use live resizing. Properties
and Measurements are separate vertical panes: drag their divider upward to enlarge
the table. The property form remains scrollable; Measurement actions provides left/
right opening, predictions, uncertainty and removal without tall permanent toolbars.
Image renders coalesce at a frame interval. While resizing, a coarser mip and nearest
preview reduce work; normal interpolation is restored after 100 ms without resizing.
Marker coordinates and original-image measurements are unchanged by preview quality.

World geometry/pose math stays float64. Native and browser adapters normalize the
display frame before float32 GPU vertex buffers, retaining detail at large world
coordinates. Camera frustums are vectorized and cached; GPU contexts and tool workers
close with their host. Control/standalone displays default to 50,000 points; the
main embedded loader retains up to 150,000, and main browser loading retains up to
one million before its selected display budget. Full exports do not use those caps.
JSON and text parsing can require more temporary RAM than the retained display.

Control/standalone comparison and loading use isolated workers. Main embedded run
loading uses a background worker; large JSON parsing can still contend for CPU time.
Browser geometry/controls remain on a loopback server. Public share requests are
denied on every M54 browser server instance, with the installed Viser hook workaround
retained. No upstream repository or installed package changes are required.

## Architecture for developers and co-development agents

| File | Responsibility |
|---|---|
| `abu_m54.py` | Shared float64 scenes, conventions, transform matrices, picking, grid and file/copy export |
| `abu_m54_native.py` | Shared Windows WGL/OpenGL renderer and interaction |
| `abu_m54_tools.py` | Native scene tools, history and background export |
| `abu_m54_browser.py` | Shared browser tools/gizmo/pivots and standalone browser surface |
| `abu_m54_host.py` | Minimal host services/callback interface |
| `abu_m54_app.py`, `abu_m54_launcher.cs`, `Abu_M54.exe` | Standalone app, isolated loading and Open with registration |
| `native_inspection.py`, `viewer_workspace.py` | Main-app host adapters and existing inspection features |
| `control_points_native_3d.py`, `control_points_3d.py` | Control-tool compatibility/browser adapters |

New tools should use these adapters and shared math instead of introducing another
viewer fork. `control_points_native_3d.NativeControlNavigator` is a compatibility
alias for `M54NativeViewer`. Engine outputs remain the authoritative inputs until
the user explicitly exports a transformed copy.

## Research and verification

The selection/pivot workflow follows established
[CloudCompare point-picking](https://www.cloudcompare.org/doc/wiki/index.php/Point_picking)
and [Open3D interaction](https://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html)
patterns. Group transformations and explicit matrix copies follow
[CloudCompare transformation concepts](https://www.cloudcompare.org/doc/wiki/index.php/Apply_Transformation).
[COLMAP's format specification](https://colmap.github.io/format.html) defines pose
direction, quaternion order and sparse point records.
[Viser's scene API](https://viser.studio/main/api/core/scene_api/) provides browser
gizmos and scene interaction; implementation was checked against installed 0.2.23,
whose gizmo has translation/rotation and a display-size parameter rather than a
uniform model-scale drag handle.

See `ABU_M54_VERIFICATION.md` for measured runtime, tests, limits and reproduction.
