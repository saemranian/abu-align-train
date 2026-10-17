# Manual control points — image, grid and 3D views

Choose **Image**, **Grid** or **3D** beside **Left** and **Right**. Each view has
its own content, image zoom and 3D viewpoint. A useful measuring layout is Grid
on the left and Image on the right. Two Image views let you compare measurements;
3D beside Image lets you inspect a camera and its original photograph.

## Browse and measure

1. Open the original images or their alignment and select a control point.
2. Switch one view to Grid. Search names or filter measured, missing or predicted
   images. Choose the destination view; an embedded grid defaults to the other view.
3. Double-click a thumbnail or use the arrow keys and Enter. The destination
   becomes Image and the grid stays available. If you choose the grid's own view
   as destination, that view becomes Image deliberately.
4. Zoom and mark the same physical feature. Measurements use original pixels.
5. Review the measurements and save controls before refinement.

All images occupy one continuous scroll region. Two local workers prefetch the
small catalog and nearby thumbnails. Ready cached thumbnails appear immediately;
a neutral image icon can appear on the first uncached visit. There is no Loading
banner. Invalid or missing originals show **Image unavailable** and are logged.
The popup **Image grid** remains available alongside the embedded grids.

## Resize the workspace

Drag the visible divider handles between Points, the central views and
Properties/Settings; drag the middle divider for Left/Right widths, and the
Properties divider for measurement-table height. Dragging/releasing works over
child images, notebooks and tables. Side widths fit when the window shrinks and
restore when enlarged. Project → Reset pane sizes restores the initial layout.

Settings → Pane resizing offers Auto, Fast and Live. Auto defaults to a responsive
divider guide above2 million window pixels, committing content sizes on release;
small windows resize live. Fast always uses the guide. Live redraws while dragging
and can be slower at large/high-DPI sizes. The choice saves separately from the
solver. Grid wheel/scrollbar input coalesces into one GUI-idle layout, uses real
wheel deltas and avoids unchanged tile layouts. Read [the measured limits and
workflow](MANUAL_UI_RESPONSIVENESS.md).

## Read angular error colours

Only the **° error** numbers are coloured; the rest of the row remains readable.

| Colour | Default angular residual |
|---|---:|
| Green | 0 through 0.10° |
| Yellow | Above 0.10° through 0.50° |
| Red | Above 0.50° |
| Uncoloured dash | Not available |

**Settings → Green error limit / Yellow error limit** changes the limits live.
Both must be positive, finite and in increasing order. Incomplete typing keeps
the last valid colours. Save controls validates and retains these display settings
separately from the solver settings; older control files use the defaults.

These colours describe agreement between measured rays and the triangulated
point. They do not prove surveyed accuracy. Inspect red measurements for wrong
features, stitching, reflections or poor parallax before tightening uncertainty.
The equivalent pixel error depends on image resolution, projection and latitude.

## Use shared Abu M54

A 3D view needs an open alignment. It uses the shared **Abu M54** GPU renderer,
scene/file math, selection, transforms, nulls and comparison tools. Source
geometry remains unchanged. Both panes receive the same scene data and retain
independent navigation and preview state. Their tools are host adapters, not
another renderer or parser.

Drag to orbit, middle/right-drag to pan and wheel to zoom. Click a control or
camera. Orbit selected, O or double-click sets the pivot. A picked camera opens
its original in the other view only when that view is Image; it does not collapse
a second Grid or 3D workspace. **Tools** opens M54 settings and **Compare** retains
the existing read-only B workflow. Save independent nulls explicitly; previews
and nulls are not bundle-adjustment measurements.

The in-app **Help & shortcuts** includes locally drawn diagrams for the view
choices and error colours. Hover guides explain destination selection and limits.

See [verification and performance limits](MANUAL_VIEWS_VERIFICATION.md) and
[the shared viewer guide](ABU_M54.md).
