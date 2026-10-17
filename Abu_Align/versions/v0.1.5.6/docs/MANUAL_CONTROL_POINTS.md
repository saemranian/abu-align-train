# Manual control points — 360 and perspective

The same tool measures original equirectangular panoramas and supported calibrated
perspective images. Open it from **Prepare → Manual control points**. All images,
measurements, browser viewing and computation stay on this PC; no cubemap conversion
or upstream modifications are required.

## Choose the starting alignment

For further improvement, open the **latest good refined reconstruction JSON**, keep
the same original images and saved controls, add/correct measurements, then refine
again. Camera poses and supported sparse observations from the last solve provide
a better initialization and often reduce work. It is a starting point, not a proof
of accuracy. Use the initial alignment to restart, compare alternatives, or recover
from an incorrect control/survey configuration. Repeating unchanged constraints
after convergence does not create further accuracy.

Use **Refine alignment** inside this tool to update the alignment currently open.
**Use with main Align** saves/attaches controls, budgets, hardware selection and
the saved accuracy settings. The main **Align** button then runs initial engine
registration followed by manual refinement before packaging. It starts a fresh
alignment pipeline; it does not resume the JSON open in this editor.

## Load and measure

Open alignment accepts a single native/OpenSfM JSON component or a complete COLMAP
model: select its `images.bin`, `images.txt`, `cameras.bin` or `cameras.txt`.
VGGT-Omega and other engines' calibrated COLMAP outputs can be used. Choose the
original image folder if it is not discovered. Names, dimensions and projection
must match; pixels from cropped, resized, perspective or spherical images cannot
be interchanged. A point-only PLY or an uncalibrated transforms JSON is insufficient
for this measurement/refinement interface; the M54 viewer can still display those
supported geometry inputs separately.

Before alignment, choose **Settings → Before-alignment image type**, then **Open
images**. This selection is explicit because a 2:1 aspect ratio does not prove
an image is equirectangular. Camera calibration in an opened alignment takes
precedence. Panorama seam wrapping stays disabled on perspective views.

Mark a stable, identifiable feature in several translated views. Three to six
well-separated views help find mistakes. Spread multiple controls over the scene,
including depth and height variation. Use the image grid, arrows and predicted
image list to navigate. Perspective predictions are clipped to the camera's
visible image and positive depth. Predictions remain guidance: inspect the actual
feature before clicking it. CSV measurements can be imported directly for the
same perspective originals, as well as for the same spherical originals.

## Accuracy settings

| Setting | Starting value | When to change it |
|---|---:|---|
| Automatic landmarks | 10000 | Try 25000–50000 if good track coverage and independent checks justify the cost |
| Solver budget | 100 | Try 200–500 when the limit is reached while useful progress remains |
| Robust loss | soft_l1 | Huber is another robust choice; linear suits verified clean observations |
| Robust transition | 2 σ units | Change after setting realistic uncertainties; too low discards useful influence |
| Automatic track uncertainty | 1 original px | Increase for noisy tracks so verified manual observations carry more relative influence |
| New measurement uncertainty | 1 original px | About 0.5–1 for crisp marks, 2–5 for blurred/stitching-sensitive features; existing observations have their own values |
| Minimum automatic views | 2 | Try 3–4 if enough overlapping tracks remain |
| Minimum automatic parallax | 0° (extra filter off) | Try 0.5–1° for ordinary scenes; distant scenes may need lower values |
| Objective / step tolerance | 1e-8 / 1e-8 | Tighten only if convergence and independent checks justify more work |
| Gradient tolerance | 1e-7 | Same principle; CPU and CUDA stopping tests differ |
| Hardware | Auto | Small jobs use analytic CPU; larger supported jobs use CUDA; explicit CUDA is available |

Image uncertainty is in original pixels; survey/distance uncertainty is in metres.
Enter real precision, not the accuracy you hope to achieve. Smaller uncertainty
increases influence without making a measurement more trustworthy. Enabled manual
ties/GCPs remain included when automatic track filters/budgets change.

Ground controls constrain the local metric frame. At least three non-collinear
surveyed controls are required; good spatial/depth distribution is better than
clustering them. A measured distance constrains scale. **Check points are withheld
from optimization**: compare their errors between runs. Lower fitted residual,
more iterations or a tighter tolerance alone does not demonstrate better absolute
accuracy. Hover guides explain each setting and tradeoff.

## Camera models and VGGT limits

Supported native models are spherical/equirectangular, OpenSfM perspective and
Brown, and explicit calibrated pinhole/OpenCV/OpenCV fisheye. Supported COLMAP
models are PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV and
OPENCV_FISHEYE. Focal lengths, principal points and lens distortion remain fixed;
this tool refines camera poses and sparse landmarks. It does not guess a lens
model or silently recalibrate intrinsics. Unsupported models fail with an
explanation. Calibrate a wrong model in the originating engine first.

A VGGT neural cloud may have no multi-view tracks. Its cloud centres do not become
automatic observations. Enough distributed manual measurements can still refine
the observed cameras; unobserved cameras remain at their initialized poses. The
solver rejects an insufficient constraint count or disconnected selected track
graph. Passing those checks is necessary, not sufficient evidence of good geometry.

Untracked source cloud points cannot be corrected after camera changes and are
excluded from the corrected sparse output. Their original source cloud remains
untouched. A manual-only result can therefore contain far fewer points than a
neural cloud and is not a regenerated dense model or a guaranteed better training
initialization. Obtain/export real multi-view tracks for a stronger full-scene
refinement and dense reconstruction after pose changes.

## Outputs and M54

Refinement writes a new JSON, exact controls/settings snapshots, report,
double-precision sparse PLY and exact camera CSV. Homogeneous perspective results
also receive a `.colmap` model folder with intrinsics and valid sparse observations.
Reports record settings, source, camera families, optimized/fixed cameras, track
filter counts, excluded untracked points, convergence, checks and solver VRAM.

M54 scene-tools numbers update the preview immediately. Type, or drag a number
left/right. Shift gives finer steps; Ctrl gives coarser steps. Scale changes
proportionally and stays positive. One drag is one native Undo step; incomplete
keyboard numbers retain the last valid scene. Export saves a separate full-source
model transform; source geometry/controls are never overwritten.

Browser orbit/pan is the default. **Pick one cloud sample** temporarily captures
a click, then returns to normal navigation. Camera/control clicks still work.
Changing browser numeric transform fields also updates the preview immediately.

## Research basis and scope

[OpenSfM's control-point documentation](https://opensfm.org/docs/using.html#ground-control-points)
distinguishes global frame alignment from bundle adjustment.
[COLMAP's calibration guidance](https://colmap.github.io/faq.html#fix-intrinsics)
explains fixed calibration and the risks of self-calibration; its
[format specification](https://colmap.github.io/format.html) defines poses, tracks
and camera records. The adapter uses the installed pycolmap 4.2.0 API.
[VGGT-Omega's official repository](https://github.com/facebookresearch/vggt-omega)
provides camera/intrinsic predictions; actual Abu export tracks are checked locally.
[SciPy's least-squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
defines robust losses and stopping tolerances. CPU and CUDA share the validated
loss/noise settings and analytic geometry derivatives. These sources support the
design; synthetic tests are not a surveyed benchmark of your client project.

The full calibrated input API is `abu_m54.load_calibrated_alignment`.
`manual_camera_geometry` handles pixel/ray/lens projection and native COLMAP
conversion; `manual_refinement_settings` defines settings and hover guidance.
Reuse these with M54 rather than introducing a new file/parser/navigation fork.
