# Manual control points — accuracy settings and image evidence

There is no setting that guarantees the highest accuracy for every scene. Start
with correctly identified features in translated views, realistic uncertainties,
distributed controls and independent check points. Refine the latest good model.
Inspect red residuals and the termination/check report before increasing budgets.
Camera calibration stays fixed; wrong focal lengths or distortion require a
correctly calibrated input model.

## Settings in the refinement panel

| Parameter | New default | Increasing / decreasing / practical effect |
|---|---:|---|
| Hardware | Auto | CUDA speeds supported large solves; CPU suits small jobs. This choice does not improve the evidence. |
| Robust loss | soft_l1 | Huber is also robust; linear trusts all residuals equally. Use linear only after verifying observations. Robust loss cannot repair a wrong feature. |
| Automatic landmarks | 10,000 | More good tracks can constrain more cameras but use more time/RAM/VRAM; fewer save resources and can weaken coverage. Manual optimization controls remain included. |
| Solver budget | 100 | Increase to 200–500 if the budget was reached while improving. Decrease for quicker trials. A limit reached is not convergence. CPU counts residual evaluations; CUDA counts LM steps. |
| New measurement σ | 1 original pixel | Smaller gives new clicks more influence; larger expresses less confidence. Use about 0.5–1 for crisp carefully marked corners, 2–5 for noisy/stitching-sensitive features. Existing observations retain their own σ. |
| Marker radius | 6 screen pixels | Larger is easier to see; smaller reveals more detail. Display only: no accuracy effect. |
| Robust transition | 1 σ unit | Higher trusts larger discrepancies; lower rejects them sooner. Too high trusts outliers; too low discards useful evidence. Set realistic σ before tuning it. |
| Automatic track σ | 1 original pixel | Higher reduces automatic tracks' influence relative to verified manual evidence; lower increases it. It must describe real track noise. |
| Minimum automatic views | 2 | Raise to 3–4 for stronger track support if coverage remains. Lower to 2 for sparse overlap. Excessive filtering can disconnect cameras. Manual controls are retained. |
| Minimum automatic parallax | 1° | Higher rejects weak depth geometry but can discard distant tracks; lower keeps low-baseline/distant tracks but can leave unstable depth. 0 disables the extra filter. |
| Objective tolerance | 1e-8 | Smaller demands less relative cost change before stopping; larger stops sooner. Extra iterations do not correct bad data. |
| Step tolerance | 1e-8 | Smaller requests smaller parameter steps before stopping; larger can stop earlier. Scene scale and solver definitions matter. |
| Gradient tolerance | 1e-8 | Smaller demands a smaller gradient; larger stops sooner. CPU/CUDA stopping criteria differ; tighter values can only exhaust the budget. |

A useful accuracy trial is 25,000 automatic landmarks, budget 200, 3 automatic
views and 1° parallax, with realistic existing measurement/survey uncertainties.
Keep the default tolerances initially. Compare independent check errors and camera
coverage against the default run. Retain the better supported result, not merely
the one with the smallest fitted RMS. Lower a filter if it disconnects the model.

**Restore standard defaults** restores the new numbers without changing existing
measurement uncertainties. Loading an older control file keeps its saved overrides.
Each hover guide and **Help → Refine** explains the parameter and its default source.

## Which defaults actually come from upstream?

The local SciPy 1.17.1 signature gives ftol/xtol/gtol = 1e-8 and f_scale = 1.
Installed OpenSfM revision `c4985ced723888d2529f9b7a1902d4aa682de1d0`
uses SoftLOneLoss, threshold 1, min_track_length 2, triangulation_min_ray_angle 1°
and bundle_max_iterations 100. Abu uses the same robust family and appropriate
numeric defaults, but its normalized residuals and CPU/CUDA budgets are not
identical to Ceres' definitions. New OpenSfM master uses IRLS instead.

The 10,000-track budget, 1-pixel noise assumptions, Auto dispatch and 6-pixel marker
radius are explicitly **Abu defaults**, not universal upstream standards. OpenSfM
reprojection SD 0.004 is in normalized image coordinates; copying 0.004 into a
pixel uncertainty would give unrealistic weight. 360DVO's native patch/window
settings configure its initial odometry, not this post-alignment solver.

Sources: [SciPy least_squares](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.optimize.least_squares.html),
[installed OpenSfM revision](https://github.com/OpenSfM/OpenSfM/blob/c4985ced723888d2529f9b7a1902d4aa682de1d0/opensfm/config.py),
[current OpenSfM configuration](https://github.com/OpenSfM/OpenSfM/blob/master/opensfm/config.py),
[360DVO native configuration](https://github.com/chris1004336379/360DVO/blob/master/config/360.yaml).

## Image highlighting, exclusions and masks

Measured-image rows show a mint background for Left, blue for Right, and purple
for a selected 3D camera. The angular numbers keep their independent error colours.
**Ctrl+F** focuses the current measurement/prediction; in a 3D pane it focuses the
selected object. **F** still fits the complete view.

**Project → Images & masks** manages image inclusion and a separate mask folder.
Select rows and Enable/Disable. Measurement actions also toggles its selected image.
Exclusions affect subsequent initial Align and Refine; existing originals and
measurements remain in the saved control document. Undo restores policy changes.

Use original-resolution binary mask layers named `image.jpg.png`, `image.mask.png`
or `image.png`. Black ignores, white keeps (threshold 128); inversion is explicit.
Missing masks keep the whole image. Ambiguous names, reused masks and wrong sizes
are rejected. Apply validates the layers, then Save controls and use Refine or
Use with main Align. Masks cannot share the originals folder.
Fully black masks omit the corresponding original from engine input rather than
feeding an empty feature set into OpenSfM. Partial masks retain usable regions.

Initial masks are connected to OpenSfM's native mask folder, Abu's spherical
fallback, native COLMAP's ImageReaderOptions and the app-owned 360DVO source/target
weight adapter. Masked 360DVO cloud patches are filtered too. Other initial engines
reject mask requests explicitly, while image exclusions and supported imported
model refinement still work. Use fresh engine output folders after policy changes.
An enabled control must retain at least two usable measurements; add views or
disable that control. Unregistered measurements need initial Align before Refine.

## Stronger local visual predictions

Open **Predicted images → Visual search**. `all` searches every enabled original,
including unregistered images; `registered` limits the job to cameras with poses.
No nearest-camera candidate cap is imposed. Registered views use geometric search
locations; unknown views use a wider visual search. Both retain original pixel
coordinates. Native panoramas are not converted into cubemap datasets. Small
transient tangent patches correct local panorama distortion only for matching.

`auto` uses local ALIKED/LightGlue when available, otherwise SIFT; `learned` requires
the learned weights. **Set up local AI model** downloads public weights only.
Image inference runs in an isolated process with outbound connections blocked,
supports cancellation, reports coverage and releases its GPU memory on exit.

The matcher verifies nearby correspondences with robust local transforms, avoids
long extrapolation, checks masks and geometric displacement, and compares proposals
from up to three measured source views. Results label one-view support, agreement
or ambiguity. The score is a matching heuristic, not an accuracy probability or
proof of visibility. Open & focus, inspect the same physical feature and click it
yourself. No automatic proposal is saved as a measurement.

See [prediction research](MANUAL_PREDICTION_RESEARCH.md),
[measured validation](MANUAL_EVIDENCE_VERIFICATION.md)
and [the mixed-view workflow](MANUAL_VIEWS.md).
