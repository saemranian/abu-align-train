# Manual 360 controls and reproducible evaluation

Open **Prepare input ▾ → 360° manual control points**. The redesigned workspace has a searchable point list, two independent image views, a Properties/Settings/Results inspector, and a Help tab. See [the user guide](360_CONTROL_POINTS_USER_GUIDE.md). Select one spherical
reconstruction.json and its original images. Create a named point and click
the same physical location in each pane. Choose other images to add more
observations. Wheel zooms, middle-button drag pans, right click removes an
observation, and Undo restores edits. Zoom renders only the visible region.

You can also **Open panoramas** before registration and prepare measurements
without camera poses. **Use with main Align** saves and attaches the controls
to the main window. Its enabled controls option runs initial registration,
then joint refinement, and packages the refined poses. The CLI validates the
selected images and snapshots the input controls before inference. These are
post-registration constraints, not initialization seeds. Opening an alignment
for the exact same originals retains prepared measurements.

**Import CSV** previews RealityScan image measurements or local XYZ, with
explicit column mapping, units, merge policy and undo. Perspective-image
measurements cannot be transferred onto panoramas: import point names and
re-mark the features instead. **Image grid** provides paged thumbnails,
filename search and measurement filters. Plain Left/Right browse images;
Ctrl/Shift arrows nudge measurements. **3D view** serves a local loopback
WebGL scene with sampled sparse points, camera centres, triangulated controls
and selected sight lines. It does not overlay unregistered survey XYZ.

Tie points need translated views and measurable parallax. Add observations
across the trajectory, not just adjacent frames. Longitude wraps around the
panorama seam in the solver. GCP/check positions currently use **local XYZ in
meters**, not latitude/longitude; three non-collinear GCPs establish the world
frame. Check points are never optimized. Distance endpoints are existing
landmark names. Every position/observation has an uncertainty.

**Refine alignment** writes a new reconstruction and report. The source stays
unchanged. The native solver jointly optimizes cameras and landmarks using
sparse SciPy CPU BA and robust bearing residuals; this is not advertised as a
GPU solver. The automatic point budget is 10,000 and manual points are kept.
Unselected tracked points are retriangulated after camera refinement. For a
large reconstruction, start with a small budget and inspect the termination
and angular residual report. Angular residual improvement alone is not proof
of absolute accuracy: use independent check points and reference trajectories.

```powershell
python spherical_refinement.py --input reconstruction.json --controls control_points.json --output refined.json
python align_cli.py --engine opensfm --input images --spherical_refine --control_points control_points.json
python benchmark_360.py --estimate refined.json --reference ground_truth.json --alignment sim3 --output benchmark.json
```

Evaluation reports named-camera completeness, Sim(3)/SE(3)/unaligned ATE,
translation and rotation RPE, and fitted scale. Reference-coordinate units
are explicit. Use SE(3) or no alignment for externally established metric scale.
The Python API also offers solid-angle weighted panoramic PSNR on RGB in [0,1].

Synthetic regression: four cameras/18 landmarks, perturbed poses/points,
joint refinement reduces independent ATE by >5x and angular RMS by >10x.
Three surveyed controls plus one withheld check produce check error <1 cm.
These tests establish correctness on generated geometry, not real-scene SOTA.

Disconnected component merging, geographic GCP conversion and suggested
automatic annotations remain separate future extensions.


Neural 360DVO now preserves confidence-filtered patch measurements in
`dvo_tracks.json`. Stable source-frame/patch IDs survive window removal. The
OpenSfM handoff triangulates them against the final full trajectory and restores
original-image pixel coordinates. The tracked model excludes loose display
cloud points that cannot participate in joint BA; their separate PLY remains.
The default main CLI/desktop DVO route requires the neural backend to initialize.
Use `--dvo_allow_fallback` only for an explicitly experimental CPU run.


Disabled manual points and distance constraints attached to them are excluded from optimization. Controls retain their disabled state on disk. Use Apply properties to commit property edits; Save as keeps separate control-file variants. The workspace includes measured angular residuals and unconfirmed projection guidance. Predictions never become observations automatically.
