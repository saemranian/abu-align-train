# Expanded manual 360 control workflow verification

This records v0.1.4.4. For the current acceleration, continuous grid and
embedded 3D workflow, see [the newer verification](CONTROL_ACCELERATION_VERIFICATION.md).

## Design and scope

The editor supports measurements before registration and inspection/refinement
after registration. Main Align accepts a saved control file, validates it
against the selected originals, saves an immutable input snapshot, registers
images, performs joint spherical refinement and exports the refined poses.
The engine adapters do not consume manual points as initialization seeds.
Disconnected-component merging remains unsupported.

CSV import follows RealityScan's documented image-path, point-name, X-pixel,
Y-pixel export. It also supports mapped local XYZ with explicit source units
and Ground control/Check roles. A preview precedes a transactional, undoable
merge; conflicts and invalid coordinates are rejected. Perspective/cropped
or differently sized source images require re-marking on the panoramas.

The supplied RealityScan CSV was inspected locally: 113 measurement rows,
22 point names and 96 referenced images. The referenced source image checked
was 2160×3840 portrait perspective imagery. Its safe default is names-only:
22 point identities, zero transferred pixel measurements. No client file or
image was uploaded or copied into this repository.

The thumbnail browser pages 24 images, searches filenames, filters by selected
point measurements and chooses the destination image pane. Its cache is capped
at 128 thumbnails of at most 195×105 RGB pixels (about 7.5 MiB of raw pixels,
excluding objects and active Tk images). One worker decodes thumbnails; stale
jobs are canceled. Original-image measurement coordinates are unchanged.

3D inspection uses Viser on 127.0.0.1 with sharing hidden. Camera centres,
triangulated controls, labels and sight lines share the current alignment
frame. Unregistered survey targets are not guessed into this frame. Sparse
cloud display is sampled to 50,000 points; solver inputs are unaffected.

## Verification

- 39 Tk/model/import/main-window workflow tests passed, including BOM and
  quoted paths, Unicode, multiple delimiters, units, atomic undo, invalid
  projection handling, image grid selection, arrow navigation, pre-alignment
  handoff, preservation on opening an alignment and minimum-width actions.
- Five spherical-refinement/integrity tests passed, including surveyed and
  withheld check geometry. Source parsing and `git diff --check` passed.
- Synthetic local Viser startup, scene population/replacement, control
  selection, loopback HTTP response and shutdown passed. Windows socket tests
  required execution outside the sandbox; no external data service was used.
- Full CLI smoke test on the user's 48 local 7680×3840 panoramas used native
  spherical COLMAP, 1024 feature width, 3000 features, 500 automatic landmarks
  and 10 refinement evaluations. All 48 cameras registered; 503 landmarks and
  2587 observations were refined in 6.41 seconds. Total pipeline runtime was
  43.1 seconds. Angular RMS changed from 0.071761° to 0.067308°.
- The run reached its evaluation budget and did not converge. Its three tie
  points were generated from the existing model for integration testing;
  they are not independent surveyed ground truth or an accuracy benchmark.
  OpenSfM/360DVO-specific fresh runs were not repeated for this UI release.
- `control_points.input.json` exactly matched the retained
  `refined/reconstruction.controls.json`. Exports completed using the refined
  reconstruction. Private test outputs remain outside the repository.

## Memory and performance limits

Thumbnail and viewport work use CPU memory; joint refinement uses sparse
SciPy CPU optimization. The viewer uses the browser's graphics renderer.
This release introduces no GPU training tensors or cubemap conversion.
RTX 4090 memory after testing was 4365 / 24564 MiB across all PC processes;
this is an idle observation, not this workflow's peak or exclusive allocation.
Peak RAM/VRAM was not instrumented in this UI verification run.

## Research references

- [RealityScan measurement CSV export](https://rshelp.capturingreality.com/en-US/tools/cpmeasurementsexport.htm)
- [RealityScan control-point roles and inspection](https://rshelp.capturingreality.com/en-US/tools/controlpoints.htm)
- [RealityScan measured versus estimated coordinates](https://rshelp.capturingreality.com/en-US/appbasics/reports_fav_controlPoints.htm)

The workflow borrows explicit point roles, measurement review and image/3D
selection from these references. It does not claim complete RealityScan parity.
