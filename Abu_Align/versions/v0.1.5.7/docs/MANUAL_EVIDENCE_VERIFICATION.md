# Manual control point image evidence — v0.1.5.4

## Changes and root causes

Properties now marks a measured-image row mint for Left, blue for Right and purple
for the selected native/browser M54 camera. Error numbers retain independent
green/yellow/red colouring. Excluded or masked measurements are grey; the raw
observation is preserved. Ctrl+F focuses the active image measurement/proposal or
the selected 3D object. F continues to fit the entire view.

Project → Images & masks manages all image inclusions, original-resolution layers
and explicit inversion. Measurement actions can toggle the selected image. The
policy is saved with controls, Undoable, logged and validated before a solve.
Disabled shots and masked manual/automatic observations are removed from the
effective solve and exported new model, without modifying input documents/images.
Each enabled optimization control still needs two usable views. Unregistered
measurements require initial Align before Refine. Policy-only input is accepted
for initial Align without forcing a manual post-refinement.

Native OpenSfM's supported mask folder and explicit image_list.txt prevent stale
links re-entering the job. Masked/excluded runs require fresh feature workspaces.
The native test exposed an OpenSfM no-feature failure for all-black masks; Abu now
omits these originals before engine input. Native COLMAP treats missing masks as
MASK_ERROR; Abu stages white layers to implement the documented whole-image
fallback. Initial masks are frozen in the run folder, with source SHA256 records;
the same staged policy drives post-refinement. Fully masked omitted images remain
explicitly excluded when their masks are absent from that staged selected set.

The 360DVO app runner applies a bounded GPU mask atlas to source patch centres
and predicted target locations, zeros disallowed neural weights, and rejects
masked output cloud patches. It records counters and hashes masks/adapter into
cache fingerprints. Conservative mask downsampling/erosion avoids boundary leaks.
CPU spherical and DVO fallback paths respect the layers too. Initial masks are
supported for OpenSfM, native COLMAP and 360DVO only; other engines explicitly
reject them, but image exclusions and supported model post-refinement remain
available. No installed upstream repository or package source was edited by this
work; existing upstream checkout edits from other work were preserved.

Old predictions were geometry projections into registered cameras, ranked near
measured views, with no image-content evidence. Visual search now examines every
enabled original in all scope, including unregistered originals. The local,
cancellable worker combines ALIKED/LightGlue CUDA/CPU or SIFT, bounded feature
counts, native spherical tangent neighbourhoods, robust local correspondence
transforms, a geometry gate and support from up to three measured source views.
Sparse context widens adaptively; expanded support must surround the query.
Uncalibrated perspective images use pixel search rather than guessed optics.
No cubemap alignment dataset is created. Proposals map back to original pixels,
label single-view/agreement/ambiguity and require the user's verification/click.
Geometry-only suggestions also respect target masks and exclusions.

Inference blocks outbound socket connections and loads local tensor-only weights;
setup separately downloads public checkpoints. No photo inference API is used.
Weights live in an ignored app cache and are omitted from Git and backups.
The two public local checkpoints were verified:

- ALIKED SHA256: 5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2
- LightGlue SHA256: d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503

Default robust transition is 1 sigma unit, automatic parallax 1 degree and all
three tolerances 1e-8. These are verified against installed SciPy/OpenSfM, with
solver-unit/budget differences explained. Auto dispatch, 10000 track budget,
1-pixel uncertainties and marker radius 6 are Abu choices. Restore standard
defaults preserves existing per-measurement uncertainties. Hover/Help/guide
explain increasing/decreasing each parameter, failure modes and independent checks.

Broad tests exposed intermittent Tcl handle finalization on Viser background
threads after windows closed. Owned variables/images and native compare/editing
controllers now retire on the GUI owner thread. M54 standalone closes its own
root handles; the Manual tool retires only its window, preserving the host root.
Workers/timers/GL contexts close normally. No global GC suppression or installed
Tk package patch was used.

## Verification and limitations

Final broad regression: **173 tests passed in 108.361s**.
After freezing the initial/refinement mask handoff, focused evidence, CSV/main
handoff, CUDA solver and manifest checks: **34 passed in
26.746s**. The sets overlap; their counts are not added. Both processes
exited0 without Tk finalizer/fatal warnings. PyTorch's sparse CSR beta warning is
informational. Checks include mask dimensions/names/inversion, immutable source
documents, whole-image omissions, white fallback, GPU/CPU seam/bounds sampling,
initial preflight and frozen masks, excluded camera/track export, genuine
original-pixel prediction geometry, held-out/unregistered views, realistic
defaults, row highlighting, unsaved proposals, Undo/save and Ctrl+F.

```powershell
$env:PYTHONPATH = "$PWD;$PWD\tests"
& C:\Users\USer\miniconda3\envs\vggt\python.exe -B -m unittest test_manual_evidence test_manual_views test_control_points_workspace test_control_import_navigation test_control_comparison_help test_control_acceleration test_manual_cameras test_spherical_refinement test_spherical_integrity test_desktop_ui test_abu_m54 test_m54_objects test_native_inspection test_viewer_workspace test_fast_loader test_fast_viewer test_run_manifest test_360dvo_cache test_dvo_tracks
```

Native OpenSfM installed revision c4985ced723888d2529f9b7a1902d4aa682de1d0:
synthetic original panoramas, one wholly masked omission, half-mask feature
boundary check. Extraction exited0 in 5.915s; the partially masked
image had322 features with normalized X >=0 and the unmasked image711 including
negative X. Native COLMAP4.2.0 CPU extraction exited0 in 0.569s,
with203 masked features whose original X >=720.504 and465 unmasked features.
The installed Windows PyCOLMAP build has no CUDA. These tests verify mask
extraction, not a complete native OpenSfM/COLMAP accuracy benchmark.

Native neural 360DVO CUDA, revision 46f5ed0735d39aff29a75f28c360600340879b35, on16 authorized
local panoramas: 19.844s entire subprocess,
9.310s runner,16/16 trajectory rows,6123 persistent observations,
no truncation. Original-resolution synthetic binary masks ignored a small stripe.
The adapter recorded7480 masked weight evaluations
and36 rejected cloud patches. Weight counters count
repeated optimizer evaluations, not unique observations. Peak allocated Torch
VRAM 644.89MiB; atlas
0.50MiB. This validates hardware/mask plumbing rather
than surveyed alignment accuracy.

Known-geometry textured planar scene rendered as six original ERP panoramas:
two supporting images, three targets including an unregistered held-out camera,
one excluded. Both SIFT and ALIKED/LightGlue returned3/3 targets with multi-view
agreement. SIFT median/max original pixel error
0.06177/0.21372px,
scan 1.465s, whole process 2.112s,
peak RSS 101.59MiB. Learned median/max error
0.05445/0.40789px,
scan 7.378s, process 8.969s,
peak Torch VRAM 282.56MiB, RSS
1493.95MiB. The easy plane favoured SIFT speed; these
results do not prove learned matching universally better. A separate uncalibrated
perspective translation test found2/2 targets, max original pixel error
0.18958px.

Authorized48-image real panorama set, one CV-selected seed, no ground truth:
47/47 candidate originals searched locally, 2 unverified
proposals, 81.032s, peak Torch VRAM
281.83MiB, RSS 1580.92MiB.
Coverage is not recall or accuracy. Sparse/repeated texture, occlusion, stitching,
wrong source clicks and bad initial poses remain limitations. Low proposal count
is explicit; nothing is auto-accepted. Reference image-pair benchmark rankings
do not establish native panorama control-point accuracy. RoMaV2 and Efficient
LoFTR are research candidates, not installed/tested production options here.

Decoded prediction buffers retain two previews; model feature count1024 and
source views3 are bounded. Mask LRU caps at128MiB (one very large mask can exceed
the cap); DVO atlas rejects requests above128MiB. Worker exit releases its CUDA
context; normal matching close empties allocator caches. RSS includes Torch and
library/context overhead. Timings include cold model/import costs. Existing Tk
large-window painting limits from v0.1.5.3 are not changed or claimed eliminated.

All dataset reads/inference/geometry outputs remained local during this work;
no screenshot capture was used. Private event logs/coordinates/photo names are
outside Git in the local artifact directory. No client pixels or dataset files
were uploaded in this turn. The earlier screenshot incident remains documented
in the preceding walkthrough; this statement does not erase that prior incident.

## Release isolation

Verified checkout: `C:\Users\USer\.codex\worktrees\m54-viewer-release\abu align`; branch codex/m54-viewer-release. v0.1.5.4 is published
here to preserve another agent's canonical v0.1.5.0 GUI/splitter/rig work and counter.
Owned modules/guides and their Abu_Align copies are updated in the shared checkout;
its release headers stay at the canonical counter. Full parity and clean Git
status refer to this isolated release. Both complete version snapshots, root,
mirror and artifact walkthroughs are verified. Public weights are cached locally
for offline inference; the isolated launcher can open this verified release.

See [accuracy guide](MANUAL_ACCURACY_GUIDE.md), [primary research comparison](MANUAL_PREDICTION_RESEARCH.md)
and [modified files](MANUAL_EVIDENCE_RELEASE_FILES.md).
