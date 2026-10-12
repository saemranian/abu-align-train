# Manual 360 refinement acceleration and troubleshooting

## Root causes and local changes

The supplied refinement report used sparse SciPy finite differences with
Python observation loops: 593 cameras, 10,001 landmarks, 39,572 optimized
observations, 476.98 seconds, evaluation budget exhausted. Refinement ran in
the Tk process on a worker thread, allowing Python work to compete with the
interface. This release uses a separate cancellable process and limits its
BLAS/OpenMP workers, while supplying progress, device and memory reporting.

The local solver now has batched analytic bearing and metric Jacobians.
CUDA uses PyTorch float64 sparse CSR matrix products and damped, column-scaled
conjugate-gradient normal solves; no dense all-parameter Hessian is formed.
Auto chooses CUDA at 2,000 or more observations when available. Smaller jobs
and the CPU selection use SciPy sparse least squares with analytic Jacobians.
Explicit CUDA errors are reported; Auto falls back on CUDA out-of-memory.
Retrieval and triangulation of remaining landmarks are batched on the CPU.
Output JSON is compact and atomic; source models and images are unchanged.

The blank browser was reproduced on synthetic geometry. The installed Viser
0.2.23 client reports React error #300 when `show_share_button=False` changes
its hook count. Abu now preserves that client setting and blocks the server
instance's public-share method instead. The browser visibly loaded the cloud,
camera markers, selected control and sight lines after this change. Normalized
display coordinates, explicit clipping planes and scene-fit controls reduce
precision/clipping problems. No upstream checkout or installed package was
edited. The optional share icon cannot create a tunnel.

An embedded Windows OpenGL navigator renders the sampled sparse cloud on the
graphics driver. It supports orbit/pan/zoom, fit and view presets, selected
focus, visibility switches, camera orientation axes, sight lines and camera/
point picking or selection lists. The 2D workspace remains available for
measurement. The observed renderer was NVIDIA GeForce RTX 4090, OpenGL 4.6.

The image grid is one virtualized scrolling page; all filenames are reachable
without pagination. Only visible and nearby rows instantiate thumbnails.
The cache stays bounded to 128 thumbnails. Predicted-image lists and grid
filters rank unmeasured cameras near supporting observations, show projected
pixels and focus the image on opening. They do not assert visibility or
automatically create a measurement; occlusion remains unknown.

Session JSONL logs store document edits, image/CSV choices, settings, saves,
worker progress, outputs, cancellation and exceptions. They remain local and
contain no image contents. Results provides an Open troubleshooting log action.

## Local performance results

All measurements used the same existing spherical input model and saved
manual controls on the user's PC, with a 10,000 automatic-landmark budget.

| Run | Budget | Angular RMS after | Time | Peak solver VRAM |
| --- | ---: | ---: | ---: | ---: |
| Supplied old finite-difference report | prior evaluation budget | 0.075988° | 476.98 s reported | unavailable |
| Initial CUDA integration | 40 iterations | 0.077473° | 63.97 s total | 60.32 MiB |
| Longer CUDA integration | 100 iterations | 0.073879° | 67.84 s total | 60.32 MiB |
| Final batched retriangulation and compact export | 100 iterations | 0.073880° | 40.95 s total | 60.32 MiB |

The final run spent 13.58 seconds in setup and 20.32 seconds in optimization;
remaining time covered retriangulation, verification and exports. It retained
593 cameras and 156,785 sparse landmarks. Initial angular RMS was 0.135842°.
All listed real runs exhausted their budgets; none proves convergence or
independent absolute accuracy. The old timer excluded setup/JSON writing,
while the new total includes them, and iteration budgets differ. These are
local workflow comparisons, not a controlled accuracy benchmark or SOTA claim.

Peak GPU allocation is measured by PyTorch for the solver, excluding other
applications and browser/native OpenGL resources. CUDA cached memory is
released after solving; the isolated process exits after export/cancellation.
Peak host RAM was not instrumented. No photos were uploaded, no cubemaps were
created, and no upstream repositories were modified.

## Output files and using an existing refined model

Open the refined reconstruction in the editor to inspect its corrected camera
poses and cloud. Results > Export cloud & camera CSV works for an older JSON.
Successful new solves create these companions automatically:

- `<name>.points.ply`: binary sparse points, native axes/scale and RGB.
- `<name>.cameras.csv`: centres, scalar-first camera-to-world quaternions,
  world-to-camera rotation vectors/translations, projection and dimensions.
- `<name>.exports.json`: paths, counts and coordinate conventions.

The requested existing result was exported beside its original JSON: 156,785
points and 593 cameras. That JSON, its saved control file and its report were
not replaced. This is a sparse cloud, not a new dense cloud or trained splat.
The CSV is a native camera-pose table, not a RealityScan measurement import.
Training still needs the full prepared dataset. Main Align with saved controls
enabled runs registration, refinement and the normal training-dataset exports.

## Verification and references

Targeted verification covers analytic Jacobians versus independent finite
differences, CPU and CUDA synthetic pose accuracy, surveyed GCPs and withheld
checks, batch/scalar triangulation agreement and parallel-ray rejection,
lossless cloud/CSV coordinates, actual isolated worker responsiveness and
cancellation, edit/prediction logs, 600-image virtualization and NVIDIA 3D
camera/control selection. The Viser browser was checked through its rendered
UI using only synthetic data; sharing was checked to return no URL/tunnel.
The final targeted suite passed 53 tests in 45.14 seconds.

- [PyTorch sparse matrix multiplication](https://docs.pytorch.org/docs/stable/generated/torch.sparse.mm.html)
- [Viser server and client controls](https://viser.studio/main/api/core/server/)
- [React error #300](https://react.dev/errors/300)

The implementation targets the installed PyTorch 2.5.1 CUDA 12.1 and Viser
0.2.23 APIs; the linked current documentation is background reference, not
evidence that the installed dependencies were upgraded.
