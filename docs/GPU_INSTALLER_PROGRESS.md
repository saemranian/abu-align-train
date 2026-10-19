# Full GPU installer implementation — 18 October 2026

Target: one installer per platform, supplying qualified prebuilt components without
manual Python, pip, Conda, compiler or CUDA toolkit setup. A compatible NVIDIA
GPU/driver is a Windows GPU prerequisite. The existing published v0.1.6.3 assets
remain CPU candidates until newly built assets are published.

## Implemented and tested

* Managed components: pinned hashes/sizes, resumable downloads, selectable storage,
  background progress/cancellation, safe extraction and atomic active receipts.
* Official COLMAP 4.2.1 Windows CUDA extraction/matching; native spherical Ceres
  mapping remains CPU. Original panoramas are used without cubemaps.
* Isolated portable LichtFeld build at revision
  `00a96b63c0ce45691d3dd9b4dcfd02069269c40c`, with its actual NVIDIA exception.
  The older v0.5.4 revision lacked that exception and is not the distribution pin.
* Private CUDA 13.0.2 SDK assembled from verified NVIDIA archives, MSVC 14.44,
  six build workers, minimum SM75 PTX. Profiler, NVTX and NVML headers included;
  DLL layout normalized for upstream packaging. No upstream source edits.
* A separate vcpkg manifest removes optional x264/default FFmpeg features.
  The selected FFmpeg build is LGPL; native NVIDIA video support remains.
* Portable third-party notice generation and coverage passed for 72 vcpkg ports.
  Matching dependency source/build recipes and a hashed source offer accompany
  the redistributed trainer; this does not grant permission for other engines.
* Frozen training uses managed binaries, native camera metadata selects GUT,
  failed subprocess exits cannot falsely report an old PLY as a new success.
* The online setup installs qualified alignment and training packs automatically.
  Offline re-embedding requires separate clearance; the official COLMAP archive
  is downloaded directly from its upstream publisher.
* 37 component, payload, CLI/worker safety and distribution regressions pass.
  Frozen GUI, native-library, refinement worker and loopback M54 checks passed
  on the preliminary runtime; final versioned installer checks remain required.

## Local runtime evidence (RTX 4090)

Earlier tests registered 8/8 panoramas in 5.543 s and 48/48 in 56.387 s. The new
user-authorized fixture registered 30/30 in one reconstruction in 47.620 s at
width 2048 / 6000 features. CUDA extraction/matching and CPU Ceres mapping were
used. These are integration results, not surveyed accuracy or matched CPU
comparison benchmarks.

The native GUT trainer completed 200 iterations on all 30 panoramas at width 512,
cap 15,000, exported 14,210 finite Gaussians, saved `.licht` projects and resumed
to iteration 250 with another validated export. Loss was finite (0.3626).
Training/export/resume took 7.011 s with a warm CUDA cache; steady steps were
2.65 ms/iteration. This short run tests execution, not final-quality convergence.
A generated eight-camera fixture also passed; its first CUDA initialization took
450.572 s, while a fresh-output warm repeat took 4.148 s. First launch can have a
substantial JIT delay. Shutdown emitted upstream CUDA context-release warnings,
despite zero exits and valid exports; retain this candidate limitation.

VRAM metrics from the upstream benchmark are CUDA device-wide and include other
applications. They must not be presented as the trainer's isolated allocation.
The component manager streams archives and creates no CUDA tensors. Client
images, reconstructions, checkpoints, detailed logs and private snapshots remain
local and excluded from publication.

## Remaining acceptance and platform scope

Archive/hash qualification, managed installation, final frozen GUI/worker checks,
online isolated setup, installed CLI training, same-version reinstall and uninstall
have passed locally. User component storage is retained on uninstall. Public
source CI and final desktop release asset verification remain at commit time. No stable or signed/notarized status is implied. Linux GPU
packs need independent qualification. macOS does not support CUDA; Metal training
parity needs a separately implemented and tested backend. Restricted research
integrations, including unresolved PFGS360 redistribution/checkpoint terms, cannot
be bundled simply because the application is free.

See [distribution review](DISTRIBUTION_FINAL_REVIEW.md),
[upstream GPU permission](https://github.com/MrNeRF/LichtFeld-Studio/blob/00a96b63c0ce45691d3dd9b4dcfd02069269c40c/LICENSE-EXCEPTION.txt),
[COLMAP license scope](https://colmap.github.io/license.html) and THIRD_PARTY_NOTICES.md.
