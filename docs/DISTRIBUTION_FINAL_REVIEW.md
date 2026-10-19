# Distribution review and implementation status

Reviewed against Antigravity's distribution/licensing strategy and the public
`saemranian/abu-align-train` repository on 18 October 2026.

## Verdict on the original implementation

It was not a complete standalone distribution. Copying source into an installer
did not supply an interpreter, native libraries, GPU engines or training runtime.
The original publisher could expose private workspace files and rewrite public
history. The original license conclusion was too broad. These were material
problems, rather than cosmetic differences between the two plans.

The subsequent CPU candidate work supplied actual frozen runtimes and verified
Windows, Linux x64 and macOS arm64 builds. Version 0.1.6.3 passed all three public
build jobs. Those assets are CPU candidates; they must not be described as full
GPU training installers. See DISTRIBUTION_AUDIT.md for the original corrections.

## Corrections that must remain in every release

1. Free pricing does not override research, model, redistribution or commercial
   restrictions. A label or automatic model download does not grant permission.
2. The retained VGGT custom grant, Omega's grant and VGG-T3's grant are distinct.
   The suite cannot be labelled uniformly Apache-2.0. ACE0 restricts binary
   redistribution; PFGS360's complete grant and depth/checkpoint terms remain
   unresolved. Those integrations cannot simply be frozen into a public installer.
3. A CUDA dependency does not become Metal-compatible on macOS. Splatfacto's
   perspective training is not native panorama training. Full-platform parity
   requires actual implementations and hardware acceptance, not a build badge.
4. COLMAP's BSD license does not clear all native dependencies. The verified
   upstream CUDA archive is acquired directly from its original publisher;
   offline re-embedding needs separate dependency/source/notice clearance.
5. LichtFeld's GPU permission must exist in the **exact source revision** being
   built. The older v0.5.4 tag lacked the later NVIDIA exception. The distribution
   builder now pins `00a96b63c0ce45691d3dd9b4dcfd02069269c40c`, which adds that
   permission and generated third-party notices. Its GPL obligations remain.
6. The CUDA trainer distribution uses a separate vcpkg manifest without the
   optional GPL x264 encoder. It retains native NVIDIA video codecs and builds
   FFmpeg as LGPL 2.1-or-later. Upstream source is not modified. Corresponding
   source, build settings and notices accompany any redistributed trainer.

## Implemented GPU distribution infrastructure

* Verified, pinned prebuilt components, resumable downloads, background progress,
  user-selected storage and atomic installation receipts.
* Windows native equirectangular alignment uses the official COLMAP CUDA process
  for feature extraction/matching. Mapping remains explicitly identified as
  Ceres; no spherical Caspar acceleration is claimed.
* Managed trainer resolution takes priority over personal developer paths.
  Frozen installs cannot silently locate the developer's private environment.
* Real spherical/distorted camera metadata automatically selects LichtFeld GUT.
  No panorama-to-cubemap conversion is introduced.
* Failed trainer exits raise an error before inspecting old PLY outputs.
* One online installer can acquire all qualified default components itself.
  Users do not run Python, pip, Conda, a compiler or a CUDA toolkit installer.
  A supported NVIDIA driver remains a system prerequisite.
* Offline packaging has a separate redistribution gate. A CPU-only payload,
  missing trainer, wrong binary hash, perspective-only test or missing training,
  export/resume evidence cannot pass the full-payload checks.
* A pinned maintainer build recipe assembles a private local CUDA SDK from
  NVIDIA's hash-verified components. It does not change installed drivers/CUDA.
* Public publication continues to use an explicit source allowlist and immutable
  release tags. Client images, reconstructed scenes, credentials, local logs,
  model caches and private snapshots are excluded.

## Evidence and acceptance

The prior RTX 4090 acceptance registered 48/48 original panoramas in one model
in 56.387 seconds at width 2048 / 6000 features. This is integration evidence,
not surveyed pose accuracy or a fair CPU/GPU speed comparison. Source component,
full-payload, failed-training and distribution safety checks currently pass;
final counts and the final binary version belong in the release walkthrough.

The pinned Windows trainer compiled without upstream source changes. Its portable
install passed upstream notice generation and coverage checks for 72 vcpkg ports.
On a generated eight-camera equirectangular fixture, native GUT training completed
200 iterations, exported 1,076 finite Gaussians, saved a `.licht` project and
resumed to iteration 250 with a second validated export. Recorded loss was finite
(0.1940). A repeat in fresh output folders completed training/export/resume in
4.148 seconds; reported steady training time was 2.183 ms/iteration. The cold run
took 450.572 seconds including CUDA initialization/JIT. These small synthetic
tests establish execution, not real-scene quality or a general speed benchmark.
Upstream emitted `cuDevicePrimaryCtxRelease failed: 4` during shutdown despite
zero exits and validated exports; retain this limitation in candidate notes.

The user subsequently authorized a new 30-panorama fixture. All 30 registered in
one native spherical reconstruction in 47.620 seconds (width 2048, 6000 features,
CUDA extraction/matching and Ceres mapping). Native GUT training at width 512,
200 iterations and a 15,000 Gaussian cap exported 14,210 finite Gaussians, saved
projects and resumed to iteration 250 with another validated export. Recorded
loss was finite (0.3626); training/export/resume took 7.011 seconds with a warm
CUDA cache. Reported steady training time was 2.65 ms/iteration. This short test
is integration evidence, not final-quality convergence or surveyed accuracy.
All images, scenes, checkpoints and detailed logs remain local and excluded
from public export.

Final local v0.1.6.6 acceptance passed: frozen GUI, native-library self-test,
refinement worker and CLI dispatch; the M54 loopback check passed on the preceding
v0.1.6.5 build with unchanged viewer sources. An isolated app-ID installation
fetched both components from public GitHub URLs without Python/toolkit setup.
Installed frozen CLI training automatically selected GUT, trained the 30-image
fixture at width 1024 and exported 14,212 finite Gaussians. Same-version reinstall
and uninstall returned zero; uninstall removed the desktop runtime and retained
user component storage. This is not a real old-version migration or independent
clean-machine/hardware certification.

The trainer/runtime ZIP is 297,426,613 bytes, SHA256
`6dd2b6495340f7db1ed15401a29f624da83149a04a1d44c99004e1c5d29f3768`.
Matching source ZIP is 761,683,778 bytes, SHA256
`1b6d29c45e4ae4439ec64419e01ec96ff4aae71a044cbf54041fd92c99267bc0`.
Both, plus sanitized qualification, are public in the
[Windows runtime prerelease](https://github.com/saemranian/abu-align-train/releases/tag/gpu-runtimes-2026-10-18-1).
The local desktop installer SHA256 is
`35328fd854f2b352e70bea3ace672e6d394e337cc2ff908a701ef6000885a229`;
CI builds have their own manifest hashes. At the source commit, public platform
CI and final desktop release assembly are still to be checked before promotion.

## Modified implementation files

Desktop integration: `abu_align_gui.py`, `abu_components.py`,
`abu_engine_packs.py`, `abu_runtime.py`, `train_engine.py`.
Distribution: `.github/workflows/release.yml`, `installer/core-files.json`,
`installer/public-files.json`, `installer/engine-packs.json`, `installer/engines.json`,
`installer/build_gpu_trainer.py`, `installer/gpu-build-lock.json`,
`installer/package_gpu_trainer.py`, `installer/full_payload.py`,
`installer/windows/build_setup.py`, `installer/windows/abu_align_setup.iss`,
`installer/license-summary.txt`, `installer/prepare_candidate.py`,
`installer/github_draft.py`, `tests/test_full_payload.py`.
Documentation: README, THIRD_PARTY_NOTICES, RELEASE_GUIDE, GPU_INSTALLER_PROGRESS,
this review and walkthrough. VERSION, generated C# launchers and version headers,
Abu_Align mirrors, artifacts walkthrough and v0.1.6.5/v0.1.6.6 snapshots were
updated through the mandatory release protocol. Private build/test helpers and
artifacts are ignored and are not part of the public source manifest.

## Practical update workflow

"Push to GitHub" means: test the change; complete the mandatory version, launcher,
mirror and snapshot protocol; commit; run the safe export dry run; publish the
new allowlisted commit/tag; inspect the actual platform jobs and release assets.
An unchanged qualified engine archive can be reused by its immutable hash. Engine,
toolchain or adapter changes require relevant qualification again. Failed builds
or incomplete GPU coverage must not be promoted as a successful full release.

One installer **per OS/architecture** is practical. One executable that runs on
Windows, Linux and macOS is not the distribution format. A full Windows NVIDIA
candidate does not establish a full Linux CUDA or macOS Metal training release.
Signing/notarization identities and rights-holder permissions cannot be created
by the build script. The exact remaining platform and release gates must be
stated beside downloads.

Primary references: [COLMAP license scope](https://colmap.github.io/license.html),
[pinned LichtFeld GPU permission](https://github.com/MrNeRF/LichtFeld-Studio/blob/00a96b63c0ce45691d3dd9b4dcfd02069269c40c/LICENSE-EXCEPTION.txt),
[LichtFeld GPL license](https://github.com/MrNeRF/LichtFeld-Studio/blob/00a96b63c0ce45691d3dd9b4dcfd02069269c40c/LICENSE),
[CUDA 13 driver requirements](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/),
and THIRD_PARTY_NOTICES.md for engine-specific upstream sources.

Training adapter correction: resolution selection now matches exact aliases or the first GUI label token. `4k` and `3840` select 3840 pixels; `1024` selects 1024, and `1920` selects 1920. Regression checks inspect the launched native command, preventing substring collisions that previously changed quality and VRAM unexpectedly.
