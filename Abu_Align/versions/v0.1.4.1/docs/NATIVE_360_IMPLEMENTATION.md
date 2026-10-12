# Native 360 implementation ledger

All user panoramas remain on this PC. Local tests used the authorized 48-image
Container_03 folder. Alignment and training use equirectangular images directly;
no cubemap conversion is part of these routes.

## Implemented releases

1. v0.1.3.7: fixed quaternion/half-turn exports and retained OpenSfM observations.
   Existing 48-camera model preserves 210,526 observations; rotation matrix
   round-trip error at most 7.4e-16.
2. v0.1.3.9: ordered content manifests, exact DVO selection, neural polar masks,
   tracked WSL runner, explicit fallback identity and safe width limit 1024.
   Masked neural DVO completed a local 16-frame smoke test.
3. v0.1.4.0: manual spherical tie/GCP/check/distance controls, Prepare-menu editor,
   joint camera/landmark refinement, independent trajectory and weighted-PSNR
   utilities. Synthetic accuracy tests include withheld checks and seam wrapping.
4. Current release: bounded PFGS360 image/depth access, immutable actual-resolution
   training inputs, disjoint splits, explicit pose modes, runtime Gaussian cap,
   fixed-camera invariant, CPU checkpoint export, unique per-run outputs, and
   native COLMAP candidate accessible from the alignment GUI and CLI.

## Local acceptance evidence (2026-10-12)

- Native COLMAP 4.2.0 CLI: 48/48 images, one component, 5,866 landmarks and
  30,600 observations; complete wrapper/export runtime 41.4 seconds at feature
  width 1024 and budget 3000. Median angular residual 0.05261 degrees; 95th
  percentile 0.13609 degrees. These are fitted residuals, not absolute accuracy.
- PFGS360 full wrapper: 48 panoramas resized to actual 480x240 images, 43 training
  and five held-out views, five iterations, 8192 Gaussians, depth and sky masking,
  fixed cameras, and verified binary Gaussian PLY export. Runtime 47.62 seconds
  for training worker; peak allocated VRAM 4,743,700,992 bytes (4.42 GiB), reserved
  VRAM 4,838,129,664 bytes (4.51 GiB), peak RAM 4,469,325,824 bytes (4.16 GiB).
  Camera pose parameter drift relative to initialization: exactly 0.0.
- 28 focused geometry, manifest, memory, training-input and desktop tests pass.
- Export validates Gaussian attributes and finite values, and fails explicitly
  when the current run has no usable export.

## Gates and remaining research work

The installed native PyCOLMAP is CPU-only; selecting CUDA fails explicitly.
Spherical Ceres BA runs on CPU. Caspar is not enabled for spherical cameras.
SIFT guided matching aborts in this installed build on spherical cameras and is
therefore disabled. Global mapping is selectable but not locally promoted.

PFGS360 fixed-pose training is tested. Refinement is an experimental soft absolute
pose prior, not direct GCP-constrained training; pose-free still follows upstream
incremental growth. The posed initializer/schedule is a locally modified upstream
integration and is not equivalent to a reproduced published benchmark. Long-run
rendering quality, density schedules, component merging, persistent neural DVO
track retention and competing panoramic trainer/retrieval backends remain gated.
Depth cache currently includes image content, image shape and adapter source;
locking external model weights/revisions is still a reproducibility improvement.

No SOTA ranking, surveyed accuracy, full-resolution 8K memory claim or production
render-quality improvement is inferred from these smoke checks. Independent
surveyed controls/reference poses and longer held-out rendering comparisons are
needed for such claims.

## Primary upstream references

- [COLMAP camera/solver releases](https://colmap.github.io/changelog.html)
- [COLMAP solver FAQ](https://colmap.github.io/faq.html)
- [PFGS360 official code](https://github.com/zcq15/PFGS360)
- [PFGS360 CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_Pose-Free_Omnidirectional_Gaussian_Splatting_for_360-Degree_Videos_with_Consistent_Depth_CVPR_2026_paper.html)
- [360DVO author project](https://chris1004336379.github.io/360DVO-homepage/)
