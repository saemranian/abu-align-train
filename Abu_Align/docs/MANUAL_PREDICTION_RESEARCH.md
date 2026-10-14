# Manual point prediction research — 2026-10-13

The old implementation already projected into every registered camera. Its ranking
favoured proximity to measured cameras, and it could not look at image content or
unregistered originals. A projected pixel is not evidence of visibility, identity
or absence of occlusion. More candidate names alone would not solve those limits.

## Reviewed primary implementations

| Method | Evidence and suitability for this app |
|---|---|
| [ALIKED + LightGlue](https://github.com/cvg/LightGlue) | Learned sparse local features and adaptive transformer matching; supports CUDA and CPU. The installed local library and ALIKED weights made a tested Windows adapter feasible. It provides correspondences, not arbitrary control-point measurements by itself. |
| [RoMa v2](https://github.com/Parskatt/RoMaV2) | Newer dense correspondence model, with official MegaDepth/ScanNet evaluations and overlap/precision outputs. Its reference environment is tested on Linux/Python 3.12 and includes DINOv3 with a separate licence. It is not installed or benchmarked in this Windows environment, so it is not labelled production-ready here. |
| [Efficient LoFTR](https://github.com/zju3dv/EfficientLoFTR) | Semi-dense matching with official evaluation code; a Transformers integration was announced in 2025. Dense neighbourhood evidence is attractive for difficult texture, but the reference setup/weights and panorama adaptations need their own local acceptance tests. |
| [XFeat + LighterGlue](https://github.com/verlab/accelerated_features) | Compact features and a smaller learned matcher. The official repository reports CPU/GPU and MegaDepth results. A promising efficiency alternative, not a claim that it is more accurate on arbitrary panorama points. |
| SIFT + mutual ratio filtering + robust local transform | Available locally without weights, useful for textured scenes and a transparent CPU fallback. The synthetic plane test was faster with SIFT than with the learned pipeline. |

These public benchmarks evaluate conventional image-pair matching/pose tasks.
They do not establish accuracy of a chosen feature in full equirectangular images.
No ranking from them is copied into an unsupported “best 360 predictor” claim.
The shipped learned method is modern, tested ALIKED/LightGlue; it is not claimed
to be the newest model or universally superior to the reviewed dense methods.

## App-owned integration

The predictor runs as an isolated, cancellable local worker. The UI and control
document stay separate from image decoding and inference. A fixed memory cache
retains two decoded previews; feature count and support images are bounded. Target
coverage has no nearest-camera cap: all scope visits every enabled original.

Registered views combine geometric locations with visual correspondence search.
Unregistered panoramas use overlapping local views across longitude and latitude;
uncalibrated perspective photos use native pixel search without invented optics.
Small in-memory tangent patches correct local spherical distortion. Original
alignment inputs remain equirectangular; no cubemap images or substitute alignment
dataset are created. Proposals map back to original pixels with pixel-centre
resampling corrections.

Sparse source context is widened adaptively. Robust local transforms require
multiple inliers, bounded residuals and local support around the query. Expanded
support must surround it, avoiding distant extrapolation. Masks reject excluded
pixels; a geometric displacement gate rejects grossly inconsistent registered
proposals. Results compare up to three measured source views and explicitly label
single-view evidence, agreement and ambiguity. Occlusion and repeated patterns
remain real limits. A score is a heuristic, not a calibrated probability.

The UI exposes Matching, Hardware and Scope, live coverage/progress, Cancel and
Open & focus. Nothing becomes a measurement without the user marking the feature.
Unknown images can be inspected/measured; initial Align must register them before
they participate in Refine. Existing geometric suggestions remain available.

## Privacy and upstream boundary

Setup downloads only the two public checkpoints into an ignored app model cache
and records their SHA256 hashes. During image inference, outbound socket connections
are blocked. Checkpoints load from local paths with tensor-only loading; the
installed libraries' loader is redirected in the worker's memory only. No library
or upstream repository file is patched. No hosted image-matching API is used.

Alignment masks use existing OpenSfM/COLMAP interfaces and an app-owned 360DVO
wrapper. The native adapter zeros source/target mask weights and filters masked
cloud patches, recording rejection counts. This extends the app's runner rather
than editing 360DVO or OpenSfM source. See the versioned verification record for
local runtime tests, memory measurements and the exact supported engine scope.

Further dense matching adapters should be added only after original-pixel mapping,
mask handling, cancellation/offline behaviour, ambiguity rejection and held-out
native-panorama accuracy tests pass on this PC. They must retain the same proposal
and control-document contracts rather than create another measurement pipeline.
