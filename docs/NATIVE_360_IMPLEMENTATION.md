# Native 360 implementation ledger

The original equirectangular images remain the alignment and training inputs.

1. Geometry/export integrity: implemented in v0.1.3.7. Quaternion order,
   half-turn rotations, upstream/fallback observations, and row-wise Gaussian
   validity. Real 48-camera model preserves 210,526 observations with maximum
   rotation matrix error 7.4e-16. Ten targeted regression tests pass.
2. Reproducible image manifest and engine provenance: next.
3. Independent trajectory, geometry, rendering and memory benchmarks: pending.
4. 360DVO masks, persistent input selection, explicit fallback and safe presets: pending.
5. OpenSfM matching and joint global refinement: pending.
6. Shared spherical manual controls: pending.
7. Memory-bounded PFGS360 datasets/depth and runtime Gaussian cap: pending.
8. Distinct faithful pose-free/fixed-pose/constrained-refinement schedules: pending.
9. Isolated native COLMAP/new panoramic trainer comparisons: pending.
10. Promote only benchmark-validated presets and release: pending.

Installation availability, smoke checks, and real accuracy benchmarks are
different acceptance criteria. No unpublished accuracy or speedup is claimed.
