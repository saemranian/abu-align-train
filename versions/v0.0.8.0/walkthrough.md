# v0.0.8.0 — NVIDIA VGG-T³ Precision Architecture Fix & Alignment Validation

## Overview & Incident Diagnosis
- Diagnosed and resolved the fatal PyTorch LayerNorm type mismatch reported in `abu_align_2026-09-26_13-27-29.log`:
  ```
  RuntimeError: expected scalar type Float but found BFloat16
  ```
- **Root Cause**:
  In `vggttt_engine.py`, `load_vggttt_model` was casting the entire neural network to `torch.bfloat16` when run on compute capability >= 8.0 GPUs (e.g. RTX 4090).
  NVIDIA's VGG-T³ reference design operates as a hybrid precision system:
  1. Primary network weights and heads reside in FP32 (`torch.float32`).
  2. Heavy sequence attention in `aggregator` runs with `torch.autocast("cuda", dtype=dtype)`.
  3. Geometric estimation modules (`camera_head`, `depth_head`, unprojection, SE3 inversion) execute in FP32 with autocast disabled (`torch.autocast("cuda", enabled=False)`) for numerical stability.
  Casting `camera_head` weights to BFloat16 caused `F.layer_norm` inside `token_norm` to fail when encountering FP32 camera tokens.

## Key Changes
1. **Engine Model Loader (`abu align/vggttt_engine.py`)**:
   - Updated `load_vggttt_model()` to maintain network weights in `Float32` on CUDA, aligning with NVIDIA's reference specification (`VGGT.from_pretrained(...).eval().cuda()`).
   - Kept mixed-precision acceleration intact by passing `dtype=dtype` (e.g. `torch.bfloat16`) to `model.infer()`, enabling fast autocast attention while preserving FP32 precision for geometry heads.

2. **Upstream Architecture Hardening (`vgg-ttt/vggttt/nets/vggt/`)**:
   - `models/vggt.py`: Explicitly cast input camera tokens to `camera_head` parameter dtype before invocation inside the `autocast(enabled=False)` block.
   - `heads/camera_head.py`: Added explicit dtype validation in `forward()` (`token_norm`) and `trunk_fn()` (`embed_pose`), ensuring input tokens automatically match module weight dtypes regardless of user-specified runtime casting.
   - `heads/dpt_head.py`: Added defensive type alignment in `apply_proj_resize()` before `self.norm(x)` LayerNorm execution.

## Verification & Validation Results
- **Isolated Unit Test**: Verified that `VGGT.infer()` on dummy input with `dtype=torch.bfloat16` succeeds and outputs clean `pose` (shape: `[2, 4, 4]`) and `pts3d` (shape: `[2, 518, 518, 3]`).
- **End-to-End CLI Pipeline Run**:
  Executed `align_cli.py --engine vggttt --stride 15 --vggttt_conf 2.0 --vggttt_ttt_steps 1` on the user dataset:
  - Input: 50 images from `Schemata IS4S Huntsville Iphone/Sel Copy Del 4`
  - Registered Views: **4 / 4 (100.0%)** keyframes
  - 3D Points Created: **183,824 clean vertices**
  - Exported standard packages:
    - `3dgs_dataset/` (`sparse/0/`, `images/`, `transforms.json`)
    - `COLMAP_Lichtfield/` (`cameras.bin/.txt`, `images.bin/.txt`, `points3D.bin/.txt`)
    - `Postshot_RealityCapture/` (`cameras.csv`, `transforms.json`, `points.ply`)
    - Root point cloud: `points3D.ply`
  - Total Runtime: **24.1s** (0.353s/frame inference)
  - Exit Code: **0** (Complete Success)

## Release & Versioning
- Base-10 version bump: `v0.0.7.9` -> `v0.0.8.0`.
- All entrypoints, version banners, C# launchers, batch scripts, and version snapshots synchronized.
