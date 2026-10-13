# Abu Align & Train v0.0.5.6 — Fix Unwanted 360 Panoramic Splitting on 2:1 Aspect Ratio Perspective Images

## Overview
Version **v0.0.5.6** eliminates a critical auto-detection false positive in `align_cli.py` where regular perspective cameras (such as iPhone photos/video frames with a 2:1 / 18:9 aspect ratio, e.g. 1920x960) were erroneously classified as 360° omnidirectional equirectangular panoramas and sliced into 8-camera virtual pinhole rigs. 

360° rig projection is now **strictly opt-in** and will only trigger if explicitly requested by the user.

---

## What Happened in the Screenshot?

### 1. Log & Screenshot Evidence (`media_1790282765447.png`)
In the user's run:
- **Dataset**: `...\Schemata IS4S Huntsville Iphone\...\Sel Copy Del 2` (55 iPhone frames).
- **Resolution**: `1920 x 960` (Aspect ratio = 2.0).
- **Engine Selected**: `vggt_omega` (Neural Perspective SfM).
- **Observed Behavior**:
  ```text
  [1/5] Scanning Image Dataset...
        * Image Size:    1920 x 960 (1.8 MP)
        * Target Output: ...\results\sparse_02_omega
        * [360 RIG PROJECTOR] 360° Equirectangular dataset detected (1920x960).
        * Projecting into RING multi-camera pinhole rig (8 views/pano, 90.0° FOV, 1024x1024px)...
          Splitting panorama 1/55: frame000035.png
          ...
        * [OK] Generated 440 pinhole views with transforms.json in ...\rig_ring_frames
        * Projected rig pinhole views for alignment: 440 (1600x1025)
  ```
- In File Explorer, 440 sliced ring views with pitch and yaw angles (`frame000041_ing 37 yaw015...`) were created in `rig_ring_frames`.
- VGGT-Omega was then launched on the 440 split views instead of the 55 original iPhone frames.

### 2. Root Cause Analysis
- **Code Location**: `align_cli.py`, line 1360:
  ```python
  is_equirectangular = (orig_w == 2 * orig_h)
  needs_splitting = (
      getattr(args, "split_panorama", False)
      or getattr(args, "split_only", False)
      or getattr(args, "cubemap_transcode", False)
      or (is_equirectangular and args.engine not in ["opensfm", "360dvo", "dvo360", "pfgs360"])
  )
  ```
- **Why It Triggered**:
  - The iPhone frames in `Sel Copy Del 2` were `1920x960`.
  - The naive check `orig_w == 2 * orig_h` evaluated to `True`.
  - Because `vggt_omega` is not in `["opensfm", "360dvo", "dvo360", "pfgs360"]`, the condition forced `needs_splitting = True`.
- **Why This Was Wrong**:
  - A 2:1 aspect ratio (Univisium / 18:9) is common in smartphone video, anamorphic capture, and custom crops. It does **not** make an image an equirectangular 360° sphere.
  - Perspective neural models like VGGT-Omega should **never** split perspective images.
  - Slicing perspective images into spherical ring angles distorted the projection and multiplied the frame count from 55 to 440.

---

## Fixes Implemented in v0.0.5.6

### 1. Eliminated Naive Aspect Ratio Auto-Splitting (`align_cli.py`)
Removed `(is_equirectangular and args.engine not in ...)` from `needs_splitting`. Rig projection and panoramic splitting are now **strictly explicit**:
```python
# 360° Multi-Camera Rig Projector & Panoramic Splitter (Phase 3)
# ONLY perform rig projection / splitting if explicitly requested by user CLI/GUI flags
needs_splitting = (
    getattr(args, "split_panorama", False)
    or getattr(args, "split_only", False)
    or getattr(args, "cubemap_transcode", False)
)
```
- Standard perspective datasets (including 2:1 aspect ratio images like 1920x960) remain untouched as perspective pinholes.
- 55 frames will now align as exactly 55 perspective frames in VGGT-Omega.

### 2. Fixed Camera Model Fallback Reporting (`align_cli.py`)
In the terminal telemetry summary, if a camera reconstruction metadata object is empty, the report now correctly distinguishes perspective from spherical engines:
```python
if args.engine in ["360dvo", "dvo360", "pfgs360"]:
    report_lines.append(f" Camera Model:         EQUIRECTANGULAR (Omnidirectional 360°)")
else:
    report_lines.append(f" Camera Model:         PINHOLE / PERSPECTIVE")
```

---

## Verification
- Validated via unit test: 1920x960 perspective images with `--engine vggt_omega` now evaluate `needs_splitting = False`.
- Verified that explicit 360 commands (`--split_panorama`, `--cubemap_transcode`) continue to function when intentionally requested.
- Bumped version to `v0.0.5.6` with synchronized GUI, CLI, and executables.

---

## Modified Files
- `align_cli.py`
- `VERSION.txt`
- `abu_align_gui.py`
- `desktop_ui.py`
- `train_engine.py`
- `launcher.cs`
- `Abu_Align.bat`
- `Abu_Align_and_Train.bat`
