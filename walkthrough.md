# Walkthrough: Abu Align & Train v0.1.3.4
## 360 Splitter: Up/Down Zenith/Nadir Inversion Fix & Full Dataset True XMP Application

### 1. Executive Summary
Release **v0.1.3.4** addresses the root causes identified in the GPT Camera Rig audit (`CAMERA_RIG_README.md`), fixes a fundamental pitch sign inversion in the 360 Splitter's equirectangular remapper, and updates the entire 3,558-image dataset at:
`C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Insta Exports\VID_20260902_094505_00_003_frames\CC\Masked\pinhole_rig\perspectives`

---

### 2. Issues Discovered in the Audit & GPT Solutions

#### Issue 1: Pitch Inversion in Equirectangular Remapping (`up` = floor, `down` = ceiling)
- **Root Cause**: In `panoramic_splitter.py`, the rotation matrix generator `euler_to_rotation_matrix()` had inverted `sin(pitch)` signs in `Rx`. When sampling rays via `rays_cam @ R.T`, a positive pitch ($+90^\circ$ for `up`) rotated camera forward vector into $-Y$ (latitude $-90^\circ$, nadir/floor), while negative pitch ($-90^\circ$ for `down`) rotated into $+Y$ (latitude $+90^\circ$, zenith/ceiling).
- Meanwhile, `basis()` in the metadata compounding code treated positive pitch as pointing up ($+Z$). As a result:
  - The image saved as `up.jpg` contained the **floor/tripod**, but its metadata claimed it looked at the **ceiling**.
  - The image saved as `down.jpg` contained the **ceiling**, but its metadata claimed it looked at the **floor**.
- **RealityScan Impact**: When SIFT tie points on the floor matched with the bottom of `front.jpg` and `up.jpg`, RealityScan looked at `up.jpg`'s metadata (pointing at the ceiling) and flagged massive reprojection errors or shattered the component.
- **GPT Correction**: GPT discovered this via pixel feature correspondences and swapped the base orientation matrices of `up` and `down` for the existing image files.
- **Code Fix in `panoramic_splitter.py`**: Fixed `Rx` in `euler_to_rotation_matrix` so that $+90^\circ$ pitches upward to $+Y$ (Zenith/Ceiling) and $-90^\circ$ pitches downward to $-Y$ (Nadir/Floor), ensuring all future extractions are physically and geometrically coherent.

#### Issue 2: Uniform World Orientations vs. True Trajectory Orientation
- **Root Cause**: The original split had `override_orientations_from_rig = True`, which discarded the panorama's actual heading (`yaw`, `pitch`, `roll`) from the 360DVO trajectory and stamped every single station with identity (`yaw=0.0000`, `pitch=90.0000`). With `PosePrior="locked"`, this told RealityScan that the camera never rotated while walking.
- **GPT Correction**: GPT estimated each station's true orientation $Q_{\text{station}}$ from image features, then applied $R_{\text{face}} = B_{\text{face}} @ Q_{\text{station}}$.
- **Dataset Update**: For the 36 stations (Mask042 to Mask077), the verified GPT solutions were preserved. For the remaining 557 stations (Mask000 to Mask041, Mask078 to Mask592), station rotations $Q = [r, f, u]^T$ were computed from `sparse_01_360dvo\Postshot_RealityCapture\cameras.csv`, ensuring every station reflects its true heading and orientation.

#### Issue 3: Schema Conformance to `True_Schema_360_Camera_Rig.xmp`
- Cleanly generated all 3,558 XMP sidecars with:
  - `xcr:PosePrior="locked"`
  - `xcr:DistortionModel="perspective"`
  - `xcr:DistortionPrior="exact"`
  - `xcr:DistortionCoeficients="0 0 0 0 0 0"`
  - `xcr:FocalLength35mm="11.6893366775552"` (13 decimal places)
  - `<xcr:Rotation>` and `<xcr:Position>` child nodes
  - Full removal of `Rig`, `RigInstance`, `ComponentId`, and `LockedPoseGroup`.

---

### 3. Dataset Update Summary

| Action | Path / Target | Count | Status |
| :--- | :--- | :--- | :--- |
| **Backup** | `pinhole_rig\Original_XMP_Backup_20261009_001000` | 3,558 XMP files | **Backed up safely** |
| **GPT Verified Copy** | Stations `Container1_Mask042` to `Mask077` | 216 XMP files (36 stations) | **Copied & verified** |
| **Trajectory Update** | Stations `Mask000`–`Mask041` & `Mask078`–`Mask592` | 3,342 XMP files (557 stations) | **Generated & applied** |
| **Total Sidecars** | `perspectives/` (`front`, `back`, `left`, `right`, `up`, `down`) | **3,558 XMP files** | **100% Complete** |

---

### 4. Verification and Regression Tests

- `tests/test_cubemap_rs.py`: **PASSED (100%)**
- `tests/test_rig_cameras_csv.py`: **PASSED (100%, 10 tests)**
- `tests/test_splitter_masks.py`: **PASSED (100%, 4 tests)**
