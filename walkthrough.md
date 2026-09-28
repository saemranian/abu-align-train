# v0.1.0.5 — 360 Splitter: RealityScan Rig Parser XMP Sidecars & all_cameras.csv

## Overview & Outcome
Abu Align & Train **v0.1.0.5** expands the **360° Panoramic Splitter & Rig Projector** with automated multi-camera rig export for Epic Games RealityScan and RealityCapture:

1. **`all_cameras.csv` Master Manifest Next to Perspective Subfolders**:
   * **Location**: Generated directly in `perspectives/all_cameras.csv`, alongside perspective subfolders (`back`, `down`, `front`, `left`, `right`, `up`).
   * **Content**: Consolidates all extracted perspective views across all frames in a single unified table with relative subfolder paths (e.g. `front/Container1_Mask000_front.jpg,...`).
   * **Per-Perspective Tables**: Renamed generic `cameras.csv` inside each perspective folder to `cameras_<cam_name>.csv` (e.g., `cameras_front.csv`, `cameras_back.csv`, `cameras_right.csv`, `cameras_left.csv`, `cameras_up.csv`, `cameras_down.csv`) for clear identification when dragging individual perspective folders into photogrammetry tools.

2. **RealityScan / RealityCapture Rig Parser `.xmp` XML Sidecars**:
   * **Background**: Splitting 360 equirectangular panoramas into 6 cubemap faces results in mathematically exact 90° orthogonal orientations and identical shared optical centers $(0, 0, 0\text{ offset})$, even when trajectory positions have minor SLAM/GPS uncertainty.
   * **Rig Constraints**: For every extracted view, the splitter generates sidecar `.xmp` files (`<image>.jpg.xmp` and `<image>.xmp`) complying with the CapturingReality `mxmp 1.0` XML schema:
     * `<xcr:RigId>`: Groups all synchronized cubemap cameras into a rigid camera rig (`cubemap_rig`).
     * `<xcr:RigInstanceId>`: Shared per-frame exposure key (`{FRAME_ID}` / `base_name`, e.g. `Container1_Mask000`) binding all 6 perspective faces of each frame together.
     * `<xcr:Position>`: Absolute $(x, y, z)$ position from the SLAM/GPS trajectory prior.
     * `<xcr:Rotation>`: Row-major $3 \times 3$ rotation matrix ($R_{00}\dots R_{22}$) representing the camera's orthogonal orientation in RealityCapture coordinate space.
     * `<xcr:FocalLength35mm>`: 35mm equivalent focal length calculated from camera FOV and aspect ratio.
     * `<xcr:PrincipalPointU>` / `<xcr:PrincipalPointV>`: Exact principal points (`0.000000 0.000000`).
     * `<xcr:DistortionModel>`: Brown polynomial model (`brown3`).
     * `<xcr:DistortionCoeficients>`: `0 0 0 0 0 0` (unwrapped rectilinear pinhole).
     * `<xcr:CalibrationPrior>` & `<xcr:LensDistortionPrior>`: Set to `locked`.
     * `<xcr:PosePrior>`: Set to `approximate` with `PoseAccuracyPosition` ($0.3\text{m}$) and `PoseAccuracyRotation` ($0.001\text{ rad}$), allowing RealityScan's Rig Parser to solve the trajectory as a rigid multi-camera body without drifting faces apart.

3. **In-Place Migration of Existing Dataset**:
   * Executed on:
     `C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Insta Exports\VID_20260902_094505_00_003_frames\CC\Masked\pinhole_rig\perspectives`
   * Renamed all 6 perspective CSV files: `cameras.csv` -> `cameras_<cam_name>.csv`.
   * Generated `all_cameras.csv` (4,092 view rows) in `perspectives/`.
   * Generated 4,092 paired `.xmp` XML metadata sidecars (`.jpg.xmp` and `.xmp`) with exact RigId, RigInstanceId, and 3x3 rotation matrices across all 6 perspectives.

---

## Verification & Automated Test Results

* **Dedicated Rig Test Suite**: [`tests/test_rig_cameras_csv.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/tests/test_rig_cameras_csv.py) (10 tests, 0.200s)
  * Verified `perspectives/all_cameras.csv` creation with correct row count and paths.
  * Verified `cameras_<cam_name>.csv` in all perspective subfolders.
  * Verified XML schema validity, `<xcr:RigId>`, `<xcr:RigInstanceId>`, `<xcr:Rotation>`, and `<xcr:Position>` in `.xmp` sidecars.
* **Full Regression Test Suite**:
  * **135 tests passed in 42.802s** (100% pass rate, 0 failures).
* **Upstream Cleanliness**:
  * Upstream submodules (`vgg-ttt/`, `vggt-omega/`) and WSL `/home/ahmad/360DVO/` remain 100% clean and untouched.
* **SHA256 Parity**:
  * Byte-for-byte identity maintained between `360dvo_engine.py` and `dvo360_engine.py` (`60CF1F1A0E51DC0D25C20702623C658A25F463A4F46B7771B25E3BF760B8ECAC`).
