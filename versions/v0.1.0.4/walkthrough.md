# v0.1.0.4 — 360 Splitter: Perspective Suffix Naming & Clean Rig Orientation Override

## Overview & Outcome
Abu Align & Train **v0.1.0.4** enhances the **360° Panoramic Splitter & Rig Projector** with proper perspective image naming and decoupled orientation overriding:

1. **Perspective-Specific Image Suffix Naming**:
   * **Problem**: When exporting panoramas into perspective subfolders (`perspectives/<cam_name>/`), files previously retained the panorama base name without the camera identifier (e.g. `Container1_Mask000.jpg` in all 6 folders), causing ambiguity when mixing or referencing files externally.
   * **Solution**: Every exported image in perspective subfolders is now uniquely named with its camera angle suffix:
     * `perspectives/back/Container1_Mask000_back.jpg`
     * `perspectives/front/Container1_Mask000_front.jpg`
     * `perspectives/right/Container1_Mask000_right.jpg`
     * `perspectives/left/Container1_Mask000_left.jpg`
     * `perspectives/up/Container1_Mask000_up.jpg`
     * `perspectives/down/Container1_Mask000_down.jpg`
   * **Manifest Synchronization**: Both the local `perspectives/<cam>/cameras.csv` and master `cameras.csv` index these exact filenames, guaranteeing 100% path parity when dragging individual folders into RealityCapture or RealityScan.

2. **Clean Rig Orientation Override in `cameras.csv`**:
   * **Problem**: In previous versions, when importing 360DVO trajectory priors, visual odometry drift and heading adjustments slightly shifted the exported `yaw`, `pitch`, and `roll` values across frames (e.g., `yaw=180.1318°` in frame 003).
   * **Solution**: `compound_realitycapture_pose()` now takes metric camera positions $(x, y, \text{alt})$ from the 360DVO trajectory prior, discards/deletes the imported visual odometry rotation angles, and overrides them with the clean, canonical RealityCapture Euler angles defined by the splitting rig:
     * **Back**: `yaw = 180.0000°, pitch = 90.0000°, roll = 0.0000°`
     * **Front**: `yaw = 0.0000°, pitch = 90.0000°, roll = 0.0000°`
     * **Right**: `yaw = 90.0000°, pitch = 90.0000°, roll = 0.0000°`
     * **Left**: `yaw = 270.0000°, pitch = 90.0000°, roll = 0.0000°`
     * **Up**: `yaw = 0.0000°, pitch = 180.0000°, roll = 0.0000°`
     * **Down**: `yaw = 0.0000°, pitch = 0.0000°, roll = 0.0000°`
     * **Custom Rigs**: Any custom rig camera angles configured in the UI are cleanly converted into constant RealityCapture Euler angles across all trajectory frames.
   * **Optional Compounding**: Added `--compound-rotations` CLI flag if full $SO(3)$ rotation compounding is desired.

3. **In-Place Migration of Existing Dataset**:
   * Migrated all 4,092 image files across the 6 perspective folders in:
     `C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Insta360\Insta Exports\VID_20260902_094505_00_003_frames\CC\Masked\pinhole_rig\perspectives`
   * Updated all local `cameras.csv`, master `cameras.csv`, and `transforms.json` with the new naming scheme and clean rig orientations.

---

## Verification & Automated Test Results

* **Dedicated Rig Test Suite**: [`tests/test_rig_cameras_csv.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/tests/test_rig_cameras_csv.py) (10 tests, 0.103s)
  * Verified perspective subfolder image naming suffix generation (`pano_001_<cam>.jpg`).
  * Verified default orientation override with clean rig angles while preserving metric $(x, y, \text{alt})$.
  * Verified optional $SO(3)$ rotation compounding mode.
* **Full Regression Test Suite**:
  * **135 tests passed in 42.869s** (100% pass rate, 0 failures).
* **Upstream Cleanliness**:
  * `vgg-ttt/`, `vggt-omega/`, and WSL `/home/ahmad/360DVO/` remain 100% clean and untouched.
* **SHA256 Parity**:
  * `360dvo_engine.py` and `dvo360_engine.py` match SHA256 exactly (`60CF1F1A0E51DC0D25C20702623C658A25F463A4F46B7771B25E3BF760B8ECAC`).
