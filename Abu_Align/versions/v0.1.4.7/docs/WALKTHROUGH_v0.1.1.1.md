# Walkthrough: Abu Align & Train v0.1.1.1
## Native RealityScan 2.2 Rig Recognition Architecture & XMP Metadata Schema

### Overview
In `v0.1.1.1`, we performed deep binary reverse engineering and comprehensive documentation analysis of the installed **RealityScan 2.2** engine (`C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe`, `Help\en-US\tools\xmpalign.htm`) to solve the issue where camera rigs imported via sidecar `.xmp` files were not recognized as a rigid multi-camera rig.

---

### Root Cause Analysis

1. **Why `Locked pose group` was empty**:
   - `Locked pose group` (CLI parameter `inpPosePriorRelativeGroup`) in the RealityScan UI is an arbitrary grouping field intended for manually or programmatically clustering unconstrained cameras during alignment.
   - Deep inspection of the `RealityScan.exe` binary strings and XMP descriptor tables revealed that `xcr:LockedPoseGroup` **does not exist in RealityScan's XMP parser code**. Any `<xcr:LockedPoseGroup>` element or attribute is completely disregarded by the XMP parser.
   - Guides 02, 03, and 04 assumed `LockedPoseGroup` was the XML tag because of the UI text label.

2. **The True RealityScan Rig Architecture**:
   - In RealityScan / RealityCapture, multi-camera rigs are defined exclusively through three dedicated attributes:
     - `xcr:Rig="{RIG_GUID}"`: Hardware rig definition GUID shared by all cameras.
     - `xcr:RigInstance="{FRAME_GUID}"`: Synchronous exposure / frame capture GUID shared by all cameras belonging to the same frame.
     - `xcr:RigPoseIndex="{CAMERA_INDEX}"`: Integer index (0, 1, 2, 3, 4, 5) designating each camera face.
     - `xcr:PosePrior="exact"`: Critical flag informing RealityScan that relative poses between cameras are rigid and mathematically fixed.
     - `xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"`: Official XML namespace with trailing `#`.

3. **Single Perspective Import Issue**:
   - In the user's screenshot, only images from the `perspectives/back/` folder were loaded (16 images, all `_back.jpg`).
   - A multi-camera rig requires at least two cameras sharing the same `RigInstance` GUID. When only `_back.jpg` is imported, each image is an isolated single-camera instance, preventing RealityScan from forming a rig constellation.
   - Importing all 6 perspectives (`front`, `back`, `left`, `right`, `up`, `down`) allows RealityScan to automatically match matching instances into complete 6-camera rigid units.

---

### Verification & Test Results

1. **RealityScan 2.2 Headless Engine Verification**:
   - Executed `RealityScan.exe -headless -newScene -addFolder ... -save test2.rcproj -quit` with our updated XMP files containing all 4 perspectives of frame 000.
   - RealityScan successfully generated the native rig project structure:
     ```xml
     <overrides>
         <rename type="rig" id="{6038A0B9-11D8-5A31-8B8C-A1AE9AA9A4F4}" name="{6038A0B9-11D8-5A31-8B8C-A1AE9AA9A4F4}"/>
         <rename type="rigInstance" id="{EAA5ECDA-F642-5CA2-8C3E-1D31C3634F1D}" name="{EAA5ECDA-F642-5CA2-8C3E-1D31C3634F1D}"/>
     </overrides>
     <source>
         <input fileName="frame_000_back.jpg" width="2400" height="2400" rigId="{6038A0B9-11D8-5A31-8B8C-A1AE9AA9A4F4}" rigInstance="{EAA5ECDA-F642-5CA2-8C3E-1D31C3634F1D}" rigCameraIndex="1"/>
         <input fileName="frame_000_front.jpg" width="2400" height="2400" rigId="{6038A0B9-11D8-5A31-8B8C-A1AE9AA9A4F4}" rigInstance="{EAA5ECDA-F642-5CA2-8C3E-1D31C3634F1D}" rigCameraIndex="0"/>
         <input fileName="frame_000_left.jpg" width="2400" height="2400" rigId="{6038A0B9-11D8-5A31-8B8C-A1AE9AA9A4F4}" rigInstance="{EAA5ECDA-F642-5CA2-8C3E-1D31C3634F1D}" rigCameraIndex="2"/>
         <input fileName="frame_000_right.jpg" width="2400" height="2400" rigId="{6038A0B9-11D8-5A31-8B8C-A1AE9AA9A4F4}" rigInstance="{EAA5ECDA-F642-5CA2-8C3E-1D31C3634F1D}" rigCameraIndex="3"/>
     </source>
     ```
2. **Dataset Migration**:
   - Migrated all **4,092 `.xmp` files** in `VID_20260902_094505_00_003_frames\CC\Masked\pinhole_rig\perspectives` across all 6 perspectives (`front`, `back`, `left`, `right`, `up`, `down`).
3. **Unit Tests**:
   - `test_rig_cameras_csv.py`: 10/10 passed.
   - `test_splitter_masks.py`: 4/4 passed.
   - `test_realityscan_engine.py`: 5/5 passed.
4. **Launchers**:
   - `Abu_Align.exe` and `Abu_Align_and_Train.exe` recompiled cleanly.
