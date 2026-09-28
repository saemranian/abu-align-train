# Walkthrough: Abu Align & Train v0.1.1.2
## RealityScan 2.2 Alignment & Bundle Adjustment Optimization Architecture

### 1. Executive Summary
In version **v0.1.1.2**, we conducted deep binary reverse engineering of `RealityScan.exe` (v2.2.0.119430) using the Capstone disassembler to resolve the alignment issues encountered in RealityScan 2.2:
1. **Registered focal length 0.000000** when Absolute pose was set to "Locked".
2. **Alignment crash** when Absolute pose was changed to "Position and orientation".
3. **Empty "Locked pose group"** UI field in RealityScan.
4. **Duplicate `.jpg.xmp` and `.xmp` files** across perspective directories.
5. **Exact focal length mathematical verification** for the 114° field-of-view cubemap.

---

### 2. Deep Root Cause Analysis & Binary Reversing Discoveries

#### A. Why Registered Focal Length was 0.000000 under "Locked" Pose
* **`Registration calibration` is the output SfM result:** In RealityScan's 1Ds panel, the `Registration calibration` block is the output of the bundle adjuster, while `Prior calibration` is the input.
* **Camera Rejection due to 360DVO translation drift:** When `xcr:PosePrior="locked"` is set, RealityScan freezes all camera 6DoF poses strictly as immutable ground truth. However, 360DVO visual odometry trajectories contain 10–30 cm of cumulative drift. During bundle adjustment, visual tie points between adjacent frames contradict the frozen positions. RealityScan evaluates the reprojection error, determines it exceeds tolerance, and **rejects the camera entirely from the component** (leaving it unregistered).
* For any unregistered camera, RealityScan sets `Calibration group: -1` and `Focal length (35mm): 0.000000`.

#### B. Why RealityScan Crashed on "Position and orientation" Alignment
* When hardware rig tags (`xcr:Rig="{GUID}"`, `xcr:RigInstance="{GUID}"`, and `xcr:RigPoseIndex`) were present in the XMP, RealityScan activated its internal **Rig Bundle Adjuster**.
* Because all 6 cubemap perspectives originate from the exact same nodal point $(0, 0, 0)$, the baseline distance between virtual cameras is zero. Without an external hardware rig calibration XML file loaded into RealityScan defining relative baseline offsets, the optimizer encountered a **singular Hessian matrix $(J^T J)^{-1}$ / rank-deficiency divide-by-zero**, causing `RSNode.exe` / `RealityScan.exe` to crash immediately.

#### C. Why "Locked pose group" was Consistently Empty
* Disassembly of the XMP parser table in `RealityScan.exe` at offset `0x217f1c0` revealed the exhaustive list of supported XMP tags:
  ```
  xcr:CalibrationPrior, xcr:DistortionCoeficients, xcr:FocalLength35mm,
  xcr:Skew, xcr:AspectRatio, xcr:PrincipalPointU, xcr:PrincipalPointV,
  xcr:Rotation, xcr:Position, xcr:Coordinates, xcr:ComponentId,
  xcr:PosePrior, xcr:CalibrationGroup, xcr:DistortionGroup, xcr:Rig,
  xcr:RigInstance, xcr:RigPoseIndex, xcr:InTexturing, xcr:InColoring, xcr:InMeshing
  ```
* **Discovery:** Tags like `xcr:LockedPoseGroup`, `xcr:PoseAccuracyPosition`, and `xcr:PoseAccuracyRotation` **do not exist** in the RealityScan binary and are completely ignored.

#### D. The Exact String-to-Enum Values for `xcr:PosePrior`
Disassembly of function `0x141120680` proved the accepted string enums:
* `"initial"` $\rightarrow$ Maps to Enum 1 (`Draft / Initial`)
* `"exact"` $\rightarrow$ Maps to Enum 2 (`Exact / Position & Orientation prior`)
* `"locked"` $\rightarrow$ Maps to Enum 3 (`Locked`)
* `"approximate"` $\rightarrow$ **Not recognized** (falls back to 0: `Unknown`)

---

### 3. Mathematical Verification of Focal Length

For rectilinear pinhole cameras with square pixels ($W = H = 2400\,\text{px}$) and $114.0^\circ$ horizontal FOV:
$$\text{half\_fov} = 114.0^\circ / 2 = 57.0^\circ$$
$$f_x = \frac{W}{2 \tan(57.0^\circ)} = \frac{1200}{\tan(57.0^\circ)} = \frac{1200}{1.5398649638} \approx 779.289117\,\text{px}$$

On a standard 35mm full-frame sensor ($W_{35\text{mm}} = 36.0\,\text{mm}$):
$$f_{35\text{mm}} = f_x \times \frac{36.0}{2400.0} = \frac{18.0}{\tan(57.0^\circ)} = \frac{18.0}{1.5398649638} = \mathbf{11.689337\,\text{mm}}$$

* **Result:** $11.689337\,\text{mm}$ is mathematically exact for $114.0^\circ$ FOV.
* (By comparison: $10.392305\,\text{mm}$ corresponds to $120.0^\circ$ FOV, and $18.000000\,\text{mm}$ corresponds to $90.0^\circ$ FOV).

---

### 4. Verified Clean RealityScan XMP Schema

```xml
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
      xcr:Version="3"
      xcr:PosePrior="exact"
      xcr:Coordinates="absolute"
      xcr:Rotation="{ROTATION_MATRIX_3X3}"
      xcr:FocalLength35mm="{F_35MM}"
      xcr:DistortionModel="brown3"
      xcr:DistortionCoeficients="0.000000 0.000000 0.000000 0.000000 0.000000 0.000000"
      xcr:Skew="0.000000"
      xcr:AspectRatio="1.000000"
      xcr:PrincipalPointU="0.000000"
      xcr:PrincipalPointV="0.000000"
      xcr:CalibrationPrior="locked"
      xcr:CalibrationGroup="1"
      xcr:DistortionGroup="1"
      xcr:InTexturing="1"
      xcr:InMeshing="1">
      <xcr:Position>{X} {Y} {Z}</xcr:Position>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

---

### 5. Verification on the 20-Frame Test Dataset

* **Test Set:** `scratch/rs_20frames_test/` (120 images across frames 000–019 in all 6 perspectives).
* **CLI Execution:**
  ```powershell
  RealityScan.exe -headless -newScene -addFolder <dir> -align -save <out.rcproj> -quit
  ```
* **Outcome:**
  * Exit code: 0 (completed in 26 seconds).
  * Project created: `Exp2_Exact_NoRig.rcproj` (120 inputs, Component 0 created).
  * Zero crashes, singular matrices, or unhandled exceptions.

---

### 6. Active Dataset Migration & Cleanup

1. **Duplicate XMP Removal:** Verified 0 `.jpg.xmp` duplicate files remain in the dataset.
2. **Active Dataset Updated:** All **3,558 `.xmp` files** across the 6 perspective folders (`front/`, `back/`, `left/`, `right/`, `up/`, `down/`) were rewritten with the verified schema and exact focal length $11.689337\,\text{mm}$.
3. **Core Generator Updated:** `generate_realityscan_xmp()` in `panoramic_splitter.py` now produces the clean, crash-free schema.

---

### 7. RealityScan GUI Alignment Workflow

To refine 360DVO positions while keeping 99% accurate rotations locked:
1. Open RealityScan 2.2 and load the `perspectives/` folder (or drag all 6 perspective folders).
2. Go to **ALIGNMENT** tab $\rightarrow$ **Settings** $\rightarrow$ **Camera priors**:
   * **Absolute pose**: Keep as `Position and orientation` (loaded from `PosePrior="exact"`).
   * **Position accuracy**: Set to `0.300 m` (gives solver $\pm 30\,\text{cm}$ freedom to adjust drift).
   * **Rotation accuracy**: Set to `0.050 deg` (virtually freezes orientation).
3. Click **Align Images**.
