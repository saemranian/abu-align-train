# Walkthrough: Abu Align & Train v0.1.3.3
## 360 Splitter: True Camera Rig XMP Integration (`True_Schema_360_Camera_Rig.xmp`)

### 1. Executive Summary
Release **v0.1.3.3** integrates the user-validated, industry-grade **`True_Schema_360_Camera_Rig.xmp`** into the 360 Splitter pipeline. 

RealityScan 2.2 requires explicit coordinate priors and perspective pinhole distortion models to lock relative camera angles without recalculating rotations during bundle adjustment. By implementing the verified schema:
1. **Locked Camera Poses (`xcr:PosePrior="locked"`)**: RealityScan preserves the exact relative 90° orientation between the 6 cubemap perspectives and does not recalculate rotations.
2. **Perspective Pinhole Distortion Model (`xcr:DistortionModel="perspective"`)**: Synthetic rectilinear crops are tagged with `DistortionModel="perspective"`, `DistortionPrior="exact"`, and `DistortionCoeficients="0 0 0 0 0 0"`, preventing polynomial radial distortion solver instability.
3. **Structured Rotation Matrix and Metric Position**: `<xcr:Rotation>` (3x3 row-major rotation matrix) and `<xcr:Position>` (X Y Z metric translation) are emitted as child elements of `<rdf:Description>`, aligning with RealityScan 2.2's schema parser.
4. **Sub-Nanometer Precision Focal Length**: Computed using $f = 18.0 / \tan(\text{FOV} / 2)$ to 13 decimal digits (`11.6893366775552` for 114° FOV).
5. **Turnkey Defaults**: Default `xmp_pose_prior` is now `"locked"` in both programmatic APIs and CLI arguments.

---

### 2. Schema Comparison: Previous vs. `True_Schema_360_Camera_Rig.xmp`

| Attribute / Element | Guide 06 Approach | `True_Schema_360_Camera_Rig.xmp` (v0.1.3.3) | Purpose / Impact in RealityScan 2.2 |
| :--- | :--- | :--- | :--- |
| **`xcr:PosePrior`** | Omitted | `"locked"` | Instructs RealityScan to fix rotation and position, preventing rotation recalculation |
| **`xcr:Coordinates`** | Omitted | `"absolute"` | Establishes coordinates in absolute metric frame |
| **`xcr:DistortionModel`** | `"brown3"` | `"perspective"` | Uses pure rectilinear pinhole model instead of radial polynomial |
| **`xcr:DistortionPrior`** | Omitted | `"exact"` | Locks distortion to zero without optimizer drift |
| **`xcr:DistortionCoeficients`**| Omitted | `"0 0 0 0 0 0"` | 6 zero-coefficients for pinhole geometry |
| **`<xcr:Rotation>`** | Stripped | Child XML Element | Embeds full 3x3 rotation matrix for the perspective |
| **`<xcr:Position>`** | Stripped | Child XML Element | Metric X, Y, Z camera coordinates |
| **`xcr:FocalLength35mm`** | `11.689337` (6 decimals) | `11.6893366775552` (13 decimals) | Exact matching with RealityScan internal sensor calibration |

---

### 3. XML Structure of Generated Sidecars

```xml
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
      xcr:Version="3"
      xcr:PosePrior="locked"
      xcr:Coordinates="absolute"
      xcr:FocalLength35mm="11.6893366775552"
      xcr:DistortionModel="perspective"
      xcr:DistortionCoeficients="0 0 0 0 0 0"
      xcr:DistortionPrior="exact"
      xcr:Skew="0"
      xcr:AspectRatio="1"
      xcr:PrincipalPointU="0"
      xcr:PrincipalPointV="0"
      xcr:CalibrationPrior="exact"
      xcr:CalibrationGroup="1"
      xcr:DistortionGroup="1"
      xcr:InTexturing="1"
      xcr:InMeshing="1">
      <xcr:Rotation>{R00} {R01} {R02} {R10} {R11} {R12} {R20} {R21} {R22}</xcr:Rotation>
      <xcr:Position>{X} {Y} {Z}</xcr:Position>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
```

---

### 4. Verification and Regression Testing

- **`tests/test_cubemap_rs.py`**:
  - Validated generation of 6 cubemap views with 114° FOV.
  - Confirmed all 6 XMP sidecars contain `PosePrior="locked"`, `DistortionModel="perspective"`, `FocalLength35mm="11.6893366775552"`, and `<xcr:Rotation>` / `<xcr:Position>` child nodes.
  - Status: **PASSED (100%)**.
- **`tests/test_rig_cameras_csv.py`**:
  - Validated `cameras.csv` generation and canonical XMP sidecars across multi-perspective ring layouts.
  - Status: **PASSED (100%, 10 tests)**.
- **`tests/test_splitter_masks.py`**:
  - Validated projection of equirectangular AI masks alongside perspective views without XMP conflicts.
  - Status: **PASSED (100%, 4 tests)**.
