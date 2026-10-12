# Abu Align & Train v0.0.5.8 — Fix OpenSfM 0-Camera COLMAP Export & LichtFeld Studio Dataset Corruption

## Overview
Version **v0.0.5.8** resolves a critical bug in the **Mapillary OpenSfM** alignment pipeline that caused LichtFeld Studio training to crash on dataset import with the following fatal error:
```text
Import Failed. Path: 3dgs_dataset.
[Corrupted data] cameras.bin must declare at least one camera (camera_count=0; path=".../results/sparse_06_opensfm/3dgs_dataset/sparse/0/cameras.bin", remaining_bytes=0)
```

---

## Root Cause Analysis

### 1. The Empty Overwrite Bug in `align_cli.py`
In `align_cli.py` (lines 1630–1638), OpenSfM output handling looked for an existing `colmap/` or `sparse/` directory:
```python
colmap_out = os.path.join(opensfm.output_dir, "colmap")
if not os.path.isdir(colmap_out):
    colmap_out = os.path.join(opensfm.output_dir, "sparse")
if os.path.isdir(colmap_out):
    reconstruction = pycolmap.Reconstruction(colmap_out)
else:
    reconstruction = pycolmap.Reconstruction()  # <-- Created an EMPTY object (0 cameras, 0 images)
```
Because OpenSfM writes its primary photogrammetric solution to `reconstruction.json` (and optionally `colmap_export/`), neither `colmap/` nor `sparse/` existed. Consequently, `reconstruction` became an empty `pycolmap.Reconstruction()` instance.

Later in `align_cli.py` Stage 5 (lines 1747–1753):
```python
if reconstruction is not None:  # In Python, an empty pycolmap.Reconstruction is NOT None!
    try:
        reconstruction.write(sparse_0_dir)      # Overwrites cameras.bin (8 bytes, camera_count=0)
        reconstruction.write_text(sparse_0_dir) # Overwrites cameras.txt (# Number of cameras: 0)
    except Exception:
        pass
```
This obliterated the valid text files generated from `reconstruction.json` and generated corrupt 8-byte empty binary files (`cameras.bin`, `images.bin`, `points3D.bin`, `frames.bin`, `rigs.bin`).

### 2. Binary Priority in LichtFeld Studio (`colmap_loader.cpp`)
LichtFeld Studio's COLMAP loader discovers all available files. When `cameras.bin` is found on disk, it prioritizes binary over text:
```cpp
if ((has_cameras || has_images || has_points) && (has_cameras_text || has_images_text || has_points_text)) {
    LOG_WARN("Found both binary and text COLMAP files. Prioritizing binary files.");
}
```
In `colmap.cpp` (line 1472):
```cpp
const uint64_t n_cams = read_u64(cur, end, "cameras.bin camera count");
if (n_cams == 0) {
    throw_colmap_error(lfs::ErrorCode::DataLoss,
        std::format("cameras.bin must declare at least one camera (camera_count={}, path='{}', remaining_bytes={})",
                    n_cams, lfs::core::path_to_utf8(file_path), static_cast<size_t>(end - cur)));
}
```
Reading the 8-byte empty `cameras.bin` yielded `n_cams = 0`, causing LichtFeld Studio to immediately abort with exit code 1.

### 3. OpenSfM `export_colmap` Binary Index Inconsistency
OpenSfM's internal `export_colmap --binary` command exports 2D feature indices from the full unfiltered keypoint array, but only writes a filtered subset into `images.bin`. When loaded in COLMAP / `pycolmap`, point tracks reference out-of-bounds 2D feature indices (e.g. `p2d_idx=4613 >= num_pts=1971`), triggering `std::out_of_range` (`IndexError: invalid vector subscript`).

### 4. Hardcoded Equirectangular Camera Model
`opensfm_to_transforms_json()` previously hardcoded `"camera_model": "EQUIRECTANGULAR"` in `transforms.json` and `1 EQUIRECTANGULAR` in `cameras.txt`, even when the dataset was captured with perspective cameras (`--projection perspective`).

---

## Fixes Implemented in v0.0.5.8

### 1. Native OpenSfM to `pycolmap.Reconstruction` Converter (`opensfm_engine.py`)
Implemented `opensfm_to_pycolmap(rec_json_path: str) -> Optional[pycolmap.Reconstruction]` which translates OpenSfM's `reconstruction.json` into a fully valid `pycolmap.Reconstruction`:
- **Intrinsics**: Converts `perspective` models to `PINHOLE` or `RADIAL` (with accurate focal length scaling $f_{px} = f_{norm} \cdot \max(W, H)$), and `equirectangular`/`spherical` models to `EQUIRECTANGULAR`.
- **Extrinsics**: Converts Rodrigues rotation vectors to unit quaternions $(q_w, q_x, q_y, q_z)$ with a numerically stable SciPy fallback for 180° rotations, and pairs them with world-to-cam translation vectors into `pycolmap.Rigid3d`.
- **Tie Points**: Incorporates all Ceres bundle-adjusted 3D points directly into `reconstruction.add_point3D()`.

### 2. Guarded COLMAP Model Serialization (`align_cli.py`)
- Guaranteed that `reconstruction.write()` and `reconstruction.write_text()` are **only called when `reconstruction.num_reg_images() > 0`**:
  ```python
  if reconstruction is not None and reconstruction.num_reg_images() > 0:
      reconstruction.write(sparse_0_dir)
      reconstruction.write_text(sparse_0_dir)
  ```
- Added automated post-export purging of any corrupt `*.bin` files with size $\le 8$ bytes in `sparse/0/` and `COLMAP_Lichtfield/`.

### 3. Dynamic Projection Support in `transforms.json` (`opensfm_engine.py`)
`opensfm_to_transforms_json()` now inspects the camera's actual projection:
- **Perspective**: Writes `"camera_model": "OPENCV"` with $fl_x, fl_y, cx, cy, k1, k2$ and computed horizontal/vertical FOV angles.
- **Equirectangular**: Writes `"camera_model": "EQUIRECTANGULAR"` with canvas dimensions.
- Correctly formats `cameras.txt` as `1 PINHOLE ...`, `1 RADIAL ...`, or `1 EQUIRECTANGULAR ...`.

### 4. Robust Camera Model Unpacking (`align_cli.py`)
Fixed `export_transforms_json()` and `export_realitycapture_csv()` to safely unpack parameter vectors depending on `cam.model_name`:
- `PINHOLE` / `OPENCV`: 4 parameters $[f_x, f_y, c_x, c_y]$.
- `SIMPLE_PINHOLE` / `SIMPLE_RADIAL` / `RADIAL`: 3 to 5 parameters $[f, c_x, c_y, (k_1, k_2)]$.
- `EQUIRECTANGULAR`: Derives dummy intrinsics $[W, H, W/2, H/2]$.

### 5. Flat Image Linking Hygeine
When staging `3dgs_dataset/images`, directory junctions to directories that contain recursive `results/` folders are avoided. Instead, images are linked directly from `opensfm_workspace/images` or individual file links, preventing LichtFeld Studio contract violations caused by duplicate basenames across subfolders.

### 6. Repaired User Dataset (`results/sparse_06_opensfm`)
Directly regenerated and repaired the user's dataset:
- `3dgs_dataset/sparse/0/cameras.bin`: 64 bytes (1 valid PINHOLE camera).
- `3dgs_dataset/sparse/0/images.bin`: 4,408 bytes (50 registered images).
- `3dgs_dataset/sparse/0/points3D.bin`: 1,005,014 bytes (19,706 tie points).
- `3dgs_dataset/transforms.json`: Updated with valid OPENCV perspective intrinsics.
- `COLMAP_Lichtfield/`: Repaired with matching COLMAP binary and text files.
- `Postshot_RealityCapture/`: Populated with valid `cameras.csv` and `transforms.json`.

---

## Verification & Validation

1. **LichtFeld Studio Headless Training**:
   Executed `LichtFeld-Studio.exe` directly on the repaired `results/sparse_06_opensfm/3dgs_dataset`:
   ```text
   [info] colmap.cpp:4403  Read 1 cameras and 50 images from COLMAP
   [info] colmap_loader.cpp:325  Loaded 19706 points from COLMAP
   [info] colmap_loader.cpp:399  COLMAP dataset loaded successfully in 84ms
   [info] colmap_loader.cpp:400    - 50 cameras
   [info] training_setup.cpp:756  Dataset loaded successfully using COLMAP loader
   ...
   [info] trainer.cpp:8992  Training completed successfully
   [info] training_setup.cpp:1446  Exported final splat: .../splat_5.ply
   ```
   Zero errors, complete training pipeline execution, and successful Gaussian splat export.

2. **Full Automated Test Suite**:
   Ran the entire unit test suite via `unittest discover tests`:
   ```text
   Ran 25 tests in 31.038s - OK
   ```

3. **Version Sync**:
   Bumped version to `v0.0.5.8` across CLI, GUI, C# launchers, and created historical binaries.
