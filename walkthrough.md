# Abu Align & Train v0.0.7.2 — COLMAP Image Coordinate Fix & 3D Browser Viewport Auto-Centering

Version **v0.0.7.2** fixes a critical photogrammetry camera coordinate parsing bug in the 3D Browser Viewer (`fast_viewer.py`), resolving the issue where point clouds were erroneously wiped out by outlier clipping and camera frustums were pushed into deep outer space.

---

## 1. Problem Description & Root Cause

Even after run discovery was resolved in `v0.0.7.1`:
1. **Console Showed 0 Points Rendered**:
   `[Viewer] Rendered 0 points from points3D.ply (sparse_10_omega)` despite `points3D.ply` containing 300,000 valid vertices.
2. **Viewport Blank**:
   Nothing appeared in the browser viewport.
3. **Markdown Error**:
   The metadata panel displayed `Markdown Failed to Render`.

### Technical Root Causes:
1. **COLMAP Alternating Line Specification**:
   - In COLMAP's `images.txt` specification, every registered image has **two lines**:
     - Line 1: `IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME`
     - Line 2: `POINTS2D[] as (X, Y, POINT3D_ID)...` (2D feature detections and 3D tie point IDs)
   - `load_cameras_from_colmap_images` iterated line-by-line without consuming the second line. Because the 2D points line typically contains dozens or hundreds of tokens, it was parsed as a camera pose!
   - Floating-point pixel coordinates and integer point IDs (e.g. `1194`, `404.01`) were treated as rotation quaternions and translation vectors.
   - When converted to camera-to-world via $R^T t$, this generated astronomical camera positions ($10^{11}$ meters).
2. **Outlier Centroid Distortion**:
   - Outlier filtering computed distances relative to `latest_camera_center`.
   - With corrupted camera poses at $10^{11}$ meters, `latest_camera_center` became $1.9 \times 10^9$ meters.
   - Consequently, all 300,000 valid points were evaluated as $> 2.7 \times 10^9$ meters from the center, exceeding the 35 m cutoff and causing 100% of the point cloud to be discarded as outliers (`inlier_pts = 0`).
3. **React-Markdown Parsing Failure**:
   - The info card contained non-BMP emojis (`🟢`, `⚠️`) and HTML `<small>` tags which caused Viser's React-Markdown parser to fail on Windows.

---

## 2. Technical Fixes

### A. COLMAP `images.txt` Two-Line Parser ([`fast_viewer.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/fast_viewer.py))
- Properly consumes both lines per image (`f.readline()` for pose, `f.readline()` to discard 2D keypoints).
- Normalizes quaternions: validates $\|q\| \ge 10^{-4}$ and divides by norm.
- Adds bounding guards rejecting non-finite positions or translations with $\|t\| > 50,000$ m.

### B. Point Centroid Outlier Filtering ([`fast_viewer.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/fast_viewer.py))
- Outlier filtering now measures Euclidean distance from `np.median(pts, axis=0)` (the geometric centroid of the point cloud itself).
- Immune to camera coordinate system scale discrepancies.
- Result: **300,000 out of 300,000 points (100%)** render as inliers.

### C. Bulletproof React-Markdown Info Panel ([`fast_viewer.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/fast_viewer.py))
- Converted info card to clean, standard Markdown bullet points without raw HTML tags or non-BMP emojis.
- Renders cleanly in dark theme without parser errors.

### D. Viewport Orientation & Reframe ([`fast_viewer.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/fast_viewer.py))
- Enforces `client.camera.up_direction = np.array([0.0, 0.0, 1.0])` on connection and on "Fit trajectory" (`focus_btn`).
- Positions camera eye at `[0.0, -diag * 1.5, diag * 0.8]` looking directly at object center.

---

## 3. Verification

- `tests/test_fast_viewer.py`: Added `test_colmap_images_txt_parser` verifying alternating 2D point measurement line skipping. All 6 unit tests pass.
- Verified on user's dataset:
  - Cameras loaded: **240 cameras** (COLMAP images.txt), span: `[1.93, 0.52, 1.78]` m, center: `[0.05, -0.17, 0.82]`.
  - Points loaded: **300,000 vertices** (`points3D.ply`), Inliers: **300,000 / 300,000 (100%)**.
- Version bumped from `v0.0.7.1` to `v0.0.7.2` (base-10 odometer) and native launchers recompiled.
