# Walkthrough: Abu Align & Train v0.1.2.0
## Video Frame Extractor: Native Portrait Ratio Preservation, Custom Resolution Export & Orientation Controls

### 1. Executive Summary
Release **v0.1.2.0** brings a major overhaul to the **Video to Frame Sequence Extractor** (`video_extractor.py`, `abu_align_gui.py`):
- **Native Orientation & Aspect-Ratio Preservation**: Fixed the issue where vertical (portrait) videos were forcibly squashed into 2:1 horizontal equirectangular frames. By default, the extractor strictly preserves the original input ratio: **vertical video inputs extract into native vertical frames**, and horizontal video inputs extract into native horizontal frames.
- **Orientation & Display Matrix Detection**: `get_video_info` now parses both `Display Matrix` side data and stream/format `rotate` tags via `ffprobe` (and `cv2.CAP_PROP_ORIENTATION_META` in OpenCV), detecting upright display dimensions and distinguishing portrait from landscape videos automatically.
- **Smart Adaptive Preset Scaling (4K, 2K, 1K)**:
  - For planar videos, presets now scale the **longest edge** while strictly maintaining native aspect ratio:
    - Vertical 9:16 at 4K $\to$ $2160 \times 3840$ (vertical portrait).
    - Vertical 9:16 at 2K $\to$ $1080 \times 1920$ (vertical portrait).
    - Vertical 4:5 at 2K $\to$ $1536 \times 1920$ (vertical portrait).
    - Horizontal 16:9 at 2K $\to$ $1920 \times 1080$ (horizontal landscape).
  - 2:1 scaling is retained exclusively for 360° Equirectangular (ERP) videos.
- **Custom Resolution Export**:
  - Full support for arbitrary user-defined resolutions (e.g. `1080x1920`, `2160x3840`, `720x1280`, `1440x2560`) in both GUI and CLI.
  - GUI dialog features a dedicated custom `WxH` entry field that automatically enables when `"custom"` is selected and auto-populates with the detected video's upright dimensions.
- **Orientation / Rotation Control Option**:
  - `Original (Follow Input)`: Default. Vertical stays vertical; horizontal stays horizontal.
  - `Rotate to Horizontal (Landscape)`: If input video is vertical, rotates it 90° CW so that frames render horizontally in landscape format.
  - `Rotate 90° CW`, `Rotate 90° CCW`, `Rotate 180°`: Explicit rotation overrides.
  - Fully integrated with NVIDIA NVDEC/CUDA hardware acceleration (`-hwaccel cuda`) and pure-Python OpenCV fallback.
- **Live Video Detection Telemetry**:
  - Displays instant badge upon selecting a video file: e.g. `Detected Video: 1080x1920 (Vertical (Portrait) 0.56:1) | 30.0 FPS | 12.4s (372 frames) | H.264`.

---

### 2. Architecture & Pipeline Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│             Video Frame Extractor Engine Architecture (v0.1.2.0)       │
│                                                                        │
│  [Video Input: e.g. 1080 x 1920 Portrait / Smartphone Video]           │
│                                                            │           │
│  ┌─────────────────────────────────────────────────────────┴────────┐  │
│  │          METADATA & DISPLAY MATRIX PROBING (get_video_info)      │  │
│  │  • Analyzes raw width, height, codec, fps, total frames          │  │
│  │  • Parses Display Matrix & rotate tags (0°, 90°, 180°, 270°)     │  │
│  │  • Determines upright dimensions & is_vertical boolean           │  │
│  └─────────────────────────────────────────────────────────┬────────┘  │
│                                                            │           │
│  ┌─────────────────────────────────────────────────────────┴────────┐  │
│  │               ORIENTATION & RESOLUTION RESOLVER                  │  │
│  │  • Mode: "Original" -> Keeps 1080 x 1920 vertical                │  │
│  │  • Mode: "Horizontal" -> Applies transpose=1 (1920 x 1080)       │  │
│  │  • Scaling: "2K" on vertical -> 1080 x 1920 (longest edge 1920)  │  │
│  │  • Scaling: "Custom" -> user-specified exact W x H               │  │
│  └─────────────────────────────────────────────────────────┬────────┘  │
│                                                            │           │
│  ┌─────────────────────────────────────────────────────────┴────────┐  │
│  │        MULTI-THREADED HARDWARE EXTRACTION PIPELINE               │  │
│  │  • FFmpeg + NVDEC CUDA: -vf fps=...,[transpose],[scale=W:H]      │  │
│  │  • OpenCV Fallback: cv2.rotate() + cv2.resize() + ThreadPool     │  │
│  │  • Output: Exact aspect ratio matching requested orientation     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Verification & Test Suite Results

The comprehensive test suite in `test_video_extractor.py` was executed with all 8 verification tests passing:
- **Test 1: Technical metadata and orientation detection**: **PASS** (Detected `288x360`, `is_vertical=True`).
- **Test 2: Native vertical extraction (follows original ratio)**: **PASS** (Extracted native portrait frames `288x360`).
- **Test 3: Smart preset scaling on vertical video (2K portrait)**: **PASS** (Extracted portrait frame `1536x1920`, longest edge 1920).
- **Test 4: Custom resolution export (1080x1920)**: **PASS** (Extracted exact custom frame `1080x1920`).
- **Test 5: Rotate vertical video to Horizontal (Landscape)**: **PASS** (Extracted rotated landscape frame `1920x1536`).
- **Test 6: Explicit rotation transforms (180°)**: **PASS** (Extracted rotated 180° frame `288x360`).
- **Test 7: OpenCV fallback pipeline with custom resolution**: **PASS** (Extracted 36 frames at `720x1280`).
- **Test 8: Manifest telemetry verification**: **PASS** (Validated JSON manifest schema and extraction configuration).
