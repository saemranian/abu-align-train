# Walkthrough: Abu Align & Train v0.1.1.9
## Adaptive Aspect Ratio Engine, Per-Frame Orientation Preservation & Multi-Ratio Export

### 1. Executive Summary
In release **v0.1.1.9**, Abu Align's **AI Masking Studio** (`sam2_engine.py`, `sam2_masker_ui.py`) introduces a smart **Adaptive Aspect Ratio & Orientation Engine**:
- **Smart Horizontal & Vertical Frame Handling**: Resolved the issue where mixed-orientation datasets (containing both landscape horizontal and portrait vertical images) had all mask layers forced into horizontal ratio. The studio now natively tracks each frame's individual dimensions and orientation throughout ingestion, interactive prompting, temporal tracking, viewport blitting, and batch export.
- **Dynamic SAM 2 Predictor Hooks**: Hooked SAM 2's `_get_orig_video_res_output` and prompt normalization pipeline to dynamically resolve each frame's native $(W_i, H_i)$ geometry. Horizontal frames generate horizontal masks; vertical frames generate vertical masks without stretching, transposing, or distorting.
- **Orientation-Preserving Hardware Exporters**:
  - `export_masks` computes target dimensions `(frame_target_w, frame_target_h) = compute_scaled_size(cur_w, cur_h, scale_mode)` on a per-frame basis.
  - **Horizontal input $\to$ horizontal export** (e.g. $6000 \times 4000$ native, or $2048 \times 1365$ at 2K).
  - **Vertical input $\to$ vertical export** (e.g. $4000 \times 6000$ native, or $1365 \times 2048$ at 2K).
  - Preserves exact aspect ratios across all downscale modes (4K UHD, 2K QHD, 1080p FHD, 50% Half-Scale, 25% Quarter-Scale) on both CUDA GPU and CPU pipelines.
- **Responsive Viewport Navigation**:
  - Viewport auto-fits (`_zoom_to_fit()`) whenever orientation toggles between consecutive frames during scrubbing or playback, ensuring portrait photos are never cropped at the top or bottom.
  - Frame readout displays individual frame resolution and orientation: `Frame: 2 / 50 (IMG_002.JPG) [4000x6000 Portrait]`.
  - Ingestion header detects mixed orientation sequences: `Res: Mixed Aspect (Auto H/V)`.
- **Rigorous Verification**: Added `test_mixed_orientations()` to `test_sam2_masker.py`, verifying 100% dimensional accuracy across mixed landscape and portrait image sequences.

---

### 2. Multi-Ratio Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│             Adaptive Aspect Ratio & Orientation Engine (v0.1.1.9)     │
│                                                                        │
│  [Mixed Image Dataset]                                                 │
│  ├── Frame 0: 6000 x 4000 (Landscape)                                  │
│  ├── Frame 1: 4000 x 6000 (Portrait)                                   │
│  └── Frame 2: 6000 x 4000 (Landscape)                                  │
│                                                            │           │
│  ┌─────────────────────────────────────────────────────────┴────────┐  │
│  │               PER-FRAME GEOMETRY REGISTRY (self.frame_dims)       │  │
│  │  • Frame 0: W=6000, H=4000 | Aspect Ratio = 1.50 (Horizontal)    │  │
│  │  • Frame 1: W=4000, H=6000 | Aspect Ratio = 0.67 (Vertical)      │  │
│  └─────────────────────────────────────────────────────────┬────────┘  │
│                                                            │           │
│  ┌─────────────────────────────────────────────────────────┴────────┐  │
│  │                 DYNAMIC SAM 2 TEMPORAL TRACKING                  │  │
│  │  • Prompt clicks normalized by native (W_i, H_i)                 │  │
│  │  • Dynamic _get_orig_video_res_output queries frame geometry     │  │
│  │  • Frame 0 mask: (4000, 6000) [Horizontal]                       │  │
│  │  • Frame 1 mask: (6000, 4000) [Vertical]                         │  │
│  └─────────────────────────────────────────────────────────┬────────┘  │
│                                                            │           │
│  ┌─────────────────────────────────────────────────────────┴────────┐  │
│  │               ADAPTIVE HARDWARE EXPORTER (RTX 4090 GPU)          │  │
│  │  • Frame 0: Exported as Horizontal (e.g. 2048 x 1365 at 2K)      │  │
│  │  • Frame 1: Exported as Vertical   (e.g. 1365 x 2048 at 2K)      │  │
│  │  • Zero Distortion · Zero Squashing · Bit-Exact Aspect Ratio      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Verification & Test Suite Results

The comprehensive test suite in `test_sam2_masker.py` was executed with all assertions passing:
- `SAM 2.1 Hiera Tiny` predictor model load on CUDA: **PASS**
- Standard 5-frame sequence prompts, propagation, and filters: **PASS**
- Hardware-accelerated multi-threaded WebP & PNG export: **PASS**
- **Mixed Horizontal & Vertical Orientation Suite (`test_mixed_orientations`)**:
  - Horizontal Frame 0 ($640 \times 480$) click and prompt evaluation: **PASS** (Mask shape: `(480, 640)`)
  - Vertical Frame 1 ($480 \times 640$) click and prompt evaluation: **PASS** (Mask shape: `(640, 480)`)
  - Bidirectional temporal tracking across mixed aspect sequence: **PASS**
  - Native export:
    - Frame 0 (Horizontal): Mask `640x480` | Cutout `640x480`: **PASS**
    - Frame 1 (Vertical):   Mask `480x640` | Cutout `480x640`: **PASS**
    - Frame 2 (Horizontal): Mask `800x600` | Cutout `800x600`: **PASS**
    - Frame 3 (Vertical):   Mask `600x800` | Cutout `600x800`: **PASS**
  - 50% Downscaled export:
    - Frame 0 (Horizontal): `320x240`: **PASS**
    - Frame 1 (Vertical):   `240x320`: **PASS**
    - Frame 2 (Horizontal): `400x300`: **PASS**
    - Frame 3 (Vertical):   `300x400`: **PASS**
