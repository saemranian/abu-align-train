# Walkthrough: Abu Align & Train v0.1.2.3
## AI Masker: Grounded-SAM 2 Text Prompts, Zero-Lag Brush & Box Cut, Model Switching Verification & Hover Tooltips

### 1. Executive Summary
Release **v0.1.2.3** delivers four major user-requested enhancements to the **AI Subject Masker** (`sam2_engine.py`, `sam2_masker_ui.py`):

1. **Grounded-SAM 2 Natural Language Text-Prompt Masking**:
   - Integrated open-vocabulary zero-shot grounding detector (`google/owlv2-base-patch16-ensemble`) coupled with SAM 2.1.
   - Users can type natural language descriptions (e.g., `"turntable"`, `"white disk"`, `"bracket"`, `"screw"`, `"person"`) in Card 2 and click **`✨ AI Mask`** (or press Enter).
   - OwLV2 computes sub-pixel bounding box predictions with confidence scoring, which are directly injected into SAM 2.1's memory predictor to mask the subject cleanly in one step.
2. **Zero-Lag Brush Paint, Eraser & Box Cut**:
   - **Root Cause Diagnosed**: Previously, every individual mouse drag event (60–120Hz) invoked `video_predictor.add_new_mask()` (triggering synchronous PyTorch GPU kernel dispatches) and `_render_current_frame()` (executing expensive OpenCV `floodFill` and connected-components stats on high-resolution images per pixel of motion).
   - **Fix Implemented**:
     - Decoupled real-time brush dragging from neural memory synchronization: `brush_stroke_mask` and `box_cut_mask` update the numpy mask in RAM with `sync_to_sam2=False`.
     - Canvas rendering during active drag uses `fast_brush=True`, bypassing morphological passes for 120 FPS instantaneous blitting.
     - On mouse button release (`<ButtonRelease>`), `sync_frame_mask_to_sam2` synchronizes the mask into SAM 2's video memory predictor asynchronously in a background thread without blocking the UI thread.
3. **Verified Model Dropdown & Dynamic Switching Architecture**:
   - Verified that all 4 SAM 2.1 model checkpoints exist and load cleanly (`tiny`, `small`, `base_plus`, `large`).
   - Added thread-safe model switching lock `_model_lock`.
   - Added feature cache invalidation (`cached_features.clear()`) and native predictor state reset (`video_predictor.reset_state(inference_state)`) when switching models, preventing stale backbone embeddings from corrupting the new model.
   - All active manual keyframe prompts are automatically re-evaluated using the newly selected model weights.
   - Dropdown is disabled during switching and the hardware device badge is dynamically updated.
4. **Responsive Non-Clipped Sidebar UI & Tool Grid**:
   - Expanded control sidebar container width to **500px**.
   - Replaced horizontal packing in Card 2 with a 4-column responsive grid layout (`sticky="w"`).
   - `Box Cut (−)` and all direct tools now have ample breathing room on any screen size or DPI scaling.
5. **Instructive Hover Tooltips Across All Parameters**:
   - Created a modern, dark-themed `ToolTip` class with smooth 350ms hover delay and screen-boundary awareness.
   - Attached instructive guides to every interactive parameter: model dropdown, device badge, prompt tools, brush slider, text prompt input, undo/clear/reset, timeline tracking, mask refinement sliders, export options, and viewport controls.

---

### 2. Architecture & Pipeline Diagram

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      Enhanced AI Masker System Architecture (v0.1.2.3)                 │
│                                                                                        │
│  [Natural Language Text Prompt] ──> [OwLV2 Zero-Shot Detector] ──> [Bounding Box]      │
│                                                                           │            │
│  [User Canvas Mouse Interaction] ──┬──> AI Modes: Point (+), Erase (−), Box (+)        │
│                                    │                                      │            │
│                                    └──> Direct Edit: Brush (+), Erase (−), Box Cut (−) │
│                                                                           │            │
│  ┌────────────────────────────────────────────────────────────────────────┴─────────┐  │
│  │                     SAM 2 ENGINE & ZERO-LAG MEMORY ARCHITECTURE                  │  │
│  │  • OwLV2 Grounded Pipeline: detect_box_from_text -> generate_mask_from_text      │  │
│  │  • Thread-Safe Model Switching: _model_lock, reset_state(), prompt re-evaluation │  │
│  │  • Zero-Lag Dragging: Decoupled in-memory mask update (sync_to_sam2=False)        │  │
│  │  • Async Memory Commit: sync_frame_mask_to_sam2 on ButtonRelease                 │  │
│  │  • mask_revisions Counter: Increments on every stroke for instant cache invalid  │  │
│  │  • Full Undo / Redo History Stack per frame                                      │  │
│  │  • Hardware Exporters: Batch WebP & PNG with pause_token & cancel_token          │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                            │
│  ┌────────────────────────────────────────┴─────────────────────────────────────────┐  │
│  │                        SPACIOUS CONTROL SIDEBAR (500px)                          │  │
│  │  • Card 1: Source Media Ingestion & Sequence Diagnostics                         │  │
│  │  • Card 2: 4-Column Tool Grid, Text Prompt Entry & AI Mask Button                │  │
│  │  • Card 3: Timeline Propagation (Forward, Backward, Full Sequence)               │  │
│  │  • Card 4: Mask Refinement (Dilation, Feather, Threshold, Speckles, Holes, Inv)  │  │
│  │  • Card 5: Hardware-Accelerated Exporter with Pause & Stop Controls              │  │
│  │  • ToolTip System: Comprehensive instructive hover guides on all controls        │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Verification & Test Suite Results

The comprehensive test suite in `test_sam2_masker.py` was executed with all 5 test suites passing with zero errors:

- **Suite 1: SAM 2 AI Masking Core & Hardware Exporters**: **PASS**
  - Model catalog & download verification.
  - Interactive click and box prompts on synthetic moving target sequence.
  - Bidirectional temporal propagation across sequence.
  - Mask post-processing ops (dilation, erosion, feathering, invert).
  - Hardware-accelerated WebP & PNG batch export (95+ fps).
- **Suite 2: Mixed Horizontal & Vertical Orientation Suite**: **PASS**
  - Ingestion of mixed landscape and portrait frames.
  - Adaptive per-frame aspect ratio mask generation.
  - Native resolution export preserving independent orientation ratios.
- **Suite 3: Advanced Selection, Erase, Box Cut & Brush Tools**: **PASS**
  - AI Bounding Box Marquee with coordinate clamping: **PASS**.
  - Direct Manual Brush Eraser stroke: **PASS** (erased subject center to 0).
  - Multi-level Undo of Brush Eraser: **PASS** (restored pixel to 255).
  - Box Cut / Marquee Deselect: **PASS** (zeroed out rectangular region).
  - Multi-level Undo of Box Cut: **PASS** (restored marquee region to 255).
  - Smart Anchor Positive Point on Tracked Frame Erase: **PASS** (prevented mask collapse).
- **Suite 4: Export Stop, Pause & Mask Revision Suite**: **PASS**
  - Mask revision tracking across box, brush, and undo (`0 -> 2 -> 3 -> 4`): **PASS**.
  - Reset frame conditioning purging tracked memory: **PASS**.
  - Export cancellation stopping cleanly after 2/8 frames: **PASS** (`cancelled: True`).
  - Export pause and resume completing all 8 frames: **PASS** (`cancelled: False`).
- **Suite 5: Grounded-SAM 2 Text Prompt & Model Switching Suite**: **PASS**
  - OwLV2 Zero-Shot Text Prompt Detection (`"white disk"`): **PASS** (Box: `[125.6, 125.7, 276.4, 277.2]`, Score: 82.6%).
  - Dynamic Model Switching (`tiny` -> `small` -> `tiny`): **PASS** (Flushed backbone feature cache, cleanly reset tracking state, and re-evaluated keyframe mask).
  - Zero-Lag Brush Stroke with decoupled SAM 2 sync: **PASS** (Instant in-memory mutation + async neural synchronization).
