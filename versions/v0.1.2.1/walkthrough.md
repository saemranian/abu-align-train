# Walkthrough: Abu Align & Train v0.1.2.1
## AI Masker: Comprehensive Selection Optimization, Direct Brush/Cut Precision Tools & Scrollable Non-Clipped UI

### 1. Executive Summary
Release **v0.1.2.1** addresses core user experience and AI selection challenges in the **AI Subject Masker** (`sam2_engine.py`, `sam2_masker_ui.py`):
- **Erase / Deselect Optimization & Smart Anchor Detection**:
  - Fixed the root cause where AI Erase clicks on tracked/propagated frames did nothing or caused the entire mask to collapse to zero. Because SAM 2 had no positive prompts on propagated frames, receiving only a negative prompt wiped out object tokens.
  - The engine now automatically extracts a high-confidence medial foreground anchor point via `cv2.distanceTransform` from the existing mask interior, giving SAM 2 the foreground reference needed to retain the subject while carving out the negative region.
  - Multi-point negative reinforcement (cardinal cross cluster around the click point) and a localized fallback eraser ensure that an Erase click **never fails or does nothing**.
- **SAM 2.1 Bounding Box Marquee Prompt Optimization**:
  - Implemented strict boundary clamping `[0, W-1]` and `[0, H-1]` so dragging near or outside the canvas never passes out-of-bounds positional embeddings to SAM 2.
  - Filters out conflicting out-of-box points that previously corrupted SAM 2's attention logits.
  - Live interactive dashed rectangle feedback on the viewport canvas (`ACCENT` color for AI Box (+), `DANGER` color for Box Cut (−)).
- **Direct Manual Precision Editing Suite (100% Guaranteed Override)**:
  - **Manual Brush Paint (+)** & **Manual Eraser Brush (−)**: Freehand drawing on the mask buffer with interactive brush radius slider (2px to 120px) and live circular cursor overlay on canvas. Hotkeys `[` and `]` quickly decrease and increase brush size.
  - **Box Cut / Marquee Deselect (−)**: Drag a rectangular marquee to immediately zero out any region on the mask.
  - **SAM 2 Video Memory Sync (`add_new_mask`)**: Manual brush strokes and box cuts feed directly into SAM 2's video memory predictor so that subsequent timeline propagation tracks the manual edits forward and backward across the video sequence.
  - **Multi-Level Undo History (`Ctrl+Z`)**: Snapshot-based undo history stack on each frame, allowing instant reversion of AI clicks, box prompts, brush strokes, and box cuts.
- **Scrollable Non-Clipped Left Control Sidebar**:
  - Fixed left panel options getting cropped from the bottom on small windows and low-resolution displays (720p/768p/1080p).
  - Repositioned vertical scrollbar to pack first on the right, ensuring it is always visible and never squashed by canvas expansion.
  - Implemented centralized mousewheel event routing: hovering over any card, button, slider, or entry in the sidebar smoothly scrolls the sidebar up and down, while hovering over the viewport canvas zooms into the image.

---

### 2. Architecture & Pipeline Diagram

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      Enhanced AI Masker System Architecture (v0.1.2.1)                 │
│                                                                                        │
│  [User Canvas Interaction] ──┬──> AI Modes: Point (+), Erase (−), Box (+)              │
│                              │                                                         │
│                              └──> Direct Edit: Brush Paint (+), Eraser (−), Box Cut (−)│
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     SAM 2 ENGINE & INTELLIGENT REINFORCEMENT                     │  │
│  │  • Smart Anchor Point Detection: Injects distance-transform peak for Erase clicks │  │
│  │  • Multi-point Negative Reinforcement: Cardinal cross cluster flips local logits │  │
│  │  • Box Coordinate Clamping: Bounds ix0, iy0, ix1, iy1 strictly to [0, W], [0, H] │  │
│  │  • Direct Modifiers: cv2.circle/cv2.line strokes & rectangular marquee zeroing   │  │
│  │  • SAM 2 Memory Synchronization: add_new_mask updates video predictor state      │  │
│  │  • Multi-Level Undo Stack: Push & pop frame mask & prompt snapshots              │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                            │
│  ┌────────────────────────────────────────┴─────────────────────────────────────────┐  │
│  │                        SCROLLABLE CONTROL SIDEBAR (440px)                        │  │
│  │  • Scrollbar docked first on right (always visible & draggable)                  │  │
│  │  • Centralized MouseWheel router: seamless sidebar scrolling & canvas zooming    │  │
│  │  • Card 1: Source Media Ingestion                                                │  │
│  │  • Card 2: 2-Row Tool Matrix (AI + Direct), Brush Size Slider & Undo Buttons     │  │
│  │  • Card 3: Timeline Propagation (Forward, Backward, Both)                        │  │
│  │  • Card 4: Mask Refinement (Dilation, Feather, Threshold, Speckles, Holes, Inv)  │  │
│  │  • Card 5: Hardware-Accelerated RealityScan Mask & RGBA WebP Cutout Exporter     │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Verification & Test Suite Results

The comprehensive test suite in `test_sam2_masker.py` was executed with all 3 test suites passing:
- **Suite 1: SAM 2 AI Masking Core & Hardware Exporters**: **PASS**
  - Model catalog & download verification.
  - Point and box prompts on synthetic moving target sequence.
  - Bidirectional temporal propagation across 5 frames.
  - Mask post-processing ops (dilation, erosion, feathering, invert).
  - Hardware-accelerated WebP & PNG batch export (106+ fps).
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
