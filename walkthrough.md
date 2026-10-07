# Walkthrough: Abu Align & Train v0.1.2.5
## AI Masker: Native Resolution Export Default, Box Cut `sync_to_sam2` Decoupling & Canvas Keyboard Focus Routing

### 1. Executive Summary
Release **v0.1.2.5** addresses three critical user requests in the **AI Subject Masker** (`sam2_engine.py`, `sam2_masker_ui.py`):

1. **Default Resolution for Mask Export as Native**:
   - **User Request**: *"Make the default resolution for mask export as native."*
   - **Implementation**:
     - `self.export_scale_var` is now initialized to `"Original (100% Native)"` by default.
     - `"Original (100% Native)"` is placed as the first entry in the export resolution combobox (`self.cb_scale`).
     - Preserves 100% native photogrammetry pixel resolution across individual landscape and portrait images without auto-downsampling to 2K.
2. **Fixed Box Cut Selection Tool Error**:
   - **User Request**: *"The 'box cut' selection too is not working. 2nd screenshot."*
   - **Root Cause Diagnosed**:
     - `sam2_masker_ui.py:1599` called `self.engine.box_cut_mask(self.current_frame_idx, box, sync_to_sam2=False)` to decouple the instant raster cutout from the background SAM 2 neural memory sync.
     - `sam2_engine.py:904` defined `def box_cut_mask(self, frame_idx: int, box: List[float], obj_id: int = 1) -> np.ndarray:`, which lacked the `sync_to_sam2` keyword parameter, raising: `TypeError: SAM2MaskingEngine.box_cut_mask() got an unexpected keyword argument 'sync_to_sam2'`.
   - **Fix Implemented**:
     - Updated `box_cut_mask` in `sam2_engine.py` to accept `sync_to_sam2: bool = True`.
     - When `sync_to_sam2=False`, the bounding box pixels are zeroed immediately in memory with zero UI latency, and `self.engine.sync_frame_mask_to_sam2` is asynchronously dispatched to commit the changes to SAM 2 memory.
3. **Keyboard Focus Routing Fix (Shortcuts Unblocked Outside Text Entry)**:
   - **User Request**: *"in the 2nd screenshot, now I'm not typing but the keyboard buttons do not work!"*
   - **Root Cause Diagnosed**:
     - In `sam2_masker_ui.py`, `_is_typing_focused()` previously checked `w_class in ("Entry", "TEntry", "Text", "Combobox", "TCombobox")`.
     - Whenever any dropdown in the UI (e.g. Model selection, Overlay color, Export resolution, Cutout format) was clicked or selected, Tkinter maintained focus on that readonly `ttk.Combobox`.
     - Because `Combobox` was classified as typing focused, `_is_typing_focused()` continuously returned `True` even when the user was no longer typing. Consequently, all global viewport shortcuts (Spacebar for playback, Left/Right for stepping, Delete for clearing frame, F for fit to view, `[`/`]` for brush size) were permanently suppressed.
     - Additionally, clicking on `self.canvas` did not automatically transfer keyboard focus (`self.canvas.focus_set()`).
   - **Fix Implemented**:
     - Removed `Combobox` and `TCombobox` from `_is_typing_focused()`. Verified that only editable `Entry` and `Text` widgets with `state not in ("readonly", "disabled")` are treated as typing focused.
     - Added `takefocus=True` to `self.canvas`.
     - Bound `<<ComboboxSelected>>` on all comboboxes (`model_combo`, `cb_scale`, `cb_mask`, `cb_cut`, `color_combo`) to automatically yield focus to `self.canvas` as soon as an item is selected.
     - Added `self.canvas.focus_set()` across canvas mouse events (`_on_canvas_left_down`, `_on_canvas_right_down`, `_on_canvas_alt_down`, `_on_canvas_pan_start`, and `_on_canvas_enter`).
     - Added `<Return>` and `<Escape>` bindings on `text_prompt_entry`, `brush_size_entry`, and all refinement parameter entries to immediately release focus to `self.canvas`.

---

### 2. Focus Routing & Event Handling Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   Keyboard Focus & Tool Event Architecture (v0.1.2.5)                  │
│                                                                                        │
│  [User Keyboard Event] (Space, Left/Right, Delete, F, [, ])                            │
│          │                                                                             │
│          ▼                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     _is_typing_focused(event) EVALUATION                         │  │
│  │                                                                                  │  │
│  │   Focused widget in (Entry, TEntry, Text) AND state not in (readonly, disabled)? │  │
│  │                                                                                  │  │
│  │      ├── YES ──> [Text Input Mode]                                               │  │
│  │      │           • Space types ' ' into Text Prompt / numeric input              │  │
│  │      │           • Left/Right moves cursor within text field                     │  │
│  │      │           • Viewport hotkeys pass through untouched                       │  │
│  │      │                                                                           │  │
│  │      └── NO  ──> [Global Viewport Shortcuts Active]                              │  │
│  │                  • Readonly Comboboxes (Model, Color, Export Scale) NEVER block  │  │
│  │                  • Spacebar ──> Toggles playback                                 │  │
│  │                  • Left/Right ──> Steps frames                                   │  │
│  │                  • Delete ──> Clears current frame prompt                        │  │
│  │                  • 'F' ──> Fits view to canvas                                   │  │
│  │                  • '[' / ']' ──> Adjusts brush radius                            │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     AUTOMATIC FOCUS RECOVERY TO CANVAS                           │  │
│  │                                                                                  │  │
│  │   • Canvas Click / Pan / Drag / Hover ──> self.canvas.focus_set()                │  │
│  │   • Combobox Item Selected (<<ComboboxSelected>>) ──> self.canvas.focus_set()    │  │
│  │   • Text Prompt <Return> (Submit) or <Escape> ──> self.canvas.focus_set()        │  │
│  │   • Slider/Entry Commit (Dilation, Feather, Threshold) ──> self.canvas.focus_set│  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Verification & Test Suite Results

The comprehensive test suite in `test_sam2_masker.py` was executed with all 6 test suites passing with zero errors:

- **Suite 1: SAM 2 AI Masking Core & Hardware Exporters**: **PASS**
  - Model catalog & download verification.
  - Interactive click and box prompts on moving target sequence.
  - Bidirectional temporal propagation across sequence.
  - Mask post-processing ops (dilation, erosion, feathering, invert).
  - Hardware-accelerated WebP & PNG batch export (95+ fps).
- **Suite 2: Mixed Horizontal & Vertical Orientation Suite**: **PASS**
  - Ingestion of mixed landscape and portrait frames via hardware pipeline.
  - Adaptive per-frame aspect ratio mask generation.
  - Native resolution export preserving independent orientation ratios.
- **Suite 3: Advanced Selection, Erase, Box Cut & Brush Tools**: **PASS**
  - AI Bounding Box Marquee with coordinate clamping: **PASS**.
  - Direct Manual Brush Eraser stroke: **PASS**.
  - Multi-level Undo of Brush Eraser: **PASS**.
  - **Box Cut / Marquee Deselect with `sync_to_sam2=False` & Async Sync**: **PASS** (zeroed marquee region without crashing).
  - Multi-level Undo of Box Cut: **PASS** (restored marquee region to 255).
  - Smart Anchor Positive Point on Tracked Frame Erase: **PASS**.
- **Suite 4: Export Stop, Pause & Mask Revision Suite**: **PASS**
  - Mask revision tracking across box, brush, and undo (`0 -> 2 -> 3 -> 4`): **PASS**.
  - Reset frame conditioning purging tracked memory: **PASS**.
  - Export cancellation stopping cleanly after 2/8 frames: **PASS**.
  - Export pause and resume completing all 8 frames: **PASS**.
- **Suite 5: Grounded-SAM 2 Text Prompt & Model Switching Suite**: **PASS**
  - OwLV2 Zero-Shot Text Prompt Detection (`"white disk"`): **PASS** (Box: `[125.6, 125.7, 276.4, 277.2]`, Score: 82.6%).
  - Dynamic Model Switching (`tiny` -> `small` -> `tiny`): **PASS**.
  - Zero-Lag Brush Stroke with decoupled SAM 2 sync: **PASS**.
- **Suite 6: Hardware-Accelerated Ingestion & UI Shortcut Suite**: **PASS**
  - Ingested 8 high-res frames in 0.08s (100.4 fps): **PASS**.
  - LRU frame cache verified (`_frame_rgb_cache`): **PASS**.
  - **Default Export Resolution Verification**: `export_scale_var.get() == "Original (100% Native)"`: **PASS**.
  - **Combobox Focus Non-Blocking Verification**: Readonly comboboxes do NOT set `_is_typing_focused()` to True: **PASS**.
  - **Keyboard Shortcut Execution while Combobox Focused**: Spacebar toggles playback cleanly: **PASS**.
  - **Text Prompt Typing Focus Isolation**: Spacebar and Arrow keys do not trigger playback while actively typing in text entry: **PASS**.
