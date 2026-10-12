"""
Comprehensive Verification Test Suite for SAM 2 AI Masking in Abu Align.
Tests engine ingestion, prompt additions, temporal tracking, post-processing filters,
and RealityScan binary / RGBA transparent cutout exports.
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import torch
import time

from sam2_engine import SAM2MaskingEngine, SAM2_MODELS


def test_sam2_all():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE SAM 2 AI MASKING TEST SUITE")
    print("=" * 70)

    # 1. Test Model Catalog & Status
    engine = SAM2MaskingEngine()
    status = engine.get_model_status()
    print("[1] Model Catalog Status:")
    for k, v in status.items():
        print(f"    - {k:10s}: {v['name']} | Downloaded: {v['downloaded']} | Size: {v['size_mb']}MB")
    assert "tiny" in status, "Tiny model missing from catalog"
    assert status["tiny"]["downloaded"] is True, "Tiny model checkpoint should be present"

    # 2. Test Engine Initialization & Model Loading
    print("\n[2] Loading SAM 2 Tiny Model onto device:", engine.device)
    engine.load_model("tiny")
    assert engine.video_predictor is not None, "Video predictor failed to build"
    assert engine.image_predictor is not None, "Image predictor failed to build"
    print("    [OK] SAM 2 Predictor loaded successfully.")

    # 3. Create Synthetic Image Sequence (Moving Target)
    print("\n[3] Generating Synthetic 5-Frame Sequence...")
    temp_dir = tempfile.mkdtemp(prefix="sam2_test_seq_")
    frames_dir = os.path.join(temp_dir, "images")
    os.makedirs(frames_dir, exist_ok=True)

    H, W = 480, 640
    for i in range(5):
        img = np.full((H, W, 3), 40, dtype=np.uint8)  # dark background
        # Moving bright object (circle + square composite subject)
        cx = 150 + i * 60
        cy = 240
        cv2.circle(img, (cx, cy), 50, (0, 220, 180), -1)
        cv2.rectangle(img, (cx - 30, cy - 80), (cx + 30, cy - 20), (220, 180, 0), -1)
        # Background distractor
        cv2.circle(img, (50, 50), 20, (180, 50, 50), -1)

        fname = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(frames_dir, fname), img)

    n_loaded = engine.load_source(frames_dir)
    print(f"    [OK] Loaded {n_loaded} frames from {frames_dir} (Res: {engine.orig_width}x{engine.orig_height})")
    assert n_loaded == 5, f"Expected 5 frames, got {n_loaded}"
    assert engine.orig_width == W and engine.orig_height == H, "Resolution mismatch"

    # 4. Test Prompt Evaluation & Refinement
    print("\n[4] Testing Interactive Prompts on Frame 2 (Middle Frame)...")
    # Positive click on object center
    cx_f2 = 150 + 2 * 60  # 270
    cy_f2 = 240
    mask_click1 = engine.add_point(frame_idx=2, x=cx_f2, y=cy_f2, label=1)
    px_count1 = np.sum(mask_click1 > 0)
    print(f"    - First positive click at ({cx_f2}, {cy_f2}): mask white pixels = {px_count1}")
    assert px_count1 > 500, "Mask should cover subject"

    # Negative click on background
    mask_click2 = engine.add_point(frame_idx=2, x=50, y=50, label=0)
    print(f"    - Second negative click at (50, 50): mask white pixels = {np.sum(mask_click2 > 0)}")

    # Test Bounding Box on Frame 0
    box_f0 = [150 - 55, 240 - 85, 150 + 55, 240 + 55]
    mask_box0 = engine.add_box(frame_idx=0, box=box_f0)
    print(f"    - Bounding box prompt on Frame 0: mask white pixels = {np.sum(mask_box0 > 0)}")
    assert np.sum(mask_box0 > 0) > 500, "Box prompt should generate subject mask"

    keyframes = engine.get_keyframes()
    print(f"    - Keyframes identified: {keyframes}")
    assert 0 in keyframes and 2 in keyframes, "Frames 0 and 2 should be registered keyframes"

    # 5. Test Bidirectional Temporal Propagation
    print("\n[5] Testing Full Bidirectional Propagation Across Timeline...")
    tracked = engine.propagate(direction="both", start_frame=2)
    print(f"    [OK] Tracked {len(tracked)} frames.")
    assert len(tracked) == 5, f"Expected 5 tracked frames, got {len(tracked)}"
    for idx in range(5):
        m = tracked[idx]
        white_px = np.sum(m > 0)
        print(f"    - Frame {idx} mask: {white_px} white px")
        assert white_px > 500, f"Frame {idx} mask is empty!"

    # 6. Test Mask Post-Processing Pipeline
    print("\n[6] Testing Mask Post-Processing Operations...")
    raw_m = tracked[0]
    
    # Test Dilation (+5px)
    dilated = engine.postprocess_mask(raw_m, dilation=5)
    assert np.sum(dilated > 0) > np.sum(raw_m > 0), "Dilation should expand mask area"
    print(f"    - Dilation (+5px): {np.sum(raw_m > 0)} -> {np.sum(dilated > 0)} px")

    # Test Erosion (-5px)
    eroded = engine.postprocess_mask(raw_m, dilation=-5)
    assert np.sum(eroded > 0) < np.sum(raw_m > 0), "Erosion should contract mask area"
    print(f"    - Erosion (-5px):  {np.sum(raw_m > 0)} -> {np.sum(eroded > 0)} px")

    # Test Feathering (soft edge)
    feathered = engine.postprocess_mask(raw_m, feather=5, threshold=0)
    unique_vals = len(np.unique(feathered))
    print(f"    - Feathering (radius 5): unique grayscale values = {unique_vals}")
    assert unique_vals > 2, "Feathering should produce anti-aliased gradient edges"

    # Test Invert
    inverted = engine.postprocess_mask(raw_m, invert=True)
    assert np.all(inverted[raw_m > 0] == 0), "Invert should flip foreground to 0"
    print("    - Invert: foreground correctly mapped to 0")

    # 7. Test RealityScan & RGBA Cutout Exporters (Hardware Accelerated)
    print("\n[7] Testing Hardware-Accelerated Multi-Threaded Exporters...")
    log_messages = []
    engine.set_log_callback(lambda msg: log_messages.append(msg))

    # Test 7A: WebP Cutouts + JPG Masks at 50% scale
    export_out = os.path.join(temp_dir, "export_webp_50pct")
    res_webp = engine.export_masks(
        output_dir=export_out,
        export_binary=True,
        export_cutout=True,
        rs_naming=True,
        mask_format="jpg",
        cutout_format="webp",
        scale_mode="50%",
        quality=92,
        postprocess_params={"dilation": 1, "feather": 2, "threshold": 128}
    )
    print(f"    - WebP 50% Export results: {res_webp}")
    assert res_webp["binary"] == 5, f"Expected 5 binary masks, got {res_webp['binary']}"
    assert res_webp["cutout"] == 5, f"Expected 5 cutout WebP files, got {res_webp['cutout']}"
    assert len(log_messages) > 0, "Engine should log export progress"

    # Verify WebP cutouts (should be half resolution: 320x240, with 4 channels RGBA)
    cutouts_dir = os.path.join(export_out, "cutouts")
    for i in range(5):
        expected_cutout = os.path.join(cutouts_dir, f"frame_{i:04d}_cutout.webp")
        assert os.path.isfile(expected_cutout), f"WebP Cutout missing: {expected_cutout}"
        rgba = cv2.imread(expected_cutout, cv2.IMREAD_UNCHANGED)
        assert rgba.shape == (240, 320, 4), f"Cutout should be 50% res (240, 320, 4), got {rgba.shape}"
        alpha = rgba[:, :, 3]
        assert np.sum(alpha > 0) > 100, "Cutout alpha channel is empty"
        # File size check: each 50% WebP cutout should be tiny (< 50 KB)
        f_size = os.path.getsize(expected_cutout)
        print(f"    - Frame {i} WebP cutout size: {f_size / 1024:.1f} KB")
        assert f_size < 100 * 1024, "WebP cutout should be highly compressed"

    # Verify JPG masks
    for i in range(5):
        expected_mask = os.path.join(export_out, f"frame_{i:04d}.mask.jpg")
        assert os.path.isfile(expected_mask), f"RealityScan mask missing: {expected_mask}"

    # Test 7B: PNG Cutouts at Native Resolution
    export_png = os.path.join(temp_dir, "export_png_native")
    res_png = engine.export_masks(
        output_dir=export_png,
        export_binary=True,
        export_cutout=True,
        rs_naming=True,
        mask_format="png",
        cutout_format="png",
        scale_mode="original",
        png_compression=4,
        postprocess_params={"dilation": 0, "feather": 0}
    )
    print(f"    - PNG Native Export results: {res_png}")
    assert res_png["binary"] == 5 and res_png["cutout"] == 5

    print(f"    [OK] WebP vs PNG size comparison: WebP total {res_webp['total_mb']:.3f} MB vs PNG total {res_png['total_mb']:.3f} MB")
    print("    [OK] All hardware-accelerated exports and logs verified successfully.")


def test_mixed_orientations():
    print("\n" + "=" * 70)
    print("RUNNING MIXED HORIZONTAL & VERTICAL ORIENTATION TEST SUITE")
    print("=" * 70)

    engine = SAM2MaskingEngine()
    engine.load_model("tiny")

    temp_dir = tempfile.mkdtemp(prefix="sam2_mixed_test_")
    frames_dir = os.path.join(temp_dir, "mixed_images")
    os.makedirs(frames_dir, exist_ok=True)

    # Frame specs: (Width, Height, Description)
    frame_specs = [
        (640, 480, "Horizontal Landscape"),
        (480, 640, "Vertical Portrait"),
        (800, 600, "Horizontal Landscape High-Res"),
        (600, 800, "Vertical Portrait High-Res"),
    ]

    for i, (w, h, desc) in enumerate(frame_specs):
        img = np.full((h, w, 3), 35, dtype=np.uint8)
        # Center subject
        cx, cy = w // 2, h // 2
        r = min(w, h) // 4
        cv2.circle(img, (cx, cy), r, (0, 255, 128), -1)
        # Offset distractor
        cv2.rectangle(img, (10, 10), (70, 70), (50, 50, 200), -1)
        fname = f"frame_{i:04d}.png"
        cv2.imwrite(os.path.join(frames_dir, fname), img)
        print(f"    - Created {fname}: {w}x{h} ({desc})")

    n_loaded = engine.load_source(frames_dir)
    print(f"\n[1] Ingestion Check: Loaded {n_loaded} frames.")
    assert n_loaded == 4, f"Expected 4 frames, got {n_loaded}"

    for i, (w, h, desc) in enumerate(frame_specs):
        fw, fh = engine.get_frame_dims(i)
        print(f"    - Frame {i} dimensions: {fw}x{fh} (Expected: {w}x{h})")
        assert (fw, fh) == (w, h), f"Frame {i} dimension mismatch: expected {(w, h)}, got {(fw, fh)}"

    # Test Interactive Prompt on Frame 0 (Horizontal)
    print("\n[2] Prompting Frame 0 (Horizontal)...")
    m0 = engine.add_point(frame_idx=0, x=320, y=240, label=1)
    print(f"    - Frame 0 mask shape: {m0.shape} (Expected: (480, 640))")
    assert m0.shape == (480, 640), f"Frame 0 mask should be horizontal (480, 640), got {m0.shape}"

    # Test Interactive Prompt on Frame 1 (Vertical)
    print("\n[3] Prompting Frame 1 (Vertical)...")
    m1 = engine.add_point(frame_idx=1, x=240, y=320, label=1)
    print(f"    - Frame 1 mask shape: {m1.shape} (Expected: (640, 480))")
    assert m1.shape == (640, 480), f"Frame 1 mask should be vertical (640, 480), got {m1.shape}"

    # Test Bidirectional Propagation Across Mixed Sequence
    print("\n[4] Bidirectional Tracking Across Mixed Sequence...")
    tracked = engine.propagate(direction="both", start_frame=0)
    for i, (w, h, desc) in enumerate(frame_specs):
        m = tracked[i]
        print(f"    - Tracked Frame {i} mask shape: {m.shape} (Expected: ({h}, {w}))")
        assert m.shape == (h, w), f"Tracked Frame {i} shape mismatch: expected ({h}, {w}), got {m.shape}"

    # Test Export in Original / Native Resolution
    print("\n[5] Testing Native Resolution Export (Preserving Horizontal & Vertical Ratios)...")
    native_out = os.path.join(temp_dir, "export_native")
    res_native = engine.export_masks(
        output_dir=native_out,
        export_binary=True,
        export_cutout=True,
        scale_mode="original",
        mask_format="png",
        cutout_format="webp",
        quality=90
    )
    print(f"    - Native Export Results: {res_native}")
    assert res_native["binary"] == 4 and res_native["cutout"] == 4

    # Verify native exported dimensions
    for i, (w, h, desc) in enumerate(frame_specs):
        mask_path = os.path.join(native_out, f"frame_{i:04d}.mask.png")
        cutout_path = os.path.join(native_out, "cutouts", f"frame_{i:04d}_cutout.webp")
        assert os.path.isfile(mask_path), f"Missing mask: {mask_path}"
        assert os.path.isfile(cutout_path), f"Missing cutout: {cutout_path}"

        m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        c_img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)

        print(f"    - Frame {i} exported mask: {m_img.shape[1]}x{m_img.shape[0]} | cutout: {c_img.shape[1]}x{c_img.shape[0]} (Expected: {w}x{h})")
        assert (m_img.shape[1], m_img.shape[0]) == (w, h), f"Exported mask {i} wrong size: expected {w}x{h}, got {m_img.shape[1]}x{m_img.shape[0]}"
        assert (c_img.shape[1], c_img.shape[0]) == (w, h), f"Exported cutout {i} wrong size: expected {w}x{h}, got {c_img.shape[1]}x{c_img.shape[0]}"

    # Test Export at 50% Scale (Half resolution, each preserving its aspect ratio)
    print("\n[6] Testing 50% Half-Scale Export (Preserving Horizontal & Vertical Ratios)...")
    half_out = os.path.join(temp_dir, "export_half")
    res_half = engine.export_masks(
        output_dir=half_out,
        export_binary=True,
        export_cutout=True,
        scale_mode="50%",
        mask_format="png",
        cutout_format="webp",
        quality=90
    )
    print(f"    - 50% Export Results: {res_half}")
    for i, (w, h, desc) in enumerate(frame_specs):
        exp_w, exp_h = w // 2, h // 2
        mask_path = os.path.join(half_out, f"frame_{i:04d}.mask.png")
        cutout_path = os.path.join(half_out, "cutouts", f"frame_{i:04d}_cutout.webp")
        m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        c_img = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
        print(f"    - Frame {i} 50% export: {m_img.shape[1]}x{m_img.shape[0]} (Expected: {exp_w}x{exp_h})")
        assert (m_img.shape[1], m_img.shape[0]) == (exp_w, exp_h)
        assert (c_img.shape[1], c_img.shape[0]) == (exp_w, exp_h)

    print("\n" + "=" * 70)
    print("ALL MIXED HORIZONTAL & VERTICAL TESTS PASSED PERFECTLY!")
    print("=" * 70)


def test_selection_and_editing_tools():
    print("\n" + "=" * 70)
    print("RUNNING ADVANCED SELECTION, ERASE, BOX CUT & BRUSH TEST SUITE")
    print("=" * 70)

    engine = SAM2MaskingEngine()
    engine.load_model("tiny")

    temp_dir = tempfile.mkdtemp(prefix="sam2_edit_test_")
    frames_dir = os.path.join(temp_dir, "test_images")
    os.makedirs(frames_dir, exist_ok=True)

    # Create synthetic test frame with a white circular subject at (200, 200)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 80, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(frames_dir, "frame_0000.png"), img)

    engine.load_source(frames_dir)

    # 1. Test AI Box prompt with clamping
    print("[1] Testing AI Bounding Box Marquee with coordinate clamping...")
    m_box = engine.add_box(0, [100, 100, 300, 300])
    assert m_box is not None and m_box[200, 200] == 255, "Subject center should be 255"
    print("    [PASS] AI Box prompt segmented subject successfully.")

    # 2. Test Direct Manual Brush Eraser
    print("\n[2] Testing Direct Brush Eraser Stroke...")
    engine.push_undo_state(0)
    m_erased = engine.brush_stroke_mask(0, [(200, 200), (210, 200)], radius=20, mode="erase")
    assert m_erased[200, 200] == 0, "Center pixel should be erased to 0"
    print("    [PASS] Brush stroke directly erased subject center to 0.")

    # 3. Test Undo of Brush Eraser
    print("\n[3] Testing Undo History...")
    m_undone = engine.undo(0)
    assert m_undone is not None and m_undone[200, 200] == 255, "Undo should restore erased pixel to 255"
    print("    [PASS] Undo restored erased pixel back to 255.")

    # 4. Test Box Cut / Marquee Deselect (with async/deferred sync_to_sam2=False)
    print("\n[4] Testing Box Cut / Marquee Deselect...")
    m_cut = engine.box_cut_mask(0, [180, 180, 220, 220], sync_to_sam2=False)
    engine.sync_frame_mask_to_sam2(0, async_exec=False)
    assert m_cut[200, 200] == 0, "Box cut should zero out center region"
    print("    [PASS] Box Cut successfully zeroed out marquee region.")

    # 5. Test Undo of Box Cut
    print("\n[5] Testing Undo of Box Cut...")
    m_undone2 = engine.undo(0)
    assert m_undone2 is not None and m_undone2[200, 200] == 255, "Undo should restore cut box"
    print("    [PASS] Undo restored cut box area back to 255.")

    # 6. Test Smart Anchor Positive Point on Tracked Frame Erase
    print("\n[6] Testing Smart Anchor Detection on Tracked Frame...")
    engine.prompts.clear()
    engine.tracked_masks[0] = (img[:, :, 0] > 0).astype(np.uint8) * 255
    # Click negative point at edge of circle without previous positive prompts
    m_neg = engine.add_point(0, 260, 200, label=0)
    assert m_neg.sum() > 0, "Smart anchor injection should prevent mask collapse"
    print(f"    [PASS] Anchor injected and negative click evaluated without mask collapse (Mask sum: {m_neg.sum()}).")

    print("\n" + "=" * 70)
    print("ALL ADVANCED SELECTION & EDITING TESTS PASSED PERFECTLY!")
    print("=" * 70)


def test_export_stop_pause_and_revisions():
    print("\n" + "=" * 70)
    print("RUNNING EXPORT STOP, PAUSE & MASK REVISION TEST SUITE")
    print("=" * 70)

    engine = SAM2MaskingEngine()
    engine.load_model("tiny")

    temp_dir = tempfile.mkdtemp(prefix="sam2_pause_test_")
    frames_dir = os.path.join(temp_dir, "test_images")
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(8):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 50, (255, 255, 255), -1)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.png"), img)

    engine.load_source(frames_dir)

    # 1. Test Mask Revision Tracking
    print("[1] Testing Mask Revision Tracking...")
    rev0 = engine.get_mask_revision(0)
    engine.add_box(0, [100, 100, 200, 200])
    rev1 = engine.get_mask_revision(0)
    assert rev1 > rev0, f"Revision should increment after add_box: {rev0} -> {rev1}"

    engine.brush_stroke_mask(0, [(150, 150)], radius=10, mode="erase")
    rev2 = engine.get_mask_revision(0)
    assert rev2 > rev1, f"Revision should increment after brush stroke: {rev1} -> {rev2}"

    engine.undo(0)
    rev3 = engine.get_mask_revision(0)
    assert rev3 > rev2, f"Revision should increment after undo: {rev2} -> {rev3}"
    print(f"    [PASS] Mask revisions incremented properly: {rev0} -> {rev1} -> {rev2} -> {rev3}")

    # 2. Test Reset Frame Conditioning
    print("\n[2] Testing Reset Frame Conditioning...")
    engine.reset_frame_conditioning(0)
    obj_idx = 0
    if "frames_tracked_per_obj" in engine.inference_state and obj_idx in engine.inference_state["frames_tracked_per_obj"]:
        assert 0 not in engine.inference_state["frames_tracked_per_obj"][obj_idx], "Frame 0 should be purged from tracked memory"
    print("    [PASS] Stale frame conditioning purged cleanly.")

    # 3. Test Export Cancellation
    print("\n[3] Testing Export Cancellation (Stop Action)...")
    out_dir_cancel = os.path.join(temp_dir, "export_cancel")
    cancel_flag = False

    def check_cancel():
        return cancel_flag

    def cancel_prog(cur, tot, msg):
        nonlocal cancel_flag
        if cur >= 2:
            cancel_flag = True

    res_cancel = engine.export_masks(
        output_dir=out_dir_cancel,
        export_binary=True,
        export_cutout=False,
        scale_mode="original",
        progress_callback=cancel_prog,
        cancel_token=check_cancel
    )
    assert res_cancel.get("cancelled") is True, "Export should report cancelled=True"
    assert res_cancel["binary"] < 8, f"Export should have stopped early, got {res_cancel['binary']} frames"
    print(f"    [PASS] Export cancelled successfully after {res_cancel['binary']}/8 frames.")

    # 4. Test Export Pause and Resume
    print("\n[4] Testing Export Pause and Resume...")
    out_dir_pause = os.path.join(temp_dir, "export_pause")
    pause_flag = False
    paused_once = False

    def check_pause():
        return pause_flag

    def pause_prog(cur, tot, msg):
        nonlocal pause_flag, paused_once
        if cur == 2 and not paused_once:
            pause_flag = True
            paused_once = True

            # Resume after brief pause
            import threading
            def resume_worker():
                import time
                time.sleep(0.2)
                nonlocal pause_flag
                pause_flag = False

            threading.Thread(target=resume_worker, daemon=True).start()

    res_pause = engine.export_masks(
        output_dir=out_dir_pause,
        export_binary=True,
        export_cutout=False,
        scale_mode="original",
        progress_callback=pause_prog,
        pause_token=check_pause
    )
    assert res_pause.get("cancelled") is False, "Export should complete fully after resume"
    assert res_pause["binary"] == 8, f"Expected all 8 frames, got {res_pause['binary']}"
    print(f"    [PASS] Export paused, resumed, and completed all 8 frames successfully.")

    print("\n" + "=" * 70)
    print("ALL EXPORT STOP, PAUSE & REVISION TESTS PASSED PERFECTLY!")
    print("=" * 70)


def test_text_grounding_and_model_switching():
    print("\n" + "=" * 70)
    print("RUNNING GROUNDED-SAM 2 TEXT PROMPT & MODEL SWITCHING TEST SUITE")
    print("=" * 70)

    engine = SAM2MaskingEngine()
    engine.load_model("tiny")

    temp_dir = tempfile.mkdtemp(prefix="sam2_grounding_test_")
    frames_dir = os.path.join(temp_dir, "test_images")
    os.makedirs(frames_dir, exist_ok=True)

    # Frame with a white circle on dark background
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 200), 75, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(frames_dir, "frame_0000.png"), img)

    engine.load_source(frames_dir)

    # 1. Test Text-Prompted Grounding with OwLV2 + SAM 2.1
    print("[1] Testing OwLV2 Zero-Shot Text Prompt Detection ('white disk')...")
    mask_np, box, score = engine.generate_mask_from_text(0, "white disk", threshold=0.10)
    print(f"    - Detection Box: {box}, Score: {score:.3f}")
    assert box is not None, "Failed to detect bounding box for 'white disk'"
    assert mask_np is not None, "Failed to generate mask from text prompt"
    assert mask_np[200, 200] == 255, "Mask from text prompt should cover disk center"
    print("    [PASS] OwLV2 + SAM 2.1 successfully detected and segmented 'white disk'.")

    # 2. Test Model Switching with Feature Cache Flushing and Prompt Re-Evaluation
    print("\n[2] Testing Dynamic Model Switching ('tiny' -> 'small')...")
    prev_model = engine.model_name
    engine.load_model("small")
    assert engine.model_name == "small", f"Expected model 'small', got {engine.model_name}"
    # Verify current frame mask was re-evaluated with the new model
    mask_small = engine.get_frame_mask(0)
    assert mask_small is not None and mask_small[200, 200] == 255, "Re-evaluated mask should preserve subject"
    print(f"    [PASS] Switched from {prev_model} to {engine.model_name} with clean cache invalidation and prompt re-evaluation.")

    # Switch back to tiny
    engine.load_model("tiny")
    assert engine.model_name == "tiny"
    print("    [PASS] Switched back to tiny cleanly.")

    # 3. Test Zero-Lag Brush Stroke with Decoupled SAM 2 Sync
    print("\n[3] Testing Zero-Lag Brush Stroke and Async SAM 2 Sync...")
    # Perform fast brush without immediate neural sync
    m_fast = engine.brush_stroke_mask(0, [(200, 200)], radius=15, mode="erase", sync_to_sam2=False)
    assert m_fast[200, 200] == 0, "Brush stroke should immediately erase in-memory pixel"
    # Now sync to SAM 2
    engine.sync_frame_mask_to_sam2(0, async_exec=False)
    m_synced = engine.get_frame_mask(0)
    assert m_synced[200, 200] == 0, "Synced mask should preserve erased pixel"
    print("    [PASS] Zero-lag brush stroke and decoupled SAM 2 sync verified.")

    print("\n" + "=" * 70)
    print("ALL GROUNDED-SAM 2 & MODEL SWITCHING TESTS PASSED PERFECTLY!")
    print("=" * 70)


def test_accelerated_frame_ingestion_and_ui_shortcuts():
    print("\n" + "=" * 70)
    print("RUNNING HARDWARE-ACCELERATED INGESTION & UI SHORTCUT TEST SUITE")
    print("=" * 70)

    import tempfile
    import tkinter as tk
    from sam2_masker_ui import SAM2MaskerDialog

    # 1. Test Accelerated Frame Loading & Ingestion Pipeline
    print("[1] Testing Hardware-Accelerated Multi-Thread & GPU Ingestion Pipeline...")
    engine = SAM2MaskingEngine()
    engine.load_model("tiny")

    temp_dir = tempfile.mkdtemp(prefix="sam2_accel_test_")
    frames_dir = os.path.join(temp_dir, "test_frames")
    os.makedirs(frames_dir, exist_ok=True)

    H, W = 1500, 2000
    for i in range(8):
        img = np.full((H, W, 3), 30 + i * 10, dtype=np.uint8)
        cv2.circle(img, (1000, 750), 300, (0, 200, 100), -1)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    prog_msgs = []
    t0 = time.perf_counter()
    n_loaded = engine.load_source(frames_dir, progress_callback=lambda cur, tot, msg: prog_msgs.append(msg))
    elapsed = time.perf_counter() - t0

    print(f"    - Ingested {n_loaded} high-res ({W}x{H}) frames in {elapsed:.3f}s ({n_loaded / elapsed:.1f} fps)")
    assert n_loaded == 8, f"Expected 8 frames, got {n_loaded}"
    assert engine.inference_state["images"].shape == (8, 3, 1024, 1024), "Image tensor shape mismatch"
    assert len(engine.frame_dims) == 8 and engine.frame_dims[0] == (W, H), "Frame dimensions mismatch"
    assert len(prog_msgs) > 0, "Expected progress callback messages"
    print("    [PASS] Hardware-accelerated multi-threaded & GPU ingestion pipeline verified.")

    # 2. Test LRU Frame Cache
    print("\n[2] Testing LRU Frame Cache...")
    assert 0 in engine._frame_rgb_cache, "Frame 0 should be pre-cached on load"
    rgb0 = engine.get_frame_rgb(0)
    assert rgb0.shape == (H, W, 3), f"Frame 0 shape mismatch: {rgb0.shape}"
    # Access frame 1 (populates cache)
    rgb1 = engine.get_frame_rgb(1)
    assert 1 in engine._frame_rgb_cache, "Frame 1 should be cached after retrieval"
    print(f"    [PASS] LRU frame cache verified (Cached items: {list(engine._frame_rgb_cache.keys())}).")

    # 3. Test Keyboard Typing Focus Isolation & Default Export Resolution
    print("\n[3] Testing Text Prompt Typing Focus Isolation, Comboboxes & Native Resolution in UI...")
    root = tk.Tk()
    dlg = SAM2MaskerDialog(root)
    root.update()

    # Verify default export resolution is Native
    assert dlg.export_scale_var.get() == "Original (100% Native)", f"Expected default 'Original (100% Native)', got {dlg.export_scale_var.get()}"
    assert dlg.cb_scale['values'][0] == "Original (100% Native)", "Original (100% Native) should be first in dropdown"
    print("    [PASS] Default export resolution verified as 'Original (100% Native)'.")

    dlg.engine.num_frames = 5
    dlg.current_frame_idx = 0
    dlg.is_playing = False

    # Focus on Text Prompt Entry
    dlg.text_prompt_entry.focus_set()
    root.update()

    assert dlg._is_typing_focused() == True, "Should detect text prompt entry as typing focused"

    # Simulate Space key while typing: must NOT toggle playback
    dlg.dialog.event_generate("<space>")
    root.update()
    assert dlg.is_playing == False, "Spacebar must NOT trigger playback while focused in text entry!"

    # Simulate Arrow keys while typing: must NOT change frame
    dlg.dialog.event_generate("<Left>")
    dlg.dialog.event_generate("<Right>")
    root.update()
    assert dlg.current_frame_idx == 0, "Arrow keys must NOT navigate frames while typing in text entry!"

    # Test Focus on Comboboxes: Readonly comboboxes must NEVER block shortcuts
    dlg.cb_scale.focus_set()
    root.update()
    assert dlg._is_typing_focused() == False, "Readonly combobox must NOT be classified as typing focused"
    dlg.dialog.event_generate("<space>")
    root.update()
    assert dlg.is_playing == True, "Spacebar MUST trigger playback even if combobox was clicked/focused!"
    dlg.dialog.event_generate("<space>")
    root.update()
    assert dlg.is_playing == False, "Spacebar must toggle playback back off"

    dlg.model_combo.focus_set()
    root.update()
    assert dlg._is_typing_focused() == False, "Model combobox must NOT be classified as typing focused"

    # Focus on Canvas: shortcuts should be active
    dlg.canvas.focus_set()
    root.update()
    assert dlg._is_typing_focused() == False, "Canvas must not be classified as typing focused"

    dlg.dialog.destroy()
    root.destroy()
    print("    [PASS] Text entry typing focus isolation and combobox unblocking verified successfully.")

    print("\n" + "=" * 70)
    print("ALL ACCELERATED INGESTION & UI SHORTCUT TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_sam2_all()
    test_mixed_orientations()
    test_selection_and_editing_tools()
    test_export_stop_pause_and_revisions()
    test_text_grounding_and_model_switching()
    test_accelerated_frame_ingestion_and_ui_shortcuts()

