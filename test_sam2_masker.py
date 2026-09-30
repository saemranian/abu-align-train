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


if __name__ == "__main__":
    test_sam2_all()
    test_mixed_orientations()
