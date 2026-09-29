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

    # 7. Test RealityScan & RGBA Cutout Exporters
    print("\n[7] Testing Exporters...")
    export_out = os.path.join(temp_dir, "export_output")
    counts = engine.export_masks(
        output_dir=export_out,
        export_binary=True,
        export_cutout=True,
        rs_naming=True,
        mask_format="png",
        postprocess_params={"dilation": 1, "feather": 2, "threshold": 128}
    )
    print(f"    - Export counts: {counts}")
    assert counts["binary"] == 5, f"Expected 5 binary masks, got {counts['binary']}"
    assert counts["cutout"] == 5, f"Expected 5 cutout PNGs, got {counts['cutout']}"

    # Verify RealityScan naming convention
    for i in range(5):
        expected_mask = os.path.join(export_out, f"frame_{i:04d}.mask.png")
        assert os.path.isfile(expected_mask), f"RealityScan mask missing: {expected_mask}"
        # Read mask to check dimensions and values
        m_img = cv2.imread(expected_mask, cv2.IMREAD_GRAYSCALE)
        assert m_img.shape == (H, W), f"Mask shape wrong: {m_img.shape}"
        assert set(np.unique(m_img)).issubset({0, 255}), "Binary mask should contain only 0 and 255"

    # Verify RGBA Cutout
    cutouts_dir = os.path.join(export_out, "cutouts")
    for i in range(5):
        expected_cutout = os.path.join(cutouts_dir, f"frame_{i:04d}_cutout.png")
        assert os.path.isfile(expected_cutout), f"Cutout missing: {expected_cutout}"
        rgba = cv2.imread(expected_cutout, cv2.IMREAD_UNCHANGED)
        assert rgba.shape == (H, W, 4), f"Cutout should have 4 channels (RGBA), got {rgba.shape}"
        # Alpha channel should match subject mask
        alpha = rgba[:, :, 3]
        assert np.sum(alpha > 0) > 500, "Cutout alpha channel is empty"
    print("    [OK] All RealityScan binary masks and RGBA cutouts verified successfully.")

    print("\n" + "=" * 70)
    print("ALL SAM 2 AI MASKING TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_sam2_all()
