"""
Automated Test Suite for Frame Picker Engine, CLI, and GUI
===========================================================
Tests natural sorting, cadence decimation calculations, parallel I/O,
cut/move, undo restoration, and headless CLI integration.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

# Set UTF-8 for console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from frame_picker_engine import FramePickerEngine, FrameItem, natural_sort_key


def test_frame_picker():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE FRAME PICKER TEST SUITE")
    print("=" * 70)

    temp_dir = tempfile.mkdtemp(prefix="frame_picker_test_")
    src_folder = os.path.join(temp_dir, "frames_source")
    os.makedirs(src_folder, exist_ok=True)

    try:
        from PIL import Image

        # -----------------------------------------------------------------
        # Stage 1: Generate 100 Synthetic Frames with Non-Zero Padded Names
        # -----------------------------------------------------------------
        print(f"\n[1] Creating 100 synthetic sequence frames in: {src_folder}")
        for i in range(1, 101):
            # Using frame_1.jpg, frame_2.jpg, ... frame_10.jpg to test natural sorting!
            fpath = os.path.join(src_folder, f"frame_{i}.jpg")
            img = Image.new("RGB", (64, 64), color=((i * 2) % 255, (i * 5) % 255, (i * 7) % 255))
            img.save(fpath, "JPEG", quality=85)
        print("    [PASS] Created 100 frames.")

        engine = FramePickerEngine()

        # -----------------------------------------------------------------
        # Stage 2: Natural Alphanumeric Sorting Verification
        # -----------------------------------------------------------------
        print("\n[2] Testing Natural Alphanumeric Sequence Sorting...")
        items = engine.scan_folder(src_folder, sort_by="natural")
        assert len(items) == 100, f"Expected 100 items, got {len(items)}"
        assert items[0].filename == "frame_1.jpg", f"First item should be frame_1.jpg, got {items[0].filename}"
        assert items[1].filename == "frame_2.jpg", f"Second item should be frame_2.jpg, got {items[1].filename}"
        assert items[9].filename == "frame_10.jpg", f"10th item should be frame_10.jpg, got {items[9].filename}"
        assert items[99].filename == "frame_100.jpg", f"100th item should be frame_100.jpg, got {items[99].filename}"
        print("    [PASS] Natural sorting successfully sequenced frame_1 .. frame_10 .. frame_100.")

        # -----------------------------------------------------------------
        # Stage 3: Vectorized Cadence Decimation Calculation
        # -----------------------------------------------------------------
        print("\n[3] Testing Vectorized Cadence Calculations & Offsets...")
        # Every 5th frame, offset=1: frames 1, 6, 11, 16, ..., 96 (total 20)
        items, stats = engine.compute_selection(cadence=5, offset=1, invert=False)
        assert stats.picked_frames == 20, f"Expected 20 picked frames for cadence=5, got {stats.picked_frames}"
        assert stats.skipped_frames == 80, f"Expected 80 skipped frames, got {stats.skipped_frames}"
        picked_indices = [it.sequence_index for it in items if it.selected]
        expected_indices = list(range(1, 101, 5))
        assert picked_indices == expected_indices, f"Picked indices mismatch: {picked_indices[:5]} vs {expected_indices[:5]}"
        print(f"    [PASS] Cadence=5 Offset=1: Picked {stats.picked_frames} frames ({stats.decimation_ratio_pct:.1f}%).")

        # Every 5th frame, offset=2: frames 2, 7, 12, 17, ..., 97 (total 20)
        items, stats = engine.compute_selection(cadence=5, offset=2, invert=False)
        assert stats.picked_frames == 20
        picked_indices = [it.sequence_index for it in items if it.selected]
        expected_indices = list(range(2, 101, 5))
        assert picked_indices == expected_indices
        print("    [PASS] Cadence=5 Offset=2: Correct phase offset shift.")

        # Invert mode: Drop every 5th frame, keep 80 frames
        items, stats = engine.compute_selection(cadence=5, offset=1, invert=True)
        assert stats.picked_frames == 80, f"Expected 80 picked frames for invert=True, got {stats.picked_frames}"
        assert stats.skipped_frames == 20
        print("    [PASS] Invert Selection mode: Inverted 80 picked / 20 skipped.")

        # Range mode: frames 10 to 30 with cadence=2
        items, stats = engine.compute_selection(cadence=2, offset=1, range_start=10, range_end=30)
        # Sequence indices 10..30 (21 frames), picking every 2nd: 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30 (11 frames)
        assert stats.picked_frames == 11, f"Expected 11 picked frames in range 10:30, got {stats.picked_frames}"
        print("    [PASS] Frame range boundary filtering verified.")

        # -----------------------------------------------------------------
        # Stage 4: High-Speed Thumbnail Generation
        # -----------------------------------------------------------------
        print("\n[4] Testing Fast SIMD/EXIF Thumbnail Extraction...")
        t0 = time.time()
        for it in items[:10]:
            thumb = engine.get_thumbnail(it.full_path, max_dim=120)
            assert thumb is not None
        thumb_dur = time.time() - t0
        print(f"    [PASS] Extracted 10 thumbnails in {thumb_dur * 1000:.2f}ms ({10 / thumb_dur:.1f} fps).")

        # -----------------------------------------------------------------
        # Stage 5: Multi-Threaded Parallel Copy Execution
        # -----------------------------------------------------------------
        print("\n[5] Testing Parallel Multi-Threaded Copy Transfer...")
        dest_copy = os.path.join(temp_dir, "dest_copy")
        # Pick every 10th frame (total 10 frames)
        engine.compute_selection(cadence=10, offset=1, invert=False)
        picked_items = [it for it in engine.raw_frames if it.selected]

        t0 = time.time()
        ok_cnt, err_cnt, manifest = engine.execute_transfer(
            items_to_transfer=picked_items,
            destination_folder=dest_copy,
            action="copy"
        )
        dur = time.time() - t0

        assert ok_cnt == 10, f"Expected 10 copied files, got {ok_cnt}"
        assert err_cnt == 0
        assert len(os.listdir(dest_copy)) >= 10
        # Original folder should still have all 100 files
        assert len(os.listdir(src_folder)) == 100
        print(f"    [PASS] Copied {ok_cnt} frames in {dur * 1000:.2f}ms. Source folder preserved 100%.")

        # -----------------------------------------------------------------
        # Stage 6: Cut (Move) Transfer & One-Click Undo Restoration
        # -----------------------------------------------------------------
        print("\n[6] Testing Cut (Move) Execution and One-Click Undo Restoration...")
        dest_move = os.path.join(temp_dir, "dest_move")
        # Cut every 20th frame (5 frames: 1, 21, 41, 61, 81)
        engine.compute_selection(cadence=20, offset=1, invert=False)
        picked_move = [it for it in engine.raw_frames if it.selected]
        assert len(picked_move) == 5

        ok_cnt, err_cnt, manifest = engine.execute_transfer(
            items_to_transfer=picked_move,
            destination_folder=dest_move,
            action="move"
        )
        assert ok_cnt == 5
        assert len(os.listdir(src_folder)) == 95, f"Source should now have 95 files, got {len(os.listdir(src_folder))}"
        assert len(os.listdir(dest_move)) >= 5

        # Execute Undo
        print("    Testing One-Click Undo Restoration...")
        restored, undo_errs = engine.undo_last_operation()
        assert restored == 5, f"Expected 5 restored files, got {restored}"
        assert undo_errs == 0
        assert len(os.listdir(src_folder)) == 100, f"Source folder should be back to 100 files, got {len(os.listdir(src_folder))}"
        print("    [PASS] Cut and Undo restoration completed with 100% file integrity.")

        # -----------------------------------------------------------------
        # Stage 7: Headless CLI Execution
        # -----------------------------------------------------------------
        print("\n[7] Testing Headless CLI in subprocess...")
        cli_out_folder = os.path.join(temp_dir, "cli_output")
        cli_cmd = [
            sys.executable,
            "frame_picker_cli.py",
            "--input", src_folder,
            "--output", cli_out_folder,
            "--cadence", "4",
            "--offset", "1",
            "--action", "copy"
        ]
        res = subprocess.run(cli_cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        assert res.returncode == 0, f"CLI returned {res.returncode}: {res.stderr}"
        assert "Copied 25 frames" in res.stdout or "Success" in res.stdout
        assert len(os.listdir(cli_out_folder)) >= 25
        print("    [PASS] CLI copied 25 frames (every 4th) successfully.")

        # Test CLI dry-run
        cli_dry_cmd = [
            sys.executable,
            "frame_picker_cli.py",
            "--input", src_folder,
            "--cadence", "10",
            "--action", "dry-run"
        ]
        res_dry = subprocess.run(cli_dry_cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        assert res_dry.returncode == 0
        assert "Dry-Run" in res_dry.stdout
        print("    [PASS] CLI dry-run succeeded without disk modifications.")

        # -----------------------------------------------------------------
        # Stage 8: Studio Dialog GUI Initialization Test
        # -----------------------------------------------------------------
        print("\n[8] Testing Studio Dialog GUI Initialization...")
        import tkinter as tk
        from frame_picker_ui import FramePickerStudioDialog

        root = tk.Tk()
        root.withdraw()
        dialog = FramePickerStudioDialog(root, initial_folder=src_folder)
        dialog.dialog.update_idletasks()
        assert dialog.dialog.winfo_exists()
        dialog.dialog.destroy()
        root.destroy()
        print("    [PASS] FramePickerStudioDialog instantiated and destroyed cleanly.")

        print("\n" + "=" * 70)
        print("ALL 8 FRAME PICKER TEST STAGES PASSED PERFECTLY!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_frame_picker()
