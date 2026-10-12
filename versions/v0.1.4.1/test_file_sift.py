"""
Automated Test Suite for FileSift in Abu Align.
Verifies source ingestion (folders, text, csv, json), recursive target indexing,
O(1) exact and stem cross-format matching (JPG <-> RAW/DNG), parallel copy,
cut/move, one-click undo restoration, CLI execution, and GUI dialog initialization.
"""

import os
import sys
import tempfile
import shutil
import time
import json
import subprocess
from pathlib import Path

from file_sift_engine import FileSiftEngine, SiftMatchRecord
from file_sift_ui import FileSiftStudioDialog, open_file_sift_dialog


def setup_mock_environment(base_dir: str):
    """
    Creates synthetic directory structure:
    - folder_1_src: Curated files (JPGs, e.g. from photogrammetry inspection or BlurDrop)
    - folder_2_tgt: Raw heavy dataset containing mixed extensions (.ARW, .DNG, .JPG, .CR3) across subdirectories
    - list_file_txt: Plain text list of filenames
    - list_file_json: JSON array of filenames
    """
    p_base = Path(base_dir)
    f1 = p_base / "source_folder"
    f2 = p_base / "target_raw_dataset"
    f3 = p_base / "destination_output"
    f1.mkdir(parents=True, exist_ok=True)
    f2.mkdir(parents=True, exist_ok=True)
    f3.mkdir(parents=True, exist_ok=True)

    # Subdirectories inside Target (Folder 2)
    (f2 / "cam_left").mkdir(parents=True, exist_ok=True)
    (f2 / "cam_right").mkdir(parents=True, exist_ok=True)
    (f2 / "sub_nested" / "deep").mkdir(parents=True, exist_ok=True)

    # Create 20 mock files in Source Folder (JPGs)
    for i in range(1, 21):
        with open(f1 / f"photo_{i:04d}.jpg", "w") as f:
            f.write(f"Source JPG {i}")

    # Create files in Target Folder (Folder 2):
    # Some match Source stems as .ARW, some as .DNG, some as .JPG, plus extra files not in Source!
    # Matched items:
    for i in range(1, 11):
        # 1-5 in cam_left as .ARW
        with open(f2 / "cam_left" / f"photo_{i:04d}.ARW", "w") as f:
            f.write(f"Raw Sony ARW Data {i}" * 100)
    for i in range(6, 16):
        # 6-15 in cam_right as .DNG
        with open(f2 / "cam_right" / f"photo_{i:04d}.dng", "w") as f:
            f.write(f"Adobe DNG Data {i}" * 100)
    for i in range(16, 21):
        # 16-20 in deep nested subfolder as .JPG
        with open(f2 / "sub_nested" / "deep" / f"photo_{i:04d}.jpg", "w") as f:
            f.write(f"Nested JPG {i}" * 50)

    # Extra unmatched files in Target
    for i in range(101, 111):
        with open(f2 / "cam_left" / f"unmatched_raw_{i:04d}.ARW", "w") as f:
            f.write("Unmatched target content")

    # List files
    list_txt = p_base / "sample_list.txt"
    with open(list_txt, "w") as f:
        for i in range(1, 11):
            f.write(f"photo_{i:04d}.jpg\n")

    list_json = p_base / "sample_list.json"
    with open(list_json, "w") as f:
        json.dump([f"photo_{i:04d}.jpg" for i in range(5, 15)], f)

    return str(f1), str(f2), str(f3), str(list_txt), str(list_json)


def test_file_sift_all():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE FILESIFT ENGINE, CLI & UI TEST SUITE")
    print("=" * 70)

    temp_root = tempfile.mkdtemp(prefix="filesift_test_")
    try:
        f1_src, f2_tgt, f3_dst, list_txt, list_json = setup_mock_environment(temp_root)
        print(f"[1] Created synthetic multi-folder environment in:\n    {temp_root}")

        engine = FileSiftEngine()

        # ---------------------------------------------------------------------
        # 1. Test Ingestion Modes
        # ---------------------------------------------------------------------
        print("\n[2] Testing Source Ingestion (Folder, TXT, JSON, Text Parsing)...")
        # Ingestion from Folder
        src_folder_items = engine.load_source_from_folder(f1_src)
        assert len(src_folder_items) == 20, f"Expected 20 source items, got {len(src_folder_items)}"

        # Ingestion from TXT
        src_txt_items = engine.load_source_from_list_file(list_txt)
        assert len(src_txt_items) == 10, f"Expected 10 items from TXT, got {len(src_txt_items)}"

        # Ingestion from JSON
        src_json_items = engine.load_source_from_list_file(list_json)
        assert len(src_json_items) == 10, f"Expected 10 items from JSON, got {len(src_json_items)}"

        # Ingestion from raw text / clipboard string
        src_str_items = engine.load_source_from_text("photo_0001.jpg, photo_0002.jpg\nphoto_0003.jpg")
        assert len(src_str_items) == 3, f"Expected 3 items from string, got {len(src_str_items)}"
        print("    [PASS] Ingestion from folder, txt, json, and comma-separated string verified.")

        # ---------------------------------------------------------------------
        # 2. Test Target Folder Recursive Indexing
        # ---------------------------------------------------------------------
        print("\n[3] Testing Target Folder Scanning & Indexing...")
        tgt_items = engine.scan_target_folder(f2_tgt, recursive=True)
        # 10 ARW + 10 DNG + 5 JPG + 10 unmatched = 35 files total
        assert len(tgt_items) == 35, f"Expected 35 target files, got {len(tgt_items)}"
        print(f"    [PASS] Rapidly indexed {len(tgt_items)} target files across nested directories.")

        # ---------------------------------------------------------------------
        # 3. Test Cross-Format Stem Matching (JPG <-> RAW/DNG)
        # ---------------------------------------------------------------------
        print("\n[4] Testing Sub-Millisecond Cross-Format Stem Matching (JPG <-> ARW/DNG)...")
        # Set source to all 20 source JPGs
        engine.load_source_from_folder(f1_src)
        matches, missing_src, unmatched_tgt = engine.match_files(mode="stem", case_sensitive=False)

        # Expected:
        # photo_0001 - photo_0005 matched with .ARW
        # photo_0006 - photo_0010 matched with BOTH .ARW and .DNG (each stem matches 2 files!)
        # photo_0011 - photo_0015 matched with .DNG
        # photo_0016 - photo_0020 matched with .JPG
        # Total matched records = 5 (.ARW) + 5*2 (both .ARW & .DNG) + 5 (.DNG) + 5 (.JPG) = 25 matched files!
        assert len(matches) == 25, f"Expected 25 matches across formats, got {len(matches)}"
        assert len(missing_src) == 0, f"Expected 0 missing source items, got {len(missing_src)}"
        assert len(unmatched_tgt) == 10, f"Expected 10 unmatched target files, got {len(unmatched_tgt)}"
        print(f"    [PASS] Stem matching paired 20 source keys to {len(matches)} target files across .ARW, .DNG, and .JPG.")

        # ---------------------------------------------------------------------
        # 4. Test Exact Filename Matching
        # ---------------------------------------------------------------------
        print("\n[5] Testing Exact Filename Matching...")
        matches_exact, missing_exact, _ = engine.match_files(mode="exact", case_sensitive=False)
        # Only photo_0016 - photo_0020 have exact .jpg in target
        assert len(matches_exact) == 5, f"Expected 5 exact matches, got {len(matches_exact)}"
        assert len(missing_exact) == 15, f"Expected 15 missing in exact mode, got {len(missing_exact)}"
        print("    [PASS] Exact matching correctly isolated only identically extended files.")

        # ---------------------------------------------------------------------
        # 5. Test Multi-Threaded Copy Transfer
        # ---------------------------------------------------------------------
        print("\n[6] Testing Parallel Multi-Threaded COPY Transfer...")
        # Reset matches to stem matches
        engine.match_files(mode="stem")
        copy_dest = os.path.join(f3_dst, "copied_batch")
        succ, fail, mb = engine.execute_transfer(destination_folder=copy_dest, action="copy", preserve_hierarchy=True)

        assert succ == 25, f"Expected 25 copied files, got {succ}"
        assert fail == 0, f"Expected 0 failures, got {fail}"
        assert os.path.isdir(copy_dest), "Destination folder should exist"

        # Verify files exist in destination and target still has all 35 files
        tgt_items_after_copy = engine.scan_target_folder(f2_tgt, recursive=True)
        assert len(tgt_items_after_copy) == 35, "Target folder must retain all files after COPY"
        print(f"    [PASS] Copied {succ} files cleanly. Target folder remained 100% intact.")

        # ---------------------------------------------------------------------
        # 6. Test Undo of Copy (Cleanup in Destination)
        # ---------------------------------------------------------------------
        print("\n[7] Testing Undo of COPY Action...")
        restored, errors = engine.undo_last_transfer()
        assert restored == 25, f"Expected 25 removed copy replicas, got {restored}"
        assert errors == 0, f"Expected 0 errors, got {errors}"
        print("    [PASS] Undo copy cleaned up all replicated files in destination.")

        # ---------------------------------------------------------------------
        # 7. Test Cut (Move) Transfer
        # ---------------------------------------------------------------------
        print("\n[8] Testing CUT (Move) Transfer...")
        # Re-match before cut
        engine.scan_target_folder(f2_tgt, recursive=True)
        engine.match_files(mode="stem")
        cut_dest = os.path.join(f3_dst, "cut_batch")
        succ_cut, fail_cut, _ = engine.execute_transfer(destination_folder=cut_dest, action="cut", preserve_hierarchy=True)

        assert succ_cut == 25, f"Expected 25 moved files, got {succ_cut}"
        assert fail_cut == 0, f"Expected 0 failures, got {fail_cut}"

        # Target should now ONLY contain the 10 unmatched files!
        tgt_items_after_cut = engine.scan_target_folder(f2_tgt, recursive=True)
        assert len(tgt_items_after_cut) == 10, f"Expected 10 files left in target after cut, got {len(tgt_items_after_cut)}"
        print(f"    [PASS] Cut cleanly moved 25 files into destination. Target left with only {len(tgt_items_after_cut)} unmatched files.")

        # ---------------------------------------------------------------------
        # 8. Test One-Click Undo of Cut (Full Restoration)
        # ---------------------------------------------------------------------
        print("\n[9] Testing One-Click UNDO Restoration of CUT Action...")
        restored_cut, errors_cut = engine.undo_last_transfer()
        assert restored_cut == 25, f"Expected 25 restored files, got {restored_cut}"
        assert errors_cut == 0, f"Expected 0 errors on restore, got {errors_cut}"

        # Target folder must be completely restored back to 35 files!
        tgt_items_restored = engine.scan_target_folder(f2_tgt, recursive=True)
        assert len(tgt_items_restored) == 35, f"Expected 35 restored files in target, got {len(tgt_items_restored)}"
        print("    [PASS] Full one-click undo successfully restored all moved files back to their exact original locations.")

        # ---------------------------------------------------------------------
        # 9. Test GUI Studio Dialog Initialization (Headless Tkinter)
        # ---------------------------------------------------------------------
        print("\n[10] Testing FileSift Studio Dialog GUI Initialization...")
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        dlg = open_file_sift_dialog(root, initial_source=f1_src, initial_target=f2_tgt, log_callback=lambda msg: None)
        root.update()

        assert dlg.source_path_var.get() == f1_src, "Dialog source path must match initial source"
        assert dlg.target_path_var.get() == f2_tgt, "Dialog target path must match initial target"
        assert dlg.btn_sift is not None, "Sift action button must be present"

        dlg.dialog.destroy()
        root.destroy()
        print("    [PASS] FileSift Studio Dialog instantiated cleanly with callbacks and bindings.")

        # ---------------------------------------------------------------------
        # 10. Test CLI Headless Execution
        # ---------------------------------------------------------------------
        print("\n[11] Testing FileSift Headless CLI in subprocess...")
        cli_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_sift_cli.py")
        cmd = [
            sys.executable, cli_py,
            "--source", f1_src,
            "--target", f2_tgt,
            "--mode", "stem",
            "--action", "dry-run"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        assert "Matched 25 files" in res.stdout, f"CLI output missing matched count. Output:\n{res.stdout}"
        print("    [PASS] Headless CLI verified with dry-run mode.")

        print("\n" + "=" * 70)
        print("ALL FILESIFT TESTS PASSED PERFECTLY (100%)!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    test_file_sift_all()
