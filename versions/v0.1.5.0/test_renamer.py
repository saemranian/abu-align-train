"""
Automated Test Suite for Renamer in Abu Align.
Verifies live preview generation, natural and chronological sorting, safe two-phase
in-place collision renaming, destination copying, one-click undo restoration,
headless CLI execution, and GUI dialog initialization.
"""

import os
import sys
import tempfile
import shutil
import time
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from renamer_engine import RenamerEngine, RenameItem
from renamer_ui import RenamerStudioDialog, open_renamer_dialog


def setup_mock_folder(base_dir: str):
    """Creates synthetic files with varying naming conventions and timestamps."""
    fld = Path(base_dir) / "images_to_rename"
    fld.mkdir(parents=True, exist_ok=True)

    # 1. Unordered alphanumeric names (to verify natural sort)
    # img_1, img_10, img_2, img_20, img_3
    test_names = ["img_1.jpg", "img_10.jpg", "img_2.jpg", "img_20.jpg", "img_3.jpg",
                  "DSC0042.PNG", "DSC0043.PNG", "scan_A.tif", "scan_B.tif"]

    for idx, name in enumerate(test_names):
        p = fld / name
        with open(p, "w") as f:
            f.write(f"Sample content for {name}")
        # Set synthetic timestamps with 5s increments
        t_mtime = time.time() - (len(test_names) - idx) * 10
        os.utime(p, (t_mtime, t_mtime))

    return str(fld)


def test_renamer_all():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE RENAMER ENGINE, CLI & UI TEST SUITE")
    print("=" * 70)

    temp_root = tempfile.mkdtemp(prefix="renamer_test_")
    try:
        fld = setup_mock_folder(temp_root)
        print(f"[1] Created synthetic test folder with 9 files in:\n    {fld}")

        engine = RenamerEngine()

        # ---------------------------------------------------------------------
        # 1. Test Natural Sorting vs Alphabetical Sorting
        # ---------------------------------------------------------------------
        print("\n[2] Testing Natural Alphanumeric Sorting...")
        files_natural = engine.scan_files(fld, sort_by="natural")
        names_nat = [f[1] for f in files_natural if f[1].startswith("img_")]
        # Natural sort MUST put img_1, img_2, img_3, img_10, img_20
        expected_nat = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_10.jpg", "img_20.jpg"]
        assert names_nat == expected_nat, f"Natural sort failed! Got: {names_nat}"
        print(f"    [PASS] Natural sort correctly ordered: {names_nat}")

        # ---------------------------------------------------------------------
        # 2. Test Chronological Sorting (Date Modified)
        # ---------------------------------------------------------------------
        print("\n[3] Testing Chronological Timeline Sorting...")
        files_date = engine.scan_files(fld, sort_by="date_asc")
        # Earlier created files should come first
        mtimes = [f[3] for f in files_date]
        assert mtimes == sorted(mtimes), "Date ascending sort must be monotonically non-decreasing"
        print("    [PASS] Chronological sorting ordered files by timeline timestamps.")

        # ---------------------------------------------------------------------
        # 3. Test Live Preview Generation ("Seeing what final naming looks like")
        # ---------------------------------------------------------------------
        print("\n[4] Testing Live Preview Generation without Disk Modification...")
        preview = engine.generate_preview(
            folder_path=fld,
            base_name="frame",
            start_index=1,
            padding=4,
            delimiter="_",
            sort_by="natural",
            lowercase_ext=True
        )
        assert len(preview) == 9, f"Expected 9 preview items, got {len(preview)}"
        assert preview[0].new_filename == "frame_0001.png" or preview[0].new_filename.startswith("frame_0001"), "First sequenced name should be frame_0001"
        assert preview[-1].new_filename.startswith("frame_0009"), "Last sequenced name should be frame_0009"

        # Verify disk files remain UNTOUCHED after preview
        disk_files_now = set(os.listdir(fld))
        assert "img_1.jpg" in disk_files_now, "Preview must not modify files on disk!"
        print(f"    [PASS] Preview generated in memory. Sample: {preview[0].original_filename} -> {preview[0].new_filename}")

        # ---------------------------------------------------------------------
        # 4. Test Collision-Safe In-Place Renaming (Two-Phase Atomic)
        # ---------------------------------------------------------------------
        print("\n[5] Testing Safe In-Place Renaming (Overwriting inside Input Folder)...")
        # Let's create an intentional shift collision case:
        # e.g., files 'f_1.txt', 'f_2.txt' renaming to 'f_0.txt', 'f_1.txt'
        col_dir = Path(temp_root) / "collision_test"
        col_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 4):
            with open(col_dir / f"item_{i}.txt", "w") as f:
                f.write(f"Content {i}")

        col_items = engine.generate_preview(str(col_dir), base_name="item", start_index=0, padding=1)
        # item_1 -> item_0.txt, item_2 -> item_1.txt (COLLISION with existing item_1!)
        succ_col, err_col = engine.execute_rename(col_items, mode="inplace")
        assert succ_col == 3, f"Expected 3 collision-free renames, got {succ_col}"
        assert err_col == 0, f"Expected 0 errors in collision rename, got {err_col}"

        renamed_in_col = sorted([f for f in os.listdir(col_dir) if not f.startswith(".")])
        assert renamed_in_col == ["item_0.txt", "item_1.txt", "item_2.txt"], f"Unexpected collision results: {renamed_in_col}"
        print("    [PASS] Two-phase atomic rename resolved shift collisions with zero clobbering.")

        # Test Undo of collision rename
        rest_col, err_u = engine.undo_last_rename()
        assert rest_col == 3, f"Expected 3 restored files, got {rest_col}"
        assert sorted([f for f in os.listdir(col_dir) if not f.startswith(".")]) == ["item_1.txt", "item_2.txt", "item_3.txt"]
        print("    [PASS] Undo restored original filenames.")

        # ---------------------------------------------------------------------
        # 5. Test Copy Mode (Copy with new names to another folder)
        # ---------------------------------------------------------------------
        print("\n[6] Testing Copy to Destination Folder with New Names...")
        copy_dest = Path(temp_root) / "renamed_copy_output"
        preview_copy = engine.generate_preview(
            folder_path=fld,
            base_name="scan_orbit",
            start_index=1,
            padding=5,
            dest_folder=str(copy_dest)
        )
        succ_cp, err_cp = engine.execute_rename(preview_copy, mode="copy", dest_folder=str(copy_dest))

        assert succ_cp == 9, f"Expected 9 copied files, got {succ_cp}"
        assert err_cp == 0, f"Expected 0 errors, got {err_cp}"

        # Verify source folder is 100% intact
        assert len([f for f in os.listdir(fld) if not f.startswith(".")]) == 9, "Source folder files must remain untouched in copy mode"
        # Verify destination has sequenced files
        dest_files = sorted([f for f in os.listdir(copy_dest) if not f.startswith(".")])
        assert len(dest_files) == 9, f"Destination folder should have 9 files, got {len(dest_files)}"
        assert any(f.startswith("scan_orbit_00001") for f in dest_files), "First destination file should be scan_orbit_00001"
        print(f"    [PASS] Copy mode generated {len(dest_files)} renamed files in destination while leaving source untouched.")

        # Undo of copy
        rest_cp, err_cpu = engine.undo_last_rename()
        assert rest_cp == 9, "Undo should delete the 9 copied replicas"
        assert len([f for f in os.listdir(copy_dest) if not f.startswith(".")]) == 0, "Destination folder should be empty after copy undo"
        print("    [PASS] Undo copy cleanly removed replicas from destination.")

        # ---------------------------------------------------------------------
        # 6. Test GUI Studio Dialog Initialization
        # ---------------------------------------------------------------------
        print("\n[7] Testing Renamer Studio Dialog GUI Initialization...")
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        dlg = open_renamer_dialog(root, initial_folder=fld, log_callback=lambda msg: None)
        root.update()

        assert dlg.folder_var.get() == fld, "Dialog must bind initial folder"
        assert dlg.base_name_var.get() == "frame", "Default base name should be 'frame'"
        assert dlg.btn_execute is not None, "Execution button must be present"

        dlg.dialog.destroy()
        root.destroy()
        print("    [PASS] Renamer Studio Dialog instantiated cleanly with callbacks and bindings.")

        # ---------------------------------------------------------------------
        # 7. Test Headless CLI in Subprocess
        # ---------------------------------------------------------------------
        print("\n[8] Testing Renamer Headless CLI...")
        cli_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "renamer_cli.py")
        cmd_dry = [
            sys.executable, cli_py,
            "--input", fld,
            "--name", "cam_pass",
            "--dry-run"
        ]
        res_dry = subprocess.run(
            cmd_dry,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        assert "[LIVE PREVIEW]" in res_dry.stdout, f"CLI output missing preview banner. Output:\n{res_dry.stdout}"
        assert "cam_pass_0001" in res_dry.stdout, f"CLI output missing formatted name. Output:\n{res_dry.stdout}"
        print("    [PASS] Headless CLI verified with live dry-run preview.")

        print("\n" + "=" * 70)
        print("ALL RENAMER ENGINE, CLI & UI TESTS PASSED PERFECTLY (100%)!")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    test_renamer_all()
