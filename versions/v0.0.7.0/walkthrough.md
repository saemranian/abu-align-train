# Abu Align & Train v0.0.6.9 — Session Diagnostic Log Attribution & Automated Run Archival

Version **v0.0.6.9** introduces full provenance telemetry linking every Photogrammetry Alignment and 3D Gaussian Splatting Training run directly to its exact session diagnostic log (`Log File` and absolute disk `Log Location`), recording these in `reconstruction_report.txt`, `training_report.txt`, and the Fast 3D Interactive Viewer (`fast_viewer.py`), while archiving a duplicate copy of the log directly inside the output results directory.

---

## 1. Problem Description & Requirements

Previously, session telemetry and diagnostic output were captured by the GUI into timestamped log files located in `logs/` (e.g. `logs/abu_align_YYYY-MM-DD_HH-MM-SS.log` or `logs/abu_train_YYYY-MM-DD_HH-MM-SS.log`), but:
1. **Missing Report Attribution**: Neither `reconstruction_report.txt` nor any training summary explicitly recorded the log filename or its location on disk.
2. **Post-Mortem Debugging Gap**: When inspecting an older run in `results/sparse_XX/`, users had to manually search the `logs/` directory and correlate timestamps to find the corresponding console log.
3. **Viewer Disconnect**: The 3D interactive viewer (`fast_viewer.py`) parsed dataset provenance, cameras, and tie point statistics from `reconstruction_report.txt`, but had no visibility into which log captured the run's solver metrics.

---

## 2. Technical Implementation

### A. CLI & GUI Parameter Plumbing (`align_cli.py`, `abu_align_gui.py`, `train_engine.py`)
- Added `--log_file` / `--log_path` CLI arguments to both `align_cli.py` and `train_engine.py`.
- In `abu_align_gui.py`:
  - `_start_alignment`: Automatically forwards `self.active_log_path` (`abu_align_<timestamp>.log`) to `align_cli.py`.
  - `_start_train_only`: Automatically forwards `self.active_log_path` (`abu_train_<timestamp>.log`) to `train_engine.py`.
- **Smart Fallback Discovery**: If the CLI or standalone training engine is executed outside the GUI without `--log_file`, an automated discovery routine scans `logs/` for the latest timestamped `abu_align_*.log` or `abu_train_*.log` (or `abu_align_latest.log`), ensuring seamless attribution across headless runs.

### B. Reconstruction & Training Reports (`reconstruction_report.txt`, `training_report.txt`)
- In `align_cli.py`:
  - Added `Log File: {log_display_name}` and `Log Location: {log_display_path}` directly in the header banner of `reconstruction_report.txt`.
  - Included the log file in the displayed output directory tree.
  - When 3DGS training completes, the `[3DGS TRAINING COMPLETED]` block appends `Log File` and `Log Location`.
  - Automatically copies the session log file directly into `output_dir` (e.g. `results/sparse_XX/abu_align_<timestamp>.log`) using `shutil.copy2`.
- In `train_engine.py`:
  - When training completes, detects any `reconstruction_report.txt` in parent run directories and appends the training completion block with `Log File` and `Log Location`.
  - Writes a standalone `training_report.txt` into `3dgs_models/`.
  - Copies the active log into `3dgs_models/` and the run parent folder.

### C. Fast Interactive 3D Viewer Provenance Card (`fast_viewer.py`)
- Updated `parse_run_report(run_dir)` to parse `log_file` and `log_location` from `reconstruction_report.txt`.
- Displayed `| **Log File** | <logfile> |` directly inside the metadata table in the viewer info panel.

---

## 3. Verification & Testing

1. **Automated Unit Tests** (`tests/test_report_logging.py`):
   - Verified metadata extraction of `log_file` and `log_location` in `fast_viewer.py`.
   - Verified argument parsing for `--log_file` in both CLI runners.
   - Verified automated log archival into result directories.
2. **Full Regression Suite**:
   - Ran `python -m unittest discover tests`: All 49 unit tests passed successfully.
3. **Module Compilation**:
   - Successfully compiled all Python scripts with `py_compile`.

---

## 4. Versioning & Executables
- Version bumped from `v0.0.6.8` to `v0.0.6.9` in compliance with strict base-10 odometer versioning.
- Recompiled Windows C# native launcher executables (`Abu_Align.exe`, `Abu_Align_and_Train.exe`, and historical archives).
- Generated version snapshot archive at `versions/v0.0.6.9/`.
