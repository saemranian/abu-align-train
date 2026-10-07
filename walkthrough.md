# Walkthrough: Abu Align & Train v0.1.2.8
## Renamer: Interactive Sequence Naming, Live Preview, and Atomic Batch Renaming Studio

### 1. Executive Summary
Release **v0.1.2.8** introduces **Renamer**, a high-performance, sequence and batch file renaming studio designed for computer vision, 3D Gaussian Splatting (3DGS), VFX, photogrammetry, and neural rendering pipelines.

Datasets often arrive with erratic, unordered, or camera-generated file names (e.g. `DSC0042.PNG`, `IMG_1.jpg`, `img_10.jpg`, `DJI_0088.JPG`). Aligning and training engines require consistent, deterministic sequence numbering (e.g. `frame_0001.png`, `frame_0002.png`). **Renamer** provides an intuitive, robust solution with:
1. **Live Interactive Preview**: A reactive side-by-side table displaying the exact mapping of `Original Filename` ➔ `New Sequenced Name` in real-time as you type, before making any changes to files on disk.
2. **Two Execution Modes**:
   - **✏️ Overwrite in Input Folder (In-Place)**: Renames files directly inside the source directory using a collision-free two-phase atomic rename algorithm.
   - **📋 Copy with New Names to Another Folder**: Preserves the original source files completely untouched, duplicating and sequencing the files into a specified destination directory.
3. **Smart Sorting Modes**:
   - **Human Natural Alphanumeric Sort**: Recognizes embedded numbers so `img_1.jpg`, `img_2.jpg`, ..., `img_10.jpg` are sorted correctly rather than standard ASCII (`img_1.jpg`, `img_10.jpg`, `img_2.jpg`).
   - **Timeline Chronological Sort**: Orders files by filesystem creation/modification timestamps (`Oldest First` or `Newest First`), ideal for continuous video frame grabs or burst shots.
   - **Reverse Natural Sort & Raw ASCII Sort**.
4. **Flexible Sequence Configuration**:
   - Base sequence name prompt (e.g. `frame`, `cam01`, `scan`).
   - Configurable start index (e.g. `0`, `1`, `100`).
   - Configurable zero-padding width (e.g. `4` for `0001`, `5` for `00001`, `0` for unpadded `1`).
   - Delimiter selection (`_`, `-`, `.` or empty).
   - Extension casing control (preserve, lowercase, uppercase).
5. **One-Click Safe Undo Restoration**: Automatically logs a timestamped JSON manifest (`.renamer_manifest_<timestamp>.json`) in the folder, enabling instant, one-click reversal of in-place renames or destination copies.
6. **Headless CLI (`renamer_cli.py`)**: Scriptable command-line interface with `--dry-run` support for terminal pipelines and automated batch jobs.

---

### 2. Algorithmic Architecture & Safe Two-Phase Renaming

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Renamer Safe In-Place Renaming Pipeline                         │
│                                                                                        │
│  [Input Folder] (Unordered files: img_1.jpg, img_2.jpg, img_3.jpg...)                  │
│          │                                                                             │
│          ▼                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     PHASE 0: LIVE REACTIVE PREVIEW ENGINE                        │  │
│  │                                                                                  │  │
│  │  • Natural Alphanumeric Key: re.split(r'(\d+)', filename)                        │  │
│  │  • Sub-millisecond formatting without touching disk                              │  │
│  │  • Real-time validation against duplicate target stems                           │  │
│  └────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                           │                                            │
│                      User Clicks 'Rename Files'                                        │
│                                           ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     PHASE 1: ATOMIC TEMPORARY SENTINELS                          │  │
│  │                                                                                  │  │
│  │  • Files renamed to: .tmp_renamer_<uuid>_<index>.tmp                             │  │
│  │  • Eliminates cyclic collisions (e.g., img_1 -> img_0, img_2 -> img_1)           │  │
│  │  • Prevents in-order overwrite clobbering on Windows filesystems                 │  │
│  └────────────────────────────────────────┬─────────────────────────────────────────┘  │
│                                           │                                            │
│                                           ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
│  │                     PHASE 2: FINAL SEQUENCE COMMIT                               │  │
│  │                                                                                  │  │
│  │  • Temp sentinels committed to final target names (e.g. frame_0001.jpg)          │  │
│  │  • Timestamped manifest generated for 100% reversible Undo                       │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Why Two-Phase Renaming Matters
In naive single-pass renaming, shifting filenames (such as renaming `img_1.jpg` to `img_0.jpg`, `img_2.jpg` to `img_1.jpg`) can fail or overwrite data because the target name already exists on disk. 
**Renamer Engine** solves this by:
1. Renaming all source files into temporary, cryptographically random sentinel filenames (`.tmp_renamer_<uuid>_<idx>`).
2. Committing temporary sentinels into the final sequenced names.
If any step encounters an unexpected error or lock, the engine automatically rolls back all intermediate operations.

---

### 3. User Interface & Controls

The **Renamer Studio** dialog is accessed from:
- Main Window: **Dataset Row** ➔ Click `🏷️ Renamer...`
- Main Window: **Prepare Dropdown** ➔ Click `🏷️ Renamer (Sequence Naming)`
- Main Window: **Tools Menu** ➔ Click `🏷️ Renamer: Sequence & Batch File Renamer...`

#### Studio Dialog Controls
1. **Directories**:
   - **Input Folder**: Select source directory containing images/files to rename.
   - **Mode Selection**:
     - `✏️ Overwrite in Input Folder (In-Place)`: Renames files in their current folder.
     - `📋 Copy with New Names to Another Folder`: Duplicates files into an output directory.
   - **Destination Folder**: Enabled when copy mode is active.
2. **Sequence Settings**:
   - **Base Sequence Name**: Prompt text (e.g. `frame`, `cam_rig`, `capture`).
   - **Start Index**: Starting number (default `1`).
   - **Zero Padding**: Digits width (default `4` ➔ `0001`, `0002`).
   - **Delimiter**: Delimiter between base name and number (`_`, `-`, `.` or None).
   - **Sort Order**: `Natural (1, 2, 10)`, `Chronological (Oldest First)`, `Chronological (Newest First)`, `Reverse Natural`, or `Raw ASCII`.
   - **Extension Casing**: `Preserve`, `lowercase (.jpg)`, `UPPERCASE (.JPG)`.
3. **Live Preview Table**:
   - Side-by-side columns: `#`, `Original Filename`, `New Sequenced Name`, `Size`, `Status`.
   - As you change any sequence setting, the table updates in real-time instantly without touching disk.
4. **Execution & Restoration**:
   - **Rename Files Button**: Executes in-place rename or destination copy with progress updates.
   - **Undo Last Rename Button**: Reverts the last operation using the saved manifest.

---

### 4. Headless CLI Usage

Renamer is fully usable without a GUI via `renamer_cli.py`:

```bash
# Preview what the renaming will look like (Dry-Run, no disk writes):
python renamer_cli.py --input "C:\Datasets\MyScans" --name "frame" --padding 4 --dry-run

# In-place sequence rename:
python renamer_cli.py --input "C:\Datasets\MyScans" --name "frame" --padding 4 --start-index 1

# Copy files with sequenced names to another folder:
python renamer_cli.py --input "C:\Datasets\RawShots" --dest "C:\Datasets\Sequenced" --name "cam01" --copy

# Sort chronologically by capture date:
python renamer_cli.py --input "C:\Datasets\Burst" --name "burst" --sort date_asc
```

---

### 5. Automated Verification & Test Results

The automated test suite in [`test_renamer.py`](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/test_renamer.py) passed all 8 stages with 100% success:

```
======================================================================
RUNNING COMPREHENSIVE RENAMER ENGINE, CLI & UI TEST SUITE
======================================================================
[1] Created synthetic test folder with 9 files
[2] Testing Natural Alphanumeric Sorting...
    [PASS] Natural sort correctly ordered: ['img_1.jpg', 'img_2.jpg', 'img_3.jpg', 'img_10.jpg', 'img_20.jpg']
[3] Testing Chronological Timeline Sorting...
    [PASS] Chronological sorting ordered files by timeline timestamps.
[4] Testing Live Preview Generation without Disk Modification...
    [PASS] Preview generated in memory. Sample: DSC0042.PNG -> frame_0001.png
[5] Testing Safe In-Place Renaming (Overwriting inside Input Folder)...
    [PASS] Two-phase atomic rename resolved shift collisions with zero clobbering.
    [PASS] Undo restored original filenames.
[6] Testing Copy to Destination Folder with New Names...
    [PASS] Copy mode generated 9 renamed files in destination while leaving source untouched.
    [PASS] Undo copy cleanly removed replicas from destination.
[7] Testing Renamer Studio Dialog GUI Initialization...
    [PASS] Renamer Studio Dialog instantiated cleanly with callbacks and bindings.
[8] Testing Renamer Headless CLI...
    [PASS] Headless CLI verified with live dry-run preview.
======================================================================
ALL RENAMER ENGINE, CLI & UI TESTS PASSED PERFECTLY (100%)!
======================================================================
```

---

## Prior Release Notes

### v0.1.2.7: FileSift Cross-Folder File Matcher & Sifter
- High-speed file list and folder matching between two directories.
- Cross-format stem matching (`.jpg` <-> `.ARW`, `.DNG`, `.CR3`).
- Sub-millisecond $O(1)$ set and hash lookup engine.
- Parallel multi-threaded Copy and Cut (Move) to 3rd destination folder.
- Reversible one-click Undo restoration.

### v0.1.2.6: BlurDrop Image Sharpness & Blur Filter
- Hardware-accelerated Laplacian variance, FFT frequency analysis, and Tenengrad edge gradient evaluation.
- Subject-aware depth-of-field mode (Bokeh invariance).
- Real-time dynamic threshold slider reclassification.
- Instant cut & move to curated blur folder with full one-click Undo.
