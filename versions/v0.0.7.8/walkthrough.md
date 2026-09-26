# Engine Repository Version Tracker & VGGT-Omega Upstream Verification Walkthrough (v0.0.7.7)

## Overview & Objectives

In response to the user's request:
1. **VGGT-Omega Upstream Inspection & Sync**:
   Checked the official upstream repository (`facebookresearch/vggt-omega`). Synchronized local repository (`..\vggt-omega`) to commit `48b23c8` (100% up to date with latest GitHub `origin/main`).
2. **Automated Multi-Engine GitHub Version Tracker (`engine_tracker.py`)**:
   Engineered an automated version tracking module covering all 10 photogrammetry, Structure-from-Motion, and 3D Gaussian Splatting engines in Abu Align & Train.
3. **Background Auto-Check & UI Update Badge**:
   Integrated non-blocking background query on application launch when internet connectivity is available. Displays a dynamic badge next to the version label in the UI (`🔔 Update Available (N)`, `✓ Engines Up to Date`, or `⚡ Offline Mode`). Clicking the badge opens a comprehensive interactive dialog.
4. **Interactive Administrator Report Dialog**:
   Full modal dialog with dark styling matching Abu Align:
   - Live color-coded Treeview table of all 10 engines.
   - Selected engine inspection card (local path, commit message, upstream HEAD, commit link).
   - "🌐 View GitHub Repository" and "📁 Open Local Directory" buttons.
   - "🔄 Refresh Now" (runs concurrent background check) and "📋 Copy Diagnostic Report to Clipboard" (enables easy copy-pasting for requesting updates).
5. **CLI Integration**:
   Added `--check_updates` / `--check_engines` flag to `align_cli.py` to print a structured diagnostic table directly to the console.
6. **Version Bump & Executable Compilation**:
   Bumped from `v0.0.7.6` to `v0.0.7.7` via `version_manager.py`, recompiled standalone launchers `Abu_Align.exe` and `Abu_Align_and_Train.exe`, and created archive snapshot `versions/v0.0.7.7/`.

---

## 1. VGGT-Omega Upstream Inspection & Verification

- **Repository**: `https://github.com/facebookresearch/vggt-omega.git`
- **Initial Local Commit**: `a3ab014`
- **Remote `origin/main` Commit**: `48b23c8`
- **Diff Analysis (`git diff a3ab014..48b23c8`)**:
  - The core `vggt_omega` model code, coordinate transformers, and forward pipeline are completely unchanged.
  - The upstream commit updated `eval/model.py` to change the default eval resolution to `512` (which Abu Align already leverages by default).
- **Action Taken**:
  Executed `git -C "..\vggt-omega" pull --ff-only` to bring the local repository to `48b23c8`.
  Abu Align's VGGT-Omega engine is now **100% up to date** with upstream GitHub HEAD.

---

## 2. All 10 Tracked Engines

The tracker engine monitors the following 10 photogrammetry and 3DGS engines:

| Engine | Category | Role | Upstream Repository / Integration |
| :--- | :--- | :--- | :--- |
| **VGGT-Omega** | Alignment Engine | Neural Foundation Transformer | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) |
| **Niantic Labs ACE0** | Alignment Engine | Scene Coordinate Regression & 25k Refit SfM | [nianticlabs/acezero](https://github.com/nianticlabs/acezero) |
| **NVIDIA 3DGRUT** | 3DGS Training Engine | Ray Traced Gaussian Splatting & BVH | [nv-tlabs/3dgrut](https://github.com/nv-tlabs/3dgrut) |
| **PFGS360** | 3DGS Training Engine | CVPR 2026 Pose-Free Gaussian Splatting | [zcq15/PFGS360](https://github.com/zcq15/PFGS360) |
| **NVIDIA Isaac pyCuSFM** | Alignment Engine | GPU Structure-from-Motion | [nvidia-isaac/pyCuSFM](https://github.com/nvidia-isaac/pyCuSFM) |
| **Mapillary OpenSfM** | Alignment Engine | Native 360° Spherical & $S^2$ Bundle Adjustment | [mapillary/OpenSfM](https://github.com/mapillary/OpenSfM) |
| **360DVO** | Alignment Engine | Deep Omnidirectional Visual Odometry | Native DAS-Feat & ODBA (IEEE RA-L 2026) |
| **LichtFeld Studio** | 3DGS Training Engine | Real-time Vulkan 3DGS GUI + GUT Optics | [MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) |
| **Nerfstudio** | 3DGS Training Engine | Splatfacto Optimization Pipeline | [nerfstudio-project/nerfstudio](https://github.com/nerfstudio-project/nerfstudio) |
| **VGGT (Original)** | Foundation Model | Original Vision Geometry Transformer | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) |

---

## 3. Engine Tracker Architecture (`engine_tracker.py`)

- **Concurrent Multi-Threading**: Uses `ThreadPoolExecutor` to evaluate all 10 engines concurrently within 2–3 seconds.
- **Dual Query Strategy (API + ls-remote)**:
  1. Primary: Fast GitHub REST API (`api.github.com/repos/.../commits/HEAD`) providing commit SHA, ISO timestamp, and commit message.
  2. Fallback: `git ls-remote <url> HEAD` if the unauthenticated GitHub API rate limit is reached.
- **Offline Safety**: Performs initial socket connectivity test to `github.com:443` with a 2-second timeout; if offline, displays local versions without stalling or throwing exceptions.
- **Windows CP1252 Encoding Protection**: Formats output using clean ASCII tags (`[OK]`, `[UPDATE]`, `[OFFLINE]`) to prevent `UnicodeEncodeError` in standard Windows cmd/PowerShell consoles.

---

## 4. UI & CLI Features

### A. Desktop UI Header Badge
In [desktop_ui.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/desktop_ui.py#L430-L445), an interactive update monitor button is positioned directly next to the version label `v0.0.7.7`:
- On launch: `⏳ Checking engines...`
- When an update exists: `🔔 Update Available (1)` in bold amber (`#d97706`) with white text.
- When up to date: `✓ Engines Up to Date` in emerald green (`#10b981`).
- When offline: `⚡ Offline (Engines)` in muted slate (`#a0adb5`).

### B. Interactive Administrator Report Dialog
In [abu_align_gui.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/abu%20align/abu_align_gui.py#L4610-L4840), clicking the badge (or selecting `Tools -> Check Engine Repository Updates...` / `Engines -> Check GitHub Updates...`) opens a modal dialog:
- Displays summary status (Internet connection, last checked time, update count).
- Full table showing Engine Name, Category, Local Applied Version, GitHub Upstream (HEAD), and Status.
- Clicking any row loads detailed local and remote metadata into the detail card.
- One-click buttons to open the GitHub repository or the local engine folder.
- `🔄 Refresh Now` button with live background checking.
- `📋 Copy Diagnostic Report` button to copy the formatted text to the Windows clipboard.

### C. Menus
- **Tools Menu**: Added `🔄 Check Engine Repository Updates...`.
- **Engines Menu**: Added dedicated menu cascade with update check and direct links to upstream repositories for all 10 engines.

### D. CLI Diagnostic Report (`--check_updates`)
Command:
```bash
python align_cli.py --check_updates
```
Example Output:
```
Checking upstream GitHub repositories for Abu Align engines...

==================================================================================
 ABU ALIGN & TRAIN - ENGINE REPOSITORY VERSION REPORT
==================================================================================
 Check Time:          2026-09-26 12:34:54
 Internet Status:     CONNECTED
 Updates Available:   1 of 10 engines
----------------------------------------------------------------------------------
ENGINE             | CATEGORY             | APPLIED VERSION  | STATUS              
----------------------------------------------------------------------------------
VGGT-Omega         | Alignment Engine     | 48b23c8          | [OK] Up to Date     
Niantic Labs ACE0  | Alignment Engine     | 821762f          | [OK] Custom Commits 
NVIDIA 3DGRUT      | 3DGS Training Engine | 7397cc9          | [UPDATE] Available  
PFGS360            | 3DGS Training Engine | e79413f          | [OK] Up to Date     
NVIDIA Isaac pyCuSFM | Alignment Engine     | v0.1.3 (0c97a..  | [OK] v0.1.3 (0c97a67
Mapillary OpenSfM  | Alignment Engine     | v0.5.2 / Nati..  | [OK] v0.5.2 / Native
360DVO             | Alignment Engine     | Native DAS-Fe..  | [OK] Native Engine  
LichtFeld Studio   | 3DGS Training Engine | Integrated Li..  | [OK] Integrated Lich
Nerfstudio         | 3DGS Training Engine | Conda Environ..  | [OK] Conda Environme
VGGT (Original)    | Foundation Model     | e304eae          | [OK] Custom Commits 
----------------------------------------------------------------------------------

==================================================================================
 [!] AVAILABLE ENGINE UPDATES ON GITHUB:
==================================================================================

>>> NVIDIA 3DGRUT (3DGS Training Engine)
    * Role:            Differentiable Ray Tracing & Accelerated Gaussian Splatting
    * Repository:      https://github.com/nv-tlabs/3dgrut.git
    * Current Local:   7397cc9 (2026-09-17) - fix(playground): restore build after particleKernelMaxAlpha change (#267)
    * Latest Remote:   e6f4552 (Latest) - Latest commit on GitHub HEAD
    * Commit URL:      https://github.com/nv-tlabs/3dgrut/commit/e6f4552b408af020a96de09ab5757f9b57430aaa
    * Action Required: Request agent or run git pull to update this engine.
==================================================================================
```

---

## 5. Verification Results

| Test / Command | Result | Notes |
| :--- | :--- | :--- |
| `python -m unittest tests/test_engine_tracker.py` | **PASSED** (5 tests in 0.233s) | Verifies registry integrity, git SHA extraction, offline fallback, and report formatting |
| `python -m unittest discover -s tests -p "test_desktop_ui.py"` | **PASSED** (13 tests in 13.1s) | Verifies GUI widgets, theme rendering, and engine update badge |
| `python -m unittest discover tests` | **PASSED** (61 tests in 36.0s) | Entire test suite clean with zero regressions |
| `python align_cli.py --check_updates` | **PASSED** (Code 0) | Correctly detected NVIDIA 3DGRUT update `e6f4552` and confirmed VGGT-Omega `48b23c8` |
| `python version_manager.py --bump` | **PASSED** | Bumped version to `v0.0.7.7`, recompiled C# `.exe` launchers, and generated snapshot |

---

## 6. Files Modified & Created

- **`engine_tracker.py`** [NEW]: Multi-threaded GitHub repository version comparison and reporting engine.
- **`tests/test_engine_tracker.py`** [NEW]: Unit tests for repository checking, git extraction, and report formatting.
- **`desktop_ui.py`** [MODIFIED]: Added `app.engine_update_btn` in header next to version label.
- **`abu_align_gui.py`** [MODIFIED]: Added background auto-check thread on launch, badge updater, interactive report dialog, and Engines menu.
- **`align_cli.py`** [MODIFIED]: Added `--check_updates` / `--check_engines` flag and updated banner.
- **`version_manager.py`** [MODIFIED]: Added `engine_tracker.py` to snapshot archiving list.
- **`VERSION.txt`** [MODIFIED]: Updated to `0.0.7.7`.
- **`Abu_Align.exe` & `Abu_Align_and_Train.exe`** [RECOMPILED]: Recompiled standalone C# launcher executables.
- **`versions/v0.0.7.7/`** [ARCHIVED]: Complete code and binary snapshot.
