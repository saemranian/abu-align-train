# Abu Align & Train v0.0.1.3 Walkthrough: Universal Pause/Resume, High-Visibility Progress & Automated Base-10 Versioning

## Executive Summary

**Abu Align & Train v0.0.1.3** introduces core workflow controls, visual feedback upgrades, and robust build engineering:
1. **Universal Pause / Resume Control (`⏸ PAUSE` / `▶ RESUME`):** Allows users to freeze and thaw active photogrammetry alignment and 3DGS training processes at any moment. Using recursive process-tree thread suspension via `psutil`, GPU VRAM allocations and in-memory progress are 100% preserved.
2. **High-Contrast Progress Gauge & Numeric Percentage Badge:** Resolves dark mode visibility by wrapping the progress bar in a bright `#3b82f6` border frame with a `#11111b` deep trough and a dedicated `[ 0.0% ]` to `[ 100.0% ]` percentage readout (`self.pct_lbl`), ensuring instant readability even when idle.
3. **6 Stage Breadcrumb Badges:** Live visual pipeline flow displaying:
   `[1. Scan] ➔ [2. GPU Init] ➔ [3. Preprocess] ➔ [4. 3D Inf] ➔ [5. Tracking & BA] ➔ [6. 3DGS Train]`
   Badges dynamically shift colors: Active (`#2563eb` Royal Blue), Completed (`#065f46` Deep Emerald Green), and Pending (`#232338` Charcoal Slate).
4. **Automated Base-10 Odometer Version Manager (`version_manager.py`):** Enforces strict base-10 digit rules ($0 \to 9$ per digit, cascading rollover on 9, e.g., `v0.0.0.9 -> v0.0.1.0`, `v0.0.1.2 -> v0.0.1.3`). Automatically synchronizes version strings across all GUI, CLI, train engine, launcher, batch scripts, compiles C# executables (`Abu_Align_and_Train_v0.0.1.3.exe`), and maintains Git commit hygiene.

---

## 1. Universal Pause / Resume Architecture

### Problem & Challenge:
Photogrammetry alignment (especially dense bundle adjustment with Google Ceres) and 3D Gaussian Splatting training consume heavy GPU and CPU resources for extended durations. Users needed the ability to temporarily pause execution (e.g. to inspect intermediate output, free up compute for another application, or check system thermals) without losing their progress or freeing CUDA allocations.

### Solution:
A dedicated `⏸ PAUSE` button in the action bar wired to `AbuAlignApp._toggle_pause()`:
- **Pause Action:**
  - Queries child processes recursively via `psutil.Process(worker_process.pid).children(recursive=True)`.
  - Suspends all child worker threads followed by the parent process.
  - Updates button to `▶ RESUME` (Emerald Green `#10b981`) and sets telemetry to:
    `Status: ⏸ PAUSED (GPU VRAM & state preserved in RAM)`
- **Resume Action:**
  - Resumes parent process followed by child workers.
  - Updates button back to `⏸ PAUSE` (Amber `#f59e0b`).
  - Restores active status: `Status: ▶ Resumed and executing...`
- **Stop Safety Guard:**
  - If the user clicks `■ STOP` while paused, the GUI safely thaws all threads before terminating to prevent hung or zombie processes on Windows.

---

## 2. High-Visibility Progress Gauge & Stage Breadcrumbs

### Visibility Resolution:
In standard Tkinter dark themes on Windows, an empty `ttk.Progressbar` (value = 0) renders without any border or fill, causing it to disappear into the dark panel background.
In v0.0.1.3:
- **Progress Gauge Frame:** Enclosed in a styled container (`bg="#11111b"`, `highlightbackground="#3b82f6"`, `highlightthickness=1`, `padx=2`, `pady=2`).
- **Percentage Badge (`self.pct_lbl`):** Bold Segoe UI label permanently visible on the right side of the gauge (`  0.0% ` to `100.0%`).
- **Stage Breadcrumbs:** 6 discrete badges placed above the telemetry row showing pipeline stages:
  1. `1. Scan` (Image discovery & validation)
  2. `2. GPU Init` (CUDA setup & PyTorch foundation model loading)
  3. `3. Preprocess` (Feature extraction & batch inference)
  4. `4. 3D Inf` (Dense unprojection & point map prediction)
  5. `5. Tracking & BA` (Multi-view optical flow & Ceres bundle adjustment)
  6. `6. 3DGS Train` (LichtFeld Studio / Nerfstudio Splatfacto / NVIDIA 3DGRUT)

---

## 3. Automated Base-10 Version Manager (`version_manager.py`)

### Odometer Rollover Specification:
Version strings strictly follow `vMAJOR.MINOR.PATCH.BUILD` where each digit is $0 \le d \le 9$:
$$\begin{aligned}
\text{v0.0.0.9} &\longrightarrow \text{v0.0.1.0} \\
\text{v0.0.1.2} &\longrightarrow \text{v0.0.1.3} \\
\text{v0.0.1.9} &\longrightarrow \text{v0.0.2.0} \\
\text{v0.0.9.9} &\longrightarrow \text{v0.1.0.0} \\
\text{v0.9.9.9} &\longrightarrow \text{v1.0.0.0}
\end{aligned}$$

### Automated Sync Targets:
When incremented (`--bump`) or set (`--set 0.0.1.3`), `version_manager.py` automatically updates:
1. `Abu_Align/VERSION.txt` and `VERSION.txt`
2. `Abu_Align/abu_align_gui.py` and `abu_align_gui.py` (Title, Badge, About dialog, Headers)
3. `Abu_Align/align_cli.py` and `align_cli.py` (Banners, Telemetry)
4. `Abu_Align/train_engine.py` and `train_engine.py`
5. `Abu_Align/launcher.cs`
6. `Abu_Align/Abu_Align_and_Train.bat` and `Abu_Align.bat`
7. Recompiles C# executables with `csc.exe`:
   - `Abu_Align/Abu_Align_and_Train.exe`
   - `Abu_Align/Abu_Align_and_Train_v0.0.1.3.exe`
   - `Abu_Align_and_Train.exe`

---

## 4. Verification & Git Commit Log

### Unit & GUI Verification:
- **Base-10 Rollover Unit Tests:** Passed all carry-over conditions ($0.0.0.9 \to 0.0.1.0$, $0.0.1.2 \to 0.0.1.3$, $0.0.9.9 \to 0.1.0.0$, $0.9.9.9 \to 1.0.0.0$).
- **Widget Initialization Test:** Verified all Tkinter widgets (`pause_btn`, `pct_lbl`, `stage_badges`) instantiate cleanly, stage transitions function properly (`#065f46`, `#2563eb`, `#232338`), and `psutil` process suspension works.
- **Smart Cache Performance:** Verified 8× speedup on `SEL Copy` (04:23 $\to$ 00:33) upon changing parameters.

### Git Version Control Record:
- **Commit:** `2c021d5`
- **Branch:** `main`
- **Message:** `feat(v0.0.1.3): add universal pause/resume, high-contrast progress bar with stage indicators, and automated base-10 version manager`
- **Files Modified/Created:** 17 files (+1,036 insertions, -406 deletions)
