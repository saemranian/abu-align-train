# Abu Align & Train - Mandatory Agent Instructions

## Shared 3D viewer architecture
The shared viewer is **Abu 3D Viewer M54**, shortened to **Abu M54** during development.
Use `abu_m54.py` for scene/file/transform math, `abu_m54_native.py` for the embedded
GPU renderer, `abu_m54_browser.py` for browser navigation/transform controls and
`abu_m54_app.py` for standalone opening. Main and control-point workspaces are host
adapters; do not add another independent orbit, transform or file-parser fork.
Keep source geometry immutable, transform cameras/cloud/controls together, retain
float64 world coordinates and export explicit copies. See `docs/ABU_M54.md`.

## MANDATORY RULE: Base-10 Version Control & Git Commit on Every Modification

### Absolute Requirement
After **EVERY** modification, bugfix, refactor, or feature addition to the Abu Align & Train software:
**YOU MUST EXECUTE THE FOLLOWING 5-STEP PROTOCOL IMMEDIATELY BEFORE RESPONDING TO THE USER. NEVER WAIT FOR OR EXPECT THE USER TO REMIND YOU.**

---

### Step 1: Base-10 Odometer Version Increment
* Run `python version_manager.py --bump`.
* Verify the new version string in `VERSION.txt` (format `vMAJOR.MINOR.PATCH.BUILD`, e.g., `v0.0.2.4`).
* Ensure all GUI, CLI, and engine headers reflect the new version number.
* Recompile the C# launcher executables (`Abu_Align_and_Train.exe` and `Abu_Align_and_Train_vX.Y.Z.W.exe`) via `csc.exe`.

---

### Step 2: 100% Workspace & `Abu_Align/` Parity
* Synchronize all modified files from workspace root to `Abu_Align/`:
  * `train_engine.py`
  * `abu_align_gui.py`
  * `align_cli.py`
  * `video_extractor.py`
  * Any engine files (`360dvo_engine.py`, `opensfm_engine.py`, `panoramic_splitter.py`, `equirectangular_transcoder.py`, etc.)
  * `VERSION.txt`

---

### Step 3: Version Controlling Backup Snapshot
* Create the version snapshot directory: `Abu_Align/versions/vX.Y.Z.W/`.
* Copy all active executables, scripts, engines, and walkthrough documentation into `Abu_Align/versions/vX.Y.Z.W/`.

---

### Step 4: Documentation Update
* Update `walkthrough.md` (in artifacts, workspace root, and `Abu_Align/`) detailing:
  * Exact modifications, root cause analysis, and design rationale.
  * Verification commands and benchmark results.
  * VRAM hygiene and memory metrics.

---

### Step 5: Git Commit & Status
* Stage all modified and new files: `git add -A`.
* Commit with a detailed, professional commit message:
  ```bash
  git commit -m "vX.Y.Z.W: <Summary of changes>

  - Detailed list of modifications
  - Bugfixes and memory optimizations
  - Verification and test results"
  ```
* Verify `git status` is clean.
* In your user response, explicitly report:
  1. The new version number.
  2. The git commit hash and message summary.
  3. The backup snapshot path.
  4. The list of modified files.

---

### Step 6: "Push to GitHub" Automated Release Trigger
Whenever the user instructs **"Push to GitHub"** (or asks to update the GitHub repository / installers):
1. Run `python push_to_github.py "<Concise summary of changes>" --details "<Detailed release notes>"` (or pass `--no-bump` if the version was already bumped in the current turn).
2. This automatically bumps the Base-10 version, synchronizes `Abu_Align/` and `versions/vX.Y.Z.W/`, updates installer metadata, commits, creates git tag `vX.Y.Z.W`, and pushes `HEAD:main` and the tag to `https://github.com/saemranian/abu-align-train.git`.
3. GitHub Actions (`.github/workflows/release.yml`) will automatically build and attach the three single-file installers (`Abu_Align_and_Train_Setup_Windows_x64.exe`, `Abu_Align_and_Train_macOS_Universal.dmg`, `Abu_Align_and_Train_Linux_x86_64.AppImage`) to GitHub Releases.
