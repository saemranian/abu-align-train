# Integration of NVIDIA PyCuSFM (cuSFM) Engine into Abu Align

Integrate **NVIDIA PyCuSFM** (`https://github.com/nvidia-isaac/pyCuSFM`, Isaac cuSFM), a state-of-the-art CUDA-accelerated Structure-from-Motion (SfM) framework, into **Abu Align**. This upgrades Abu Align to a **Quad-Engine Architecture**:
1. **VGGT Foundation (1.18B)** - Global feed-forward attention, unconstrained wide baselines.
2. **VGGT-Omega** - Compressed register token architecture (~60% lower VRAM, fast).
3. **Niantic Labs ACE0** - Scene coordinate regression MLP (ECCV 2024 Oral, ~2GB VRAM).
4. **NVIDIA PyCuSFM (cuSFM)** - GPU-accelerated classical/learned hybrid SfM with ALIKED / SuperPoint feature extraction, LightGlue matching, pose graph optimization, and global bundle adjustment with cuDSS GPU sparse solvers.

---

## User Review Required

> [!IMPORTANT]
> **Runtime Environment: Windows-to-WSL2 Seamless GPU Bridge**
> NVIDIA's `pyCuSFM` compiles and publishes its CUDA C++ binaries (`feature_extractor_main`, `feature_matcher_main`, `pose_graph_main`, `keypoints_mapper_main`, `kpmap_to_colmap`, `bundle_adjustment_runner`) exclusively for **Linux x86_64 (Ubuntu 24.04 LTS)** using CUDA 12/13 and glibc. 
> 
> System verification confirmed:
> - **Host:** Windows 11 x64 with NVIDIA GeForce RTX 4090 (24 GB VRAM, Driver 610.43 / CUDA 13.3).
> - **WSL:** **Ubuntu 24.04.2 LTS** is installed and ready in WSL2 (`wsl -d Ubuntu`).
> - **GPU in WSL:** Direct NVIDIA RTX 4090 GPU passthrough is tested and active (`nvidia-smi` working).
> - **Permissions:** Passwordless root execution (`wsl -u root`) is supported.
> 
> Abu Align will implement a zero-friction **Windows-to-WSL Bridge Engine** (`pycusfm_engine.py`):
> - The user runs Abu Align normally from Windows (CLI or GUI).
> - Abu Align handles path translation (`C:\...` $\leftrightarrow$ `/mnt/c/...`), runs PyCuSFM on the RTX 4090 via WSL2, captures real-time stdout progress, and delivers standard COLMAP, RealityCapture/Postshot CSV, `transforms.json`, and RGB PLY point clouds directly to the Windows output folder.

> [!NOTE]
> **The Coarse-to-Fine "Killer Feature": VGGT + PyCuSFM Hybrid Alignment**
> PyCuSFM documentation states that **monocular and unordered image reconstruction requires initial camera poses in `camera_to_world`** (PyCuSFM does not perform un-posed monocular incremental SfM from scratch without initial poses or sequential stereo).
> 
> Abu Align is uniquely positioned to solve this:
> - **Stage 1 (Coarse Initialization):** Abu Align runs VGGT-Omega (or ACE0) to estimate initial 6DOF camera poses and focal lengths in 5–10 seconds.
> - **Stage 2 (Metadata Conversion):** Converts estimated poses into PyCuSFM `frames_meta.json` format (axis-angle rotations, metric translations, camera matrices).
> - **Stage 3 (High-Precision cuSFM Optimization):** Launches PyCuSFM to extract ALIKED/SuperPoint GPU features, match with LightGlue, build pose graphs, run global bundle adjustment with cuDSS, and colorize the 3D map points with RGB values (`--output_rgb`).
> - **Direct Mode Support:** If sequential stereo images or pre-existing pose files (`poses.csv`) are provided, PyCuSFM can also run directly without Stage 1.

---

## Open Questions

> [!TIP]
> **Default Feature Extractor Preference:**
> PyCuSFM defaults to **ALIKED** (state-of-the-art fast keypoints) with LightGlue matcher, with options for **SuperPoint** and **SIFT**. We propose keeping **ALIKED** as the default with dropdown selection in GUI and CLI flags (`--cusfm_features aliked|superpoint|sift`).

---

## Proposed Changes

Grouped by component and layer:

```
Abu_Align/
├── pycusfm_engine.py                [NEW]    # NVIDIA PyCuSFM adapter & WSL bridge
├── align_cli.py                     [MODIFY] # Add --engine pycusfm and CLI flags
├── abu_align_gui.py                 [MODIFY] # Add 4th engine RadioButton, options, & tooltips
├── fast_viewer.py                   [MODIFY] # Support viewing cuSFM sparse point clouds
├── versions/v0.0.1.1/               [NEW]    # Version snapshot
├── Abu_Align_CLI.exe                [MODIFY] # Recompiled Windows launcher
└── Abu_Align_GUI.exe                [MODIFY] # Recompiled Windows launcher
```

---

### Component 1: PyCuSFM Engine Adapter (`pycusfm_engine.py`)

#### [NEW] [pycusfm_engine.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/pycusfm_engine.py)
A specialized engine adapter for NVIDIA PyCuSFM with the following responsibilities:
1. **WSL2 Environment Inspector & Auto-Installer:**
   - Detects Ubuntu 24.04 distribution in WSL2 (`wsl -l -v`).
   - Verifies / auto-installs required host dependencies (`git-lfs`, `libopencv-dev`, `libgoogle-glog-dev`, `libgflags-dev`, `libabsl-dev`, `libprotobuf-dev`, `protobuf-compiler`).
   - Automatically clones `https://github.com/nvidia-isaac/pyCuSFM` (or uses existing clone), pulls Git LFS binaries, and runs `./setup.bash cuda12`.
2. **Metadata & Trajectory Generator:**
   - If initial poses are provided or computed via Stage 1 (VGGT-Omega):
     - Computes axis-angle representations (`R.from_matrix(...).as_rotvec()`).
     - Extracts camera intrinsic matrix $K = \begin{bmatrix} fx & 0 & cx \\ 0 & fy & cy \\ 0 & 0 & 1 \end{bmatrix}$.
     - Synthesizes compliant `frames_meta.json` adhering to Isaac `KeyframesMetadataCollection` schema.
3. **Execution & Live Progress Streaming:**
   - Invokes `cusfm_cli` via WSL subprocess with real-time stdout line parsing.
   - Monitors pipeline stages:
     - `feature_extractor` $\rightarrow$ `vocab_generator` $\rightarrow$ `pose_graph` $\rightarrow$ `matcher` $\rightarrow$ `mapper` $\rightarrow$ `map_convertor`.
   - Streams stage progress and timing to the Abu Align progress callback.
4. **Reconstruction Import & Universal Handshake:**
   - PyCuSFM outputs COLMAP sparse reconstruction into `<output_dir>/sparse/` (`cameras.txt`, `images.txt`, `points3D.txt` with RGB colors).
   - Loads the result into `pycolmap.Reconstruction`.
   - Extracts dense 3D points `(N, 3)` and RGB colors `(N, 3)`.
   - Returns standard tuple: `(pycolmap.Reconstruction, np.ndarray, np.ndarray)` matching the Abu Align contract.

---

### Component 2: Alignment CLI (`align_cli.py`)

#### [MODIFY] [align_cli.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/align_cli.py)
1. **Engine Argument Expansion:**
   - Add `"pycusfm"` and `"cusfm"` to `--engine` choices:
     `choices=["vggt", "vggt_omega", "ace0", "acezero", "pycusfm", "cusfm"]`.
   - Add PyCuSFM options:
     - `--cusfm_features`: Feature extractor (`aliked`, `superpoint`, `sift`, default: `aliked`).
     - `--cusfm_init`: Initial pose provider (`omega`, `vggt`, `ace0`, `none`, default: `omega`).
     - `--cusfm_output_rgb`: Enable RGB color sampling for sparse 3D points (default: `True`).
     - `--cusfm_ba_iterations`: Maximum bundle adjustment iterations (default: `100`).
2. **Directory Naming:**
   - Update `get_next_sparse_dir`: generates `sparse_01_cusfm`, `sparse_02_cusfm`, etc.
3. **Engine Dispatching:**
   - Branch for `engine in ["pycusfm", "cusfm"]`:
     - Run coarse initialization if needed.
     - Call `run_pycusfm_reconstruction(...)` from `pycusfm_engine.py`.
     - Forward reconstructed model to standard export targets:
       - `COLMAP_Lichtfield/`
       - `Postshot_RealityCapture/` (`cameras.csv`, `transforms.json`, `points.ply`)
4. **Banner & Logging:**
   - Update banner to: `Abu Align v0.0.1.1 - Neural & GPU Photogrammetry Engine`.
   - List all 4 engines in banner and output summaries.

---

### Component 3: GUI Application (`abu_align_gui.py`)

#### [MODIFY] [abu_align_gui.py](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/abu_align_gui.py)
1. **Engine RadioButton:**
   - Add 4th radio button in `Engine & Canvas`:
     `rb_cusfm = tk.Radiobutton(col1, text="NVIDIA PyCuSFM (GPU SfM + ALIKED)", variable=self.engine_var, value="pycusfm", ...)`
2. **Interactive Tooltip:**
   - Tooltip: `NVIDIA PyCuSFM (cuSFM): High-throughput CUDA Structure-from-Motion framework by NVIDIA Isaac. Combines learned ALIKED/SuperPoint keypoints, LightGlue GPU matching, and global bundle adjustment with cuDSS sparse GPU solver.`
3. **Engine Settings:**
   - Add PyCuSFM settings tab or parameters (Feature Type: ALIKED / SuperPoint / SIFT; Coarse Initializer: Auto-Omega / None).
4. **Log & About Updates:**
   - Update startup banner to describe Quad-Engine Architecture.
   - Update About dialog to version `v0.0.1.1`.

---

### Component 4: Version Snapshot & Standalone Executables

#### [NEW] [Abu_Align/versions/v0.0.1.1/](file:///c:/Users/USer/.gemini/antigravity/scratch/General%20Stuff%2001/vggt/Abu_Align/versions/v0.0.1.1/)
- Full versioned backup of `align_cli.py`, `abu_align_gui.py`, `pycusfm_engine.py`, `acezero_engine.py`, and docs.
- Recompile `Abu_Align_CLI.exe` and `Abu_Align_GUI.exe` with C# compiler (`csc.exe`) targeting the new version.

---

## Verification Plan

### Automated Verification
1. **WSL2 Bridge Verification:**
   - Test `wsl -d Ubuntu` communication, CUDA availability, and Python bridge execution.
2. **Adapter Unit Test (`test_pycusfm_adapter.py`):**
   - Synthesize a minimal 4-view test scene with synthetic poses.
   - Test `generate_frame_meta` JSON serialization (axis-angle and camera matrix format).
   - Test COLMAP model import and point cloud conversion.
3. **End-to-End CLI Dry Run:**
   - Run `python Abu_Align/align_cli.py --engine pycusfm --help` to verify argument parsing and banner.
   - Run a test alignment on sample dataset to verify output generation in `COLMAP_Lichtfield/` and `Postshot_RealityCapture/`.

### Manual Verification
1. Launch `Abu_Align_GUI.exe` and verify:
   - "NVIDIA PyCuSFM (GPU SfM + ALIKED)" radio button is selectable.
   - Tooltip displays correctly.
   - Real-time logging displays PyCuSFM stages cleanly.
   - Outputs load properly into `fast_viewer.py` and Postshot/COLMAP.
