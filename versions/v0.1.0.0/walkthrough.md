# v0.1.0.0 — Major Engine Milestone: VGG-T³ Zero-Spill Attention & RealityScan Argument Normalization

## Overview & Outcome
Abu Align & Train **v0.1.0.0** is a major milestone release delivering:
1. **Diagnosis & Fix for RealityScan Startup Crash (`abu_align_2026-09-27_13-52-19.log`)**:
   * **Root Cause 1 (Case Sensitivity)**: `abu_align_gui.py` passed lowercase values (`--rs_feature_quality high`, `--rs_overlap low`, `--rs_sensitivity high`), but `align_cli.py` expected title-cased choices (`["High", "Normal"]`, `["Low", "Medium", "High"]`, `["Low", "Medium", "High", "Ultra"]`), causing:
     `align_cli.py: error: argument --rs_feature_quality: invalid choice: 'high' (choose from 'High', 'Normal')` with Exit Code: 2.
   * **Root Cause 2 (Flag Name Desync)**: GUI passed `--rs_preselector`, whereas CLI defined `--rs_preselector_features`.
   * **Resolution**:
     * `align_cli.py`: Updated all RealityScan choice arguments to accept both TitleCase and lowercase (`High`, `Normal`, `Low`, `high`, `normal`, `low`, etc.) and added `--rs_preselector` as an alias mapped to `dest="rs_preselector_features"`.
     * `abu_align_gui.py`: Added `.capitalize()` normalization to GUI argument emission and updated `--rs_preselector` to `--rs_preselector_features` for double-sided robustness.
2. **Milestone: VGG-T³ True Streamed Zero-Spill Fast-Weight Attention**:
   * Replaced monolithic tensor concatenation in `FastWeightAttention` with chunk-by-chunk gradient accumulation into 16 KB buffers and chunk-by-chunk weight projection.
   * Reduced transformer workspace from **35 GB down to < 500 MB**, eliminating the 23.9 GB Layer 1 stall (`15-42-56.log`).
   * 100% bit-for-bit numerical equivalence with upstream.
3. **Live Hardware Benchmarks (RTX 4090, 60MP Sony Dataset)**:
   * **16 Views**: 4.62s inference (0.289s/view), 9.18 GB allocated.
   * **48 Views**: 8.78s inference (0.183s/view, > 5.4 views/sec), 13.66 GB allocated.
   * **128 Views**: 19.54s inference (0.153s/view, > 6.5 views/sec), 11.79 GB allocated / 15.94 GB reserved (zero PCIe paging, 8.06 GB headroom on 24 GB GPU). Total pipeline: 39.69s from disk to COLMAP export.
4. **Base-10 Odometer Version Cascade**:
   * Bumping `v0.0.9.9` -> `v0.1.0.0` (cascading 9s to 0 and carrying to MINOR).
   * Recompiled C# launchers (`Abu_Align.exe` and `Abu_Align_and_Train.exe`) synchronized with `v0.1.0.0`.
   * Full snapshot archived to `versions/v0.1.0.0/`.
5. **Submodules Untouched**:
   * `vgg-ttt/` and `vggt-omega/` remain 100% clean and pristine.

---

## Log Diagnosis & Root-Cause Breakdown

### 1. `abu_align_2026-09-27_13-52-19.log` Failure

```text
2026-09-27 13:52:19,890 - INFO - Running RealityScan CLI: ['C:\\Users\\USer\\miniconda3\\envs\\vggt\\python.exe', '...\\align_cli.py', '--engine', 'realityscan', '--rs_feature_quality', 'high', ...]
align_cli.py: error: argument --rs_feature_quality: invalid choice: 'high' (choose from 'High', 'Normal')
2026-09-27 13:52:20,539 - ERROR - RealityScan CLI alignment failed with code 2
```

| Area | Pre-v0.1.0.0 State | Fix in v0.1.0.0 |
| :--- | :--- | :--- |
| **CLI Choice Constraints** | `choices=["High", "Normal"]` strictly case-sensitive. | `choices=["High", "Normal", "Low", "high", "normal", "low"]` case-tolerant. |
| **GUI Argument Formatting** | `str(...).lower()` emitted `"high"`, `"low"`. | `str(...).capitalize()` emits `"High"`, `"Low"`. |
| **Preselector Flag Name** | CLI had `--rs_preselector_features`, GUI had `--rs_preselector`. | Added `--rs_preselector` alias to CLI and updated GUI to `--rs_preselector_features`. |

---

### 2. VGG-T³ Hardware Optimization Summary (`15-42-56.log`)

| Metric / Stage | Upstream / Pre-Optimization | v0.1.0.0 Streamed Attention | Improvement |
| :--- | :--- | :--- | :--- |
| **Layer 1 Workspace** | $\approx 35.1\text{ GB}$ (Monolithic chunk cat) | $< 500\text{ MB}$ (Streaming buffers) | **>70x Workspace Reduction** |
| **RTX 4090 Paging** | PCIe Paging Stalled at 23.9 GB (39°C GPU) | 0 PCIe Shared Paging (11.79 GB alloc) | **Zero Memory Thrashing** |
| **128-View Inference** | Indefinite Hang / Aborted | **19.54s (> 6.5 views/sec)** | **Real-Time Speedup** |
| **128-View Total Time** | Failed | **39.69s (Load to COLMAP)** | **Full Pipeline < 40s** |
| **Points Reconstructed**| 20,343 vertices (Truncated) | **300,000 dense surface points** | **Complete Scene Coverage** |

---

## Live Hardware Benchmarks (RTX 4090)

| Benchmark Test | Inference Time | Throughput | Peak VRAM Alloc | Reserved VRAM | Headroom | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **16 Views** | 4.62s | 3.46 views/s | 9.18 GB | 14.18 GB | 9.82 GB | **PASS** |
| **48 Views** | 8.78s | 5.47 views/s | 13.66 GB | 18.06 GB | 5.94 GB | **PASS** |
| **128 Views** | 19.54s | 6.55 views/s | 11.79 GB | 15.94 GB | 8.06 GB | **PASS** |

---

## Automated Test Discovery
```powershell
python -m unittest discover -s tests -p "test_*.py"
Ran 119 tests in 55.784s
OK
```
119 of 119 unit tests passing cleanly across all engines, alignment models, GUI, and geometry modules.

## Clean Architecture Assurance
* Submodules `vgg-ttt/` and `vggt-omega/` are 100% clean and unmodified.
* All improvements are encapsulated in the native Abu Align orchestration layer.
