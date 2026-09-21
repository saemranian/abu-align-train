"""
Abu Align & Train - 3D Gaussian Splatting Training Engine
==========================================================
Integrates 3 premier 3DGS training engines:
1. LichtFeld Studio (C++ / Vulkan Real-Time Viewport & Headless with GUT)
2. Nerfstudio Splatfacto (State-of-the-Art PyTorch 3DGS + PLY exporter)
3. NVIDIA 3DGRUT / 3DGUT (Gaussian Ray Tracing & Unscented Transform)

Provides unified execution, real-time telemetry streaming, and automated
3D Gaussian Splat (.ply, .spz, .splat) model generation.
"""

import os
import sys
import time
import json
import shutil
import subprocess
import glob
import re
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List


# ----------------------------------------------------------------------
# Paths & Environment Detectors
# ----------------------------------------------------------------------
LICHTFELD_CANDIDATES = [
    r"C:\Users\USer\Ahmad_LichtFeld\LichtFeld-Studio\build\LichtFeld-Studio.exe",
    r"C:\Program Files\LichtFeld Studio\LichtFeld-Studio.exe",
]

NERFSTUDIO_NS_TRAIN_CANDIDATES = [
    r"C:\Users\USer\miniconda3\envs\gs_clean\Scripts\ns-train.exe",
    r"C:\Users\USer\miniconda3\envs\nerfstudio\Scripts\ns-train.exe",
]

NERFSTUDIO_NS_EXPORT_CANDIDATES = [
    r"C:\Users\USer\miniconda3\envs\gs_clean\Scripts\ns-export.exe",
    r"C:\Users\USer\miniconda3\envs\nerfstudio\Scripts\ns-export.exe",
]

THREEDGRUT_REPO_CANDIDATES = [
    r"C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\3dgrut",
    r"C:\Users\USer\3dgrut",
]


def get_lichtfeld_executable() -> Optional[str]:
    for p in LICHTFELD_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def get_nerfstudio_train_executable() -> Optional[str]:
    for p in NERFSTUDIO_NS_TRAIN_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def get_nerfstudio_export_executable() -> Optional[str]:
    for p in NERFSTUDIO_NS_EXPORT_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def get_3dgrut_repo() -> Optional[str]:
    for p in THREEDGRUT_REPO_CANDIDATES:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.py")):
            return p
    return None


# ----------------------------------------------------------------------
# 1. LichtFeld Studio Trainer
# ----------------------------------------------------------------------
def train_lichtfeld(
    dataset_dir: str,
    output_dir: str,
    iterations: int = 30000,
    headless: bool = False,
    gut: bool = False,
    sh_degree: int = 3,
    bilateral_grid: bool = False,
    exposure_correction: bool = False,
    enable_sparsity: bool = False,
    enable_mip: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes 3DGS training using LichtFeld Studio.
    
    :param dataset_dir: Directory containing 3dgs_dataset (sparse/0/ and images/)
    :param output_dir: Destination directory for trained model
    :param iterations: Max training iterations
    :param headless: Run headless in background vs live interactive Vulkan window
    :param gut: Enable Gaussian Unscented Transform
    :param sh_degree: Spherical harmonics degree
    :param bilateral_grid: Enable bilateral grid color correction
    :param exposure_correction: Enable per-photo auto-exposure correction
    :param enable_sparsity: Enable ADMM sparsity pruning
    :param enable_mip: Enable anti-aliased mip-filtering
    :param progress_callback: Telemetry callback
    :return: Path to trained .ply file
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    lfs_exe = get_lichtfeld_executable()
    if not lfs_exe:
        raise FileNotFoundError(
            f"LichtFeld Studio executable not found. Checked: {LICHTFELD_CANDIDATES}"
        )

    os.makedirs(output_dir, exist_ok=True)
    log(f"      * Initializing LichtFeld Studio Training Engine...")
    log(f"      * Executable: {lfs_exe}")
    log(f"      * Dataset:    {dataset_dir}")
    log(f"      * Output Dir: {output_dir}")
    log(f"      * Iterations: {iterations:,}")
    log(f"      * GUT Optics: {'Enabled (Gaussian Unscented Transform)' if gut else 'Disabled'}")
    log(f"      * Bilateral:  {'Enabled' if bilateral_grid else 'Disabled'}")
    log(f"      * Exposure:   {'Enabled' if exposure_correction else 'Disabled'}")
    log(f"      * Sparsity:   {'Enabled' if enable_sparsity else 'Disabled'}")
    log(f"      * Mip-Filter: {'Enabled' if enable_mip else 'Disabled'}")
    log(f"      * Mode:       {'Headless Background' if headless else 'Live Vulkan Viewport'}")

    cmd = [
        lfs_exe,
        "-d", os.path.abspath(dataset_dir),
        "-o", os.path.abspath(output_dir),
        "-i", str(iterations),
        "--export", "ply",
    ]

    if headless:
        cmd.append("--headless")
    else:
        cmd.append("--train")
    if gut:
        cmd.append("--gut")
    if bilateral_grid:
        cmd.append("--bilateral-grid")
    if exposure_correction:
        cmd.append("--exposure-correction")
    if enable_sparsity:
        cmd.append("--enable-sparsity")
    if enable_mip:
        cmd.append("--enable-mip")
    if sh_degree != 3:
        cmd.extend(["--sh-degree", str(sh_degree)])

    log(f"      * Launching command: {' '.join(cmd)}")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    last_report_time = time.time()
    for line in proc.stdout:
        line_s = line.strip()
        if not line_s:
            continue

        # Check for iteration progress lines
        # Example: [1000/30000] loss: 0.0123 psnr: 28.4dB points: 154,200
        lower = line_s.lower()
        if any(keyword in lower for keyword in ["iter", "step", "loss", "psnr", "point", "training", "export"]):
            now = time.time()
            if now - last_report_time >= 0.25 or "export" in lower or "done" in lower or "finish" in lower:
                log(f"        [LichtFeld] {line_s}")
                last_report_time = now

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"LichtFeld Studio exited with return code {proc.returncode}")

    # Locate output PLY file
    ply_candidates = glob.glob(os.path.join(output_dir, "*.ply")) + \
                     glob.glob(os.path.join(output_dir, "**", "*.ply"), recursive=True)
    if ply_candidates:
        best_ply = max(ply_candidates, key=os.path.getmtime)
        log(f"      * [LichtFeld] Training complete! Exported 3DGS model: {best_ply} ({time.time()-start_time:.1f}s)")
        return best_ply

    log(f"      * [LichtFeld] Training finished in {time.time()-start_time:.1f}s.")
    return output_dir


# ----------------------------------------------------------------------
# 2. Nerfstudio Splatfacto Trainer
# ----------------------------------------------------------------------
def train_nerfstudio_splatfacto(
    dataset_dir: str,
    output_dir: str,
    iterations: int = 30000,
    model_type: str = "splatfacto",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes 3DGS training using Nerfstudio Splatfacto.
    
    :param dataset_dir: Directory containing 3dgs_dataset (sparse/0/ and images/)
    :param output_dir: Destination directory for trained model
    :param iterations: Max training iterations (e.g. 30,000)
    :param model_type: 'splatfacto' or 'splatfacto-big'
    :param progress_callback: Telemetry callback
    :return: Path to exported Gaussian splat .ply file
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    ns_train_exe = get_nerfstudio_train_executable()
    ns_export_exe = get_nerfstudio_export_executable()
    if not ns_train_exe:
        raise FileNotFoundError(
            f"Nerfstudio ns-train executable not found. Checked: {NERFSTUDIO_NS_TRAIN_CANDIDATES}"
        )

    os.makedirs(output_dir, exist_ok=True)
    log(f"      * Initializing Nerfstudio Splatfacto Engine...")
    log(f"      * ns-train:   {ns_train_exe}")
    log(f"      * Model:      {model_type.upper()}")
    log(f"      * Dataset:    {dataset_dir}")
    log(f"      * Output Dir: {output_dir}")
    log(f"      * Max Steps:  {iterations:,}")

    # Ensure transforms.json or colmap format is matched
    dataset_abs = os.path.abspath(dataset_dir)
    transforms_json = os.path.join(dataset_abs, "transforms.json")
    colmap_sparse = os.path.join(dataset_abs, "sparse", "0")

    # Command construction
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    scripts_dir = os.path.dirname(ns_train_exe)
    env["PATH"] = f"{scripts_dir};{os.path.dirname(scripts_dir)};{env.get('PATH', '')}"

    # Sanitize transforms.json if present for mixed image orientations
    if os.path.isfile(transforms_json):
        try:
            with open(transforms_json, "r", encoding="utf-8") as f:
                tf_data = json.load(f)
            frames = tf_data.get("frames", [])
            if frames:
                frame_dims = {(f.get("w"), f.get("h")) for f in frames if "w" in f and "h" in f}
                if len(frame_dims) > 1:
                    # Mixed resolutions: ensure root w/h/cx/cy are removed so per-frame settings are used
                    for key in ["w", "h", "fl_x", "fl_y", "cx", "cy"]:
                        tf_data.pop(key, None)
                    with open(transforms_json, "w", encoding="utf-8") as f:
                        json.dump(tf_data, f, indent=2)
                    log(f"        [Nerfstudio] Detected mixed orientations ({len(frame_dims)} dimensions) - calibrated transforms.json to per-frame mode")
        except Exception as e:
            log(f"        [Nerfstudio Warning] Failed to inspect transforms.json: {e}")

    cmd = [
        ns_train_exe,
        model_type,
        "--data", dataset_abs,
        "--output-dir", os.path.abspath(output_dir),
        f"--max-num-iterations={iterations}",
        "--vis", "tensorboard",
        "--logging.steps-per-log=50",
    ]

    log(f"      * Launching Nerfstudio Training: {' '.join(cmd)}")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        encoding="utf-8",
        errors="replace"
    )

    config_path = None
    step_pattern = re.compile(r"(\d+)\s*/\s*(\d+)")
    loss_pattern = re.compile(r"loss\s*[:=]\s*([0-9\.]+)", re.IGNORECASE)

    last_report_time = time.time()
    for line in proc.stdout:
        line_s = line.strip()
        if not line_s:
            continue

        if "Saving config to:" in line_s or "config.yml" in line_s:
            parts = line_s.split()
            for p in parts:
                if p.endswith(".yml") or p.endswith(".yaml"):
                    config_path = p.strip()
                    log(f"        [Nerfstudio] Registered experiment config: {config_path}")

        # Stream training steps periodically
        if any(k in line_s for k in ["Step", "step", "loss", "Saving", "Printing"]):
            now = time.time()
            if now - last_report_time >= 0.5 or "Saving" in line_s:
                log(f"        [Nerfstudio] {line_s}")
                last_report_time = now

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Nerfstudio ns-train exited with code {proc.returncode}")

    log(f"      * [Nerfstudio] Optimization finished in {time.time()-start_time:.1f}s.")

    # Locate config.yml if not parsed from output
    if not config_path or not os.path.isfile(config_path):
        yml_candidates = glob.glob(os.path.join(output_dir, "**", "config.yml"), recursive=True)
        if yml_candidates:
            config_path = max(yml_candidates, key=os.path.getmtime)

    if not config_path or not os.path.isfile(config_path):
        log("      [Warning] Could not locate config.yml to export .ply model.")
        return output_dir

    # Export to standalone Gaussian Splatting .ply
    if ns_export_exe and os.path.isfile(ns_export_exe):
        log("      * [Nerfstudio] Exporting high-fidelity Gaussian Splatting .ply via ns-export...")
        export_out = os.path.join(output_dir, "export")
        os.makedirs(export_out, exist_ok=True)
        export_cmd = [
            ns_export_exe,
            "gaussian-splat",
            "--load-config", os.path.abspath(config_path),
            "--output-dir", os.path.abspath(export_out),
        ]
        log(f"      * Running export: {' '.join(export_cmd)}")
        res = subprocess.run(export_cmd, capture_output=True, text=True, env=env)
        if res.returncode == 0:
            exported_plys = glob.glob(os.path.join(export_out, "*.ply"))
            if exported_plys:
                best_ply = max(exported_plys, key=os.path.getmtime)
                log(f"      * [Nerfstudio] Exported standalone 3DGS PLY: {best_ply}")
                return best_ply
        else:
            log(f"      [Warning] ns-export returned code {res.returncode}: {res.stderr}")

    return output_dir


# ----------------------------------------------------------------------
# 3. NVIDIA 3DGRUT / 3DGUT Trainer
# ----------------------------------------------------------------------
def train_3dgrut(
    dataset_dir: str,
    output_dir: str,
    variant: str = "3dgut",
    iterations: int = 30000,
    downsample_factor: int = 1,
    with_viser_gui: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes 3DGS training using NVIDIA 3DGRUT / 3DGUT.
    
    :param dataset_dir: Directory containing 3dgs_dataset (sparse/0/ and images/)
    :param output_dir: Destination directory for trained model
    :param variant: '3dgut' (Gaussian Unscented Transform) or '3dgrt' (Gaussian Ray Tracing)
    :param iterations: Max iterations
    :param downsample_factor: Image downsample factor (1, 2, 4)
    :param with_viser_gui: Launch interactive browser GUI
    :param progress_callback: Telemetry callback
    :return: Output directory path
    """
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    repo_dir = get_3dgrut_repo()
    if not repo_dir:
        raise FileNotFoundError(
            f"NVIDIA 3DGRUT repository not found. Checked: {THREEDGRUT_REPO_CANDIDATES}"
        )

    os.makedirs(output_dir, exist_ok=True)
    config_name = "apps/colmap_3dgut.yaml" if variant == "3dgut" else "apps/colmap_3dgrt.yaml"
    log(f"      * Initializing NVIDIA 3DGRUT / 3DGUT Engine...")
    log(f"      * Repository: {repo_dir}")
    log(f"      * Config:     {config_name}")
    log(f"      * Dataset:    {dataset_dir}")
    log(f"      * Output Dir: {output_dir}")

    # Use python environment with torch and hydra
    python_exe = sys.executable
    cmd = [
        python_exe,
        "train.py",
        f"--config-name={config_name}",
        f"path={os.path.abspath(dataset_dir)}",
        f"out_dir={os.path.abspath(output_dir)}",
        f"experiment_name=run_{variant}",
        f"dataset.downsample_factor={downsample_factor}",
        f"with_viser_gui={str(with_viser_gui)}",
    ]

    log(f"      * Launching command: {' '.join(cmd)}")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace"
    )

    for line in proc.stdout:
        line_s = line.strip()
        if not line_s:
            continue
        if any(k in line_s.lower() for k in ["iter", "step", "loss", "psnr", "viser", "http"]):
            log(f"        [3DGRUT] {line_s}")

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"3DGRUT exited with code {proc.returncode}")

    log(f"      * [3DGRUT] Training complete in {time.time()-start_time:.1f}s.")
    return output_dir


# ----------------------------------------------------------------------
# 4. Master 3DGS Training Dispatcher
# ----------------------------------------------------------------------
def run_3dgs_training(
    engine: str,
    dataset_dir: str,
    output_dir: str,
    iterations: int = 30000,
    mode: str = "interactive",
    gut: bool = False,
    sh_degree: int = 3,
    bilateral_grid: bool = False,
    exposure_correction: bool = False,
    enable_sparsity: bool = False,
    enable_mip: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Unified entrypoint to trigger 3DGS training on any standardized dataset folder.
    
    :param engine: 'lichtfeld', 'splatfacto', or '3dgrut'
    :param dataset_dir: Standardized 3dgs_dataset directory (containing images/ and sparse/0/)
    :param output_dir: Base directory for 3dgs_models
    :param iterations: Number of iterations
    :param mode: 'interactive' or 'headless'
    :param gut: Gaussian Unscented Transform toggle
    :param sh_degree: Spherical harmonics degree
    :param bilateral_grid: Bilateral grid color correction
    :param exposure_correction: Per-photo exposure correction
    :param enable_sparsity: ADMM sparsity pruning
    :param enable_mip: Anti-aliased mip-filtering
    :param progress_callback: Callable for live status logs
    :return: Path to output directory or exported .ply file
    """
    engine_clean = engine.lower().strip()
    target_models_dir = os.path.join(output_dir, f"{engine_clean}_models")
    os.makedirs(target_models_dir, exist_ok=True)

    if progress_callback:
        progress_callback(f"======================================================================")
        progress_callback(f"                STARTING 3D GAUSSIAN SPLATTING TRAINING               ")
        progress_callback(f"======================================================================")
        progress_callback(f" Engine:      {engine.upper()}")
        progress_callback(f" Dataset:     {dataset_dir}")
        progress_callback(f" Models Dir:  {target_models_dir}")
        progress_callback(f" Iterations:  {iterations:,}")
        progress_callback(f" Viewport:    {mode.upper()}")
        progress_callback(f" GUT Optics:  {'Enabled' if gut else 'Disabled'}")
        progress_callback(f" Bilateral:   {'Enabled' if bilateral_grid else 'Disabled'}")
        progress_callback(f" Exposure:    {'Enabled' if exposure_correction else 'Disabled'}")
        progress_callback(f" Sparsity:    {'Enabled' if enable_sparsity else 'Disabled'}")
        progress_callback(f" Mip-Filter:  {'Enabled' if enable_mip else 'Disabled'}")
        progress_callback(f" SH Degree:   {sh_degree}")
        progress_callback(f"======================================================================")

    if engine_clean in ["lichtfeld", "lichtfeld_studio", "lfs"]:
        is_headless = (mode == "headless")
        return train_lichtfeld(
            dataset_dir=dataset_dir,
            output_dir=target_models_dir,
            iterations=iterations,
            headless=is_headless,
            gut=gut,
            sh_degree=sh_degree,
            bilateral_grid=bilateral_grid,
            exposure_correction=exposure_correction,
            enable_sparsity=enable_sparsity,
            enable_mip=enable_mip,
            progress_callback=progress_callback
        )
    elif engine_clean in ["splatfacto", "nerfstudio", "ns"]:
        return train_nerfstudio_splatfacto(
            dataset_dir=dataset_dir,
            output_dir=target_models_dir,
            iterations=iterations,
            model_type="splatfacto",
            progress_callback=progress_callback
        )
    elif engine_clean in ["3dgrut", "3dgut", "grut"]:
        variant = "3dgut" if gut else "3dgrt"
        return train_3dgrut(
            dataset_dir=dataset_dir,
            output_dir=target_models_dir,
            variant=variant,
            iterations=iterations,
            with_viser_gui=(mode == "interactive"),
            progress_callback=progress_callback
        )
    else:
        raise ValueError(f"Unknown 3DGS training engine: '{engine}'. Supported: 'lichtfeld', 'splatfacto', '3dgrut'")


# ----------------------------------------------------------------------
# 5. Standalone CLI Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Abu Align & Train - 3D Gaussian Splatting Training Engine CLI")
    parser.add_argument("--dataset", "-d", required=True, help="Path to dataset directory or 3dgs_dataset folder")
    parser.add_argument("--output", "-o", default=None, help="Destination directory for 3dgs_models")
    parser.add_argument("--engine", "-e", choices=["lichtfeld", "splatfacto", "3dgrut"], default="lichtfeld", help="3DGS Training Engine")
    parser.add_argument("--iterations", "-i", type=int, default=30000, help="Max training iterations (default: 30000; quick test: 7000)")
    parser.add_argument("--mode", "-m", choices=["interactive", "headless"], default="interactive", help="Viewport mode")
    parser.add_argument("--gut", action="store_true", help="Enable Gaussian Unscented Transform (GUT)")
    parser.add_argument("--bilateral_grid", action="store_true", help="Enable bilateral grid color correction")
    parser.add_argument("--exposure_correction", action="store_true", help="Enable per-photo auto-exposure correction")
    parser.add_argument("--enable_sparsity", action="store_true", help="Enable ADMM sparsity pruning")
    parser.add_argument("--enable_mip", action="store_true", help="Enable anti-aliased mip-filtering")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")

    args = parser.parse_args()

    # Smart discovery of 3dgs_dataset
    raw_path = os.path.abspath(args.dataset)
    target_3dgs = None

    if os.path.basename(raw_path).lower() == "3dgs_dataset" and os.path.isdir(raw_path):
        target_3dgs = raw_path
    elif os.path.isdir(os.path.join(raw_path, "3dgs_dataset")):
        target_3dgs = os.path.join(raw_path, "3dgs_dataset")
    elif os.path.isdir(os.path.join(raw_path, "sparse", "0")):
        target_3dgs = raw_path
    else:
        # Search inside results/
        results_dir = os.path.join(raw_path, "results")
        if os.path.isdir(results_dir):
            cands = []
            for d in os.listdir(results_dir):
                sub = os.path.join(results_dir, d, "3dgs_dataset")
                if os.path.isdir(sub):
                    cands.append((os.path.getmtime(sub), sub))
            if cands:
                cands.sort(reverse=True)
                target_3dgs = cands[0][1]

    if not target_3dgs:
        target_3dgs = raw_path

    if args.output:
        out_models = os.path.abspath(args.output)
    else:
        parent = os.path.dirname(target_3dgs) if os.path.basename(target_3dgs).lower() == "3dgs_dataset" else target_3dgs
        out_models = os.path.join(parent, "3dgs_models")

    os.makedirs(out_models, exist_ok=True)

    try:
        run_3dgs_training(
            engine=args.engine,
            dataset_dir=target_3dgs,
            output_dir=out_models,
            iterations=args.iterations,
            mode=args.mode,
            gut=args.gut,
            sh_degree=args.sh_degree,
            bilateral_grid=args.bilateral_grid,
            exposure_correction=args.exposure_correction,
            enable_sparsity=args.enable_sparsity,
            enable_mip=args.enable_mip,
            progress_callback=print
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 3DGS Training failed: {e}")
        sys.exit(1)
