"""
Abu Align & Train v0.0.4.8 - 3D Gaussian Splatting Training Engine
==================================================================
Integrates 4 premier 3DGS training engines:
1. LichtFeld Studio (C++ / Vulkan Real-Time Viewport & Headless with GUT)
2. Nerfstudio Splatfacto (State-of-the-Art PyTorch 3DGS + PLY exporter)
3. NVIDIA 3DGRUT / 3DGUT (Gaussian Ray Tracing & Unscented Transform)
4. PFGS360 (CVPR 2026 Pose-Free Omnidirectional Gaussian Splatting + Consistent Depth Priors)

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

try:
    import wsl_bridge
except ImportError:
    wsl_bridge = None

# Ensure stdout and stderr use UTF-8 on Windows with error replacement
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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

PFGS360_NS_TRAIN_CANDIDATES = [
    r"C:\Users\USer\miniconda3\envs\pfgs360\Scripts\ns-train.exe",
    r"C:\Users\USer\miniconda3\envs\nerfstudio\Scripts\ns-train.exe",
    r"C:\Users\USer\miniconda3\envs\gs_clean\Scripts\ns-train.exe",
    os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "pfgs360", "Scripts", "ns-train.exe"),
    "/opt/conda/envs/pfgs360/bin/ns-train",
]

PFGS360_NS_EXPORT_CANDIDATES = [
    r"C:\Users\USer\miniconda3\envs\pfgs360\Scripts\ns-export.exe",
    r"C:\Users\USer\miniconda3\envs\nerfstudio\Scripts\ns-export.exe",
    r"C:\Users\USer\miniconda3\envs\gs_clean\Scripts\ns-export.exe",
    os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "pfgs360", "Scripts", "ns-export.exe"),
    "/opt/conda/envs/pfgs360/bin/ns-export",
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


def get_pfgs360_train_executable() -> Optional[str]:
    if wsl_bridge and wsl_bridge.is_pfgs360_available():
        return "wsl_ns_train"
    for p in PFGS360_NS_TRAIN_CANDIDATES:
        if os.path.isfile(p):
            return p
    # Fallback to general ns-train
    return get_nerfstudio_train_executable()


def get_pfgs360_export_executable() -> Optional[str]:
    if wsl_bridge and wsl_bridge.is_pfgs360_available():
        return "wsl_ns_export"
    for p in PFGS360_NS_EXPORT_CANDIDATES:
        if os.path.isfile(p):
            return p
    # Fallback to general ns-export
    return get_nerfstudio_export_executable()


def check_save_and_finish_sentinel(output_dir: str, dataset_dir: Optional[str] = None) -> bool:
    """Checks whether the user requested an early Save & Finish in Abu Train."""
    candidates = [
        os.path.join(output_dir, ".finish_and_save"),
        os.path.join(tempfile.gettempdir(), "abu_finish_and_save.flag"),
    ]
    if dataset_dir:
        candidates.append(os.path.join(dataset_dir, ".finish_and_save"))
    parent_out = os.path.dirname(output_dir)
    if parent_out:
        candidates.append(os.path.join(parent_out, ".finish_and_save"))
    for c in candidates:
        try:
            if os.path.exists(c):
                return True
        except Exception:
            pass
    return False


def consume_save_and_finish_sentinel(output_dir: str, dataset_dir: Optional[str] = None):
    """Removes any existing sentinel flags after handling."""
    candidates = [
        os.path.join(output_dir, ".finish_and_save"),
        os.path.join(tempfile.gettempdir(), "abu_finish_and_save.flag"),
    ]
    if dataset_dir:
        candidates.append(os.path.join(dataset_dir, ".finish_and_save"))
    parent_out = os.path.dirname(output_dir)
    if parent_out:
        candidates.append(os.path.join(parent_out, ".finish_and_save"))
    for c in candidates:
        try:
            if os.path.exists(c):
                os.remove(c)
        except Exception:
            pass


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
    export_spz: bool = False,
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
    :param export_spz: Whether to export compressed .spz (10x smaller) in addition to .ply
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
    log(f"      * SH Degree:  {sh_degree}")
    log(f"      * Format:     {'.spz (Compressed Splat)' if export_spz else '.ply (Standard 3DGS)'}")
    log(f"      * Bilateral:  {'Enabled' if bilateral_grid else 'Disabled'}")
    log(f"      * Exposure:   {'Enabled' if exposure_correction else 'Disabled'}")
    log(f"      * Sparsity:   {'Enabled' if enable_sparsity else 'Disabled'}")
    log(f"      * Mip-Filter: {'Enabled' if enable_mip else 'Disabled'}")
    log(f"      * Mode:       {'Headless Background' if headless else 'Live Vulkan Viewport'}")

    export_fmt = "spz" if export_spz else "ply"
    cmd = [
        lfs_exe,
        "-d", os.path.abspath(dataset_dir),
        "-o", os.path.abspath(output_dir),
        "-i", str(iterations),
        "--export", export_fmt,
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
    camera_optimizer: str = "none",
    bilateral_grid: bool = False,
    enable_mip: bool = False,
    sh_degree: int = 3,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes 3DGS training using Nerfstudio Splatfacto.
    
    :param dataset_dir: Directory containing 3dgs_dataset (sparse/0/ and images/)
    :param output_dir: Destination directory for trained model
    :param iterations: Max training iterations (e.g. 30,000)
    :param model_type: 'splatfacto' or 'splatfacto-big'
    :param camera_optimizer: Camera optimizer mode ('none', 'so3_x_r3', 'se3')
    :param bilateral_grid: Whether to train with bilateral grid color correction
    :param enable_mip: Whether to enable anti-aliased mip-filtering
    :param sh_degree: Spherical harmonics degree (0..4)
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
    log(f"      * Cam Opt:    {camera_optimizer.upper()}")
    log(f"      * Bilateral:  {'Enabled' if bilateral_grid else 'Disabled'}")
    log(f"      * Mip-Filter: {'Enabled' if enable_mip else 'Disabled'}")
    log(f"      * SH Degree:  {sh_degree}")
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
        "--vis", "viewer+tensorboard",
        "--viewer.quit-on-train-completion", "True",
        "--logging.steps-per-log=50",
    ]

    if camera_optimizer and camera_optimizer.lower() != "none":
        cmd.append(f"--pipeline.model.camera-optimizer.mode={camera_optimizer.lower()}")
    if sh_degree is not None:
        cmd.append(f"--pipeline.model.sh-degree={sh_degree}")
    if enable_mip:
        cmd.append("--pipeline.model.antialiased=True")

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

        # Live 3D Stream: Detect Viser web viewer URL and auto-launch
        if any(token in line_s for token in ["http://localhost:7007", "http://127.0.0.1:7007", "Viewer at:"]):
            log(f"        [LIVE 3D STREAM] Viser Interactive 3D Viewer Live at: http://localhost:7007")
            try:
                import webbrowser
                webbrowser.open("http://localhost:7007")
            except Exception:
                pass

        # Stream training steps periodically
        if any(k in line_s for k in ["Step", "step", "loss", "Saving", "Printing"]):
            now = time.time()
            if now - last_report_time >= 0.5 or "Saving" in line_s:
                log(f"        [Nerfstudio] {line_s}")
                last_report_time = now

    proc.wait()
    is_early_finish = check_save_and_finish_sentinel(output_dir, dataset_abs)
    if proc.returncode != 0:
        yml_candidates = glob.glob(os.path.join(output_dir, "**", "config.yml"), recursive=True)
        if is_early_finish and yml_candidates:
            log(f"      * [Nerfstudio] Early 'Save & finish' detected (exit code {proc.returncode}). Proceeding to export 3DGS .ply...")
        else:
            raise RuntimeError(f"Nerfstudio ns-train exited with code {proc.returncode}")
    elif is_early_finish:
        log("      * [Nerfstudio] Early 'Save & finish' detected. Proceeding to export 3DGS .ply...")

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
                consume_save_and_finish_sentinel(output_dir, dataset_abs)
                return best_ply
        else:
            log(f"      [Warning] ns-export returned code {res.returncode}: {res.stderr}")

    consume_save_and_finish_sentinel(output_dir, dataset_abs)
    return output_dir


# ----------------------------------------------------------------------
# 3. NVIDIA 3DGRUT / 3DGUT Trainer
# ----------------------------------------------------------------------
def train_3dgrut(
    dataset_dir: str,
    output_dir: str,
    variant: str = "3dgut",
    bvh_backend: str = "optix",
    iterations: int = 30000,
    downsample_factor: int = 1,
    with_viser_gui: bool = True,
    exposure_correction: bool = False,
    enable_sparsity: bool = False,
    sh_degree: int = 3,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes 3DGS training using NVIDIA 3DGRUT / 3DGUT.
    
    :param dataset_dir: Directory containing 3dgs_dataset (sparse/0/ and images/)
    :param output_dir: Destination directory for trained model
    :param variant: '3dgut' (Gaussian Unscented Transform) or '3dgrt' (Gaussian Ray Tracing)
    :param bvh_backend: 'optix' (Hardware RT Cores) or 'cuda' (CUDA Linear BVH)
    :param iterations: Max iterations
    :param downsample_factor: Image downsample factor (1, 2, 4)
    :param with_viser_gui: Launch interactive browser GUI
    :param exposure_correction: Affine exposure appearance correction
    :param enable_sparsity: ADMM sparsity pruning
    :param sh_degree: Spherical harmonics degree (0..3)
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
    log(f"      * Repository:  {repo_dir}")
    log(f"      * Variant:     {variant.upper()}")
    log(f"      * BVH Backend: {bvh_backend.upper()}")
    log(f"      * Config:      {config_name}")
    log(f"      * Exposure:    {'Enabled' if exposure_correction else 'Disabled'}")
    log(f"      * Sparsity:    {'Enabled (ADMM lambda=0.001)' if enable_sparsity else 'Disabled'}")
    log(f"      * SH Degree:   {sh_degree}")
    log(f"      * Dataset:     {dataset_dir}")
    log(f"      * Output Dir:  {output_dir}")

    python_exe = sys.executable
    cmd = [
        python_exe,
        "train.py",
        f"--config-name={config_name}",
        f"path={os.path.abspath(dataset_dir)}",
        f"out_dir={os.path.abspath(output_dir)}",
        f"experiment_name=run_{variant}",
        f"n_iterations={iterations}",
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
# Helper Functions: Video Ingestion & Pose-Free Manifest Bridge
# ----------------------------------------------------------------------
def ensure_dataset_has_images(dataset_dir: str, fps: float = 2.0, log_func=None) -> str:
    """
    Ensures that dataset_dir has an images/ directory with frames.
    If dataset_dir only contains video files (.mp4, .mov, .insv, etc.),
    automatically extracts 360° equirectangular frames using video_extractor.
    """
    images_dir = os.path.join(dataset_dir, "images")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Check if images_dir exists and has images
    if os.path.isdir(images_dir):
        existing = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in valid_exts]
        if len(existing) > 0:
            # Check resolution: if existing images exceed 2K (e.g. 4K/8K), automatically downscale to 2K (1920x960)
            # This protects GPU VRAM from ViT-L quadratic self-attention exhaustion and prevents system freeze.
            try:
                from PIL import Image
                def nat_sort(s):
                    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]
                existing.sort(key=nat_sort)
                first_img = os.path.join(images_dir, existing[0])
                with Image.open(first_img) as im:
                    ew, eh = im.size
                if ew > 2048 or eh > 1024:
                    if log_func:
                        log_func(f"      * [VRAM Shield] Existing frames are {ew}x{eh} (>2K). Resizing to 2K (1920x960) for optimal VRAM hygiene & speed...")
                    for fn in existing:
                        fpath = os.path.join(images_dir, fn)
                        try:
                            with Image.open(fpath) as im:
                                if im.size[0] > 2048 or im.size[1] > 1024:
                                    im_resized = im.resize((1920, 960), Image.Resampling.LANCZOS)
                                    im_resized.save(fpath, quality=95)
                        except Exception:
                            pass
                    if log_func:
                        log_func(f"      * [VRAM Shield] Successfully downscaled {len(existing)} frames to 2K (1920x960).")
            except Exception as ex:
                if log_func:
                    log_func(f"      * [VRAM Shield Warning] Frame downscaling check: {ex}")
            return images_dir

    # Check if dataset_dir itself contains images directly
    direct_imgs = [f for f in os.listdir(dataset_dir) if Path(f).suffix.lower() in valid_exts]
    if len(direct_imgs) > 0:
        os.makedirs(images_dir, exist_ok=True)
        for f in direct_imgs:
            src = os.path.join(dataset_dir, f)
            dst = os.path.join(images_dir, f)
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
        return images_dir

    # Search for video files in dataset_dir
    video_exts = {".mp4", ".mov", ".insv", ".avi", ".mkv", ".webm"}
    video_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if Path(f).suffix.lower() in video_exts and os.path.isfile(os.path.join(dataset_dir, f))
    ]

    if not video_files:
        return images_dir

    target_video = video_files[0]
    vid_name = os.path.basename(target_video)
    if log_func:
        log_func(f"      * [Auto-Extractor] No images found, but detected video: {vid_name}")
        log_func(f"      * [Auto-Extractor] Extracting 2K (1920x960) 360° ERP frames @ {fps} FPS to images/ ...")

    try:
        from video_extractor import extract_video_frames, get_video_info
        info = get_video_info(target_video)
        # For 360 pose-free training, 2K (1920x960) is the optimal standard for monocular depth ViT models
        # (UniK3D) and omnidirectional Gaussian Splatting, ensuring ultra-fast training and <=6GB VRAM footprint.
        target_res = "2k"

        os.makedirs(images_dir, exist_ok=True)
        res = extract_video_frames(
            video_path=target_video,
            output_dir=images_dir,
            mode="fps",
            fps=fps,
            image_format="jpg",
            quality=95,
            resolution=target_res,
            progress_callback=lambda p: log_func(f"        [Extractor] Frame {p.get('current_frame')}/{p.get('total_frames')} ({p.get('percent', 0):.1f}%)") if log_func and p.get("current_frame", 0) % 20 == 0 else None
        )
        if log_func:
            log_func(f"      * [Auto-Extractor] Extracted {res.get('total_frames', 0)} frames successfully in {res.get('elapsed_seconds', 0)}s!")
    except Exception as ex:
        if log_func:
            log_func(f"      * [Auto-Extractor Warning] Video extraction failed: {ex}")

    return images_dir


def ensure_transforms_json_for_posefree(dataset_dir: str, log_func=None) -> Optional[str]:
    """
    Creates a standard transforms.json with EQUIRECTANGULAR camera model and identity poses
    for an unposed sequence in dataset_dir/images.
    """
    tf_path = os.path.join(dataset_dir, "transforms.json")
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.isdir(images_dir):
        images_dir = dataset_dir

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [f for f in os.listdir(images_dir) if Path(f).suffix.lower() in valid_exts]
    def nat_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]
    imgs.sort(key=nat_key)

    if os.path.isfile(tf_path):
        if imgs:
            try:
                from PIL import Image
                first_img = os.path.join(images_dir, imgs[0])
                with Image.open(first_img) as im:
                    cur_w, cur_h = im.size
                with open(tf_path, "r", encoding="utf-8") as f:
                    tf_data = json.load(f)
                if tf_data.get("w") != cur_w or tf_data.get("h") != cur_h:
                    tf_data["w"] = int(cur_w)
                    tf_data["h"] = int(cur_h)
                    tf_data["fl_x"] = float(cur_h)
                    tf_data["fl_y"] = float(cur_h)
                    tf_data["cx"] = float(cur_w) / 2.0
                    tf_data["cy"] = float(cur_h) / 2.0
                    with open(tf_path, "w", encoding="utf-8") as f:
                        json.dump(tf_data, f, indent=2)
                    if log_func:
                        log_func(f"      * [Pose-Free Manifest] Synchronized transforms.json dimensions to {cur_w}x{cur_h}")
            except Exception:
                pass
        return tf_path

    if not imgs:
        return None

    try:
        from PIL import Image
        first_img = os.path.join(images_dir, imgs[0])
        with Image.open(first_img) as im:
            w, h = im.size
    except Exception:
        w, h = 1920, 960

    ident = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
    frames = []
    for fn in imgs:
        rel_path = f"images/{fn}" if os.path.isdir(os.path.join(dataset_dir, "images")) else fn
        frames.append({
            "file_path": rel_path,
            "transform_matrix": ident
        })

    tf_data = {
        "camera_model": "EQUIRECTANGULAR",
        "w": int(w),
        "h": int(h),
        "fl_x": float(h),
        "fl_y": float(h),
        "cx": float(w) / 2.0,
        "cy": float(h) / 2.0,
        "frames": frames
    }

    try:
        with open(tf_path, "w", encoding="utf-8") as f:
            json.dump(tf_data, f, indent=2)
        if log_func:
            log_func(f"      * [Pose-Free Manifest] Generated transforms.json with {len(frames)} frames (EQUIRECTANGULAR {w}x{h})")
        return tf_path
    except Exception as ex:
        if log_func:
            log_func(f"      * [Warning] Could not generate transforms.json: {ex}")
        return None


def export_pfgs360_checkpoint_to_ply(ckpt_path: str, out_ply: str) -> str:
    """
    Directly extracts 3D Gaussian Splatting parameters from a PFGS360 PyTorch checkpoint
    and writes a standard binary PLY point cloud file compatible with all 3DGS viewers.
    """
    import torch
    import numpy as np
    from collections import OrderedDict

    data = torch.load(ckpt_path, map_location="cpu")
    pipe = data.get("pipeline", {})
    if "_model.gauss_params.means" not in pipe:
        prefix = ""
        for k in pipe.keys():
            if "gauss_params.means" in k:
                prefix = k.split("gauss_params.means")[0]
                break
    else:
        prefix = "_model."

    means = pipe[f"{prefix}gauss_params.means"].float().numpy()
    opacities = pipe[f"{prefix}gauss_params.opacities"].float().numpy()
    scales = pipe[f"{prefix}gauss_params.scales"].float().numpy()
    quats = pipe[f"{prefix}gauss_params.quats"].float().numpy()
    f_dc = pipe[f"{prefix}gauss_params.features_dc"].float().numpy()
    f_rest = pipe[f"{prefix}gauss_params.features_rest"].float().numpy()

    N = means.shape[0]
    map_to_tensors = OrderedDict()
    map_to_tensors["x"] = means[:, 0]
    map_to_tensors["y"] = means[:, 1]
    map_to_tensors["z"] = means[:, 2]
    map_to_tensors["nx"] = np.zeros(N, dtype=np.float32)
    map_to_tensors["ny"] = np.zeros(N, dtype=np.float32)
    map_to_tensors["nz"] = np.zeros(N, dtype=np.float32)

    for i in range(3):
        map_to_tensors[f"f_dc_{i}"] = f_dc[:, i, None]

    if len(f_rest.shape) == 3 and f_rest.shape[1] > 0:
        shs_rest = np.transpose(f_rest, (0, 2, 1)).reshape(N, -1)
        for i in range(shs_rest.shape[-1]):
            map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

    map_to_tensors["opacity"] = opacities
    for i in range(3):
        map_to_tensors[f"scale_{i}"] = scales[:, i, None]
    for i in range(4):
        map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    valid = np.ones(N, dtype=bool)
    for k, v in map_to_tensors.items():
        valid = np.logical_and(valid, np.isfinite(v).all(axis=-1))

    # Opacity threshold logit(1/255) approx -5.5373
    valid = np.logical_and(valid, (opacities.squeeze(-1) >= -5.5373))
    count = int(np.sum(valid))

    os.makedirs(os.path.dirname(os.path.abspath(out_ply)), exist_ok=True)
    type_list = [(k, "f4") for k in map_to_tensors.keys()]
    rec_arr = np.empty(count, dtype=type_list)
    for k in map_to_tensors.keys():
        rec_arr[k] = map_to_tensors[k][valid].squeeze()

    with open(out_ply, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(b"comment Vertical Axis: z\n")
        f.write(f"element vertex {count}\n".encode())
        for k in map_to_tensors.keys():
            f.write(f"property float {k}\n".encode())
        f.write(b"end_header\n")
        rec_arr.tofile(f)

    return out_ply


# ----------------------------------------------------------------------
# 4. PFGS360 Trainer (CVPR 2026 Pose-Free 360° Omnidirectional 3DGS)
# ----------------------------------------------------------------------
def train_pfgs360(
    dataset_dir: str,
    output_dir: str,
    iterations: int = 30000,
    depth_type: str = "unik3d",
    dataparser_mode: str = "auto",
    spherical_aware: bool = True,
    consist_aware: bool = True,
    inlier_growth: bool = True,
    align_depth: bool = True,
    patch_filter: bool = True,
    filter_sky: bool = True,
    sky_depth: float = 50.0,
    match_views: int = 8,
    densify_views: int = 8,
    sh_degree: int = 3,
    downscale_factor: Optional[str] = "1",
    mode: str = "interactive",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Executes 360° Omnidirectional Gaussian Splatting using PFGS360 (CVPR 2026).
    Supports both pose-free direct learning and OpenSfM / 360DVO alignment handoff.
    """
    viewer_opened = False
    def log(msg):
        nonlocal viewer_opened
        s = str(msg)
        if ("localhost:7007" in s or "Viewer running at" in s) and not viewer_opened:
            viewer_opened = True
            try:
                import webbrowser
                webbrowser.open("http://localhost:7007")
            except Exception:
                pass
        try:
            if progress_callback:
                progress_callback(s)
            else:
                print(s)
        except UnicodeEncodeError:
            clean = s.encode("ascii", errors="replace").decode("ascii")
            if progress_callback:
                progress_callback(clean)
            else:
                print(clean)

    ns_train_exe = get_pfgs360_train_executable()
    ns_export_exe = get_pfgs360_export_executable()
    if not ns_train_exe:
        raise FileNotFoundError(
            f"PFGS360 ns-train executable not found. Checked: {PFGS360_NS_TRAIN_CANDIDATES}"
        )

    os.makedirs(output_dir, exist_ok=True)
    dataset_abs = os.path.abspath(dataset_dir)

    # 1. Ensure images exist (auto-extract from video if video file was provided)
    ensure_dataset_has_images(dataset_abs, fps=2.0, log_func=log)

    # 2. In Pose-Free mode, ensure transforms.json is generated as bridge file
    if dataparser_mode in ["auto", "posefree", "pose-free", "direct"]:
        ensure_transforms_json_for_posefree(dataset_abs, log_func=log)

    # Determine optimal dataparser according to 360° Data Intake & Alignment Pipeline specification:
    # 1. Condition 1 & 2 / Paradigm B: If transforms.json exists with real camera poses (from 360DVO or OpenSfM), route to nerfstudio-data
    # 2. OpenSfM native handoff: If opensfm/reconstruction.json exists, route to opensfm-dataparser
    # 3. Condition 3 / Paradigm A: Pose-Free mode (unposed ERP frames in images/)
    tf_json = os.path.join(dataset_abs, "transforms.json")
    rec_json = os.path.join(dataset_abs, "opensfm", "reconstruction.json")
    if not os.path.isfile(rec_json):
        rec_json = os.path.join(dataset_abs, "reconstruction.json")

    has_sparse = os.path.isdir(os.path.join(dataset_abs, "sparse", "0"))

    if dataparser_mode == "auto":
        is_posed_tf = False
        if os.path.isfile(tf_json):
            try:
                with open(tf_json, "r", encoding="utf-8") as f:
                    tf_meta = json.load(f)
                frames = tf_meta.get("frames", [])
                if len(frames) > 1 and "transform_matrix" in frames[0] and "transform_matrix" in frames[1]:
                    if frames[0]["transform_matrix"] != frames[1]["transform_matrix"]:
                        is_posed_tf = True
            except Exception:
                pass

        if os.path.isfile(rec_json):
            active_dp = "opensfm-dataparser"
            dp_mode_label = "OpenSfM / 360DVO Handoff (opensfm-dataparser - Instant 100% GPU)"
        elif is_posed_tf:
            active_dp = "nerfstudio-data"
            dp_mode_label = "Transforms.json Handoff (nerfstudio-data - Instant 100% GPU)"
        elif has_sparse:
            active_dp = "nerfstudio-data"
            dp_mode_label = "COLMAP Sparse Model Handoff (nerfstudio-data - Instant 100% GPU)"
        else:
            active_dp = "posefree-dataparser"
            dp_mode_label = "Pose-Free Direct (Paradigm A / Condition 3 - CPU Matching Phase)"
    elif dataparser_mode in ["nerfstudio", "nerfstudio-data", "transforms"]:
        active_dp = "nerfstudio-data"
        dp_mode_label = "Transforms.json Handoff (nerfstudio-data - Instant 100% GPU)"
    elif dataparser_mode in ["opensfm", "handoff"]:
        active_dp = "opensfm-dataparser"
        dp_mode_label = "OpenSfM Handoff (opensfm-dataparser - Instant 100% GPU)"
    elif dataparser_mode in ["ob3d", "egocentric"]:
        active_dp = "ob3d-dataparser"
        dp_mode_label = "OB3D Egocentric Dataparser"
    elif dataparser_mode in ["posefree", "pose-free", "direct"]:
        active_dp = "posefree-dataparser"
        dp_mode_label = "Pose-Free Direct (Paradigm A / Condition 3 - CPU Matching Phase)"
        if os.path.isfile(rec_json) or os.path.isfile(tf_json) or has_sparse:
            log("      * [NOTICE] Dataset already contains pre-aligned camera poses (from OpenSfM/360DVO/COLMAP).")
            log("      * [NOTICE] 'Pose-Free Direct' mode forces PFGS360 to discard prior poses and solve camera")
            log("      *          trajectories from scratch on CPU view-by-view (NearestNeighbors).")
            log("      * [TIP] To train immediately at full 100% RTX 4090 GPU capacity without waiting for CPU pose")
            log("      *       calculation, set Dataparser Mode to 'Auto-Detect' or 'OpenSfM / 360DVO Handoff'.")
        if match_views == 0:
            log("      * [PERFORMANCE WARNING] 'match_views' is set to 0 (All Visited Views).")
            log("      *                     PFGS360 will perform O(N²) sequential pairwise SIFT matching on CPU.")
            log("      *                     This will bottleneck progress and temporarily keep GPU utilization low (~10-25%).")
            log("      * [RECOMMENDATION] Set match_views=8 for a fast sliding window, or pre-align with OpenSfM/360DVO.")
        else:
            log(f"      * [FAST SLIDING WINDOW] match_views={match_views}: Restricts CPU pose matching to recent {match_views} views (O(N·K) complexity).")
            log("      *                       Prevents CPU bottlenecking and maintains steady GPU training progress!")
    else:
        active_dp = f"{dataparser_mode}-dataparser" if not dataparser_mode.endswith("-dataparser") and dataparser_mode != "nerfstudio-data" else dataparser_mode
        dp_mode_label = active_dp

    if active_dp in ["opensfm-dataparser", "nerfstudio-data"]:
        log("      * [FULL GPU POWER] Pre-aligned camera poses detected! Bypassing CPU matching to engage 100% RTX 4090 GPU immediately.")

    # Ensure opensfm/reconstruction.json and metadata.json exist if opensfm-dataparser is active
    if active_dp == "opensfm-dataparser":
        osfm_dir = os.path.join(dataset_abs, "opensfm")
        os.makedirs(osfm_dir, exist_ok=True)
        target_rec = os.path.join(osfm_dir, "reconstruction.json")
        if not os.path.isfile(target_rec):
            for cand in [
                os.path.join(dataset_abs, "reconstruction.json"),
                os.path.join(os.path.dirname(dataset_abs), "reconstruction.json"),
                os.path.join(os.path.dirname(dataset_abs), "opensfm", "reconstruction.json")
            ]:
                if os.path.isfile(cand):
                    shutil.copy2(cand, target_rec)
                    break

        # Auto-heal: If reconstruction.json exists but points is empty, inject points from points.ply or 360dvo_points.ply
        if os.path.isfile(target_rec):
            try:
                with open(target_rec, "r", encoding="utf-8") as f:
                    rec_content = json.load(f)
                if isinstance(rec_content, list) and len(rec_content) > 0:
                    pts_meta = rec_content[0].get("points", {})
                    if len(pts_meta) == 0:
                        ply_cands = [
                            os.path.join(dataset_abs, "points.ply"),
                            os.path.join(dataset_abs, "360dvo_points.ply"),
                            os.path.join(dataset_abs, "points3D.ply"),
                            os.path.join(os.path.dirname(dataset_abs), "360dvo_points.ply"),
                            os.path.join(os.path.dirname(dataset_abs), "points.ply"),
                            os.path.join(os.path.dirname(dataset_abs), "points3D.ply"),
                        ]
                        for pc in ply_cands:
                            if os.path.isfile(pc):
                                from dvo360_engine import load_ply_points_for_opensfm
                                healed_pts = load_ply_points_for_opensfm(pc)
                                if healed_pts:
                                    rec_content[0]["points"] = healed_pts
                                    with open(target_rec, "w", encoding="utf-8") as fw:
                                        json.dump(rec_content, fw, indent=4)
                                    log(f"      * [Auto-Heal] Injected {len(healed_pts)} 3D tie points from {os.path.basename(pc)} into reconstruction.json")
                                    root_rec = os.path.join(dataset_abs, "reconstruction.json")
                                    if os.path.isfile(root_rec):
                                        with open(root_rec, "w", encoding="utf-8") as fwr:
                                            json.dump(rec_content, fwr, indent=4)
                                    break
            except Exception as ex:
                log(f"      * [Auto-Heal Warning] Point injection check: {ex}")

        try:
            from opensfm_engine import ensure_opensfm_metadata
            ensure_opensfm_metadata(dataset_abs)
        except Exception:
            pass

    log(f"      * Initializing PFGS360 Engine (CVPR 2026)...")
    log(f"      * ns-train:      {ns_train_exe}")
    log(f"      * Dataparser:    {dp_mode_label}")
    log(f"      * Depth Prior:   {depth_type.upper()} (Omnidirectional)")
    log(f"      * Match Views:   {f'{match_views} Keyframes (Sliding Window)' if match_views > 0 else 'All Visited Views (O(N²) CPU Matching)'}")
    log(f"      * Densify Views: {f'{densify_views} Keyframes (Sliding Window)' if densify_views > 0 else 'All Keyframes'}")
    log(f"      * Spherical:     {'Enabled (S² Manifold)' if spherical_aware else 'Disabled'}")
    log(f"      * Consist-Aware: {'Enabled (Epipolar Masks)' if consist_aware else 'Disabled'}")
    log(f"      * Inlier-Grow:   {'Enabled (DIA-Densify)' if inlier_growth else 'Disabled'}")
    log(f"      * Align-Depth:   {'Enabled' if align_depth else 'Disabled'}")
    log(f"      * Filter Sky:    {f'Enabled (>{sky_depth:.0f}m)' if filter_sky else 'Disabled'}")
    log(f"      * Downscale:     {f'{ds_flag[1]}x' if ds_flag else 'Auto (<=1600px Max)'}")
    log(f"      * Viewer Mode:   {'Headless (100% Dedicated GPU)' if mode == 'headless' else 'Interactive Viser (http://localhost:7007)'}")
    log(f"      * Dataset:       {dataset_abs}")
    log(f"      * Output Dir:    {output_dir}")
    log(f"      * Max Steps:     {iterations:,}")

    if ns_train_exe == "wsl_ns_train" and wsl_bridge:
        log("      * Engaging Native Linux PFGS360 via WSL2 Ubuntu (RTX 4090 GPU Accelerated)...")
        wsl_data = wsl_bridge.win_to_wsl_path(dataset_abs)
        wsl_out = wsl_bridge.win_to_wsl_path(output_dir)

        wsl_cmd_parts = [
            f"{wsl_bridge.WSL_NS_TRAIN}",
            "pfgs360",
            "--max-num-iterations", str(iterations),
            "--output-dir", f"'{wsl_out}'",
            "--pipeline.model.sh-degree", str(sh_degree),
            "--pipeline.model.depth-type", depth_type,
        ] + vis_flag + [
            "--viewer.quit-on-train-completion", "True",
        ]
        if not spherical_aware:
            wsl_cmd_parts.extend(["--pipeline.model.spherical-aware", "False"])
        if not consist_aware:
            wsl_cmd_parts.extend(["--pipeline.model.consist-aware", "False"])
        if not inlier_growth:
            wsl_cmd_parts.extend(["--pipeline.model.inlier-growth", "False"])
        if not align_depth:
            wsl_cmd_parts.extend(["--pipeline.model.align-depth", "False"])
        if not patch_filter:
            wsl_cmd_parts.extend(["--pipeline.model.patch-filter", "False"])
        if filter_sky:
            wsl_cmd_parts.extend(["--pipeline.model.filter-sky", "True"])
            if sky_depth != 50.0:
                wsl_cmd_parts.extend(["--pipeline.model.sky-depth", str(sky_depth)])
        else:
            wsl_cmd_parts.extend(["--pipeline.model.filter-sky", "False"])
        if match_views > 0:
            wsl_cmd_parts.extend(["--pipeline.model.match-views", str(match_views)])
        if densify_views > 0:
            wsl_cmd_parts.extend(["--pipeline.model.densify-views", str(densify_views)])

        if active_dp == "nerfstudio-data":
            wsl_cmd_parts.extend(["nerfstudio-data", "--data", f"'{wsl_data}'", "--auto-scale-poses", "False"] + (ds_flag if ds_flag else ["--downscale-factor", "1"]))
        elif active_dp == "opensfm-dataparser":
            wsl_cmd_parts.extend(["opensfm-dataparser", "--data", f"'{wsl_data}'", "--eval-mode", "all"] + ds_flag)
        elif active_dp == "ob3d-dataparser":
            wsl_cmd_parts.extend(["ob3d-dataparser", "--data", f"'{wsl_data}'", "--trajectory-type", "Egocentric"] + ds_flag)
        elif active_dp in ["posefree-dataparser", "posefree"]:
            wsl_cmd_parts.extend(["posefree-dataparser", "--data", f"'{wsl_data}'"] + ds_flag)
        elif active_dp:
            wsl_cmd_parts.extend([active_dp, "--data", f"'{wsl_data}'"] + ds_flag)
        else:
            wsl_cmd_parts.extend(["--data", f"'{wsl_data}'"] + ds_flag)

        full_wsl_cmd = " ".join(wsl_cmd_parts)
        log(f"      * Launching WSL command: {full_wsl_cmd}")
        ret, out = wsl_bridge.run_command_in_wsl(full_wsl_cmd, log_callback=log)
        is_early_finish = check_save_and_finish_sentinel(output_dir, dataset_dir)
        if ret != 0:
            config_candidates = glob.glob(os.path.join(output_dir, "**", "config.yml"), recursive=True)
            if is_early_finish and config_candidates:
                log(f"      * [PFGS360] Early 'Save & finish' detected (exit code {ret}). Proceeding to export 3DGS .ply...")
            else:
                raise RuntimeError(f"WSL2 PFGS360 training failed (exit code {ret})")
        elif is_early_finish:
            log("      * [PFGS360] Early 'Save & finish' detected. Proceeding to export 3DGS .ply...")

        # Automatically export 3DGS .ply point cloud if needed
        config_candidates = glob.glob(os.path.join(output_dir, "**", "config.yml"), recursive=True)
        if config_candidates:
            latest_config = max(config_candidates, key=os.path.getmtime)
            log(f"      * [PFGS360] Exporting Gaussian Splats to standard 3DGS .ply format...")
            wsl_cfg = wsl_bridge.win_to_wsl_path(latest_config)
            export_cmd = f"{wsl_bridge.WSL_NS_EXPORT} gaussian-splat --load-config '{wsl_cfg}' --output-dir '{wsl_out}' --output-filename 'pfgs360_splat.ply'"
            wsl_bridge.run_command_in_wsl(export_cmd, log_callback=log)

        consume_save_and_finish_sentinel(output_dir, dataset_dir)

        target_splat = os.path.join(output_dir, "pfgs360_splat.ply")
        if os.path.isfile(target_splat):
            log(f"      * [PFGS360] Training complete! Model exported: {target_splat}")
            return target_splat

        splat_candidates = [p for p in glob.glob(os.path.join(output_dir, "**", "*.ply"), recursive=True) if "splat" in os.path.basename(p).lower()]
        if splat_candidates:
            best_splat = max(splat_candidates, key=os.path.getmtime)
            log(f"      * [PFGS360] Training complete! Model exported: {best_splat}")
            return best_splat

        ply_candidates = glob.glob(os.path.join(output_dir, "*.ply")) + \
                         glob.glob(os.path.join(output_dir, "**", "*.ply"), recursive=True)
        if ply_candidates:
            best_ply = max(ply_candidates, key=os.path.getmtime)
            log(f"      * [PFGS360] Training complete! Model exported: {best_ply}")
            return best_ply
        return output_dir

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    scripts_dir = os.path.dirname(ns_train_exe)
    env["PATH"] = f"{scripts_dir};{os.path.dirname(scripts_dir)};{env.get('PATH', '')}"

    cmd = [
        ns_train_exe,
        "pfgs360",
        "--max-num-iterations", str(iterations),
        "--output-dir", output_dir,
        "--pipeline.model.sh-degree", str(sh_degree),
        "--pipeline.model.depth-type", depth_type,
    ] + vis_flag + [
        "--viewer.quit-on-train-completion", "True",
    ]
    if not spherical_aware:
        cmd.extend(["--pipeline.model.spherical-aware", "False"])
    if not consist_aware:
        cmd.extend(["--pipeline.model.consist-aware", "False"])
    if not inlier_growth:
        cmd.extend(["--pipeline.model.inlier-growth", "False"])
    if not align_depth:
        cmd.extend(["--pipeline.model.align-depth", "False"])
    if not patch_filter:
        cmd.extend(["--pipeline.model.patch-filter", "False"])
    if filter_sky:
        cmd.extend(["--pipeline.model.filter-sky", "True"])
        if sky_depth != 50.0:
            cmd.extend(["--pipeline.model.sky-depth", str(sky_depth)])
    else:
        cmd.extend(["--pipeline.model.filter-sky", "False"])
    if match_views > 0:
        cmd.extend(["--pipeline.model.match-views", str(match_views)])
    if densify_views > 0:
        cmd.extend(["--pipeline.model.densify-views", str(densify_views)])

    # Append dataparser subcommand and options in correct Tyro order
    if active_dp == "nerfstudio-data":
        cmd.extend(["nerfstudio-data", "--data", dataset_abs, "--auto-scale-poses", "False"] + (ds_flag if ds_flag else ["--downscale-factor", "1"]))
    elif active_dp == "opensfm-dataparser":
        cmd.extend(["opensfm-dataparser", "--data", dataset_abs, "--eval-mode", "all"] + ds_flag)
    elif active_dp == "ob3d-dataparser":
        cmd.extend(["ob3d-dataparser", "--data", dataset_abs, "--trajectory-type", "Egocentric"] + ds_flag)
    elif active_dp in ["posefree-dataparser", "posefree"]:
        cmd.extend(["posefree-dataparser", "--data", dataset_abs] + ds_flag)
    elif active_dp:
        cmd.extend([active_dp, "--data", dataset_abs] + ds_flag)
    else:
        # Paradigm A: Pose-Free Direct command
        cmd.extend(["--data", dataset_abs] + ds_flag)

    log(f"      * Launching command: {' '.join(cmd)}")
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        env=env
    )

    captured_lines = []
    last_report_time = time.time()
    for line in proc.stdout:
        line_s = line.strip()
        if not line_s:
            continue
        captured_lines.append(line_s)
        if len(captured_lines) > 40:
            captured_lines.pop(0)

        # Sanitize any non-encodable box-drawing characters for legacy consoles
        clean_line = line_s
        if sys.platform == "win32":
            try:
                clean_line.encode(sys.stdout.encoding or "utf-8")
            except UnicodeEncodeError:
                clean_line = clean_line.encode("ascii", errors="replace").decode("ascii")

        lower = clean_line.lower()
        if any(keyword in lower for keyword in ["iter", "step", "loss", "psnr"]):
            now = time.time()
            if now - last_report_time >= 0.5 or any(k in lower for k in ["done", "finish", "export", "saved"]):
                log(f"        [PFGS360] {clean_line}")
                last_report_time = now
        elif any(keyword in lower for keyword in ["error", "modulenotfounderror", "traceback", "fatal", "unsupported"]):
            log(f"        [PFGS360-ERR] {clean_line}")
        else:
            log(f"        [PFGS360] {clean_line}")

    proc.wait()
    is_early_finish = check_save_and_finish_sentinel(output_dir, dataset_abs)
    if proc.returncode != 0:
        ckpt_files = glob.glob(os.path.join(output_dir, "**", "*.ckpt"), recursive=True)
        if is_early_finish and ckpt_files:
            log(f"      * [PFGS360] Early 'Save & finish' detected (exit code {proc.returncode}). Proceeding to export 3DGS .ply...")
        else:
            err_block = "\n".join(captured_lines[-15:]) if captured_lines else f"Exited with code {proc.returncode}"
            if "spherical_distortion" in err_block or "No module named" in err_block or "invalid choice: 'pfgs360'" in err_block:
                missing_pkg = "spherical_distortion" if "spherical_distortion" in err_block else "PFGS360 C++ extensions"
                raise RuntimeError(
                    f"PFGS360 requires '{missing_pkg}'.\n"
                    f"Please run 'setup_pfgs360.bat' (Windows) or 'setup_pfgs360.sh' (Linux/RunPod) to build extensions.\n"
                    f"Details:\n{err_block}"
                )
            raise RuntimeError(f"PFGS360 training failed (exit code {proc.returncode}):\n{err_block}")
    elif is_early_finish:
        log("      * [PFGS360] Early 'Save & finish' detected. Proceeding to export 3DGS .ply...")

    # Export Gaussian Splatting .ply directly from checkpoint or via ns-export fallback
    ckpt_files = glob.glob(os.path.join(output_dir, "**", "*.ckpt"), recursive=True)
    export_ply_path = os.path.join(output_dir, "pfgs360_splat.ply")
    exported = False
    if ckpt_files:
        best_ckpt = max(ckpt_files, key=os.path.getmtime)
        log(f"      * Exporting PFGS360 Gaussian Splat to PLY: {export_ply_path}")
        try:
            export_pfgs360_checkpoint_to_ply(best_ckpt, export_ply_path)
            log(f"      * [PFGS360] PLY exported successfully ({os.path.getsize(export_ply_path):,} bytes)!")
            exported = True
        except Exception as ex:
            log(f"      * [Warning] Direct PLY export failed: {ex}")

    if not exported:
        config_files = glob.glob(os.path.join(output_dir, "**", "config.yml"), recursive=True)
        if config_files and ns_export_exe:
            best_config = max(config_files, key=os.path.getmtime)
            log(f"      * Exporting PFGS360 Gaussian Splat via ns-export: {export_ply_path}")
            export_cmd = [
                ns_export_exe,
                "gaussian-splat",
                "--load-config", best_config,
                "--output-dir", output_dir
            ]
            try:
                subprocess.run(export_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            except Exception as ex:
                log(f"      * [Warning] ns-export failed: {ex}")

    ply_candidates = glob.glob(os.path.join(output_dir, "*.ply")) + \
                     glob.glob(os.path.join(output_dir, "**", "*.ply"), recursive=True)
    if ply_candidates:
        best_ply = max(ply_candidates, key=os.path.getmtime)
        log(f"      * [PFGS360] Training complete! Model exported: {best_ply} ({time.time()-start_time:.1f}s)")
        consume_save_and_finish_sentinel(output_dir, dataset_abs)
        return best_ply

    log(f"      * [PFGS360] Training complete in {time.time()-start_time:.1f}s.")
    consume_save_and_finish_sentinel(output_dir, dataset_abs)
    return output_dir


# ----------------------------------------------------------------------
# 5. Master 3DGS Training Dispatcher
# ----------------------------------------------------------------------
def run_3dgs_training(
    engine: str,
    dataset_dir: str,
    output_dir: str,
    iterations: int = 30000,
    mode: str = "interactive",
    gut: bool = False,
    sh_degree: int = 3,
    export_spz: bool = False,
    ns_model: str = "splatfacto",
    camera_optimizer: str = "none",
    bilateral_grid: bool = False,
    exposure_correction: bool = False,
    enable_sparsity: bool = False,
    enable_mip: bool = False,
    representation: str = "3dgut",
    bvh_backend: str = "optix",
    pfgs_depth: str = "unik3d",
    pfgs_dataparser: str = "auto",
    pfgs_filter_sky: bool = True,
    pfgs_sky_depth: float = 50.0,
    pfgs_match_views: int = 8,
    pfgs_densify_views: int = 8,
    pfgs_spherical: bool = True,
    pfgs_consist: bool = True,
    pfgs_inlier: bool = True,
    pfgs_align_depth: bool = True,
    pfgs_downscale: str = "1",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Unified entrypoint to trigger 3DGS training on any standardized dataset folder.
    """
    engine_clean = engine.lower().strip()
    target_models_dir = os.path.join(output_dir, f"{engine_clean}_models")
    os.makedirs(target_models_dir, exist_ok=True)

    # Smart dataset discovery: if dataset_dir doesn't directly contain poses,
    # search standard Abu Align subdirectories (e.g. 3dgs_dataset, results/sparse_*/3dgs_dataset)
    discovered_dataset = None
    if not (os.path.isfile(os.path.join(dataset_dir, "transforms.json")) or 
            os.path.isfile(os.path.join(dataset_dir, "reconstruction.json")) or
            os.path.isfile(os.path.join(dataset_dir, "opensfm", "reconstruction.json")) or
            os.path.isdir(os.path.join(dataset_dir, "sparse", "0"))):
        
        cand_3dgs = os.path.join(dataset_dir, "3dgs_dataset")
        if (os.path.isfile(os.path.join(cand_3dgs, "transforms.json")) or 
            os.path.isfile(os.path.join(cand_3dgs, "opensfm", "reconstruction.json")) or
            os.path.isdir(os.path.join(cand_3dgs, "sparse", "0"))):
            discovered_dataset = cand_3dgs
        else:
            cand_results = os.path.join(dataset_dir, "results")
            if not os.path.isdir(cand_results):
                cand_parent_results = os.path.join(os.path.dirname(dataset_dir), "results")
                if os.path.isdir(cand_parent_results):
                    cand_results = cand_parent_results
            
            if os.path.isdir(cand_results):
                subdirs = [os.path.join(cand_results, d) for d in os.listdir(cand_results) if d.startswith("sparse_")]
                subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                for sd in subdirs:
                    target_cand = os.path.join(sd, "3dgs_dataset")
                    if (os.path.isfile(os.path.join(target_cand, "transforms.json")) or 
                        os.path.isfile(os.path.join(target_cand, "opensfm", "reconstruction.json")) or
                        os.path.isdir(os.path.join(target_cand, "sparse", "0"))):
                        discovered_dataset = target_cand
                        break
                    elif (os.path.isfile(os.path.join(sd, "transforms.json")) or 
                          os.path.isdir(os.path.join(sd, "sparse", "0"))):
                        discovered_dataset = sd
                        break

    if discovered_dataset:
        if engine_clean in ["lichtfeld", "splatfacto", "3dgrut"] or (engine_clean in ["pfgs360", "pfgs"] and str(pfgs_dataparser).lower() != "posefree"):
            if progress_callback:
                progress_callback(f"      * [SMART DISCOVERY] Auto-resolved raw image folder to latest aligned dataset:")
                progress_callback(f"        -> {discovered_dataset}")
            dataset_dir = discovered_dataset

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
        progress_callback(f" Format/Spz:  {'.spz' if export_spz else '.ply'}")
        progress_callback(f" Model/Rep:   {ns_model if 'splat' in engine_clean else representation}")
        progress_callback(f" Cam Opt:     {camera_optimizer}")
        progress_callback(f" Bilateral:   {'Enabled' if bilateral_grid else 'Disabled'}")
        progress_callback(f" Exposure:    {'Enabled' if exposure_correction else 'Disabled'}")
        progress_callback(f" Sparsity:    {'Enabled' if enable_sparsity else 'Disabled'}")
        progress_callback(f" Mip-Filter:  {'Enabled' if enable_mip else 'Disabled'}")
        progress_callback(f" SH Degree:   {sh_degree}")
        if "pfgs" in engine_clean:
            progress_callback(f" Depth Model: {pfgs_depth.upper()}")
            rec_check = os.path.isfile(os.path.join(dataset_dir, "opensfm", "reconstruction.json")) or os.path.isfile(os.path.join(dataset_dir, "reconstruction.json"))
            tf_check = os.path.isfile(os.path.join(dataset_dir, "transforms.json"))
            if pfgs_dataparser.lower() == "auto":
                if rec_check:
                    dp_disp = "AUTO (Auto-Routed: OpenSfM Handoff [100% GPU])"
                elif tf_check:
                    dp_disp = "AUTO (Auto-Routed: Transforms.json Handoff [100% GPU])"
                else:
                    dp_disp = "AUTO (Auto-Routed: Pose-Free Direct [posefree-dataparser])"
            elif pfgs_dataparser.lower() in ["posefree", "pose-free", "direct"]:
                dp_disp = "POSE-FREE DIRECT (posefree-dataparser - Paradigm A)"
            elif pfgs_dataparser.lower() in ["opensfm", "handoff"]:
                dp_disp = "SFM HANDOFF (opensfm-dataparser [100% GPU])"
            elif pfgs_dataparser.lower() in ["ob3d", "egocentric"]:
                dp_disp = "OB3D BENCHMARK (ob3d-dataparser)"
            else:
                dp_disp = pfgs_dataparser.upper()
            progress_callback(f" Dataparser:  {dp_disp}")
            progress_callback(f" Match Views: {f'{pfgs_match_views} Keyframes' if pfgs_match_views > 0 else 'All Views (O(N²))'}")
            progress_callback(f" Filter Sky:  {f'Enabled (>{pfgs_sky_depth:.0f}m)' if pfgs_filter_sky else 'Disabled'}")
        progress_callback(f"======================================================================")

    # Ensure dataset has images extracted if raw video was supplied
    ensure_dataset_has_images(dataset_dir, fps=2.0, log_func=progress_callback)

    if engine_clean in ["lichtfeld", "lichtfeld_studio", "lfs"]:
        is_headless = (mode == "headless")
        return train_lichtfeld(
            dataset_dir=dataset_dir,
            output_dir=target_models_dir,
            iterations=iterations,
            headless=is_headless,
            gut=gut,
            sh_degree=sh_degree,
            export_spz=export_spz,
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
            model_type=ns_model,
            camera_optimizer=camera_optimizer,
            bilateral_grid=bilateral_grid,
            enable_mip=enable_mip,
            sh_degree=sh_degree,
            progress_callback=progress_callback
        )
    elif engine_clean in ["3dgrut", "3dgut", "grut"]:
        variant = representation if representation in ["3dgut", "3dgrut", "3dgrt"] else ("3dgut" if gut else "3dgrt")
        return train_3dgrut(
            dataset_dir=dataset_dir,
            output_dir=target_models_dir,
            variant=variant,
            bvh_backend=bvh_backend,
            iterations=iterations,
            with_viser_gui=(mode == "interactive"),
            exposure_correction=exposure_correction,
            enable_sparsity=enable_sparsity,
            sh_degree=sh_degree,
            progress_callback=progress_callback
        )
    elif engine_clean in ["pfgs360", "pfgs", "pfgs_360"]:
        return train_pfgs360(
            dataset_dir=dataset_dir,
            output_dir=target_models_dir,
            iterations=iterations,
            depth_type=pfgs_depth,
            dataparser_mode=pfgs_dataparser,
            spherical_aware=pfgs_spherical,
            consist_aware=pfgs_consist,
            inlier_growth=pfgs_inlier,
            align_depth=pfgs_align_depth,
            filter_sky=pfgs_filter_sky,
            sky_depth=pfgs_sky_depth,
            match_views=pfgs_match_views,
            densify_views=pfgs_densify_views,
            sh_degree=sh_degree,
            downscale_factor=pfgs_downscale,
            mode=mode,
            progress_callback=progress_callback
        )
    else:
        raise ValueError(f"Unknown 3DGS training engine: '{engine}'. Supported: 'lichtfeld', 'splatfacto', '3dgrut', 'pfgs360'")


# ----------------------------------------------------------------------
# 5. Standalone CLI Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Abu Align & Train - 3D Gaussian Splatting Training Engine CLI")
    parser.add_argument("--dataset", "-d", required=True, help="Path to dataset directory or 3dgs_dataset folder")
    parser.add_argument("--output", "-o", default=None, help="Destination directory for 3dgs_models")
    parser.add_argument("--engine", "-e", choices=["lichtfeld", "splatfacto", "3dgrut", "pfgs360"], default="lichtfeld", help="3DGS Training Engine")
    parser.add_argument("--iterations", "-i", type=int, default=30000, help="Max training iterations (default: 30000; quick test: 7000)")
    parser.add_argument("--mode", "-m", choices=["interactive", "headless"], default="interactive", help="Viewport mode")
    parser.add_argument("--gut", action="store_true", help="Enable Gaussian Unscented Transform (GUT)")
    parser.add_argument("--export_spz", action="store_true", default=False, help="Export compressed .spz format in LichtFeld Studio")
    parser.add_argument("--ns_model", choices=["splatfacto", "splatfacto-big"], default="splatfacto", help="Model variant for Nerfstudio (default: splatfacto)")
    parser.add_argument("--camera_optimizer", choices=["none", "so3_x_r3", "se3"], default="none", help="Camera optimizer for Nerfstudio (default: none)")
    parser.add_argument("--bilateral_grid", action="store_true", help="Enable bilateral grid color correction")
    parser.add_argument("--exposure_correction", action="store_true", help="Enable per-photo auto-exposure correction")
    parser.add_argument("--enable_sparsity", action="store_true", help="Enable ADMM sparsity pruning")
    parser.add_argument("--enable_mip", action="store_true", help="Enable anti-aliased mip-filtering")
    parser.add_argument("--representation", choices=["3dgut", "3dgrut"], default="3dgut", help="Representation mode for 3DGRUT")
    parser.add_argument("--bvh_backend", choices=["optix", "cuda"], default="optix", help="BVH backend for 3DGRUT")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")
    parser.add_argument("--pfgs_depth", choices=["unik3d", "midas", "zoedepth"], default="unik3d", help="Depth prior for PFGS360")
    parser.add_argument("--pfgs_dataparser", default="auto", help="Dataparser for PFGS360 (auto, opensfm, posefree, ob3d, odgs)")
    parser.add_argument("--pfgs_downscale", choices=["auto", "1", "2", "4"], default="1", help="Image downscale factor for PFGS360 ('1' for full res, '2' for half, 'auto' for auto <=1600px)")
    parser.add_argument("--pfgs_filter_sky", dest="pfgs_filter_sky", action="store_true", default=True, help="Filter sky in PFGS360")
    parser.add_argument("--no_pfgs_filter_sky", dest="pfgs_filter_sky", action="store_false", help="Disable sky filtering in PFGS360")
    parser.add_argument("--pfgs_sky_depth", type=float, default=50.0, help="Max depth threshold for sky filtering in meters (default: 50.0)")
    parser.add_argument("--pfgs_match_views", type=int, default=8, help="Sliding window size for PFGS360 pose matching (default: 8; 0 for all views)")
    parser.add_argument("--pfgs_densify_views", type=int, default=8, help="Sliding window size for PFGS360 densification (default: 8; 0 for all views)")
    parser.add_argument("--pfgs_spherical", dest="pfgs_spherical", action="store_true", default=True, help="Enable SCA-PE spherical coordinate positional encoding")
    parser.add_argument("--no_pfgs_spherical", dest="pfgs_spherical", action="store_false", help="Disable SCA-PE")
    parser.add_argument("--pfgs_consist", dest="pfgs_consist", action="store_true", default=True, help="Enable consist-aware epipolar masks")
    parser.add_argument("--no_pfgs_consist", dest="pfgs_consist", action="store_false", help="Disable consist-aware")
    parser.add_argument("--pfgs_inlier", dest="pfgs_inlier", action="store_true", default=True, help="Enable DIA-Densify depth inlier growth")
    parser.add_argument("--no_pfgs_inlier", dest="pfgs_inlier", action="store_false", help="Disable DIA-Densify")
    parser.add_argument("--pfgs_align_depth", dest="pfgs_align_depth", action="store_true", default=True, help="Align monocular depth with rendered internal depth")
    parser.add_argument("--no_pfgs_align_depth", dest="pfgs_align_depth", action="store_false", help="Disable align depth")

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
            export_spz=args.export_spz,
            ns_model=args.ns_model,
            camera_optimizer=args.camera_optimizer,
            bilateral_grid=args.bilateral_grid,
            exposure_correction=args.exposure_correction,
            enable_sparsity=args.enable_sparsity,
            enable_mip=args.enable_mip,
            representation=args.representation,
            bvh_backend=args.bvh_backend,
            pfgs_depth=args.pfgs_depth,
            pfgs_dataparser=args.pfgs_dataparser,
            pfgs_filter_sky=args.pfgs_filter_sky,
            pfgs_match_views=args.pfgs_match_views,
            pfgs_densify_views=args.pfgs_densify_views,
            pfgs_downscale=args.pfgs_downscale,
            progress_callback=print
        )
    except Exception as e:
        print(f"\n[CRITICAL ERROR] 3DGS Training failed: {e}")
        sys.exit(1)
