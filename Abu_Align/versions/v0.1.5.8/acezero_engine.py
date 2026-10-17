"""
ACE0 Engine Adapter for Abu Align
=================================
Integrates Niantic Labs' ACE0 (Scene Coordinate Regression Structure-from-Motion,
ECCV 2024 Oral) into the Abu Align pipeline.

Provides seamless execution of ACE0 mapping and relocalization, parses estimated
poses and dense point clouds, and builds a standard pycolmap.Reconstruction object
ready for multi-application export (COLMAP, Postshot/RealityCapture CSV, transforms.json).
"""

import os
import sys
import re
import time
import subprocess
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import pycolmap
import trimesh

try:
    import wsl_bridge
except ImportError:
    wsl_bridge = None

# Locate the ACE0 directory
CURRENT_DIR = Path(__file__).parent.resolve()
ACEZERO_DIR_CANDIDATES = [
    CURRENT_DIR / "acezero",
    CURRENT_DIR.parent / "acezero",
    CURRENT_DIR.parent.parent / "acezero",
]

ACEZERO_DIR = None
for candidate in ACEZERO_DIR_CANDIDATES:
    if candidate.is_dir() and (candidate / "ace_zero.py").is_file():
        ACEZERO_DIR = candidate.resolve()
        break

if ACEZERO_DIR is None:
    # Fallback to local junction/subfolder
    ACEZERO_DIR = CURRENT_DIR / "acezero"

if str(ACEZERO_DIR) not in sys.path:
    sys.path.insert(0, str(ACEZERO_DIR))


def find_acezero_script(script_name):
    script_path = ACEZERO_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"ACE0 script not found: {script_path}")
    return str(script_path)


def run_acezero_reconstruction(
    image_paths,
    output_dir,
    iterations_max=30,
    refinement="mlp",
    try_seeds=3,
    external_focal_length=-1.0,
    refine_calibration=True,
    image_resolution=480,
    export_point_cloud=True,
    dense_point_cloud=False,
    shared_camera=True,
    final_refit=True,
    final_refine=True,
    seed_parallel_workers=3,
    seed_iterations=3000,
    cooldown_iterations=1500,
    mapping_iterations=8000,
    refit_iterations=25000,
    final_refit_posewait=5000,
    seed_threshold_early_exit=0.75,
    stage_res="native",
    progress_callback=None,
):
    """
    Executes the full ACE0 Structure-from-Motion pipeline on a set of images.

    :param image_paths: List of file paths to the photos to align.
    :param output_dir: Destination directory for alignment results.
    :param iterations_max: Maximum rounds of alternating mapping and relocalization.
    :param refinement: Camera pose refinement strategy ('mlp', 'naive', 'none').
    :param try_seeds: Number of candidate seeds evaluated (3 is default; 3-5 for complex scenes).
    :param external_focal_length: Initial focal length guess (-1 to auto-estimate from image diagonal).
    :param refine_calibration: Whether to optimize focal length during mapping.
    :param image_resolution: Resolution for feature extraction / scene coordinate regression.
    :param export_point_cloud: Whether to generate the 3D scene point cloud.
    :param dense_point_cloud: If True, skips reprojection filtering (only for ultra-dense datasets >2000 photos).
                              If False (default), applies strict 1-3px reprojection filtering for clean surfaces.
    :param shared_camera: Whether all views share a single camera intrinsic parameter set.
    :param final_refit: Whether to run a 25k iteration refit from scratch after registration.
                       Defaults to True (official Niantic benchmark mode) for pristine global geometry.
    :param final_refine: Whether to execute an additional mapping round after 100% registration.
    :param seed_parallel_workers: Number of concurrent workers for candidate seed evaluation.
    :param seed_iterations: Training iterations per candidate seed (default: 3000 for fast convergence).
    :param cooldown_iterations: Training iterations during learning rate cooldown (default: 1500).
    :param mapping_iterations: Maximum training iterations per mapping round (default: 8000).
    :param refit_iterations: Training iterations for the final refit (default: 25000).
    :param final_refit_posewait: Iterations to keep camera poses frozen during refit (default: 5000).
    :param seed_threshold_early_exit: Registration ratio (0-1) to early-exit candidate seed search (default: 0.75).
    :param stage_res: Staging resolution for ACE0 ('native' for 100% full res, '2k' for 1920px max, '1k' for 1024px max).
    :param progress_callback: Optional callable(str) for real-time console updates.

    :return: (pycolmap.Reconstruction, np.ndarray (points_3d), np.ndarray (points_rgb))
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "acezero_work")
    os.makedirs(work_dir, exist_ok=True)

    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    if seed_parallel_workers is None or seed_parallel_workers <= 0:
        seed_parallel_workers = 2
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 20.0:
                    seed_parallel_workers = 3
                elif vram_gb >= 12.0:
                    seed_parallel_workers = 2
                else:
                    seed_parallel_workers = 1
        except Exception:
            seed_parallel_workers = 2

    log(f"      * Initializing ACE0 Engine (Niantic Labs Scene Coordinate Regression)...")
    log(f"      * ACE0 Root:           {ACEZERO_DIR}")
    log(f"      * Photos Count:        {len(image_paths)}")
    log(f"      * Seeds:               {try_seeds} (Early-exit threshold: {seed_threshold_early_exit*100:.0f}%)")
    log(f"      * Seed Iterations:     {seed_iterations} (Fast init)")
    log(f"      * Mapping Iterations:  {mapping_iterations} (Cooldown: {cooldown_iterations})")
    log(f"      * Point Cloud Filter:  {'Unfiltered (Dense)' if dense_point_cloud else 'Strict Reprojection Filtered (Crisp)'}")
    log(f"      * Fast SfM (No Refit): {not final_refit}")
    log(f"      * Work Dir:            {work_dir}")

    # Stage images: only pre-scale if explicitly requested by user (default: native original full res)
    stage_res_str = str(stage_res).strip().lower() if stage_res else "native"
    if any(k in stage_res_str for k in ["2k", "1920"]):
        max_stage_dim = 1920
        log(f"      * Staging Resolution:  2K Downscale (1920px Max)")
    elif any(k in stage_res_str for k in ["1k", "1024"]):
        max_stage_dim = 1024
        log(f"      * Staging Resolution:  1K Downscale (1024px Max)")
    elif any(k in stage_res_str for k in ["4k", "3840"]):
        max_stage_dim = 3840
        log(f"      * Staging Resolution:  4K Downscale (3840px Max)")
    else:
        max_stage_dim = None
        log(f"      * Staging Resolution:  Original Native (Full Res, Zero Downscaling)")

    stage_dir = Path(work_dir) / "staged_images"
    os.makedirs(stage_dir, exist_ok=True)
    staged_paths = []
    source_mapping = {}
    scale_factors = {}

    for idx, src_path in enumerate(image_paths):
        ext = Path(src_path).suffix.lower()
        clean_stem = re.sub(r'[^\w\-]', '_', Path(src_path).stem)
        dst_name = f"{idx:05d}_{clean_stem}.jpg"
        dst_path = stage_dir / dst_name
        needs_scale = False
        scale_val = 1.0
        if max_stage_dim is not None and max_stage_dim > 0:
            try:
                from PIL import Image
                with Image.open(src_path) as im:
                    w, h = im.size
                    if max(w, h) > max_stage_dim:
                        needs_scale = True
                        scale_val = max_stage_dim / max(w, h)
                        if not dst_path.exists():
                            new_w, new_h = int(w * scale_val), int(h * scale_val)
                            im_resized = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                            im_resized.save(dst_path, "JPEG", quality=95)
            except Exception:
                needs_scale = False
                scale_val = 1.0

        if not needs_scale and not dst_path.exists():
            try:
                os.link(src_path, dst_path)
            except Exception:
                import shutil
                shutil.copy2(src_path, dst_path)

        staged_paths.append(str(dst_path))
        source_mapping[str(dst_path)] = src_path
        scale_factors[src_path] = scale_val
        scale_factors[str(Path(src_path).resolve())] = scale_val
        scale_factors[Path(src_path).name] = scale_val

    input_target = str(stage_dir)
    ace_zero_script = find_acezero_script("ace_zero.py")
    def handle_ace_line(raw_line, prefix="[ACE0-WSL]"):
        line_str = raw_line.strip()
        if not line_str:
            return
        log(f"      {prefix} {line_str}")

        # Live structured telemetry emission for GUI & CLI tracking
        if "Processing seed " in line_str:
            m = re.search(r"Processing seed\s+(\d+)", line_str)
            if m:
                s_idx = int(m.group(1))
                log(f"[TELEMETRY:STAGE2] status=evaluating_seed seed={s_idx+1} total={try_seeds}")
        elif "Early-stopping seed" in line_str or "Selected best seed" in line_str:
            log(f"[TELEMETRY:STAGE2] status=seed_selected")
        elif "Iteration:" in line_str and "|" in line_str:
            m = re.search(r"Iteration:\s*(\d+)\s*\|\s*(\d+).*Loss:\s*([\d\.]+).*inliers.*:\s*([\d\.]+)%", line_str)
            if m:
                cur_it = int(m.group(1))
                tot_it = int(m.group(2))
                loss_v = float(m.group(3))
                inl_v = float(m.group(4))
                pct = (cur_it / max(1, tot_it)) * 100.0
                log(f"[TELEMETRY:STAGE3] pct={pct:.1f} iter={cur_it} total={tot_it} loss={loss_v} inliers={inl_v}")
        elif "Current Focal Length:" in line_str:
            m = re.search(r"Current Focal Length:\s*([\d\.]+)", line_str)
            if m:
                f_val = float(m.group(1))
                log(f"[TELEMETRY:ACE_OPTICS] focal={f_val:.1f}")
        elif "Successfully registered" in line_str and "mapping images" in line_str:
            m = re.search(r"Successfully registered\s*([\d\.]+)%", line_str)
            if m:
                reg_p = float(m.group(1))
                log(f"[TELEMETRY:STAGE4] pct={reg_p:.1f} status=registered")

    if seed_parallel_workers is None or seed_parallel_workers <= 0:
        seed_parallel_workers = 3

    use_wsl = wsl_bridge is not None and wsl_bridge.is_abu_linux_env_ready()
    if use_wsl:
        log("      [*] Engaging Native Linux Niantic ACE0 via WSL2 Ubuntu (GCC + OpenMP C++ DSAC*)...")
        wsl_bridge.ensure_wsl_packages(["roma", "pycolmap", "timm==0.6.7"], log_callback=log)
        wsl_input = wsl_bridge.win_to_wsl_path(input_target)
        wsl_work = wsl_bridge.win_to_wsl_path(work_dir)
        wsl_acezero_dir = wsl_bridge.win_to_wsl_path(str(ACEZERO_DIR))

        bash_cmd = (
            f"export PYTHONIOENCODING=utf-8 && "
            f"export LC_ALL=C.UTF-8 && "
            f"export LANG=C.UTF-8 && "
            f"export PYTHONPATH='{wsl_acezero_dir}:{wsl_acezero_dir}/dsacstar:$PYTHONPATH' && "
            f"export LD_LIBRARY_PATH='{wsl_bridge.WSL_CONDA_ENV}/lib/python3.10/site-packages/torch/lib:{wsl_bridge.WSL_CONDA_ENV}/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' && "
            f"cd '{wsl_acezero_dir}' && "
            f"{wsl_bridge.WSL_PYTHON} -u '{wsl_acezero_dir}/ace_zero.py' '{wsl_input}' '{wsl_work}' "
            f"--iterations_max {iterations_max} "
            f"--refinement {refinement} "
            f"--refine_calibration {refine_calibration} "
            f"--use_external_focal_length {external_focal_length} "
            f"--image_resolution {image_resolution} "
            f"--export_point_cloud {export_point_cloud} "
            f"--dense_point_cloud {dense_point_cloud} "
            f"--try_seeds {try_seeds} "
            f"--seed_parallel_workers {seed_parallel_workers} "
            f"--seed_iterations {seed_iterations} "
            f"--seed_threshold_early_exit {seed_threshold_early_exit} "
            f"--cooldown_iterations {cooldown_iterations} "
            f"--mapping_iterations {mapping_iterations} "
            f"--refit_iterations {refit_iterations} "
            f"--final_refit_posewait {final_refit_posewait} "
            f"--final_refit {final_refit} "
            f"--final_refine {final_refine} "
            f"--render_visualization False"
        )
        log(f"      * Executing ACE0 via WSL2 Ubuntu...")
        ret_code, _ = wsl_bridge.run_command_in_wsl(bash_cmd, log_callback=lambda l: handle_ace_line(l, "[ACE0-WSL]"))
        if ret_code != 0:
            raise RuntimeError(f"ACE0 mapping process exited with error code {ret_code} in WSL2 Ubuntu")
    else:
        log("      [*] Engaging Native Windows ACE0 Engine (Fallback)...")
        cmd = [
            sys.executable,
            "-u",
            ace_zero_script,
            input_target,
            work_dir,
            "--iterations_max", str(iterations_max),
            "--refinement", str(refinement),
            "--refine_calibration", str(refine_calibration),
            "--use_external_focal_length", str(external_focal_length),
            "--image_resolution", str(image_resolution),
            "--export_point_cloud", str(export_point_cloud),
            "--dense_point_cloud", str(dense_point_cloud),
            "--try_seeds", str(try_seeds),
            "--seed_parallel_workers", str(seed_parallel_workers),
            "--seed_iterations", str(seed_iterations),
            "--seed_threshold_early_exit", str(seed_threshold_early_exit),
            "--cooldown_iterations", str(cooldown_iterations),
            "--mapping_iterations", str(mapping_iterations),
            "--refit_iterations", str(refit_iterations),
            "--final_refit_posewait", str(final_refit_posewait),
            "--final_refit", str(final_refit),
            "--final_refine", str(final_refine),
            "--render_visualization", "False",
        ]

        log(f"      * Executing ACE0 mapping loop: {' '.join(cmd)}")
        win_env = os.environ.copy()
        win_env["PYTHONIOENCODING"] = "utf-8"
        win_env["LC_ALL"] = "C.UTF-8"
        win_env["LANG"] = "C.UTF-8"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(ACEZERO_DIR),
            env=win_env
        )

        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                handle_ace_line(line, "[ACE0]")

        ret_code = proc.wait()
        if ret_code != 0:
            raise RuntimeError(f"ACE0 mapping process exited with error code {ret_code}")


    log(f"      * ACE0 mapping completed in: {time.time()-start_time:.2f}s")

    # Locate poses_final.txt
    poses_final_path = Path(work_dir) / "poses_final.txt"
    if not poses_final_path.is_file():
        # Fallback: check latest iteration pose file
        pose_files = sorted(Path(work_dir).glob("poses_iteration*.txt"))
        if pose_files:
            poses_final_path = pose_files[-1]
        else:
            raise FileNotFoundError(f"No ACE0 pose files generated in: {work_dir}")

    log(f"      * Reading final camera poses from: {poses_final_path.name}")
    with open(poses_final_path, "r", encoding="utf-8") as f:
        pose_lines = [l.strip() for l in f if l.strip()]

    log(f"      * Registered {len(pose_lines)} views in ACE0 scene.")

    # Locate point cloud file
    pc_ply_path = Path(work_dir) / "pc_final.ply"
    points_3d = np.zeros((0, 3), dtype=np.float32)
    points_rgb = np.zeros((0, 3), dtype=np.uint8)

    if pc_ply_path.is_file():
        try:
            mesh = trimesh.load(str(pc_ply_path))
            if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                points_3d = np.asarray(mesh.vertices, dtype=np.float32)
                if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
                    points_rgb = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
                else:
                    points_rgb = np.full((len(points_3d), 3), 200, dtype=np.uint8)
                log(f"      * Loaded {len(points_3d):,} 3D scene coordinates from {pc_ply_path.name}")
        except Exception as e:
            log(f"      * [WARNING] Could not parse pc_final.ply: {e}")

    # Build pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Read original dimensions of photos
    orig_dims = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                orig_dims[os.path.basename(p)] = (img.width, img.height)
        except Exception:
            orig_dims[os.path.basename(p)] = (1024, 768)

    # In ACE0 poses_final.txt:
    # filename qw qx qy qz x y z focal_length confidence
    registered_cameras = {}
    first_focal = None

    # First pass: find dominant focal length
    for line in pose_lines:
        tokens = line.split()
        if len(tokens) >= 10:
            first_focal = float(tokens[-2])
            break
    if first_focal is None or first_focal <= 1e-4:
        first_focal = 500.0

    camera_id_seq = 1
    shared_camera_id = None

    if shared_camera:
        sample_w, sample_h = next(iter(orig_dims.values()))
        sample_src = next(iter(image_paths))
        sample_scale = scale_factors.get(str(Path(sample_src).resolve()), scale_factors.get(sample_src, scale_factors.get(Path(sample_src).name, 1.0)))
        unscaled_sample_focal = first_focal / sample_scale if sample_scale > 0 else first_focal
        cx, cy = sample_w / 2.0, sample_h / 2.0
        # Shared SIMPLE_PINHOLE (f, cx, cy) or PINHOLE (fx, fy, cx, cy)
        cam = pycolmap.Camera(
            model="PINHOLE",
            width=int(sample_w),
            height=int(sample_h),
            params=[unscaled_sample_focal, unscaled_sample_focal, cx, cy]
        )
        cam.camera_id = camera_id_seq
        if hasattr(reconstruction, "add_camera_with_trivial_rig"):
            reconstruction.add_camera_with_trivial_rig(cam)
        else:
            reconstruction.add_camera(cam)
        shared_camera_id = cam.camera_id
        camera_id_seq += 1

    img_id_seq = 1
    for line in pose_lines:
        tokens = line.split()
        if len(tokens) < 10:
            continue

        raw_img_path = " ".join(tokens[:-9])
        # Resolve original image name
        original_source_path = None
        for staged_p, orig_p in source_mapping.items():
            if Path(raw_img_path).name == Path(staged_p).name or raw_img_path in staged_p:
                original_source_path = orig_p
                break
        if original_source_path is None:
            # Match by basename
            for orig_p in image_paths:
                if Path(orig_p).name in raw_img_path or raw_img_path in orig_p:
                    original_source_path = orig_p
                    break

        if original_source_path is None:
            original_source_path = raw_img_path

        base_name = os.path.basename(original_source_path)
        img_w, img_h = orig_dims.get(base_name, (1024, 768))

        qw = float(tokens[-9])
        qx = float(tokens[-8])
        qy = float(tokens[-7])
        qz = float(tokens[-6])
        tx = float(tokens[-5])
        ty = float(tokens[-4])
        tz = float(tokens[-3])
        raw_focal = float(tokens[-2]) if float(tokens[-2]) > 1e-4 else first_focal
        confidence = float(tokens[-1])

        # Adjust focal length back to full unscaled original image resolution
        img_scale = scale_factors.get(str(Path(original_source_path).resolve()), scale_factors.get(original_source_path, scale_factors.get(base_name, 1.0)))
        unscaled_focal = raw_focal / img_scale if img_scale > 0 else raw_focal

        if shared_camera:
            cam_id = shared_camera_id
        else:
            cx, cy = img_w / 2.0, img_h / 2.0
            cam = pycolmap.Camera(
                model="PINHOLE",
                width=int(img_w),
                height=int(img_h),
                params=[unscaled_focal, unscaled_focal, cx, cy]
            )
            cam.camera_id = camera_id_seq
            if hasattr(reconstruction, "add_camera_with_trivial_rig"):
                reconstruction.add_camera_with_trivial_rig(cam)
            else:
                reconstruction.add_camera(cam)
            cam_id = cam.camera_id
            camera_id_seq += 1

        # ACE0 quaternion format is qw, qx, qy, qz (OpenCV world-to-cam)
        # pycolmap.Rotation3d accepts 3x3 matrix or [qx, qy, qz, qw]
        from scipy.spatial.transform import Rotation as SciPyRot
        R_w2c = SciPyRot.from_quat([qx, qy, qz, qw]).as_matrix()
        quat = pycolmap.Rotation3d(R_w2c)
        trans = np.array([tx, ty, tz])
        rigid = pycolmap.Rigid3d(quat, trans)

        py_img = pycolmap.Image(name=base_name, camera_id=cam_id, image_id=img_id_seq)
        if hasattr(reconstruction, "add_image_with_trivial_frame"):
            reconstruction.add_image_with_trivial_frame(py_img, rigid)
        else:
            py_img.cam_from_world = rigid
            py_img.registered = True
            reconstruction.add_image(py_img)

        img_id_seq += 1

    # Add points3D into reconstruction
    if len(points_3d) > 0:
        log(f"      * Incorporating {len(points_3d):,} 3D tie points into COLMAP reconstruction...")
        step = 1  # Preserve all valid authentic points from ACE0 for 3DGS training & visualization
        for i in range(0, len(points_3d), step):
            xyz = np.asarray(points_3d[i], dtype=np.float64)
            rgb = np.asarray(points_rgb[i] if len(points_rgb) > i else [200, 200, 200], dtype=np.uint8)
            reconstruction.add_point3D(xyz, pycolmap.Track(), rgb)

    log(f"      * Successfully built pycolmap.Reconstruction with {reconstruction.num_reg_images()} images and {reconstruction.num_points3D():,} points.")
    return reconstruction, points_3d, points_rgb
