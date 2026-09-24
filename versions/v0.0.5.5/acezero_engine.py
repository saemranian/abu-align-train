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
    progress_callback=None,
):
    """
    Executes the full ACE0 Structure-from-Motion pipeline on a set of images.

    :param image_paths: List of file paths to the photos to align.
    :param output_dir: Destination directory for alignment results.
    :param iterations_max: Maximum rounds of alternating mapping and relocalization.
    :param refinement: Camera pose refinement strategy ('mlp', 'naive', 'none').
    :param external_focal_length: Initial focal length guess (-1 to auto-estimate from image diagonal).
    :param refine_calibration: Whether to optimize focal length during mapping.
    :param image_resolution: Resolution for feature extraction / scene coordinate regression.
    :param export_point_cloud: Whether to generate the 3D scene point cloud.
    :param dense_point_cloud: If True, skips reprojection filtering for maximum point density.
    :param shared_camera: Whether all views share a single camera intrinsic parameter set.
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

    log(f"      * Initializing ACE0 Engine (Niantic Labs Scene Coordinate Regression)...")
    log(f"      * ACE0 Root:     {ACEZERO_DIR}")
    log(f"      * Photos Count:  {len(image_paths)}")
    log(f"      * Work Dir:      {work_dir}")

    # Stage images: if images exceed 1920px (e.g. 60.2MP raw exports), pre-scale once
    # to avoid massive CPU RAM pressure (57GB) and slow I/O decoding on every epoch.
    stage_dir = Path(work_dir) / "staged_images"
    os.makedirs(stage_dir, exist_ok=True)
    staged_paths = []
    source_mapping = {}
    max_stage_dim = 1920

    for idx, src_path in enumerate(image_paths):
        ext = Path(src_path).suffix.lower()
        dst_name = f"{idx:05d}_{Path(src_path).stem}.jpg"
        dst_path = stage_dir / dst_name
        if not dst_path.exists():
            needs_scale = False
            try:
                from PIL import Image
                with Image.open(src_path) as im:
                    w, h = im.size
                    if max(w, h) > max_stage_dim:
                        needs_scale = True
                        scale = max_stage_dim / max(w, h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        im_resized = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                        im_resized.save(dst_path, "JPEG", quality=95)
            except Exception:
                needs_scale = False

            if not needs_scale and not dst_path.exists():
                try:
                    os.link(src_path, dst_path)
                except Exception:
                    import shutil
                    shutil.copy2(src_path, dst_path)

        staged_paths.append(str(dst_path))
        source_mapping[str(dst_path)] = src_path

    input_target = str(stage_dir)
    ace_zero_script = find_acezero_script("ace_zero.py")

    use_wsl = wsl_bridge is not None and wsl_bridge.is_abu_linux_env_ready()
    if use_wsl:
        log("      [*] Engaging Native Linux Niantic ACE0 via WSL2 Ubuntu (GCC + OpenMP C++ DSAC*)...")
        wsl_bridge.ensure_wsl_packages(["roma", "pycolmap", "timm==0.6.7"], log_callback=log)
        wsl_input = wsl_bridge.win_to_wsl_path(input_target)
        wsl_work = wsl_bridge.win_to_wsl_path(work_dir)
        wsl_acezero_dir = wsl_bridge.win_to_wsl_path(str(ACEZERO_DIR))

        bash_cmd = (
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
            f"--seed_parallel_workers 1 "
            f"--render_visualization False"
        )
        log(f"      * Executing ACE0 via WSL2 Ubuntu...")
        ret_code, _ = wsl_bridge.run_command_in_wsl(bash_cmd, log_callback=lambda l: log(f"      [ACE0-WSL] {l}"))
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
            "--seed_parallel_workers", "1",
            "--render_visualization", "False",
        ]

        log(f"      * Executing ACE0 mapping loop: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ACEZERO_DIR)
        )

        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                stripped = line.rstrip()
                if stripped:
                    log(f"      [ACE0] {stripped}")

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
        cx, cy = sample_w / 2.0, sample_h / 2.0
        # Shared SIMPLE_PINHOLE (f, cx, cy) or PINHOLE (fx, fy, cx, cy)
        cam = pycolmap.Camera(
            model="PINHOLE",
            width=int(sample_w),
            height=int(sample_h),
            params=[first_focal, first_focal, cx, cy]
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
        focal = float(tokens[-2]) if float(tokens[-2]) > 1e-4 else first_focal
        confidence = float(tokens[-1])

        if shared_camera:
            cam_id = shared_camera_id
        else:
            cx, cy = img_w / 2.0, img_h / 2.0
            cam = pycolmap.Camera(
                model="PINHOLE",
                width=int(img_w),
                height=int(img_h),
                params=[focal, focal, cx, cy]
            )
            cam.camera_id = camera_id_seq
            if hasattr(reconstruction, "add_camera_with_trivial_rig"):
                reconstruction.add_camera_with_trivial_rig(cam)
            else:
                reconstruction.add_camera(cam)
            cam_id = cam.camera_id
            camera_id_seq += 1

        # ACE0 quaternion format is qw, qx, qy, qz (OpenCV world-to-cam)
        # pycolmap.Rotation3d accepts [qw, qx, qy, qz]
        quat = pycolmap.Rotation3d(np.array([qw, qx, qy, qz]))
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
        step = max(1, len(points_3d) // 50000)  # Sparse representation in COLMAP
        for i in range(0, len(points_3d), step):
            xyz = np.asarray(points_3d[i], dtype=np.float64)
            rgb = np.asarray(points_rgb[i] if len(points_rgb) > i else [200, 200, 200], dtype=np.uint8)
            reconstruction.add_point3D(xyz, pycolmap.Track(), rgb)

    log(f"      * Successfully built pycolmap.Reconstruction with {reconstruction.num_reg_images()} images and {reconstruction.num_points3D():,} points.")
    return reconstruction, points_3d, points_rgb
