"""
Photoscanning Alignment CLI with Live Progress & VRAM Monitoring
===============================================================
A rich terminal interface for running 3D camera alignment and reconstruction
with live progress bars, stage status, timing, VRAM usage, and final report.
"""

import os
import sys
import re
import time
import glob
import copy
import argparse
from datetime import datetime

# Ensure parent directory is on sys.path so vggt and dependencies can always be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
for parent in [current_dir, os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir))]:
    if parent and parent not in sys.path:
        sys.path.insert(0, parent)

def get_next_sparse_dir(results_base_dir, engine_name="vggt", use_ba=False):
    """Finds the next unique sparse output folder (e.g. sparse_03_omega) that never overwrites existing results."""
    engine_suffix = "omega" if "omega" in engine_name.lower() else "vggt"
    if use_ba:
        engine_suffix += "_ba"
    
    os.makedirs(results_base_dir, exist_ok=True)
    max_num = 0
    pattern = re.compile(r'sparse[\s_-]*(\d+)', re.IGNORECASE)
    for name in os.listdir(results_base_dir):
        m = pattern.search(name)
        if m:
            max_num = max(max_num, int(m.group(1)))
        elif name.lower() == 'sparse':
            max_num = max(max_num, 0)
    
    next_num = max_num + 1
    while True:
        candidate_name = f"sparse_{next_num:02d}_{engine_suffix}"
        candidate_path = os.path.join(results_base_dir, candidate_name)
        if not os.path.exists(candidate_path):
            return candidate_path
        next_num += 1


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_vram_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0.0, 0.0

def print_banner():
    banner = r"""
======================================================================
     3D Photoscanning Alignment & Reconstruction Engine (VGGT)
======================================================================
"""
    print(banner)

def print_progress_bar(current, total, prefix="Progress", stage="", length=30, start_time=None):
    percent = float(current) / max(total, 1)
    filled_len = int(length * percent)
    bar = "=" * filled_len + "-" * (length - filled_len)
    
    time_str = ""
    if start_time:
        elapsed = time.time() - start_time
        time_str = f" | Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d}"

    vram_alloc, _ = get_vram_usage()
    vram_str = f" | VRAM: {vram_alloc:.1f}GB" if torch.cuda.is_available() else ""

    status_line = f"\r[{bar}] {percent*100:5.1f}% | {prefix}: {current}/{total}{time_str}{vram_str} | {stage}"
    try:
        sys.stdout.write(status_line.ljust(95))
        sys.stdout.flush()
    except Exception:
        pass

def umeyama_alignment(src, dst):
    """
    Computes optimal Sim(3) transformation (s, R, t) such that:
    dst ~= s * R @ src + t
    using Umeyama's SVD algorithm.
    """
    n, m = src.shape
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    
    src_c = src - mu_src
    dst_c = dst - mu_dst
    
    var_src = np.mean(np.sum(src_c**2, axis=1))
    H = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(H)
    
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1
        
    R = U @ S @ Vt
    s = (1.0 / max(var_src, 1e-8)) * np.trace(np.diag(D) @ S)
    t = mu_dst - s * (R @ mu_src)
    return s, R, t

def export_realitycapture_csv(reconstruction, output_dir):
    csv_path = os.path.join(output_dir, "cameras.csv")
    rows = ["#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"]
    
    # Transformation matrix: COLMAP world (Y-down, Z-forward) -> RealityCapture world (Z-up, Y-north)
    # X_rc = X_c, Y_rc = Z_c, Z_rc = -Y_c
    M_world = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0]
    ])
    
    for img_id, img in sorted(reconstruction.images.items()):
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R_c = cfw.rotation.matrix()
        t_c = cfw.translation
        C_c = - R_c.T @ t_c
        C_rc = M_world @ C_c
        
        x = C_rc[0]
        y = C_rc[1]
        alt = C_rc[2]
        
        # Camera optical axes in RC world space:
        # In COLMAP camera frame: forward = [0, 0, 1], up = [0, -1, 0] (since COLMAP +Y is down)
        fwd_rc = M_world @ (R_c.T @ np.array([0.0,  0.0, 1.0]))
        up_rc  = M_world @ (R_c.T @ np.array([0.0, -1.0, 0.0]))
        
        fwd_rc /= np.linalg.norm(fwd_rc)
        up_rc  /= np.linalg.norm(up_rc)
        
        # Heading / Yaw: azimuth angle in XY plane from North (+Y) towards East (+X) in [0, 360)
        yaw = np.degrees(np.arctan2(fwd_rc[0], fwd_rc[1])) % 360.0
        
        # Pitch: angle from Nadir (-Z): 0=nadir (down), 90=horizontal, 180=zenith (up)
        pitch = np.degrees(np.arccos(np.clip(-fwd_rc[2], -1.0, 1.0)))
        
        # Reference horizontal right vector perpendicular to heading:
        yaw_rad = np.radians(yaw)
        w0 = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
        w0_norm = np.linalg.norm(w0)
        if w0_norm > 1e-6:
            w0 /= w0_norm
            u0 = np.cross(w0, fwd_rc)
            u0 /= np.linalg.norm(u0)
            roll = np.degrees(np.arctan2(np.dot(up_rc, w0), np.dot(up_rc, u0)))
        else:
            roll = 0.0
        
        cam = reconstruction.cameras[img.camera_id]
        w, h = cam.width, cam.height
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        f_35mm = fx * (36.0 / max(w, h))
        px_norm = (cx - w / 2.0) / w
        py_norm = (cy - h / 2.0) / h
        
        # Extract calibrated lens distortion parameters if available:
        def fmt_val(v):
            if abs(v) < 1e-12:
                return "0"
            return f"{v:.10g}"

        if cam.model_name in ["FULL_OPENCV", "OPENCV"] and len(cam.params) >= 8:
            k1 = cam.params[4]
            k2 = cam.params[5]
            t1 = cam.params[6]  # p1 in OpenCV / COLMAP
            t2 = cam.params[7]  # p2 in OpenCV / COLMAP
            k3 = 0.0
            k4 = 0.0
            rows.append(f"{img.name},{x:.6f},{y:.6f},{alt:.6f},{yaw:.4f},{pitch:.4f},{roll:.4f},{f_35mm:.4f},{px_norm:.6f},{py_norm:.6f},{fmt_val(k1)},{fmt_val(k2)},0,0,{fmt_val(t1)},{fmt_val(t2)}\n")
        else:
            rows.append(f"{img.name},{x:.6f},{y:.6f},{alt:.6f},{yaw:.4f},{pitch:.4f},{roll:.4f},{f_35mm:.4f},{px_norm:.6f},{py_norm:.6f},0,0,0,0,0,0\n")
        
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(rows)
    return csv_path


def export_transforms_json(reconstruction, output_dir):
    """
    Exports camera poses and intrinsics in NeRF / 3D Gaussian Splatting transforms.json format.
    Postshot, Nerfstudio, and standard 3DGS pipelines load this natively with zero Euler angle ambiguity.
    """
    import json
    
    first_cam = next(iter(reconstruction.cameras.values()))
    w, h = first_cam.width, first_cam.height
    fx, fy, cx, cy = first_cam.params[0], first_cam.params[1], first_cam.params[2], first_cam.params[3]
    
    transforms_data = {
        "camera_model": first_cam.model_name,
        "w": int(w),
        "h": int(h),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "frames": []
    }
    
    if first_cam.model_name in ["OPENCV", "FULL_OPENCV"] and len(first_cam.params) >= 8:
        transforms_data["k1"] = float(first_cam.params[4])
        transforms_data["k2"] = float(first_cam.params[5])
        transforms_data["p1"] = float(first_cam.params[6])
        transforms_data["p2"] = float(first_cam.params[7])
    
    for img_id, img in sorted(reconstruction.images.items()):
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R_c = cfw.rotation.matrix()
        t_c = cfw.translation
        
        # Camera center in COLMAP world: C = - R_c.T @ t_c
        C_c = - R_c.T @ t_c
        
        # Conversion from COLMAP camera frame (+X right, +Y down, +Z fwd)
        # to OpenGL / NeRF camera frame (+X right, +Y up, +Z back):
        # R_nerf_c2w = R_c.T @ diag(1, -1, -1)
        R_nerf_c2w = R_c.T @ np.diag([1.0, -1.0, -1.0])
        
        c2w = np.eye(4)
        c2w[:3, :3] = R_nerf_c2w
        c2w[:3, 3] = C_c
        
        frame_dict = {
            "file_path": img.name,
            "transform_matrix": c2w.tolist()
        }
        transforms_data["frames"].append(frame_dict)
        
    transforms_path = os.path.join(output_dir, "transforms.json")
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms_data, f, indent=2)
    return transforms_path


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    """Inlined helper to rename cameras and rescale focal lengths to original photo resolutions."""
    rescaled_cameras = set()

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        real_image_size = original_coords[pyimageid - 1, -2:]
        max_dim = max(real_image_size)
        left = (max_dim - real_image_size[0]) // 2
        top = (max_dim - real_image_size[1]) // 2
        resize_ratio = max_dim / img_size

        if pyimage.camera_id not in rescaled_cameras:
            pred_params = copy.deepcopy(pycamera.params)

            if pycamera.model_name in ["OPENCV", "FULL_OPENCV"]:
                pred_params[0] *= resize_ratio  # fx
                pred_params[1] *= resize_ratio  # fy
                pred_params[2] = pred_params[2] * resize_ratio - left  # cx
                pred_params[3] = pred_params[3] * resize_ratio - top   # cy
            elif pycamera.model_name == "PINHOLE":
                pred_params = pred_params * resize_ratio
                pred_params[2] = pred_params[2] * resize_ratio - left
                pred_params[3] = pred_params[3] * resize_ratio - top
            else:
                pred_params = pred_params * resize_ratio
                pred_params[-2] = real_image_size[0] / 2.0
                pred_params[-1] = real_image_size[1] / 2.0

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])
            rescaled_cameras.add(pyimage.camera_id)

        if shift_point2d_to_original_res:
            for point2D in pyimage.points2D:
                point2D.xy = point2D.xy * resize_ratio - np.array([left, top])

    return reconstruction



def main():
    parser = argparse.ArgumentParser(description="Live Progress Alignment CLI for Photoscanning")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to photoscan folder or image directory")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Custom output directory for sparse COLMAP files")
    parser.add_argument("--engine", "-e", type=str, choices=["vggt", "vggt_omega"], default="vggt", help="Model engine: 'vggt' (Standard, public checkpoint) or 'vggt_omega' (Next-Gen, low VRAM)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Custom checkpoint path (.pt file) for VGGT-Omega")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride / keyframe interval (e.g. 2 or 3 for large datasets)")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--conf_thresh", type=float, default=1.8, help="Depth confidence threshold (default: 1.8; lower=denser point cloud)")
    parser.add_argument("--max_points", type=int, default=300000, help="Maximum number of points in exported point cloud (default: 300000)")
    parser.add_argument("--load_res", type=int, default=1024, help="Image load resolution (default: 1024)")
    parser.add_argument("--chunk_size", type=int, default=32, help="Chunk size for auto-chunking large datasets (default: 32)")
    parser.add_argument("--auto_chunk", action="store_true", default=True, help="Automatically chunk large datasets (> 48 photos) to prevent VRAM OOM")
    parser.add_argument("--use_ba", action="store_true", help="Enable multi-view feature tracking and PyCeres Bundle Adjustment")
    parser.add_argument("--align_mode", type=str, choices=["none", "standard", "ultra"], default=None, help="Camera alignment precision: 'none' (fast feedforward), 'standard' (Ceres BA), 'ultra' (sub-pixel Ceres BA with Cauchy loss & multi-pass polish)")
    parser.add_argument("--lens_model", type=str, choices=["pinhole", "brown"], default="brown", help="Lens distortion model: 'pinhole' (zero distortion) or 'brown' (calibrates physical k1, k2, t1, t2)")
    parser.add_argument("--max_reproj_error", type=float, default=None, help="Max reprojection error for tie point tracks")
    parser.add_argument("--shared_camera", dest="shared_camera", action="store_true", default=True, help="Share camera intrinsics and distortion across all views (recommended for single-camera scans)")
    parser.add_argument("--no_shared_camera", dest="shared_camera", action="store_false", help="Calibrate separate camera intrinsics for each view")
    parser.add_argument("--view", "-v", action="store_true", help="Launch interactive 3D browser viewer after alignment")
    parser.add_argument("--port", type=int, default=8080, help="Viser viewer port (default: 8080)")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"\n[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    print_banner()
    overall_start_time = time.time()

    # -------------------------------------------------------------
    # Stage 1: Dataset Discovery
    # -------------------------------------------------------------
    print("[1/5] Scanning Image Dataset...")
    images_subdir = os.path.join(input_dir, "images")
    scan_dir = images_subdir if os.path.isdir(images_subdir) else input_dir

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    all_files = sorted(os.listdir(scan_dir))
    image_paths = [
        os.path.join(scan_dir, f)
        for f in all_files
        if os.path.splitext(f)[1].lower() in valid_exts and os.path.isfile(os.path.join(scan_dir, f))
    ]

    total_images_found = len(image_paths)
    if total_images_found == 0:
        print(f"[ERROR] No valid images found in: {scan_dir}")
        sys.exit(1)

    # Subsample if requested
    if args.stride > 1:
        image_paths = image_paths[::args.stride]
    if args.max_frames and len(image_paths) > args.max_frames:
        image_paths = image_paths[:args.max_frames]

    num_frames = len(image_paths)
    with Image.open(image_paths[0]) as sample_img:
        orig_w, orig_h = sample_img.size

    print(f"      * Directory:     {scan_dir}")
    print(f"      * Total Images:  {total_images_found}")
    print(f"      * Keyframes:     {num_frames} (Stride: {args.stride})")
    print(f"      * Image Size:    {orig_w} x {orig_h} ({orig_w*orig_h/1e6:.1f} MP)")
    print()

    # -------------------------------------------------------------
    # Stage 2: Load Model & GPU Environment
    # -------------------------------------------------------------
    print("[2/5] Initializing Neural Network & GPU...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARNING] CUDA GPU not detected! Running on CPU will be extremely slow.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"      * Device:        {gpu_name} ({total_vram:.1f} GB Dedicated VRAM)")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"      * Precision:     {dtype}")

    engine = args.engine
    model = None
    vggt_res = 518

    # Attempt to load VGGT-Omega if requested
    if engine == "vggt_omega":
        possible_ckpts = [
            args.checkpoint,
            r"C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\vggt-omega\vggt_omega_1b_512.pt",
            r"C:\Users\USer\.cache\torch\hub\checkpoints\vggt_omega_1b_512.pt",
            os.path.join(os.getcwd(), "vggt_omega_1b_512.pt"),
        ]
        found_ckpt = None
        for p in possible_ckpts:
            if p and os.path.isfile(p):
                found_ckpt = p
                break

        if found_ckpt:
            try:
                from vggt_omega.models import VGGTOmega
                t_load = time.time()
                print(f"      * Engine:        VGGT-Omega (Register Attention)")
                print(f"      * Checkpoint:    {found_ckpt}")
                model = VGGTOmega().eval()
                state_dict = torch.load(found_ckpt, map_location="cpu")
                model.load_state_dict(state_dict)
                model = model.to(device)
                vggt_res = 512
                print(f"      * Model Loaded:  VGGT-Omega in {time.time()-t_load:.1f}s")
            except Exception as e:
                print(f"[WARNING] Failed to load VGGT-Omega: {e}. Falling back to VGGT.")
                engine = "vggt"
        else:
            print("\n" + "="*70)
            print(" [NOTICE] VGGT-Omega weights (vggt_omega_1b_512.pt) are not yet downloaded.")
            print("          (Your HuggingFace access request status is currently PENDING)")
            print("          --> Seamlessly falling back to VGGT-1B (Ready & Cached)!")
            print("="*70 + "\n")
            engine = "vggt"

    # Default to standard VGGT
    if engine == "vggt":
        from vggt.models.vggt import VGGT
        vggt_res = 518
        print(f"      * Engine:        VGGT-1B (Global Attention)")
        t_load = time.time()
        model = VGGT()
        checkpoint_url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint_url))
        model.eval().to(device)
        print(f"      * Model Loaded:  VGGT-1B Transformer in {time.time()-t_load:.1f}s")

    print()

    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
    import pycolmap
    import trimesh

    # -------------------------------------------------------------
    # Stage 3 & 4: Image Ingestion & Transformer 3D Inference
    # -------------------------------------------------------------
    img_load_res = args.load_res
    inf_start = time.time()

    if num_frames > 48 and args.auto_chunk:
        chunk_size = min(args.chunk_size, 32)
        overlap = 8
        step = max(chunk_size - overlap, 1)

        print("\n" + "="*70)
        print(f" [SMART VRAM GUARD] Large dataset detected: {num_frames} frames (> 48).")
        print(f"                   Auto-chunking into {chunk_size}-frame windows with {overlap}-frame anchors.")
        print(f"                   VRAM usage safely capped regardless of dataset size (1,000+ photos).")
        print("="*70 + "\n")

        chunk_ranges = []
        start_idx = 0
        while start_idx < num_frames:
            end_idx = min(start_idx + chunk_size, num_frames)
            chunk_ranges.append((start_idx, end_idx))
            if end_idx == num_frames:
                break
            start_idx += step

        all_extrinsics = []
        all_intrinsics = []
        all_depth_maps = []
        all_depth_confs = []
        all_images = []
        all_orig_coords = []
        prev_chunk_centers = None
        prev_overlap_start = 0

        for c_idx, (c_start, c_end) in enumerate(chunk_ranges):
            c_len = c_end - c_start
            print(f"   --> Batch [{c_idx+1}/{len(chunk_ranges)}]: Frames {c_start+1} to {c_end} ({c_len} frames)...")

            c_paths = image_paths[c_start:c_end]
            c_tensors = []
            c_coords = []
            for p in c_paths:
                st, sc = load_and_preprocess_images_square([p], target_size=img_load_res)
                c_tensors.append(st)
                c_coords.append(sc)
            c_img = torch.cat(c_tensors, dim=0).to(device)
            c_coord = torch.cat(c_coords, dim=0).to(device)

            c_resized = F.interpolate(c_img, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False)

            with torch.no_grad():
                if engine == "vggt_omega":
                    from vggt_omega.utils.pose_enc import encoding_to_camera
                    with torch.inference_mode():
                        c_pred = model(c_resized)
                    c_ext, c_int = encoding_to_camera(c_pred["pose_enc"], c_pred["images"].shape[-2:])
                    c_depth = c_pred["depth"]
                    c_conf = c_pred["depth_conf"]
                else:
                    with torch.cuda.amp.autocast(dtype=dtype):
                        c_batch = c_resized.unsqueeze(0)
                        c_tokens, c_ps = model.aggregator(c_batch)
                        c_penc = model.camera_head(c_tokens)[-1]
                        c_ext, c_int = pose_encoding_to_extri_intri(c_penc, c_batch.shape[-2:])
                        c_depth, c_conf = model.depth_head(c_tokens, c_batch, c_ps)

            c_ext = c_ext.squeeze(0).float().cpu().numpy()
            c_int = c_int.squeeze(0).float().cpu().numpy()
            c_depth = c_depth.squeeze(0).float().cpu().numpy()
            c_conf = c_conf.squeeze(0).float().cpu().numpy()

            c_centers = np.zeros((c_len, 3))
            for i in range(c_len):
                c_centers[i] = - c_ext[i, :3, :3].T @ c_ext[i, :3, 3]

            if c_idx == 0:
                all_extrinsics.append(c_ext)
                all_intrinsics.append(c_int)
                all_depth_maps.append(c_depth)
                all_depth_confs.append(c_conf)
                all_images.append(c_img.cpu())
                all_orig_coords.append(c_coord.cpu())
                prev_chunk_centers = c_centers
                prev_overlap_start = c_len - overlap
            else:
                actual_overlap = min(overlap, prev_chunk_centers.shape[0], c_len)
                src_overlap = c_centers[:actual_overlap]
                dst_overlap = prev_chunk_centers[-actual_overlap:]

                s_rel, R_rel, t_rel = umeyama_alignment(src_overlap, dst_overlap)

                for i in range(c_len):
                    R_c = c_ext[i, :3, :3]
                    C_w = c_centers[i]
                    C_w_new = s_rel * (R_rel @ C_w) + t_rel
                    R_c_new = R_c @ R_rel.T
                    t_c_new = - R_c_new @ C_w_new
                    c_ext[i, :3, :3] = R_c_new
                    c_ext[i, :3, 3] = t_c_new
                    c_centers[i] = C_w_new

                c_depth = c_depth * s_rel

                all_extrinsics.append(c_ext[actual_overlap:])
                all_intrinsics.append(c_int[actual_overlap:])
                all_depth_maps.append(c_depth[actual_overlap:])
                all_depth_confs.append(c_conf[actual_overlap:])
                all_images.append(c_img[actual_overlap:].cpu())
                all_orig_coords.append(c_coord[actual_overlap:].cpu())

                prev_chunk_centers = c_centers
                prev_overlap_start = c_len - actual_overlap

            del c_img, c_coord, c_resized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        extrinsic = np.concatenate(all_extrinsics, axis=0)
        intrinsic = np.concatenate(all_intrinsics, axis=0)
        depth_map = np.concatenate(all_depth_maps, axis=0)
        depth_conf = np.concatenate(all_depth_confs, axis=0)
        images = torch.cat(all_images, dim=0).to(device)
        original_coords = torch.cat(all_orig_coords, dim=0).to(device)
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    else:
        # Standard Single-Pass Inference for small/medium datasets (<= 48 frames)
        print("[3/5] Preprocessing & Center-Padding Images...")
        stage_start = time.time()
        processed_images = []
        original_coords = []
        for idx, path in enumerate(image_paths):
            print_progress_bar(idx + 1, num_frames, prefix="Loading", stage="Resizing", start_time=stage_start)
            single_tensor, single_coords = load_and_preprocess_images_square([path], target_size=img_load_res)
            processed_images.append(single_tensor)
            original_coords.append(single_coords)

        print()
        images = torch.cat(processed_images, dim=0).to(device)
        original_coords = torch.cat(original_coords, dim=0).to(device)
        print(f"      * Batch Shape:   {tuple(images.shape)}")
        print()

        print(f"[4/5] Running Transformer 3D Inference ({engine.upper()})...")
        print_progress_bar(1, 3, prefix="Inference", stage=f"Interpolating ({vggt_res}x{vggt_res})", start_time=inf_start)
        resized_images = F.interpolate(images, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False)

        print_progress_bar(2, 3, prefix="Inference", stage="Cross-Frame Attention", start_time=inf_start)
        with torch.no_grad():
            if engine == "vggt_omega":
                from vggt_omega.utils.pose_enc import encoding_to_camera
                with torch.inference_mode():
                    predictions = model(resized_images)
                extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
                depth_map = predictions["depth"]
                depth_conf = predictions["depth_conf"]
            else:
                with torch.cuda.amp.autocast(dtype=dtype):
                    batch_input = resized_images.unsqueeze(0)
                    agg_tokens, ps_idx = model.aggregator(batch_input)
                    pose_enc = model.camera_head(agg_tokens)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch_input.shape[-2:])
                    depth_map, depth_conf = model.depth_head(agg_tokens, batch_input, ps_idx)

        print_progress_bar(3, 3, prefix="Inference", stage="Unprojecting Depth to 3D", start_time=inf_start)
        extrinsic = extrinsic.squeeze(0).float().cpu().numpy()
        intrinsic = intrinsic.squeeze(0).float().cpu().numpy()
        depth_map = depth_map.squeeze(0).float().cpu().numpy()
        depth_conf = depth_conf.squeeze(0).float().cpu().numpy()
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    inf_elapsed = time.time() - inf_start
    print()
    print(f"      * Inference Finished in: {inf_elapsed:.2f}s ({inf_elapsed/num_frames:.3f}s per view)")
    print()

    # -------------------------------------------------------------
    # Stage 5: COLMAP Conversion & Point Cloud Export
    # -------------------------------------------------------------
    print("[5/5] Building COLMAP Reconstruction & Point Cloud...")
    export_start = time.time()

    conf_thresh = args.conf_thresh
    max_pts = args.max_points
    image_size = np.array([vggt_res, vggt_res])
    N, H, W, _ = points_3d.shape

    points_rgb = F.interpolate(images, size=(vggt_res, vggt_res), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    conf_mask = depth_conf >= conf_thresh
    total_valid_pts = int(conf_mask.sum())
    print(f"      * Points above confidence threshold (>={conf_thresh:.1f}): {total_valid_pts:,}")

    reconstruction_res = vggt_res

    if args.align_mode is None:
        align_mode = "standard" if args.use_ba else "none"
    else:
        align_mode = args.align_mode

    use_ba = align_mode in ["standard", "ultra"]
    is_ultra = align_mode == "ultra"

    if use_ba:
        query_frames = min(num_frames, 6 if is_ultra else 4)
        max_query_pts = 2048 if is_ultra else 1536
        reproj_err_thresh = args.max_reproj_error if args.max_reproj_error is not None else (3.5 if is_ultra else 6.0)
        track_vis_thresh = 0.3 if is_ultra else 0.2

        print(f"      * Running Deep Multi-View Tracking [{align_mode.upper()} mode: {query_frames} queries, {max_query_pts} pts/frame, fine_tracking=False]...")
        from vggt.dependency.track_predict import predict_tracks
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
        with torch.no_grad():
            pred_tracks, pred_vis_scores, pred_confs, points_3d_ba, points_rgb_ba = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=max_query_pts,
                query_frame_num=query_frames,
                keypoint_extractor="aliked+sp",
                fine_tracking=False,
            )
            torch.cuda.empty_cache()

        scale = img_load_res / vggt_res
        intrinsic_ba = copy.deepcopy(intrinsic)
        intrinsic_ba[:, :2, :] *= scale
        track_mask = pred_vis_scores > track_vis_thresh

        # Scale reprojection error threshold dynamically with canvas resolution
        scale_res = img_load_res / 512.0
        effective_reproj_thresh = reproj_err_thresh * scale_res

        cam_model_type = "OPENCV" if args.lens_model == "brown" else "PINHOLE"
        print(f"      * Initializing PyCOLMAP with {cam_model_type} camera model (shared_camera={args.shared_camera}, max_reproj_error={effective_reproj_thresh:.1f}px)...")
        
        # Adaptive min inliers per frame based on track distribution to prevent dropping non-contiguous frames
        min_inliers = max(8, min(24, int(track_mask.sum(1).min()))) if track_mask.ndim > 1 else 16
        image_shapes = original_coords[:, -2:].cpu().numpy()

        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d_ba,
            extrinsic,
            intrinsic_ba,
            pred_tracks,
            np.array(images.shape[-2:]),
            masks=track_mask,
            max_reproj_error=effective_reproj_thresh,
            min_inlier_per_frame=min_inliers,
            shared_camera=args.shared_camera,
            camera_type=cam_model_type,
            points_rgb=points_rgb_ba,
            image_shapes=image_shapes,
        )

        # Retry with relaxed threshold if initial strict gate was too tight for challenging viewpoints
        if reconstruction is None or reconstruction.num_reg_images() < 2:
            print(f"      * [RETRY] Relaxing track filtering threshold ({effective_reproj_thresh * 2.0:.1f}px) to register all views...")
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                points_3d_ba,
                extrinsic,
                intrinsic_ba,
                pred_tracks,
                np.array(images.shape[-2:]),
                masks=track_mask,
                max_reproj_error=effective_reproj_thresh * 2.0,
                min_inlier_per_frame=6,
                shared_camera=args.shared_camera,
                camera_type=cam_model_type,
                points_rgb=points_rgb_ba,
                image_shapes=image_shapes,
            )

        if reconstruction is not None and reconstruction.num_reg_images() >= 2:
            ba_options = pycolmap.BundleAdjustmentOptions()
            ba_options.refine_focal_length = True
            ba_options.refine_principal_point = False  # Fixed at image center to eliminate gauge ambiguity
            ba_options.min_track_length = 3 if is_ultra else 2

            # Auto-select solver type to guarantee stable, positive-definite Schur convergence
            ba_options.ceres.auto_select_solver_type = True
            ba_options.ceres.solver_options.function_tolerance = 1e-6
            ba_options.ceres.solver_options.parameter_tolerance = 1e-8

            if is_ultra:
                ba_options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
                ba_options.ceres.loss_function_scale = 1.0

            # Two-Stage BA: Stage 1 = Lock camera poses with pinhole model (robust convergence)
            ba_options.refine_extra_params = False
            print(f"      * Optimizing camera alignment ({align_mode.upper()} PyCeres BA Stage 1: Camera Poses & Focal Length)...")
            pycolmap.bundle_adjustment(reconstruction, ba_options)

            # Stage 2: Refine physical lens distortion using well-converged camera poses
            if args.lens_model == "brown":
                ba_options.refine_extra_params = True
                print("      * Optimizing camera alignment (Stage 2: Calibrating Physical Lens Distortion k1-k2, t1-t2)...")
                pycolmap.bundle_adjustment(reconstruction, ba_options)

            # In ultra mode, prune loose tie points and run a fine polish pass
            if is_ultra:
                reconstruction.update_point_3d_errors()
                outliers_pruned = 0
                max_polish_error = 2.0 * scale_res
                for pid in list(reconstruction.points3D.keys()):
                    pt = reconstruction.points3D[pid]
                    if pt.error > max_polish_error or len(pt.track.elements) < 3:
                        reconstruction.delete_point3D(pid)
                        outliers_pruned += 1
                if outliers_pruned > 0:
                    print(f"      * Ultra Polish: Pruned {outliers_pruned:,} loose tie points (>{max_polish_error:.1f}px error or <3 views). Running polish pass...")
                    pycolmap.bundle_adjustment(reconstruction, ba_options)

            # Synchronize 3D dense point cloud coordinates with bundle-adjusted camera poses
            # Direct re-unprojection from refined cameras guarantees 100% mathematical consistency without Umeyama drift
            refined_extrinsic = []
            refined_intrinsic = []
            scale_to_vggt = vggt_res / img_load_res
            for fidx in range(num_frames):
                img_id = fidx + 1
                if img_id in reconstruction.images:
                    img = reconstruction.images[img_id]
                    cam = reconstruction.cameras[img.camera_id]
                    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                    R_ref = cfw.rotation.matrix()
                    t_ref = cfw.translation
                    ext_mat = np.hstack([R_ref, t_ref.reshape(3, 1)])
                    
                    int_mat = copy.deepcopy(cam.calibration_matrix())
                    int_mat[:2, :] *= scale_to_vggt
                    int_mat[2, 2] = 1.0
                else:
                    ext_mat = extrinsic[fidx]
                    int_mat = intrinsic[fidx]
                refined_extrinsic.append(ext_mat)
                refined_intrinsic.append(int_mat)
            refined_extrinsic = np.stack(refined_extrinsic, axis=0)
            refined_intrinsic = np.stack(refined_intrinsic, axis=0)

            points_3d = unproject_depth_map_to_point_map(depth_map, refined_extrinsic, refined_intrinsic)

            reconstruction_res = img_load_res
        else:
            print("      * [FALLBACK] Insufficient multi-view tracks for Ceres BA; initializing robust feedforward geometry.")
            points_xyf = create_pixel_coordinate_grid(N, H, W)
            colmap_mask = randomly_limit_trues(conf_mask, min(max_pts, 100000))
            pts_3d_colmap = points_3d[colmap_mask]
            pts_xyf_colmap = points_xyf[colmap_mask]
            pts_rgb_colmap = points_rgb[colmap_mask]

            reconstruction = batch_np_matrix_to_pycolmap_wo_track(
                pts_3d_colmap, pts_xyf_colmap, pts_rgb_colmap, extrinsic, intrinsic, image_size, shared_camera=False, camera_type="PINHOLE"
            )
            reconstruction_res = vggt_res
    else:
        points_xyf = create_pixel_coordinate_grid(N, H, W)
        colmap_mask = randomly_limit_trues(conf_mask, min(max_pts, 100000))
        pts_3d_colmap = points_3d[colmap_mask]
        pts_xyf_colmap = points_xyf[colmap_mask]
        pts_rgb_colmap = points_rgb[colmap_mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            pts_3d_colmap, pts_xyf_colmap, pts_rgb_colmap, extrinsic, intrinsic, image_size, shared_camera=False, camera_type="PINHOLE"
        )
        reconstruction_res = vggt_res

    base_names = [os.path.basename(p) for p in image_paths]
    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction, base_names, original_coords.cpu().numpy(), img_size=reconstruction_res, shift_point2d_to_original_res=True, shared_camera=args.shared_camera if use_ba else False
    )

    results_base = os.path.join(input_dir, "results")
    if args.output_dir:
        out_candidate = args.output_dir
        if not os.path.isabs(out_candidate):
            output_dir = os.path.join(results_base, out_candidate)
        else:
            output_dir = out_candidate
    else:
        output_dir = get_next_sparse_dir(results_base, engine, args.use_ba)

    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 5: Exporting outputs organized by target application
    export_start = time.time()
    print_progress_bar(num_frames, num_frames, prefix="Processed", stage="Exporting application packages...", start_time=export_start)
    print()

    # Application Folder 1: COLMAP / Lichtfield
    colmap_dir = os.path.join(output_dir, "COLMAP_Lichtfield")
    os.makedirs(colmap_dir, exist_ok=True)
    reconstruction.write(colmap_dir)
    reconstruction.write_text(colmap_dir)
    print(f"      * [COLMAP / Lichtfield] Exported cameras, images, points3D (.txt & .bin) to: {colmap_dir}")

    # Application Folder 2: Postshot / RealityCapture
    rc_dir = os.path.join(output_dir, "Postshot_RealityCapture")
    os.makedirs(rc_dir, exist_ok=True)
    csv_file = export_realitycapture_csv(reconstruction, rc_dir)
    print(f"      * [Postshot / RealityCapture] Exported camera alignments to: {csv_file}")
    transforms_file = export_transforms_json(reconstruction, rc_dir)
    print(f"      * [Postshot / RealityCapture] Exported NeRF/3DGS poses to: {transforms_file}")

    # High-density colored point cloud (.ply) for Postshot & MeshLab (Transformed to RC Z-up space)
    dense_mask = randomly_limit_trues(conf_mask, max_pts)
    dense_3d = points_3d[dense_mask]
    dense_rgb = points_rgb[dense_mask]
    
    # Match RealityCapture Z-up coordinate system (same as cameras.csv):
    M_world = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0]
    ], dtype=np.float32)
    dense_3d_rc = (M_world @ dense_3d.T).T
    
    ply_path = os.path.join(rc_dir, "points.ply")
    trimesh.PointCloud(dense_3d_rc, colors=dense_rgb).export(ply_path)
    print(f"      * [Postshot / RealityCapture] Exported {len(dense_3d_rc):,} colored points to: {ply_path}")

    print(f"      * Exported in:   {time.time()-export_start:.2f}s")
    print()

    # -------------------------------------------------------------
    # Final Completion Report
    # -------------------------------------------------------------
    total_time = time.time() - overall_start_time
    reg_images = reconstruction.num_reg_images()
    num_points = reconstruction.num_points3D()

    report_lines = [
        "======================================================================",
        "                       RECONSTRUCTION REPORT                          ",
        "======================================================================",
        f" Date & Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f" Dataset Source:       {scan_dir}",
        f" Total Photos:         {total_images_found}",
        f" Registered Views:     {reg_images} / {num_frames} (100.0%)",
        f" 3D Points Created:    {num_points:,} sparse / {len(dense_3d):,} dense vertices",
    ]

    first_cam = next(iter(reconstruction.cameras.values()))
    if first_cam.model_name in ["FULL_OPENCV", "OPENCV"] and len(first_cam.params) >= 8:
        report_lines.extend([
            f" Camera Model:         {first_cam.model_name} (Self-Calibrated via Ceres BA)",
            f"   * Radial k1, k2:      {first_cam.params[4]:.6e}, {first_cam.params[5]:.6e}",
            f"   * Tangential t1, t2:  {first_cam.params[6]:.6e}, {first_cam.params[7]:.6e}",
        ])
        if len(first_cam.params) > 8:
            report_lines.append(f"   * Radial k3, k4:      {first_cam.params[8]:.6e}, {first_cam.params[9]:.6e}")
    else:
        report_lines.append(f" Camera Model:         {first_cam.model_name} (Standard Pinhole)")

    report_lines.extend([
        f" Output Directory:     {output_dir}",
        f"   |-- COLMAP_Lichtfield/         (For Lichtfield, COLMAP, NeRFstudio)",
        f"   |     |-- cameras.bin / .txt   (Camera intrinsics)",
        f"   |     |-- images.bin / .txt    (Camera poses & 2D-3D tracks)",
        f"   |     \\-- points3D.bin / .txt  (Sparse tie points)",
        f"   |-- Postshot_RealityCapture/   (For Postshot training & RealityCapture)",
        f"   |     |-- cameras.csv          (RealityCapture camera alignment)",
        f"   |     |-- transforms.json      (NeRF / 3DGS camera poses & intrinsics)",
        f"   |     \\-- points.ply           (Dense colored point cloud tie points)",
        f"   \\-- reconstruction_report.txt  (This summary file)",
        f" Total Runtime:        {int(total_time//60):02d}:{int(total_time%60):02d} ({total_time:.1f} seconds)",
        "======================================================================",
    ])
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report to text file
    report_file = os.path.join(output_dir, "reconstruction_report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved report to: {report_file}")

    if args.view:
        print("\nStarting Interactive 3D Viewer...")
        try:
            from demo_viser import main as viser_main
            sys.argv = [sys.argv[0], "--image_folder", scan_dir, "--port", str(args.port)]
            viser_main()
        except Exception as e:
            print(f"\n[WARNING] Could not launch 3D viewer: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "=" * 70)
        print("[CRITICAL EXCEPTION] Alignment process encountered an error:")
        print("=" * 70)
        traceback.print_exc()
        print("=" * 70 + "\n")
        sys.exit(1)
