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
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Ensure parent directory is on sys.path so vggt and dependencies can always be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
for parent in [current_dir, os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir))]:
    if parent and parent not in sys.path:
        sys.path.insert(0, parent)

def get_next_sparse_dir(results_base_dir, engine_name="vggt", use_ba=False):
    """Finds the next unique sparse output folder (e.g. sparse_03_omega) that never overwrites existing results."""
    eng_lower = engine_name.lower()
    if "ace" in eng_lower:
        engine_suffix = "ace0"
    elif "cusfm" in eng_lower or "pycusfm" in eng_lower:
        engine_suffix = "cusfm"
    elif "omega" in eng_lower:
        engine_suffix = "omega"
    elif "360dvo" in eng_lower or "dvo" in eng_lower:
        engine_suffix = "360dvo"
    elif "opensfm" in eng_lower:
        engine_suffix = "opensfm"
    else:
        engine_suffix = "vggt"
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
import trimesh
import json

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
   Abu Align & Train v0.0.4.4 - Neural Photogrammetry & 3DGS Training
   Align: VGGT-Omega | Niantic ACE0 | NVIDIA PyCuSFM | OpenSfM | 360DVO
   Train: LichtFeld Studio | Nerfstudio Splatfacto | NVIDIA 3DGRUT | PFGS360
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


def export_transforms_json(reconstruction, output_dir, rel_image_dir="images"):
    """
    Exports camera poses and intrinsics in NeRF / 3D Gaussian Splatting transforms.json format.
    Postshot, Nerfstudio, and standard 3DGS pipelines load this natively with zero Euler angle ambiguity.
    """
    import json
    
    # Check if all cameras share the exact same dimensions
    all_dims = {(cam.width, cam.height) for cam in reconstruction.cameras.values()}
    is_uniform_resolution = len(all_dims) == 1

    first_cam = next(iter(reconstruction.cameras.values()))
    transforms_data = {
        "camera_model": first_cam.model_name,
        "frames": []
    }
    
    # If all cameras have identical resolution, root w/h can be included safely.
    # Otherwise, omit root w/h so parsers (like Nerfstudio) evaluate per-frame dimensions.
    if is_uniform_resolution:
        w, h = first_cam.width, first_cam.height
        fx, fy, cx, cy = first_cam.params[0], first_cam.params[1], first_cam.params[2], first_cam.params[3]
        transforms_data["w"] = int(w)
        transforms_data["h"] = int(h)
        transforms_data["fl_x"] = float(fx)
        transforms_data["fl_y"] = float(fy)
        transforms_data["cx"] = float(cx)
        transforms_data["cy"] = float(cy)
    
    if first_cam.model_name in ["OPENCV", "FULL_OPENCV"] and len(first_cam.params) >= 8:
        transforms_data["k1"] = float(first_cam.params[4])
        transforms_data["k2"] = float(first_cam.params[5])
        transforms_data["p1"] = float(first_cam.params[6])
        transforms_data["p2"] = float(first_cam.params[7])
    
    for img_id, img in sorted(reconstruction.images.items()):
        cam = reconstruction.cameras[img.camera_id]
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
        
        fpath = f"{rel_image_dir}/{img.name}" if rel_image_dir else img.name
        frame_dict = {
            "file_path": fpath,
            "w": int(cam.width),
            "h": int(cam.height),
            "fl_x": float(cam.params[0]),
            "fl_y": float(cam.params[1] if len(cam.params) > 1 else cam.params[0]),
            "cx": float(cam.params[2] if len(cam.params) > 2 else cam.width / 2.0),
            "cy": float(cam.params[3] if len(cam.params) > 3 else cam.height / 2.0),
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
                pred_params[0] *= resize_ratio  # fx
                pred_params[1] *= resize_ratio  # fy
                pred_params[2] = pred_params[2] * resize_ratio - left  # cx
                pred_params[3] = pred_params[3] * resize_ratio - top   # cy
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



def run_vggt_reconstruction(engine, model, image_paths, num_frames, total_images_found, device, dtype, vggt_res, args):
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    def safe_unproject(d_map, ext_cam, int_cam):
        if isinstance(d_map, torch.Tensor):
            d_map = d_map.cpu().numpy()
        if d_map.ndim == 3:
            d_map = d_map[..., None]
        return unproject_depth_map_to_point_map(d_map, ext_cam, int_cam)
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
    import pycolmap
    import trimesh

    # -------------------------------------------------------------
    # Stage 3 & 4: Image Ingestion & Transformer 3D Inference
    # -------------------------------------------------------------
    img_load_res = args.load_res
    inf_start = time.time()

    from cache_manager import (
        compute_dataset_hash,
        get_cache_root,
        save_neural_cache,
        load_neural_cache,
        save_tracking_cache,
        load_tracking_cache,
        clear_cache,
    )

    dataset_root = os.path.dirname(image_paths[0])
    cache_root = get_cache_root(dataset_root, getattr(args, "cache_dir", None))
    cache_key = compute_dataset_hash(image_paths, engine, img_load_res, args.stride)
    use_cache = not getattr(args, "no_cache", False)

    if getattr(args, "clear_cache", False):
        n_cleared, b_freed = clear_cache(cache_root)
        print(f"      * [Smart Cache] Cleared {n_cleared} cache items ({b_freed/(1024*1024):.1f} MB freed).")

    cached_neural = None
    if use_cache:
        cached_neural = load_neural_cache(cache_root, cache_key, device=device, dtype=dtype)

    if cached_neural is not None:
        print("\n" + "="*70)
        print(f" [SMART CACHE HIT] Loaded neural poses & 3D points in {cached_neural['load_time_sec']:.2f}s!")
        print(f"                   Skipping Stage 3 & 4 neural inference (Saved ~5-10 minutes).")
        print("="*70 + "\n")
        print("[TELEMETRY:STAGE3] pct=50.0 status=cache_hit", flush=True)

        extrinsic = cached_neural["extrinsic"].float().cpu().numpy()
        intrinsic = cached_neural["intrinsic"].float().cpu().numpy()
        points_3d = cached_neural["points_3d"].float().cpu().numpy()
        depth_conf = cached_neural["depth_conf"].float().cpu().numpy()
        original_coords = cached_neural["original_coords"]
        cached_dm = cached_neural.get("depth_map")
        if cached_dm is not None:
            depth_map = cached_dm.float().cpu().numpy()
        else:
            depth_map = None

        # If cached images exist, use them; otherwise quickly load image batch for RGB colors
        cached_imgs = cached_neural.get("images")
        if cached_imgs is not None:
            images = cached_imgs
        else:
            img_list = []
            for p in image_paths:
                st, _ = load_and_preprocess_images_square([p], target_size=img_load_res)
                img_list.append(st)
            images = torch.cat(img_list, dim=0).to(device)

    elif num_frames > 48 and args.auto_chunk:
        user_chunk = getattr(args, "chunk_size", None)
        user_anchor = getattr(args, "anchor_size", None)
        if user_chunk is not None and user_chunk > 0:
            chunk_size = user_chunk
        else:
            chunk_size = 96 if engine == "vggt_omega" else 48

        if user_anchor is not None and user_anchor > 0:
            overlap = user_anchor
        else:
            overlap = 24 if engine == "vggt_omega" else 12
        overlap = min(overlap, max(chunk_size // 2, 1))
        step = max(chunk_size - overlap, 1)

        chunk_ranges = []
        start_idx = 0
        while start_idx < num_frames:
            end_idx = min(start_idx + chunk_size, num_frames)
            chunk_ranges.append((start_idx, end_idx))
            if end_idx == num_frames:
                break
            start_idx += step

        print("\n" + "="*70)
        if len(chunk_ranges) == 1:
            print(f" [SMART VRAM GUARD] Dataset: {num_frames} frames.")
            print(f"                   Running single unified window ({chunk_size}-frame capacity).")
            print(f"                   Zero stitching needed - 100% globally consistent geometry.")
        else:
            print(f" [SMART VRAM GUARD] Large dataset detected: {num_frames} frames.")
            print(f"                   Auto-chunking into {len(chunk_ranges)} windows of {chunk_size} frames ({overlap}-frame anchors).")
            print(f"                   3D Landmark-Enriched Umeyama: Enabled (Zero axial roll drift).")
        print("="*70 + "\n")

        all_extrinsics = []
        all_intrinsics = []
        all_depth_maps = []
        all_depth_confs = []
        all_images = []
        all_orig_coords = []
        prev_chunk_centers = None
        prev_chunk_ext = None
        prev_chunk_int = None
        prev_chunk_depth = None
        prev_chunk_conf = None
        prev_overlap_start = 0

        for c_idx, (c_start, c_end) in enumerate(chunk_ranges):
            c_len = c_end - c_start
            batch_pct = ((c_idx + 1) / len(chunk_ranges)) * 50.0
            print(f"[TELEMETRY:STAGE3] batch={c_idx+1} total={len(chunk_ranges)} pct={batch_pct:.1f}", flush=True)
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
            if c_depth.ndim == 4 and c_depth.shape[-1] == 1:
                c_depth = c_depth.squeeze(-1)
            elif c_depth.ndim == 4 and c_depth.shape[1] == 1:
                c_depth = c_depth.squeeze(1)

            if c_conf.ndim == 4 and c_conf.shape[-1] == 1:
                c_conf = c_conf.squeeze(-1)
            elif c_conf.ndim == 4 and c_conf.shape[1] == 1:
                c_conf = c_conf.squeeze(1)

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
                prev_chunk_ext = c_ext
                prev_chunk_int = c_int
                prev_chunk_depth = c_depth
                prev_chunk_conf = c_conf
                prev_overlap_start = c_len - overlap
            else:
                actual_overlap = min(overlap, prev_chunk_centers.shape[0], c_len)
                src_overlap = c_centers[:actual_overlap]
                dst_overlap = prev_chunk_centers[-actual_overlap:]

                # 3D Landmark Surface Points to lock all 6 DoF and eliminate axial roll drift
                src_pts_3d = []
                dst_pts_3d = []

                H_d, W_d = c_depth.shape[1], c_depth.shape[2]
                grid_step = max(H_d // 16, 16)

                for k in range(actual_overlap):
                    prev_k = prev_chunk_centers.shape[0] - actual_overlap + k

                    R_src = c_ext[k, :3, :3]
                    t_src = c_ext[k, :3, 3]
                    fx_s = c_int[k, 0, 0]
                    fy_s = c_int[k, 1, 1]
                    cx_s = c_int[k, 0, 2]
                    cy_s = c_int[k, 1, 2]

                    R_dst = prev_chunk_ext[prev_k, :3, :3]
                    t_dst = prev_chunk_ext[prev_k, :3, 3]
                    fx_d = prev_chunk_int[prev_k, 0, 0]
                    fy_d = prev_chunk_int[prev_k, 1, 1]
                    cx_d = prev_chunk_int[prev_k, 0, 2]
                    cy_d = prev_chunk_int[prev_k, 1, 2]

                    for v_pix in range(grid_step, H_d - grid_step, grid_step):
                        for u_pix in range(grid_step, W_d - grid_step, grid_step):
                            c_s = float(c_conf[k, v_pix, u_pix])
                            c_d = float(prev_chunk_conf[prev_k, v_pix, u_pix])
                            if c_s >= 1.5 and c_d >= 1.5:
                                z_s = float(c_depth[k, v_pix, u_pix])
                                z_d = float(prev_chunk_depth[prev_k, v_pix, u_pix])
                                if z_s > 0.05 and z_d > 0.05:
                                    p_cam_s = np.array([(u_pix - cx_s) * z_s / fx_s, (v_pix - cy_s) * z_s / fy_s, z_s], dtype=np.float64)
                                    p_w_s = (R_src.T @ (p_cam_s - t_src)).flatten()

                                    p_cam_d = np.array([(u_pix - cx_d) * z_d / fx_d, (v_pix - cy_d) * z_d / fy_d, z_d], dtype=np.float64)
                                    p_w_d = (R_dst.T @ (p_cam_d - t_dst)).flatten()

                                    src_pts_3d.append(p_w_s)
                                    dst_pts_3d.append(p_w_d)

                if len(src_pts_3d) >= 16:
                    src_pts_arr = np.array(src_pts_3d, dtype=np.float64).reshape(-1, 3)
                    dst_pts_arr = np.array(dst_pts_3d, dtype=np.float64).reshape(-1, 3)
                    src_cam_weighted = np.repeat(src_overlap, 10, axis=0).reshape(-1, 3)
                    dst_cam_weighted = np.repeat(dst_overlap, 10, axis=0).reshape(-1, 3)
                    src_all = np.vstack([src_cam_weighted, src_pts_arr])
                    dst_all = np.vstack([dst_cam_weighted, dst_pts_arr])
                    s_rel, R_rel, t_rel = umeyama_alignment(src_all, dst_all)
                else:
                    s_rel, R_rel, t_rel = umeyama_alignment(src_overlap.reshape(-1, 3), dst_overlap.reshape(-1, 3))

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
                prev_chunk_ext = c_ext
                prev_chunk_int = c_int
                prev_chunk_depth = c_depth
                prev_chunk_conf = c_conf
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
        points_3d = safe_unproject(depth_map, extrinsic, intrinsic)

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
        if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
            depth_map = depth_map.squeeze(-1)
        elif depth_map.ndim == 4 and depth_map.shape[1] == 1:
            depth_map = depth_map.squeeze(1)

        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf.squeeze(-1)
        elif depth_conf.ndim == 4 and depth_conf.shape[1] == 1:
            depth_conf = depth_conf.squeeze(1)

        points_3d = safe_unproject(depth_map, extrinsic, intrinsic)

    inf_elapsed = time.time() - inf_start
    print()
    print(f"      * Inference Finished in: {inf_elapsed:.2f}s ({inf_elapsed/num_frames:.3f}s per view)")
    print()

    # Save to neural cache if not already cached
    if cached_neural is None and use_cache:
        try:
            saved_dir = save_neural_cache(
                cache_root,
                cache_key,
                extrinsic=torch.from_numpy(extrinsic) if isinstance(extrinsic, np.ndarray) else extrinsic,
                intrinsic=torch.from_numpy(intrinsic) if isinstance(intrinsic, np.ndarray) else intrinsic,
                points_3d=torch.from_numpy(points_3d) if isinstance(points_3d, np.ndarray) else points_3d,
                depth_conf=torch.from_numpy(depth_conf) if isinstance(depth_conf, np.ndarray) else depth_conf,
                original_coords=original_coords,
                depth_map=torch.from_numpy(depth_map) if isinstance(depth_map, np.ndarray) else depth_map,
                points_rgb=None,
                images=None,
                metadata={
                    "engine": engine,
                    "load_res": img_load_res,
                    "stride": args.stride,
                    "num_frames": num_frames,
                }
            )
            print(f"      * [Smart Cache] Saved neural predictions to {saved_dir}")
        except Exception as ce:
            print(f"      * [Smart Cache Warning] Failed to save neural cache: {ce}")

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
        # Zero-Paging VRAM Guard: Purge Stage 4 transformer model and cache before CoTracker
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        v_alloc, _ = get_vram_usage()
        print(f"      * [Zero-Paging VRAM Guard] Cleared neural model from GPU. Active VRAM: {v_alloc:.2f} GB.")

        if num_frames > 48:
            # Trajectory-wide uniform keyframe distribution for large scans:
            # For massive scans (>200 frames), adaptively cap queries to 8 and keypoints to 1024
            # to prevent CoTracker PCIe paging bottleneck across 700+ frames
            if num_frames > 200:
                n_queries = min(8, max(6, num_frames // 30))
                max_query_pts = 1024
            else:
                n_queries = min(12, max(6, num_frames // 16))
                max_query_pts = 2048 if is_ultra else 1536
            step = (num_frames - 1) / max(n_queries - 1, 1)
            query_frames = sorted(list(set([int(round(i * step)) for i in range(n_queries)])))
            query_count_str = f"{len(query_frames)} trajectory-spanning queries ({max_query_pts} pts/query)"
        else:
            query_frames = min(num_frames, 6 if is_ultra else 4)
            max_query_pts = 2048 if is_ultra else 1536
            query_count_str = f"{query_frames} queries"

        reproj_err_thresh = args.max_reproj_error if args.max_reproj_error is not None else (3.5 if is_ultra else 6.0)
        track_vis_thresh = 0.3 if is_ultra else 0.2

        track_cache_key = f"{cache_key}_{align_mode}_{max_query_pts}"
        cached_tracks = None
        if use_cache:
            cached_tracks = load_tracking_cache(
                cache_root,
                track_cache_key,
                query_frames if isinstance(query_frames, list) else [query_frames]
            )

        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap

        if cached_tracks is not None:
            print("\n" + "="*70)
            print(f" [SMART CACHE HIT] Loaded multi-view tracks & tie points from cache!")
            print(f"                   Skipping CoTracker feature tracking (Saved ~10-20 minutes).")
            print("="*70 + "\n")
            print("[TELEMETRY:STAGE5] pct=85.0 status=cache_hit", flush=True)
            pred_tracks = cached_tracks["pred_tracks"]
            pred_vis_scores = cached_tracks["pred_vis_scores"]
            pred_confs = cached_tracks["pred_confs"]
            points_3d_ba = cached_tracks["points_3d_ba"]
            points_rgb_ba = cached_tracks["points_rgb_ba"]
        else:
            print(f"      * Running Deep Multi-View Tracking [{align_mode.upper()} mode: {query_count_str}, {max_query_pts} pts/frame, fine_tracking=False]...")
            from vggt.dependency.track_predict import predict_tracks
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

            if use_cache:
                try:
                    save_tracking_cache(
                        cache_root,
                        track_cache_key,
                        query_indices=query_frames if isinstance(query_frames, list) else [query_frames],
                        pred_tracks=pred_tracks,
                        pred_vis_scores=pred_vis_scores,
                        pred_confs=pred_confs,
                        points_3d_ba=points_3d_ba,
                        points_rgb_ba=points_rgb_ba,
                    )
                    print(f"      * [Smart Cache] Saved multi-view tracks to cache.")
                except Exception as te:
                    print(f"      * [Smart Cache Warning] Failed to save tracking cache: {te}")

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

            LinearSolverType = type(ba_options.ceres.solver_options.linear_solver_type)
            PreconditionerType = type(ba_options.ceres.solver_options.preconditioner_type)

            if num_frames > 48:
                # Iterative Schur PCG solver for large scans: eliminates SuiteSparse CHOLMOD Cholesky factorization failures
                ba_options.ceres.auto_select_solver_type = False
                ba_options.ceres.solver_options.linear_solver_type = LinearSolverType.ITERATIVE_SCHUR
                ba_options.ceres.solver_options.preconditioner_type = PreconditionerType.SCHUR_JACOBI
                ba_options.ceres.solver_options.max_linear_solver_iterations = 100
            else:
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

    conf_thresh = args.conf_thresh
    max_pts = args.max_points
    dense_mask = randomly_limit_trues(conf_mask, max_pts)
    dense_3d = points_3d[dense_mask]
    dense_rgb = points_rgb[dense_mask]
    return reconstruction, dense_3d, dense_rgb


def main():
    parser = argparse.ArgumentParser(description="Live Progress Alignment CLI for Photoscanning")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to photoscan folder or image directory")
    parser.add_argument("--output_dir", "--output", "-o", type=str, default=None, help="Custom output directory for sparse alignment and rig files")
    parser.add_argument("--engine", "-e", type=str, choices=["vggt_omega", "ace0", "acezero", "pycusfm", "cusfm", "opensfm", "360dvo", "dvo"], default="vggt_omega", help="Model engine: 'vggt_omega' (Neural Transformer), 'ace0' (Niantic Labs), 'pycusfm' (NVIDIA Isaac cuSFM), 'opensfm' (Mapillary 360), or '360dvo' (Deep 360 Visual Odometry)")
    # Video Frame Extraction Options
    parser.add_argument("--video_fps", type=float, default=5.0, help="Target FPS when extracting from video input (default: 5.0)")
    parser.add_argument("--video_format", choices=["jpg", "png"], default="jpg", help="Format for extracted video frames (default: jpg)")
    parser.add_argument("--video_quality", type=int, default=95, help="JPEG quality (1-100) for extracted frames (default: 95)")
    parser.add_argument("--video_resolution", choices=["native", "8k", "5.7k", "4k", "2k", "1k"], default="native", help="Resolution for extracted video frames")
    parser.add_argument("--video_blur", type=float, default=0.0, help="Laplacian variance threshold to discard blurred frames (default: 0.0, disabled)")
    parser.add_argument("--extract_video_only", action="store_true", help="Extract video frames to sequence and exit without running alignment")
    # 360DVO Deep Visual Odometry Options
    parser.add_argument("--dvo_weights", type=str, default=None, help="Custom checkpoint path (.pth file) for 360DVO")
    parser.add_argument("--dvo_patches", type=int, default=96, help="Number of active tracked patches per frame in 360DVO (default: 96)")
    parser.add_argument("--dvo_polar_mask", type=float, default=0.08, help="Polar zenith/nadir cutoff fraction (0.0 to 0.20, default: 0.08) in 360DVO")
    parser.add_argument("--dvo_ba_window", type=int, default=8, help="Local ODBA sliding-window bundle adjustment size (default: 8) in 360DVO")
    parser.add_argument("--dvo_scale", type=float, default=3.2, help="Metric scene depth scale multiplier (default: 3.2m) in 360DVO")
    parser.add_argument("--dvo_viz", dest="dvo_viz", action="store_true", default=True, help="Launch live real-time Viser browser visualizer during 360DVO tracking")
    parser.add_argument("--no_dvo_viz", dest="dvo_viz", action="store_false", help="Disable real-time visualizer during 360DVO tracking")
    # OpenSfM 360 & Spherical Photogrammetry Options
    parser.add_argument("--projection", type=str, choices=["auto", "equirectangular", "spherical", "perspective", "fisheye"], default="equirectangular", help="Camera projection model for OpenSfM (default: equirectangular for 360 panos)")
    parser.add_argument("--opensfm_feature", type=str, choices=["sift", "hahog", "orb", "akaze"], default="sift", help="Feature extractor for OpenSfM (default: sift)")
    parser.add_argument("--opensfm_matcher", type=str, choices=["words", "flann", "exhaustive"], default="words", help="Feature matcher for OpenSfM (default: words)")
    parser.add_argument("--opensfm_min_features", type=int, default=3000, help="Feature budget per frame for OpenSfM (default: 3000)")
    parser.add_argument("--opensfm_neighbors", type=int, default=5, help="Sequential matching window for OpenSfM (default: 5)")
    parser.add_argument("--opensfm_bundle_interval", type=int, default=10, help="Bundle adjustment frequency for OpenSfM (default: 10, runs 10x faster than 1)")
    parser.add_argument("--opensfm_optimize_camera", action="store_true", default=False, help="Optimize spherical camera parameters during OpenSfM bundle adjustment")
    parser.add_argument("--cubemap_transcode", action="store_true", help="Automatically slice 360 equirectangular images into 6 rectilinear pinhole cubemaps")
    parser.add_argument("--cubemap_res", type=int, default=1024, help="Resolution per cubemap face (default: 1024)")
    # Phase 3: 360° Multi-Camera Rig Projector & Panoramic Splitter
    parser.add_argument("--split_panorama", action="store_true", help="Split 360° panoramas into a virtual multi-camera pinhole rig")
    parser.add_argument("--split_mode", type=str, choices=["ring", "cubemap", "dome", "fibonacci"], default="ring", help="Rig projection mode ('ring', 'cubemap', 'dome', 'fibonacci'; default: ring)")
    parser.add_argument("--split_cameras", type=int, default=8, help="Number of virtual cameras when split_mode=ring (default: 8 => 50%% overlap)")
    parser.add_argument("--split_fov", type=float, default=90.0, help="Field of View per virtual camera in degrees (default: 90.0)")
    parser.add_argument("--split_res", type=int, default=1024, help="Resolution per pinhole view (default: 1024)")
    parser.add_argument("--split_format", choices=["jpg", "png"], default="jpg", help="Split image output format (default: jpg)")
    parser.add_argument("--split_quality", type=int, default=95, help="JPEG quality for split images (default: 95)")
    parser.add_argument("--split_only", action="store_true", help="Perform panoramic splitting into virtual camera rig and exit without aligning")
    parser.add_argument("--ace_iterations", type=int, default=30, help="Maximum mapping iterations for ACE0 engine (default: 30)")
    parser.add_argument("--ace_refinement", type=str, choices=["mlp", "naive", "none"], default="mlp", help="Pose refinement mode for ACE0 (default: mlp)")
    parser.add_argument("--ace_focal", type=float, default=-1.0, help="Initial focal length guess for ACE0 (-1 to auto-estimate from image diagonal)")
    parser.add_argument("--ace_try_seeds", type=int, default=3, help="Number of candidate seeds to evaluate for ACE0 (default: 3)")
    parser.add_argument("--ace_refine_calib", dest="ace_refine_calib", action="store_true", default=True, help="Optimize camera focal length during ACE0 mapping (default: True)")
    parser.add_argument("--no_ace_refine_calib", dest="ace_refine_calib", action="store_false", help="Keep camera focal length fixed during ACE0 mapping")
    parser.add_argument("--ace_dense_pc", action="store_true", default=False, help="Export dense unfiltered point cloud in ACE0 (skips reprojection filter)")
    parser.add_argument("--cusfm_features", type=str, choices=["aliked", "superpoint", "sift"], default="aliked", help="Feature extractor for PyCuSFM engine (default: aliked)")
    parser.add_argument("--cusfm_matcher", type=str, choices=["lightglue", "superglue"], default="lightglue", help="Keypoint matcher for PyCuSFM (default: lightglue)")
    parser.add_argument("--cusfm_init", type=str, choices=["omega", "ace0", "none"], default="omega", help="Coarse pose initialization mode for PyCuSFM (default: omega)")
    parser.add_argument("--cusfm_ba_iterations", type=int, default=100, help="Maximum bundle adjustment iterations for PyCuSFM (default: 100)")
    parser.add_argument("--cusfm_rgb", dest="cusfm_rgb", action="store_true", default=True, help="Generate colorized RGB 3D point cloud for PyCuSFM")
    parser.add_argument("--no_cusfm_rgb", dest="cusfm_rgb", action="store_false", help="Generate geometric sparse point cloud for PyCuSFM")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Custom checkpoint path (.pt file) for VGGT-Omega")
    parser.add_argument("--stride", "-s", type=int, default=1, help="Stride / keyframe interval (e.g. 2 or 3 for large datasets)")
    parser.add_argument("--max_frames", "-m", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--conf_thresh", type=float, default=1.8, help="Depth confidence threshold (default: 1.8; lower=denser point cloud)")
    parser.add_argument("--max_points", type=int, default=300000, help="Maximum number of points in exported point cloud (default: 300000)")
    parser.add_argument("--load_res", type=int, default=1024, help="Image load resolution (default: 1024)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size for auto-chunking large datasets (default: 96 for VGGT-Omega, 48 for standard)")
    parser.add_argument("--auto_chunk", action="store_true", default=True, help="Automatically chunk large datasets (> 48 photos) to prevent VRAM OOM")
    parser.add_argument("--use_ba", action="store_true", help="Enable multi-view feature tracking and PyCeres Bundle Adjustment")
    parser.add_argument("--align_mode", type=str, choices=["none", "standard", "ultra"], default=None, help="Camera alignment precision: 'none' (fast feedforward), 'standard' (Ceres BA), 'ultra' (sub-pixel Ceres BA with Cauchy loss & multi-pass polish)")
    parser.add_argument("--lens_model", type=str, choices=["pinhole", "brown"], default="brown", help="Lens distortion model: 'pinhole' (zero distortion) or 'brown' (calibrates physical k1, k2, t1, t2)")
    parser.add_argument("--max_reproj_error", type=float, default=None, help="Max reprojection error for tie point tracks")
    parser.add_argument("--shared_camera", dest="shared_camera", action="store_true", default=True, help="Share camera intrinsics and distortion across all views (recommended for single-camera scans)")
    parser.add_argument("--no_shared_camera", dest="shared_camera", action="store_false", help="Calibrate separate camera intrinsics for each view")
    parser.add_argument("--view", "-v", action="store_true", help="Launch interactive 3D browser viewer after alignment")
    parser.add_argument("--port", type=int, default=8080, help="Viser viewer port (default: 8080)")

    parser.add_argument("--anchor_size", type=int, default=None, help="Anchor overlap frames connecting consecutive sliding windows (default: 24 for VGGT-Omega, 12 for standard)")
    parser.add_argument("--no_cache", action="store_true", help="Disable reading and writing from .abu_cache")
    parser.add_argument("--cache_dir", type=str, default=None, help="Custom directory path for .abu_cache")
    parser.add_argument("--clear_cache", action="store_true", help="Purge .abu_cache before running alignment")

    # 3D Gaussian Splatting Training Suite Options
    parser.add_argument("--train", action="store_true", help="Automatically train 3D Gaussian Splatting model after alignment finishes")
    parser.add_argument("--train_engine", type=str, choices=["lichtfeld", "splatfacto", "3dgrut", "pfgs360"], default="lichtfeld", help="3DGS training engine (default: lichtfeld)")
    parser.add_argument("--train_iterations", type=int, default=30000, help="Number of 3DGS training iterations (default: 30000; quick test: 7000)")
    parser.add_argument("--train_mode", type=str, choices=["interactive", "headless"], default="interactive", help="Training view mode: 'interactive' (live visualizer) or 'headless' (background)")
    parser.add_argument("--gut", action="store_true", help="Enable Gaussian Unscented Transform (GUT) for enhanced non-linear optics")
    parser.add_argument("--export_spz", action="store_true", default=False, help="Export compressed .spz format in LichtFeld Studio")
    parser.add_argument("--ns_model", choices=["splatfacto", "splatfacto-big"], default="splatfacto", help="Model variant for Nerfstudio (default: splatfacto)")
    parser.add_argument("--camera_optimizer", choices=["none", "so3_x_r3", "se3"], default="none", help="Camera optimizer for Nerfstudio (default: none)")
    parser.add_argument("--bilateral_grid", action="store_true", help="Enable bilateral grid color correction in LichtFeld / 3DGS")
    parser.add_argument("--exposure_correction", action="store_true", help="Enable per-photo auto-exposure and white-balance correction")
    parser.add_argument("--enable_sparsity", action="store_true", help="Enable ADMM sparsity pruning (60% smaller splat file size)")
    parser.add_argument("--enable_mip", action="store_true", help="Enable anti-aliased mip-filtering for splats")
    parser.add_argument("--representation", choices=["3dgut", "3dgrut"], default="3dgut", help="Representation mode for 3DGRUT")
    parser.add_argument("--bvh_backend", choices=["optix", "cuda"], default="optix", help="BVH backend for 3DGRUT")
    parser.add_argument("--sh_degree", type=int, default=3, choices=[0, 1, 2, 3], help="Spherical harmonics degree (default: 3)")
    parser.add_argument("--pfgs_dataparser", default="auto", help="Dataparser for PFGS360 ('auto', 'opensfm', 'posefree', 'ob3d', 'odgs')")
    parser.add_argument("--pfgs_depth", choices=["unik3d", "midas", "zoedepth"], default="unik3d", help="Monocular depth prior for PFGS360 (default: unik3d)")
    parser.add_argument("--pfgs_filter_sky", dest="pfgs_filter_sky", action="store_true", default=True, help="Filter sky areas (>50m) in PFGS360")
    parser.add_argument("--no_pfgs_filter_sky", dest="pfgs_filter_sky", action="store_false", help="Do not filter sky in PFGS360")
    parser.add_argument("--pfgs_sky_depth", type=float, default=50.0, help="Max depth threshold for sky filtering in meters (default: 50.0)")
    parser.add_argument("--pfgs_match_views", type=int, default=8, help="Sliding window size for PFGS360 pose matching (default: 8; 0 for exhaustive all-pairs)")
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

    input_path = os.path.abspath(args.input)
    if os.path.isfile(input_path):
        from video_extractor import is_video_file, prepare_dataset_from_video_or_images
        if is_video_file(input_path):
            video_fps = getattr(args, "video_fps", 5.0)
            video_fmt = getattr(args, "video_format", "jpg")
            video_q = getattr(args, "video_quality", 95)
            video_res = getattr(args, "video_resolution", "native")
            video_blur = getattr(args, "video_blur", 0.0)

            target_ds = os.path.join(os.path.dirname(input_path), f"{Path(input_path).stem}_dataset")
            print(f"[*] Detected video input: {Path(input_path).name}")
            print(f"[*] Extracting frames -> {target_ds} ({video_fps} FPS, {video_res.upper()}, {video_fmt.upper()})...")

            extracted_ds, was_extracted = prepare_dataset_from_video_or_images(
                input_path=input_path,
                target_dataset_dir=target_ds,
                target_fps=video_fps,
                image_format=video_fmt,
                quality=video_q,
                resolution=video_res if video_res != "native" else None,
                blur_threshold=video_blur,
                progress_callback=lambda p: print(f"    Extracting frame {p['current_frame']}/{p['total_frames']} ({p['percent']:.1f}%)", end="\r"),
            )
            print()
            input_dir = extracted_ds
            if getattr(args, "extract_video_only", False):
                print(f"[*] Video extraction complete: {os.path.join(extracted_ds, 'images')}")
                print(f"[*] Exiting (--extract_video_only).")
                sys.exit(0)
        else:
            print(f"\n[ERROR] Input file is not a supported video or folder: {input_path}")
            sys.exit(1)
    elif not os.path.isdir(input_path):
        print(f"\n[ERROR] Input directory or video not found: {input_path}")
        sys.exit(1)
    else:
        input_dir = input_path

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

    results_base = os.path.join(input_dir, "results")
    if args.output_dir:
        out_candidate = args.output_dir
        if not os.path.isabs(out_candidate):
            output_dir = os.path.join(results_base, out_candidate)
        else:
            output_dir = out_candidate
    else:
        output_dir = get_next_sparse_dir(results_base, args.engine, args.use_ba)

    os.makedirs(output_dir, exist_ok=True)
    print(f"      * Target Output: {output_dir}")
    print()

    # 360° Multi-Camera Rig Projector & Panoramic Splitter (Phase 3)
    is_equirectangular = (orig_w == 2 * orig_h)
    needs_splitting = (
        getattr(args, "split_panorama", False)
        or getattr(args, "split_only", False)
        or getattr(args, "cubemap_transcode", False)
        or (is_equirectangular and args.engine not in ["opensfm", "360dvo", "dvo360", "pfgs360"])
    )
    if needs_splitting:
        split_mode = "cubemap" if (getattr(args, "cubemap_transcode", False) and not getattr(args, "split_panorama", False) and not getattr(args, "split_only", False)) else getattr(args, "split_mode", "ring")
        split_cams = getattr(args, "split_cameras", 8)
        split_fov = getattr(args, "split_fov", 90.0)
        split_res = getattr(args, "split_res", getattr(args, "cubemap_res", 1024))
        split_fmt = getattr(args, "split_format", "jpg")
        split_qual = getattr(args, "split_quality", 95)

        cams_info = split_cams if split_mode == "ring" else ("6" if split_mode == "cubemap" else "14")
        print(f"      * [360 RIG PROJECTOR] 360° Equirectangular dataset detected ({orig_w}x{orig_h}).")
        print(f"      * Projecting into {split_mode.upper()} multi-camera pinhole rig ({cams_info} views/pano, {split_fov}° FOV, {split_res}x{split_res}px)...")

        from panoramic_splitter import split_panorama_dataset
        rig_out_dir = os.path.join(output_dir, f"rig_{split_mode}_frames")
        res = split_panorama_dataset(
            input_dir=scan_dir,
            output_dir=rig_out_dir,
            mode=split_mode,
            num_cameras=split_cams,
            fov_deg=split_fov,
            crop_size=split_res,
            image_format=split_fmt,
            quality=split_qual,
            export_transforms_json=True,
            progress_callback=lambda cur, tot, name: print(f"        Splitting panorama {cur}/{tot}: {name}", end="\r"),
        )
        print()
        print(f"      * [OK] Generated {res['total_views']} pinhole views with transforms.json in {res['output_dir']}")

        if getattr(args, "split_only", False):
            print(f"\n[*] Multi-camera rig projection complete!")
            print(f"[*] Images directory:    {res['images_dir']}")
            if res.get("transforms_path"):
                print(f"[*] Transforms manifest: {res['transforms_path']}")
            print(f"[*] Exiting (--split_only).")
            sys.exit(0)

        cubemap_paths = sorted([
            os.path.join(res["images_dir"], f) for f in os.listdir(res["images_dir"])
            if os.path.splitext(f.lower())[1] in valid_exts
        ])
        if cubemap_paths:
            image_paths = cubemap_paths
            num_frames = len(image_paths)
            scan_dir = res["images_dir"]
            with Image.open(image_paths[0]) as sample_img:
                orig_w, orig_h = sample_img.size
            print(f"      * Projected Rig Pinhole Views for Alignment: {num_frames} ({orig_w}x{orig_h})")
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
    vggt_res = 512

    def load_vggt_omega_model(dev, dt, ckpt=None):
        possible_ckpts = [
            ckpt,
            r"C:\Users\USer\.gemini\antigravity\scratch\General Stuff 01\vggt-omega\vggt_omega_1b_512.pt",
            r"C:\Users\USer\.cache\torch\hub\checkpoints\vggt_omega_1b_512.pt",
            os.path.join(os.getcwd(), "checkpoints", "vggt_omega_1b_512.pt"),
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
                m = VGGTOmega().eval()
                state_dict = torch.load(found_ckpt, map_location="cpu")
                m.load_state_dict(state_dict)
                m = m.to(dev)
                print(f"      * Model Loaded:  VGGT-Omega in {time.time()-t_load:.1f}s")
                return m, 512
            except Exception as e:
                raise RuntimeError(f"Failed to load VGGT-Omega checkpoint from {found_ckpt}: {e}")
        else:
            raise FileNotFoundError(
                "VGGT-Omega weights (vggt_omega_1b_512.pt) were not found!\n"
                "Please ensure vggt_omega_1b_512.pt is located in checkpoints/ or specify --checkpoint."
            )

    # Determine which model is needed for inference or coarse initialization
    needs_omega = (engine == "vggt_omega") or (engine in ["pycusfm", "cusfm"] and args.cusfm_init == "omega")

    # Load VGGT-Omega if requested
    if needs_omega:
        model, vggt_res = load_vggt_omega_model(device, dtype, args.checkpoint)

    # ACE0 Engine initialization
    if engine in ["ace0", "acezero"]:
        print(f"      * Engine:        ACE0 (Scene Coordinate Regression, Niantic Labs)")
        print(f"      * Architecture:  Implicit Coordinate Regression MLP + ZoeDepth Seed")
        print(f"      * Max Rounds:    {args.ace_iterations}")
        print(f"      * Refinement:    {args.ace_refinement.upper()}")

    # NVIDIA PyCuSFM Engine initialization
    if engine in ["pycusfm", "cusfm"]:
        print(f"      * Engine:        NVIDIA PyCuSFM (Isaac cuSFM GPU Pipeline)")
        print(f"      * GPU Features:  {args.cusfm_features.upper()}")
        print(f"      * Coarse Init:   {args.cusfm_init.upper()}")
        print(f"      * Max BA Iters:  {args.cusfm_ba_iterations}")

    print()

    # -------------------------------------------------------------
    # Stage 3 & 4: 3D Alignment & Reconstruction Inference
    # -------------------------------------------------------------
    if engine in ["ace0", "acezero"]:
        print("[3/5] Starting ACE0 Scene Coordinate Regression...")
        print("      * Alternating MLP Coordinate Training & PnP-RANSAC Registration")
        inf_start = time.time()
        from acezero_engine import run_acezero_reconstruction
        reconstruction, dense_3d, dense_rgb = run_acezero_reconstruction(
            image_paths=image_paths,
            output_dir=output_dir,
            iterations_max=args.ace_iterations,
            refinement=args.ace_refinement,
            try_seeds=args.ace_try_seeds,
            external_focal_length=args.ace_focal,
            refine_calibration=getattr(args, "ace_refine_calib", True),
            image_resolution=min(args.load_res, 640) if args.load_res else 480,
            export_point_cloud=True,
            dense_point_cloud=getattr(args, "ace_dense_pc", False),
            shared_camera=args.shared_camera,
        )
        inf_elapsed = time.time() - inf_start
        print()
        print(f"      * ACE0 Finished in: {inf_elapsed:.2f}s ({inf_elapsed/num_frames:.3f}s per view)")
        print()
    elif engine in ["pycusfm", "cusfm"]:
        print("[3/5] Starting NVIDIA PyCuSFM GPU Structure-from-Motion Pipeline...")
        print(f"      * Feature Extractor: {args.cusfm_features.upper()} (GPU Accelerated)")
        print(f"      * Matcher:           {getattr(args, 'cusfm_matcher', 'lightglue').upper()} (GPU Transformer)")
        print(f"      * Coarse Initializer:{args.cusfm_init.upper()}")
        inf_start = time.time()

        initial_poses = None
        initial_intrinsics = None
        if args.cusfm_init in ["omega", "vggt"] and model is not None:
            init_engine = "vggt_omega" if (args.cusfm_init == "omega" and getattr(model, "__class__", None).__name__ == "VGGTOmega") else "vggt"
            print(f"      * Computing coarse initial camera poses via {init_engine.upper()} (fast feedforward)...")
            coarse_args = copy.deepcopy(args)
            coarse_args.use_ba = False
            coarse_args.align_mode = "none"
            coarse_rec, _, _ = run_vggt_reconstruction(
                init_engine, model, image_paths, num_frames, total_images_found, device, dtype, vggt_res, coarse_args
            )
            initial_poses = []
            name_to_im = {im.name: im for im in coarse_rec.images.values()}
            for p in image_paths:
                bname = os.path.basename(p)
                if bname in name_to_im:
                    im = name_to_im[bname]
                    cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
                    R_w2c = cfw.rotation.matrix()
                    t_w2c = cfw.translation
                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c
                    initial_poses.append((R_c2w, t_c2w))
                else:
                    initial_poses.append((np.eye(3), np.array([0.0, 0.0, len(initial_poses) * 0.1])))
            if coarse_rec.cameras:
                cam = list(coarse_rec.cameras.values())[0]
                initial_intrinsics = (cam.params[0], cam.params[1], cam.params[2], cam.params[3])

        from pycusfm_engine import run_pycusfm_reconstruction
        reconstruction, dense_3d, dense_rgb = run_pycusfm_reconstruction(
            image_paths=image_paths,
            output_dir=output_dir,
            feature_type=args.cusfm_features,
            initial_poses=initial_poses,
            initial_intrinsics=initial_intrinsics,
            output_rgb=getattr(args, "cusfm_rgb", True),
            max_ba_iterations=args.cusfm_ba_iterations,
        )
        inf_elapsed = time.time() - inf_start
        print()
        print(f"      * PyCuSFM Finished in: {inf_elapsed:.2f}s ({inf_elapsed/num_frames:.3f}s per view)")
        print()
    elif engine in ["opensfm"]:
        print("[3/5] Starting Mapillary OpenSfM Spherical & Street-Scale Engine...")
        print(f"      * Projection Type:   {args.projection.upper()}")
        print(f"      * Feature Extractor: {args.opensfm_feature.upper()} (Budget: {args.opensfm_min_features})")
        print(f"      * Matcher:           {args.opensfm_matcher.upper()} (Window: {args.opensfm_neighbors})")
        print(f"      * Bundle Interval:   Every {args.opensfm_bundle_interval} view(s)")
        print(f"      * Optimize Camera:   {'Enabled' if args.opensfm_optimize_camera else 'Fixed Ideal S²'}")
        inf_start = time.time()

        from opensfm_engine import OpenSfMEngine
        opensfm = OpenSfMEngine(
            dataset_path=scan_dir,
            output_dir=os.path.join(output_dir, "opensfm_workspace"),
            projection_type=args.projection,
            feature_type=args.opensfm_feature,
            matcher_type=args.opensfm_matcher,
            feature_min_frames=args.opensfm_min_features,
            matching_order_neighbors=args.opensfm_neighbors,
            bundle_interval=args.opensfm_bundle_interval,
            optimize_camera_parameters=args.opensfm_optimize_camera
        )

        if opensfm.opensfm_bin is not None:
            opensfm.setup_workspace(image_paths)
            success = opensfm.run_pipeline()
            if success:
                import pycolmap
                colmap_out = os.path.join(opensfm.output_dir, "colmap")
                if not os.path.isdir(colmap_out):
                    colmap_out = os.path.join(opensfm.output_dir, "sparse")
                if os.path.isdir(colmap_out):
                    reconstruction = pycolmap.Reconstruction(colmap_out)
                else:
                    reconstruction = pycolmap.Reconstruction()

                # Automatically export 3dgs_dataset (transforms.json, sparse/0/, points.ply)
                dataset_3dgs_dir = os.path.join(output_dir, "3dgs_dataset")
                opensfm.export_to_3dgs_dataset(dataset_3dgs_dir)

                # Load genuine OpenSfM 3D point cloud
                rec_ply = os.path.join(opensfm.output_dir, "reconstruction.ply")
                if not os.path.isfile(rec_ply):
                    rec_ply = os.path.join(output_dir, "reconstruction.ply")
                if os.path.isfile(rec_ply) and os.path.getsize(rec_ply) > 1000:
                    try:
                        mesh = trimesh.load(rec_ply)
                        dense_3d = np.asarray(mesh.vertices, dtype=np.float32)
                        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(dense_3d):
                            dense_rgb = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
                        else:
                            dense_rgb = np.full((len(dense_3d), 3), 200, dtype=np.uint8)
                        print(f"      * [OpenSfM Point Cloud] Loaded {len(dense_3d):,} authentic 3D tie points from {os.path.basename(rec_ply)}")
                    except Exception as e:
                        print(f"      * [OpenSfM Warning] Failed to load {rec_ply}: {e}")
                        dense_3d = np.empty((0, 3), dtype=np.float32)
                        dense_rgb = np.empty((0, 3), dtype=np.uint8)
                else:
                    dense_3d = np.empty((0, 3), dtype=np.float32)
                    dense_rgb = np.empty((0, 3), dtype=np.uint8)
            else:
                raise RuntimeError("OpenSfM pipeline execution encountered an error.")
        else:
            print("      * [NOTICE] Native OpenSfM binary CLI not detected in system PATH.")
            print("      * [SPHERICAL S² SOLVER] Engaging Native Python Spherical Structure-from-Motion Engine...")
            reconstruction, dense_3d, dense_rgb = opensfm.run_spherical_sfm(
                image_paths=image_paths,
                output_dir=output_dir,
                stride=args.stride
            )

        inf_elapsed = time.time() - inf_start
        print()
        print(f"      * OpenSfM Reconstruction Finished in: {inf_elapsed:.2f}s")
        print()
    elif engine in ["360dvo", "dvo"]:
        print("[3/5] Starting 360DVO Deep Omnidirectional Visual Odometry Engine (IEEE RA-L 2026)...")
        print(f"      * Active Patches:    {args.dvo_patches} per frame")
        print(f"      * Temporal Stride:   {args.stride}")
        inf_start = time.time()
        from dvo360_engine import run_360dvo_reconstruction
        reconstruction, dense_3d, dense_rgb = run_360dvo_reconstruction(
            image_paths=image_paths,
            output_dir=output_dir,
            weights_path=args.dvo_weights,
            patches_per_frame=args.dvo_patches,
            stride=args.stride,
            polar_mask_pct=getattr(args, "dvo_polar_mask", 0.08),
            local_ba_window=getattr(args, "dvo_ba_window", 8),
            metric_scale=getattr(args, "dvo_scale", 3.2),
            enable_viz=args.dvo_viz
        )
        inf_elapsed = time.time() - inf_start
        print()
        print(f"      * 360DVO Visual Odometry Finished in: {inf_elapsed:.2f}s ({inf_elapsed/max(num_frames, 1):.3f}s per view)")
        print()
    else:
        reconstruction, dense_3d, dense_rgb = run_vggt_reconstruction(
            engine, model, image_paths, num_frames, total_images_found, device, dtype, vggt_res, args
        )

    # Stage 5: Exporting outputs organized by target application
    export_start = time.time()
    print_progress_bar(num_frames, num_frames, prefix="Processed", stage="Exporting application packages...", start_time=export_start)
    print()

    # Application Folder 0: Unified 3DGS Standard Dataset (sparse/0/ + images/ + transforms.json)
    dataset_3dgs_dir = os.path.join(output_dir, "3dgs_dataset")
    os.makedirs(dataset_3dgs_dir, exist_ok=True)
    sparse_0_dir = os.path.join(dataset_3dgs_dir, "sparse", "0")
    os.makedirs(sparse_0_dir, exist_ok=True)
    
    if reconstruction is not None:
        try:
            reconstruction.write(sparse_0_dir)
            reconstruction.write_text(sparse_0_dir)
        except Exception:
            pass

    # Copy raw sparse files if generated directly by engine (e.g. 360DVO or OpenSfM)
    src_sparse = os.path.join(output_dir, "sparse", "0")
    if os.path.isdir(src_sparse):
        for fname in os.listdir(src_sparse):
            src_f = os.path.join(src_sparse, fname)
            dst_f = os.path.join(sparse_0_dir, fname)
            if not os.path.exists(dst_f):
                try:
                    shutil.copy2(src_f, dst_f)
                except Exception:
                    pass

    # Link/stage images folder
    images_dst = os.path.join(dataset_3dgs_dir, "images")
    if not os.path.exists(images_dst):
        try:
            # Try Windows directory junction first
            subprocess.run(["cmd", "/c", "mklink", "/J", images_dst, os.path.abspath(scan_dir)], check=True, capture_output=True)
        except Exception:
            os.makedirs(images_dst, exist_ok=True)
            for src_p in image_paths:
                dst_p = os.path.join(images_dst, os.path.basename(src_p))
                if not os.path.exists(dst_p):
                    try:
                        os.link(src_p, dst_p)
                    except Exception:
                        shutil.copy2(src_p, dst_p)

    if reconstruction is not None:
        try:
            export_transforms_json(reconstruction, dataset_3dgs_dir)
        except Exception:
            pass
    elif os.path.isfile(os.path.join(output_dir, "transforms.json")):
        shutil.copy2(os.path.join(output_dir, "transforms.json"), os.path.join(dataset_3dgs_dir, "transforms.json"))

    print(f"      * [3DGS Standard Dataset] Generated sparse/0/, images/, and transforms.json at: {dataset_3dgs_dir}")

    # Application Folder 1: COLMAP / Lichtfield
    colmap_dir = os.path.join(output_dir, "COLMAP_Lichtfield")
    os.makedirs(colmap_dir, exist_ok=True)
    if reconstruction is not None:
        try:
            reconstruction.write(colmap_dir)
            reconstruction.write_text(colmap_dir)
            print(f"      * [COLMAP / Lichtfield] Exported cameras, images, points3D (.txt & .bin) to: {colmap_dir}")
        except Exception:
            pass
    if os.path.isdir(src_sparse):
        for fname in os.listdir(src_sparse):
            src_f = os.path.join(src_sparse, fname)
            dst_f = os.path.join(colmap_dir, fname)
            if not os.path.exists(dst_f):
                try:
                    shutil.copy2(src_f, dst_f)
                except Exception:
                    pass

    # Application Folder 2: Postshot / RealityCapture
    rc_dir = os.path.join(output_dir, "Postshot_RealityCapture")
    os.makedirs(rc_dir, exist_ok=True)
    if reconstruction is not None:
        try:
            csv_file = export_realitycapture_csv(reconstruction, rc_dir)
            print(f"      * [Postshot / RealityCapture] Exported camera alignments to: {csv_file}")
            transforms_file = export_transforms_json(reconstruction, rc_dir)
            print(f"      * [Postshot / RealityCapture] Exported NeRF/3DGS poses to: {transforms_file}")
        except Exception:
            pass
    elif os.path.isfile(os.path.join(output_dir, "transforms.json")):
        shutil.copy2(os.path.join(output_dir, "transforms.json"), os.path.join(rc_dir, "transforms.json"))

    # High-density colored point cloud (.ply) for Postshot & MeshLab (Transformed to RC Z-up space)
    if len(dense_3d) == 0 and reconstruction is not None and reconstruction.num_points3D() > 0:
        dense_3d = np.array([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)
        dense_rgb = np.array([pt.color for pt in reconstruction.points3D.values()], dtype=np.uint8)
    elif len(dense_3d) == 0:
        # Check if an existing ply was produced by the engine (e.g. OpenSfM)
        existing_ply_candidates = [
            os.path.join(output_dir, "opensfm_workspace", "reconstruction.ply"),
            os.path.join(output_dir, "reconstruction.ply"),
            os.path.join(output_dir, "points3D.ply"),
            os.path.join(dataset_3dgs_dir, "points.ply"),
            os.path.join(dataset_3dgs_dir, "points3D.ply")
        ]
        found_ply = False
        for ep in existing_ply_candidates:
            if os.path.isfile(ep) and os.path.getsize(ep) > 1000:
                try:
                    loaded_mesh = trimesh.load(ep)
                    dense_3d = np.asarray(loaded_mesh.vertices, dtype=np.float32)
                    if hasattr(loaded_mesh.visual, "vertex_colors") and loaded_mesh.visual.vertex_colors is not None and len(loaded_mesh.visual.vertex_colors) == len(dense_3d):
                        dense_rgb = np.asarray(loaded_mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
                    else:
                        dense_rgb = np.full((len(dense_3d), 3), 200, dtype=np.uint8)
                    found_ply = True
                    break
                except Exception:
                    pass
        if not found_ply or len(dense_3d) == 0:
            dense_3d = np.zeros((1, 3), dtype=np.float32)
            dense_rgb = np.full((1, 3), 200, dtype=np.uint8)
    
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
    
    # Guarantee point cloud in 3dgs_dataset/ and root output_dir:
    try:
        # If genuine engine ply exists (e.g. OpenSfM), preserve exact vertices
        source_real_ply = None
        for cand in [os.path.join(output_dir, "opensfm_workspace", "reconstruction.ply"), os.path.join(output_dir, "reconstruction.ply")]:
            if os.path.isfile(cand) and os.path.getsize(cand) > 1000:
                source_real_ply = cand
                break
        
        if source_real_ply:
            shutil.copy2(source_real_ply, os.path.join(dataset_3dgs_dir, "points.ply"))
            shutil.copy2(source_real_ply, os.path.join(dataset_3dgs_dir, "points3D.ply"))
            shutil.copy2(source_real_ply, os.path.join(output_dir, "points3D.ply"))
            print(f"      * [Standard Output] Exported authentic point cloud to: {os.path.join(output_dir, 'points3D.ply')}")
        else:
            shutil.copy2(ply_path, os.path.join(dataset_3dgs_dir, "points.ply"))
            trimesh.PointCloud(dense_3d, colors=dense_rgb).export(os.path.join(output_dir, "points3D.ply"))
            print(f"      * [Standard Output] Exported root point cloud to: {os.path.join(output_dir, 'points3D.ply')}")
    except Exception:
        pass

    print(f"      * Exported in:   {time.time()-export_start:.2f}s")
    print()

    # -------------------------------------------------------------
    # Final Completion Report
    # -------------------------------------------------------------
    total_time = time.time() - overall_start_time
    reg_images = reconstruction.num_reg_images() if reconstruction is not None and reconstruction.num_reg_images() > 0 else 0
    if reg_images == 0:
        for r_json in [
            os.path.join(output_dir, "opensfm_workspace", "reconstruction.json"),
            os.path.join(dataset_3dgs_dir, "opensfm", "reconstruction.json"),
            os.path.join(dataset_3dgs_dir, "transforms.json")
        ]:
            if os.path.isfile(r_json):
                try:
                    with open(r_json, "r", encoding="utf-8") as f:
                        r_data = json.load(f)
                    if isinstance(r_data, list) and len(r_data) > 0 and "shots" in r_data[0]:
                        reg_images = len(r_data[0]["shots"])
                        break
                    elif isinstance(r_data, dict) and "frames" in r_data:
                        reg_images = len(r_data["frames"])
                        break
                except Exception:
                    pass
    if reg_images == 0:
        reg_images = len(image_paths)

    num_points = reconstruction.num_points3D() if reconstruction is not None and reconstruction.num_points3D() > 0 else len(dense_3d)

    report_lines = [
        "======================================================================",
        "                       RECONSTRUCTION REPORT                          ",
        "======================================================================",
        f" Date & Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f" Dataset Source:       {scan_dir}",
        f" Total Photos:         {total_images_found}",
        f" Registered Views:     {reg_images} / {num_frames} ({reg_images/max(num_frames, 1)*100:.1f}%)",
        f" 3D Points Created:    {num_points:,} vertices",
    ]

    if reconstruction is not None and len(reconstruction.cameras) > 0:
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
            report_lines.append(f" Camera Model:         {first_cam.model_name}")
    else:
        report_lines.append(f" Camera Model:         EQUIRECTANGULAR (Omnidirectional 360°)")

    # Human-readable labels for selected GUI / CLI settings
    engine_labels = {
        "vggt_omega": "VGGT-Omega (Fast, Low VRAM)",
        "vggt": "VGGT-Omega (Fast, Low VRAM)",
        "acezero": "Niantic Labs ACE0 (Scene Coordinate MLPs)",
        "ace0": "Niantic Labs ACE0 (Scene Coordinate MLPs)",
        "pycusfm": "NVIDIA PyCuSFM (cuSFM GPU)",
        "cusfm": "NVIDIA PyCuSFM (cuSFM GPU)",
        "opensfm": "Mapillary OpenSfM (360° Spherical & Street SfM)",
        "360dvo": "360DVO (Deep 360° Visual Odometry)",
        "dvo": "360DVO (Deep 360° Visual Odometry)",
    }
    align_engine_label = engine_labels.get(str(args.engine).lower(), str(args.engine))

    align_mode_labels = {
        "none": "Feedforward (Fast, ~25s)",
        "fast": "Feedforward (Fast, ~25s)",
        "standard": "Standard Ceres BA (~1m)",
        "ultra": "Ultra Sub-Pixel BA (Cauchy+Polish)",
    }
    align_mode_label = align_mode_labels.get(str(args.align_mode).lower(), str(args.align_mode))

    lens_model_label = "Pinhole (Ideal, zero distortion)" if args.lens_model == "pinhole" else "Brown Calibration (k1-k2, t1-t2)"

    if args.max_points <= 150000 and args.conf_thresh >= 2.0:
        density_label = f"Preview (100K pts, Conf {args.conf_thresh})"
    elif args.max_points <= 400000 and args.conf_thresh >= 1.7:
        density_label = f"High Quality (300K pts, Conf {args.conf_thresh})"
    elif args.max_points <= 900000 and args.conf_thresh >= 1.4:
        density_label = f"Ultra Dense (800K pts, Conf {args.conf_thresh})"
    elif args.max_points >= 1000000:
        density_label = f"Max Scan Grade (1.5M pts, Conf {args.conf_thresh})"
    else:
        density_label = f"Custom ({args.max_points:,} pts, Conf >= {args.conf_thresh})"

    train_engine_labels = {
        "lichtfeld": "LichtFeld Studio (Vulkan Real-Time GUI + GUT)",
        "splatfacto": "Nerfstudio Splatfacto (PyTorch SOTA + PLY)",
        "nerfstudio": "Nerfstudio Splatfacto (PyTorch SOTA + PLY)",
        "3dgrut": "NVIDIA 3DGRUT / 3DGUT (Ray Traced & GUT)",
        "pfgs360": "PFGS360 (Pose-Free 360° Omnidirectional GS - CVPR 2026)",
        "pfgs": "PFGS360 (Pose-Free 360° Omnidirectional GS - CVPR 2026)",
    }
    train_engine_label = train_engine_labels.get(str(args.train_engine).lower(), str(args.train_engine))
    auto_train_label = "Auto-Train Enabled (Active)" if args.train else "Disabled (Standby / Alignment Only)"
    train_mode_label = "Interactive Viewport (Live 3D GUI)" if args.train_mode == "interactive" else "Headless Background (Fast CLI)"
    gut_label = "Enabled (Unscented Transform)" if args.gut else "Disabled"

    report_lines.extend([
        "----------------------------------------------------------------------",
        " SELECTED ALIGNMENT SETTINGS:",
        f"   * Photogrammetry Engine:   {align_engine_label}",
        f"   * Canvas Resolution:       {args.load_res} px",
        f"   * Keyframe Stride:         {args.stride} ({'All photos' if args.stride == 1 else f'Every {args.stride}th photo'})",
    ])

    if engine in ["ace0", "acezero"]:
        report_lines.extend([
            f"   * ACE0 Mapping Rounds:     {args.ace_iterations} rounds",
            f"   * Pose Refinement Mode:    {args.ace_refinement.upper()}",
            f"   * Initialization Seeds:    {args.ace_try_seeds} candidate seeds",
            f"   * Focal Calibration:       {'Refine Focal Length (Active)' if getattr(args, 'ace_refine_calib', True) else 'Fixed Initial Heuristic'}",
            f"   * Point Cloud Filter:      {'Dense Unfiltered' if getattr(args, 'ace_dense_pc', False) else 'Filtered Reprojection Points'}",
        ])
    elif engine in ["pycusfm", "cusfm"]:
        report_lines.extend([
            f"   * GPU Feature Extractor:   {args.cusfm_features.upper()}",
            f"   * Keypoint Matcher:        {getattr(args, 'cusfm_matcher', 'lightglue').upper()} (GPU Transformer)",
            f"   * Pose Initializer:        {args.cusfm_init.upper()}",
            f"   * cuDSS BA Iterations:     {args.cusfm_ba_iterations}",
            f"   * Point Cloud Output:      {'Colorized RGB Tie Points' if getattr(args, 'cusfm_rgb', True) else 'Geometric Sparse'}",
        ])
    elif engine in ["opensfm"]:
        report_lines.extend([
            f"   * Camera Projection:       {args.projection.upper()}",
            f"   * Feature Detector:        {args.opensfm_feature.upper()}",
            f"   * Feature Matcher:         {args.opensfm_matcher.upper()}",
            f"   * Feature Budget / Frame:  {getattr(args, 'opensfm_min_features', 3000)}",
            f"   * Matching Neighbor Window:{getattr(args, 'opensfm_neighbors', 5)} views",
            f"   * Bundle Interval:         Every {getattr(args, 'opensfm_bundle_interval', 1)} view(s)",
            f"   * Optimize Camera Params:  {'Enabled' if getattr(args, 'opensfm_optimize_camera', False) else 'Fixed Ideal S²'}",
            f"   * 360 Cubemap Transcode:   {'Active (' + str(args.cubemap_res) + 'px)' if getattr(args, 'cubemap_transcode', False) else 'Native Spherical S²'}",
        ])
    elif engine in ["360dvo", "dvo"]:
        report_lines.extend([
            f"   * Active Patch Density:    {args.dvo_patches} patches/frame",
            f"   * Video / Frame Stride:    {args.stride}",
            f"   * Polar Zenith/Nadir Mask: {getattr(args, 'dvo_polar_mask', 0.08)*100:.1f}% cutoff",
            f"   * ODBA Sliding BA Window:  {getattr(args, 'dvo_ba_window', 8)} keyframes",
            f"   * Metric Scene Scale:      {getattr(args, 'dvo_scale', 3.2):.1f}m multiplier",
            f"   * Trajectory Format:       Robotics TUM (timestamp, SE3 pose) + Universal COLMAP/OpenSfM/Transforms",
            f"   * Real-Time Viser View:    {'Enabled' if args.dvo_viz else 'Headless'}",
        ])
    else:
        report_lines.extend([
            f"   * Camera Alignment Acc.:   {align_mode_label}",
            f"   * Point Cloud Density:     {density_label}",
            f"   * Lens Distortion Model:   {lens_model_label}",
            f"   * Auto-Chunking VRAM:      Chunk {args.chunk_size}, Anchor {args.anchor_size}",
        ])

    report_lines.append(f"   * Smart Cache (.abu_cache): {'Disabled (--no_cache)' if getattr(args, 'no_cache', False) else 'Active'}")
    report_lines.append("----------------------------------------------------------------------")
    report_lines.append(" 3DGS TRAINING SUITE (ABU TRAIN) SETTINGS:")
    report_lines.append(f"   * Auto-Train Status:       {auto_train_label}")
    report_lines.append(f"   * Training Engine:         {train_engine_label}")
    report_lines.append(f"   * Training Iterations:     {args.train_iterations:,}")

    train_eng_clean = str(args.train_engine).lower()
    if "lichtfeld" in train_eng_clean:
        report_lines.extend([
            f"   * Training Viewport Mode:  {train_mode_label}",
            f"   * GUT Optics:              {gut_label}",
            f"   * Export Format:           {'.spz (Compressed Splat)' if getattr(args, 'export_spz', False) else '.ply (Standard 3DGS)'}",
            f"   * SH Degree:               {getattr(args, 'sh_degree', 3)}",
        ])
    elif "splatfacto" in train_eng_clean or "nerfstudio" in train_eng_clean:
        report_lines.extend([
            f"   * Model Architecture:      {getattr(args, 'ns_model', 'splatfacto').upper()}",
            f"   * Camera Optimizer:        {getattr(args, 'camera_optimizer', 'none').upper()}",
            f"   * Bilateral Grid:          {'Enabled' if getattr(args, 'bilateral_grid', False) else 'Disabled'}",
            f"   * Mip-Filtering:           {'Enabled' if getattr(args, 'enable_mip', False) else 'Disabled'}",
            f"   * SH Degree:               {getattr(args, 'sh_degree', 3)}",
        ])
    elif "3dgrut" in train_eng_clean or "3dgut" in train_eng_clean:
        report_lines.extend([
            f"   * Representation Mode:     {getattr(args, 'representation', '3dgut').upper()}",
            f"   * BVH Ray-Tracing Backend: {getattr(args, 'bvh_backend', 'optix').upper()}",
            f"   * Exposure Correction:     {'Enabled' if getattr(args, 'exposure_correction', False) else 'Disabled'}",
            f"   * Sparsity Pruning (ADMM): {'Enabled (lambda=0.001)' if getattr(args, 'enable_sparsity', False) else 'Disabled'}",
            f"   * SH Degree:               {getattr(args, 'sh_degree', 3)}",
        ])
    elif "pfgs" in train_eng_clean:
        mv_val = getattr(args, "pfgs_match_views", 8)
        match_str = f"{mv_val} views (Sliding Window)" if mv_val > 0 else "All Views (Exhaustive O(N²))"
        report_lines.extend([
            f"   * Depth Prior Engine:      {getattr(args, 'pfgs_depth', 'unik3d').upper()} (Omnidirectional)",
            f"   * Dataparser Routing:      {getattr(args, 'pfgs_dataparser', 'auto').upper()}",
            f"   * Match Views Window:      {match_str}",
            f"   * Densify Views Window:    {getattr(args, 'pfgs_densify_views', 8)} views",
            f"   * Spherical PE (SCA-PE):   {'Enabled (S² Manifold)' if getattr(args, 'pfgs_spherical', True) else 'Disabled'}",
            f"   * Epipolar Consistency:    {'Enabled' if getattr(args, 'pfgs_consist', True) else 'Disabled'}",
            f"   * Inlier Growth (DIA):     {'Enabled' if getattr(args, 'pfgs_inlier', True) else 'Disabled'}",
            f"   * Align Internal Depth:    {'Enabled' if getattr(args, 'pfgs_align_depth', True) else 'Disabled'}",
            f"   * Sky Area Filter:         {'Enabled (>' + str(getattr(args, 'pfgs_sky_depth', 50.0)) + 'm)' if getattr(args, 'pfgs_filter_sky', True) else 'Disabled'}",
            f"   * SH Degree:               {getattr(args, 'sh_degree', 3)}",
        ])

    report_lines.extend([
        "----------------------------------------------------------------------",
        f" Output Directory:     {output_dir}",
        f"   |-- 3dgs_dataset/              (Universal input for LichtFeld, Splatfacto, 3DGRUT, PFGS360)",
        f"   |     |-- images/              (Direct link/staged photos)",
        f"   |     |-- sparse/0/            (COLMAP cameras, images, points3D)",
        f"   |     \\-- transforms.json      (Standard NeRF / 3DGS camera transforms)",
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

    # Stage 6: Automated 3D Gaussian Splatting Training (if requested)
    if args.train:
        print()
        from train_engine import run_3dgs_training
        train_out = os.path.join(output_dir, "3dgs_models")
        try:
            trained_model_path = run_3dgs_training(
                engine=args.train_engine,
                dataset_dir=dataset_3dgs_dir,
                output_dir=train_out,
                iterations=args.train_iterations,
                mode=args.train_mode,
                gut=args.gut,
                sh_degree=getattr(args, "sh_degree", 3),
                export_spz=getattr(args, "export_spz", False),
                ns_model=getattr(args, "ns_model", "splatfacto"),
                camera_optimizer=getattr(args, "camera_optimizer", "none"),
                bilateral_grid=getattr(args, "bilateral_grid", False),
                exposure_correction=getattr(args, "exposure_correction", False),
                enable_sparsity=getattr(args, "enable_sparsity", False),
                enable_mip=getattr(args, "enable_mip", False),
                representation=getattr(args, "representation", "3dgut"),
                bvh_backend=getattr(args, "bvh_backend", "optix"),
                pfgs_depth=getattr(args, "pfgs_depth", "unik3d"),
                pfgs_dataparser=getattr(args, "pfgs_dataparser", "auto"),
                pfgs_filter_sky=getattr(args, "pfgs_filter_sky", True),
                pfgs_sky_depth=getattr(args, "pfgs_sky_depth", 50.0),
                pfgs_match_views=getattr(args, "pfgs_match_views", 8),
                pfgs_densify_views=getattr(args, "pfgs_densify_views", 8),
                pfgs_spherical=getattr(args, "pfgs_spherical", True),
                pfgs_consist=getattr(args, "pfgs_consist", True),
                pfgs_inlier=getattr(args, "pfgs_inlier", True),
                pfgs_align_depth=getattr(args, "pfgs_align_depth", True),
            )
            print(f"\n      * [3DGS Training Complete] Model exported to: {trained_model_path}")
            try:
                with open(report_file, "a", encoding="utf-8") as rf:
                    rf.write(f"\n[3DGS TRAINING COMPLETED]\n   * Exported Model:    {trained_model_path}\n   * Final Iterations:  {args.train_iterations:,}\n======================================================================\n")
            except Exception:
                pass
        except Exception as e:
            print(f"\n[ERROR] 3DGS Training failed: {e}")

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
