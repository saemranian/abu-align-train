"""
Abu Align & Train - Interactive 3D Browser Viewer (Viser)
Inspection suite for 3D point clouds, camera trajectories, and 3D Gaussian Splats.
Features multi-run discovery, automatic latest-run tracking, dataset provenance cards,
trajectory path rendering, and adaptive outlier clipping.
"""
import os
import sys
import time
import glob
import json
import math
import struct
import argparse
import webbrowser
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple, Optional

# =========================================================================
# Robust Point Cloud & Camera Parsers
# =========================================================================

def read_ply_data(ply_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    High-performance PLY reader supporting ASCII and Binary Little-Endian,
    with automatic fallbacks (numpy buffer -> plyfile -> trimesh -> ASCII).
    """
    if not os.path.isfile(ply_path):
        return None, None

    # Fast binary numpy reader
    try:
        with open(ply_path, 'rb') as f:
            header = []
            while True:
                raw = f.readline()
                if not raw:
                    return None, None
                line = raw.decode('utf-8', errors='ignore').strip()
                header.append(line)
                if line == 'end_header':
                    break

            format_line = [h for h in header if h.startswith('format')][0]
            element_line = [h for h in header if h.startswith('element vertex')][0]
            num_verts = int(element_line.split()[-1])

            if 'binary_little_endian' in format_line:
                props = [h.split() for h in header if h.startswith('property')]
                dt = []
                for pr in props:
                    t = pr[1]
                    name = pr[2]
                    if t in ['float', 'float32']: dt.append((name, '<f4'))
                    elif t in ['uchar', 'uint8']: dt.append((name, 'u1'))
                    elif t in ['int', 'int32']: dt.append((name, '<i4'))
                    elif t in ['double', 'float64']: dt.append((name, '<f8'))
                    else: dt.append((name, '<f4'))

                buf = f.read()
                data = np.frombuffer(buf, dtype=np.dtype(dt), count=num_verts)
                pts = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)

                if 'red' in data.dtype.names and 'green' in data.dtype.names and 'blue' in data.dtype.names:
                    cols = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.uint8)
                else:
                    cols = np.full((len(pts), 3), 180, dtype=np.uint8)
                return pts, cols

            elif 'ascii' in format_line:
                pts = []
                cols = []
                for _ in range(num_verts):
                    parts = f.readline().decode('utf-8', errors='ignore').strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
                        else:
                            cols.append([180, 180, 180])
                return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)
    except Exception:
        pass

    # Fallback to plyfile or trimesh
    try:
        from plyfile import PlyData
        pdata = PlyData.read(ply_path)
        v = pdata['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        if 'red' in v and 'green' in v and 'blue' in v:
            cols = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)
        return pts, cols
    except Exception:
        pass

    try:
        import trimesh
        mesh = trimesh.load(ply_path)
        pts = np.array(mesh.vertices, dtype=np.float32)
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            cols = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)
        return pts, cols
    except Exception as e:
        print(f"[Warning] Failed to read PLY {ply_path}: {e}")
        return None, None


def load_cameras_from_tum(traj_path: str) -> List[Dict]:
    """Loads camera poses from robotics TUM trajectory file: timestamp tx ty tz qx qy qz qw."""
    poses = []
    if not os.path.isfile(traj_path):
        return poses
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    ts = parts[0]
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    poses.append({
                        "name": f"cam_{ts}",
                        "pos": np.array([tx, ty, tz], dtype=np.float32),
                        "wxyz": np.array([qw, qx, qy, qz], dtype=np.float32)
                    })
    except Exception as e:
        print(f"[Warning] Failed to parse TUM trajectory {traj_path}: {e}")
    return poses


def load_cameras_from_colmap_images(images_txt_path: str) -> List[Dict]:
    """
    Loads camera poses from standard COLMAP images.txt format.
    COLMAP images.txt uses 2 lines per image:
    Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    Line 2: POINTS2D[] as (X, Y, POINT3D_ID)...
    """
    poses = []
    if not os.path.isfile(images_txt_path):
        return poses
    try:
        with open(images_txt_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                # Skip the following 2D points measurement line in standard COLMAP format
                f.readline()

                if len(parts) >= 10:
                    try:
                        qw = float(parts[1])
                        qx = float(parts[2])
                        qy = float(parts[3])
                        qz = float(parts[4])
                        tx = float(parts[5])
                        ty = float(parts[6])
                        tz = float(parts[7])
                        name = " ".join(parts[9:])
                    except (ValueError, IndexError):
                        continue

                    # Validate and normalize quaternion
                    q_norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                    if q_norm < 1e-4:
                        continue
                    qw, qx, qy, qz = qw / q_norm, qx / q_norm, qy / q_norm, qz / q_norm

                    # Convert w2c to c2w
                    R_w2c = np.array([
                        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
                        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
                        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
                    ], dtype=np.float64)
                    t_w2c = np.array([tx, ty, tz], dtype=np.float64)

                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c

                    if not np.all(np.isfinite(t_c2w)) or np.linalg.norm(t_c2w) > 50000.0:
                        continue

                    # R_c2w to quaternion wxyz
                    qx_c, qy_c, qz_c, qw_c = Rotation.from_matrix(R_c2w).as_quat()

                    poses.append({
                        "name": name.replace("\\", "/").removeprefix("./"),
                        "pos": t_c2w.astype(np.float32),
                        "wxyz": np.array([qw_c, qx_c, qy_c, qz_c], dtype=np.float32)
                    })
    except Exception as e:
        print(f"[Warning] Failed to parse COLMAP images.txt {images_txt_path}: {e}")
    return poses


def load_cameras_from_transforms(tf_path: str) -> List[Dict]:
    """Loads camera poses from a transforms.json file."""
    poses = []
    if not os.path.isfile(tf_path):
        return poses
    try:
        with open(tf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        for idx, frame in enumerate(frames):
            c2w = np.array(frame.get("transform_matrix", np.eye(4)), dtype=np.float32)
            # NeRF transforms use OpenGL (-Z forward, +Y up); Viser uses OpenCV.
            R = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            t = c2w[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
            name = frame.get("file_path", f"frame_{idx:04d}").replace("\\", "/").removeprefix("./")
            poses.append({
                "name": name,
                "pos": t,
                "wxyz": np.array([qw, qx, qy, qz], dtype=np.float32)
            })
    except Exception as e:
        print(f"[Warning] Failed to parse transforms.json {tf_path}: {e}")
    return poses


def load_cameras_from_colmap_bin(images_bin_path: str) -> List[Dict]:
    """Loads camera poses from binary COLMAP images.bin file: num_reg_images uint64, then records."""
    poses = []
    if not os.path.isfile(images_bin_path):
        return poses
    try:
        with open(images_bin_path, 'rb') as f:
            num_reg_images = struct.unpack('<Q', f.read(8))[0]
            for _ in range(num_reg_images):
                img_id = struct.unpack('<I', f.read(4))[0]
                qw, qx, qy, qz = struct.unpack('<4d', f.read(32))
                tx, ty, tz = struct.unpack('<3d', f.read(24))
                cam_id = struct.unpack('<I', f.read(4))[0]
                name_chars = []
                while True:
                    ch = f.read(1)
                    if ch == b'\x00' or not ch:
                        break
                    name_chars.append(ch)
                name = b''.join(name_chars).decode('utf-8', errors='ignore')
                num_pts = struct.unpack('<Q', f.read(8))[0]
                f.seek(num_pts * 24, os.SEEK_CUR)

                # Convert w2c to c2w
                R_w2c = np.array([
                    [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
                    [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
                    [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
                ], dtype=np.float64)
                t_w2c = np.array([tx, ty, tz], dtype=np.float64)

                R_c2w = R_w2c.T
                t_c2w = -R_c2w @ t_w2c

                qx_c, qy_c, qz_c, qw_c = Rotation.from_matrix(R_c2w).as_quat()

                poses.append({
                    "name": name.replace("\\", "/").removeprefix("./"),
                    "pos": t_c2w.astype(np.float32),
                    "wxyz": np.array([qw_c, qx_c, qy_c, qz_c], dtype=np.float32)
                })
    except Exception as e:
        print(f"[Warning] Failed to parse COLMAP images.bin {images_bin_path}: {e}")
    return poses


def load_cameras_from_opensfm(opensfm_path: str) -> List[Dict]:
    """Loads camera poses from OpenSfM reconstruction.json file."""
    poses = []
    if not os.path.isfile(opensfm_path):
        return poses
    try:
        with open(opensfm_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reconstructions = data if isinstance(data, list) else [data]
        for rec in reconstructions:
            shots = rec.get("shots", {})
            for name, shot in shots.items():
                r = shot.get("rotation", [0, 0, 0])
                t = shot.get("translation", [0, 0, 0])
                rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
                tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

                theta = math.sqrt(rx*rx + ry*ry + rz*rz)
                if theta > 1e-7:
                    kx, ky, kz = rx / theta, ry / theta, rz / theta
                    c = math.cos(theta)
                    s = math.sin(theta)
                    v = 1.0 - c
                    R_w2c = np.array([
                        [kx*kx*v + c,      kx*ky*v - kz*s,  kx*kz*v + ky*s],
                        [kx*ky*v + kz*s,  ky*ky*v + c,      ky*kz*v - kx*s],
                        [kx*kz*v - ky*s,  ky*kz*v + kx*s,  kz*kz*v + c]
                    ], dtype=np.float64)
                else:
                    R_w2c = np.eye(3, dtype=np.float64)

                t_w2c = np.array([tx, ty, tz], dtype=np.float64)
                R_c2w = R_w2c.T
                t_c2w = -R_c2w @ t_w2c

                qx_c, qy_c, qz_c, qw_c = Rotation.from_matrix(R_c2w).as_quat()

                poses.append({
                    "name": name.replace("\\", "/").removeprefix("./"),
                    "pos": t_c2w.astype(np.float32),
                    "wxyz": np.array([qw_c, qx_c, qy_c, qz_c], dtype=np.float32)
                })
    except Exception as e:
        print(f"[Warning] Failed to parse OpenSfM reconstruction.json {opensfm_path}: {e}")
    return poses


def load_cameras_from_ace0(ace0_path: str) -> List[Dict]:
    """Loads camera poses from ACE0 poses_final.txt or poses_subpixel.txt."""
    poses = []
    if not os.path.isfile(ace0_path):
        return poses
    try:
        with open(ace0_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 10:
                    raw_img_path = " ".join(tokens[:-9])
                    qw = float(tokens[-9])
                    qx = float(tokens[-8])
                    qy = float(tokens[-7])
                    qz = float(tokens[-6])
                    tx = float(tokens[-5])
                    ty = float(tokens[-4])
                    tz = float(tokens[-3])

                    R_w2c = np.array([
                        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
                        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
                        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
                    ], dtype=np.float64)
                    t_w2c = np.array([tx, ty, tz], dtype=np.float64)

                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c

                    qx_c, qy_c, qz_c, qw_c = Rotation.from_matrix(R_c2w).as_quat()

                    poses.append({
                        "name": os.path.basename(raw_img_path),
                        "pos": t_c2w.astype(np.float32),
                        "wxyz": np.array([qw_c, qx_c, qy_c, qz_c], dtype=np.float32)
                    })
    except Exception as e:
        print(f"[Warning] Failed to parse ACE0 pose file {ace0_path}: {e}")
    return poses


def load_cameras_from_cameras_csv(csv_path: str) -> List[Dict]:
    """Loads camera positions from RealityCapture / Postshot cameras.csv."""
    poses = []
    if not os.path.isfile(csv_path):
        return poses
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#") or header is None:
                    header = [h.strip().lstrip("#").lower() for h in line.split(",")]
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != len(header):
                    continue
                row = dict(zip(header, parts))
                name = row.get("name", "cam")
                if "x" in row and "y" in row and "alt" in row:
                    x = float(row["x"])
                    y = float(row["y"])
                    z = float(row["alt"])
                    poses.append({
                        "name": name.replace("\\", "/").removeprefix("./"),
                        "pos": np.array([x, y, z], dtype=np.float32),
                        "wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    })
    except Exception as e:
        print(f"[Warning] Failed to parse cameras.csv {csv_path}: {e}")
    return poses


def parse_run_report(run_dir: str) -> Dict[str, str]:
    """Extracts metadata from reconstruction_report.txt if available."""
    meta = {
        "engine": "Unknown",
        "date": "Unknown",
        "log_file": "N/A",
        "log_location": "N/A",
        "dataset": "",
        "photos": "0",
        "registered": "0",
        "points": "0"
    }
    report_path = os.path.join(run_dir, "reconstruction_report.txt")
    if not os.path.isfile(report_path):
        # Infer engine from run dir name
        name = os.path.basename(run_dir).lower()
        if "360dvo" in name: meta["engine"] = "360DVO (Deep Visual Odometry)"
        elif "opensfm" in name: meta["engine"] = "OpenSfM (Equirectangular SfM)"
        elif "vggt" in name: meta["engine"] = "VGGT-Omega (Feedforward)"
        elif "ace0" in name: meta["engine"] = "ACE0 (Visual Relocalization)"
        elif "pycusfm" in name: meta["engine"] = "NVIDIA PyCuSFM (GPU SfM)"
        return meta

    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip().lstrip("*").strip()
                if "date & time:" in line.lower():
                    meta["date"] = line.split(":", 1)[1].strip()
                elif "log file:" in line.lower():
                    meta["log_file"] = line.split(":", 1)[1].strip()
                elif "log location:" in line.lower():
                    meta["log_location"] = line.split(":", 1)[1].strip()
                elif "photogrammetry engine:" in line.lower():
                    meta["engine"] = line.split(":", 1)[1].strip()
                elif "dataset source:" in line.lower():
                    meta["dataset"] = line.split(":", 1)[1].strip()
                elif "total photos:" in line.lower():
                    meta["photos"] = line.split(":", 1)[1].strip()
                elif "registered views:" in line.lower():
                    meta["registered"] = line.split(":", 1)[1].strip()
                elif "3d points created:" in line.lower():
                    meta["points"] = line.split(":", 1)[1].strip()
    except Exception:
        pass

    if meta["engine"] == "Unknown":
        name = os.path.basename(run_dir).lower()
        if "360dvo" in name: meta["engine"] = "360DVO (Deep Visual Odometry)"
        elif "opensfm" in name: meta["engine"] = "OpenSfM (Equirectangular SfM)"
        elif "vggt" in name: meta["engine"] = "VGGT-Omega (Feedforward)"
        elif "ace0" in name: meta["engine"] = "ACE0 (Visual Relocalization)"
        elif "pycusfm" in name: meta["engine"] = "NVIDIA PyCuSFM (GPU SfM)"
    return meta


# =========================================================================
# Run Discovery & Workspace Resolution
# =========================================================================

def resolve_project_structure(target_dir: str) -> Tuple[str, str, Dict[str, str], Optional[str]]:
    """
    Resolves project name, dataset root path, dictionary of all runs in results/,
    and the initially selected run name.
    Performs upward directory traversal so that passing any subfolder (e.g. 3dgs_models/lichtfeld_models)
    properly identifies the results/ directory and discovers all alignment runs.
    Returns (project_title, dataset_dir, runs_dict, initial_selected).
    """
    target_dir = os.path.abspath(target_dir)

    results_dir = None
    dataset_dir = None
    child_run_name = None

    curr = target_dir
    while True:
        parent = os.path.dirname(curr)
        if os.path.basename(curr).lower() == "results":
            results_dir = curr
            dataset_dir = parent
            break
        if parent and os.path.basename(parent).lower() == "results":
            results_dir = parent
            dataset_dir = os.path.dirname(parent)
            child_run_name = os.path.basename(curr)
            break
        if not parent or parent == curr:
            break
        child_run_name = os.path.basename(curr)
        curr = parent

    # If upward traversal didn't find 'results', check if target_dir has a results/ subfolder
    if not results_dir:
        cand_results = os.path.join(target_dir, "results")
        if os.path.isdir(cand_results):
            dataset_dir = target_dir
            results_dir = cand_results
            child_run_name = None

    # Fallback: treat target_dir as results_dir
    if not results_dir:
        dataset_dir = target_dir
        results_dir = target_dir
        child_run_name = None

    # Derive human-friendly project name from directory hierarchy
    parts = [p for p in dataset_dir.replace("\\", "/").split("/") if p]
    if parts:
        project_title = parts[-1]
    else:
        project_title = "Active Project"

    # Discover runs in results_dir
    runs = {}
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            p = os.path.join(results_dir, entry)
            # Skip hidden files/directories (like .abu_cache) and standalone 3dgs_models dirs
            if not os.path.isdir(p) or entry.startswith(".") or entry.lower() == "3dgs_models":
                continue
            is_candidate = any(tag in entry.lower() for tag in ["sparse", "dvo", "vggt", "opensfm", "ace", "cusfm", "align", "colmap"])
            if not is_candidate:
                has_assets = any(f.endswith((".ply", ".txt", ".bin", ".json")) for f in os.listdir(p)) if os.path.isdir(p) else False
                if has_assets:
                    is_candidate = True

            if is_candidate:
                runs[entry] = p

    # If no runs found inside results_dir, check if target_dir itself contains reconstruction data
    if not runs:
        has_direct_ply = any(f.endswith(".ply") for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)))
        if has_direct_ply or any(f.endswith((".bin", ".txt", ".json")) for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))):
            runs[os.path.basename(target_dir)] = target_dir

    # Sort runs by directory modification time descending (latest first)
    sorted_runs = dict(sorted(runs.items(), key=lambda item: os.path.getmtime(item[1]), reverse=True))

    # Determine initial selected run
    initial_selected = None
    if child_run_name and child_run_name in sorted_runs:
        initial_selected = child_run_name
    elif sorted_runs:
        initial_selected = list(sorted_runs.keys())[0]

    return project_title, dataset_dir, sorted_runs, initial_selected


def find_run_assets(run_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Finds best point cloud (.ply) and camera files across all supported photogrammetry engines.
    Excludes 3DGS model training checkpoints and weights from point cloud selection.
    Returns:
    (best_ply, tum_traj, colmap_images_txt, colmap_images_bin, tf_json, opensfm_json, ace0_poses)
    """
    ply_files = glob.glob(os.path.join(run_dir, "*.ply")) + glob.glob(os.path.join(run_dir, "**", "*.ply"), recursive=True)

    def score_ply(path):
        norm_p = path.replace("\\", "/").lower()
        if "/3dgs_models/" in norm_p or "/checkpoints/" in norm_p or "/weights/" in norm_p:
            return -1000.0
        try:
            sz = os.path.getsize(path)
            if sz == 0:
                return -1000.0
        except Exception:
            return -1000.0

        name = os.path.basename(path).lower()
        score = 10.0
        if name == "points3d.ply": score = 100.0
        elif name == "points.ply": score = 95.0
        elif "360dvo_points" in name: score = 90.0
        elif "reconstruction.ply" in name: score = 85.0
        elif "pc_final" in name or "point_cloud" in name: score = 80.0
        return score + (os.path.getmtime(path) / 1e12)

    valid_plys = [p for p in ply_files if score_ply(p) > 0]
    best_ply = max(valid_plys, key=score_ply) if valid_plys else None

    # Camera sources in order of preference
    # 1. 360DVO TUM Trajectory
    tum_traj = None
    for cand in ["360dvo_trajectory.txt", "trajectory.txt", "poses.txt"]:
        p = os.path.join(run_dir, cand)
        if os.path.isfile(p):
            tum_traj = p
            break

    # 2. COLMAP images.txt
    colmap_images = None
    colmap_cands = [
        os.path.join(run_dir, "COLMAP_Lichtfield", "images.txt"),
        os.path.join(run_dir, "3dgs_dataset", "sparse", "0", "images.txt"),
        os.path.join(run_dir, "sparse", "0", "images.txt"),
        os.path.join(run_dir, "colmap", "images.txt"),
    ]
    for c in colmap_cands:
        if os.path.isfile(c):
            colmap_images = c
            break
    if not colmap_images:
        for p in glob.glob(os.path.join(run_dir, "**", "images.txt"), recursive=True):
            if "/3dgs_models/" not in p.replace("\\", "/").lower():
                colmap_images = p
                break

    # 3. COLMAP images.bin
    colmap_bin = None
    bin_cands = [
        os.path.join(run_dir, "COLMAP_Lichtfield", "images.bin"),
        os.path.join(run_dir, "3dgs_dataset", "sparse", "0", "images.bin"),
        os.path.join(run_dir, "sparse", "0", "images.bin"),
        os.path.join(run_dir, "colmap", "images.bin"),
    ]
    for c in bin_cands:
        if os.path.isfile(c):
            colmap_bin = c
            break
    if not colmap_bin:
        for p in glob.glob(os.path.join(run_dir, "**", "images.bin"), recursive=True):
            if "/3dgs_models/" not in p.replace("\\", "/").lower():
                colmap_bin = p
                break

    # 4. transforms.json
    tf_json = None
    tf_cands = [
        os.path.join(run_dir, "Postshot_RealityCapture", "transforms.json"),
        os.path.join(run_dir, "transforms.json"),
        os.path.join(run_dir, "3dgs_dataset", "transforms.json"),
    ]
    for tf in tf_cands:
        if os.path.isfile(tf):
            tf_json = tf
            break

    # 5. OpenSfM reconstruction.json
    opensfm_json = None
    osfm_cands = [
        os.path.join(run_dir, "reconstruction.json"),
        os.path.join(run_dir, "opensfm", "reconstruction.json"),
    ]
    for c in osfm_cands:
        if os.path.isfile(c):
            opensfm_json = c
            break

    # 6. ACE0 poses_final.txt
    ace0_poses = None
    ace_cands = [
        os.path.join(run_dir, "acezero_work", "poses_final.txt"),
        os.path.join(run_dir, "poses_final.txt"),
        os.path.join(run_dir, "poses_subpixel.txt"),
    ]
    for c in ace_cands:
        if os.path.isfile(c):
            ace0_poses = c
            break

    return best_ply, tum_traj, colmap_images, colmap_bin, tf_json, opensfm_json, ace0_poses


def load_run_cameras(run_dir: str) -> Tuple[List[Dict], str]:
    """
    Discovers and parses camera poses from any supported Abu Align engine format.
    Prioritizes formats that preserve image filenames (for photo thumbnails and matching),
    falling back to robotics trajectory logs.
    """
    _, tum_traj, colmap_images, colmap_bin, tf_json, opensfm_json, ace0_poses = find_run_assets(run_dir)

    # 1. COLMAP formats (preserve exact filenames & 6-DoF poses)
    if colmap_images:
        cams = load_cameras_from_colmap_images(colmap_images)
        if cams: return cams, "COLMAP images.txt"

    if colmap_bin:
        cams = load_cameras_from_colmap_bin(colmap_bin)
        if cams: return cams, "COLMAP images.bin"

    # 2. transforms.json (NeRF / 3DGS format with filenames)
    if tf_json:
        cams = load_cameras_from_transforms(tf_json)
        if cams: return cams, "transforms.json"

    # 3. OpenSfM reconstruction.json
    if opensfm_json:
        cams = load_cameras_from_opensfm(opensfm_json)
        if cams: return cams, "OpenSfM reconstruction.json"

    # 4. ACE0 poses
    if ace0_poses:
        cams = load_cameras_from_ace0(ace0_poses)
        if cams: return cams, "ACE0 poses"

    # 5. RealityCapture / Postshot cameras.csv
    for csv_cand in [os.path.join(run_dir, "Postshot_RealityCapture", "cameras.csv"), os.path.join(run_dir, "cameras.csv")]:
        if os.path.isfile(csv_cand):
            cams = load_cameras_from_cameras_csv(csv_cand)
            if cams: return cams, "cameras.csv"

    # 6. 360DVO TUM Trajectory (timestamp-based fallback)
    if tum_traj:
        cams = load_cameras_from_tum(tum_traj)
        if cams: return cams, "360DVO Trajectory (TUM)"

    return [], "None"


# =========================================================================
# Main Interactive Viser Viewer
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Abu Align & Train - Interactive 3D Browser Viewer")
    parser.add_argument("--results_dir", "-r", type=str, default="", help="Directory containing reconstruction results")
    parser.add_argument("--input", "-i", type=str, default="", help="Dataset or results directory")
    parser.add_argument("path", nargs="?", default="", help="Optional positional directory path")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    target_dir = args.results_dir or args.input or args.path or os.getcwd()
    target_dir = os.path.abspath(target_dir)

    project_title, dataset_dir, runs, initial_selected = resolve_project_structure(target_dir)

    print("=" * 72)
    print("   Abu Align & Train - Interactive 3D Inspection Viewer (Viser)")
    print(f"   * Project Name:     {project_title}")
    print(f"   * Dataset Root:     {dataset_dir}")
    print(f"   * Discovered Runs:  {len(runs)}")
    print(f"   * Initial Run:      {initial_selected}")
    print(f"   * Preferred Port:   {args.port}")
    print("=" * 72)

    import viser

    server = None
    active_port = args.port
    for offset in range(10):
        try:
            port_to_try = active_port + offset
            server = viser.ViserServer(host="127.0.0.1", port=port_to_try)
            active_port = port_to_try
            break
        except Exception:
            pass

    if server is None:
        print("[ERROR] Could not start Viser server on any port 8080-8089.")
        sys.exit(1)

    url = f"http://localhost:{active_port}"

    import threading
    def _open_browser_async():
        try:
            time.sleep(0.3)
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open_browser_async, daemon=True).start()

    # Configure sleek fixed dark theme matching Abu Align & Train GUI
    server.gui.configure_theme(
        titlebar_content=None,
        control_layout="collapsible",
        control_width="medium",
        dark_mode=True,
        brand_color=(112, 219, 192),  # Shared mint accent.
        show_logo=False,
        # Viser 0.2.23 conditionally calls a React hook when this is false,
        # crashing the entire frontend as the initial theme arrives.
        show_share_button=True
    )

    if not runs:
        runs[os.path.basename(target_dir)] = target_dir

    run_names = list(runs.keys())
    current_run_name = initial_selected if (initial_selected and initial_selected in runs) else run_names[0]

    from viewer_workspace import InspectionWorkspace

    def load_workspace_run(path):
        asset, *_ = find_run_assets(path)
        points, colors = read_ply_data(asset) if asset else (None, None)
        cameras, source = load_run_cameras(path)
        return dict(points=points, colors=colors, cameras=cameras, source=source,
                    asset=os.path.basename(asset) if asset else "No point cloud found")

    workspace = InspectionWorkspace(server, runs, current_run_name, project_title,
        load_workspace_run, lambda: resolve_project_structure(target_dir)[2])

    print(f"\n[READY] Interactive 3D Viewer running at: {url}")
    print(f"Viewing active run: {current_run_name}")
    print("Press Ctrl+C to stop viewer.\n")

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
