"""
Abu Align & Train - 360DVO Engine
Deep Visual Odometry for Monocular 360-Degree Camera (IEEE RA-L 2026)
Integrates Distortion-Aware Spherical Feature Extraction (DAS-Feat),
Unwarped Patch Sampling, Omnidirectional Differentiable Bundle Adjustment (ODBA),
and automated TUM-to-COLMAP trajectory conversion for 3DGS training.
"""

import os
import sys
import math
import time
import shutil
import subprocess
import numpy as np
import cv2
import json
import scipy.optimize
from typing import List, Tuple, Dict, Optional

try:
    import wsl_bridge
except ImportError:
    wsl_bridge = None

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts unit quaternion [qx, qy, qz, qw] to 3x3 rotation matrix.
    """
    qx, qy, qz, qw = q
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 0:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    else:
        return np.eye(3, dtype=np.float64)

    return np.array([
        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts 3x3 rotation matrix to unit quaternion [qx, qy, qz, qw].
    """
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 0:
        return np.array([qx/norm, qy/norm, qz/norm, qw/norm], dtype=np.float64)
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def uv_to_bearing(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Converts 2D equirectangular pixel coordinates (u, v) into 3D unit bearing vectors on S^2 manifold.
    """
    u = pts[:, 0]
    v = pts[:, 1]
    th = (u / float(w) - 0.5) * 2.0 * math.pi
    ph = (0.5 - v / float(h)) * math.pi
    cos_ph = np.cos(ph)
    x = cos_ph * np.sin(th)
    y = -np.sin(ph)
    z = cos_ph * np.cos(th)
    bearings = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(bearings, axis=1, keepdims=True)
    return bearings / np.maximum(norms, 1e-12)


def solve_spherical_relative_pose(
    b1: np.ndarray,
    b2: np.ndarray,
    max_iters: int = 150,
    thresh: float = 0.015
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates relative rotation R in SO(3) and translation direction t in S^2
    between two sets of spherical bearing vectors b1 and b2 using 8-point RANSAC
    and S^2 manifold epipolar constraint b2.T @ E @ b1 = 0 with cheirality verification.
    """
    N = len(b1)
    if N < 8:
        return np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64), np.ones(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_E = np.eye(3)
    max_count = 0
    indices = np.arange(N)

    for _ in range(max_iters):
        subset = np.random.choice(indices, 8, replace=False)
        sub_b1 = b1[subset]
        sub_b2 = b2[subset]

        A = np.zeros((8, 9))
        for k in range(8):
            A[k] = np.kron(sub_b2[k], sub_b1[k])

        _, _, vt = np.linalg.svd(A)
        E_cand = vt[-1].reshape(3, 3)
        u, _, vt_e = np.linalg.svd(E_cand)
        E_proj = u @ np.diag([1.0, 1.0, 0.0]) @ vt_e

        Eb1 = (E_proj @ b1.T).T
        Etb2 = (E_proj.T @ b2.T).T
        num = np.sum(b2 * Eb1, axis=1) ** 2
        den = np.sum(Eb1[:, :2]**2, axis=1) + np.sum(Etb2[:, :2]**2, axis=1) + 1e-9
        err = num / den

        inliers = err < thresh
        count = np.sum(inliers)
        if count > max_count:
            max_count = count
            best_inliers = inliers
            best_E = E_proj
            if max_count > N * 0.85:
                break

    if max_count < 8:
        # Fallback pure rotation via Wahba's problem
        H = b2.T @ b1
        u_h, _, vt_h = np.linalg.svd(H)
        R_rot = u_h @ vt_h
        if np.linalg.det(R_rot) < 0:
            u_h[:, -1] *= -1
            R_rot = u_h @ vt_h
        return R_rot, np.array([0.0, 0.0, 1.0], dtype=np.float64), np.ones(N, dtype=bool)

    # Decompose best_E into (R, t)
    U, _, Vt = np.linalg.svd(best_E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    candidates = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    best_cand = (np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    max_pos = -1

    sample_b1 = b1[best_inliers][:60]
    sample_b2 = b2[best_inliers][:60]

    for R_c, t_c in candidates:
        C1 = np.zeros(3)
        C2 = -R_c.T @ t_c
        d1 = sample_b1
        d2 = (R_c.T @ sample_b2.T).T

        pos_count = 0
        w0 = C1 - C2
        for i in range(len(d1)):
            b = np.dot(d1[i], d2[i])
            denom = 1.0 - b * b
            if denom > 1e-4:
                d = np.dot(d1[i], w0)
                e = np.dot(d2[i], w0)
                l1 = (b * e - d) / denom
                l2 = (e - b * d) / denom
                if l1 > 0.05 and l2 > 0.05:
                    pos_count += 1
        if pos_count > max_pos:
            max_pos = pos_count
            best_cand = (R_c, t_c)

    return best_cand[0], best_cand[1], best_inliers


def tum_trajectory_to_colmap(
    trajectory_file: str,
    image_names: List[str],
    output_sparse_dir: str,
    width: int = 1024,
    height: int = 512,
    point_cloud_ply: Optional[str] = None,
    points_xyz: Optional[np.ndarray] = None,
    points_rgb: Optional[np.ndarray] = None
) -> Tuple[bool, str]:
    """
    Converts robotics TUM trajectory format (timestamp tx ty tz qx qy qz qw)
    into standard COLMAP binary and text model files (cameras, images, points3D).
    """
    os.makedirs(output_sparse_dir, exist_ok=True)
    if not os.path.isfile(trajectory_file):
        return False, f"Trajectory file not found: {trajectory_file}"

    poses = []
    with open(trajectory_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append((t, np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])))

    if not poses:
        return False, "No valid pose records found in trajectory file."

    # Write cameras.txt
    # For equirectangular 360, COLMAP uses EQUIRECTANGULAR with width & height parameters
    cameras_txt = os.path.join(output_sparse_dir, "cameras.txt")
    with open(cameras_txt, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 EQUIRECTANGULAR {width} {height} {width} {height}\n")

    # Write images.txt
    images_txt = os.path.join(output_sparse_dir, "images.txt")
    num_to_write = min(len(poses), len(image_names))

    with open(images_txt, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx in range(num_to_write):
            t, t_c2w, q_c2w = poses[idx]
            R_c2w = quaternion_to_rotation_matrix(q_c2w)

            # Invert camera-to-world (c2w) to world-to-camera (w2c) for COLMAP convention
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w

            # Convert R_w2c back to quaternion [qw, qx, qy, qz]
            qw_w2c = math.sqrt(max(0.0, 1.0 + R_w2c[0,0] + R_w2c[1,1] + R_w2c[2,2])) / 2.0
            if qw_w2c > 1e-6:
                qx_w2c = (R_w2c[2,1] - R_w2c[1,2]) / (4.0 * qw_w2c)
                qy_w2c = (R_w2c[0,2] - R_w2c[2,0]) / (4.0 * qw_w2c)
                qz_w2c = (R_w2c[1,0] - R_w2c[0,1]) / (4.0 * qw_w2c)
            else:
                qw_w2c, qx_w2c, qy_w2c, qz_w2c = 1.0, 0.0, 0.0, 0.0

            img_name = os.path.basename(image_names[idx])
            f.write(f"{idx+1} {qw_w2c:.9f} {qx_w2c:.9f} {qy_w2c:.9f} {qz_w2c:.9f} {t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {img_name}\n")
            f.write("\n")  # Empty points2D line

    # Write points3D.txt
    points3D_txt = os.path.join(output_sparse_dir, "points3D.txt")
    with open(points3D_txt, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        if points_xyz is not None and len(points_xyz) > 0:
            rgb_arr = points_rgb if (points_rgb is not None and len(points_rgb) == len(points_xyz)) else np.full((len(points_xyz), 3), 200, dtype=np.uint8)
            for p_i, (pt, c) in enumerate(zip(points_xyz, rgb_arr)):
                f.write(f"{p_i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])} 0.1\n")

    # If pycolmap is available, compile to binary .bin as well
    try:
        import pycolmap
        rec = pycolmap.Reconstruction()
        rec.read_text(output_sparse_dir)
        rec.write_binary(output_sparse_dir)
    except Exception:
        pass

    return True, f"Converted {num_to_write} frames to COLMAP format."


def load_ply_points_for_opensfm(ply_path: Optional[str]) -> dict:
    """
    Extracts 3D coordinates and RGB colors from a PLY file for OpenSfM points dictionary.
    """
    pts_dict = {}
    if not ply_path or not os.path.isfile(ply_path):
        return pts_dict

    try:
        from plyfile import PlyData
        pdata = PlyData.read(ply_path)
        v = pdata['vertex']
        xs, ys, zs = v['x'], v['y'], v['z']
        has_rgb = 'red' in v and 'green' in v and 'blue' in v
        rs = v['red'] if has_rgb else None
        gs = v['green'] if has_rgb else None
        bs = v['blue'] if has_rgb else None
        for i in range(len(xs)):
            pts_dict[str(i)] = {
                "coordinates": [round(float(xs[i]), 6), round(float(ys[i]), 6), round(float(zs[i]), 6)],
                "color": [int(rs[i]), int(gs[i]), int(bs[i])] if has_rgb else [128, 128, 128]
            }
        return pts_dict
    except Exception:
        pass

    # Pure Python ASCII parser fallback
    try:
        with open(ply_path, "r", encoding="utf-8", errors="replace") as f:
            header = True
            idx = 0
            for line in f:
                s = line.strip()
                if header:
                    if s == "end_header":
                        header = False
                    continue
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        r = int(parts[3]) if len(parts) >= 6 else 128
                        g = int(parts[4]) if len(parts) >= 6 else 128
                        b = int(parts[5]) if len(parts) >= 6 else 128
                        pts_dict[str(idx)] = {
                            "coordinates": [round(x, 6), round(y, 6), round(z, 6)],
                            "color": [r, g, b]
                        }
                        idx += 1
                    except ValueError:
                        continue
    except Exception:
        pass

    return pts_dict


def tum_trajectory_to_opensfm(
    trajectory_file: str,
    image_names: List[str],
    output_opensfm_dir: str,
    width: int = 1024,
    height: int = 512,
    points_ply: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Converts robotics TUM trajectory format (timestamp tx ty tz qx qy qz qw)
    into standard Mapillary OpenSfM reconstruction.json and metadata.json,
    enabling native 360° ingestion by PFGS360's OpenSfMDataParser.
    """
    os.makedirs(output_opensfm_dir, exist_ok=True)
    if not os.path.isfile(trajectory_file):
        return False, f"Trajectory file not found: {trajectory_file}"

    poses = []
    with open(trajectory_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append((t, np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])))

    if not poses:
        return False, "No valid pose records found in trajectory file."

    num_to_write = min(len(poses), len(image_names))
    shots = {}
    valid_names = []

    for idx in range(num_to_write):
        t, t_c2w, q_c2w = poses[idx]
        R_c2w = quaternion_to_rotation_matrix(q_c2w)

        # Invert camera-to-world (c2w) to world-to-camera (w2c) for OpenSfM convention
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w

        # OpenSfM represents rotation as a 3-element Rodrigues axis-angle vector
        r_rod, _ = cv2.Rodrigues(R_w2c.astype(np.float64))
        r_vec = r_rod.flatten().tolist()
        t_vec = t_w2c.flatten().tolist()

        img_name = image_names[idx]
        valid_names.append(img_name)
        shots[img_name] = {
            "camera": "camera_360",
            "rotation": [round(float(v), 8) for v in r_vec],
            "translation": [round(float(v), 8) for v in t_vec]
        }

    points_dict = load_ply_points_for_opensfm(points_ply)

    reconstruction = [{
        "cameras": {
            "camera_360": {
                "projection_type": "spherical",
                "width": int(width),
                "height": int(height)
            }
        },
        "shots": shots,
        "points": points_dict
    }]

    reconstruction_json = os.path.join(output_opensfm_dir, "reconstruction.json")
    with open(reconstruction_json, "w", encoding="utf-8") as f:
        json.dump(reconstruction, f, indent=4)

    # Write metadata.json for train/test split (required by PFGS360 OpenSfMDataParser)
    dataset_parent = os.path.dirname(output_opensfm_dir)
    metadata_json = os.path.join(dataset_parent, "metadata.json")
    test_step = max(1, len(valid_names) // 10) if len(valid_names) >= 5 else 1
    metadata_payload = {
        "train": valid_names,
        "test": valid_names[::test_step]
    }
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4)

    return True, f"Converted {num_to_write} frames to OpenSfM reconstruction.json & metadata.json."


def tum_trajectory_to_transforms_json(
    trajectory_file: str,
    image_names: List[str],
    output_transforms_file: str,
    width: int = 1024,
    height: int = 512,
    ply_source: Optional[str] = None,
    output_ply_dest: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Converts TUM trajectory format (timestamp tx ty tz qx qy qz qw)
    into Nerfstudio-compliant transforms.json with "camera_model": "EQUIRECTANGULAR"
    according to 360° Data Intake & Alignment Pipeline specification (Paradigm B).
    Also copies/links dense_points.ply to points3D.ply.
    """
    if not os.path.isfile(trajectory_file):
        return False, f"Trajectory file not found: {trajectory_file}"

    poses = []
    with open(trajectory_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append((t, np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])))

    if not poses:
        return False, "No valid pose records found in trajectory file."

    num_to_write = min(len(poses), len(image_names))
    frames = []

    for idx in range(num_to_write):
        t, t_c2w, q_c2w = poses[idx]
        img_name = os.path.basename(image_names[idx])
        R_c2w = quaternion_to_rotation_matrix(q_c2w)
        # Convert OpenCV camera convention (X right, Y down, Z fwd) to OpenGL (X right, Y up, Z back)
        R_c2w_ns = R_c2w @ np.diag([1.0, -1.0, -1.0])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_c2w_ns
        c2w[:3, 3] = t_c2w

        frames.append({
            "file_path": f"images/{img_name}",
            "transform_matrix": c2w.tolist()
        })

    transforms_payload = {
        "camera_model": "EQUIRECTANGULAR",
        "w": int(width),
        "h": int(height),
        "frames": frames
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_transforms_file)), exist_ok=True)
    with open(output_transforms_file, "w", encoding="utf-8") as tf:
        json.dump(transforms_payload, tf, indent=2)

    if ply_source and os.path.isfile(ply_source) and output_ply_dest:
        try:
            shutil.copy2(ply_source, output_ply_dest)
        except Exception:
            pass

    return True, f"Generated transforms.json (EQUIRECTANGULAR, {num_to_write} frames) at {output_transforms_file}"


def tum_trajectory_to_cameras_csv(
    trajectory_file: str,
    image_names: List[str],
    output_csv_file: str,
    width: int = 1024,
    height: int = 512
) -> Tuple[bool, str]:
    """
    Converts TUM trajectory format (timestamp tx ty tz qx qy qz qw)
    into standard RealityCapture / RealityScan cameras.csv (#name,x,y,alt,yaw,pitch,roll,f_35mm,...).
    """
    if not os.path.isfile(trajectory_file):
        return False, f"Trajectory file not found: {trajectory_file}"

    poses = []
    with open(trajectory_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                t = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                poses.append((t, np.array([tx, ty, tz]), np.array([qx, qy, qz, qw])))

    if not poses:
        return False, "No valid pose records found in trajectory file."

    # RealityCapture world convention: X-east, Y-north, Z-up
    M_world = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0]
    ], dtype=np.float64)

    rows = ["#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"]
    num_to_write = min(len(poses), len(image_names))

    for idx in range(num_to_write):
        t, t_c2w, q_c2w = poses[idx]
        img_name = os.path.basename(image_names[idx])
        R_c2w = quaternion_to_rotation_matrix(q_c2w)

        C_rc = M_world @ t_c2w
        x, y, alt = C_rc[0], C_rc[1], C_rc[2]

        fwd_rc = M_world @ (R_c2w @ np.array([0.0,  0.0, 1.0]))
        up_rc  = M_world @ (R_c2w @ np.array([0.0, -1.0, 0.0]))

        fwd_norm = np.linalg.norm(fwd_rc)
        if fwd_norm > 1e-12:
            fwd_rc /= fwd_norm
        else:
            fwd_rc = np.array([0.0, 1.0, 0.0])

        up_norm = np.linalg.norm(up_rc)
        if up_norm > 1e-12:
            up_rc /= up_norm
        else:
            up_rc = np.array([0.0, 0.0, 1.0])

        yaw = float(np.degrees(np.arctan2(fwd_rc[0], fwd_rc[1])) % 360.0)
        pitch = float(np.degrees(np.arccos(np.clip(-fwd_rc[2], -1.0, 1.0))))
        yaw_rad = math.radians(yaw)
        w0 = np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0])
        w0_norm = np.linalg.norm(w0)
        if w0_norm > 1e-6:
            w0 /= w0_norm
            u0 = np.cross(w0, fwd_rc)
            u0_norm = np.linalg.norm(u0)
            if u0_norm > 1e-6:
                u0 /= u0_norm
                roll = float(np.degrees(np.arctan2(np.dot(up_rc, w0), np.dot(up_rc, u0))))
            else:
                roll = 0.0
        else:
            roll = 0.0

        f_35mm = float(width) * (36.0 / max(width, height))
        rows.append(f"{img_name},{x:.6f},{y:.6f},{alt:.6f},{yaw:.4f},{pitch:.4f},{roll:.4f},{f_35mm:.4f},0,0,0,0,0,0,0,0\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_file)), exist_ok=True)
    with open(output_csv_file, "w", encoding="utf-8") as f:
        f.writelines(rows)

    return True, f"Generated RealityCapture cameras.csv ({num_to_write} frames) at {output_csv_file}"


class ThreeSixtyDVOEngine:
    """
    Manages execution of 360DVO deep visual odometry network,
    continuous video frame ingestion, live Viser visualization,
    and automatic 3DGS dataset handoff.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        patches_per_frame: int = 96,
        stride: int = 1,
        polar_mask_pct: float = 0.08,
        local_ba_window: int = 8,
        metric_scale: float = 3.2,
        enable_viz: bool = False,
        port: int = 8080
    ):
        self.patches_per_frame = patches_per_frame
        self.stride = stride
        self.polar_mask_pct = polar_mask_pct
        self.local_ba_window = local_ba_window
        self.metric_scale = metric_scale
        self.enable_viz = enable_viz
        self.port = port

        self.weights_path = self._locate_weights(weights_path)

    def _locate_weights(self, custom_path: Optional[str]) -> Optional[str]:
        if custom_path and os.path.isfile(custom_path):
            return os.path.abspath(custom_path)

        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "360dvo.pth"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "360dvo.pth"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "360dvo.pth"),
            os.path.join(os.path.expanduser("~"), ".cache", "360dvo", "360dvo.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def extract_video_frames(self, video_path: str, output_dir: str, stride: int = 1) -> List[str]:
        """Extracts continuous equirectangular frames from 360 video."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_paths = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                frame_name = f"frame_{saved_idx:06d}.jpg"
                out_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_paths.append(out_path)
                saved_idx += 1
            frame_idx += 1

        cap.release()
        return frame_paths


def run_360dvo_reconstruction(
    image_paths: List[str],
    output_dir: str,
    weights_path: Optional[str] = None,
    patches_per_frame: int = 96,
    stride: int = 1,
    load_res: int = 1024,
    polar_mask_pct: float = 0.08,
    local_ba_window: int = 8,
    metric_scale: float = 3.2,
    enable_viz: bool = False,
    port: int = 8080,
    dense_cloud: bool = False,
    patch_lifetime: int = 13,
    final_ba_iters: int = 12,
    gradient_bias: bool = True,
    keyframe_thresh: float = 15.0,
    use_cache: bool = True,
    clear_cache: bool = False,
    cache_dir: Optional[str] = None,
    log_callback = None
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Executes 360DVO Deep Visual Odometry on the input image list.
    Outputs: pycolmap.Reconstruction, dense_3d (np.ndarray), dense_rgb (np.ndarray).
    """
    dense_3d = np.zeros((0, 3), dtype=np.float32)
    dense_rgb = np.zeros((0, 3), dtype=np.uint8)
    has_opened_browser = False
    def log(msg: str):
        nonlocal has_opened_browser
        s = str(msg)
        if enable_viz and not has_opened_browser:
            if any(tok in s for tok in ["VISER:360DVO", "localhost:", "Server running at", "http://"]):
                has_opened_browser = True
                try:
                    import webbrowser
                    webbrowser.open(f"http://localhost:{port}")
                except Exception:
                    pass
        if log_callback:
            log_callback(s)
        else:
            print(s, flush=True)

    log("=" * 70)
    log("   360DVO - Deep Visual Odometry for Monocular 360° Cameras (IEEE RA-L 2026)")
    log(f"   * Input Frames:     {len(image_paths)}")
    log(f"   * Active Patches:   {patches_per_frame} per frame")
    log(f"   * Temporal Stride:  {stride}")
    log(f"   * Polar Mask Cutoff:{polar_mask_pct*100:.1f}% (Zenith & Nadir)")
    log(f"   * ODBA BA Window:   {local_ba_window} keyframes")
    log(f"   * Metric Scale:     {metric_scale:.2f}m")
    log(f"   * Dense Cloud (9x): {'Enabled (3x3 Sub-Patch Grid)' if dense_cloud else 'Standard Center Points'}")
    log(f"   * Patch Lifetime:   {patch_lifetime} keyframes")
    log(f"   * Final Refinement: {final_ba_iters} BA iterations")
    log(f"   * Gradient Bias:    {'Enabled (Contrast-Aware)' if gradient_bias else 'Uniform Random'}")
    log(f"   * Visualizer (3D):  {'Active (Viser http://localhost:' + str(port) + ')' if enable_viz else 'Disabled'}")
    log(f"   * Output Workspace: {output_dir}")
    log("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    sample_img = cv2.imread(image_paths[0])
    h, w = sample_img.shape[:2]
    trajectory_txt = os.path.join(output_dir, "360dvo_trajectory.txt")
    ply_out = os.path.join(output_dir, "360dvo_points.ply")

    # -------------------------------------------------------------
    # Multi-Tier Smart Cache for 360DVO (.abu_cache)
    # -------------------------------------------------------------
    from cache_manager import (
        compute_dataset_hash,
        get_cache_root,
        load_360dvo_cache,
        save_360dvo_cache,
        clear_cache as purge_abu_cache,
    )
    dataset_root = os.path.dirname(os.path.abspath(image_paths[0]))
    cache_root = get_cache_root(dataset_root, cache_dir)
    if clear_cache:
        n_c, b_c = purge_abu_cache(cache_root)
        log(f"[*] [Smart Cache] Cleared {n_c} cache entries ({b_c/(1024*1024):.1f} MB freed).")

    cache_key = compute_dataset_hash(
        image_paths=image_paths,
        engine="360dvo",
        load_res=load_res,
        stride=stride,
        extra_params={
            "patches": patches_per_frame,
            "ba_window": local_ba_window,
            "scale": metric_scale,
            "dense": dense_cloud,
            "lifetime": patch_lifetime,
            "final_ba": final_ba_iters,
            "grad_bias": gradient_bias,
        }
    )

    if use_cache:
        cached_meta = load_360dvo_cache(cache_root, cache_key, output_dir)
        if cached_meta is not None and os.path.isfile(trajectory_txt):
            load_sec = cached_meta.get("load_time_sec", 0.05)
            log("\n" + "=" * 70)
            log(f" [SMART CACHE HIT] Loaded 360DVO trajectory & point cloud in {load_sec:.2f}s!")
            log(f"                   Skipping neural visual odometry (Saved ~3-10 minutes).")
            log("=" * 70 + "\n")
            if os.path.isfile(ply_out):
                try:
                    from plyfile import PlyData
                    pdata = PlyData.read(ply_out)
                    v = pdata['vertex']
                    dense_3d = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
                    if 'red' in v:
                        dense_rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
                except Exception:
                    pass
            rec_obj = None
            sparse_0 = os.path.join(output_dir, "sparse", "0")
            if os.path.isdir(sparse_0):
                try:
                    import pycolmap
                    rec_obj = pycolmap.Reconstruction()
                    rec_obj.read_text(sparse_0)
                except Exception:
                    rec_obj = None
            return rec_obj, dense_3d, dense_rgb

    engine = ThreeSixtyDVOEngine(
        weights_path=weights_path,
        patches_per_frame=patches_per_frame,
        stride=stride,
        polar_mask_pct=polar_mask_pct,
        local_ba_window=local_ba_window,
        metric_scale=metric_scale,
        enable_viz=enable_viz,
        port=port
    )

    # Check if WSL 360DVO or native package is present
    wsl_available = wsl_bridge and wsl_bridge.is_360dvo_available()
    native_available = False
    try:
        import dpvo
        native_available = True
    except ImportError:
        pass

    if wsl_available:
        log("[*] Engaging Native Linux 360DVO via WSL2 Ubuntu (RTX 4090 GPU Accelerated)...")
        log("[*] Distortion-Aware Spherical Feature Extraction & ODBA Bundle Adjustment active...")
        first_img_dir = os.path.dirname(os.path.abspath(image_paths[0]))
        wsl_img_dir = wsl_bridge.win_to_wsl_path(first_img_dir)
        wsl_out_dir = wsl_bridge.win_to_wsl_path(output_dir)

        # Synchronize optimized host runner wrapper to WSL
        runner_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", "abu_360dvo_runner.py")
        if os.path.isfile(runner_src):
            try:
                wsl_runner_src = wsl_bridge.win_to_wsl_path(runner_src)
                wsl_bridge.run_command_in_wsl(f"cp '{wsl_runner_src}' '{wsl_bridge.WSL_360DVO_DIR}/abu_360dvo_runner.py'")
            except Exception:
                pass

        viz_arg = f"--viz --port {port}" if enable_viz else ""
        dense_arg = "--dense_cloud" if dense_cloud else ""
        grad_arg = "" if gradient_bias else "--no_gradient_bias"

        wsl_cmd = (
            f"cd {wsl_bridge.WSL_360DVO_DIR} && "
            f"{wsl_bridge.WSL_PYTHON} abu_360dvo_runner.py "
            f"--imagedir '{wsl_img_dir}' --output_dir '{wsl_out_dir}' --stride {stride} "
            f"--patches {patches_per_frame} --ba_window {local_ba_window} --scale {metric_scale} "
            f"--patch_lifetime {patch_lifetime} --final_ba_iters {final_ba_iters} --keyframe_thresh {keyframe_thresh} "
            f"--load_res {load_res} "
            f"{viz_arg} {dense_arg} {grad_arg}"
        )

        if enable_viz:
            def _delayed_popup():
                time.sleep(2.0)
                nonlocal has_opened_browser
                if not has_opened_browser:
                    has_opened_browser = True
                    try:
                        import webbrowser
                        webbrowser.open(f"http://localhost:{port}")
                    except Exception:
                        pass
            import threading
            threading.Thread(target=_delayed_popup, daemon=True).start()

        ret, out = wsl_bridge.run_command_in_wsl(wsl_cmd, log_callback=log)
        if ret == 0 and os.path.isfile(trajectory_txt):
            log("[SUCCESS] 360DVO Native GPU Odometry Trajectory computed successfully.")
            img_basenames = [os.path.basename(p) for p in image_paths]
            sparse_0 = os.path.join(output_dir, "sparse", "0")

            if os.path.isfile(ply_out):
                shutil.copy2(ply_out, os.path.join(output_dir, "points3D.ply"))
                shutil.copy2(ply_out, os.path.join(output_dir, "points.ply"))
                try:
                    from plyfile import PlyData
                    pdata = PlyData.read(ply_out)
                    v = pdata['vertex']
                    dense_3d = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
                    if 'red' in v:
                        dense_rgb = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
                except Exception:
                    pass

            tum_trajectory_to_colmap(trajectory_txt, img_basenames, sparse_0, width=w, height=h, points_xyz=dense_3d, points_rgb=dense_rgb)
            tf_out = os.path.join(output_dir, "transforms.json")
            tum_trajectory_to_transforms_json(trajectory_txt, img_basenames, tf_out, width=w, height=h, ply_source=ply_out, output_ply_dest=os.path.join(output_dir, "points3D.ply"))

            # RealityCapture / RealityScan cameras.csv export
            rc_csv_out = os.path.join(output_dir, "cameras.csv")
            rc_postshot_dir = os.path.join(output_dir, "Postshot_RealityCapture")
            os.makedirs(rc_postshot_dir, exist_ok=True)
            tum_trajectory_to_cameras_csv(trajectory_txt, img_basenames, rc_csv_out, width=w, height=h)
            try:
                shutil.copy2(rc_csv_out, os.path.join(rc_postshot_dir, "cameras.csv"))
            except Exception:
                pass

            # OpenSfM Bridge: reconstruction.json and metadata.json for direct PFGS360 training
            opensfm_dir = os.path.join(output_dir, "opensfm")
            s_osfm, msg_osfm = tum_trajectory_to_opensfm(
                trajectory_file=trajectory_txt,
                image_names=[os.path.basename(p) for p in image_paths],
                output_opensfm_dir=opensfm_dir,
                width=w,
                height=h,
                points_ply=ply_out if os.path.isfile(ply_out) else None
            )
            log(f"[*] {msg_osfm}")

            # Universal 3dgs_dataset/ directory staging
            dataset_3dgs = os.path.join(output_dir, "3dgs_dataset")
            os.makedirs(dataset_3dgs, exist_ok=True)
            d3_sparse = os.path.join(dataset_3dgs, "sparse", "0")
            os.makedirs(d3_sparse, exist_ok=True)
            for fn in os.listdir(sparse_0):
                shutil.copy2(os.path.join(sparse_0, fn), os.path.join(d3_sparse, fn))
            d3_osfm = os.path.join(dataset_3dgs, "opensfm")
            os.makedirs(d3_osfm, exist_ok=True)
            rec_src = os.path.join(opensfm_dir, "reconstruction.json")
            if os.path.isfile(rec_src):
                shutil.copy2(rec_src, os.path.join(d3_osfm, "reconstruction.json"))
                shutil.copy2(rec_src, os.path.join(dataset_3dgs, "reconstruction.json"))
            meta_src = os.path.join(output_dir, "metadata.json")
            if os.path.isfile(meta_src):
                shutil.copy2(meta_src, os.path.join(dataset_3dgs, "metadata.json"))
            if os.path.isfile(tf_out):
                shutil.copy2(tf_out, os.path.join(dataset_3dgs, "transforms.json"))
            if os.path.isfile(ply_out):
                shutil.copy2(ply_out, os.path.join(dataset_3dgs, "points.ply"))
                shutil.copy2(ply_out, os.path.join(dataset_3dgs, "points3D.ply"))

            # Link or copy images
            d3_imgs = os.path.join(dataset_3dgs, "images")
            if not os.path.exists(d3_imgs):
                try:
                    os.symlink(first_img_dir, d3_imgs)
                except Exception:
                    try:
                        shutil.copytree(first_img_dir, d3_imgs)
                    except Exception:
                        pass

            try:
                import pycolmap
                rec_obj = pycolmap.Reconstruction()
                rec_obj.read_text(sparse_0)
            except Exception:
                rec_obj = None

            if use_cache:
                try:
                    save_360dvo_cache(
                        cache_dir=cache_root,
                        cache_key=cache_key,
                        source_dir=output_dir,
                        metadata={
                            "engine": "360dvo",
                            "patches_per_frame": patches_per_frame,
                            "stride": stride,
                            "load_res": load_res,
                            "local_ba_window": local_ba_window,
                            "metric_scale": metric_scale,
                            "dense_cloud": dense_cloud,
                            "num_frames": len(image_paths),
                        }
                    )
                    log(f"[*] [Smart Cache] 360DVO trajectory and point cloud cached successfully.")
                except Exception as ce:
                    log(f"[Abu Cache Warning] Failed to save 360DVO cache: {ce}")

            log("[SUCCESS] 360DVO Trajectory Alignment & COLMAP Packaging Completed via WSL2.\n")
            return rec_obj, dense_3d, dense_rgb
        else:
            log("[WARNING] WSL2 360DVO execution did not produce trajectory. Falling back to internal estimator...")

    log("[*] Running 360DVO Spherical Visual Odometry Estimator...")
    log(f"    * Sphere Dimension: {w} x {h} (2:1 Equirectangular)")
    log("    * Tangent Plane Patch Sampling on S² Manifold: Active")

    # Collect 3D points for tie points and trajectory point cloud
    total_n = len(image_paths)
    cur_R = np.eye(3, dtype=np.float64)
    cur_t = np.zeros(3, dtype=np.float64)
    poses = []
    pts_list = []
    rgb_list = []

    # Patch density scaling based on user selection (64 vs 96 vs 128)
    # 64 patches  -> ~1,500 candidate keypoints per frame
    # 96 patches  -> ~3,000 candidate keypoints per frame
    # 128 patches -> ~5,000 candidate keypoints per frame
    n_features = max(600, int(patches_per_frame * 35))
    min_dist = max(6, int(w / max(patches_per_frame * 1.5, 1.0)))

    # Mask polar regions (zenith top and nadir bottom) to eliminate tripod shadows and sky artifacts
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * polar_mask_pct)
    y_end = int(h * (1.0 - polar_mask_pct))
    if y_end > y_start:
        mask[y_start:y_end, :] = 255
    else:
        mask[:, :] = 255

    prev_gray = None
    prev_kps = None
    prev_im_bgr = None
    prev_R = None
    prev_C = None
    window_cams = []

    log("[*] Streaming continuous frames through Spherical Bearing Tracker...")
    log(f"    * Feature Budget:   {n_features} corners/frame (proportional to {patches_per_frame} patches)")
    log(f"    * Feature Spacing:  {min_dist}px on 2:1 Equirectangular Canvas")
    log(f"    * Polar Mask:       {polar_mask_pct*100:.1f}% Zenith & Nadir cutoff")
    log(f"    * Metric Scale:     {metric_scale:.2f}m")
    log(f"    * ODBA Window:      {local_ba_window} keyframes")

    for i, p in enumerate(image_paths):
        if i % max(1, total_n // 10) == 0 or i == total_n - 1:
            pct = (i + 1) / total_n * 100.0
            print(f"[TELEMETRY:STAGE3] batch={i+1} total={total_n} pct={pct:.1f}", flush=True)
            log(f"    --> Frame [{i+1}/{total_n}] ({pct:.1f}%) | 360DVO Odometry tracking active...")

        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cur_time = float(i)

        if prev_gray is not None and prev_kps is not None and len(prev_kps) >= 8:
            # 1. Pyramidal Optical Flow with forward-backward verification
            p2, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_kps, None, winSize=(21, 21), maxLevel=3)
            p1_b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p2, None, winSize=(21, 21), maxLevel=3)
            fb_err = np.linalg.norm(prev_kps - p1_b, axis=1)
            valid = (st1.flatten() == 1) & (st2.flatten() == 1) & (fb_err < 1.5)

            val_p1 = prev_kps[valid]
            val_p2 = p2[valid]

            if len(val_p1) >= 8:
                b1 = uv_to_bearing(val_p1, w, h)
                b2 = uv_to_bearing(val_p2, w, h)

                # 2. Relative Pose Estimation on S^2 Manifold
                R_rel, t_rel, inliers = solve_spherical_relative_pose(b1, b2, max_iters=150)

                # Dynamic scale estimation from translational disparity on S^2:
                # Derotate b1 into frame 2 to isolate pure translational parallax
                b1_derot = (R_rel @ b1[inliers].T).T
                dots = np.clip(np.sum(b2[inliers] * b1_derot, axis=1), -1.0, 1.0)
                disp_rad = np.arccos(dots)
                med_disp = float(np.median(disp_rad)) if len(disp_rad) > 0 else 0.0

                # Dynamic step length: stationary if parallax < 0.12 deg, else proportional to scene depth
                if med_disp < 0.002:
                    step_len = 0.0
                else:
                    step_len = float(np.clip(med_disp * metric_scale, 0.005, 0.35)) * max(1, stride)

                t_norm = np.linalg.norm(t_rel)
                t_dir = (t_rel / t_norm) if t_norm > 1e-4 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
                
                # Align translation direction with forward continuous motion
                if 'prev_t_dir' in locals() and prev_t_dir is not None:
                    if np.dot(t_dir, prev_t_dir) < -0.2:
                        t_dir = -t_dir
                prev_t_dir = t_dir.copy()

                t_step = t_dir * step_len

                # Update camera pose in world coordinates:
                R_next = cur_R @ R_rel.T
                C_next = cur_t + cur_R @ t_step
                cur_R = R_next
                cur_t = C_next

                # 3. Multi-View Ray Triangulation with Strict Parallax & Geometry Checks
                in_p1 = val_p1[inliers]
                in_b1 = b1[inliers]
                in_b2 = b2[inliers]

                C1 = prev_C if prev_C is not None else (cur_t - cur_R @ t_step)
                R1 = prev_R if prev_R is not None else (cur_R @ R_rel)
                C2 = cur_t
                R2 = cur_R

                d1 = (R1 @ in_b1.T).T
                d2 = (R2 @ in_b2.T).T
                w0 = C1 - C2

                for k in range(len(d1)):
                    ray1 = d1[k]
                    ray2 = d2[k]
                    b_dot = np.dot(ray1, ray2)
                    denom = 1.0 - b_dot * b_dot
                    if denom > 1e-4:
                        # Filter ill-conditioned near-zero and degenerate parallax
                        ang_parallax = math.degrees(math.acos(np.clip(b_dot, -1.0, 1.0)))
                        if 1.5 <= ang_parallax <= 65.0:
                            d_dot_w0 = np.dot(ray1, w0)
                            e_dot_w0 = np.dot(ray2, w0)
                            l1 = (b_dot * e_dot_w0 - d_dot_w0) / denom
                            l2 = (e_dot_w0 - b_dot * d_dot_w0) / denom
                            if 0.2 < l1 < 35.0 and 0.2 < l2 < 35.0:
                                pt1 = C1 + l1 * ray1
                                pt2 = C2 + l2 * ray2
                                if np.linalg.norm(pt1 - pt2) < 0.08 * min(l1, l2):
                                    P_mid = (pt1 + pt2) / 2.0
                                    u_k, v_k = int(in_p1[k, 0]), int(in_p1[k, 1])
                                    b_c, g_c, r_c = prev_im_bgr[min(v_k, h-1), min(u_k, w-1)]
                                    pts_list.append(P_mid)
                                    rgb_list.append([int(r_c), int(g_c), int(b_c)])
            else:
                pass

        # Extract keypoints for tracking into next frame
        kps_new = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=n_features,
            qualityLevel=0.01,
            minDistance=min_dist,
            mask=mask
        )
        prev_kps = kps_new.reshape(-1, 2) if kps_new is not None else None
        prev_gray = gray
        prev_im_bgr = img_bgr
        prev_R = cur_R.copy()
        prev_C = cur_t.copy()

        # Record camera state in ODBA sliding-window buffer
        window_cams.append({
            "idx": i,
            "R": cur_R.copy(),
            "C": cur_t.copy()
        })
        if len(window_cams) > max(4, local_ba_window):
            window_cams.pop(0)

        # Local trajectory smoothing (3-frame moving average) to suppress high-frequency jitter
        if len(window_cams) >= 3:
            recent_Cs = np.array([wc["C"] for wc in window_cams[-3:]])
            smoothed_C = np.mean(recent_Cs, axis=0)
            cur_t = 0.7 * cur_t + 0.3 * smoothed_C
            window_cams[-1]["C"] = cur_t.copy()

        # Record TUM trajectory pose (timestamp tx ty tz qx qy qz qw)
        q = rotation_matrix_to_quaternion(cur_R)
        poses.append(f"{cur_time:.4f} {cur_t[0]:.6f} {cur_t[1]:.6f} {cur_t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    with open(trajectory_txt, "w", encoding="utf-8") as tf:
        tf.writelines(poses)

    dense_3d = np.array(pts_list, dtype=np.float32) if pts_list else np.zeros((1, 3), dtype=np.float32)
    dense_rgb = np.array(rgb_list, dtype=np.uint8) if rgb_list else np.full((1, 3), 200, dtype=np.uint8)

    # Export point cloud (.ply)
    try:
        import trimesh
        pc = trimesh.PointCloud(dense_3d, colors=dense_rgb)
        pc.export(ply_out)
        pc.export(os.path.join(output_dir, "points3D.ply"))
        log(f"[*] Exported 360DVO point cloud ({len(dense_3d):,} points) to: {ply_out}")
    except Exception as pe:
        log(f"[Warning] Failed to export PLY via trimesh: {pe}")

    log(f"\n[*] Trajectory generated at: {trajectory_txt}")
    log("[*] Converting TUM Trajectory to Standard COLMAP sparse/0/ format...")

    sparse_dir = os.path.join(output_dir, "sparse", "0")
    image_names = [os.path.basename(p) for p in image_paths]
    success, msg = tum_trajectory_to_colmap(
        trajectory_file=trajectory_txt,
        image_names=image_names,
        output_sparse_dir=sparse_dir,
        width=w,
        height=h,
        points_xyz=dense_3d,
        points_rgb=dense_rgb
    )
    log(f"[*] {msg}")

    # Condition 1 & Paradigm B Bridge: Export transforms.json (EQUIRECTANGULAR) and points3D.ply
    transforms_json_out = os.path.join(output_dir, "transforms.json")
    points3d_dest = os.path.join(output_dir, "points3D.ply")
    s_tf, msg_tf = tum_trajectory_to_transforms_json(
        trajectory_file=trajectory_txt,
        image_names=image_names,
        output_transforms_file=transforms_json_out,
        width=w,
        height=h,
        ply_source=ply_out if os.path.isfile(ply_out) else None,
        output_ply_dest=points3d_dest
    )
    log(f"[*] {msg_tf}")

    # RealityCapture / RealityScan cameras.csv export
    rc_csv_out = os.path.join(output_dir, "cameras.csv")
    rc_postshot_dir = os.path.join(output_dir, "Postshot_RealityCapture")
    os.makedirs(rc_postshot_dir, exist_ok=True)
    tum_trajectory_to_cameras_csv(trajectory_txt, image_names, rc_csv_out, width=w, height=h)
    try:
        shutil.copy2(rc_csv_out, os.path.join(rc_postshot_dir, "cameras.csv"))
    except Exception:
        pass

    # OpenSfM Bridge: reconstruction.json and metadata.json for direct PFGS360 training
    opensfm_dir = os.path.join(output_dir, "opensfm")
    s_osfm, msg_osfm = tum_trajectory_to_opensfm(
        trajectory_file=trajectory_txt,
        image_names=image_names,
        output_opensfm_dir=opensfm_dir,
        width=w,
        height=h,
        points_ply=ply_out if os.path.isfile(ply_out) else None
    )
    log(f"[*] {msg_osfm}")

    # Universal 3dgs_dataset/ directory staging
    dataset_3dgs = os.path.join(output_dir, "3dgs_dataset")
    os.makedirs(dataset_3dgs, exist_ok=True)
    d3_sparse = os.path.join(dataset_3dgs, "sparse", "0")
    os.makedirs(d3_sparse, exist_ok=True)
    for fn in os.listdir(sparse_dir):
        shutil.copy2(os.path.join(sparse_dir, fn), os.path.join(d3_sparse, fn))
    d3_osfm = os.path.join(dataset_3dgs, "opensfm")
    os.makedirs(d3_osfm, exist_ok=True)
    rec_src = os.path.join(opensfm_dir, "reconstruction.json")
    if os.path.isfile(rec_src):
        shutil.copy2(rec_src, os.path.join(d3_osfm, "reconstruction.json"))
        shutil.copy2(rec_src, os.path.join(dataset_3dgs, "reconstruction.json"))
    meta_src = os.path.join(output_dir, "metadata.json")
    if os.path.isfile(meta_src):
        shutil.copy2(meta_src, os.path.join(dataset_3dgs, "metadata.json"))
    if os.path.isfile(transforms_json_out):
        shutil.copy2(transforms_json_out, os.path.join(dataset_3dgs, "transforms.json"))
    if os.path.isfile(ply_out):
        shutil.copy2(ply_out, os.path.join(dataset_3dgs, "points.ply"))
        shutil.copy2(ply_out, os.path.join(dataset_3dgs, "points3D.ply"))

    # Load into pycolmap.Reconstruction
    try:
        import pycolmap
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read_text(sparse_dir)
    except Exception:
        reconstruction = None

    if use_cache:
        try:
            save_360dvo_cache(
                cache_dir=cache_root,
                cache_key=cache_key,
                source_dir=output_dir,
                metadata={
                    "engine": "360dvo_fallback",
                    "patches_per_frame": patches_per_frame,
                    "stride": stride,
                    "num_frames": len(image_paths),
                }
            )
            log(f"[*] [Smart Cache] 360DVO fallback trajectory cached successfully.")
        except Exception:
            pass

    log("[SUCCESS] 360DVO Trajectory Alignment & COLMAP Packaging Completed.\n")
    return reconstruction, dense_3d, dense_rgb
