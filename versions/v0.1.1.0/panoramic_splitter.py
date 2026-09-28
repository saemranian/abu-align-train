"""
Abu Align & Train - Generalized 360° Multi-Camera Rig Projector & Panoramic Splitter
==================================================================================
Converts 360° Equirectangular Panoramas (ERP) and 360° Video Files into virtual
multi-camera pinhole rigs with configurable field-of-view (FOV), angular distribution,
overlap, lens distortion, and direct transforms.json export.

Rigs Supported:
1. Horizontal Ring Rig (N virtual cameras around equator, e.g. 8 cams @ 45° step, 90° FOV => 50% overlap)
2. Standard 6-Face Cubemap (Front, Back, Left, Right, Up, Down, 90° FOV)
3. Multi-Tier Spherical Dome Rig (Equator + configurable upper/lower tiers + zenith/nadir)
4. Fibonacci / Golden Spiral Sphere Rig (Uniform 4π steradian spherical coverage)
5. Custom Camera Angle Rig (arbitrary yaw/pitch/roll/fov camera arrays)

Features:
- Seamless 360° boundary wrapping and sub-pixel bicubic/bilinear interpolation
- Direct 360° video intake (.mp4, .mov, .insv, .mkv) with Target FPS & Keyframe Stride sampling
- Aspect Ratio support (1:1, 4:3, 16:9, Custom WxH) with automatic vertical FOV calculation
- Native angular pixel density solver (Auto-Res) matching equirectangular PPD
- Radial lens distortion modeling (k1, k2 barrel/pincushion compensation)
- Automatic Pinhole & OpenCV Intrinsics & Extrinsics calculation (fl_x, fl_y, cx, cy, 4x4 c2w)
- Exports Nerfstudio / 3DGS compliant transforms.json and rig_manifest.json
- Multi-threaded batch processing with live progress callbacks
"""

import os
import sys
import math
import json
import time
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ======================================================================
# Mathematical Relations & Optics Formulas
# ======================================================================

def calculate_angular_step(num_cameras: int) -> float:
    """Calculates angular separation between adjacent cameras in a 360° ring."""
    return 360.0 / max(num_cameras, 1)


def calculate_overlap(num_cameras: int, fov_deg: float) -> float:
    """
    Calculates horizontal overlap percentage between adjacent cameras in an equatorial ring.
    Formula: Overlap = max(0, (FOV - Delta_theta) / FOV) * 100%
    """
    step = calculate_angular_step(num_cameras)
    if fov_deg <= step:
        return 0.0
    return min(100.0, ((fov_deg - step) / fov_deg) * 100.0)


def calculate_fov_from_overlap(num_cameras: int, overlap_pct: float) -> float:
    """
    Calculates required FOV (in degrees) to achieve target overlap percentage for N cameras.
    Formula: FOV = (360 / N) / (1 - Overlap / 100)
    """
    step = calculate_angular_step(num_cameras)
    overlap_frac = max(0.0, min(0.95, overlap_pct / 100.0))
    return step / (1.0 - overlap_frac)


def calculate_cameras_from_fov_and_overlap(fov_deg: float, overlap_pct: float) -> int:
    """
    Calculates minimum number of cameras required to achieve target overlap given FOV.
    Formula: N = ceil(360 / (FOV * (1 - Overlap / 100)))
    """
    overlap_frac = max(0.0, min(0.95, overlap_pct / 100.0))
    effective_step = fov_deg * (1.0 - overlap_frac)
    if effective_step <= 0.1:
        return 4
    return max(3, math.ceil(360.0 / effective_step))


def calculate_optimal_resolution(pano_width: int, fov_deg: float) -> int:
    """
    Calculates optimal pinhole crop resolution matching native equirectangular angular pixel density.
    At center of pinhole view: f = (W_pin / 2) / tan(FOV / 2).
    Equirectangular angular density: PPD = pano_width / (2 * pi).
    Matching 1 px/rad at center yields: W_opt = (pano_width / pi) * tan(FOV / 2).
    """
    fov_rad = math.radians(fov_deg)
    opt_w = (pano_width / math.pi) * math.tan(fov_rad / 2.0)
    # Round to clean multiple of 16 for video encoder and GPU texture friendliness
    return max(256, int(round(opt_w / 16.0) * 16))


def calculate_vertical_fov(fov_h_deg: float, width: int, height: int) -> float:
    """
    Computes vertical FOV from horizontal FOV and aspect ratio (width, height).
    tan(FOV_v / 2) = (height / width) * tan(FOV_h / 2)
    """
    if width <= 0 or height <= 0:
        return fov_h_deg
    half_h_rad = math.radians(fov_h_deg) / 2.0
    half_v_rad = math.atan((float(height) / float(width)) * math.tan(half_h_rad))
    return math.degrees(half_v_rad) * 2.0


# ======================================================================
# Canonical Rig Configurations & Generators
# ======================================================================
# Format per camera: (name, yaw_deg, pitch_deg, roll_deg, fov_deg)

def get_cubemap_rig(fov_deg: float = 90.0) -> List[Tuple[str, float, float, float, float]]:
    """Standard 6-face orthogonal cubemap rig (Front, Right, Back, Left, Zenith, Nadir)."""
    return [
        ("front",  0.0,    0.0,   0.0, fov_deg),  # +Z
        ("right",  90.0,   0.0,   0.0, fov_deg),  # +X
        ("back",   180.0,  0.0,   0.0, fov_deg),  # -Z
        ("left",   -90.0,  0.0,   0.0, fov_deg),  # -X
        ("up",     0.0,    90.0,  0.0, fov_deg),  # +Y (Zenith)
        ("down",   0.0,   -90.0,  0.0, fov_deg),  # -Y (Nadir)
    ]


def get_horizontal_ring_rig(
    num_cameras: int = 8,
    fov_deg: float = 90.0,
    pitch_deg: float = 0.0
) -> List[Tuple[str, float, float, float, float]]:
    """
    Horizontal ring rig of N virtual cameras distributed evenly around the equator.
    Adjacent cameras overlap by: Overlap = max(0, (FOV - (360/N)) / FOV) * 100%.
    """
    num_cameras = max(num_cameras, 3)
    step = 360.0 / num_cameras
    rig = []
    for i in range(num_cameras):
        yaw = i * step
        # Normalize yaw to [-180, 180)
        yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
        name = f"ring_{i:02d}_yaw{int(round(yaw)):03d}"
        rig.append((name, yaw_norm, pitch_deg, 0.0, fov_deg))
    return rig


def get_spherical_dome_rig(
    equator_cams: int = 8,
    pitch_tier: float = 35.0,
    tier_cams: int = 2,
    include_zenith: bool = True,
    include_nadir: bool = True,
    fov_deg: float = 90.0
) -> List[Tuple[str, float, float, float, float]]:
    """
    Multi-tier spherical dome rig covering the 4π steradian sphere:
    - Equatorial ring (equator_cams @ pitch 0°)
    - Upper tier (+pitch_tier°, tier_cams distributed with yaw offset)
    - Lower tier (-pitch_tier°, tier_cams distributed with yaw offset)
    - Optional Zenith (+90°) and Nadir (-90°) cameras
    """
    rig = []
    # 1. Equator
    step_eq = 360.0 / max(equator_cams, 3)
    for i in range(equator_cams):
        yaw = i * step_eq
        yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
        rig.append((f"equator_{i:02d}_yaw{int(round(yaw)):03d}", yaw_norm, 0.0, 0.0, fov_deg))

    # 2. Upper tier
    if tier_cams > 0 and pitch_tier > 0:
        step_up = 360.0 / tier_cams
        offset_up = step_up / 2.0  # Stagger relative to equator
        for i in range(tier_cams):
            yaw = (i * step_up) + offset_up
            yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
            rig.append((f"upper_{i:02d}_yaw{int(round(yaw)):03d}_p{int(round(pitch_tier))}", yaw_norm, pitch_tier, 0.0, fov_deg))

    # 3. Lower tier
    if tier_cams > 0 and pitch_tier > 0:
        step_low = 360.0 / tier_cams
        offset_low = 0.0
        for i in range(tier_cams):
            yaw = (i * step_low) + offset_low
            yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
            rig.append((f"lower_{i:02d}_yaw{int(round(yaw)):03d}_m{int(round(pitch_tier))}", yaw_norm, -pitch_tier, 0.0, fov_deg))

    # 4. Zenith & Nadir
    if include_zenith:
        rig.append(("zenith_up", 0.0, 90.0, 0.0, fov_deg))
    if include_nadir:
        rig.append(("nadir_down", 0.0, -90.0, 0.0, fov_deg))

    return rig


def get_fibonacci_sphere_rig(
    num_cameras: int = 12,
    fov_deg: float = 90.0
) -> List[Tuple[str, float, float, float, float]]:
    """
    Generates N virtual cameras uniformly distributed across the 4π steradian sphere
    using the spherical Fibonacci (Golden Spiral) distribution.
    Guarantees optimal packing and uniform angular spacing across all directions.
    """
    num_cameras = max(num_cameras, 4)
    rig = []
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio (1.6180339887...)
    golden_angle = 2.0 * math.pi * (1.0 - 1.0 / phi)  # ~2.399963 rad (~137.5077°)

    for i in range(num_cameras):
        # Height y from +1 to -1
        y = 1.0 - (float(i) + 0.5) * (2.0 / num_cameras)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = i * golden_angle

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        pitch_deg = math.degrees(math.asin(y))
        yaw_deg = math.degrees(math.atan2(x, z))
        yaw_norm = ((yaw_deg + 180.0) % 360.0) - 180.0

        p_sign = "+" if pitch_deg >= 0 else "-"
        y_sign = "+" if yaw_norm >= 0 else "-"
        name = f"fib_{i:02d}_y{y_sign}{abs(int(round(yaw_norm))):03d}_p{p_sign}{abs(int(round(pitch_deg))):02d}"
        rig.append((name, yaw_norm, pitch_deg, 0.0, fov_deg))

    return rig


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """
    Computes 3x3 camera rotation matrix from yaw, pitch, roll (in degrees).
    Coordinate convention: X-right, Y-up, Z-forward.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    # Ry (yaw around Y axis)
    Ry = np.array([
        [ math.cos(yaw), 0.0, math.sin(yaw)],
        [ 0.0,           1.0, 0.0          ],
        [-math.sin(yaw), 0.0, math.cos(yaw)]
    ], dtype=np.float64)

    # Rx (pitch around X axis)
    Rx = np.array([
        [1.0, 0.0,            0.0          ],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch),  math.cos(pitch)]
    ], dtype=np.float64)

    # Rz (roll around Z axis)
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0.0],
        [math.sin(roll),  math.cos(roll), 0.0],
        [0.0,            0.0,             1.0]
    ], dtype=np.float64)

    return Ry @ Rx @ Rz


# ======================================================================
# RealityScan / RealityCapture Standard Photogrammetry & Compounding
# ======================================================================

RC_CSV_HEADER = "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"


def basis(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal right/up/forward basis in local rig frame, with roll applied simultaneously."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    forward = np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)], dtype=np.float64)
    right = np.array([math.cos(y), 0.0, -math.sin(y)], dtype=np.float64)
    up = np.array([-math.sin(p) * math.sin(y), math.cos(p), -math.sin(p) * math.cos(y)], dtype=np.float64)
    rolled_right = right * math.cos(r) + up * math.sin(r)
    rolled_up = up * math.cos(r) - right * math.sin(r)
    return rolled_right, rolled_up, forward


def get_rc_axes(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the 3 orthonormal body axes in RealityCapture world frame (X-east, Y-north, Z-up)
    for a panorama or camera with given (yaw, pitch, roll).
    Yaw: 0=North (+Y), 90=East (+X)
    Pitch: 0=Nadir (-Z), 90=Horizontal, 180=Zenith (+Z)
    """
    p_rad = math.radians(pitch_deg)
    y_rad = math.radians(yaw_deg)
    r_rad = math.radians(roll_deg)
    fwd = np.array([math.sin(p_rad) * math.sin(y_rad), math.sin(p_rad) * math.cos(y_rad), -math.cos(p_rad)], dtype=np.float64)
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm > 1e-12:
        fwd /= fwd_norm
    else:
        fwd = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    w0 = np.array([math.cos(y_rad), -math.sin(y_rad), 0.0], dtype=np.float64)
    w0_norm = np.linalg.norm(w0)
    if w0_norm > 1e-6:
        w0 /= w0_norm
        u0 = np.cross(w0, fwd)
        u0_norm = np.linalg.norm(u0)
        if u0_norm > 1e-6:
            u0 /= u0_norm
        else:
            u0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        u0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        w0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    up = math.cos(r_rad) * u0 + math.sin(r_rad) * w0
    up_norm = np.linalg.norm(up)
    if up_norm > 1e-12:
        up /= up_norm
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(fwd, up)
    if np.dot(right, w0) < 0:
        right = -right
    right_norm = np.linalg.norm(right)
    if right_norm > 1e-12:
        right /= right_norm
    else:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    return right, up, fwd


def rc_axes_to_euler(fwd: np.ndarray, up: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts canonical RealityCapture / RealityScan Euler angles (yaw, pitch, roll)
    from optical axes in RC world space.
    """
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm > 1e-12:
        fwd = fwd / fwd_norm
    else:
        fwd = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    up_norm = np.linalg.norm(up)
    if up_norm > 1e-12:
        up = up / up_norm
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    yaw = float(np.degrees(np.arctan2(fwd[0], fwd[1])) % 360.0)
    pitch = float(np.degrees(np.arccos(np.clip(-fwd[2], -1.0, 1.0))))
    yaw_rad = math.radians(yaw)
    w0 = np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0], dtype=np.float64)
    w0_norm = np.linalg.norm(w0)
    if w0_norm > 1e-6:
        w0 /= w0_norm
        u0 = np.cross(w0, fwd)
        u0_norm = np.linalg.norm(u0)
        if u0_norm > 1e-6:
            u0 /= u0_norm
            roll = float(np.degrees(np.arctan2(np.dot(up, w0), np.dot(up, u0))))
        else:
            roll = 0.0
    else:
        roll = 0.0
    return round(yaw, 4), round(pitch, 4), round(roll, 4)


def load_dvo_trajectory(trajectory_path: str) -> Dict[str, Any]:
    """
    Loads 360DVO camera positions and orientations from cameras.csv or 360dvo_trajectory.txt.
    Supports RealityCapture standard CSV format and robotics TUM trajectory format.
    Returns:
        {
            "count": int,
            "poses": List[Dict],
            "by_name": Dict[str, Dict],
            "by_stem": Dict[str, Dict],
            "by_idx": Dict[int, Dict]
        }
    """
    result = {
        "count": 0,
        "poses": [],
        "by_name": {},
        "by_stem": {},
        "by_idx": {},
    }
    if not trajectory_path or not os.path.isfile(trajectory_path):
        return result

    try:
        with open(trajectory_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            return result

        first_line = lines[0].lower()
        is_csv = "," in lines[0] or first_line.startswith("#name") or first_line.startswith("name") or first_line.startswith("#image")

        if is_csv:
            start_row = 1 if (first_line.startswith("#") or "name" in first_line or "yaw" in first_line) else 0
            idx = 0
            for line in lines[start_row:]:
                if line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    try:
                        name = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        alt = float(parts[3])
                        yaw = float(parts[4])
                        pitch = float(parts[5])
                        roll = float(parts[6])
                        f_35mm = float(parts[7]) if len(parts) > 7 else 35.0
                        rec = {
                            "name": name,
                            "x": x,
                            "y": y,
                            "alt": alt,
                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": roll,
                            "f_35mm": f_35mm,
                            "index": idx,
                        }
                        result["poses"].append(rec)
                        result["by_name"][name] = rec
                        result["by_name"][os.path.basename(name)] = rec
                        result["by_stem"][Path(name).stem] = rec
                        result["by_idx"][idx] = rec
                        idx += 1
                    except (ValueError, IndexError):
                        continue
        else:
            # TUM Trajectory format: timestamp tx ty tz qx qy qz qw
            M_world = np.array([
                [1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0],
                [0.0, -1.0,  0.0]
            ], dtype=np.float64)

            idx = 0
            for line in lines:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        t = float(parts[0])
                        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                        # Quat to matrix c2w
                        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                        if norm > 1e-12:
                            qx /= norm; qy /= norm; qz /= norm; qw /= norm
                        R_c2w = np.array([
                            [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw), 2.0*(qx*qz + qy*qw)],
                            [2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
                            [2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw), 1.0 - 2.0*(qx*qx + qy*qy)]
                        ], dtype=np.float64)

                        C_rc = M_world @ np.array([tx, ty, tz], dtype=np.float64)
                        fwd_rc = M_world @ (R_c2w @ np.array([0.0, 0.0, 1.0]))
                        up_rc = M_world @ (R_c2w @ np.array([0.0, -1.0, 0.0]))
                        yaw, pitch, roll = rc_axes_to_euler(fwd_rc, up_rc)

                        name = f"frame_{idx:05d}"
                        rec = {
                            "name": name,
                            "x": C_rc[0],
                            "y": C_rc[1],
                            "alt": C_rc[2],
                            "yaw": yaw,
                            "pitch": pitch,
                            "roll": roll,
                            "f_35mm": 35.0,
                            "timestamp": t,
                            "index": idx,
                        }
                        result["poses"].append(rec)
                        result["by_name"][name] = rec
                        result["by_stem"][name] = rec
                        result["by_idx"][idx] = rec
                        idx += 1
                    except (ValueError, IndexError):
                        continue

        result["count"] = len(result["poses"])
    except Exception:
        pass

    return result


def compound_realitycapture_pose(
    pano_pose: Optional[Dict[str, Any]],
    rig_cam: Tuple[str, float, float, float, float],
    crop_w: int,
    crop_h: int,
    k1: float = 0.0,
    k2: float = 0.0,
    override_orientations_from_rig: bool = True,
) -> Dict[str, Any]:
    """
    Compounds a 360 panorama 6-DoF pose with a virtual rig camera offset into
    RealityScan / RealityCapture standard Euler orientation and coordinates.

    If override_orientations_from_rig is True (default):
        - Metric position (x, y, alt) is taken from the imported panorama trajectory prior.
        - Imported yaw, pitch, roll from the trajectory are discarded and overridden with
          the clean, canonical Euler orientation derived directly from the splitting rig.
    If override_orientations_from_rig is False:
        - Panorama 3D rotation and rig orientation are multiplied in SO(3).
    """
    cam_name, rig_yaw, rig_pitch, rig_roll, fov_deg = rig_cam

    if pano_pose is not None:
        x = float(pano_pose.get("x", 0.0))
        y = float(pano_pose.get("y", 0.0))
        alt = float(pano_pose.get("alt", 0.0))
        p_yaw = float(pano_pose.get("yaw", 0.0))
        p_pitch = float(pano_pose.get("pitch", 90.0))
        p_roll = float(pano_pose.get("roll", 0.0))
    else:
        x, y, alt = 0.0, 0.0, 0.0
        p_yaw = 0.0
        p_pitch = 90.0
        p_roll = 0.0

    rig_r, rig_u, rig_f = basis(rig_yaw, rig_pitch, rig_roll)

    if override_orientations_from_rig or pano_pose is None:
        # Map local rig basis (X-right, Y-up, Z-fwd) to canonical RC world frame (X-east, Y-north, Z-up)
        # using the identity RC orientation (yaw=0°, pitch=90° horizontal, roll=0°)
        r_rc, u_rc, f_rc = get_rc_axes(0.0, 90.0, 0.0)
        M_rig_to_rc = np.column_stack([r_rc, u_rc, f_rc])
        fwd_rc = M_rig_to_rc @ rig_f
        up_rc = M_rig_to_rc @ rig_u
        yaw, pitch, roll = rc_axes_to_euler(fwd_rc, up_rc)
        fwd_norm = fwd_rc / max(float(np.linalg.norm(fwd_rc)), 1e-12)
        up_norm = up_rc / max(float(np.linalg.norm(up_rc)), 1e-12)
        right_rc = np.cross(fwd_norm, up_norm)
        if np.dot(right_rc, M_rig_to_rc @ rig_r) < 0:
            right_rc = -right_rc
        right_norm = right_rc / max(float(np.linalg.norm(right_rc)), 1e-12)
        down_norm = -up_norm
        R_mat = np.vstack([right_norm, down_norm, fwd_norm])
    else:
        r_p, u_p, f_p = get_rc_axes(p_yaw, p_pitch, p_roll)
        R_rig_to_rc = np.column_stack([r_p, u_p, f_p])
        fwd_rc = R_rig_to_rc @ rig_f
        up_rc = R_rig_to_rc @ rig_u
        yaw, pitch, roll = rc_axes_to_euler(fwd_rc, up_rc)
        fwd_norm = fwd_rc / max(float(np.linalg.norm(fwd_rc)), 1e-12)
        up_norm = up_rc / max(float(np.linalg.norm(up_rc)), 1e-12)
        right_rc = np.cross(fwd_norm, up_norm)
        if np.dot(right_rc, R_rig_to_rc @ rig_r) < 0:
            right_rc = -right_rc
        right_norm = right_rc / max(float(np.linalg.norm(right_rc)), 1e-12)
        down_norm = -up_norm
        R_mat = np.vstack([right_norm, down_norm, fwd_norm])

    fov_rad = math.radians(fov_deg)
    fx = (float(crop_w) / 2.0) / math.tan(fov_rad / 2.0)
    f_35mm = fx * (36.0 / max(crop_w, crop_h))
    R_flat = [0.0 if abs(v) < 1e-12 else float(v) for v in R_mat.flatten()]

    return {
        "x": x,
        "y": y,
        "alt": alt,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "f_35mm": f_35mm,
        "px_norm": 0.0,
        "py_norm": 0.0,
        "k1": float(k1),
        "k2": float(k2),
        "k3": 0.0,
        "k4": 0.0,
        "t1": 0.0,
        "t2": 0.0,
        "R": R_flat,
    }


def format_rc_csv_line(img_name: str, pose: Dict[str, Any]) -> str:
    """Formats a single line for RealityCapture cameras.csv."""
    def fmt_v(v):
        if abs(v) < 1e-12:
            return "0"
        return f"{v:.10g}"

    return (
        f"{img_name},{pose['x']:.6f},{pose['y']:.6f},{pose['alt']:.6f},"
        f"{pose['yaw']:.4f},{pose['pitch']:.4f},{pose['roll']:.4f},{pose['f_35mm']:.4f},"
        f"{pose['px_norm']:.6f},{pose['py_norm']:.6f},"
        f"{fmt_v(pose['k1'])},{fmt_v(pose['k2'])},0,0,0,0\n"
    )


def extract_frame_index(name: str, fallback_idx: int = 0) -> int:
    """Extracts integer index from frame/panorama filename (e.g. 'Container1_Mask036' -> 36)."""
    import re
    m = re.findall(r'\d+', name)
    if m:
        try:
            return int(m[-1])
        except Exception:
            pass
    return fallback_idx


def generate_realityscan_xmp(
    rig_id: Optional[str] = None,
    rig_instance_id: Optional[str] = None,
    pose: Dict[str, Any] = None,
    pose_prior: str = "approximate",
    pos_accuracy: Tuple[float, float, float] = (0.3, 0.3, 0.3),
    rot_accuracy: Tuple[float, float, float] = (0.001, 0.001, 0.001),
    locked_pose_group: Optional[int] = None,
) -> str:
    """
    Generates standard RealityScan 2.2 / RealityCapture compliant XMP metadata XML string.
    Follows canonical schema: http://www.capturingreality.com/ns/xcr/1.1/
    with pure element-based tags and LockedPoseGroup for multi-camera rigid grouping (Guide 03).
    """
    if pose is None:
        pose = {}
    pos_x = float(pose.get("x", 0.0))
    pos_y = float(pose.get("y", 0.0))
    pos_z = float(pose.get("alt", 0.0))

    R = pose.get("R")
    if R is not None and len(R) == 9:
        R_vals = [0.0 if abs(v) < 1e-12 else float(v) for v in R]
    else:
        r_c, u_c, f_c = get_rc_axes(pose.get("yaw", 0.0), pose.get("pitch", 90.0), pose.get("roll", 0.0))
        d_c = -u_c
        R_arr = np.vstack([r_c, d_c, f_c])
        R_vals = [0.0 if abs(v) < 1e-12 else float(v) for v in R_arr.flatten()]

    r_str = " ".join(f"{v:.8f}" for v in R_vals)

    f_35mm = float(pose.get("f_35mm", 10.392305))
    px = float(pose.get("px_norm", 0.0))
    py = float(pose.get("py_norm", 0.0))
    k1 = float(pose.get("k1", 0.0))
    k2 = float(pose.get("k2", 0.0))
    k3 = float(pose.get("k3", 0.0))
    k4 = float(pose.get("k4", 0.0))
    t1 = float(pose.get("t1", 0.0))
    t2 = float(pose.get("t2", 0.0))
    dist_str = f"{k1:.6f} {k2:.6f} {k3:.6f} {k4:.6f} {t1:.6f} {t2:.6f}"

    open_desc = [
        '    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1/"',
        '      xcr:Version="3"',
    ]
    if locked_pose_group is not None:
        open_desc.append(f'      xcr:LockedPoseGroup="{locked_pose_group}">')
    else:
        open_desc[-1] += '>'

    xml_lines = [
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
    ] + open_desc + [
        '      <xcr:CalibrationGroup>1</xcr:CalibrationGroup>',
        '      <xcr:DistortionGroup>1</xcr:DistortionGroup>',
        '      <xcr:InTexturing>1</xcr:InTexturing>',
        '      <xcr:InMeshing>1</xcr:InMeshing>',
        '      ',
        '      <!-- Calibration Priors -->',
        f'      <xcr:FocalLength35mm>{f_35mm:.6f}</xcr:FocalLength35mm>',
        f'      <xcr:PrincipalPointU>{px:.6f}</xcr:PrincipalPointU>',
        f'      <xcr:PrincipalPointV>{py:.6f}</xcr:PrincipalPointV>',
        '      <xcr:DistortionModel>brown3</xcr:DistortionModel>',
        f'      <xcr:DistortionCoeficients>{dist_str}</xcr:DistortionCoeficients>',
        '      <xcr:CalibrationPrior>locked</xcr:CalibrationPrior>',
        '      <xcr:LensDistortionPrior>locked</xcr:LensDistortionPrior>',
        '',
        '      <!-- Trajectory & Pose Priors -->',
        '      <xcr:Coordinates>absolute</xcr:Coordinates>',
        f'      <xcr:Position>{pos_x:.6f} {pos_y:.6f} {pos_z:.6f}</xcr:Position>',
        f'      <xcr:Rotation>{r_str}</xcr:Rotation>',
        f'      <xcr:PosePrior>{pose_prior}</xcr:PosePrior>',
        f'      <xcr:PoseAccuracyPosition>{pos_accuracy[0]:.6f} {pos_accuracy[1]:.6f} {pos_accuracy[2]:.6f}</xcr:PoseAccuracyPosition>',
        f'      <xcr:PoseAccuracyRotation>{rot_accuracy[0]:.6f} {rot_accuracy[1]:.6f} {rot_accuracy[2]:.6f}</xcr:PoseAccuracyRotation>',
        '    </rdf:Description>',
        '  </rdf:RDF>',
        '</x:xmpmeta>',
        ''
    ]
    return "\n".join(xml_lines)


# ======================================================================
# Panoramic Rig Splitter Class
# ======================================================================

class PanoramicRigSplitter:
    """
    Vectorized high-performance equirectangular rig projector.
    Precomputes coordinate grids and cv2.remap tables for arbitrary virtual camera arrays.
    Supports arbitrary aspect ratios (W x H) and radial distortion compensation (k1, k2).
    """

    def __init__(
        self,
        pano_width: int,
        pano_height: int,
        rig_cameras: List[Tuple[str, float, float, float, float]],
        crop_size: int = 1024,
        crop_width: Optional[int] = None,
        crop_height: Optional[int] = None,
        k1: float = 0.0,
        k2: float = 0.0,
        interpolation: int = 1,  # cv2.INTER_LINEAR = 1
    ):
        self.pano_w = pano_width
        self.pano_h = pano_height
        self.rig_cameras = rig_cameras
        self.out_w = crop_width if crop_width is not None else crop_size
        self.out_h = crop_height if crop_height is not None else crop_size
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.interpolation = interpolation

        self.maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.rotations: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, Dict[str, Any]] = {}

        self._build_remap_tables()

    def _build_remap_tables(self):
        """Precomputes coordinate grids for each camera in the rig."""
        w = self.out_w
        h = self.out_h

        cx = float(w) / 2.0
        cy = float(h) / 2.0

        for cam_name, yaw_deg, pitch_deg, roll_deg, fov_deg in self.rig_cameras:
            R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
            self.rotations[cam_name] = R

            # Focal lengths:
            # fx = (w / 2) / tan(fov_h / 2)
            # fy = (h / 2) / tan(fov_v / 2)
            fov_h_rad = math.radians(fov_deg)
            fx = (float(w) / 2.0) / math.tan(fov_h_rad / 2.0)

            # In rectilinear pinhole: fy = fx (square pixels)
            # fy matches fx when sensor pixels are square
            fy = fx
            fov_v_deg = calculate_vertical_fov(fov_deg, w, h)

            self.intrinsics[cam_name] = {
                "model": "OPENCV" if (abs(self.k1) > 1e-6 or abs(self.k2) > 1e-6) else "PINHOLE",
                "width": w,
                "height": h,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": self.k1,
                "k2": self.k2,
                "fov_h_deg": fov_deg,
                "fov_v_deg": fov_v_deg,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
            }

            # Pixel coordinate grid: px in [0..w-1], py in [0..h-1]
            px_grid = np.arange(w, dtype=np.float32)
            py_grid = np.arange(h, dtype=np.float32)
            px_m, py_m = np.meshgrid(px_grid, py_grid)

            # Normalized camera sensor plane coordinates:
            # u = (px - cx) / fx, v = -(py - cy) / fy (inverted Y for 3D camera coordinate convention)
            u = (px_m - cx) / fx
            v = -(py_m - cy) / fy

            # Apply radial lens distortion if specified (Brown-Conrady model)
            if abs(self.k1) > 1e-6 or abs(self.k2) > 1e-6:
                r2 = u * u + v * v
                radial = 1.0 + self.k1 * r2 + self.k2 * (r2 * r2)
                u = u * radial
                v = v * radial

            # Ray directions in local camera frame: [u, v, 1.0]
            rays_cam = np.stack([u, v, np.ones_like(u)], axis=-1)  # (h, w, 3)

            # Rotate rays into equirectangular global coordinate frame: ray_global = rays_cam @ R^T
            rays_global = rays_cam @ R.T

            x = rays_global[..., 0]
            y = rays_global[..., 1]
            z = rays_global[..., 2]

            # Longitude lambda in [-pi, pi], Latitude phi in [-pi/2, pi/2]
            lon = np.arctan2(x, z)
            hypot = np.sqrt(x * x + z * z)
            lat = np.arctan2(y, hypot)

            # Map to equirectangular pixel coordinates
            map_x = ((lon / (2.0 * np.pi) + 0.5) * self.pano_w).astype(np.float32)
            map_y = ((0.5 - lat / np.pi) * self.pano_h).astype(np.float32)

            # Seamless boundary wrapping for continuous 360° horizontal stitching
            map_x = np.mod(map_x, self.pano_w)
            map_y = np.clip(map_y, 0, self.pano_h - 1)

            self.maps[cam_name] = (map_x, map_y)

    def split_panorama(self, pano_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Projects a single equirectangular panorama into all rig views."""
        if not HAS_CV2:
            raise RuntimeError("OpenCV (cv2) is required for panoramic image remapping.")

        h, w = pano_bgr.shape[:2]
        if w != self.pano_w or h != self.pano_h:
            self.pano_w = w
            self.pano_h = h
            self._build_remap_tables()

        crops = {}
        for cam_name, (map_x, map_y) in self.maps.items():
            crop = cv2.remap(
                pano_bgr,
                map_x,
                map_y,
                interpolation=self.interpolation,
                borderMode=cv2.BORDER_WRAP
            )
            crops[cam_name] = crop
        return crops

    def get_c2w_matrix(self, cam_name: str) -> np.ndarray:
        """Returns 4x4 camera-to-world matrix for the virtual pinhole camera."""
        R = self.rotations[cam_name]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        return c2w


# ======================================================================
# Video Metadata & Extraction Helper
# ======================================================================

def probe_video_metadata(video_path: str) -> Dict[str, Any]:
    """Inspects video properties using OpenCV."""
    if not HAS_CV2:
        raise RuntimeError("OpenCV is required to probe video.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "width": w,
        "height": h,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
    }


# ======================================================================
# Equirectangular Mask Auto-Detection & Lookup Helpers
# ======================================================================

def find_auto_mask_dir(input_dir: str) -> Optional[str]:
    """
    Auto-detects an equirectangular mask directory inside or next to the dataset.
    Searches common conventions:
    'mask', 'masks', 'Mask', 'Masks', 'Mask Pass', 'mask_pass', 'Mask_Pass', 'mask pass',
    'Masks Pass', 'masks_pass', 'Mask 1', 'Mask 2', or any subdirectory whose name
    contains 'mask' (case-insensitive), excluding output directories.
    Validates that the candidate directory contains at least one valid image file.
    """
    if not input_dir or not os.path.exists(input_dir):
        return None
    scan_dir = input_dir if os.path.isdir(input_dir) else os.path.dirname(input_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

    # Priority 1: standard naming conventions
    priority_names = [
        "mask", "masks", "Mask", "Masks", "Mask Pass", "mask_pass", "Mask_Pass",
        "mask pass", "Masks Pass", "masks_pass", "Mask 1", "Mask 2", "masks_2"
    ]
    for name in priority_names:
        cand = os.path.join(scan_dir, name)
        if os.path.isdir(cand):
            try:
                imgs = [f for f in os.listdir(cand) if os.path.splitext(f.lower())[1] in valid_exts]
                if imgs:
                    return str(Path(cand).resolve())
            except Exception:
                pass

    # Priority 2: any subdirectory in scan_dir containing 'mask' (case-insensitive)
    ignored = {"pinhole_rig", "perspectives", "results", "images", "cameras_csv", ".git", "cache", "tmp"}
    try:
        for entry in os.listdir(scan_dir):
            if entry.lower() in ignored or entry in ignored:
                continue
            cand = os.path.join(scan_dir, entry)
            if os.path.isdir(cand) and "mask" in entry.lower():
                imgs = [f for f in os.listdir(cand) if os.path.splitext(f.lower())[1] in valid_exts]
                if imgs:
                    return str(Path(cand).resolve())
    except Exception:
        pass

    # Priority 3: check parent directory if scan_dir is e.g. an "images" or "raw" folder
    parent_dir = os.path.dirname(scan_dir)
    if parent_dir and os.path.isdir(parent_dir) and parent_dir != scan_dir:
        for name in priority_names:
            cand = os.path.join(parent_dir, name)
            if os.path.isdir(cand):
                try:
                    imgs = [f for f in os.listdir(cand) if os.path.splitext(f.lower())[1] in valid_exts]
                    if imgs:
                        return str(Path(cand).resolve())
                except Exception:
                    pass

    return None


def build_mask_lookup(mask_dir: str) -> Optional[Dict[str, Any]]:
    """
    Indexes all mask images in a directory for fast lookup by filename, stem, or index.
    """
    if not mask_dir or not os.path.isdir(mask_dir):
        return None

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    try:
        files = sorted([f for f in os.listdir(mask_dir) if os.path.splitext(f.lower())[1] in valid_exts])
    except Exception:
        return None

    if not files:
        return None

    by_name = {}
    by_name_lower = {}
    by_stem = {}
    by_stem_lower = {}
    by_idx = {}

    for idx, f in enumerate(files):
        p = os.path.join(mask_dir, f)
        by_name[f] = p
        by_name_lower[f.lower()] = p
        stem = Path(f).stem
        by_stem[stem] = p
        by_stem_lower[stem.lower()] = p
        by_idx[idx] = p

    return {
        "dir": str(Path(mask_dir).resolve()),
        "count": len(files),
        "files": files,
        "by_name": by_name,
        "by_name_lower": by_name_lower,
        "by_stem": by_stem,
        "by_stem_lower": by_stem_lower,
        "by_idx": by_idx,
    }


def find_mask_for_frame(mask_lookup: Optional[Dict[str, Any]], fname_or_stem: str, frame_idx: int = -1) -> Optional[str]:
    """
    Finds the corresponding mask file path for a given main frame filename or stem.
    """
    if not mask_lookup or mask_lookup.get("count", 0) == 0:
        return None

    base = os.path.basename(fname_or_stem)
    stem = Path(fname_or_stem).stem
    stem_l = stem.lower()

    # 1. Exact filename or lower
    if base in mask_lookup["by_name"]:
        return mask_lookup["by_name"][base]
    if base.lower() in mask_lookup["by_name_lower"]:
        return mask_lookup["by_name_lower"][base.lower()]

    # 2. Exact stem or lower
    if stem in mask_lookup["by_stem"]:
        return mask_lookup["by_stem"][stem]
    if stem_l in mask_lookup["by_stem_lower"]:
        return mask_lookup["by_stem_lower"][stem_l]

    # 3. Stripping or adding _mask / .mask variations
    cand_stems = [
        f"{stem}_mask", f"{stem}.mask",
        stem.replace("_mask", "").replace(".mask", ""),
        stem.replace("_Mask", "").replace(".Mask", ""),
    ]
    for cs in cand_stems:
        if cs in mask_lookup["by_stem"]:
            return mask_lookup["by_stem"][cs]
        if cs.lower() in mask_lookup["by_stem_lower"]:
            return mask_lookup["by_stem_lower"][cs.lower()]

    # 4. Fallback by index
    if 0 <= frame_idx < mask_lookup["count"]:
        return mask_lookup["by_idx"][frame_idx]

    return None


# ======================================================================
# Batch Dataset Processing & Transforms Exporter
# ======================================================================

def split_panorama_dataset(
    input_dir: str,
    output_dir: str,
    mode: str = "ring",             # "ring", "cubemap", "dome", "fibonacci", "custom"
    num_cameras: int = 8,           # When mode="ring" or "fibonacci"
    fov_deg: float = 90.0,          # Camera FOV in degrees
    crop_size: int = 1024,          # Output width & height per view (square)
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
    k1: float = 0.0,                # Radial distortion parameter
    k2: float = 0.0,
    pitch_tier: float = 35.0,       # For mode="dome"
    tier_cams: int = 2,             # For mode="dome"
    include_zenith: bool = True,    # For mode="dome"
    include_nadir: bool = True,     # For mode="dome"
    custom_rig: Optional[List[Tuple[str, float, float, float, float]]] = None,
    image_format: str = "jpg",      # "jpg" or "png"
    quality: int = 95,              # JPEG quality (1-100)
    video_target_fps: Optional[float] = None,
    video_frame_stride: int = 1,
    video_max_frames: int = 0,
    export_transforms_json: bool = True,
    export_realityscan_csv: bool = True,
    export_xmp: bool = True,
    rig_id: str = "cubemap_rig",
    xmp_pose_prior: str = "approximate",
    xmp_pos_accuracy: Tuple[float, float, float] = (0.3, 0.3, 0.3),
    xmp_rot_accuracy: Tuple[float, float, float] = (0.001, 0.001, 0.001),
    subfolder_per_perspective: bool = True,
    dvo_trajectory_csv: Optional[str] = None,
    override_orientations_from_rig: bool = True,
    mask_dir: Optional[str] = None,
    auto_detect_mask: bool = True,
    masks_only: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Splits an entire directory of 360° panoramas OR a 360° video file into rectilinear pinhole views
    and exports a transforms.json manifest ready for COLMAP, Nerfstudio, PFGS360, or LichtFeld,
    along with RealityScan / RealityCapture standard cameras.csv tables for every perspective,
    all_cameras.csv manifest, and .xmp XML metadata sidecars for RealityScan Rig Parser.
    Also projects equirectangular mask layers into matching <image>.mask.<ext> files alongside views.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for panoramic rig projection.")

    start_time = time.time()
    in_abs = str(Path(input_dir).resolve())
    out_abs = str(Path(output_dir).resolve())
    os.makedirs(out_abs, exist_ok=True)

    out_w = crop_width if crop_width is not None else crop_size
    out_h = crop_height if crop_height is not None else crop_size

    # Output directory layout
    if subfolder_per_perspective:
        perspectives_base = os.path.join(out_abs, "perspectives")
        os.makedirs(perspectives_base, exist_ok=True)
        images_out = perspectives_base
    else:
        perspectives_base = None
        images_out = os.path.join(out_abs, "images")
        os.makedirs(images_out, exist_ok=True)

    # 1. Build Rig Array
    mode_lower = mode.lower()
    if custom_rig:
        rig = custom_rig
        rig_title = f"Custom Rig ({len(rig)} Cameras)"
    elif mode_lower in ("cubemap", "cube", "6face"):
        rig = get_cubemap_rig(fov_deg=fov_deg)
        rig_title = "6-Face Cubemap"
    elif mode_lower in ("dome", "spherical_dome", "14cam"):
        rig = get_spherical_dome_rig(
            equator_cams=num_cameras,
            pitch_tier=pitch_tier,
            tier_cams=tier_cams,
            include_zenith=include_zenith,
            include_nadir=include_nadir,
            fov_deg=fov_deg
        )
        rig_title = f"{len(rig)}-Camera Multi-Tier Spherical Dome"
    elif mode_lower in ("fibonacci", "golden", "sphere"):
        rig = get_fibonacci_sphere_rig(num_cameras=num_cameras, fov_deg=fov_deg)
        rig_title = f"{num_cameras}-Camera Fibonacci Sphere (Golden Spiral)"
    else:  # "ring"
        rig = get_horizontal_ring_rig(num_cameras=num_cameras, fov_deg=fov_deg)
        rig_title = f"{num_cameras}-Camera Horizontal Ring"

    # Ingest 360DVO trajectory prior if provided
    traj_data = None
    if dvo_trajectory_csv and os.path.isfile(dvo_trajectory_csv):
        traj_data = load_dvo_trajectory(dvo_trajectory_csv)

    # RealityScan CSV row collections:
    # 1. Master rows for out_abs/cameras.csv
    # 2. Per-camera rows: Dict[cam_name, List[str]]
    # 3. All-perspectives rows for perspectives_base/all_cameras.csv
    rc_master_rows = [RC_CSV_HEADER]
    rc_all_persp_rows = [RC_CSV_HEADER]
    rc_camera_rows = {c[0]: [RC_CSV_HEADER] for c in rig}

    video_exts = {".mp4", ".mov", ".insv", ".mkv", ".avi"}
    is_video_input = os.path.isfile(in_abs) and (os.path.splitext(in_abs.lower())[1] in video_exts)

    fmt = "png" if image_format.lower() == "png" else "jpg"
    ext = f".{fmt}"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    total_views_saved = 0
    total_masks_saved = 0
    transforms_frames = []
    total_source_items = 0

    # Locate equirectangular masks if requested or auto-detected
    effective_mask_dir = None
    mask_lookup = None
    mask_video_path = None

    if mask_dir:
        cand_m = str(Path(mask_dir).resolve())
        if os.path.isdir(cand_m):
            effective_mask_dir = cand_m
        elif os.path.isfile(cand_m):
            if os.path.splitext(cand_m.lower())[1] in video_exts:
                mask_video_path = cand_m
            else:
                effective_mask_dir = os.path.dirname(cand_m)
    elif auto_detect_mask:
        effective_mask_dir = find_auto_mask_dir(in_abs)

    if effective_mask_dir and os.path.isdir(effective_mask_dir):
        mask_lookup = build_mask_lookup(effective_mask_dir)

    if is_video_input:
        # -------------------------------------------------------------
        # Direct Video Extraction & Projection Loop
        # -------------------------------------------------------------
        cap = cv2.VideoCapture(in_abs)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {in_abs}")

        mask_cap = None
        if mask_video_path and os.path.isfile(mask_video_path):
            mask_cap = cv2.VideoCapture(mask_video_path)

        pano_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        pano_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        splitter = PanoramicRigSplitter(
            pano_width=pano_w,
            pano_height=pano_h,
            rig_cameras=rig,
            crop_width=out_w,
            crop_height=out_h,
            k1=k1,
            k2=k2
        )

        # Determine stride from target FPS
        if video_target_fps is not None and video_target_fps > 0:
            stride = max(1, int(round(native_fps / video_target_fps)))
        else:
            stride = max(1, video_frame_stride)

        # Estimate sampled frame count
        expected_frames = total_vid_frames // stride
        if video_max_frames > 0:
            expected_frames = min(expected_frames, video_max_frames)
        total_source_items = expected_frames

        frame_idx = 0
        saved_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                saved_frame_idx += 1
                base_name = f"frame_{saved_frame_idx:05d}"
                timestamp_sec = frame_idx / native_fps

                # Locate 360DVO pose prior for this frame
                pano_pose = None
                if not masks_only and traj_data and traj_data["count"] > 0:
                    v_stem = f"frame_{saved_frame_idx:05d}"
                    v_idx = saved_frame_idx - 1
                    if v_stem in traj_data["by_stem"]:
                        pano_pose = traj_data["by_stem"][v_stem]
                    elif v_idx in traj_data["by_idx"]:
                        pano_pose = traj_data["by_idx"][v_idx]

                mask_crops = {}
                if mask_cap and mask_cap.isOpened():
                    ret_m, m_frame = mask_cap.read()
                    if ret_m and m_frame is not None:
                        mask_crops = splitter.split_panorama(m_frame)
                elif mask_lookup:
                    m_path = find_mask_for_frame(mask_lookup, base_name, saved_frame_idx - 1)
                    if m_path and os.path.isfile(m_path):
                        m_img = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
                        if m_img is not None:
                            mask_crops = splitter.split_panorama(m_img)

                crops = splitter.split_panorama(frame) if not masks_only else {}

                for rig_cam_tuple in rig:
                    cam_name = rig_cam_tuple[0]

                    if subfolder_per_perspective:
                        cam_dir = os.path.join(perspectives_base, cam_name)
                        os.makedirs(cam_dir, exist_ok=True)
                        file_in_cam = f"{base_name}_{cam_name}{ext}"
                        out_filepath = os.path.join(cam_dir, file_in_cam)
                        rel_path_root = f"perspectives/{cam_name}/{file_in_cam}"
                        local_name = file_in_cam
                        mask_filename = f"{base_name}_{cam_name}.mask{ext}"
                        mask_filepath = os.path.join(cam_dir, mask_filename)
                    else:
                        out_filename = f"{base_name}_{cam_name}{ext}"
                        out_filepath = os.path.join(images_out, out_filename)
                        rel_path_root = f"images/{out_filename}"
                        local_name = out_filename
                        mask_filename = f"{base_name}_{cam_name}.mask{ext}"
                        mask_filepath = os.path.join(images_out, mask_filename)

                    if not masks_only and cam_name in crops:
                        cv2.imwrite(out_filepath, crops[cam_name], encode_params)
                        total_views_saved += 1

                        # RealityScan / RealityCapture standard pose & XMP
                        comp_pose = None
                        if export_realityscan_csv or export_xmp:
                            comp_pose = compound_realitycapture_pose(
                                pano_pose=pano_pose,
                                rig_cam=rig_cam_tuple,
                                crop_w=out_w,
                                crop_h=out_h,
                                k1=k1,
                                k2=k2,
                                override_orientations_from_rig=override_orientations_from_rig,
                            )

                        if export_realityscan_csv and comp_pose is not None:
                            master_line = format_rc_csv_line(rel_path_root, comp_pose)
                            rc_master_rows.append(master_line)

                            if subfolder_per_perspective:
                                all_persp_line = format_rc_csv_line(f"{cam_name}/{file_in_cam}", comp_pose)
                                rc_all_persp_rows.append(all_persp_line)

                            local_line = format_rc_csv_line(local_name, comp_pose)
                            if cam_name in rc_camera_rows:
                                rc_camera_rows[cam_name].append(local_line)

                        if export_xmp and comp_pose is not None:
                            frame_int_idx = extract_frame_index(base_name, frame_idx)
                            xmp_xml = generate_realityscan_xmp(
                                rig_id=rig_id,
                                rig_instance_id=base_name,
                                pose=comp_pose,
                                pose_prior=xmp_pose_prior,
                                pos_accuracy=xmp_pos_accuracy,
                                rot_accuracy=xmp_rot_accuracy,
                                locked_pose_group=frame_int_idx,
                            )
                            xmp_file = f"{os.path.splitext(out_filepath)[0]}.xmp"
                            with open(xmp_file, "w", encoding="utf-8") as xf:
                                xf.write(xmp_xml)

                        if export_transforms_json:
                            c2w = splitter.get_c2w_matrix(cam_name)
                            transforms_frames.append({
                                "file_path": rel_path_root,
                                "transform_matrix": c2w.tolist(),
                                "camera_angle_name": cam_name,
                                "source_frame": saved_frame_idx,
                                "timestamp_sec": round(timestamp_sec, 3),
                                "yaw_deg": splitter.intrinsics[cam_name]["yaw_deg"],
                                "pitch_deg": splitter.intrinsics[cam_name]["pitch_deg"],
                            })

                    # Mask Export (NO XMP, NO CSV)
                    if cam_name in mask_crops:
                        cv2.imwrite(mask_filepath, mask_crops[cam_name], encode_params)
                        total_masks_saved += 1

                if progress_callback:
                    progress_callback(saved_frame_idx, expected_frames, f"{base_name} (t={timestamp_sec:.2f}s)")

                if video_max_frames > 0 and saved_frame_idx >= video_max_frames:
                    break

            frame_idx += 1

        cap.release()
        if mask_cap:
            mask_cap.release()
        total_source_items = saved_frame_idx

    else:
        # -------------------------------------------------------------
        # Image Directory Projection Loop
        # -------------------------------------------------------------
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
        if os.path.isdir(in_abs):
            files = sorted([
                f for f in os.listdir(in_abs)
                if os.path.splitext(f.lower())[1] in valid_exts and os.path.isfile(os.path.join(in_abs, f))
            ])
            scan_dir = in_abs
        else:
            files = [os.path.basename(in_abs)]
            scan_dir = os.path.dirname(in_abs)

        if not files:
            if mask_lookup and mask_lookup.get("files"):
                files = mask_lookup["files"]
                scan_dir = mask_lookup["dir"]
                masks_only = True
            else:
                raise FileNotFoundError(f"No valid panorama images found in {input_dir}")

        total_source_items = len(files)

        # Read first sample to initialize splitter
        sample_path = os.path.join(scan_dir, files[0])
        sample_img = cv2.imread(sample_path)
        if sample_img is None:
            if mask_lookup and mask_lookup.get("files"):
                sample_path = os.path.join(mask_lookup["dir"], mask_lookup["files"][0])
                sample_img = cv2.imread(sample_path)
            if sample_img is None:
                raise RuntimeError(f"Failed to read sample panorama: {sample_path}")
        pano_h, pano_w = sample_img.shape[:2]

        splitter = PanoramicRigSplitter(
            pano_width=pano_w,
            pano_height=pano_h,
            rig_cameras=rig,
            crop_width=out_w,
            crop_height=out_h,
            k1=k1,
            k2=k2
        )

        for idx, fname in enumerate(files):
            img_path = os.path.join(scan_dir, fname)
            base_name = Path(fname).stem

            # Locate and split equirectangular mask if available
            mask_crops = {}
            if mask_lookup:
                m_path = find_mask_for_frame(mask_lookup, fname, idx)
                if m_path and os.path.isfile(m_path):
                    m_img = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
                    if m_img is not None:
                        mask_crops = splitter.split_panorama(m_img)

            if not masks_only:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                crops = splitter.split_panorama(img)
            else:
                crops = {}

            # Locate 360DVO pose prior for this image
            pano_pose = None
            if not masks_only and traj_data and traj_data["count"] > 0:
                if fname in traj_data["by_name"]:
                    pano_pose = traj_data["by_name"][fname]
                elif os.path.basename(fname) in traj_data["by_name"]:
                    pano_pose = traj_data["by_name"][os.path.basename(fname)]
                elif base_name in traj_data["by_stem"]:
                    pano_pose = traj_data["by_stem"][base_name]
                elif idx in traj_data["by_idx"]:
                    pano_pose = traj_data["by_idx"][idx]

            for rig_cam_tuple in rig:
                cam_name = rig_cam_tuple[0]

                if subfolder_per_perspective:
                    cam_dir = os.path.join(perspectives_base, cam_name)
                    os.makedirs(cam_dir, exist_ok=True)
                    file_in_cam = f"{base_name}_{cam_name}{ext}"
                    out_filepath = os.path.join(cam_dir, file_in_cam)
                    rel_path_root = f"perspectives/{cam_name}/{file_in_cam}"
                    local_name = file_in_cam
                    mask_filename = f"{base_name}_{cam_name}.mask{ext}"
                    mask_filepath = os.path.join(cam_dir, mask_filename)
                else:
                    out_filename = f"{base_name}_{cam_name}{ext}"
                    out_filepath = os.path.join(images_out, out_filename)
                    rel_path_root = f"images/{out_filename}"
                    local_name = out_filename
                    mask_filename = f"{base_name}_{cam_name}.mask{ext}"
                    mask_filepath = os.path.join(images_out, mask_filename)

                if not masks_only and cam_name in crops:
                    cv2.imwrite(out_filepath, crops[cam_name], encode_params)
                    total_views_saved += 1

                    # RealityScan / RealityCapture standard pose & XMP
                    comp_pose = None
                    if export_realityscan_csv or export_xmp:
                        comp_pose = compound_realitycapture_pose(
                            pano_pose=pano_pose,
                            rig_cam=rig_cam_tuple,
                            crop_w=out_w,
                            crop_h=out_h,
                            k1=k1,
                            k2=k2,
                            override_orientations_from_rig=override_orientations_from_rig,
                        )

                    if export_realityscan_csv and comp_pose is not None:
                        master_line = format_rc_csv_line(rel_path_root, comp_pose)
                        rc_master_rows.append(master_line)

                        if subfolder_per_perspective:
                            all_persp_line = format_rc_csv_line(f"{cam_name}/{file_in_cam}", comp_pose)
                            rc_all_persp_rows.append(all_persp_line)

                        local_line = format_rc_csv_line(local_name, comp_pose)
                        if cam_name in rc_camera_rows:
                            rc_camera_rows[cam_name].append(local_line)

                    if export_xmp and comp_pose is not None:
                        frame_int_idx = extract_frame_index(base_name, idx)
                        xmp_xml = generate_realityscan_xmp(
                            rig_id=rig_id,
                            rig_instance_id=base_name,
                            pose=comp_pose,
                            pose_prior=xmp_pose_prior,
                            pos_accuracy=xmp_pos_accuracy,
                            rot_accuracy=xmp_rot_accuracy,
                            locked_pose_group=frame_int_idx,
                        )
                        xmp_file = f"{os.path.splitext(out_filepath)[0]}.xmp"
                        with open(xmp_file, "w", encoding="utf-8") as xf:
                            xf.write(xmp_xml)

                    if export_transforms_json:
                        c2w = splitter.get_c2w_matrix(cam_name)
                        transforms_frames.append({
                            "file_path": rel_path_root,
                            "transform_matrix": c2w.tolist(),
                            "camera_angle_name": cam_name,
                            "source_file": fname,
                            "yaw_deg": splitter.intrinsics[cam_name]["yaw_deg"],
                            "pitch_deg": splitter.intrinsics[cam_name]["pitch_deg"],
                        })

                # Mask Output: RealityCapture mask layer (<base>_<cam>.mask.<ext>)
                # CRITICAL: NO .xmp file, NO entry in cameras.csv
                if cam_name in mask_crops:
                    cv2.imwrite(mask_filepath, mask_crops[cam_name], encode_params)
                    total_masks_saved += 1

            if progress_callback:
                progress_callback(idx + 1, len(files), fname)

    # 2. Export transforms.json Manifest
    first_cam = rig[0][0]
    cam_intr = splitter.intrinsics[first_cam]
    transforms_path = os.path.join(out_abs, "transforms.json")

    if export_transforms_json and transforms_frames:
        transforms_data = {
            "camera_model": cam_intr["model"],
            "fl_x": cam_intr["fx"],
            "fl_y": cam_intr["fy"],
            "cx": cam_intr["cx"],
            "cy": cam_intr["cy"],
            "w": out_w,
            "h": out_h,
            "k1": k1,
            "k2": k2,
            "fov_h_deg": fov_deg,
            "fov_v_deg": cam_intr["fov_v_deg"],
            "rig_mode": mode,
            "total_cameras_per_panorama": len(rig),
            "frames": transforms_frames,
        }
        with open(transforms_path, "w", encoding="utf-8") as tf:
            json.dump(transforms_data, tf, indent=2)

    # 3. Export RealityScan / RealityCapture cameras.csv Files
    cameras_csv_path = None
    all_cameras_csv_path = None
    if export_realityscan_csv and len(rc_master_rows) > 1:
        # A. Master cameras.csv in root output directory
        cameras_csv_path = os.path.join(out_abs, "cameras.csv")
        with open(cameras_csv_path, "w", encoding="utf-8") as f:
            f.writelines(rc_master_rows)

        # B. If flat images/ directory exists, also mirror cameras.csv in images/
        if not subfolder_per_perspective and os.path.isdir(images_out):
            try:
                shutil.copy2(cameras_csv_path, os.path.join(images_out, "cameras.csv"))
            except Exception:
                pass

        # C. Export all_cameras.csv in perspectives/ and cameras_<cam>.csv inside each perspective folder
        if subfolder_per_perspective and perspectives_base and os.path.isdir(perspectives_base):
            # 1. Export all_cameras.csv next to perspective folders (containing all perspectives in one file)
            all_cameras_csv_path = os.path.join(perspectives_base, "all_cameras.csv")
            with open(all_cameras_csv_path, "w", encoding="utf-8") as f:
                f.writelines(rc_all_persp_rows)

            # 2. Export per-perspective cameras_<cam_name>.csv in each perspective subfolder
            for cam_name, rows in rc_camera_rows.items():
                cam_dir = os.path.join(perspectives_base, cam_name)
                if os.path.isdir(cam_dir):
                    cam_csv = os.path.join(cam_dir, f"cameras_{cam_name}.csv")
                    with open(cam_csv, "w", encoding="utf-8") as f:
                        f.writelines(rows)
                    # Clean up legacy cameras.csv to avoid naming confusion
                    old_csv = os.path.join(cam_dir, "cameras.csv")
                    if os.path.isfile(old_csv):
                        try:
                            os.remove(old_csv)
                        except Exception:
                            pass

        # D. Always export a dedicated cameras_csv/ folder with per-perspective tables
        csv_dir = os.path.join(out_abs, "cameras_csv")
        os.makedirs(csv_dir, exist_ok=True)
        for cam_name, rows in rc_camera_rows.items():
            cam_csv = os.path.join(csv_dir, f"{cam_name}_cameras.csv")
            with open(cam_csv, "w", encoding="utf-8") as f:
                f.writelines(rows)

    elapsed = time.time() - start_time

    # 4. Export rig_manifest.json with Full Telemetry
    overlap_val = calculate_overlap(len(rig), fov_deg) if mode_lower == "ring" else None
    rig_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rig_title": rig_title,
        "mode": mode,
        "num_views_per_pano": len(rig),
        "fov_h_deg": fov_deg,
        "fov_v_deg": cam_intr["fov_v_deg"],
        "overlap_pct": round(overlap_val, 2) if overlap_val is not None else None,
        "crop_width": out_w,
        "crop_height": out_h,
        "aspect_ratio": f"{out_w}:{out_h}",
        "k1": k1,
        "k2": k2,
        "is_video_source": is_video_input,
        "total_source_panoramas": total_source_items,
        "total_views_exported": total_views_saved,
        "subfolder_per_perspective": subfolder_per_perspective,
        "dvo_trajectory_used": dvo_trajectory_csv if (traj_data and traj_data["count"] > 0) else None,
        "mask_dir_used": effective_mask_dir,
        "total_masks_exported": total_masks_saved,
        "cameras_csv_path": cameras_csv_path,
        "elapsed_seconds": round(elapsed, 2),
        "transforms_path": transforms_path if export_transforms_json else None,
        "cameras": [
            {
                "name": c[0],
                "yaw_deg": c[1],
                "pitch_deg": c[2],
                "roll_deg": c[3],
                "fov_deg": c[4]
            }
            for c in rig
        ]
    }
    meta_path = os.path.join(out_abs, "rig_manifest.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(rig_meta, mf, indent=2)

    return {
        "success": True,
        "rig_mode": mode,
        "total_panoramas": total_source_items,
        "total_views": total_views_saved,
        "total_masks": total_masks_saved,
        "mask_dir": effective_mask_dir,
        "output_dir": out_abs,
        "images_dir": images_out,
        "subfolder_per_perspective": subfolder_per_perspective,
        "cameras_csv_path": cameras_csv_path,
        "all_cameras_csv_path": all_cameras_csv_path,
        "transforms_path": transforms_path if export_transforms_json else None,
        "elapsed_seconds": round(elapsed, 2),
    }


def split_masks_for_dataset(
    input_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
    mode: Optional[str] = None,
    fov_deg: Optional[float] = None,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
    k1: float = 0.0,
    k2: float = 0.0,
    subfolder_per_perspective: bool = True,
    image_format: str = "jpg",
    quality: int = 95,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Dedicated equirectangular mask projection helper.
    Automatically matches existing rig settings from rig_manifest.json if present in output_dir,
    and exports <base>_<cam>.mask.<ext> perspective masks without generating XMP sidecars or altering cameras.csv.
    """
    out_abs = str(Path(output_dir).resolve())
    manifest_path = os.path.join(out_abs, "rig_manifest.json")
    custom_rig = None

    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if mode is None:
                mode = meta.get("mode", "cubemap")
            if fov_deg is None and "fov_h_deg" in meta:
                fov_deg = float(meta["fov_h_deg"])
            if crop_width is None and "crop_width" in meta:
                crop_width = int(meta["crop_width"])
            if crop_height is None and "crop_height" in meta:
                crop_height = int(meta["crop_height"])
            if abs(k1) < 1e-9 and "k1" in meta:
                k1 = float(meta["k1"])
            if abs(k2) < 1e-9 and "k2" in meta:
                k2 = float(meta["k2"])
            if "subfolder_per_perspective" in meta:
                subfolder_per_perspective = bool(meta.get("subfolder_per_perspective", True))
            if "cameras" in meta and isinstance(meta["cameras"], list) and meta["cameras"]:
                custom_rig = [
                    (c["name"], float(c["yaw_deg"]), float(c["pitch_deg"]), float(c["roll_deg"]), float(c["fov_deg"]))
                    for c in meta["cameras"]
                ]
        except Exception:
            pass

    return split_panorama_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        mode=mode or "cubemap",
        fov_deg=fov_deg if fov_deg is not None else 90.0,
        crop_width=crop_width or 1024,
        crop_height=crop_height or 1024,
        k1=k1,
        k2=k2,
        custom_rig=custom_rig,
        subfolder_per_perspective=subfolder_per_perspective,
        image_format=image_format,
        quality=quality,
        mask_dir=mask_dir,
        auto_detect_mask=True,
        masks_only=True,
        export_transforms_json=False,
        export_realityscan_csv=False,
        export_xmp=False,
        progress_callback=progress_callback,
    )


# ======================================================================
# CLI Interface
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Abu Align & Train - 360° Multi-Camera Rig Projector & Panoramic Splitter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory of 360° panoramas or single image/video file")
    parser.add_argument("-o", "--output", help="Output directory (defaults to <input>_pinhole_rig)")
    parser.add_argument("-m", "--mode", choices=["ring", "cubemap", "dome", "fibonacci"], default="ring", help="Virtual camera rig type")
    parser.add_argument("-n", "--num_cameras", type=int, default=8, help="Number of virtual cameras when mode=ring or fibonacci")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of View (degrees) per camera view")
    parser.add_argument("-s", "--size", type=int, default=1024, help="Output image width and height (square crop)")
    parser.add_argument("--width", type=int, help="Output width (overrides --size)")
    parser.add_argument("--height", type=int, help="Output height (overrides --size)")
    parser.add_argument("-k1", "--distortion-k1", type=float, default=0.0, help="Radial lens distortion k1 parameter")
    parser.add_argument("--video-fps", type=float, default=None, help="Target FPS to extract when input is a 360° video")
    parser.add_argument("--video-stride", type=int, default=1, help="Frame stride when extracting from video")
    parser.add_argument("-f", "--format", choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--no-transforms", action="store_true", help="Skip generating transforms.json")
    parser.add_argument("--no-rc-csv", action="store_true", help="Skip exporting RealityScan / RealityCapture cameras.csv")
    parser.add_argument("--no-xmp", action="store_true", help="Skip exporting RealityScan / RealityCapture .xmp XML metadata sidecars")
    parser.add_argument("--rig-id", type=str, default="cubemap_rig", help="Rig identifier in XMP metadata (for RealityScan Rig Parser)")
    parser.add_argument("--xmp-pose-prior", type=str, default="approximate", choices=["approximate", "locked", "exact", "initial"], help="PosePrior in XMP metadata")
    parser.add_argument("--flat-images", action="store_true", help="Export flat images/ layout instead of per-perspective subfolders")
    parser.add_argument("--dvo-trajectory", type=str, default=None, help="Path to 360DVO cameras.csv or trajectory.txt prior")
    parser.add_argument("--compound-rotations", action="store_true", help="Compound full 3D rotation instead of overriding orientations with clean modified rig angles")
    parser.add_argument("--mask-dir", type=str, default=None, help="Directory of equirectangular mask images or mask video")
    parser.add_argument("--no-auto-mask", action="store_true", help="Disable auto-detection of mask folder in dataset")
    parser.add_argument("--masks-only", action="store_true", help="Only project and export .mask images for an existing dataset")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    out_w = args.width or args.size
    out_h = args.height or args.size
    out_dir = args.output or (f"{input_path}_pinhole_rig" if os.path.isdir(input_path) else f"{os.path.splitext(input_path)[0]}_pinhole_rig")

    print("=" * 70)
    print("   Abu Align - 360° Multi-Camera Rig Projector & Panoramic Splitter")
    print("=" * 70)
    print(f"[*] Input:             {input_path}")
    print(f"[*] Output Directory:  {out_dir}")
    print(f"[*] Rig Mode:          {args.mode.upper()} ({args.num_cameras} cameras if ring/fibonacci)")
    print(f"[*] Field of View:     {args.fov}°")
    print(f"[*] Resolution:        {out_w} x {out_h} px")
    if abs(args.distortion_k1) > 1e-6:
        print(f"[*] Distortion (k1):   {args.distortion_k1}")
    print(f"[*] Format & Quality:  {args.format.upper()} (quality {args.quality})")
    if args.dvo_trajectory:
        print(f"[*] 360DVO Trajectory: {args.dvo_trajectory}")
    if not args.no_xmp:
        print(f"[*] RealityScan Rig XMP: RigId='{args.rig_id}', PosePrior='{args.xmp_pose_prior}'")
    if args.mask_dir:
        print(f"[*] Mask Directory:    {args.mask_dir}")
    if args.masks_only:
        print(f"[*] Mode:              MASKS ONLY (generating .mask perspective layers)")

    res = split_panorama_dataset(
        input_dir=input_path,
        output_dir=out_dir,
        mode=args.mode,
        num_cameras=args.num_cameras,
        fov_deg=args.fov,
        crop_size=args.size,
        crop_width=out_w,
        crop_height=out_h,
        k1=args.distortion_k1,
        video_target_fps=args.video_fps,
        video_frame_stride=args.video_stride,
        image_format=args.format,
        quality=args.quality,
        export_transforms_json=not args.no_transforms and not args.masks_only,
        export_realityscan_csv=not args.no_rc_csv and not args.masks_only,
        export_xmp=not args.no_xmp and not args.masks_only,
        rig_id=args.rig_id,
        xmp_pose_prior=args.xmp_pose_prior,
        subfolder_per_perspective=not args.flat_images,
        dvo_trajectory_csv=args.dvo_trajectory,
        override_orientations_from_rig=not args.compound_rotations,
        mask_dir=args.mask_dir,
        auto_detect_mask=not args.no_auto_mask,
        masks_only=args.masks_only,
        progress_callback=lambda cur, tot, name: print(f"    Processing {cur}/{tot}: {name}", end="\r"),
    )
    print()
    print("=" * 70)
    print(f"[*] Rig projection complete!")
    print(f"    Source Panos/Frames: {res['total_panoramas']}")
    print(f"    Pinhole Views:       {res['total_views']}")
    if res.get("total_masks", 0) > 0:
        print(f"    Mask Layers (.mask): {res['total_masks']} exported to perspective folders")
    print(f"    Elapsed Time:        {res['elapsed_seconds']}s")
    print(f"    Images Folder:       {res['images_dir']}")
    if res.get("transforms_path"):
        print(f"    Transforms JSON:     {res['transforms_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
