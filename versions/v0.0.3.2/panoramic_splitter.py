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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Splits an entire directory of 360° panoramas OR a 360° video file into rectilinear pinhole views
    and exports a transforms.json manifest ready for COLMAP, Nerfstudio, PFGS360, or LichtFeld.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for panoramic rig projection.")

    start_time = time.time()
    in_abs = str(Path(input_dir).resolve())
    out_abs = str(Path(output_dir).resolve())
    images_out = os.path.join(out_abs, "images")
    os.makedirs(images_out, exist_ok=True)

    out_w = crop_width if crop_width is not None else crop_size
    out_h = crop_height if crop_height is not None else crop_size

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

    video_exts = {".mp4", ".mov", ".insv", ".mkv", ".avi"}
    is_video_input = os.path.isfile(in_abs) and (os.path.splitext(in_abs.lower())[1] in video_exts)

    fmt = "png" if image_format.lower() == "png" else "jpg"
    ext = f".{fmt}"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    total_views_saved = 0
    transforms_frames = []
    total_source_items = 0

    if is_video_input:
        # -------------------------------------------------------------
        # Direct Video Extraction & Projection Loop
        # -------------------------------------------------------------
        cap = cv2.VideoCapture(in_abs)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {in_abs}")

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

                crops = splitter.split_panorama(frame)
                for cam_name, crop_img in crops.items():
                    out_filename = f"{base_name}_{cam_name}{ext}"
                    out_filepath = os.path.join(images_out, out_filename)
                    cv2.imwrite(out_filepath, crop_img, encode_params)
                    total_views_saved += 1

                    if export_transforms_json:
                        c2w = splitter.get_c2w_matrix(cam_name)
                        transforms_frames.append({
                            "file_path": f"images/{out_filename}",
                            "transform_matrix": c2w.tolist(),
                            "camera_angle_name": cam_name,
                            "source_frame": saved_frame_idx,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "yaw_deg": splitter.intrinsics[cam_name]["yaw_deg"],
                            "pitch_deg": splitter.intrinsics[cam_name]["pitch_deg"],
                        })

                if progress_callback:
                    progress_callback(saved_frame_idx, expected_frames, f"{base_name} (t={timestamp_sec:.2f}s)")

                if video_max_frames > 0 and saved_frame_idx >= video_max_frames:
                    break

            frame_idx += 1

        cap.release()
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
            raise FileNotFoundError(f"No valid panorama images found in {input_dir}")

        total_source_items = len(files)

        # Read first sample to initialize splitter
        sample_path = os.path.join(scan_dir, files[0])
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
            img = cv2.imread(img_path)
            if img is None:
                continue

            base_name = Path(fname).stem
            crops = splitter.split_panorama(img)

            for cam_name, crop_img in crops.items():
                out_filename = f"{base_name}_{cam_name}{ext}"
                out_filepath = os.path.join(images_out, out_filename)
                cv2.imwrite(out_filepath, crop_img, encode_params)
                total_views_saved += 1

                if export_transforms_json:
                    c2w = splitter.get_c2w_matrix(cam_name)
                    transforms_frames.append({
                        "file_path": f"images/{out_filename}",
                        "transform_matrix": c2w.tolist(),
                        "camera_angle_name": cam_name,
                        "source_file": fname,
                        "yaw_deg": splitter.intrinsics[cam_name]["yaw_deg"],
                        "pitch_deg": splitter.intrinsics[cam_name]["pitch_deg"],
                    })

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

    elapsed = time.time() - start_time

    # 3. Export rig_manifest.json with Full Telemetry
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
        "output_dir": out_abs,
        "images_dir": images_out,
        "transforms_path": transforms_path if export_transforms_json else None,
        "elapsed_seconds": round(elapsed, 2),
    }


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
        export_transforms_json=not args.no_transforms,
        progress_callback=lambda cur, tot, name: print(f"    Processing {cur}/{tot}: {name}", end="\r"),
    )
    print()
    print("=" * 70)
    print(f"[*] Rig projection complete!")
    print(f"    Source Panos/Frames: {res['total_panoramas']}")
    print(f"    Pinhole Views:       {res['total_views']}")
    print(f"    Elapsed Time:        {res['elapsed_seconds']}s")
    print(f"    Images Folder:       {res['images_dir']}")
    if res.get("transforms_path"):
        print(f"    Transforms JSON:     {res['transforms_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
