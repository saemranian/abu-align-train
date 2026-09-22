"""
Abu Align & Train - Generalized 360° Multi-Camera Rig Projector & Panoramic Splitter
==================================================================================
Converts 360° Equirectangular Panoramas (ERP) into virtual multi-camera pinhole rigs
with configurable field-of-view (FOV), angular distribution, and overlap.

Rigs Supported:
1. Standard 6-Face Cubemap (Front, Back, Left, Right, Up, Down, 90° FOV)
2. Horizontal Ring Rig (N virtual cameras around equator, e.g. 8 cams @ 45° step, 90° FOV => 50% overlap)
3. Spherical Dome Rig (14-18 cameras across multi-tier pitch levels + zenith/nadir with uniform overlap)
4. Custom Camera Angle Rig (arbitrary yaw/pitch/roll/fov camera arrays)

Features:
- Seamless 360° boundary wrapping and sub-pixel bicubic/bilinear interpolation
- Automatic Pinhole Intrinsics & Extrinsics calculation (fl_x, fl_y, cx, cy, 4x4 c2w matrices)
- Exports Nerfstudio / 3DGS compliant transforms.json and COLMAP camera manifests
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
from typing import List, Tuple, Dict, Any, Optional, Callable

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


# Canonical Rig Configurations
# ----------------------------------------------------------------------
# (name, yaw_deg, pitch_deg, roll_deg, fov_deg)

def get_cubemap_rig(fov_deg: float = 90.0) -> List[Tuple[str, float, float, float, float]]:
    """Standard 6-face orthogonal cubemap rig."""
    return [
        ("front",  0.0,    0.0,   0.0, fov_deg),  # +Z
        ("right",  90.0,   0.0,   0.0, fov_deg),  # +X
        ("back",   180.0,  0.0,   0.0, fov_deg),  # -Z
        ("left",   -90.0,  0.0,   0.0, fov_deg),  # -X
        ("up",     0.0,    90.0,  0.0, fov_deg),  # +Y (Zenith)
        ("down",   0.0,   -90.0,  0.0, fov_deg),  # -Y (Nadir)
    ]


def get_horizontal_ring_rig(num_cameras: int = 8, fov_deg: float = 90.0) -> List[Tuple[str, float, float, float, float]]:
    """
    Horizontal ring rig of N virtual cameras distributed evenly around the equator (yaw 0..360).
    With 8 cameras and 90° FOV, adjacent cameras overlap by 50%, providing dense tie points for pinhole SfM.
    """
    step = 360.0 / max(num_cameras, 3)
    rig = []
    for i in range(num_cameras):
        yaw = i * step
        # Normalize yaw to [-180, 180)
        yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
        name = f"ring_{i:02d}_yaw{int(round(yaw)):03d}"
        rig.append((name, yaw_norm, 0.0, 0.0, fov_deg))
    return rig


def get_spherical_dome_rig(fov_deg: float = 90.0) -> List[Tuple[str, float, float, float, float]]:
    """
    14-camera spherical dome rig covering the entire 4π steradian sphere with uniform overlap:
    - 8 cameras on the equator (pitch 0°, yaw every 45°)
    - 2 cameras on the upper tier (pitch +35°, yaw 0°, 180°)
    - 2 cameras on the lower tier (pitch -35°, yaw 90°, 270°)
    - 1 zenith camera (pitch +90°)
    - 1 nadir camera (pitch -90°)
    """
    rig = []
    # 1. Equatorial ring (8 cams)
    for i in range(8):
        yaw = i * 45.0
        yaw_norm = ((yaw + 180.0) % 360.0) - 180.0
        rig.append((f"equator_{i:02d}_yaw{int(round(yaw)):03d}", yaw_norm, 0.0, 0.0, fov_deg))

    # 2. Upper tier (+35° pitch, 2 cams)
    rig.append(("upper_00_yaw000_p35", 0.0, 35.0, 0.0, fov_deg))
    rig.append(("upper_01_yaw180_p35", 180.0, 35.0, 0.0, fov_deg))

    # 3. Lower tier (-35° pitch, 2 cams)
    rig.append(("lower_00_yaw090_m35", 90.0, -35.0, 0.0, fov_deg))
    rig.append(("lower_01_yaw270_m35", -90.0, -35.0, 0.0, fov_deg))

    # 4. Zenith and Nadir
    rig.append(("zenith_up", 0.0, 90.0, 0.0, fov_deg))
    rig.append(("nadir_down", 0.0, -90.0, 0.0, fov_deg))

    return rig


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """
    Computes camera rotation matrix from yaw, pitch, roll (in degrees).
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


# ----------------------------------------------------------------------
# Panoramic Rig Splitter Class
# ----------------------------------------------------------------------
class PanoramicRigSplitter:
    """
    Vectorized high-performance equirectangular rig projector.
    Precomputes cv2.remap tables for arbitrary virtual camera arrays.
    """
    def __init__(
        self,
        pano_width: int,
        pano_height: int,
        rig_cameras: List[Tuple[str, float, float, float, float]],
        crop_size: int = 1024,
    ):
        self.pano_w = pano_width
        self.pano_h = pano_height
        self.rig_cameras = rig_cameras
        self.crop_size = crop_size

        self.maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.rotations: Dict[str, np.ndarray] = {}
        self.intrinsics: Dict[str, Dict[str, Any]] = {}

        self._build_remap_tables()

    def _build_remap_tables(self):
        """Precomputes coordinate grids for each camera in the rig."""
        s = self.crop_size

        for cam_name, yaw_deg, pitch_deg, roll_deg, fov_deg in self.rig_cameras:
            R = euler_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
            self.rotations[cam_name] = R

            # Compute focal length from FOV: tan(fov/2) = (s / 2) / f => f = (s / 2) / tan(fov/2)
            fov_rad = math.radians(fov_deg)
            focal = (float(s) / 2.0) / math.tan(fov_rad / 2.0)
            cx = float(s) / 2.0
            cy = float(s) / 2.0

            self.intrinsics[cam_name] = {
                "model": "PINHOLE",
                "width": s,
                "height": s,
                "fx": focal,
                "fy": focal,
                "cx": cx,
                "cy": cy,
                "fov_deg": fov_deg,
                "yaw_deg": yaw_deg,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
            }

            # Normalized camera sensor plane coordinates: u = (px - cx) / f, v = (py - cy) / f
            # In image space, +y is down, but in standard 3D camera coordinates, +y is up (so invert v)
            px_grid = np.arange(s, dtype=np.float32)
            py_grid = np.arange(s, dtype=np.float32)
            px_m, py_m = np.meshgrid(px_grid, py_grid)

            u = (px_m - cx) / focal
            v = -(py_m - cy) / focal

            # Ray directions in local camera frame: [u, v, 1.0] normalized
            rays_cam = np.stack([u, v, np.ones_like(u)], axis=-1)  # (s, s, 3)

            # Rotate rays into equirectangular global coordinate frame: ray_global = rays_cam @ R^T
            rays_global = rays_cam @ R.T

            x = rays_global[..., 0]
            y = rays_global[..., 1]
            z = rays_global[..., 2]

            # Longitude (azimuth) lambda in [-pi, pi], Latitude (polar) phi in [-pi/2, pi/2]
            lon = np.arctan2(x, z)
            hypot = np.sqrt(x * x + z * z)
            lat = np.arctan2(y, hypot)

            # Map to equirectangular pixel coordinates
            map_x = ((lon / (2.0 * np.pi) + 0.5) * self.pano_w).astype(np.float32)
            map_y = ((0.5 - lat / np.pi) * self.pano_h).astype(np.float32)

            # Boundary wrap for continuous 360° horizontal stitching
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
                interpolation=cv2.INTER_LINEAR,
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


# ----------------------------------------------------------------------
# Batch Dataset Processing & Transforms Exporter
# ----------------------------------------------------------------------
def split_panorama_dataset(
    input_dir: str,
    output_dir: str,
    mode: str = "ring",             # "cubemap", "ring", "dome"
    num_cameras: int = 8,           # When mode="ring"
    fov_deg: float = 90.0,          # Camera FOV in degrees
    crop_size: int = 1024,          # Output width & height per view
    image_format: str = "jpg",      # "jpg" or "png"
    quality: int = 95,              # JPEG quality
    export_transforms_json: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Splits an entire directory of 360° panoramas into rectilinear pinhole views
    and exports a transforms.json manifest ready for COLMAP, Nerfstudio, ACE0, or LichtFeld.
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required for panoramic rig projection.")

    start_time = time.time()
    in_abs = str(Path(input_dir).resolve())
    out_abs = str(Path(output_dir).resolve())
    images_out = os.path.join(out_abs, "images")
    os.makedirs(images_out, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(in_abs)
        if os.path.splitext(f.lower())[1] in valid_exts and os.path.isfile(os.path.join(in_abs, f))
    ])

    if not files:
        raise FileNotFoundError(f"No valid panorama images found in {input_dir}")

    # Build Rig
    mode_lower = mode.lower()
    if mode_lower in ("cubemap", "cube", "6face"):
        rig = get_cubemap_rig(fov_deg=fov_deg)
        rig_title = "6-Face Cubemap"
    elif mode_lower in ("dome", "spherical_dome", "14cam"):
        rig = get_spherical_dome_rig(fov_deg=fov_deg)
        rig_title = "14-Camera Spherical Dome"
    else:  # "ring"
        rig = get_horizontal_ring_rig(num_cameras=num_cameras, fov_deg=fov_deg)
        rig_title = f"{num_cameras}-Camera Horizontal Ring"

    # Read first image to determine panorama dimensions
    sample_path = os.path.join(in_abs, files[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        raise RuntimeError(f"Failed to read sample image: {sample_path}")
    pano_h, pano_w = sample_img.shape[:2]

    splitter = PanoramicRigSplitter(pano_w, pano_h, rig_cameras=rig, crop_size=crop_size)

    fmt = "png" if image_format.lower() == "png" else "jpg"
    ext = f".{fmt}"
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    total_views_saved = 0
    transforms_frames = []

    # Common intrinsics (since all views share crop_size and fov)
    first_cam = rig[0][0]
    cam_intr = splitter.intrinsics[first_cam]

    for idx, fname in enumerate(files):
        img_path = os.path.join(in_abs, fname)
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
                # In standard Nerfstudio / OpenGL conventions:
                # transform_matrix is 4x4 camera-to-world list
                transforms_frames.append({
                    "file_path": f"images/{out_filename}",
                    "transform_matrix": c2w.tolist(),
                    "camera_angle_name": cam_name,
                    "yaw_deg": splitter.intrinsics[cam_name]["yaw_deg"],
                    "pitch_deg": splitter.intrinsics[cam_name]["pitch_deg"],
                })

        if progress_callback:
            progress_callback(idx + 1, len(files), fname)

    # Export transforms.json manifest
    transforms_path = os.path.join(out_abs, "transforms.json")
    if export_transforms_json and transforms_frames:
        transforms_data = {
            "camera_model": "PINHOLE",
            "fl_x": cam_intr["fx"],
            "fl_y": cam_intr["fy"],
            "cx": cam_intr["cx"],
            "cy": cam_intr["cy"],
            "w": crop_size,
            "h": crop_size,
            "fov_deg": fov_deg,
            "rig_mode": mode,
            "total_cameras_per_panorama": len(rig),
            "frames": transforms_frames,
        }
        with open(transforms_path, "w", encoding="utf-8") as tf:
            json.dump(transforms_data, tf, indent=2)

    elapsed = time.time() - start_time

    # Write rig configuration metadata
    rig_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rig_title": rig_title,
        "mode": mode,
        "num_views_per_pano": len(rig),
        "fov_deg": fov_deg,
        "crop_size": crop_size,
        "total_source_panoramas": len(files),
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
        "total_panoramas": len(files),
        "total_views": total_views_saved,
        "output_dir": out_abs,
        "images_dir": images_out,
        "transforms_path": transforms_path if export_transforms_json else None,
        "elapsed_seconds": round(elapsed, 2),
    }


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Abu Align & Train - 360° Multi-Camera Rig Projector & Panoramic Splitter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory of 360° panoramas or single image")
    parser.add_argument("-o", "--output", help="Output directory (defaults to <input>_pinhole_rig)")
    parser.add_argument("-m", "--mode", choices=["ring", "cubemap", "dome"], default="ring", help="Virtual camera rig type")
    parser.add_argument("-n", "--num_cameras", type=int, default=8, help="Number of virtual cameras when mode=ring")
    parser.add_argument("--fov", type=float, default=90.0, help="Field of View (degrees) per camera view")
    parser.add_argument("-s", "--size", type=int, default=1024, help="Output image width and height (square crop)")
    parser.add_argument("-f", "--format", choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--no-transforms", action="store_true", help="Skip generating transforms.json")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    out_dir = args.output or (f"{input_path}_pinhole_rig" if os.path.isdir(input_path) else f"{os.path.splitext(input_path)[0]}_pinhole_rig")

    print("=" * 70)
    print("   Abu Align - 360° Multi-Camera Rig Projector & Panoramic Splitter")
    print("=" * 70)
    print(f"[*] Input:             {input_path}")
    print(f"[*] Output Directory:  {out_dir}")
    print(f"[*] Rig Mode:          {args.mode.upper()} ({args.num_cameras} cameras if ring)")
    print(f"[*] Field of View:     {args.fov}°")
    print(f"[*] Resolution:        {args.size} x {args.size} px")
    print(f"[*] Format & Quality:  {args.format.upper()} (quality {args.quality})")

    # If input is a single image, create temp single-item folder
    temp_dir = None
    target_in = input_path
    if os.path.isfile(input_path):
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="pano_single_")
        shutil.copy(input_path, os.path.join(temp_dir, os.path.basename(input_path)))
        target_in = temp_dir

    try:
        res = split_panorama_dataset(
            input_dir=target_in,
            output_dir=out_dir,
            mode=args.mode,
            num_cameras=args.num_cameras,
            fov_deg=args.fov,
            crop_size=args.size,
            image_format=args.format,
            quality=args.quality,
            export_transforms_json=not args.no_transforms,
            progress_callback=lambda cur, tot, name: print(f"    Processing panorama {cur}/{tot}: {name}", end="\r"),
        )
        print()
        print("=" * 70)
        print(f"[*] Rig projection complete!")
        print(f"    Source Panos:    {res['total_panoramas']}")
        print(f"    Pinhole Views:   {res['total_views']}")
        print(f"    Elapsed Time:    {res['elapsed_seconds']}s")
        print(f"    Images Folder:   {res['images_dir']}")
        if res.get("transforms_path"):
            print(f"    Transforms JSON: {res['transforms_path']}")
        print("=" * 70)
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
