"""
Abu Align & Train - 360° Equirectangular to Pinhole Cubemap Transcoder
High-speed vectorized conversion of 360° spherical panoramas into 6 rectilinear 90° pinhole views
Enables VGGT-Omega, ACE0, PyCuSFM, and 3DGS engines (LichtFeld Studio / Splatfacto)
to process 360° cameras with zero optical curvature distortion.
"""

import os
import math
import numpy as np
import cv2
from typing import List, Tuple, Dict

# Face names and their canonical rotation angles (Yaw, Pitch, Roll in degrees)
CUBEMAP_FACES = [
    ("front",  0.0,    0.0),    # +Z
    ("right",  90.0,   0.0),    # +X
    ("back",   180.0,  0.0),    # -Z
    ("left",   -90.0,  0.0),    # -X
    ("up",     0.0,    90.0),   # +Y (Zenith)
    ("down",   0.0,   -90.0),   # -Y (Nadir)
]

def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Computes camera rotation matrix from yaw and pitch (in degrees).
    Convention: Y-up, Z-forward, X-right.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Rotation around Y (yaw)
    Ry = np.array([
        [ math.cos(yaw), 0.0, math.sin(yaw)],
        [ 0.0,           1.0, 0.0          ],
        [-math.sin(yaw), 0.0, math.cos(yaw)]
    ], dtype=np.float64)

    # Rotation around X (pitch)
    Rx = np.array([
        [1.0, 0.0,            0.0          ],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch),  math.cos(pitch)]
    ], dtype=np.float64)

    return Ry @ Rx

class EquirectangularCubemapTranscoder:
    """
    Precomputed remap tables for lightning-fast transformation of 360° panoramas
    into 6 distortion-free rectilinear pinhole faces.
    """
    def __init__(self, pano_width: int, pano_height: int, face_size: int = 1024):
        self.pano_w = pano_width
        self.pano_h = pano_height
        self.face_size = face_size
        self.maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.rotations: Dict[str, np.ndarray] = {}

        self._build_remap_tables()

    def _build_remap_tables(self):
        """Precomputes cv2.remap coordinate grids (map_x, map_y) for all 6 faces."""
        s = self.face_size
        # Normalized grid from -1.0 to 1.0
        grid = np.linspace(-1.0 + 1.0/s, 1.0 - 1.0/s, s, dtype=np.float32)
        u, v = np.meshgrid(grid, -grid)  # invert v for standard image coordinate system

        for face_name, yaw_deg, pitch_deg in CUBEMAP_FACES:
            R = euler_to_rotation_matrix(yaw_deg, pitch_deg)
            self.rotations[face_name] = R

            # Pinhole rays with 90° FoV (focal length = s / 2 => z = 1.0)
            # Ray direction in face coordinate frame: (u, v, 1.0)
            rays_face = np.stack([u, v, np.ones_like(u)], axis=-1)  # (s, s, 3)
            # Rotate ray into global equirectangular frame
            rays_global = rays_face @ R.T  # (s, s, 3)

            x = rays_global[..., 0]
            y = rays_global[..., 1]
            z = rays_global[..., 2]

            # Convert 3D ray to spherical longitude (lambda) and latitude (phi)
            # Longitude: -pi to +pi
            lon = np.arctan2(x, z)
            # Latitude: -pi/2 to +pi/2
            hypot = np.sqrt(x * x + z * z)
            lat = np.arctan2(y, hypot)

            # Map to equirectangular pixel coordinates
            map_x = ((lon / (2.0 * np.pi) + 0.5) * self.pano_w).astype(np.float32)
            map_y = ((0.5 - lat / np.pi) * self.pano_h).astype(np.float32)

            # Boundary wrap for 360 continuity
            map_x = np.mod(map_x, self.pano_w)
            map_y = np.clip(map_y, 0, self.pano_h - 1)

            self.maps[face_name] = (map_x, map_y)

    def transcode_image(self, pano_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Transforms a single 360° equirectangular image into 6 pinhole faces.
        Returns a dictionary mapping face_name -> face_bgr_image.
        """
        h, w = pano_bgr.shape[:2]
        if w != self.pano_w or h != self.pano_h:
            # Recompute tables if resolution differs
            self.pano_w = w
            self.pano_h = h
            self._build_remap_tables()

        faces = {}
        for face_name in self.maps:
            map_x, map_y = self.maps[face_name]
            face_img = cv2.remap(
                pano_bgr, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
            faces[face_name] = face_img
        return faces

    def get_pinhole_intrinsics(self) -> Dict[str, float]:
        """Returns standard COLMAP/OpenCV pinhole parameters for the generated faces."""
        focal = float(self.face_size) / 2.0
        cx = float(self.face_size) / 2.0
        cy = float(self.face_size) / 2.0
        return {
            "model": "PINHOLE",
            "width": self.face_size,
            "height": self.face_size,
            "fx": focal,
            "fy": focal,
            "cx": cx,
            "cy": cy,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0
        }


def transcode_dataset(
    input_dir: str,
    output_dir: str,
    face_size: int = 1024,
    selected_faces: List[str] = None,
    progress_callback = None
) -> Tuple[int, str]:
    """
    Batch transcode an entire directory of 360° equirectangular images into
    rectilinear pinhole cubemaps ready for photogrammetry alignment and 3DGS training.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ])

    if not files:
        return 0, "No image files found in input directory."

    # Read first image to determine panorama dimensions
    sample_path = os.path.join(input_dir, files[0])
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        return 0, f"Failed to load sample image: {sample_path}"

    pano_h, pano_w = sample_img.shape[:2]
    transcoder = EquirectangularCubemapTranscoder(pano_w, pano_h, face_size=face_size)

    faces_to_save = selected_faces or [f[0] for f in CUBEMAP_FACES]
    total_saved = 0

    for idx, fname in enumerate(files):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        base_name = os.path.splitext(fname)[0]
        faces = transcoder.transcode_image(img)

        for face_name in faces_to_save:
            if face_name in faces:
                out_name = f"{base_name}_{face_name}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, faces[face_name], [cv2.IMWRITE_JPEG_QUALITY, 95])
                total_saved += 1

        if progress_callback:
            progress_callback(idx + 1, len(files), fname)

    return total_saved, f"Successfully transcoded {len(files)} panoramas into {total_saved} rectilinear pinhole images."
