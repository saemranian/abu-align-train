"""Shared native panorama geometry and lossless OpenSfM observation adapters."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_xyzw(matrix):
    """Stable for identity and 180-degree rotations; PyCOLMAP uses scalar-last."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("Rotation must belong to SO(3)")
    return Rotation.from_matrix(matrix).as_quat()


def load_observations(reconstruction_path, component):
    """Return (shot, landmark, pixel_xy), including upstream v1/v2 tracks.csv.

    OpenSfM stores centered coordinates normalized by the largest image dimension.
    COLMAP pixel centers use half-pixel coordinates, so x*max_dimension + width/2
    converts directly without an additional half-pixel shift.
    """
    observations = []
    for point_id, point in component.get("points", {}).items():
        for obs in point.get("observations", []):
            if obs["shot_id"] in component["shots"]:
                observations.append((obs["shot_id"], str(point_id), np.asarray(obs["pixel"], dtype=float)))
    path = Path(reconstruction_path).parent / "tracks.csv"
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream, delimiter="\t"):
                if len(row) < 5 or row[0] not in component.get("shots", {}) or row[1] not in component.get("points", {}):
                    continue
                camera = component["cameras"][component["shots"][row[0]]["camera"]]
                w, h = camera["width"], camera["height"]
                xy = np.array([float(row[3]) * max(w, h) + w / 2,
                               float(row[4]) * max(w, h) + h / 2])
                observations.append((row[0], row[1], xy))
    seen = set()
    result = []
    for shot, point, xy in observations:
        key = (shot, point)
        if key in seen or xy.shape != (2,) or not np.isfinite(xy).all():
            continue
        seen.add(key)
        result.append((shot, point, xy))
    return result


def validate_gaussian_ply(path):
    """Reject empty clouds and ordinary point PLY files masquerading as splats."""
    from plyfile import PlyData
    data = PlyData.read(str(path), mmap=False)
    vertices = data["vertex"].data
    required = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"}
    required.update(f"scale_{i}" for i in range(3))
    required.update(f"rot_{i}" for i in range(4))
    if not len(vertices) or not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"Not a nonempty Gaussian splat: {path}")
    if any(not np.isfinite(vertices[name]).all() for name in vertices.dtype.names):
        raise ValueError(f"Nonfinite Gaussian parameters in {path}")
    return len(vertices)
