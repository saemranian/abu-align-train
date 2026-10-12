"""Regression tests for pose convention, observation retention and splat exports."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from scipy.spatial.transform import Rotation

from opensfm_engine import opensfm_to_pycolmap
from dvo360_engine import tum_trajectory_to_colmap
from train_engine import export_pfgs360_checkpoint_to_ply
from spherical_geometry import validate_gaussian_ply


class TestSphericalIntegrity(unittest.TestCase):
    def test_opensfm_pose_and_tracks_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction.json"
            rotation = Rotation.from_rotvec([0, np.pi, 0]).as_matrix()
            center = np.array([2., 3., 4.])
            component = {"cameras": {"pano": {"projection_type": "spherical", "width": 1024, "height": 512}},
                         "shots": {"a.jpg": {"camera": "pano", "rotation": [0, np.pi, 0], "translation": (-rotation @ center).tolist()}},
                         "points": {"12": {"coordinates": [1, 2, 3], "color": [5, 6, 7]}}}
            path.write_text(json.dumps([component]), encoding="utf-8")
            (path.parent / "tracks.csv").write_text("OPENSFM_TRACKS_VERSION_v2\na.jpg\t12\t99\t0.1\t-0.1\t1\t5\t6\t7\t-1\t-1\n", encoding="utf-8")
            reconstruction = opensfm_to_pycolmap(str(path))
            image = reconstruction.images[1]
            np.testing.assert_allclose(image.cam_from_world().rotation.matrix(), rotation, atol=1e-8)
            np.testing.assert_allclose(image.projection_center(), center, atol=1e-8)
            point = next(iter(reconstruction.points3D.values()))
            self.assertEqual(len(point.track.elements), 1)
            np.testing.assert_allclose(image.points2D[0].xy, [614.4, 153.6])
            self.assertEqual(image.points2D[0].point3D_id, next(iter(reconstruction.points3D)))

    def test_tum_half_turn_survives_colmap_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.txt"
            trajectory.write_text("0 2 3 4 0 1 0 0\n", encoding="utf-8")
            tum_trajectory_to_colmap(str(trajectory), ["a.jpg"], str(root / "model"))
            fields = (root / "model" / "images.txt").read_text().splitlines()[3].split()
            self.assertAlmostEqual(abs(float(fields[2])), 0)
            self.assertAlmostEqual(abs(float(fields[3])), 1)

    def test_one_invalid_gaussian_does_not_remove_valid_rows(self):
        from plyfile import PlyData
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = {"means": torch.tensor([[1., 2., 3.], [float("nan"), 0., 0.]]),
                      "opacities": torch.ones(2, 1), "scales": torch.zeros(2, 3),
                      "quats": torch.tensor([[1., 0., 0., 0.]] * 2),
                      "features_dc": torch.ones(2, 3), "features_rest": torch.zeros(2, 15, 3)}
            torch.save({"pipeline": {"_model.gauss_params." + k: v for k, v in params.items()}}, root / "input.ckpt")
            export_pfgs360_checkpoint_to_ply(str(root / "input.ckpt"), str(root / "splat.ply"))
            self.assertEqual(validate_gaussian_ply(root / "splat.ply"), 1)
            vertices = PlyData.read(root / "splat.ply", mmap=False)["vertex"].data
            self.assertEqual(vertices["x"].tolist(), [1.])


if __name__ == "__main__":
    unittest.main()
