import os
import sys
import unittest
import numpy as np
import tempfile
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dvo360_engine import (
    tum_trajectory_to_colmap,
    tum_trajectory_to_transforms_json,
    tum_trajectory_to_opensfm,
    run_360dvo_reconstruction
)


class Test360DVOEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.traj_file = os.path.join(self.test_dir, "test_traj.txt")
        with open(self.traj_file, "w", encoding="utf-8") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.write("1.0 0.1 0.0 0.0 0.0 0.0 0.0 1.0\n")
        self.image_names = ["img_001.jpg", "img_002.jpg"]

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tum_trajectory_to_colmap(self):
        sparse_dir = os.path.join(self.test_dir, "sparse", "0")
        dense_3d = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        dense_rgb = np.array([[255, 0, 0]], dtype=np.uint8)
        success, msg = tum_trajectory_to_colmap(
            self.traj_file,
            self.image_names,
            sparse_dir,
            width=7680,
            height=3840,
            points_xyz=dense_3d,
            points_rgb=dense_rgb
        )
        self.assertTrue(success)
        self.assertTrue(os.path.isfile(os.path.join(sparse_dir, "cameras.txt")))
        self.assertTrue(os.path.isfile(os.path.join(sparse_dir, "images.txt")))
        self.assertTrue(os.path.isfile(os.path.join(sparse_dir, "points3D.txt")))

        # Check cameras.txt contains full 8K resolution
        with open(os.path.join(sparse_dir, "cameras.txt"), "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("7680 3840", content)

    def test_tum_trajectory_to_transforms_json(self):
        tf_file = os.path.join(self.test_dir, "transforms.json")
        success, msg = tum_trajectory_to_transforms_json(
            self.traj_file,
            self.image_names,
            tf_file,
            width=7680,
            height=3840
        )
        self.assertTrue(success)
        self.assertTrue(os.path.isfile(tf_file))
        with open(tf_file, "r", encoding="utf-8") as f:
            import json
            data = json.load(f)
            self.assertEqual(data["camera_model"], "EQUIRECTANGULAR")
            self.assertEqual(data["w"], 7680)
            self.assertEqual(data["h"], 3840)
            self.assertEqual(len(data["frames"]), 2)

    def test_run_360dvo_signature_and_fallback(self):
        import inspect
        sig = inspect.signature(run_360dvo_reconstruction)
        self.assertIn("load_res", sig.parameters)
        self.assertEqual(sig.parameters["load_res"].default, 1024)
        self.assertIn("use_cache", sig.parameters)
        self.assertTrue(sig.parameters["use_cache"].default)
        self.assertIn("clear_cache", sig.parameters)
        self.assertFalse(sig.parameters["clear_cache"].default)
        self.assertIn("cache_dir", sig.parameters)


if __name__ == "__main__":
    unittest.main()
