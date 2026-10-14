"""
Unit tests for fast_viewer.py:
- resolve_project_structure upward directory traversal
- find_run_assets priority scoring & 3dgs_models rejection
- load_cameras_from_colmap_bin binary parser
- load_cameras_from_opensfm angle-axis rotation parser
- load_cameras_from_ace0 OpenCV w2c pose parser
- load_run_cameras unified dispatcher
"""
import os
import json
import math
import struct
import shutil
import tempfile
import unittest
import numpy as np

import fast_viewer


class TestFastViewer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_viewer_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_resolve_project_structure_upward_traversal(self):
        """Passing deep subpath inside a run (e.g. 3dgs_models/lichtfeld_models) must resolve results/."""
        dataset = os.path.join(self.test_dir, "My_Scan_Dataset")
        results = os.path.join(dataset, "results")
        run1 = os.path.join(results, "sparse_01_omega")
        run2 = os.path.join(results, "sparse_02_omega")
        deep_model = os.path.join(run2, "3dgs_models", "lichtfeld_models")
        os.makedirs(run1)
        os.makedirs(deep_model)

        # Target is deep subfolder
        title, ds, runs, init = fast_viewer.resolve_project_structure(deep_model)
        self.assertEqual(ds, os.path.abspath(dataset))
        self.assertIn("sparse_01_omega", runs)
        self.assertIn("sparse_02_omega", runs)
        self.assertEqual(init, "sparse_02_omega")

        # Target is dataset root
        title, ds, runs, init = fast_viewer.resolve_project_structure(dataset)
        self.assertEqual(ds, os.path.abspath(dataset))
        self.assertEqual(len(runs), 2)

    def test_find_run_assets_scoring_and_rejection(self):
        """Verifies point cloud scoring prioritizes points3D.ply and rejects 3dgs_models splats."""
        run_dir = os.path.join(self.test_dir, "sparse_03_vggt")
        models_dir = os.path.join(run_dir, "3dgs_models")
        os.makedirs(models_dir)

        # Fake splat in 3dgs_models
        fake_splat = os.path.join(models_dir, "splat.ply")
        with open(fake_splat, "wb") as f:
            f.write(b"ply\nformat ascii 1.0\nelement vertex 10\nend_header\n" + b"0 0 0\n"*10)

        # Real points3D.ply
        real_ply = os.path.join(run_dir, "points3D.ply")
        with open(real_ply, "wb") as f:
            f.write(b"ply\nformat ascii 1.0\nelement vertex 5\nend_header\n" + b"1 2 3\n"*5)

        best_ply, _, _, _, _, _, _ = fast_viewer.find_run_assets(run_dir)
        self.assertEqual(os.path.abspath(best_ply), os.path.abspath(real_ply))

    def test_colmap_bin_parser(self):
        """Verifies pure-Python COLMAP images.bin parser."""
        bin_path = os.path.join(self.test_dir, "images.bin")
        # Pack 1 registered image:
        # num_reg_images: uint64 (1)
        # img_id: uint32 (1)
        # qw, qx, qy, qz: 4 doubles (identity: 1.0, 0, 0, 0)
        # tx, ty, tz: 3 doubles (1.0, 2.0, 3.0)
        # cam_id: uint32 (1)
        # name: "test_frame.png\0"
        # num_points2D: uint64 (0)
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
            f.write(struct.pack("<3d", 1.0, 2.0, 3.0))
            f.write(struct.pack("<I", 1))
            f.write(b"test_frame.png\x00")
            f.write(struct.pack("<Q", 0))

        cams = fast_viewer.load_cameras_from_colmap_bin(bin_path)
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["name"], "test_frame.png")
        # In identity rotation, c2w position is -t_w2c = [-1.0, -2.0, -3.0]
        np.testing.assert_allclose(cams[0]["pos"], [-1.0, -2.0, -3.0], atol=1e-5)

    def test_opensfm_parser(self):
        """Verifies OpenSfM reconstruction.json parser."""
        json_path = os.path.join(self.test_dir, "reconstruction.json")
        data = [{
            "shots": {
                "cam01.jpg": {
                    "rotation": [0.0, 0.0, 0.0],
                    "translation": [2.0, -4.0, 6.0]
                }
            }
        }]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        cams = fast_viewer.load_cameras_from_opensfm(json_path)
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["name"], "cam01.jpg")
        np.testing.assert_allclose(cams[0]["pos"], [-2.0, 4.0, -6.0], atol=1e-5)

    def test_ace0_parser(self):
        """Verifies ACE0 poses_final.txt parser."""
        ace_path = os.path.join(self.test_dir, "poses_final.txt")
        # format: filename qw qx qy qz x y z focal confidence
        with open(ace_path, "w", encoding="utf-8") as f:
            f.write("img_001.png 1.0 0.0 0.0 0.0 0.5 1.5 -2.5 800.0 1.0\n")

        cams = fast_viewer.load_cameras_from_ace0(ace_path)
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["name"], "img_001.png")
        np.testing.assert_allclose(cams[0]["pos"], [-0.5, -1.5, 2.5], atol=1e-5)


    def test_colmap_images_txt_parser(self):
        """Verifies COLMAP images.txt parser properly skips alternating 2D point measurement lines."""
        txt_path = os.path.join(self.test_dir, "images.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            # Image 1
            f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame000001.png\n")
            # 2D points for image 1 (must be skipped, NOT parsed as camera)
            f.write("100.5 200.5 1 300.2 400.1 2 500.0 600.0 3 700.0 800.0 4\n")
            # Image 2
            f.write("2 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 frame000002.png\n")
            # 2D points for image 2
            f.write("150.0 250.0 1\n")

        cams = fast_viewer.load_cameras_from_colmap_images(txt_path)
        self.assertEqual(len(cams), 2)
        self.assertEqual(cams[0]["name"], "frame000001.png")
        self.assertEqual(cams[1]["name"], "frame000002.png")
        np.testing.assert_allclose(cams[0]["pos"], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(cams[1]["pos"], [-1.0, -2.0, -3.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
