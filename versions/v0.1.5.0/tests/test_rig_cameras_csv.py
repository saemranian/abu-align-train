"""
Comprehensive unit tests for RealityScan / RealityCapture standard cameras.csv export,
360DVO trajectory prior ingestion, and 3D pose compounding in 360 Rig Designer.
"""
import os
import sys
import math
import shutil
import tempfile
import unittest
import numpy as np

import panoramic_splitter as ps


class TestRealityCaptureMath(unittest.TestCase):
    """Verifies mathematical correctness of RealityCapture world frame and Euler angles."""

    def test_rc_csv_header(self):
        expected = "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"
        self.assertEqual(ps.RC_CSV_HEADER, expected)

    def test_basis_orthonormality(self):
        for yaw in [0.0, 45.0, 90.0, 180.0, 270.0]:
            for pitch in [-90.0, -35.0, 0.0, 35.0, 90.0]:
                for roll in [-30.0, 0.0, 25.0]:
                    r, u, f = ps.basis(yaw, pitch, roll)
                    self.assertAlmostEqual(np.linalg.norm(r), 1.0, places=6)
                    self.assertAlmostEqual(np.linalg.norm(u), 1.0, places=6)
                    self.assertAlmostEqual(np.linalg.norm(f), 1.0, places=6)
                    self.assertAlmostEqual(float(np.dot(r, u)), 0.0, places=6)
                    self.assertAlmostEqual(float(np.dot(r, f)), 0.0, places=6)
                    self.assertAlmostEqual(float(np.dot(u, f)), 0.0, places=6)

    def test_euler_matrix_roundtrip(self):
        """Tests that (yaw, pitch, roll) -> matrix -> (yaw, pitch, roll) preserves the rotation matrix."""
        for y_in in [0.0, 30.0, 90.0, 135.0, 220.0, 315.0]:
            for p_in in [-45.0, 0.0, 45.0, 85.0]:  # non-gimbal lock pitch
                for r_in in [-25.0, 0.0, 15.0]:
                    M = ps.euler_zyx_to_matrix(y_in, p_in, r_in)
                    y_out, p_out, r_out = ps.matrix_to_euler_zyx(M)
                    M_reconstructed = ps.euler_zyx_to_matrix(y_out, p_out, r_out)
                    diff = np.max(np.abs(M - M_reconstructed))
                    self.assertLess(diff, 1e-6, f"Matrix reconstruction failed for ({y_in}, {p_in}, {r_in})")

    def test_cubemap_perpendicularity(self):
        """Verifies that all 6 cubemap faces compounded from an arbitrary panorama pose are exactly orthogonal."""
        pano_pose = {"x": 10.0, "y": 20.0, "alt": 5.0, "yaw": 35.0, "pitch": 80.0, "roll": -12.0}
        rig = ps.get_cubemap_rig(fov_deg=90.0)

        poses = {}
        for cam in rig:
            comp = ps.compound_realitycapture_pose(pano_pose, cam, 1024, 1024)
            M = ps.euler_zyx_to_matrix(comp["yaw"], comp["pitch"], comp["roll"])
            poses[cam[0]] = M[:, 2]  # optical axis (Forward)

        # 90° pairs
        pairs_90 = [("front", "right"), ("front", "left"), ("front", "up"), ("front", "down"),
                    ("back", "right"), ("back", "left"), ("back", "up"), ("back", "down"),
                    ("right", "up"), ("right", "down"), ("left", "up"), ("left", "down")]
        for c1, c2 in pairs_90:
            dot = np.dot(poses[c1], poses[c2])
            self.assertAlmostEqual(dot, 0.0, places=5, msg=f"Non-perpendicular pair: {c1} vs {c2} (dot={dot})")

        # 180° pairs
        pairs_180 = [("front", "back"), ("left", "right"), ("up", "down")]
        for c1, c2 in pairs_180:
            dot = np.dot(poses[c1], poses[c2])
            self.assertAlmostEqual(dot, -1.0, places=5, msg=f"Non-opposite pair: {c1} vs {c2} (dot={dot})")



class TestTrajectoryIngestionAndCompounding(unittest.TestCase):
    """Verifies loading 360DVO trajectory prior and compounding virtual rig cameras."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="abu_test_dvo_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_realitycapture_cameras_csv(self):
        csv_file = os.path.join(self.test_dir, "cameras.csv")
        content = (
            "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"
            "frame_00001.jpg,1.250000,-3.400000,0.850000,45.0000,90.0000,0.0000,35.0000,0,0,0,0,0,0,0,0\n"
            "frame_00002.jpg,1.550000,-3.100000,0.860000,50.0000,92.0000,-1.0000,35.0000,0,0,0,0,0,0,0,0\n"
        )
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(content)

        traj = ps.load_dvo_trajectory(csv_file)
        self.assertEqual(traj["count"], 2)
        self.assertIn("frame_00001.jpg", traj["by_name"])
        self.assertIn("frame_00001", traj["by_stem"])
        self.assertIn(0, traj["by_idx"])

        rec0 = traj["by_name"]["frame_00001.jpg"]
        self.assertAlmostEqual(rec0["x"], 1.25)
        self.assertAlmostEqual(rec0["y"], -3.40)
        self.assertAlmostEqual(rec0["alt"], 0.85)
        self.assertAlmostEqual(rec0["yaw"], 45.0)
        self.assertAlmostEqual(rec0["pitch"], 90.0)

    def test_load_tum_trajectory(self):
        tum_file = os.path.join(self.test_dir, "360dvo_trajectory.txt")
        # Identity rotation (qx=0, qy=0, qz=0, qw=1) at position (0.5, 1.0, 2.0)
        content = (
            "0.000 0.500000 1.000000 2.000000 0.000000 0.000000 0.000000 1.000000\n"
        )
        with open(tum_file, "w", encoding="utf-8") as f:
            f.write(content)

        traj = ps.load_dvo_trajectory(tum_file)
        self.assertEqual(traj["count"], 1)
        rec = traj["by_idx"][0]
        # In RC: X=X_colmap (0.5), Y=Z_colmap (2.0), Alt=-Y_colmap (-1.0)
        self.assertAlmostEqual(rec["x"], 0.5, places=4)
        self.assertAlmostEqual(rec["y"], 2.0, places=4)
        self.assertAlmostEqual(rec["alt"], -1.0, places=4)

    def test_compound_pose_identity_panorama(self):
        """Front and orthogonal cameras on a horizontal North-facing panorama."""
        pano_pose = {"x": 10.0, "y": 20.0, "alt": 5.0, "yaw": 0.0, "pitch": 90.0, "roll": 0.0}

        # 1. Front camera (0° yaw, 0° pitch)
        front_cam = ("front", 0.0, 0.0, 0.0, 90.0)
        pose_front = ps.compound_realitycapture_pose(pano_pose, front_cam, 1024, 1024)
        self.assertAlmostEqual(pose_front["x"], 10.0)
        self.assertAlmostEqual(pose_front["y"], 20.0)
        self.assertAlmostEqual(pose_front["alt"], 5.0)

        # 2. Verify all cubemap faces are orthogonal
        rig = ps.get_cubemap_rig(fov_deg=90.0)
        poses = {}
        for cam in rig:
            comp = ps.compound_realitycapture_pose(pano_pose, cam, 1024, 1024)
            M = ps.euler_zyx_to_matrix(comp["yaw"], comp["pitch"], comp["roll"])
            poses[cam[0]] = M[:, 2]

        for c1, c2 in [("front", "right"), ("front", "left"), ("front", "up"), ("front", "down")]:
            dot = np.dot(poses[c1], poses[c2])
            self.assertAlmostEqual(dot, 0.0, places=5)

        dot_fb = np.dot(poses["front"], poses["back"])
        self.assertAlmostEqual(dot_fb, -1.0, places=5)

    def test_compound_pose_rotated_panorama(self):
        """Panorama facing East (yaw=90°), horizontal."""
        pano_pose = {"x": 10.0, "y": 20.0, "alt": 30.0, "yaw": 90.0, "pitch": 90.0, "roll": 0.0}
        front_cam = ("front", 0.0, 0.0, 0.0, 90.0)
        right_cam = ("right", 90.0, 0.0, 0.0, 90.0)

        p_front = ps.compound_realitycapture_pose(pano_pose, front_cam, 1024, 1024, override_orientations_from_rig=True)
        self.assertAlmostEqual(p_front["x"], 10.0)
        p_right = ps.compound_realitycapture_pose(pano_pose, right_cam, 1024, 1024, override_orientations_from_rig=True)
        self.assertAlmostEqual(p_right["x"], 10.0)

        # Compounded mode preserves position and maintains orthogonal rig relationship
        p_c_front = ps.compound_realitycapture_pose(pano_pose, front_cam, 1024, 1024, override_orientations_from_rig=False)
        p_c_right = ps.compound_realitycapture_pose(pano_pose, right_cam, 1024, 1024, override_orientations_from_rig=False)
        M_f = ps.euler_zyx_to_matrix(p_c_front["yaw"], p_c_front["pitch"], p_c_front["roll"])
        M_r = ps.euler_zyx_to_matrix(p_c_right["yaw"], p_c_right["pitch"], p_c_right["roll"])
        dot = np.dot(M_f[:, 2], M_r[:, 2])
        self.assertAlmostEqual(dot, 0.0, places=5)



class TestDatasetSplittingExport(unittest.TestCase):
    """Tests end-to-end dataset splitting with RealityScan cameras.csv and perspective folders."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="abu_test_split_")
        self.in_dir = os.path.join(self.test_dir, "input")
        self.out_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.in_dir, exist_ok=True)

        # Create 2 synthetic equirectangular panoramas (64x32)
        try:
            import cv2
            img1 = np.zeros((32, 64, 3), dtype=np.uint8)
            img1[:, :, 0] = 200  # Blueish
            cv2.imwrite(os.path.join(self.in_dir, "pano_001.jpg"), img1)

            img2 = np.zeros((32, 64, 3), dtype=np.uint8)
            img2[:, :, 2] = 200  # Reddish
            cv2.imwrite(os.path.join(self.in_dir, "pano_002.jpg"), img2)
        except Exception:
            pass

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_split_with_subfolders_and_cameras_csv(self):
        # Create mock 360DVO cameras.csv in input dir
        dvo_csv = os.path.join(self.in_dir, "cameras.csv")
        content = (
            "#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"
            "pano_001.jpg,1.0,2.0,3.0,0.0,90.0,0.0,35.0,0,0,0,0,0,0,0,0\n"
            "pano_002.jpg,4.0,5.0,6.0,45.0,90.0,0.0,35.0,0,0,0,0,0,0,0,0\n"
        )
        with open(dvo_csv, "w", encoding="utf-8") as f:
            f.write(content)

        res = ps.split_panorama_dataset(
            input_dir=self.in_dir,
            output_dir=self.out_dir,
            mode="ring",
            num_cameras=4,
            fov_deg=90.0,
            crop_size=64,
            export_transforms_json=True,
            export_realityscan_csv=True,
            subfolder_per_perspective=True,
            dvo_trajectory_csv=dvo_csv,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["total_panoramas"], 2)
        self.assertEqual(res["total_views"], 8)  # 2 panos * 4 ring cameras

        # 1. Verify root cameras.csv
        root_csv = os.path.join(self.out_dir, "cameras.csv")
        self.assertTrue(os.path.isfile(root_csv))
        with open(root_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(lines[0], ps.RC_CSV_HEADER)
        # Header + 8 image rows = 9 lines
        self.assertEqual(len(lines), 9)

        # Verify first row contains compound pose matching pano_001.jpg (x=1.0, y=2.0, alt=3.0)
        parts = lines[1].split(",")
        self.assertEqual(len(parts), 16)
        self.assertAlmostEqual(float(parts[1]), 1.0)
        self.assertAlmostEqual(float(parts[2]), 2.0)
        self.assertAlmostEqual(float(parts[3]), 3.0)

        # 2. Verify perspective subfolders and all_cameras.csv
        persp_dir = os.path.join(self.out_dir, "perspectives")
        self.assertTrue(os.path.isdir(persp_dir))

        # Verify all_cameras.csv exists in perspectives/ directory
        all_cam_csv = os.path.join(persp_dir, "all_cameras.csv")
        self.assertTrue(os.path.isfile(all_cam_csv), "Missing perspectives/all_cameras.csv")
        with open(all_cam_csv, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        self.assertEqual(len(all_lines), 9)  # Header + 8 images

        cam_dirs = [d for d in os.listdir(persp_dir) if os.path.isdir(os.path.join(persp_dir, d))]
        self.assertEqual(len(cam_dirs), 4)

        for cam in cam_dirs:
            local_cam_dir = os.path.join(persp_dir, cam)
            # Verify cameras_<cam>.csv is written inside each perspective folder
            local_csv = os.path.join(local_cam_dir, f"cameras_{cam}.csv")
            self.assertTrue(os.path.isfile(local_csv), f"Missing local cameras_{cam}.csv in {cam}")
            with open(local_csv, "r", encoding="utf-8") as f:
                c_lines = f.readlines()
            # Header + 2 images = 3 lines
            self.assertEqual(len(c_lines), 3)
            # The name should have camera suffix (e.g. pano_001_ring_00_yaw000.jpg)
            first_name = c_lines[1].split(",")[0]
            expected_img_name = f"pano_001_{cam}.jpg"
            self.assertEqual(first_name, expected_img_name)
            # Verify file exists on disk with that exact name
            self.assertTrue(os.path.isfile(os.path.join(local_cam_dir, expected_img_name)))

            # Verify canonical XMP sidecar file exists for RealityScan Rig Parser (no redundant .jpg.xmp)
            xmp_stem = os.path.join(local_cam_dir, f"pano_001_{cam}.xmp")
            xmp_ext = os.path.join(local_cam_dir, f"{expected_img_name}.xmp")
            self.assertTrue(os.path.isfile(xmp_stem), f"Missing canonical XMP: {xmp_stem}")
            self.assertFalse(os.path.isfile(xmp_ext), f"Redundant .jpg.xmp should not exist: {xmp_ext}")

            # Verify XMP XML contents conform to True_Schema_360_Camera_Rig
            with open(xmp_stem, "r", encoding="utf-8") as xf:
                xmp_content = xf.read()
            self.assertIn('xcr:PosePrior="locked"', xmp_content)
            self.assertIn('xcr:CalibrationPrior="exact"', xmp_content)
            self.assertIn('xcr:CalibrationGroup="1"', xmp_content)
            self.assertIn('xcr:DistortionGroup="1"', xmp_content)
            self.assertIn('xcr:DistortionModel="perspective"', xmp_content)
            self.assertIn('xcr:DistortionCoeficients="0 0 0 0 0 0"', xmp_content)
            self.assertIn('xcr:DistortionPrior="exact"', xmp_content)
            self.assertIn('xcr:FocalLength35mm="', xmp_content)
            self.assertIn("<xcr:Rotation>", xmp_content)
            self.assertIn("<xcr:Position>", xmp_content)
            self.assertNotIn('xcr:Rig="', xmp_content)
            self.assertNotIn('xcr:LockedPoseGroup="', xmp_content)

        # 3. Verify cameras_csv/ folder contains separate CSV per camera
        csv_dir = os.path.join(self.out_dir, "cameras_csv")
        self.assertTrue(os.path.isdir(csv_dir))
        for cam in cam_dirs:
            p_csv = os.path.join(csv_dir, f"{cam}_cameras.csv")
            self.assertTrue(os.path.isfile(p_csv))

    def test_split_flat_layout(self):
        """Tests flat images/ layout where subfolder_per_perspective is False."""
        res = ps.split_panorama_dataset(
            input_dir=self.in_dir,
            output_dir=self.out_dir,
            mode="cubemap",
            fov_deg=90.0,
            crop_size=64,
            export_transforms_json=True,
            export_realityscan_csv=True,
            subfolder_per_perspective=False,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["total_views"], 12)  # 2 panos * 6 cubemap faces

        root_csv = os.path.join(self.out_dir, "cameras.csv")
        self.assertTrue(os.path.isfile(root_csv))
        images_csv = os.path.join(self.out_dir, "images", "cameras.csv")
        self.assertTrue(os.path.isfile(images_csv))


if __name__ == "__main__":
    unittest.main()
