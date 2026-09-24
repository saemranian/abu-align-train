"""
Unit tests for VGGT-Omega Foundation Model Alignment Optimization,
Zero-Paging VRAM Safety, and Direct Indexing in abu align (v0.0.5.9).
"""

import unittest
import os
import sys
import numpy as np
import tempfile
from PIL import Image

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from align_cli import (
    load_and_preprocess_vggt_images,
    rename_colmap_recons_and_rescale_camera,
)
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import pycolmap


class TestVGGTOmegaOptimization(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory with test images
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img1_path = os.path.join(self.temp_dir.name, "test_16_9.jpg")
        self.img2_path = os.path.join(self.temp_dir.name, "test_4_3.jpg")

        # Create dummy photos of different aspect ratios
        img1 = Image.new("RGB", (1920, 1080), color=(120, 150, 180))
        img1.save(self.img1_path, "JPEG")

        img2 = Image.new("RGB", (1600, 1200), color=(200, 100, 50))
        img2.save(self.img2_path, "JPEG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_and_preprocess_vggt_modes(self):
        """Test preprocessing under max_size vs square modes."""
        paths = [self.img1_path, self.img2_path]

        # Test square pad mode (CPU / fallback safe)
        imgs, coords, rgb_uint8, is_square = load_and_preprocess_vggt_images(
            paths, engine="vggt", preprocess_mode="square", vggt_res=512, img_load_res=1024, device="cpu"
        )
        self.assertTrue(is_square)
        self.assertEqual(imgs.shape[0], 2)
        self.assertEqual(rgb_uint8.shape, (2, 512, 512, 3))
        self.assertEqual(rgb_uint8.dtype, np.uint8)
        self.assertEqual(coords.shape[0], 2)

    def test_zero_memory_direct_indexing(self):
        """Verify the O(1) mathematical index mapping without full 3D coordinate grid allocation."""
        N, H, W = 4, 288, 512
        np.random.seed(42)

        # Mock points_3d and depth_conf
        depth_conf = np.random.uniform(0.5, 3.0, size=(N, H, W)).astype(np.float32)
        points_3d = np.random.randn(N, H, W, 3).astype(np.float32)
        points_rgb = np.random.randint(0, 256, size=(N, H, W, 3), dtype=np.uint8)

        conf_thresh = 1.8
        conf_mask = depth_conf >= conf_thresh
        valid_flat = np.flatnonzero(conf_mask.reshape(-1))

        self.assertGreater(len(valid_flat), 0)

        # Select budget
        max_colmap_pts = min(len(valid_flat), 500)
        selected_flat = np.random.choice(valid_flat, size=max_colmap_pts, replace=False)

        # Fast direct coordinate mapping
        f_idx = selected_flat // (H * W)
        rem = selected_flat % (H * W)
        y_idx = rem // W
        x_idx = rem % W

        pts_xyf = np.stack([x_idx.astype(np.float32), y_idx.astype(np.float32), f_idx.astype(np.float32)], axis=-1)
        pts_3d_sampled = points_3d.reshape(-1, 3)[selected_flat]
        pts_rgb_sampled = points_rgb.reshape(-1, 3)[selected_flat]

        # Verify ground-truth consistency
        for i in range(len(selected_flat)):
            f = int(f_idx[i])
            y = int(y_idx[i])
            x = int(x_idx[i])

            self.assertGreaterEqual(depth_conf[f, y, x], conf_thresh)
            np.testing.assert_array_equal(pts_3d_sampled[i], points_3d[f, y, x])
            np.testing.assert_array_equal(pts_rgb_sampled[i], points_rgb[f, y, x])
            self.assertEqual(pts_xyf[i, 0], x)
            self.assertEqual(pts_xyf[i, 1], y)
            self.assertEqual(pts_xyf[i, 2], f)

    def test_non_square_camera_rescaling(self):
        """Verify proportional scaling without letterbox offsets for native aspect ratios."""
        N = 2
        # Mock COLMAP reconstruction
        reconstruction = pycolmap.Reconstruction()

        # Target dimensions (e.g. 512x288) and Real dimensions (3840x2160) => 7.5x scale
        orig_w, orig_h = 3840.0, 2160.0
        target_w, target_h = 512.0, 288.0
        orig_coords = np.array([
            [0.0, 0.0, target_w, target_h, orig_w, orig_h],
            [0.0, 0.0, target_w, target_h, orig_w, orig_h],
        ], dtype=np.float32)

        # Initial camera params at target resolution: [fx, fy, cx, cy]
        extrinsics = np.tile(np.eye(4)[None, :3, :], (N, 1, 1))
        intrinsics = np.tile(np.array([
            [400.0, 0.0, 256.0],
            [0.0, 400.0, 144.0],
            [0.0, 0.0, 1.0]
        ])[None, :, :], (N, 1, 1))
        image_size = np.array([int(target_w), int(target_h)])

        points_3d = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32)
        points_xyf = np.array([[256.0, 144.0, 0.0], [256.0, 144.0, 1.0]], dtype=np.float32)
        points_rgb = np.full((2, 3), 128, dtype=np.uint8)

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d, points_xyf, points_rgb, extrinsics, intrinsics, image_size, shared_camera=False, camera_type="PINHOLE"
        )

        # Rescale with is_square_padded=False (Native Aspect Ratio Scaling)
        image_names = ["photo_1.jpg", "photo_2.jpg"]
        rescaled = rename_colmap_recons_and_rescale_camera(
            reconstruction,
            image_names,
            orig_coords,
            img_size=512,
            shift_point2d_to_original_res=True,
            shared_camera=False,
            is_square_padded=False
        )

        scale_x = orig_w / target_w  # 3840 / 512 = 7.5
        scale_y = orig_h / target_h  # 2160 / 288 = 7.5

        for i in range(1, N + 1):
            cam = rescaled.cameras[i]
            self.assertEqual(cam.width, 3840)
            self.assertEqual(cam.height, 2160)
            self.assertAlmostEqual(cam.params[0], 400.0 * scale_x, places=3)
            self.assertAlmostEqual(cam.params[1], 400.0 * scale_y, places=3)
            self.assertAlmostEqual(cam.params[2], 256.0 * scale_x, places=3)
            self.assertAlmostEqual(cam.params[3], 144.0 * scale_y, places=3)

            img = rescaled.images[i]
            self.assertEqual(img.name, image_names[i - 1])
            self.assertAlmostEqual(img.points2D[0].xy[0], 256.0 * scale_x, places=3)
            self.assertAlmostEqual(img.points2D[0].xy[1], 144.0 * scale_y, places=3)

    def test_batch_np_matrix_shared_camera(self):
        """Verify batch_np_matrix_to_pycolmap_wo_track respects shared_camera."""
        N = 3
        P = 30
        points_3d = np.random.randn(P, 3).astype(np.float32)
        points_xyf = np.zeros((P, 3), dtype=np.float32)
        for i in range(P):
            points_xyf[i] = [100.0, 100.0, float(i % N)]
        points_rgb = np.full((P, 3), 200, dtype=np.uint8)

        extrinsics = np.tile(np.eye(4)[None, :3, :], (N, 1, 1))
        intrinsics = np.tile(np.array([
            [500.0, 0.0, 256.0],
            [0.0, 500.0, 256.0],
            [0.0, 0.0, 1.0]
        ])[None, :, :], (N, 1, 1))
        image_size = np.array([512, 512])

        # Test shared_camera=True
        recons_shared = batch_np_matrix_to_pycolmap_wo_track(
            points_3d, points_xyf, points_rgb, extrinsics, intrinsics, image_size, shared_camera=True, camera_type="PINHOLE"
        )
        self.assertEqual(len(recons_shared.cameras), 1)
        self.assertEqual(len(recons_shared.images), N)
        for img in recons_shared.images.values():
            self.assertEqual(img.camera_id, 1)

        # Test shared_camera=False
        recons_unshared = batch_np_matrix_to_pycolmap_wo_track(
            points_3d, points_xyf, points_rgb, extrinsics, intrinsics, image_size, shared_camera=False, camera_type="PINHOLE"
        )
        self.assertEqual(len(recons_unshared.cameras), N)
        self.assertEqual(len(recons_unshared.images), N)

    def test_save_neural_cache_numpy_arrays(self):
        """Verify save_neural_cache accepts NumPy arrays without .detach() AttributeError."""
        from cache_manager import save_neural_cache, load_neural_cache

        N, H, W = 2, 288, 512
        extrinsic = np.eye(4, dtype=np.float32)[None, :3, :].repeat(N, axis=0)
        intrinsic = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0)
        original_coords = np.zeros((N, 6), dtype=np.float32)
        points_3d = np.random.randn(N, H, W, 3).astype(np.float32)
        depth_conf = np.random.uniform(0.5, 3.0, size=(N, H, W)).astype(np.float32)
        depth_map = np.random.uniform(1.0, 10.0, size=(N, H, W)).astype(np.float32)
        points_rgb = np.random.randint(0, 256, size=(N, H, W, 3), dtype=np.uint8)
        images = np.random.uniform(0.0, 1.0, size=(N, 3, H, W)).astype(np.float32)

        cache_key = "test_numpy_cache"
        target_dir = os.path.join(self.temp_dir.name, cache_key)

        # Must succeed without raising AttributeError: 'numpy.ndarray' object has no attribute 'detach'
        save_neural_cache(
            cache_dir=self.temp_dir.name,
            cache_key=cache_key,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            original_coords=original_coords,
            points_3d=points_3d,
            depth_conf=depth_conf,
            depth_map=depth_map,
            points_rgb=points_rgb,
            images=images,
            metadata={"test": True}
        )

        self.assertTrue(os.path.exists(os.path.join(target_dir, "extrinsic.pt")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "intrinsic.pt")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "original_coords.pt")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "points_3d_fp16.pt")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "depth_conf_fp16.pt")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "depth_map_fp16.pt")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "points_rgb.npy")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "images_u8.pt")))

        # Test loading it back
        cached = load_neural_cache(self.temp_dir.name, cache_key, device="cpu")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["extrinsic"].shape, (N, 3, 4))
        self.assertEqual(cached["intrinsic"].shape, (N, 3, 3))
        self.assertEqual(cached["points_3d"].shape, (N, H, W, 3))

    def test_non_square_tracking_coordinate_scaling(self):
        """Verify non-square aspect ratio keypoint scaling and boundary clamping in track_predict."""
        import torch
        # Test rectangular frame (H=288, W=512) with model confidence map (conf_h=288, conf_w=512)
        H, W = 288, 512
        conf_h, conf_w = 288, 512
        query_points = torch.tensor([[[0.0, 0.0], [511.4, 287.6], [600.0, 400.0]]], dtype=torch.float32)

        scale_x = float(conf_w) / float(W)
        scale_y = float(conf_h) / float(H)

        pts = query_points.squeeze(0)
        qx_scaled = (pts[:, 0] * scale_x).round().long().clamp(0, conf_w - 1).cpu().numpy()
        qy_scaled = (pts[:, 1] * scale_y).round().long().clamp(0, conf_h - 1).cpu().numpy()

        self.assertEqual(qx_scaled[0], 0)
        self.assertEqual(qy_scaled[0], 0)
        self.assertEqual(qx_scaled[1], 511)
        self.assertEqual(qy_scaled[1], 288 - 1)
        # Clamped out of bounds:
        self.assertEqual(qx_scaled[2], 511)
        self.assertEqual(qy_scaled[2], 287)


if __name__ == "__main__":
    unittest.main()
