"""
Test Suite for Deep Multi-View Tracking & Trajectory Co-visibility Safeguards (v0.0.6.5)
========================================================================================
Verifies:
1. batch_np_matrix_to_pycolmap works with partial inlier distributions (trajectory scans)
2. _augment_non_visible_frames respects max_augment_queries ceiling
3. _augment_non_visible_frames safely terminates when scene has open trajectory co-visibility
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
from vggt.dependency.track_predict import _augment_non_visible_frames


class TestTrackingOptimization(unittest.TestCase):
    def test_partial_inlier_reconstruction_pycolmap(self):
        """
        Verify that batch_np_matrix_to_pycolmap successfully builds a reconstruction
        and enables Ceres BA even when distant frames along an open trajectory have 0 inliers.
        """
        N = 6
        P = 40
        points3d = np.random.randn(P, 3).astype(np.float32)
        points3d[:, 2] += 5.0  # in front of camera

        extrinsics = np.zeros((N, 3, 4), dtype=np.float32)
        for i in range(N):
            extrinsics[i, :3, :3] = np.eye(3)
            extrinsics[i, 0, 3] = i * 0.1

        intrinsics = np.zeros((N, 3, 3), dtype=np.float32)
        for i in range(N):
            intrinsics[i] = np.array([[500, 0, 256], [0, 500, 256], [0, 0, 1]], dtype=np.float32)

        tracks = np.random.uniform(0, 512, (N, P, 2)).astype(np.float32)
        image_size = np.array([512, 512])

        # Frames 0, 1, 2 have 25 inliers each. Frames 3, 4, 5 have 0 inliers (un-co-visible distant frames).
        masks = np.zeros((N, P), dtype=bool)
        masks[0, :25] = True
        masks[1, :25] = True
        masks[2, :25] = True

        reconstruction, valid_mask = batch_np_matrix_to_pycolmap(
            points3d,
            extrinsics,
            intrinsics,
            tracks,
            image_size,
            masks=masks,
            min_inlier_per_frame=8,
            camera_type="PINHOLE",
        )

        self.assertIsNotNone(reconstruction)
        self.assertGreaterEqual(reconstruction.num_reg_images(), 2)
        self.assertEqual(len(reconstruction.images), N)

    def test_augment_zero_limit_noop(self):
        """Verify that max_augment_queries <= 0 returns immediately without tracking extra frames."""
        pred_tracks = [np.zeros((10, 10, 2))]
        pred_vis_scores = [np.zeros((10, 10))]
        pred_confs = [np.zeros(10)]
        pred_points_3d = [np.zeros((10, 3))]
        pred_colors = [np.zeros((10, 3))]

        out_tracks, out_vis, out_confs, out_pts, out_cols = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images=None,
            conf=None,
            points_3d=None,
            fmaps_for_tracker=None,
            keypoint_extractors=None,
            tracker=None,
            max_points_num=1000,
            fine_tracking=False,
            max_augment_queries=0,
        )

        # Output lists should be unchanged
        self.assertEqual(len(out_tracks), 1)
        self.assertEqual(len(out_vis), 1)

    def test_augment_trajectory_guard_early_exit(self):
        """Verify that having >20 non-visible frames immediately breaks out of augmentation loop."""
        # 50 frames with 0 visibility
        pred_tracks = [np.zeros((50, 5, 2))]
        pred_vis_scores = [np.zeros((50, 5))]
        pred_confs = [np.zeros(5)]
        pred_points_3d = [np.zeros((5, 3))]
        pred_colors = [np.zeros((5, 3))]

        out_tracks, out_vis, out_confs, out_pts, out_cols = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images=None,
            conf=None,
            points_3d=None,
            fmaps_for_tracker=None,
            keypoint_extractors=None,
            tracker=None,
            max_points_num=1000,
            fine_tracking=False,
            max_augment_queries=5,
        )

        # Should have exited without entering the forward loop (which would crash on images=None)
        self.assertEqual(len(out_tracks), 1)

    def test_switch_tensor_order_cross_device(self):
        """Verify switch_tensor_order dynamically matches order device to target tensor device."""
        import torch
        from vggt.dependency.vggsfm_utils import switch_tensor_order

        t_cpu = torch.randn(2, 10, 4)
        order_cuda = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], device="cuda" if torch.cuda.is_available() else "cpu")

        res = switch_tensor_order([t_cpu], order_cuda, dim=1)
        self.assertEqual(res[0].device, t_cpu.device)
        self.assertEqual(res[0].shape, t_cpu.shape)

        if torch.cuda.is_available():
            t_cuda = torch.randn(2, 10, 4, device="cuda")
            order_cpu = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device="cpu")
            res_cuda = switch_tensor_order([t_cuda], order_cpu, dim=1)
            self.assertEqual(res_cuda[0].device, t_cuda.device)
            self.assertEqual(res_cuda[0].shape, t_cuda.shape)


if __name__ == "__main__":
    unittest.main()

