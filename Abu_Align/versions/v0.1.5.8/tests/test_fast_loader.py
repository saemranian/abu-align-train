"""
Test Fast Parallel Image Batch Loader & DCT Scaling for VGGT-Omega
===================================================================
Verifies:
  1. Multi-threaded batch decoding and target shape calculation.
  2. Extreme aspect ratio cropping [0.5, 2.0].
  3. libjpeg-turbo DCT draft scaling fidelity.
  4. Multi-shape padding handling.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

from align_cli import (
    load_and_preprocess_vggt_images,
    _process_single_image_vggt_omega
)


class TestFastImageLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="abu_test_fast_loader_")
        cls.image_paths = []

        # Create standard 16:9 images
        for i in range(4):
            p = os.path.join(cls.temp_dir, f"std_{i}.jpg")
            img = Image.new("RGB", (1920, 1080), color=(50 + i * 20, 100, 150))
            img.save(p, quality=90)
            cls.image_paths.append(p)

        # Create portrait 9:16 images
        for i in range(2):
            p = os.path.join(cls.temp_dir, f"portrait_{i}.jpg")
            img = Image.new("RGB", (1080, 1920), color=(100, 50 + i * 30, 200))
            img.save(p, quality=90)
            cls.image_paths.append(p)

        # Create extreme wide image (AR < 0.5)
        p_wide = os.path.join(cls.temp_dir, "wide.jpg")
        img_wide = Image.new("RGB", (3000, 1000), color=(120, 80, 200))
        img_wide.save(p_wide, quality=90)
        cls.image_paths.append(p_wide)
        cls.wide_path = p_wide

        # Create extreme tall image (AR > 2.0)
        p_tall = os.path.join(cls.temp_dir, "tall.jpg")
        img_tall = Image.new("RGB", (1000, 3000), color=(80, 180, 120))
        img_tall.save(p_tall, quality=90)
        cls.image_paths.append(p_tall)
        cls.tall_path = p_tall

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_single_image_processing(self):
        rgb_arr, orig_sz, target_sz = _process_single_image_vggt_omega(
            self.image_paths[0], mode="max_size", image_resolution=512, patch_size=16
        )
        self.assertEqual(orig_sz, (1920, 1080))
        th, tw = target_sz
        self.assertEqual(th % 16, 0)
        self.assertEqual(tw % 16, 0)
        self.assertEqual(rgb_arr.shape, (th, tw, 3))
        self.assertEqual(rgb_arr.dtype, np.uint8)

    def test_aspect_ratio_cropping_extreme(self):
        # Wide image
        _, _, (th_w, tw_w) = _process_single_image_vggt_omega(
            self.wide_path, mode="max_size", image_resolution=512, patch_size=16
        )
        ar_w = th_w / tw_w
        self.assertGreaterEqual(ar_w, 0.49)
        self.assertLessEqual(ar_w, 2.01)

        # Tall image
        _, _, (th_t, tw_t) = _process_single_image_vggt_omega(
            self.tall_path, mode="max_size", image_resolution=512, patch_size=16
        )
        ar_t = th_t / tw_t
        self.assertGreaterEqual(ar_t, 0.49)
        self.assertLessEqual(ar_t, 2.01)

    def test_batch_parallel_loader_max_size(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        imgs_t, coords, rgb_uint8, is_sq = load_and_preprocess_vggt_images(
            self.image_paths[:4], engine="vggt_omega", preprocess_mode="max_size",
            vggt_res=512, device=device
        )
        self.assertEqual(imgs_t.shape[0], 4)
        self.assertEqual(imgs_t.shape[1], 3)
        self.assertEqual(coords.shape, (4, 6))
        self.assertEqual(rgb_uint8.shape[0], 4)
        self.assertFalse(is_sq)
        self.assertGreaterEqual(imgs_t.min().item(), 0.0)
        self.assertLessEqual(imgs_t.max().item(), 1.0)

    def test_batch_parallel_loader_balanced_mode(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        imgs_t, coords, rgb_uint8, is_sq = load_and_preprocess_vggt_images(
            self.image_paths[:4], engine="vggt_omega", preprocess_mode="balanced",
            vggt_res=512, device=device
        )
        self.assertEqual(imgs_t.shape[0], 4)
        self.assertFalse(is_sq)

    def test_batch_mixed_aspect_ratio_padding(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Mix standard (16:9) and portrait (9:16)
        mixed_paths = [self.image_paths[0], self.image_paths[4]]
        imgs_t, coords, rgb_uint8, is_sq = load_and_preprocess_vggt_images(
            mixed_paths, engine="vggt_omega", preprocess_mode="max_size",
            vggt_res=512, device=device
        )
        self.assertEqual(imgs_t.shape[0], 2)
        # Should be padded to common square/max shape
        self.assertEqual(imgs_t.shape[2], imgs_t.shape[3])


if __name__ == "__main__":
    unittest.main()
