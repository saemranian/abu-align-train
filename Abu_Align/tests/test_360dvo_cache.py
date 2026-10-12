import os
import sys
import unittest
import tempfile
import shutil
import numpy as np

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cache_manager import (
    compute_dataset_hash,
    save_360dvo_cache,
    load_360dvo_cache,
    clear_cache,
)


class Test360DVOCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_360dvo_cache_")
        self.cache_dir = os.path.join(self.test_dir, ".abu_cache")
        self.source_dir = os.path.join(self.test_dir, "run_01")
        self.dest_dir = os.path.join(self.test_dir, "run_02")
        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.dest_dir, exist_ok=True)

        # Create dummy image files
        self.images_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.image_paths = []
        for i in range(5):
            p = os.path.join(self.images_dir, f"frame_{i:04d}.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            self.image_paths.append(p)

        # Create dummy 360DVO artifacts in source_dir
        self.traj_file = os.path.join(self.source_dir, "360dvo_trajectory.txt")
        with open(self.traj_file, "w", encoding="utf-8") as f:
            f.write("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")
            f.write("0.1 0.1 0.0 0.0 0.0 0.0 0.0 1.0\n")

        self.ply_file = os.path.join(self.source_dir, "360dvo_points.ply")
        with open(self.ply_file, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n0.0 0.0 0.0\n")

        # Create dummy sparse folder
        self.sparse_dir = os.path.join(self.source_dir, "sparse", "0")
        os.makedirs(self.sparse_dir, exist_ok=True)
        with open(os.path.join(self.sparse_dir, "cameras.txt"), "w") as f:
            f.write("1 PINHOLE 1024 512 500 500 512 256\n")

    def tearDown(self):
        if os.path.isdir(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_compute_dataset_hash_with_extra_params(self):
        hash1 = compute_dataset_hash(
            image_paths=self.image_paths,
            engine="360dvo",
            load_res=1024,
            stride=1,
            extra_params={"patches": 96, "ba_window": 8}
        )
        hash2 = compute_dataset_hash(
            image_paths=self.image_paths,
            engine="360dvo",
            load_res=1024,
            stride=1,
            extra_params={"patches": 96, "ba_window": 8}
        )
        hash3 = compute_dataset_hash(
            image_paths=self.image_paths,
            engine="360dvo",
            load_res=1024,
            stride=1,
            extra_params={"patches": 128, "ba_window": 8}
        )
        self.assertEqual(hash1, hash2, "Identical 360DVO configs must produce identical cache keys")
        self.assertNotEqual(hash1, hash3, "Different patch configurations must produce different cache keys")

    def test_save_and_load_360dvo_cache(self):
        cache_key = "test_key_123"
        saved_dir = save_360dvo_cache(
            cache_dir=self.cache_dir,
            cache_key=cache_key,
            source_dir=self.source_dir,
            metadata={"num_frames": 5, "patches": 96}
        )
        self.assertTrue(os.path.isdir(saved_dir))
        self.assertTrue(os.path.isfile(os.path.join(saved_dir, "360dvo_trajectory.txt")))
        self.assertTrue(os.path.isfile(os.path.join(saved_dir, "360dvo_points.ply")))
        self.assertTrue(os.path.isfile(os.path.join(saved_dir, "sparse", "0", "cameras.txt")))

        # Test loading into a new output destination
        loaded_meta = load_360dvo_cache(
            cache_dir=self.cache_dir,
            cache_key=cache_key,
            output_dir=self.dest_dir
        )
        self.assertIsNotNone(loaded_meta, "Should return metadata dictionary on cache hit")
        self.assertEqual(loaded_meta["patches"], 96)
        self.assertTrue(os.path.isfile(os.path.join(self.dest_dir, "360dvo_trajectory.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.dest_dir, "360dvo_points.ply")))
        self.assertTrue(os.path.isfile(os.path.join(self.dest_dir, "sparse", "0", "cameras.txt")))

    def test_360dvo_cache_miss(self):
        loaded_meta = load_360dvo_cache(
            cache_dir=self.cache_dir,
            cache_key="non_existent_key",
            output_dir=self.dest_dir
        )
        self.assertIsNone(loaded_meta, "Cache miss must return None")

    def test_clear_cache(self):
        save_360dvo_cache(
            cache_dir=self.cache_dir,
            cache_key="key_to_clear",
            source_dir=self.source_dir
        )
        self.assertTrue(os.path.isdir(self.cache_dir))
        n_cleared, b_freed = clear_cache(self.cache_dir)
        self.assertGreater(b_freed, 0)
        self.assertFalse(os.path.isdir(self.cache_dir))


if __name__ == "__main__":
    unittest.main()
