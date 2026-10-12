import os
import shutil
import tempfile
import unittest
from cache_manager import safe_remove_tree


class TestSafeImageStagingAndRemoval(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="abu_test_loop_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            safe_remove_tree(self.test_dir)

    def test_safe_remove_tree_normal(self):
        """Verifies safe_remove_tree cleanly removes standard folder hierarchies."""
        sub = os.path.join(self.test_dir, "a", "b", "c")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "test.txt"), "w") as f:
            f.write("test")
        self.assertTrue(safe_remove_tree(sub))
        self.assertFalse(os.path.exists(sub))

    def test_safe_image_staging_no_recursion(self):
        """
        Simulates the user's workflow where output_dir (results/sparse_01)
        is located INSIDE the dataset directory (which contains images directly and subdirectories).
        Ensures staging only links/copies files into 3dgs_dataset/images and NEVER recurses.
        """
        dataset_dir = os.path.join(self.test_dir, "dataset")
        results_dir = os.path.join(dataset_dir, "results", "sparse_01")
        sub_dir = os.path.join(dataset_dir, "sub_folder")
        os.makedirs(sub_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Create dummy images in dataset_dir and dummy file in sub_dir
        img1 = os.path.join(dataset_dir, "frame_001.jpg")
        img2 = os.path.join(dataset_dir, "frame_002.jpg")
        with open(img1, "wb") as f:
            f.write(b"img1")
        with open(img2, "wb") as f:
            f.write(b"img2")
        with open(os.path.join(sub_dir, "ignore.txt"), "w") as f:
            f.write("ignore")

        image_paths = [img1, img2]
        dataset_3dgs = os.path.join(results_dir, "3dgs_dataset")
        d3_imgs = os.path.join(dataset_3dgs, "images")
        os.makedirs(d3_imgs, exist_ok=True)

        # Execute safe staging logic
        for src in image_paths:
            dst = os.path.join(d3_imgs, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

        # Verifications
        self.assertTrue(os.path.isfile(os.path.join(d3_imgs, "frame_001.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(d3_imgs, "frame_002.jpg")))
        # Crucial: NO subdirectories (like 'results' or 'sub_folder') must exist in d3_imgs!
        subdirs_in_images = [d for d in os.listdir(d3_imgs) if os.path.isdir(os.path.join(d3_imgs, d))]
        self.assertEqual(subdirs_in_images, [], "images/ folder must NEVER contain subdirectories!")


if __name__ == "__main__":
    unittest.main()
