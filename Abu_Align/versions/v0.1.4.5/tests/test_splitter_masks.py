"""
Unit tests for Equirectangular Mask Layer Auto-Detection, Splitting, and RealityScan
Layer Alignment in panoramic_splitter.py.
"""
import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np

import panoramic_splitter as ps

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@unittest.skipUnless(HAS_CV2, "OpenCV (cv2) is required for mask splitting tests")
class TestMaskDiscoveryAndLookup(unittest.TestCase):
    """Tests automatic discovery and indexing of equirectangular mask directories."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_mask_disc_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def test_find_auto_mask_dir_conventions(self):
        # 1. No mask folder initially
        self.assertIsNone(ps.find_auto_mask_dir(self.temp_dir))

        # 2. Ignored output folders should NOT be treated as masks even if named mask
        ignored_p = os.path.join(self.temp_dir, "pinhole_rig")
        os.makedirs(ignored_p, exist_ok=True)
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(ignored_p, "dummy_mask.jpg"), dummy_img)
        self.assertIsNone(ps.find_auto_mask_dir(self.temp_dir))

        # 3. Create 'Mask Pass' folder with image
        mask_pass = os.path.join(self.temp_dir, "Mask Pass")
        os.makedirs(mask_pass, exist_ok=True)
        cv2.imwrite(os.path.join(mask_pass, "pano_001.jpg"), dummy_img)

        found = ps.find_auto_mask_dir(self.temp_dir)
        self.assertIsNotNone(found)
        self.assertEqual(os.path.abspath(found), os.path.abspath(mask_pass))

    def test_build_mask_lookup_and_matching(self):
        mask_dir = os.path.join(self.temp_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(mask_dir, "Container1_Mask000.jpg"), dummy_img)
        cv2.imwrite(os.path.join(mask_dir, "Container1_Mask001.jpg"), dummy_img)

        lookup = ps.build_mask_lookup(mask_dir)
        self.assertIsNotNone(lookup)
        self.assertEqual(lookup["count"], 2)

        # 1. Match by exact name
        m0 = ps.find_mask_for_frame(lookup, "Container1_Mask000.jpg", 0)
        self.assertIsNotNone(m0)
        self.assertTrue(m0.endswith("Container1_Mask000.jpg"))

        # 2. Match by stem
        m1 = ps.find_mask_for_frame(lookup, "Container1_Mask001", 1)
        self.assertIsNotNone(m1)
        self.assertTrue(m1.endswith("Container1_Mask001.jpg"))

        # 3. Match by index
        m_idx = ps.find_mask_for_frame(lookup, "other_name_000", 0)
        self.assertIsNotNone(m_idx)
        self.assertTrue(m_idx.endswith("Container1_Mask000.jpg"))


@unittest.skipUnless(HAS_CV2, "OpenCV (cv2) is required for mask splitting tests")
class TestMaskRigProjection(unittest.TestCase):
    """Verifies end-to-end splitting of main panoramas and equirectangular masks into perspective views."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_mask_proj_")
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

        # Create 2:1 ERP synthetic main image (gradient)
        h, w = 120, 240
        self.main_erp = np.zeros((h, w, 3), dtype=np.uint8)
        self.main_erp[:, :, 0] = 128
        self.main_erp[:, :, 1] = 64
        self.main_erp[:, :, 2] = 200

        # Create 2:1 ERP synthetic mask image (binary circle)
        self.mask_erp = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(self.mask_erp, (w // 2, h // 2), 30, (255, 255, 255), -1)

        # Write to input dataset
        self.input_dir = os.path.join(self.temp_dir, "dataset")
        os.makedirs(self.input_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.input_dir, "frame_001.jpg"), self.main_erp)

        # Write to mask folder
        self.mask_dir = os.path.join(self.input_dir, "Mask Pass")
        os.makedirs(self.mask_dir, exist_ok=True)
        cv2.imwrite(os.path.join(self.mask_dir, "frame_001.jpg"), self.mask_erp)

        self.output_dir = os.path.join(self.temp_dir, "pinhole_rig")

    def test_split_panorama_with_auto_detected_masks(self):
        """Tests that split_panorama_dataset auto-detects Mask Pass and creates .mask.jpg files."""
        res = ps.split_panorama_dataset(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            mode="cubemap",
            crop_size=64,
            subfolder_per_perspective=True,
            export_realityscan_csv=True,
            export_xmp=True,
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["total_panoramas"], 1)
        self.assertEqual(res["total_views"], 6)
        self.assertEqual(res["total_masks"], 6)

        perspectives_dir = os.path.join(self.output_dir, "perspectives")
        for face in ["front", "right", "back", "left", "up", "down"]:
            face_dir = os.path.join(perspectives_dir, face)
            self.assertTrue(os.path.isdir(face_dir))

            # Main image exists
            main_img_path = os.path.join(face_dir, f"frame_001_{face}.jpg")
            self.assertTrue(os.path.isfile(main_img_path))

            # Mask image exists next to main image with .mask.jpg
            mask_img_path = os.path.join(face_dir, f"frame_001_{face}.mask.jpg")
            self.assertTrue(os.path.isfile(mask_img_path), f"Mask image not found: {mask_img_path}")

            # Verify mask dimensions match main image dimensions
            main_shape = cv2.imread(main_img_path).shape
            mask_shape = cv2.imread(mask_img_path).shape
            self.assertEqual(main_shape, mask_shape)

            # Main image canonical XMP sidecar exists (redundant .jpg.xmp does not exist)
            self.assertTrue(os.path.isfile(os.path.join(face_dir, f"frame_001_{face}.xmp")))
            self.assertFalse(os.path.isfile(f"{main_img_path}.xmp"))

            # CRITICAL CONSTRAINT: NO XMP sidecar should exist for the mask image
            self.assertFalse(os.path.isfile(f"{mask_img_path}.xmp"))
            self.assertFalse(os.path.isfile(os.path.join(face_dir, f"frame_001_{face}.mask.xmp")))

        # Verify all_cameras.csv does NOT contain any .mask rows
        all_csv = os.path.join(perspectives_dir, "all_cameras.csv")
        self.assertTrue(os.path.isfile(all_csv))
        with open(all_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # 1 header + 6 faces = 7 lines
        self.assertEqual(len(lines), 7)
        for line in lines:
            self.assertNotIn(".mask", line)

        # Verify manifest recorded mask telemetry
        manifest_path = os.path.join(self.output_dir, "rig_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["total_masks_exported"], 6)
        self.assertIsNotNone(manifest["mask_dir_used"])

    def test_split_masks_for_dataset_standalone(self):
        """Tests that split_masks_for_dataset can project masks on an already exported rig."""
        # 1. Export rig without masks first
        ps.split_panorama_dataset(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            mode="cubemap",
            crop_size=64,
            subfolder_per_perspective=True,
            auto_detect_mask=False,  # disabled initially
        )

        perspectives_dir = os.path.join(self.output_dir, "perspectives")
        for face in ["front", "right", "back", "left", "up", "down"]:
            mask_p = os.path.join(perspectives_dir, face, f"frame_001_{face}.mask.jpg")
            self.assertFalse(os.path.isfile(mask_p))

        # 2. Call dedicated split_masks_for_dataset
        m_res = ps.split_masks_for_dataset(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            mask_dir=self.mask_dir,
        )

        self.assertTrue(m_res["success"])
        self.assertEqual(m_res["total_masks"], 6)

        # 3. Verify .mask.jpg now exist
        for face in ["front", "right", "back", "left", "up", "down"]:
            mask_p = os.path.join(perspectives_dir, face, f"frame_001_{face}.mask.jpg")
            self.assertTrue(os.path.isfile(mask_p))
            # And no XMP sidecars for mask
            self.assertFalse(os.path.isfile(f"{mask_p}.xmp"))


if __name__ == "__main__":
    unittest.main()
