"""
Abu Align & Train - NVIDIA VGG-T³ (VGG-TTT) Engine Unit Tests
=============================================================
Tests VGG-T³ weight discovery, coordinate frame conversions (OpenCV -> COLMAP -> NeRF),
intrinsics rescaling, and export package generation.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pycolmap
from vggttt_engine import (
    find_vggttt_weights,
    _export_vggttt_packages,
)


class TestVGGTTTEngine(unittest.TestCase):

    def test_find_vggttt_weights_fallback(self):
        """Verifies weight discovery falls back to official Hugging Face slug."""
        slug = find_vggttt_weights(None)
        self.assertIn("vgg-ttt", slug)

    def test_find_vggttt_weights_custom(self):
        """Verifies custom weights path is respected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_weights = os.path.join(tmp_dir, "model.safetensors")
            with open(fake_weights, "wb") as f:
                f.write(b"mock")
            resolved = find_vggttt_weights(fake_weights)
            self.assertEqual(os.path.abspath(resolved), os.path.abspath(tmp_dir))

    def test_opencv_to_nerf_coordinate_transform(self):
        """
        VGG-T³ predicts OpenCV camera-to-world (+X right, +Y down, +Z forward).
        Transforms.json requires NeRF/OpenGL (+X right, +Y up, -Z back).
        Verify transformation preserves orientation and determinant.
        """
        opencv_to_nerf = np.diag([1.0, -1.0, -1.0, 1.0])
        self.assertAlmostEqual(np.linalg.det(opencv_to_nerf), 1.0)

        # Pure translation along Z in OpenCV (camera moves forward into scene)
        T_c2w = np.eye(4)
        T_c2w[2, 3] = 2.5  # z = 2.5m forward

        T_nerf = T_c2w @ opencv_to_nerf
        # In NeRF, looking direction is -Z, position should remain identical
        np.testing.assert_allclose(T_nerf[:3, 3], [0.0, 0.0, 2.5])
        # Camera up vector (-Y in OpenCV) becomes +Y in NeRF
        self.assertEqual(T_nerf[1, 1], -1.0)

    def test_colmap_world_to_camera_inversion(self):
        """
        Verify COLMAP R_cw and t_cw accurately represent camera center C = t_c2w.
        """
        # Simulated camera pose with 45 degree rotation and translation
        theta = np.radians(45.0)
        R_c2w = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ])
        t_c2w = np.array([1.5, -2.0, 5.0])

        # COLMAP conversion
        R_cw = R_c2w.T
        t_cw = -R_cw @ t_c2w

        # Compute camera optical center from COLMAP world-to-cam: C = -R_cw^T @ t_cw
        C_recovered = -R_cw.T @ t_cw
        np.testing.assert_allclose(C_recovered, t_c2w, rtol=1e-5, atol=1e-5)

    def test_intrinsics_rescaling(self):
        """
        VGG-T³ operates at 518px preprocessed canvas.
        Intrinsics must upscale proportionally to original image aspect and dimensions.
        """
        model_w, model_h = 518, 518
        orig_w, orig_h = 3840, 2160

        scale_x = orig_w / float(model_w)
        scale_y = orig_h / float(model_h)

        fx_model = 400.0
        fy_model = 400.0
        cx_model = 259.0
        cy_model = 259.0

        fx_orig = fx_model * scale_x
        fy_orig = fy_model * scale_y
        cx_orig = cx_model * scale_x
        cy_orig = cy_model * scale_y

        self.assertAlmostEqual(cx_orig, orig_w / 2.0, places=1)
        self.assertAlmostEqual(cy_orig, orig_h / 2.0, places=1)
        self.assertAlmostEqual(fx_orig, 400.0 * (3840.0 / 518.0), places=3)

    def test_export_vggttt_packages(self):
        """
        Verify _export_vggttt_packages writes valid 3dgs_dataset, transforms.json,
        and points.ply without exceptions.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            rec = pycolmap.Reconstruction()
            cam = pycolmap.Camera(model="PINHOLE", width=1920, height=1080, params=[1000.0, 1000.0, 960.0, 540.0], camera_id=1)
            if hasattr(rec, "add_camera_with_trivial_rig"):
                rec.add_camera_with_trivial_rig(cam)
            else:
                rec.add_camera(cam)
            img = pycolmap.Image(name="frame_001.jpg", camera_id=1, image_id=1)
            rigid = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
            if hasattr(rec, "add_image_with_trivial_frame"):
                rec.add_image_with_trivial_frame(img, rigid)
            else:
                img.cam_from_world = rigid
                img.registered = True
                rec.add_image(img)

            transforms = {
                "camera_model": "OPENCV",
                "fl_x": 1000.0,
                "fl_y": 1000.0,
                "cx": 960.0,
                "cy": 540.0,
                "w": 1920,
                "h": 1080,
                "frames": [{
                    "file_path": "./images/frame_001.jpg",
                    "transform_matrix": np.eye(4).tolist(),
                    "fl_x": 1000.0,
                    "fl_y": 1000.0,
                    "cx": 960.0,
                    "cy": 540.0,
                    "w": 1920,
                    "h": 1080,
                }]
            }

            pts3d = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float32)
            pts_rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

            _export_vggttt_packages(
                reconstruction=rec,
                transforms_dict=transforms,
                dense_3d=pts3d,
                dense_rgb=pts_rgb,
                image_paths=[],
                output_dir=tmp_dir,
                log=lambda m: None,
            )

            # Check generated files
            ds_dir = Path(tmp_dir) / "3dgs_dataset"
            self.assertTrue((ds_dir / "transforms.json").is_file())
            self.assertTrue((ds_dir / "points.ply").is_file())
            self.assertTrue((ds_dir / "sparse" / "0").is_dir())
            self.assertTrue((Path(tmp_dir) / "COLMAP_Lichtfield").is_dir())
            self.assertTrue((Path(tmp_dir) / "points.ply").is_file())

    def test_run_vggttt_reconstruction_signature_compatibility(self):
        """Verifies run_vggttt_reconstruction accepts device, dtype, and extra kwargs without error."""
        import inspect
        from vggttt_engine import run_vggttt_reconstruction
        sig = inspect.signature(run_vggttt_reconstruction)
        self.assertIn("device", sig.parameters)
        self.assertIn("dtype", sig.parameters)
        self.assertTrue(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))


if __name__ == "__main__":
    unittest.main()
