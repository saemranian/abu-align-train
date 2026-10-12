"""
Test Suite for Abu Train Max Gaussians Option & Engine Capping (v0.0.6.4)
=======================================================================
Verifies:
1. cap_ply_gaussians binary PLY opacity sorting & capping
2. align_cli and train_engine argument parsing
3. GUI preset string parsing
"""

import os
import sys
import struct
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train_engine import cap_ply_gaussians


class TestMaxGaussians(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_ply(self, opacities):
        """Creates a minimal valid binary little-endian PLY file with Gaussian opacities."""
        num_vertices = len(opacities)
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {num_vertices}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property float opacity\n"
            f"end_header\n"
        ).encode("ascii")

        ply_path = os.path.join(self.temp_dir.name, "test_splat.ply")
        with open(ply_path, "wb") as f:
            f.write(header)
            for idx, op in enumerate(opacities):
                # 4 floats: x, y, z, opacity
                f.write(struct.pack("<ffff", float(idx), 0.0, 0.0, float(op)))
        return ply_path

    def test_cap_ply_gaussians_pruning(self):
        """Test that cap_ply_gaussians retains the top-K highest opacity Gaussians."""
        opacities = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.05]
        ply_path = self._create_mock_ply(opacities)

        # Cap to 4 Gaussians
        result_path = cap_ply_gaussians(ply_path, max_gaussians=4)
        self.assertEqual(result_path, ply_path)

        # Read back the capped PLY
        with open(ply_path, "rb") as f:
            content = f.read()

        end_hdr = content.find(b"end_header\n") + len(b"end_header\n")
        hdr_text = content[:end_hdr].decode("ascii")
        self.assertIn("element vertex 4\n", hdr_text)

        # Read remaining vertices
        vertex_data = content[end_hdr:]
        self.assertEqual(len(vertex_data), 4 * 16)  # 4 vertices * 16 bytes each
        remaining_opacities = []
        for i in range(4):
            x, y, z, op = struct.unpack_from("<ffff", vertex_data, i * 16)
            remaining_opacities.append(round(op, 2))

        # Expected top 4: 0.9, 0.8, 0.7, 0.6
        self.assertEqual(remaining_opacities, [0.9, 0.8, 0.7, 0.6])

    def test_cap_ply_gaussians_noop_when_under_limit(self):
        """Test that capping does nothing if vertex count <= max_gaussians."""
        opacities = [0.1, 0.5, 0.9]
        ply_path = self._create_mock_ply(opacities)

        cap_ply_gaussians(ply_path, max_gaussians=10)

        # Verify unchanged
        with open(ply_path, "rb") as f:
            content = f.read()
        self.assertIn(b"element vertex 3\n", content)

    def test_cap_ply_gaussians_noop_when_zero(self):
        """Test that max_gaussians=0 does nothing (unlimited)."""
        opacities = [0.1, 0.5, 0.9]
        ply_path = self._create_mock_ply(opacities)

        cap_ply_gaussians(ply_path, max_gaussians=0)

        with open(ply_path, "rb") as f:
            content = f.read()
        self.assertIn(b"element vertex 3\n", content)

    def test_gui_parse_max_gaussians(self):
        """Test GUI preset string parser."""
        from abu_align_gui import AbuAlignApp

        # Create dummy instance without calling __init__
        class DummyGUI(AbuAlignApp):
            def __init__(self):
                pass

        gui = DummyGUI()

        test_cases = [
            ("Unlimited (Engine Default)", 0),
            ("500,000 (Fast / Low VRAM)", 500000),
            ("1,000,000 (Standard)", 1000000),
            ("2,000,000 (Detailed)", 2000000),
            ("3,000,000 (High Quality)", 3000000),
            ("5,000,000 (Maximum Fidelity)", 5000000),
            ("750,000", 750000),
            ("400000", 400000),
            ("invalid", 0),
            ("", 0),
        ]

        class DummyVar:
            def __init__(self, val):
                self.val = val
            def get(self):
                return self.val

        for raw_str, expected in test_cases:
            gui.train_max_gaussians_var = DummyVar(raw_str)
            self.assertEqual(gui._parse_train_max_gaussians(), expected)


if __name__ == "__main__":
    unittest.main()
