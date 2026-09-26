import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from acezero import dataset_io

class TestACEZeroUnicode(unittest.TestCase):
    def test_write_pose_unicode_path(self):
        """Verify that paths containing unicode characters (e.g. \u2026 ellipsis) can be written and read without errors."""
        unicode_path = "Orbiting_object_product_promotio\u2026_20260922095106 frames/frame_00001.jpg"
        pose_34 = np.eye(4)[:3, :]
        focal_length = 500.0
        confidence = 100.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            pose_file = Path(tmp_dir) / "test_pose.txt"
            with open(pose_file, "w", encoding="utf-8") as f:
                dataset_io.write_pose_to_pose_file(
                    f,
                    rgb_file=unicode_path,
                    pose=pose_34,
                    confidence=confidence,
                    focal_length=focal_length
                )

            self.assertTrue(pose_file.exists())
            rgb_files, poses, focal_lengths, kept = dataset_io.load_dataset_ace(pose_file, confidence_threshold=0.0)
            self.assertEqual(len(rgb_files), 1)
            self.assertIn("\u2026", rgb_files[0])
            self.assertEqual(focal_lengths[0], 500.0)

    def test_write_pose_ascii_file_fallback(self):
        """Verify that if a stream is opened without utf-8 encoding, fallback works without crashing."""
        unicode_path = "Orbiting_object_product_promotio\u2026_20260922095106 frames/frame_00001.jpg"
        pose_34 = np.eye(4)[:3, :]
        focal_length = 500.0
        confidence = 100.0

        # StringIO or buffer with ascii encoding
        class MockAsciiStream:
            def __init__(self):
                self.content = []
            def write(self, s):
                s.encode('ascii') # will raise UnicodeEncodeError if non-ascii
                self.content.append(s)

        stream = MockAsciiStream()
        # Should not crash thanks to the defensive fallback in write_pose_to_pose_file
        dataset_io.write_pose_to_pose_file(
            stream,
            rgb_file=unicode_path,
            pose=pose_34,
            confidence=confidence,
            focal_length=focal_length
        )
        self.assertEqual(len(stream.content), 1)

    def test_ace_zero_util_run_cmd(self):
        """Verify that ace_zero_util.run_cmd executes without NameError: name 'os' is not defined."""
        from acezero import ace_zero_util
        ret = ace_zero_util.run_cmd(["python", "-c", "import sys; sys.exit(0)"], verbose=False)
        self.assertEqual(ret, 0)

if __name__ == "__main__":
    unittest.main()
