"""Unit tests for RealityScan photogrammetry and prior trajectory alignment engine."""
import os
import tempfile
import json
import unittest
from pathlib import Path

from realityscan_engine import (
    find_realityscan_executable,
    generate_trajectory_import_xml,
    find_prior_trajectory,
    convert_transforms_to_realityscan_csv,
)


class RealityScanEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_xml_matches_guid_and_prior_format(self):
        xml_path = self.dir / "import_def.xml"
        generate_trajectory_import_xml(str(xml_path))
        self.assertTrue(xml_path.is_file())
        content = xml_path.read_text(encoding="utf-8")
        self.assertIn("{35CD8B84-6573-417D-8FEE-BE8BBEEC00D3}", content)
        self.assertIn("<separator>,</separator>", content)
        self.assertIn("Global camera prior settings", content)
        self.assertIn("local:1 - Euclidean", content)
        self.assertIn("ZYX", content)

    def test_convert_transforms_to_realityscan_csv(self):
        transforms = {
            "camera_angle_x": 1.0,
            "fl_x": 1000.0,
            "fl_y": 1000.0,
            "cx": 960.0,
            "cy": 540.0,
            "w": 1920,
            "h": 1080,
            "frames": [
                {
                    "file_path": "images/frame_0001.jpg",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.5],
                        [0.0, 1.0, 0.0, 1.2],
                        [0.0, 0.0, 1.0, 3.4],
                        [0.0, 0.0, 0.0, 1.0]
                    ]
                }
            ]
        }
        json_path = self.dir / "transforms.json"
        json_path.write_text(json.dumps(transforms), encoding="utf-8")
        out_csv = self.dir / "cameras.csv"
        res = convert_transforms_to_realityscan_csv(str(json_path), str(out_csv))
        self.assertTrue(res)
        self.assertTrue(out_csv.is_file())
        csv_text = out_csv.read_text(encoding="utf-8")
        self.assertIn("frame_0001.jpg", csv_text)
        self.assertIn("0.5", csv_text)
        self.assertIn("1.2", csv_text)
        self.assertIn("3.4", csv_text)

    def test_find_prior_trajectory_prior_runs(self):
        res_dir = self.dir / "results"
        res_dir.mkdir()
        run1 = res_dir / "vggt_run1"
        run1.mkdir()
        csv_content = "# camera_name,x,y,z,yaw,pitch,roll\n" + "\n".join([f"frame_{i:04d}.jpg,1.0,2.0,3.0,0.0,0.0,0.0" for i in range(15)])
        (run1 / "cameras.csv").write_text(csv_content, encoding="utf-8")
        cand, r_dir = find_prior_trajectory(str(self.dir))
        self.assertIsNotNone(cand)
        self.assertTrue(cand.endswith("cameras.csv"))
        self.assertEqual(r_dir, str(run1))


if __name__ == "__main__":
    unittest.main()
