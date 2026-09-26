"""
Abu Align & Train - Engine Repository Tracker Unit Tests
Tests registry completeness, local git info extraction, report formatting,
and non-blocking evaluation across all 10 photogrammetry and 3DGS engines.
"""

import unittest
from pathlib import Path
from engine_tracker import (
    ENGINE_REGISTRY,
    get_local_git_info,
    check_internet_connectivity,
    evaluate_engine,
    generate_report_text,
    check_all_engines,
)


class TestEngineTracker(unittest.TestCase):

    def test_registry_contains_all_10_engines(self):
        """Ensure all 10 alignment, training, and foundation engines are registered."""
        self.assertEqual(len(ENGINE_REGISTRY), 10)
        expected_ids = {
            "vggt_omega", "ace0", "3dgrut", "pfgs360",
            "pycusfm", "opensfm", "360dvo", "lichtfeld",
            "nerfstudio", "vggt_base"
        }
        actual_ids = {eng["id"] for eng in ENGINE_REGISTRY}
        self.assertEqual(actual_ids, expected_ids)

    def test_local_git_detection(self):
        """Test local git inspection on an existing repository directory."""
        root_dir = Path(__file__).resolve().parent.parent
        info = get_local_git_info([root_dir])
        self.assertTrue(info["found"])
        self.assertTrue(info["is_git"])
        self.assertIsNotNone(info["sha"])
        self.assertEqual(len(info["short_sha"]), 7)

    def test_evaluate_engine_offline_safe(self):
        """Test evaluate_engine runs gracefully without internet access."""
        eng_def = ENGINE_REGISTRY[0]  # vggt_omega
        res = evaluate_engine(eng_def, has_internet=False)
        self.assertEqual(res["id"], "vggt_omega")
        self.assertIn(res["status"], ["offline", "not_installed"])
        self.assertIn("status_label", res)

    def test_evaluate_native_engine(self):
        """Test that internal native engines (like 360DVO) resolve to native status."""
        dvo_eng = next(e for e in ENGINE_REGISTRY if e["id"] == "360dvo")
        res = evaluate_engine(dvo_eng, has_internet=True)
        self.assertEqual(res["status"], "integrated")
        self.assertIn("Native Engine", res["status_label"])

    def test_generate_report_formatting(self):
        """Test that generate_report_text produces clean ASCII tables with all engines."""
        mock_data = {
            "timestamp": "2026-09-26 12:00:00",
            "has_internet": True,
            "update_count": 1,
            "total_engines": len(ENGINE_REGISTRY),
            "engines": [
                {
                    "id": "vggt_omega",
                    "name": "VGGT-Omega",
                    "category": "Alignment Engine",
                    "role": "Neural Foundation Transformer",
                    "repo_slug": "facebookresearch/vggt-omega",
                    "repo_url": "https://github.com/facebookresearch/vggt-omega.git",
                    "local": {"found": True, "is_git": True, "short_sha": "48b23c8"},
                    "remote": {"short_sha": "48b23c8"},
                    "status": "up_to_date",
                    "status_label": "[OK] Up to Date"
                },
                {
                    "id": "3dgrut",
                    "name": "NVIDIA 3DGRUT",
                    "category": "3DGS Training Engine",
                    "role": "Ray Traced Gaussian Splatting",
                    "repo_slug": "nv-tlabs/3dgrut",
                    "repo_url": "https://github.com/nv-tlabs/3dgrut.git",
                    "local": {"found": True, "is_git": True, "short_sha": "7397cc9", "date": "2026-09-17", "message": "fix"},
                    "remote": {"short_sha": "e6f4552", "date": "2026-09-22", "message": "update", "url": "https://github.com/..."},
                    "status": "update_available",
                    "status_label": "[UPDATE] Available"
                }
            ]
        }
        report = generate_report_text(mock_data)
        self.assertIn("ABU ALIGN & TRAIN - ENGINE REPOSITORY VERSION REPORT", report)
        self.assertIn("VGGT-Omega", report)
        self.assertIn("NVIDIA 3DGRUT", report)
        self.assertIn("[UPDATE] Available", report)
        self.assertIn("[!] AVAILABLE ENGINE UPDATES ON GITHUB:", report)


if __name__ == "__main__":
    unittest.main()
