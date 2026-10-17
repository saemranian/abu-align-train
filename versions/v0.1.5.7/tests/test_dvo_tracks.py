import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch
from dvo_tracks import PersistentTracks, attach_tracks


class TestPersistentDVOTracks(unittest.TestCase):
    def test_strict_mode_rejects_unavailable_neural_backend(self):
        import dvo360_engine as engine
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = root/'a.jpg'
            Image.new('RGB', (64, 32)).save(image)
            with patch.object(engine.wsl_bridge, 'is_360dvo_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'Neural 360DVO failed'):
                    engine.run_360dvo_reconstruction([image], str(root/'run'), use_cache=False,
                                                   allow_fallback=False, log_callback=lambda _: None)

    def test_patch_identity_bounds_and_longitude_wrap(self):
        tracks = PersistentTracks(max_observations=2)
        tracks.record([4], [8], [2], [[1.1, .5]], [[-.1, .5]], [1], [[255, 0, 0]])
        tracks.record([4], [9], [2], [[.1, .5]], [[.2, .5]], [1], [[255, 0, 0]])
        self.assertEqual(tracks.count, 2)
        self.assertTrue(tracks.truncated)
        self.assertAlmostEqual(tracks.tracks['dvo_4_2']['observations'][8][0], .9)

    def test_final_pose_triangulation_and_original_pixel_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            point = np.array([.3, .2, 3.])
            def pixel(center):
                ray = point - center; ray /= np.linalg.norm(ray)
                return [.5 + np.arctan2(ray[0], ray[2])/(2*np.pi), .5 + np.arcsin(ray[1])/np.pi]
            tracks = PersistentTracks()
            tracks.record([0], [1], [0], [pixel(np.zeros(3))], [pixel(np.array([1, 0, 0]))], [1], [[100, 150, 200]])
            tracks.save(root/'tracks.json')
            model = [{'cameras': {'s': {'projection_type': 'spherical', 'width': 1000, 'height': 500}},
                      'shots': {'a.jpg': {'camera': 's', 'rotation': [0, 0, 0], 'translation': [0, 0, 0]},
                                'b.jpg': {'camera': 's', 'rotation': [0, 0, 0], 'translation': [-1, 0, 0]}}, 'points': {}}]
            (root/'reconstruction.json').write_text(json.dumps(model))
            report = attach_tracks(root/'reconstruction.json', root/'tracks.json', ['a.jpg', 'b.jpg'])
            self.assertEqual(report['observations'], 2)
            result = json.loads((root/'reconstruction.json').read_text())[0]['points']['dvo_0_0']
            np.testing.assert_allclose(result['coordinates'], point, atol=1e-10)
            np.testing.assert_allclose(result['observations'][0]['pixel'], np.array(pixel(np.zeros(3)))*[1000, 500])
