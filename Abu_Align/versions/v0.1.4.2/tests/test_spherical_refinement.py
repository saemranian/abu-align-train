import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spherical_refinement import refine, pixel_bearing
from benchmark_360 import evaluate, spherical_psnr


class TestSphericalRefinement(unittest.TestCase):
    def test_joint_refinement_reduces_independent_error(self):
        rng = np.random.default_rng(5)
        width, height = 1024, 512
        points = rng.uniform([-1, -.7, 3], [1, .7, 5], (18, 3))
        shots = {f'{i:03}.jpg': {'camera': 'pano', 'rotation': [0, 0, 0],
                                'translation': [-i * .3, 0, 0]} for i in range(4)}
        component = {'cameras': {'pano': {'projection_type': 'spherical', 'width': width, 'height': height}},
                     'shots': shots, 'points': {}}
        for i, xyz in enumerate(points):
            observations = []
            for name, shot in shots.items():
                ray = xyz + np.asarray(shot['translation'])
                ray /= np.linalg.norm(ray)
                u = (np.arctan2(ray[0], ray[2]) / (2*np.pi) + .5) * width
                v = (.5 + np.arcsin(ray[1]) / np.pi) * height
                observations.append({'shot_id': name, 'pixel': [u, v]})
            component['points'][str(i)] = {'coordinates': xyz.tolist(), 'color': [255, 0, 0], 'observations': observations}
        noisy = copy.deepcopy(component)
        noisy['shots']['001.jpg']['rotation'] = [0, .035, .012]
        noisy['shots']['001.jpg']['translation'][1] += .04
        for point in noisy['points'].values():
            point['coordinates'] = (np.array(point['coordinates']) + rng.normal(0, .03, 3)).tolist()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'truth.json').write_text(json.dumps([component]))
            (root/'noisy.json').write_text(json.dumps([noisy]))
            before = evaluate(root/'noisy.json', root/'truth.json', 'none')
            report = refine(root/'noisy.json', root/'refined.json', max_iterations=80)
            after = evaluate(root/'refined.json', root/'truth.json', 'none')
            self.assertLess(after['ate_rmse'], before['ate_rmse'] / 5)
            self.assertLess(report['after_angular_rms_deg'], report['before_angular_rms_deg'] / 10)
            self.assertEqual(json.loads((root/'noisy.json').read_text()), [noisy])
            self.assertEqual(report['observations'], 72)
            controls = {'points': [{'id': str(i), 'type': 'gcp', 'position': points[i].tolist(),
                                    'observations': component['points'][str(i)]['observations']} for i in [0, 4, 12]]}
            controls['points'].append({'id': '17', 'type': 'check', 'position': points[17].tolist(),
                                       'observations': component['points']['17']['observations']})
            (root/'controls.json').write_text(json.dumps(controls))
            controlled = refine(root/'noisy.json', root/'controlled.json', root/'controls.json', max_iterations=100)
            self.assertLess(controlled['checks'][0]['error_units'], .01)
            self.assertEqual(controlled['observations'], 68)

    def test_spherical_seam_and_metric_weighting(self):
        np.testing.assert_allclose(pixel_bearing([0, 256], 1024, 512), pixel_bearing([1024, 256], 1024, 512), atol=1e-12)
        truth = np.zeros((64, 128, 3))
        pole = truth.copy(); pole[0] = 1
        equator = truth.copy(); equator[32] = 1
        self.assertGreater(spherical_psnr(pole, truth), spherical_psnr(equator, truth))
