"""Geometry correctness and real Viser API integration; no reconstruction jobs."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viewer_workspace import InspectionWorkspace, camera_pairs, similarity, transform_cameras, cloud_distances
from fast_viewer import load_cameras_from_transforms, load_cameras_from_opensfm, read_ply_data


def sample():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=float)
    return dict(points=positions.copy(), colors=np.full((4, 3), 170, np.uint8),
        cameras=[dict(name=f'images/{i}.jpg', pos=p.copy(), wxyz=np.array([1., 0, 0, 0])) for i, p in enumerate(positions)],
        source='Synthetic fixture', asset='fixture.ply')


class GeometryTests(unittest.TestCase):
    def test_matching_export_prefixes_without_ambiguous_basenames(self):
        a = [dict(name='images/a.jpg'), dict(name='left/b.jpg'), dict(name='right/b.jpg')]
        b = [dict(name='C:/dataset/a.jpg'), dict(name='b.jpg')]
        self.assertEqual(camera_pairs(a, b), [(0, 0)])

    def test_directed_cloud_distances_and_empty_input(self):
        distances, count = cloud_distances(np.array([[0, 0, 0], [2, 0, 0]]), np.array([[0, 3, 0], [2, 0, 0]]))
        np.testing.assert_allclose(distances, [3, 0])
        self.assertEqual(count, 2)
        self.assertEqual(cloud_distances([], [1])[1], 0)

    def test_similarity_recovers_scale_rotation_translation(self):
        source = sample()['points']
        rotation = Rotation.from_euler('xyz', [30, -80, 130], degrees=True).as_matrix()
        target = 2.5*(source @ rotation.T)+[7, -3, 1]
        scale, actual, offset = similarity(source, target)
        np.testing.assert_allclose(scale*(source @ actual.T)+offset, target, atol=1e-10)
        transformed = transform_cameras(sample()['cameras'], scale, actual, offset)
        np.testing.assert_allclose(transformed[0]['pos'], target[0])
        q = transformed[0]['wxyz']
        np.testing.assert_allclose(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), rotation, atol=1e-10)

    def test_degenerate_registration_rejected(self):
        for points in (np.zeros((2, 3)), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])):
            with self.assertRaises(ValueError):
                similarity(points, points)

    def test_matching_ignores_duplicates_and_preserves_subfolders(self):
        a = [dict(name='a/image.jpg'), dict(name='b/image.jpg'), dict(name='duplicate'), dict(name='duplicate')]
        b = [dict(name='b\\image.jpg'), dict(name='a/image.jpg'), dict(name='duplicate')]
        self.assertEqual(set(camera_pairs(a, b)), {(0, 1), (1, 0)})

    def test_opengl_and_half_turn_camera_orientations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'transforms.json'
            path.write_text(json.dumps({'frames':[{'file_path':'./images/a.jpg', 'transform_matrix':np.eye(4).tolist()}]}))
            camera = load_cameras_from_transforms(str(path))[0]
            q = camera['wxyz']
            matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(matrix, np.diag([1, -1, -1]), atol=1e-7)
            self.assertEqual(camera['name'], 'images/a.jpg')
            path = Path(directory)/'reconstruction.json'
            path.write_text(json.dumps([{'shots':{'image.jpg':{'rotation':[0, 0, np.pi], 'translation':[0, 0, 0]}}}]))
            q = load_cameras_from_opensfm(str(path))[0]['wxyz']
            np.testing.assert_allclose(Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix(), np.diag([-1, -1, 1]), atol=1e-6)

    def test_truncated_ply_returns_without_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.ply'
            path.write_text('ply\nformat ascii 1.0\n')
            self.assertEqual(read_ply_data(str(path)), (None, None))


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        # Use the installed frontend, avoiding an unrelated process-discovery
        # race in Viser's development-server auto-builder on Windows.
        with patch('viser._client_autobuild._check_viser_yarn_running', return_value=False):
            self.server = viser.ViserServer(host='127.0.0.1', port=8896, verbose=False)
        self.addCleanup(self.server.stop)
        self.loader = Mock(side_effect=lambda path: sample())
        self.ui = InspectionWorkspace(self.server, {'Run A':'a', 'Run B':'b'}, 'Run A',
                                      'Fixture', self.loader, lambda: {'Run A':'a', 'Run B':'b'})

    def test_switch_and_color_preserve_viewpoint(self):
        client = SimpleNamespace(camera=SimpleNamespace(position=np.array([9.,8,7]), look_at=np.zeros(3), up_direction=np.array([0.,0,1])))
        self.server.get_clients = lambda: {1:client}
        self.ui.switch_comparison()
        self.assertEqual(self.ui.mode.value, 'B only')
        self.ui.switch_comparison()
        self.assertEqual(self.ui.mode.value, 'A only')
        self.ui.color.value = 'Layer color'
        self.ui.color_changed()
        self.assertEqual(self.ui.mode.value, 'A only')
        np.testing.assert_equal(client.camera.position, [9,8,7])
        self.assertFalse(self.ui.camera_thumbs.value)
        self.assertTrue(self.ui.open_photo_btn.disabled)

    def test_small_points_and_explicit_comparison_activation(self):
        self.ui.point_size.value = 0.0001
        self.ui.comparison_changed()
        self.assertEqual(self.ui.point_size.value, 0.0001)
        self.assertEqual(self.ui.mode.value, 'Overlay')
        self.assertEqual(len(self.ui.camera_options), 8)
        self.assertIn('RMS', self.ui.comparison.content)

    def test_photo_preview_does_not_launch_external_application(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            photo = Path(directory)/'test.jpg'
            Image.new('RGB', (40, 20)).save(photo)
            with patch.object(self.ui, '_resolve_photo', return_value=str(photo)), patch('os.startfile') as launch:
                self.ui.open_selected_photo()
                self.ui.open_selected_photo()
                launch.assert_not_called()
                self.assertIsNotNone(self.ui._photo_modal)

    def test_real_api_loads_and_keeps_camera_on_display_changes(self):
        self.assertIn('Ready', self.ui.status.content)
        client = SimpleNamespace(camera=SimpleNamespace(position=np.array([9, 8, 7]), look_at=np.zeros(3), up_direction=np.array([0, 0, 1])))
        self.server.get_clients = lambda: {1:client}
        self.ui.update()
        self.assertEqual(self.loader.call_count, 2)
        np.testing.assert_equal(client.camera.position, [9, 8, 7])
        self.ui.mode.value = 'Overlay'
        self.ui.registration.value = True
        self.ui.update()
        self.assertIn('Registered B', self.ui.comparison.content)
        self.assertEqual(len(self.ui.camera_options), 8)
        self.assertIn('RMS', self.ui.comparison.content)
        self.ui.color.value = 'B distance to A'
        self.ui.update()
        self.assertIn('Cloud distance', self.ui.comparison.content)

    def test_navigation_camera_lookthrough_and_saved_view(self):
        client = SimpleNamespace(client_id=1, camera=SimpleNamespace(position=np.array([9., 8, 7]), look_at=np.zeros(3), up_direction=np.array([0., 0, 1])))
        self.ui.set_view(client, 'Top')
        self.assertGreater(client.camera.position[2], client.camera.look_at[2])
        np.testing.assert_equal(client.camera.up_direction, [0, 1, 0])
        self.ui.camera_actions._impl.value = 'Look through'
        self.ui.camera_action(SimpleNamespace(client=client))
        self.assertGreater(client.camera.look_at[2], client.camera.position[2])
        np.testing.assert_equal(client.camera.up_direction, [0, -1, 0])
        self.ui.bookmark._impl.value = 'Save view'
        self.ui.saved_view(SimpleNamespace(client=client))
        saved = np.array(client.camera.position).copy()
        client.camera.position = np.array([100, 100, 100])
        self.ui.bookmark._impl.value = 'Restore view'
        self.ui.saved_view(SimpleNamespace(client=client))
        np.testing.assert_equal(client.camera.position, saved)

    def test_registration_failure_keeps_previous_scene(self):
        old = list(self.ui.handles)
        self.ui.cache['Run B']['cameras'] = []
        self.ui.registration.value = True
        self.ui.update()
        self.assertIn('Unable to apply', self.ui.status.content)
        self.assertEqual(self.ui.handles, old)

    def test_refresh_preserves_selected_runs(self):
        self.ui.refresh_runs(None)
        self.assertEqual(self.ui.a.value, 'Run A')
        self.assertEqual(self.ui.b.value, 'Run B')
        self.assertEqual(self.loader.call_count, 4)


if __name__ == '__main__':
    unittest.main()
