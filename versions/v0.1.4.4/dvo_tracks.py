"""Bounded persistent neural patch measurements for native spherical BA."""
import json
from pathlib import Path
import numpy as np


class PersistentTracks:
    def __init__(self, max_observations=500000):
        self.tracks = {}
        self.count = 0
        self.limit = max_observations
        self.truncated = False

    def record(self, owners, targets, slots, source, target, weights, colors):
        for owner, destination, slot, src, dst, confidence, color in zip(
                owners, targets, slots, source, target, weights, colors):
            if owner == destination or confidence < .1 or not np.isfinite([*src, *dst, confidence]).all():
                continue
            if not 0 <= src[1] < 1 or not 0 <= dst[1] < 1:
                continue
            key = f'dvo_{int(owner)}_{int(slot)}'
            if key not in self.tracks:
                if self.count + 2 > self.limit:
                    self.truncated = True
                    continue
                self.tracks[key] = {'color': np.asarray(color).tolist(), 'observations': {}}
            observations = self.tracks[key]['observations']
            for frame, pixel in [(int(owner), src), (int(destination), dst)]:
                if frame not in observations and self.count >= self.limit:
                    self.truncated = True
                    continue
                self.count += frame not in observations
                observations[frame] = [float(pixel[0] % 1), float(pixel[1])]

    def save(self, path):
        Path(path).write_text(json.dumps({'schema_version': 1, 'coordinates': 'normalized_erp',
                                         'truncated': self.truncated, 'tracks': self.tracks}), encoding='utf-8')


def attach_tracks(reconstruction_path, tracks_path, image_names, max_error_deg=2.0):
    """Triangulate against the final trajectory, never stale window point positions."""
    from scipy.spatial.transform import Rotation
    from spherical_refinement import triangulate, pixel_bearing
    path = Path(reconstruction_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if len(data) != 1: raise ValueError('DVO track attachment requires one component')
    component = data[0]
    payload = json.loads(Path(tracks_path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1 or payload.get('coordinates') != 'normalized_erp':
        raise ValueError('Unsupported DVO track coordinates')
    cameras = {}
    for name, shot in component['shots'].items():
        camera = component['cameras'][shot['camera']]
        cameras[name] = (Rotation.from_rotvec(shot['rotation']).as_matrix(),
                         np.asarray(shot['translation']), camera['width'], camera['height'])
    added = observations_count = 0
    for key, track in payload['tracks'].items():
        measurements = []
        for index, pixel in track['observations'].items():
            index = int(index)
            if not 0 <= index < len(image_names): continue
            name = image_names[index]
            if name not in cameras: continue
            width, height = cameras[name][2:]
            measurements.append((name, [pixel[0]*width, pixel[1]*height], 1.0))
        try:
            xyz = triangulate(measurements, cameras)
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = []
        for name, pixel, _ in measurements:
            rotation, translation, width, height = cameras[name]
            ray = rotation @ xyz + translation
            ray /= np.linalg.norm(ray)
            errors.append(np.degrees(np.arccos(np.clip(ray @ pixel_bearing(pixel, width, height), -1, 1))))
        if not np.isfinite(errors).all() or max(errors) > max_error_deg: continue
        color = np.asarray(track['color'])
        if color.max() <= 1: color = color * 255
        component['points'][key] = {'coordinates': xyz.tolist(), 'color': np.clip(color, 0, 255).astype(int).tolist(),
                                   'observations': [{'shot_id': name, 'pixel': pixel} for name, pixel, _ in measurements]}
        added += 1; observations_count += len(measurements)
    discarded = 0
    if added:
        loose = [key for key, point in component['points'].items() if not point.get('observations')]
        discarded = len(loose)
        for key in loose: del component['points'][key]
    path.write_text(json.dumps(data), encoding='utf-8')
    return {'landmarks': added, 'observations': observations_count, 'truncated': payload['truncated'],
            'max_angular_error_deg': max_error_deg, 'untracked_cloud_points_excluded_from_ba_model': discarded}
