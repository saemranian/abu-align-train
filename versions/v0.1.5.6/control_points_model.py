"""Validated, undoable manual-control documents and spherical point diagnostics."""
import copy
import json
import math
from pathlib import Path


def validate_controls(payload):
    if not isinstance(payload, dict) or payload.get('schema_version') != 1 or not isinstance(payload.get('points'), list):
        raise ValueError('Choose an Abu control-points file (schema version 1).')
    data = copy.deepcopy(payload)
    if 'image_policy' in data:
        from manual_image_policy import validate_policy
        data['image_policy']=validate_policy(data['image_policy'])
    data.setdefault('distances', [])
    if not isinstance(data['distances'], list): raise ValueError('Distances must be a list.')
    names = set()
    for point in data['points']:
        if not isinstance(point, dict): raise ValueError('Invalid point record.')
        name = point.get('id')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Point names must be unique and non-empty.')
        names.add(name)
        point.setdefault('type', 'tie')
        if 'enabled' in point and not isinstance(point['enabled'], bool): raise ValueError('Enabled must be true or false.')
        if point.get('type', 'tie') not in {'tie', 'gcp', 'check'}:
            raise ValueError(f'Unknown point type: {name}')
        if point.get('type', 'tie') != 'tie':
            xyz = point.get('position', [])
            if len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
                raise ValueError(f'{name}: enter finite local X, Y and Z coordinates.')
            point['position'] = [float(v) for v in xyz]
        for key in ['sigma_m']:
            if key in point and (not math.isfinite(float(point[key])) or float(point[key]) <= 0):
                raise ValueError(f'{name}: uncertainty must be positive.')
            if key in point: point[key] = float(point[key])
        shots = set()
        if not isinstance(point.get('observations', []), list): raise ValueError('Invalid measurements.')
        point.setdefault('observations', [])
        for observation in point['observations']:
            if not isinstance(observation, dict): raise ValueError('Invalid measurement record.')
            shot = observation.get('shot_id')
            pixel = observation.get('pixel', [])
            sigma = float(observation.get('sigma_px', 1))
            if not isinstance(shot, str) or not shot or shot in shots or len(pixel) != 2:
                raise ValueError(f'{name}: each image can have one measurement.')
            if not all(math.isfinite(float(v)) for v in pixel) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError(f'{name}: invalid pixel or measurement uncertainty.')
            shots.add(shot)
            observation.update(pixel=[float(v) for v in pixel], sigma_px=sigma)
    for distance in data['distances']:
        if not isinstance(distance, dict): raise ValueError('Invalid distance record.')
        if distance.get('a') not in names or distance.get('b') not in names or distance['a'] == distance['b']:
            raise ValueError('A distance needs two different existing points.')
        for key, default in [('meters', 0), ('sigma_m', .01)]:
            value = float(distance.get(key, default))
            if not math.isfinite(value) or value <= 0: raise ValueError('Distance and uncertainty must be positive.')
            distance[key] = value
    return data


class ControlDocument:
    def __init__(self):
        self.data = {'schema_version': 1, 'points': [], 'distances': []}
        self.undo_stack = []
        self.redo_stack = []
        self.saved = self.signature()

    def signature(self): return json.dumps(self.data, sort_keys=True)
    @property
    def dirty(self): return self.signature() != self.saved
    def mark_saved(self): self.saved = self.signature()
    def checkpoint(self):
        self.undo_stack.append(copy.deepcopy(self.data))
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()
    def undo(self):
        if not self.undo_stack: return False
        self.redo_stack.append(copy.deepcopy(self.data)); self.data = self.undo_stack.pop(); return True
    def redo(self):
        if not self.redo_stack: return False
        self.undo_stack.append(copy.deepcopy(self.data)); self.data = self.redo_stack.pop(); return True
    def point(self, name): return next((p for p in self.data['points'] if p['id'] == name), None)
    def next_name(self):
        index = 1
        while self.point(f'CP{index:03}'): index += 1
        return f'CP{index:03}'
    def rename(self, old, new):
        new = new.strip()
        if not new or (new != old and self.point(new)): raise ValueError('Choose a unique, non-empty name.')
        self.point(old)['id'] = new
        for distance in self.data['distances']:
            for key in ['a', 'b']:
                if distance[key] == old: distance[key] = new
    def delete(self, name):
        self.data['points'] = [p for p in self.data['points'] if p['id'] != name]
        self.data['distances'] = [d for d in self.data['distances'] if name not in [d['a'], d['b']]]
    def measure(self, name, shot, pixel, sigma=1):
        point = self.point(name)
        point['observations'] = [o for o in point['observations'] if o['shot_id'] != shot]
        if pixel is not None:
            point['observations'].append({'shot_id': shot, 'pixel': list(pixel), 'sigma_px': float(sigma)})


def atomic_json(path, data, indent=2):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(data, indent=indent), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def point_geometry(point, cameras):
    """Triangulated guidance only; never treat a prediction as a measurement."""
    import numpy as np
    from spherical_refinement import triangulate
    from manual_camera_geometry import bearing
    observations = [(o['shot_id'], o['pixel'], o.get('sigma_px', 1)) for o in point['observations']]
    if len(observations) < 2: return None, 'Needs at least 2 images', {}
    try:
        xyz = triangulate(observations, cameras)
        errors = {}
        for name, pixel, _ in observations:
            rotation, translation, width, height = cameras[name]
            ray = rotation @ xyz + translation; ray /= np.linalg.norm(ray)
            errors[name] = float(np.degrees(np.arccos(np.clip(ray @ bearing(pixel,cameras[name]), -1, 1))))
        rms = math.sqrt(sum(v*v for v in errors.values()) / len(errors))
        return xyz, f'Angular RMS {rms:.4f}°', errors
    except (ValueError, KeyError, np.linalg.LinAlgError) as error:
        return None, str(error).replace('Manual ', ''), {}


def projected_pixel(xyz, camera):
    from manual_camera_geometry import project
    return project(xyz,camera)
