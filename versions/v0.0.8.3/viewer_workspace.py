"""Navigation and honest A/B inspection for the Viser reconstruction workspace."""
from collections import OrderedDict
from threading import RLock
import os
import numpy as np
from scipy.spatial.transform import Rotation


MINT = (112, 219, 192)
CORAL = (255, 155, 127)


def camera_pairs(a, b):
    """Match unique, normalized relative image names; never guess by list order."""
    def index(cameras):
        names, duplicates = {}, set()
        for i, camera in enumerate(cameras):
            name = camera['name'].replace('\\', '/').removeprefix('./')
            if name in names:
                duplicates.add(name)
            names[name] = i
        return {name: i for name, i in names.items() if name not in duplicates}
    left, right = index(a), index(b)
    return [(left[name], right[name]) for name in left.keys() & right.keys()]


def similarity(source, target):
    """Least-squares, positive-scale similarity mapping source onto target."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Registration needs at least three finite matched cameras.')
    x, y = source-source.mean(0), target-target.mean(0)
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError('Matched cameras are collinear; registration is ambiguous.')
    u, values, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(values * signs) / np.mean(np.sum(x*x, axis=1)))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Unable to determine a valid registration scale.')
    return scale, rotation, target.mean(0)-scale*(rotation @ source.mean(0))


def transform_cameras(cameras, scale, rotation, translation):
    result = []
    for camera in cameras:
        q = np.asarray(camera['wxyz'])
        matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        xyzw = Rotation.from_matrix(rotation @ matrix).as_quat()
        result.append(dict(camera, pos=scale*(rotation @ camera['pos'])+translation,
                           wxyz=xyzw[[3, 0, 1, 2]]))
    return result


def scene_bounds(points):
    valid = [np.asarray(p) for p in points if len(p)]
    if not valid:
        return np.zeros(3), 1.0
    coords = np.concatenate(valid)
    low, high = coords.min(0), coords.max(0)
    return (low+high)*0.5, max(float(np.linalg.norm(high-low)), 0.01)


class InspectionWorkspace:
    """Two cached source layers; navigation never changes source reconstruction data."""
    def __init__(self, server, runs, selected, title, loader, discover):
        self.server, self.runs = server, dict(runs)
        self.loader, self.discover = loader, discover
        self.lock = RLock()
        self.cache = OrderedDict()
        self.handles, self.layers, self.bookmarks = [], {}, {}
        self.selection_handle = None
        self.center, self.span = np.zeros(3), 1.0
        self.camera_options = {}
        gui = server.gui
        gui.add_markdown(f'## Reconstruction workspace\n{title}\n\nInspect · Navigate · Compare')
        self.status = gui.add_markdown('Loading reconstruction…')
        with gui.add_folder('Navigate'):
            self.views = gui.add_button_group('View', ('Fit', 'Top', 'Front', 'Right'))
            self.zoom = gui.add_button_group('Distance', ('Closer', 'Farther'))
            self.bookmark = gui.add_button_group('Viewpoint', ('Save view', 'Restore view'))
            gui.add_markdown('**Orbit** drag · **Pan** right-drag · **Zoom** scroll\n\n'
                             'Use Fit to recover the scene. Display changes keep your viewpoint.')
        with gui.add_folder('Compare reconstructions'):
            names = list(runs)
            self.a = gui.add_dropdown('A · Reference', names, initial_value=selected)
            self.b = gui.add_dropdown('B · Comparison', names,
                initial_value=next((n for n in names if n != selected), selected), disabled=len(names)<2)
            self.mode = gui.add_dropdown('Show', ('A only', 'B only', 'Overlay'), initial_value='A only', disabled=len(names)<2)
            self.registration = gui.add_checkbox('Register B to A using cameras', False,
                hint='Fits rotation, translation and scale from identical image names. Changes the preview only.')
            self.links = gui.add_checkbox('Show camera differences', True)
            gui.add_markdown('**A · mint** / **B · coral** in Overlay. Switch A/B to compare from the same viewpoint.\n\n'
                             'Raw coordinates are used unless registration is enabled. Independent runs may have different origins and scales.')
            self.comparison = gui.add_markdown('Choose a different run for B to compare.')
            self.refresh = gui.add_button('Refresh available runs')
        with gui.add_folder('Scene layers', expand_by_default=False):
            self.points = gui.add_checkbox('Point clouds', True)
            self.cameras = gui.add_checkbox('Camera frustums', True)
            self.paths = gui.add_checkbox('Camera paths', True)
            self.grid = gui.add_checkbox('Ground grid', False)
            self.point_size = gui.add_slider('Point size', 0.001, 0.03, 0.001, 0.004)
            self.camera_size = gui.add_slider('Camera size', 0.2, 3.0, 0.1, 1.0)
            self.color = gui.add_dropdown('Point color', ('Original', 'Layer color'), initial_value='Original')
            self.budget = gui.add_dropdown('Points per run', ('100,000', '300,000', '1,000,000'), initial_value='300,000')
            self.trim = gui.add_checkbox('Hide outer 1% of points', False,
                hint='Preview only. Off preserves all finite points before display sampling.')
        with gui.add_folder('Inspect a camera', expand_by_default=False):
            self.camera = gui.add_dropdown('Camera', ('No cameras',))
            self.camera_actions = gui.add_button_group('Selection', ('Previous', 'Next', 'Focus', 'Look through'))
            self.camera_info = gui.add_markdown('Click a camera frustum or choose an image above. Frustum optics are illustrative.')
        with gui.add_folder('Run details', expand_by_default=False):
            self.details = gui.add_markdown('')

        for control in (self.a, self.b, self.mode, self.registration, self.links, self.points,
                        self.cameras, self.paths, self.grid, self.point_size, self.camera_size,
                        self.color, self.budget, self.trim):
            control.on_update(self.update)
        self.views.on_click(self.navigate)
        self.zoom.on_click(self.zoom_view)
        self.bookmark.on_click(self.saved_view)
        self.refresh.on_click(self.refresh_runs)
        self.camera.on_update(self.inspect_camera)
        self.camera_actions.on_click(self.camera_action)
        server.on_client_connect(lambda client: self.set_view(client, 'Fit'))
        self.update()

    def clients(self, event):
        return [event.client] if getattr(event, 'client', None) else list(self.server.get_clients().values())

    def load(self, name):
        if name not in self.cache:
            data = self.loader(self.runs[name])
            points, colors = data['points'], data['colors']
            if points is None:
                points, colors = np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
            mask = np.isfinite(points).all(axis=1)
            data['points'] = points[mask]
            data['colors'] = colors[mask] if colors is not None else np.full((mask.sum(), 3), 180, np.uint8)
            data['cameras'] = [c for c in data['cameras'] if np.isfinite(c['pos']).all()
                               and np.isfinite(c['wxyz']).all() and np.linalg.norm(c['wxyz']) > 0]
            self.cache[name] = data
        self.cache.move_to_end(name)
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return self.cache[name]

    def update(self, event=None):
        with self.lock:
            try:
                a = self.load(self.a.value)
                b = self.load(self.b.value) if self.b.value != self.a.value else None
                if b is None and self.mode.value != 'A only':
                    raise ValueError('Choose two different runs for B only or Overlay.')
                layers = {'A': dict(a)}
                summary = 'Choose a different run for B to compare.'
                if b is not None:
                    b = dict(b)
                    pairs = camera_pairs(a['cameras'], b['cameras'])
                    left = np.array([a['cameras'][i]['pos'] for i, _ in pairs])
                    right = np.array([b['cameras'][j]['pos'] for _, j in pairs])
                    registration = 'Raw coordinates'
                    if self.registration.value:
                        scale, rotation, translation = similarity(right, left)
                        b['points'] = (scale*(b['points'] @ rotation.T)+translation).astype(np.float32)
                        b['cameras'] = transform_cameras(b['cameras'], scale, rotation, translation)
                        right = scale*(right @ rotation.T)+translation
                        registration = f'Registered B → A · scale {scale:.4g}'
                    summary = f'**{registration}**\n\n{len(pairs)} uniquely matched image names'
                    if pairs:
                        errors = np.linalg.norm(left-right, axis=1)
                        summary += (f'\n\nCamera-center difference · median **{np.median(errors):.4g}** · '
                                    f'RMS **{np.sqrt(np.mean(errors**2)):.4g}** scene units.\n\n'
                                    'Agreement between runs, not ground-truth accuracy. Registration residuals use the fitted cameras.')
                    else:
                        summary += '\n\nNo camera differences available. Names must match exactly, including relative paths.'
                    layers['B'] = b
                self.layers = layers
                self.comparison.content = summary
                self.render()
                count = sum(len(layer['points']) for layer in layers.values())
                self.status.content = f'**{self.mode.value}** · {count:,} finite source points\n\n' + (
                    'No geometry found. Select a completed reconstruction or refresh.' if not count and not any(l['cameras'] for l in layers.values())
                    else 'Ready · display sampling leaves source files unchanged.')
            except Exception as error:
                self.status.content = f'**Unable to apply selection**\n\n{error}\n\nPrevious scene retained. Change the selection or disable registration.'

    def render(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.camera_options.clear()
        bounds, details = [], []
        for key, layer in self.layers.items():
            visible = self.mode.value == 'Overlay' or self.mode.value.startswith(key)
            tint = MINT if key == 'A' else CORAL
            points, colors, cameras = layer['points'], layer['colors'], layer['cameras']
            if self.trim.value and len(points):
                distance = np.linalg.norm(points-np.median(points, axis=0), axis=1)
                mask = distance <= np.percentile(distance, 99)
                points, colors = points[mask], colors[mask]
            limit = int(self.budget.value.replace(',', ''))
            indices = np.linspace(0, max(0, len(points)-1), min(limit, len(points)), dtype=int)
            sampled = points[indices]
            positions = np.array([c['pos'] for c in cameras]).reshape(-1, 3)
            if visible:
                bounds.extend([sampled, positions])
            _, span = scene_bounds([sampled, positions])
            if len(sampled):
                self.handles.append(self.server.scene.add_point_cloud(f'/inspection/{key}/points', sampled,
                    tint if self.mode.value == 'Overlay' or self.color.value == 'Layer color' else colors[indices],
                    point_size=self.point_size.value, point_shape='circle', visible=visible and self.points.value))
            if len(positions) > 1:
                self.handles.append(self.server.scene.add_line_segments(f'/inspection/{key}/path',
                    np.stack((positions[:-1], positions[1:]), axis=1), tint, line_width=2,
                    visible=visible and self.paths.value))
            step = max(1, int(np.ceil(len(cameras)/250)))
            for i, camera in enumerate(cameras):
                name = f'{key} · {i+1:04d} · {camera["name"]}'
                if visible:
                    self.camera_options[name] = (key, i)
                if i % step:
                    continue
                handle = self.server.scene.add_camera_frustum(f'/inspection/{key}/camera/{i}',
                    fov=np.radians(60), aspect=1.5, scale=span*.018*self.camera_size.value,
                    color=tint, position=camera['pos'], wxyz=camera['wxyz'], visible=visible and self.cameras.value)
                handle.on_click(lambda event, value=name: self.select_camera(value))
                self.handles.append(handle)
            details.append(f'**{key} · {self.a.value if key == "A" else self.b.value}**\n\n'
                           f'{len(sampled):,} displayed / {len(layer["points"]):,} finite points · '
                           f'{len(cameras)} cameras · up to 250 frustums\n\n'
                           f'Camera source: {layer["source"]}\n\nCloud: `{layer["asset"]}`')
        self.center, self.span = scene_bounds(bounds)
        self.handles.append(self.server.scene.add_grid('/inspection/grid', width=self.span*2, height=self.span*2,
            cell_size=self.span/20, section_size=self.span/4, plane='xy', position=(self.center[0], self.center[1], self.center[2]-self.span/2),
            cell_color=(48, 60, 65), section_color=(70, 86, 92), visible=self.grid.value))
        if self.links.value and self.mode.value == 'Overlay' and 'B' in self.layers:
            a, b = self.layers['A']['cameras'], self.layers['B']['cameras']
            pairs = camera_pairs(a, b)
            if pairs:
                self.handles.append(self.server.scene.add_line_segments('/inspection/differences',
                    np.array([[a[i]['pos'], b[j]['pos']] for i, j in pairs]), (230, 203, 126), line_width=1.5))
        old = self.camera.value
        self.camera.options = tuple(self.camera_options) or ('No cameras',)
        self.camera.value = old if old in self.camera_options else self.camera.options[0]
        self.camera.disabled = not bool(self.camera_options)
        self.camera_actions.disabled = not bool(self.camera_options)
        self.inspect_camera()
        self.details.content = '\n\n---\n\n'.join(details)

    def set_view(self, client, name):
        directions = {'Fit': (1, -1.5, 1), 'Top': (0, 0, 1), 'Front': (0, -1, 0), 'Right': (1, 0, 0)}
        direction = np.array(directions[name], dtype=float)
        direction /= np.linalg.norm(direction)
        client.camera.up_direction = (0, 1, 0) if name == 'Top' else (0, 0, 1)
        client.camera.position = self.center + direction*self.span*1.5
        client.camera.look_at = self.center

    def navigate(self, event):
        for client in self.clients(event):
            self.set_view(client, self.views.value)

    def zoom_view(self, event):
        for client in self.clients(event):
            camera = client.camera
            factor = .8 if self.zoom.value == 'Closer' else 1.25
            camera.position = np.asarray(camera.look_at)+(np.asarray(camera.position)-camera.look_at)*factor

    def saved_view(self, event):
        for client in self.clients(event):
            if self.bookmark.value == 'Save view':
                self.bookmarks[client.client_id] = tuple(np.array(v).copy() for v in
                    (client.camera.position, client.camera.look_at, client.camera.up_direction))
            elif client.client_id in self.bookmarks:
                client.camera.position, client.camera.look_at, client.camera.up_direction = self.bookmarks[client.client_id]

    def select_camera(self, name):
        self.camera.value = name
        self.inspect_camera()

    def inspect_camera(self, event=None):
        if self.selection_handle is not None:
            self.selection_handle.remove()
            self.selection_handle = None
        if self.camera.value not in self.camera_options:
            return
        key, index = self.camera_options[self.camera.value]
        camera = self.layers[key]['cameras'][index]
        self.selection_handle = self.server.scene.add_camera_frustum('/inspection/selected',
            fov=np.radians(60), aspect=1.5, scale=self.span*.022*self.camera_size.value,
            color=(255, 220, 125), line_width=3, position=camera['pos'], wxyz=camera['wxyz'],
            visible=self.cameras.value)
        self.camera_info.content = f'**{camera["name"]}**\n\nLayer {key} · position ' + ', '.join(f'{v:.4g}' for v in camera['pos']) + '\n\nFrustum FOV is illustrative (60°), not calibrated optics.'

    def camera_action(self, event):
        with self.lock:
            if not self.camera_options:
                return
            action = self.camera_actions.value
            if action in ('Previous', 'Next'):
                names = list(self.camera_options)
                index = names.index(self.camera.value)
                self.select_camera(names[(index+(-1 if action == 'Previous' else 1)) % len(names)])
                return
            key, index = self.camera_options[self.camera.value]
            camera = self.layers[key]['cameras'][index]
            q = np.asarray(camera['wxyz'])
            matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            for client in self.clients(event):
                position = np.asarray(camera['pos'])
                if action == 'Focus':
                    client.camera.position = position + np.array([0, -.15, .1])*self.span
                    client.camera.up_direction = (0, 0, 1)
                    client.camera.look_at = position
                else:
                    client.camera.position = position
                    client.camera.up_direction = -matrix[:, 1]
                    client.camera.look_at = position + matrix[:, 2]*self.span*.1

    def refresh_runs(self, event):
        with self.lock:
            runs = self.discover()
            if not runs:
                self.status.content = '**No runs found.** Existing scene retained.'
                return
            self.runs = runs
            self.cache.clear()
            for control in (self.a, self.b):
                previous = control.value
                control.options = list(runs)
                control.value = previous if previous in runs else next(iter(runs))
            self.b.disabled = self.mode.disabled = len(runs) < 2
            self.update()
