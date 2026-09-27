"""Navigation and honest A/B inspection for the Viser reconstruction workspace."""
from collections import OrderedDict
from threading import RLock
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


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
    pairs = [(left[name], right[name]) for name in sorted(left.keys() & right.keys())]
    # Exporters often prepend images/ or an absolute dataset directory.
    # Only use basename fallback when it is unique in both complete inputs.
    def basenames(cameras):
        result = {}
        for i, camera in enumerate(cameras):
            name = camera['name'].replace('\\', '/').rsplit('/', 1)[-1]
            result.setdefault(name, []).append(i)
        return result
    aa, bb = basenames(a), basenames(b)
    used_a, used_b = {i for i,j in pairs}, {j for i,j in pairs}
    for name in sorted(aa.keys() & bb.keys()):
        if len(aa[name]) == len(bb[name]) == 1:
            i,j = aa[name][0],bb[name][0]
            if i not in used_a and j not in used_b:
                pairs.append((i,j))
    return pairs


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


def cloud_distances(reference, query, limit=50000):
    """Directed nearest-sample distances, not calibrated reconstruction error."""
    if not len(reference) or not len(query):
        return np.empty(0), 0
    indices = np.linspace(0, len(reference)-1, min(limit, len(reference)), dtype=int)
    distance, _ = cKDTree(reference[indices]).query(query, workers=1)
    return distance, len(indices)


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
        self._thumb_cache = {}
        self._thumb_executor = ThreadPoolExecutor(max_workers=6)
        gui = server.gui
        gui.add_markdown(f'## 3D workspace\n{title}')
        self.status = gui.add_markdown('Loading reconstruction…')
        with gui.add_folder('Navigate'):
            self.views = gui.add_button_group('View', ('Fit', 'Top', 'Front', 'Right'))
            self.fit_traj_btn = gui.add_button('🎯 Fit the trajectory')
            self.open_run_btn = gui.add_button('📂 Open Run Folder')
            self.zoom = gui.add_button_group('Distance', ('Closer', 'Farther'))
            self.bookmark = gui.add_button_group('Viewpoint', ('Save view', 'Restore view'))
            self.up_axis = gui.add_dropdown('World up', ('Z up', 'Y up'), initial_value='Z up',
                hint='Used by standard views. Camera look-through uses the source camera orientation.')
            self.navigation_info = gui.add_markdown('**Orbit** drag · **Pan** right-drag · **Zoom** scroll')
        with gui.add_folder('Compare reconstructions'):
            names = list(runs)
            self.a = gui.add_dropdown('A · Reference', names, initial_value=selected)
            self.b = gui.add_dropdown('B · Comparison', names,
                initial_value=next((n for n in names if n != selected), selected), disabled=len(names)<2)
            self.mode = gui.add_dropdown('Show', ('A only', 'B only', 'Overlay'), initial_value='A only', disabled=len(names)<2)
            self.registration = gui.add_checkbox('Register B to A', False,
                hint='Fits rotation, translation and scale from identical image names. Changes the preview only.')
            self.compare_button = gui.add_button('Compare A and B', disabled=len(names)<2)
            self.links = gui.add_checkbox('Camera offsets', True)
            gui.add_markdown('**A · mint** / **B · coral** · Overlay uses layer colors.')
            with gui.add_folder('How comparison works', expand_by_default=False):
                gui.add_markdown('Switch A/B from the same viewpoint. Display settings never reset navigation.\n\n'
                    'Independent reconstructions may have different origins and scales. Register B to A fits '
                    'rotation, translation, and scale using uniquely matched camera image names.\n\n'
                    'Residuals measure agreement, not ground-truth accuracy. They use the same cameras fitted '
                    'during registration. Scene units are not necessarily meters. Source files are never changed.')
            self.comparison = gui.add_markdown('Choose a different run for B to compare.')
            self.refresh = gui.add_button('Refresh available runs')
        with gui.add_folder('Scene layers', expand_by_default=False):
            self.points = gui.add_checkbox('Point clouds', True)
            self.cameras = gui.add_checkbox('Camera frustums', True)
            self.camera_thumbs = gui.add_checkbox('Frustum thumbnails', True)
            self.paths = gui.add_checkbox('Camera trajectory', True, hint='Connects camera centers in file order; not an inferred capture path.')
            self.grid = gui.add_checkbox('Ground grid', False)
            self.point_size = gui.add_number('Point size', initial_value=0.004, min=0.000001, max=1.0, step=0.000001, hint='Scene units. Values down to 0.000001 are supported.')
            self.camera_size = gui.add_slider('Camera size', 0.2, 3.0, 0.1, 1.0)
            self.color = gui.add_dropdown('Point color', ('Original', 'Layer color', 'B distance to A'), initial_value='Original')
            self.budget = gui.add_dropdown('Points per run', ('100,000', '300,000', '1,000,000'), initial_value='300,000')
            self.trim = gui.add_checkbox('Hide outer 1% of points', False,
                hint='Preview only. Off preserves all finite points before display sampling.')
        with gui.add_folder('Inspect a camera', expand_by_default=False):
            self.camera = gui.add_dropdown('Camera', ('No cameras',))
            self.camera_actions = gui.add_button_group('Selection', ('Previous', 'Next', 'Focus', 'Look through'))
            self.open_photo_btn = gui.add_button('View photo')
            self.auto_open_photo = gui.add_checkbox('Preview photo on select', False,
                hint='Show a single in-browser photo dialog when you select a camera.')
            self.camera_info = gui.add_markdown('Click a camera frustum or choose an image above. Frustum optics are illustrative.')
        with gui.add_folder('Run details', expand_by_default=False):
            self.details = gui.add_markdown('')

        for control in (self.a, self.b, self.mode, self.registration, self.links, self.points,
                        self.cameras, self.camera_thumbs, self.paths, self.grid, self.point_size, self.camera_size,
                        self.color, self.budget, self.trim):
            control.on_update(self.comparison_changed if control in (self.b, self.color) else self.update)
        self.compare_button.on_click(self.comparison_changed)
        self.views.on_click(self.navigate)
        self.fit_traj_btn.on_click(self.fit_trajectory)
        self.open_run_btn.on_click(self.open_run_folder)
        self.zoom.on_click(self.zoom_view)
        self.bookmark.on_click(self.saved_view)
        self.refresh.on_click(self.refresh_runs)
        self.camera.on_update(self.inspect_camera)
        self.camera_actions.on_click(self.camera_action)
        self.open_photo_btn.on_click(self.open_selected_photo)
        server.on_client_connect(lambda client: self.set_view(client, 'Fit'))
        self.update()

    def comparison_changed(self, event=None):
        if self.a.value != self.b.value:
            self.mode.value = 'Overlay'
        self.update()
        if self.mode.value == 'Overlay':
            for client in self.clients(event):
                self.set_view(client, 'Fit')

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
                                    'Inter-run agreement, not ground-truth accuracy.')
                    else:
                        summary += '\n\nNo camera differences available. Use matching image names; ambiguous duplicate filenames are excluded.'
                    layers['B'] = b
                self.layers = layers
                self.comparison.content = summary
                with self.server.atomic():
                    self.render()
                count = sum(len(layer['points']) for key, layer in layers.items()
                            if self.mode.value == 'Overlay' or self.mode.value.startswith(key))
                self.status.content = f'**{self.mode.value}** · {count:,} finite source points\n\n' + (
                    'No geometry found. Select a completed reconstruction or refresh.' if not count and not any(l['cameras'] for l in layers.values())
                    else 'Ready · viewpoint preserved')
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
            point_colors = tint if self.mode.value == 'Overlay' or self.color.value == 'Layer color' else colors[indices]
            if key == 'B' and self.color.value == 'B distance to A':
                distance, reference_count = cloud_distances(self.layers['A']['points'], sampled)
                if len(distance):
                    maximum = max(float(np.percentile(distance, 95)), 1e-9)
                    fraction = np.clip(distance/maximum, 0, 1)[:, None]
                    point_colors = ((1-fraction)*np.array([95, 190, 240])+fraction*np.array([255, 115, 90])).astype(np.uint8)
                    self.comparison.content += (f'\n\n**Cloud distance · B → A**\n\nMedian {np.median(distance):.4g} · '
                        f'95th percentile {np.percentile(distance, 95):.4g} scene units.\n\n'
                        f'Blue → coral: 0 → {maximum:.4g}. Nearest sample in A ({reference_count:,} reference points). '
                        'Directed, sampled distances; not correspondence or ground-truth errors.')
                else:
                    self.comparison.content += '\n\nCloud distances unavailable: both runs need finite points.'
            if len(sampled):
                self.handles.append(self.server.scene.add_point_cloud(f'/inspection/{key}/points', sampled,
                    point_colors,
                    point_size=self.point_size.value, point_shape='circle', visible=visible and self.points.value))
            if len(positions) > 1:
                self.handles.append(self.server.scene.add_line_segments(f'/inspection/{key}/path',
                    np.stack((positions[:-1], positions[1:]), axis=1), tint, line_width=2,
                    visible=visible and self.paths.value))
            run_path = self.runs.get(self.a.value if key == 'A' else self.b.value, '')
            step = max(1, int(np.ceil(len(cameras)/250)))
            for i, camera in enumerate(cameras):
                name = f'{key} · {i+1:04d} · {camera["name"]}'
                if visible:
                    self.camera_options[name] = (key, i)
                if i % step:
                    continue
                cam_name = camera.get('name', '')
                cache_key = (cam_name, run_path)
                cached_thumb = self._thumb_cache.get(cache_key)
                handle = self.server.scene.add_camera_frustum(f'/inspection/{key}/camera/{i}',
                    fov=np.radians(60), aspect=1.5, scale=span*.018*self.camera_size.value,
                    color=tint, position=camera['pos'], wxyz=camera['wxyz'],
                    image=cached_thumb if self.camera_thumbs.value else None, format='jpeg',
                    visible=visible and self.cameras.value)
                handle.on_click(lambda event, value=name: self.select_camera(value, event))
                self.handles.append(handle)
                if self.camera_thumbs.value and cache_key not in self._thumb_cache and cam_name:
                    self._thumb_executor.submit(self._fetch_thumbnail_and_update, cam_name, run_path, handle)
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
        up = np.array((0, 1, 0) if name == 'Top' else (0, 0, 1))
        if self.up_axis.value == 'Y up':
            direction = direction[[0, 2, 1]] * [1, 1, -1]
            up = up[[0, 2, 1]] * [1, 1, -1]
        client.camera.up_direction = up
        fov = float(getattr(client.camera, 'fov', np.radians(60)))
        aspect = max(.1, float(getattr(client.camera, 'aspect', 1)))
        half_angle = min(fov/2, np.arctan(np.tan(fov/2)*aspect))
        distance = self.span*.5/max(.05, np.sin(half_angle))*1.12
        client.camera.position = self.center + direction*distance
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
                self.navigation_info.content = 'View saved. Use Restore view to return after inspecting another angle.'
            elif client.client_id in self.bookmarks:
                client.camera.position, client.camera.look_at, client.camera.up_direction = self.bookmarks[client.client_id]
                self.navigation_info.content = 'Saved viewpoint restored.'
            else:
                self.navigation_info.content = 'No saved viewpoint yet. Choose Save view first.'

    def _resolve_photo(self, cam_name: str, run_dir: str = "") -> Optional[str]:
        if not cam_name:
            return None
        if os.path.isabs(cam_name) and os.path.isfile(cam_name):
            return cam_name
        clean_name = os.path.basename(cam_name)
        dirs_to_check = []
        if run_dir and os.path.isdir(run_dir):
            dirs_to_check.extend([
                run_dir,
                os.path.join(run_dir, "images"),
                os.path.join(run_dir, "3dgs_dataset", "images"),
                os.path.join(run_dir, "input"),
                os.path.dirname(run_dir),
                os.path.dirname(os.path.dirname(run_dir)),
                os.path.join(os.path.dirname(os.path.dirname(run_dir)), "images"),
                os.path.join(os.path.dirname(os.path.dirname(run_dir)), "input"),
            ])
        for d in dirs_to_check:
            if not d or not os.path.isdir(d):
                continue
            for c in [os.path.join(d, cam_name), os.path.join(d, clean_name)]:
                if os.path.isfile(c):
                    return c
            base, _ = os.path.splitext(clean_name)
            for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]:
                c = os.path.join(d, base + ext)
                if os.path.isfile(c):
                    return c
        return None

    def _fetch_thumbnail_and_update(self, cam_name: str, run_dir: str, handle):
        thumb = self._get_thumbnail(cam_name, run_dir)
        if thumb is not None and handle is not None:
            try:
                handle.image = thumb
            except Exception:
                pass

    def _get_thumbnail(self, cam_name: str, run_dir: str = "") -> Optional[np.ndarray]:
        if not cam_name:
            return None
        cache_key = (cam_name, run_dir)
        if cache_key in self._thumb_cache:
            return self._thumb_cache[cache_key]
        photo_path = self._resolve_photo(cam_name, run_dir)
        if not photo_path:
            self._thumb_cache[cache_key] = None
            return None
        try:
            with Image.open(photo_path) as img:
                try:
                    img.draft("RGB", (160, 106))
                except Exception:
                    pass
                img = img.convert("RGB")
                img.thumbnail((160, 106), Image.Resampling.BILINEAR)
                arr = np.asarray(img, dtype=np.uint8)
                self._thumb_cache[cache_key] = arr
                return arr
        except Exception:
            self._thumb_cache[cache_key] = None
            return None

    def fit_trajectory(self, event=None):
        with self.lock:
            positions = []
            for key in ('A', 'B'):
                if key in self.layers and (self.mode.value == 'Overlay' or (key == 'A' and self.mode.value == 'A only') or (key == 'B' and self.mode.value == 'B only')):
                    cams = self.layers[key].get('cameras', [])
                    for c in cams:
                        positions.append(c['pos'])
            if not positions:
                for client in self.clients(event):
                    self.set_view(client, 'Fit')
                return
            positions = np.asarray(positions)
            traj_center = np.mean(positions, axis=0)
            diffs = positions - traj_center
            dist = np.linalg.norm(diffs, axis=1)
            traj_span = max(float(np.max(dist)) * 2.0, 0.5)

            direction = np.array((1.0, -1.5, 1.0), dtype=float)
            direction /= np.linalg.norm(direction)
            up = np.array((0.0, 0.0, 1.0))
            if self.up_axis.value == 'Y up':
                direction = direction[[0, 2, 1]] * [1, 1, -1]
                up = up[[0, 2, 1]] * [1, 1, -1]

            for client in self.clients(event):
                client.camera.up_direction = up
                fov = float(getattr(client.camera, 'fov', np.radians(60)))
                aspect = max(.1, float(getattr(client.camera, 'aspect', 1)))
                half_angle = min(fov/2, np.arctan(np.tan(fov/2)*aspect))
                distance = traj_span*.5/max(.05, np.sin(half_angle))*1.25
                client.camera.position = traj_center + direction*distance
                client.camera.look_at = traj_center
            self.navigation_info.content = f'Fitted viewpoint to trajectory ({len(positions)} cameras).'

    def open_run_folder(self, event=None):
        run_name = self.a.value if self.mode.value != 'B only' else self.b.value
        run_path = self.runs.get(run_name, '')
        if run_path and os.path.isdir(run_path):
            try:
                os.startfile(run_path)
            except Exception as e:
                print(f"[viewer] Error opening run folder: {e}")

    def select_camera(self, name, event=None):
        self.camera.value = name
        self.inspect_camera()
        if event is not None and self.auto_open_photo.value:
            self.open_selected_photo(event)


    def inspect_camera(self, event=None):
        if self.selection_handle is not None:
            self.selection_handle.remove()
            self.selection_handle = None
        if self.camera.value not in self.camera_options:
            return
        key, index = self.camera_options[self.camera.value]
        camera = self.layers[key]['cameras'][index]
        run_name = self.a.value if key == 'A' else self.b.value
        run_path = self.runs.get(run_name, '')
        cam_name = camera.get('name', '')
        thumb = self._get_thumbnail(cam_name, run_path) if self.camera_thumbs.value else None
        photo_path = self._resolve_photo(cam_name, run_path)

        self.selection_handle = self.server.scene.add_camera_frustum('/inspection/selected',
            fov=np.radians(60), aspect=1.5, scale=self.span*.022*self.camera_size.value,
            color=(255, 220, 125), line_width=3, position=camera['pos'], wxyz=camera['wxyz'],
            image=thumb, format='jpeg',
            visible=self.cameras.value)
        photo_info = f'\n\n📷 Photo: `{photo_path}`' if photo_path else '\n\n📷 Photo: *(not found on disk)*'
        self.camera_info.content = (
            f'**{cam_name}**\n\n'
            f'Layer {key} · position ' + ', '.join(f'{v:.4g}' for v in camera['pos']) +
            photo_info + '\n\nFrustum FOV is illustrative (60°), not calibrated optics.'
        )

        if event is not None and self.auto_open_photo.value:
            self.open_selected_photo(event)

    def open_selected_photo(self, event=None):
        if self.camera.value not in self.camera_options:
            return
        key, index = self.camera_options[self.camera.value]
        camera = self.layers[key]['cameras'][index]
        run_name = self.a.value if key == 'A' else self.b.value
        run_path = self.runs.get(run_name, '')
        cam_name = camera.get('name', '')
        photo_path = self._resolve_photo(cam_name, run_path)
        if not photo_path or not os.path.isfile(photo_path):
            self.camera_info.content = f'**{cam_name}**\n\n⚠️ Image file not found on disk for this camera.'
            return
        try:
            with Image.open(photo_path) as img:
                img = img.convert('RGB')
                img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                arr = np.asarray(img, dtype=np.uint8)
            previous = getattr(self, '_photo_modal', None)
            if previous is not None:
                previous.close()
            gui = event.client.gui if getattr(event, 'client', None) else self.server.gui
            modal = gui.add_modal(os.path.basename(photo_path))
            self._photo_modal = modal
            with modal:
                gui.add_markdown(f'Layer {key} · Camera {index + 1} / {len(self.layers[key]["cameras"])}')
                gui.add_image(arr, format='jpeg')
                close_btn = gui.add_button('Close preview')
                close_btn.on_click(lambda _: modal.close())
        except Exception as e:
            print(f"[viewer] Error opening browser photo modal: {e}")

    def camera_action(self, event):
        with self.lock:
            if not self.camera_options:
                return
            action = self.camera_actions.value
            if action in ('Previous', 'Next'):
                names = list(self.camera_options)
                index = names.index(self.camera.value)
                self.select_camera(names[(index+(-1 if action == 'Previous' else 1)) % len(names)], event)
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
            self.b.disabled = self.mode.disabled = self.compare_button.disabled = len(runs) < 2
            if len(runs) > 1 and self.b.value == self.a.value:
                self.b.value = next(n for n in runs if n != self.a.value)
            self.update()
