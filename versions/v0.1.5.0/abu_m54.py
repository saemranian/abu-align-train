"""Abu 3D Viewer M54: shared scene, navigation, transforms and local file I/O.

World geometry stays float64; adapters normalize only for display. Model transforms
move cloud, cameras and triangulated controls together. Sources are immutable.
"""
import copy
import csv
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

NAME = 'Abu 3D Viewer M54'
FORMATS = '*.json *.ply *.csv *.txt *.bin *.xyz *.pts *.npz'


def read_ply_geometry(source, limit=50000):
    """Preserve double-precision global coordinates; sample before array copies."""
    from plyfile import PlyData
    vertices=PlyData.read(str(source),mmap='r')['vertex'].data;count=len(vertices)
    indices=np.linspace(0,max(0,count-1),min(count,limit),dtype=int);v=vertices[indices]
    points=np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)
    colors=np.column_stack([v[k] for k in ['red','green','blue']]).astype(np.uint8) if all(k in v.dtype.names for k in ['red','green','blue']) else np.full((len(v),3),180,np.uint8)
    valid=np.isfinite(points).all(1)
    return points[valid],colors[valid],count


def read_colmap_points(source,limit=50000):
    source=Path(source);points=[];colors=[]
    if source.suffix.lower()=='.bin':
        size=source.stat().st_size
        with source.open('rb') as stream:
            def read_exact(n):
                value=stream.read(n)
                if len(value)!=n:raise ValueError('Truncated COLMAP points file.')
                return value
            count=struct.unpack('<Q',read_exact(8))[0]
            if count>(size-8)//51:raise ValueError('Invalid COLMAP point count.')
            indices=set(np.linspace(0,max(0,count-1),min(count,limit),dtype=int))
            for index in range(count):
                record=struct.unpack('<QdddBBBd',read_exact(43));track=struct.unpack('<Q',read_exact(8))[0]
                end=stream.tell()+track*8
                if end>size:raise ValueError('Truncated COLMAP point track.')
                if index in indices:points.append(record[1:4]);colors.append(record[4:7])
                stream.seek(end)
    else:
        with source.open(encoding='utf-8') as stream:count=sum(1 for line in stream if line.strip() and not line.lstrip().startswith('#'))
        indices=set(np.linspace(0,max(0,count-1),min(count,limit),dtype=int));index=0
        with source.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith('#'):continue
                parts=line.split()
                if len(parts)<8:raise ValueError('Invalid COLMAP point row.')
                if index in indices:points.append([float(v) for v in parts[1:4]]);colors.append([int(v) for v in parts[4:7]])
                index+=1
    points=np.asarray(points,dtype=np.float64).reshape(-1,3);colors=np.asarray(colors,dtype=np.uint8).reshape(-1,3);valid=np.isfinite(points).all(1)
    return points[valid],colors[valid],count


@dataclass
class SceneTransform:
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angles: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_matrix(cls,matrix):
        m=np.asarray(matrix,dtype=float)
        if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1]):raise ValueError('Expected a finite affine 4 × 4 matrix.')
        scale=float(np.cbrt(np.linalg.det(m[:3,:3])))
        if scale<=0:raise ValueError('Only positive uniform scale and proper rotation are supported.')
        r=m[:3,:3]/scale
        if not np.allclose(r.T@r,np.eye(3),atol=1e-8):raise ValueError('Non-uniform scale/shear cannot preserve camera optics.')
        return cls(m[:3,3].copy(),Rotation.from_matrix(r).as_euler('xyz',degrees=True),scale)

    def matrix(self):
        t, a, p = [np.asarray(v, dtype=np.float64) for v in (self.translation, self.angles, self.pivot)]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (t, a, p)):
            raise ValueError('Translation, rotation and pivot must contain three finite numbers.')
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Uniform scale must be positive and finite.')
        r = Rotation.from_euler('xyz', a, degrees=True).as_matrix()
        m = np.eye(4); m[:3, :3] = float(self.scale) * r; m[:3, 3] = p + t - m[:3, :3] @ p
        return m

    def report(self):
        return {'translation': np.asarray(self.translation).tolist(), 'rotation_xyz_degrees': np.asarray(self.angles).tolist(),
                'uniform_scale': float(self.scale), 'pivot': np.asarray(self.pivot).tolist(), 'matrix': self.matrix().tolist()}


def bounds(scene):
    p = np.asarray(scene['points']).reshape(-1, 3)
    pieces = [np.quantile(p, [.01, .99], axis=0)] if len(p) else []
    pieces += [np.asarray(list(scene['centers'].values())).reshape(-1, 3)] if scene['centers'] else []
    pieces += [np.asarray([c['position'] for c in scene.get('controls', {}).values()])] if scene.get('controls') else []
    if not pieces: return np.zeros(3), 1.
    xyz = np.concatenate(pieces); lo, hi = xyz.min(0), xyz.max(0)
    return (lo + hi) / 2, max(float(np.linalg.norm(hi - lo)), 1e-6)


def transform_scene(scene, transform):
    m = transform.matrix(); linear, offset = m[:3, :3], m[:3, 3]
    r = linear / float(transform.scale)
    def apply(p): return np.asarray(p, dtype=np.float64) @ linear.T + offset
    result = dict(scene, points=apply(scene['points']), centers={n: apply(p) for n, p in scene['centers'].items()},
                  camera_rotations={n: r @ v for n, v in scene.get('camera_rotations', {}).items()},
                  controls={n: dict(v, position=apply(v['position'])) for n, v in scene.get('controls', {}).items()})
    result['center'], result['extent'] = bounds(result)
    return result


def from_records(data):
    """Adapter for the existing engine result readers; preserve exact image names."""
    centers, rotations = {}, {}
    for c in data.get('cameras', []):
        name = c['name']; q = np.asarray(c['wxyz'])
        if name in centers: raise ValueError('Duplicate camera name; select one reconstruction component.')
        centers[name] = np.asarray(c['pos'], dtype=float)
        rotations[name] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    p = data.get('points'); p = np.empty((0, 3)) if p is None else np.asarray(p, dtype=float).reshape(-1, 3)
    colors = data.get('colors'); colors = np.full((len(p), 3), 180, np.uint8) if colors is None else np.asarray(colors, dtype=np.uint8)
    result = dict(points=p, colors=colors, centers=centers, camera_rotations=rotations, controls={},
                  source=data.get('path', data.get('source', 'scene')), total_points=data.get('count', len(p)))
    if 'm54_base_transform' in data:result['m54_base_transform']=data['m54_base_transform']
    result['center'], result['extent'] = bounds(result)
    return result


def to_records(scene):
    cameras = []
    for name, pos in scene['centers'].items():
        r = scene.get('camera_rotations', {}).get(name, np.eye(3)); q = Rotation.from_matrix(r).as_quat()
        cameras.append(dict(name=name, pos=pos, wxyz=q[[3, 0, 1, 2]]))
    return dict(points=scene['points'], colors=scene['colors'], cameras=cameras, path=scene.get('source', ''),
                source='Abu M54', count=scene.get('total_points', len(scene['points'])))


def load_scene(source, limit=50000, attach_companions=True):
    source = Path(source)
    if source.name.lower() in {'points3d.bin','points3d.txt'}:
        points,colors,count=read_colmap_points(source,limit)
        camera_file=next((source.with_name(n) for n in ['images.bin','images.txt'] if source.with_name(n).is_file()),None)
        scene=load_scene(camera_file,limit,False) if camera_file else from_records(dict(path=str(source)))
        scene.update(points=points,colors=colors,total_points=count,source=str(source))
    elif source.name.lower() in {'cameras.bin','cameras.txt','rigs.bin','rigs.txt','frames.bin','frames.txt'}:
        return load_scene(source.parent,limit,attach_companions)
    elif source.suffix.lower() == '.ply':
        points,colors,count=read_ply_geometry(source,limit)
        scene=from_records(dict(points=points,colors=colors,path=str(source),count=count))
    elif source.suffix.lower() == '.csv':
        with source.open(encoding='utf-8-sig', newline='') as stream:
            sample = stream.read(4096); stream.seek(0)
            try: dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except csv.Error: dialect = csv.excel
            rows = list(csv.DictReader(stream, dialect=dialect))
        records = []
        for row in rows:
            row = {str(k).strip().lstrip('#').lower(): v for k, v in row.items() if k is not None}
            if all(k in row for k in ['name','x','y','alt','yaw','pitch','roll']):
                # This is the documented Abu exporter convention: azimuth from
                # North, pitch from nadir, roll about forward, camera Y down.
                yaw,pitch,roll=np.radians([float(row[k]) for k in ['yaw','pitch','roll']])
                forward=np.array([np.sin(pitch)*np.sin(yaw),np.sin(pitch)*np.cos(yaw),-np.cos(pitch)])
                right0=np.array([np.cos(yaw),-np.sin(yaw),0.]);up0=np.cross(right0,forward)
                up=np.cos(roll)*up0+np.sin(roll)*right0;right=np.cross(forward,up)
                matrix=np.column_stack([right,-up,forward]);q=Rotation.from_matrix(matrix).as_quat()
                records.append(dict(name=row['name'].replace('\\','/').removeprefix('./'),pos=np.array([float(row[k]) for k in ['x','y','alt']]),wxyz=q[[3,0,1,2]]))
                continue
            if not all(k in row for k in ['center_x', 'center_y', 'center_z']):
                raise ValueError('Camera CSV needs image, center_x/y/z and camera_to_world_qw/qx/qy/qz. A control-point measurement CSV is not a camera alignment.')
            qkeys = ['camera_to_world_q' + c for c in 'wxyz']
            if not all(k in row for k in qkeys):
                raise ValueError('Camera CSV orientation must be explicit: camera_to_world_qw/qx/qy/qz.')
            records.append(dict(name=row.get('image', row.get('name', '')), pos=np.array([float(row[k]) for k in ['center_x', 'center_y', 'center_z']]),
                                wxyz=np.array([float(row[k]) for k in qkeys])))
        if not records or any(not c['name'] or not np.isfinite(c['pos']).all() or not np.isfinite(c['wxyz']).all() or np.linalg.norm(c['wxyz']) == 0 for c in records):
            raise ValueError('Camera CSV has no valid, named finite cameras.')
        scene = from_records(dict(cameras=records, path=str(source)))
    elif source.suffix.lower() in {'.xyz', '.pts'}:
        rows = np.loadtxt(source, ndmin=2, skiprows=1 if source.suffix.lower() == '.pts' else 0)
        if rows.shape[1] < 3: raise ValueError('XYZ/PTS needs at least three coordinate columns.')
        mask = np.isfinite(rows[:, :3]).all(1); rows = rows[mask]; count = len(rows)
        rows = rows[np.linspace(0, max(0, count-1), min(count, limit), dtype=int)]
        colors = np.clip(rows[:, -3:], 0, 255).astype(np.uint8) if rows.shape[1] >= 6 else None
        scene = from_records(dict(points=rows[:, :3], colors=colors, path=str(source), count=count))
    elif source.suffix.lower() in {'.txt', '.bin'}:
        from fast_viewer import load_cameras_from_colmap_images, load_cameras_from_colmap_bin, load_cameras_from_tum,load_cameras_from_ace0
        if source.name.lower() == 'images.txt': records = load_cameras_from_colmap_images(str(source))
        elif source.name.lower() == 'images.bin': records = load_cameras_from_colmap_bin(str(source))
        elif source.name.lower().startswith('poses_'):records=load_cameras_from_ace0(str(source))
        elif source.suffix.lower() == '.txt': records = load_cameras_from_tum(str(source))
        else: raise ValueError('Select COLMAP images.bin or its completed run folder; cameras.bin alone has intrinsics, not poses.')
        if not records: raise ValueError('No camera poses found. Select a supported trajectory, images.txt or a completed run folder.')
        scene = from_records(dict(cameras=records, path=str(source)))
    elif source.suffix.lower() == '.npz':
        from control_points_comparison import read_scene
        scene = read_scene(source); scene['source'] = str(source)
        if len(scene['points'])>limit:
            indices=np.linspace(0,len(scene['points'])-1,limit,dtype=int);scene['points']=scene['points'][indices];scene['colors']=scene['colors'][indices]
    else:
        from control_points_comparison import load_scene as load_reconstruction
        scene = load_reconstruction(source, limit)
        if source.is_file() and source.suffix.lower()=='.json' and not scene.get('controls'):
            data=json.loads(source.read_text(encoding='utf-8'))
            if isinstance(data,dict) and 'frames' in data:
                companion=next((p for p in [source.with_suffix('.points.ply'),source.parent/'points.ply',source.parent/'points3D.ply'] if p.is_file()),None)
                if companion:scene['points'],scene['colors'],scene['total_points']=read_ply_geometry(companion,limit)
        if source.is_dir():
            from fast_viewer import find_run_assets
            assets=find_run_assets(str(source))
            if not any(assets[i] for i in [2,3,4,5]):
                camera_csv=next((p for p in [source/'Postshot_RealityCapture'/'cameras.csv',source/'cameras.csv'] if p.is_file()),None)
                if camera_csv:
                    camera_scene=load_scene(camera_csv,limit,False);scene['centers']=camera_scene['centers'];scene['camera_rotations']=camera_scene['camera_rotations']
            # Read PLY coordinates at their stored precision after shared discovery.
            if assets[0] and not (assets[5] and not any(assets[i] for i in [2,3,4])):
                scene['points'],scene['colors'],scene['total_points']=read_ply_geometry(assets[0],limit)
            if not len(scene['points']):
                camera_file=assets[2] or assets[3]
                directory=Path(camera_file).parent if camera_file else source
                points_file=next((directory/name for name in ['points3D.bin','points3D.txt'] if (directory/name).is_file()),None)
                if points_file:scene['points'],scene['colors'],scene['total_points']=read_colmap_points(points_file,limit)
    if attach_companions and source.suffix.lower() in {'.ply','.csv'}:
        if source.suffix.lower()=='.ply':
            candidates=[source.with_suffix('.cameras.csv')]
            if source.name.endswith('.points.ply'):candidates.insert(0,source.with_name(source.name[:-11]+'.cameras.csv'))
            companion=next((p for p in candidates if p.is_file()),None)
            if companion:
                cameras=load_scene(companion,limit,False);scene['centers']=cameras['centers'];scene['camera_rotations']=cameras['camera_rotations'];scene['companion_cameras']=str(companion)
        else:
            stem=source.name[:-12] if source.name.endswith('.cameras.csv') else source.stem
            companion=next((source.with_name(stem+suffix) for suffix in ['.points.ply','.ply'] if source.with_name(stem+suffix).is_file()),None)
            if companion:
                cloud=load_scene(companion,limit,False);scene['points']=cloud['points'];scene['colors']=cloud['colors'];scene['total_points']=cloud['total_points'];scene['companion_cloud']=str(companion)
    scene['center'], scene['extent'] = bounds(scene)
    return scene


def screen_pick(points, screen, valid, depth, x, y, radius=8):
    """Pick a displayed sample; depth resolves overlapping points, then distance."""
    distance = np.linalg.norm(screen - [x, y], axis=1)
    indices = np.flatnonzero(valid & (distance <= radius))
    if not len(indices): return None
    return int(indices[np.lexsort((distance[indices], depth[indices]))[0]])


def ray_pick(points, origin, direction, aperture=.008):
    p = np.asarray(points); origin = np.asarray(origin); direction = np.asarray(direction)
    norm = np.linalg.norm(direction)
    if not len(p) or not np.isfinite(norm) or norm == 0: return None
    direction = direction / norm; delta = p - origin; along = delta @ direction
    perpendicular = np.linalg.norm(delta - along[:, None] * direction, axis=1)
    indices = np.flatnonzero((along > 0) & (perpendicular <= np.maximum(along, 1e-8) * aperture))
    return int(indices[np.argmin(along[indices])]) if len(indices) else None


def grid_lines(center, extent, plane='XY', elevation=0., cells=20):
    axes = {'XY': (0, 1, 2), 'XZ': (0, 2, 1), 'YZ': (1, 2, 0)}[plane]
    center = np.asarray(center); step = 10. ** np.floor(np.log10(max(extent, 1e-8) / cells))
    step *= 2 if extent / cells / step < 5 else 5
    half = cells * step / 2; segments = []
    for offset in np.arange(-cells // 2, cells // 2 + 1) * step:
        for a, b in [(axes[0], axes[1]), (axes[1], axes[0])]:
            pair = np.tile(center, (2, 1)); pair[:, axes[2]] = elevation
            pair[:, a] = center[a] + offset; pair[:, b] = center[b] + np.array([-half, half]); segments.append(pair)
    return np.asarray(segments), step


def transformed_component(component, transform):
    result = copy.deepcopy(component); m = transform.matrix(); r = m[:3, :3] / transform.scale
    for point in result.get('points', {}).values(): point['coordinates'] = (m[:3, :3] @ point['coordinates'] + m[:3, 3]).tolist()
    for shot in result['shots'].values():
        old = Rotation.from_rotvec(shot['rotation']).as_matrix(); c = -old.T @ np.asarray(shot['translation'])
        new = old @ r.T; c = m[:3, :3] @ c + m[:3, 3]
        shot['rotation'] = Rotation.from_matrix(new).as_rotvec().tolist(); shot['translation'] = (-new @ c).tolist()
    result['abu_m54_transform'] = transform.report()
    return result


def export_copy(source, target, transform):
    """Export full source geometry in a new copy; display sampling is not exported."""
    source, target = Path(source), Path(target)
    outputs = [target, target.with_suffix('.ply'), target.with_suffix('.points.ply'), target.with_suffix('.points.cameras.csv'), target.with_suffix('.cameras.csv'), target.with_suffix('.exports.json'), target.with_suffix('.m54.json')]
    if source.resolve() in [p.resolve() for p in outputs]: raise ValueError('Choose a new output name; the source must remain unchanged.')
    if any(p.exists() for p in outputs):raise ValueError('Choose a new output basename; a destination or companion file already exists.')
    from control_points_model import atomic_json
    if source.suffix.lower() == '.json':
        data = json.loads(source.read_text(encoding='utf-8'))
        component = data[0] if isinstance(data, list) and len(data) == 1 else data
        if isinstance(component, dict) and 'shots' in component and 'cameras' in component:
            result = transformed_component(component, transform)
            atomic_json(target, [result] if isinstance(data, list) else result)
            from refined_exports import export_refined
            report = export_refined(target, result)
            # The established refinement exporter writes float32 display PLY.
            # M54's coordinate-copy path retains float64 XYZ for global frames.
            precise=load_scene(target,limit=2**63-1)
            export_geometry(precise,target.with_suffix('.points.ply'),write_cameras=False)
            report['points']=len(precise['points']);atomic_json(target.with_suffix('.exports.json'),report)
        elif isinstance(component, dict) and 'frames' in component:
            result = copy.deepcopy(component); m = transform.matrix(); r = m[:3, :3] / transform.scale
            for frame in result['frames']:
                pose = np.asarray(frame['transform_matrix'],dtype=float); pose[:3, 3] = m[:3, :3] @ pose[:3, 3] + m[:3, 3]; pose[:3, :3] = r @ pose[:3, :3]
                frame['transform_matrix'] = pose.tolist()
                image=Path(frame.get('file_path',''))
                if str(image)!='.' and not image.is_absolute() and (source.parent/image).is_file():frame['file_path']=str((source.parent/image).resolve())
            atomic_json(target, result); report = export_geometry(transform_scene(load_scene(source, limit=2**63-1), transform), target.with_suffix('.points.ply'))
        else: raise ValueError('Select one reconstruction component or a supported transforms JSON.')
    else:
        report = export_geometry(transform_scene(load_scene(source, limit=2**63-1), transform), target.with_suffix('.ply'))
    report['source'] = str(source); report['transform'] = transform.report(); report['viewer'] = NAME
    atomic_json(target.with_suffix('.m54.json'), report)
    return report


def export_geometry(scene, cloud,write_cameras=True):
    cloud = Path(cloud); cloud.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(len(scene['points']), dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, name in enumerate(['x', 'y', 'z']): vertices[name] = scene['points'][:, i]
    for i, name in enumerate(['red', 'green', 'blue']): vertices[name] = scene['colors'][:, i]
    temporary = cloud.with_name(cloud.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(('ply\nformat binary_little_endian 1.0\ncomment Abu M54 transformed full source copy\n' + f'element vertex {len(vertices)}\n' +
                          'property double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
            vertices.tofile(stream)
        temporary.replace(cloud)
    finally: temporary.unlink(missing_ok=True)
    cameras = cloud.with_suffix('.cameras.csv')
    if write_cameras:
        with cameras.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['image', 'center_x', 'center_y', 'center_z', 'camera_to_world_qw', 'camera_to_world_qx', 'camera_to_world_qy', 'camera_to_world_qz'])
            for camera in to_records(scene)['cameras']: writer.writerow([camera['name'], *camera['pos'], *camera['wxyz']])
    return dict(point_cloud=str(cloud), camera_alignment_csv=str(cameras) if write_cameras else None, points=len(vertices), cameras=len(scene['centers']))
