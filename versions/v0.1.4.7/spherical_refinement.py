"""Native equirectangular joint BA with manual landmarks and metric constraints.

CLI: python spherical_refinement.py --input reconstruction.json --controls controls.json
     --output refined.json. Input files are immutable; each refinement is a new result.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from spherical_geometry import load_observations
def pixel_bearing(pixel, width, height):
    x, y = pixel
    longitude = (x / width - .5) * 2 * np.pi
    latitude = (.5 - y / height) * np.pi
    return np.array([np.cos(latitude) * np.sin(longitude), -np.sin(latitude),
                     np.cos(latitude) * np.cos(longitude)])
def triangulate(observations, cameras):
    rays, centers = [], []
    for shot, pixel, _ in observations:
        rotation, translation, width, height = cameras[shot]
        rays.append(rotation.T @ pixel_bearing(pixel, width, height))
        centers.append(-rotation.T @ translation)
    if len(rays) < 2:
        raise ValueError('A manual tie needs observations in at least two registered images')
    rays, centers = np.asarray(rays), np.asarray(centers)
    projectors = np.eye(3)[None] - rays[:, :, None] * rays[:, None, :]
    normal = projectors.sum(axis=0)
    if np.linalg.cond(normal) > 1e6:
        raise ValueError('Manual point has insufficient parallax for triangulation')
    point = np.linalg.solve(normal, np.einsum('nij,nj->i', projectors, centers))
    if np.any(np.sum((point - centers) * rays, axis=1) <= 0):
        raise ValueError('Manual observations triangulate behind an observed spherical ray')
    return point
def refine(input_path, output_path, controls_path=None, max_points=10000, max_iterations=100, backend="auto", progress=None):
    started=time.perf_counter()
    def notify(phase, **values):
        if progress:progress(dict(phase=phase,elapsed=time.perf_counter()-started,**values))
    notify("load")
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.absolute() == output_path.absolute():
        raise ValueError('Refinement output must differ from its immutable input')
    data = json.loads(input_path.read_text(encoding='utf-8'))
    components = data if isinstance(data, list) else [data]
    if len(components) != 1:
        raise ValueError('Refine each disconnected component separately; do not silently select the first')
    component = copy.deepcopy(components[0])
    controls = json.loads(Path(controls_path).read_text(encoding='utf-8')) if controls_path else {'points': []}
    cameras = {}
    for name, shot in component['shots'].items():
        camera = component['cameras'][shot['camera']]
        if camera['projection_type'] not in ['spherical', 'equirectangular']:
            raise ValueError('This solver requires native spherical cameras')
        cameras[name] = (cv2.Rodrigues(np.asarray(shot['rotation'], dtype=float))[0],
                         np.asarray(shot['translation'], dtype=float), camera['width'], camera['height'])
    notify("tracks", cameras=len(cameras))
    observations = {}
    for shot, point, pixel in load_observations(input_path, component):
        observations.setdefault(point, []).append((shot, pixel, 1.0))
    controls_by_id, checks = {}, []
    for control in controls.get('points', []):
        if not control.get('enabled', True):
            continue
        point_id = str(control['id'])
        if point_id in controls_by_id:
            raise ValueError(f'Duplicate control id: {point_id}')
        controls_by_id[point_id] = control
        values = []
        seen = set()
        for obs in control.get('observations', []):
            shot = obs['shot_id']
            if shot not in cameras or shot in seen:
                raise ValueError(f'Unknown/duplicate shot in control {point_id}: {shot}')
            seen.add(shot)
            pixel = np.asarray(obs['pixel'], dtype=float)
            sigma = float(obs.get('sigma_px', 1.0))
            width, height = cameras[shot][2:]
            if pixel.shape != (2,) or not np.isfinite(pixel).all() or not 0 <= pixel[1] < height or sigma <= 0:
                raise ValueError(f'Invalid control measurement: {point_id}')
            # Longitude wraps; latitude does not.
            pixel[0] %= width
            values.append((shot, pixel, sigma))
        if control.get('type', 'tie') == 'check':
            checks.append((control, values))
            observations.pop(point_id, None)  # Never optimize a check landmark.
            continue
        if len(values) < 2:
            raise ValueError(f'Control {point_id} needs at least two observations')
        replaced_shots = {shot for shot, _, _ in values}
        observations[point_id] = [obs for obs in observations.get(point_id, []) if obs[0] not in replaced_shots] + values
        point = triangulate(values, cameras)
        component['points'].setdefault(point_id, {'coordinates': point.tolist(), 'color': [255, 180, 0]})
    # Keep every manual landmark while bounding the automatic optimization budget.
    manual = {k for k, v in controls_by_id.items() if v.get('type', 'tie') != 'check'}
    automatic = sorted(k for k, obs in observations.items() if k not in manual and len(obs) >= 2)
    if max_points > 0 and len(automatic) > max_points:
        automatic = [automatic[i] for i in np.linspace(0, len(automatic) - 1, max_points, dtype=int)]
    point_ids = sorted(manual | set(automatic))
    if not point_ids:
        raise ValueError('No multi-view tracks available; rebuild/export observations first')
    names = sorted(cameras)
    gcps = {k: v for k, v in controls_by_id.items() if v.get('type') == 'gcp'}
    if gcps and len(gcps) < 3:
        raise ValueError('World-constrained refinement needs at least three non-collinear local-coordinate GCPs')
    if gcps:
        positions = np.array([gcps[k]['position'] for k in gcps], dtype=float)
        if positions.shape != (len(gcps), 3) or not np.isfinite(positions).all() or np.linalg.matrix_rank(positions - positions.mean(axis=0)) < 2:
            raise ValueError('GCP positions must be finite, non-collinear local XYZ coordinates')
        # Initialize the gauge from surveyed coordinates before nonlinear BA.
        from benchmark_360 import similarity
        source = np.array([component['points'][k]['coordinates'] for k in gcps])
        scale, world_rotation, world_translation = similarity(source, positions)
        for name, (rotation, translation, width, height) in list(cameras.items()):
            center = scale * world_rotation @ (-rotation.T @ translation) + world_translation
            rotation = rotation @ world_rotation.T
            cameras[name] = (rotation, -rotation @ center, width, height)
        for point in component['points'].values():
            point['coordinates'] = (scale * world_rotation @ np.asarray(point['coordinates']) + world_translation).tolist()
    free_names = names if gcps else names[1:]
    camera_index = {name: i * 6 for i, name in enumerate(free_names)}
    point_offset = len(free_names) * 6
    point_index = {name: point_offset + i * 3 for i, name in enumerate(point_ids)}
    params = []
    for name in free_names:
        rotation, translation, _, _ = cameras[name]
        params.extend(Rotation.from_matrix(rotation).as_rotvec())
        params.extend(translation)
    for point in point_ids:
        params.extend(component['points'][point]['coordinates'])
    initial = np.asarray(params, dtype=float)
    measurements = [(point, shot, pixel_bearing(pixel, *cameras[shot][2:]),
                     max(2 * np.pi / cameras[shot][2] * sigma, 1e-8))
                    for point in point_ids for shot, pixel, sigma in observations[point]]
    disabled = {str(p['id']) for p in controls.get('points', []) if not p.get('enabled', True)}
    distances = [d for d in controls.get('distances', []) if d.get('enabled', True)
                 and str(d['a']) not in disabled and str(d['b']) not in disabled]
    for distance in distances:
        if str(distance['a']) not in point_index or str(distance['b']) not in point_index or float(distance['meters']) <= 0 or float(distance.get('sigma_m', .01)) <= 0:
            raise ValueError('Invalid distance constraint or unknown endpoint')
    # Fix the arbitrary monocular scale only when no metric/world constraints exist.
    centers = {name: -r.T @ t for name, (r, t, _, _) in cameras.items()}
    scale_name = max(names[1:], key=lambda n: np.linalg.norm(centers[n] - centers[names[0]])) if len(names) > 1 else None
    baseline = np.linalg.norm(centers[scale_name] - centers[names[0]]) if scale_name else 0
    scale_anchor = bool(not gcps and not distances and baseline > 1e-8)
    if not gcps and not distances and not scale_anchor:
        raise ValueError('A translated camera baseline or metric controls are required')
    def unpack(values):
        poses = {name: (r, t) for name, (r, t, _, _) in cameras.items()}
        for name, start in camera_index.items():
            poses[name] = (Rotation.from_rotvec(values[start:start+3]).as_matrix(), values[start+3:start+6])
        points = {name: values[start:start+3] for name, start in point_index.items()}
        return poses, points
    from spherical_acceleration import BearingProblem, cuda_available, solve_cuda
    problem=BearingProblem(cameras,free_names,point_ids,measurements,gcps,distances,scale_name,baseline,scale_anchor)
    if backend not in ['auto','cpu','cuda']:raise ValueError('Choose auto, cpu or cuda refinement.')
    use_cuda=backend=='cuda' or (backend=='auto' and len(measurements)>=2000 and cuda_available())
    if use_cuda and not cuda_available():raise ValueError('CUDA requested but PyTorch CUDA is unavailable. Select CPU or Auto.')
    def residual(values,weighted=True):return problem.evaluate(values,weighted=weighted)
    notify('optimize',backend='cuda' if use_cuda else 'cpu',points=len(point_ids),observations=len(measurements))
    setup_seconds=time.perf_counter()-started;solve_started=time.perf_counter();fallback_reason=None
    if use_cuda:
        import torch
        def gpu_progress(event):
            event=dict(event);phase=event.pop('phase');event['solver_elapsed']=event.pop('elapsed',0);notify(phase,**event)
        try:result=solve_cuda(problem,initial,max_iterations,gpu_progress)
        except torch.cuda.OutOfMemoryError:
            if backend!='auto':raise
            use_cuda=False;fallback_reason='CUDA out of memory; using analytic CPU solver';notify('fallback',reason=fallback_reason)
        finally:torch.cuda.empty_cache()
    if not use_cuda:
        evaluations=0;last_update=0.
        def jacobian(values):
            nonlocal evaluations,last_update
            evaluations+=1
            if time.perf_counter()-last_update>1:
                notify('optimize',iteration=evaluations,iterations=max_iterations,backend='cpu');last_update=time.perf_counter()
            return problem.evaluate(values,jac=True)[1]
        result=least_squares(residual,initial,jac=jacobian,loss='soft_l1',f_scale=2.,max_nfev=max_iterations,x_scale='jac')
    solve_seconds=time.perf_counter()-solve_started
    notify('retriangulate')
    if not np.isfinite(result.x).all() or np.sum(residual(result.x)**2) > np.sum(residual(initial)**2) * 1.001:
        raise RuntimeError('Refinement failed verification; input reconstruction preserved')
    poses, points = unpack(result.x)
    for name, (rotation, translation) in poses.items():
        component['shots'][name]['rotation'] = Rotation.from_matrix(rotation).as_rotvec().tolist()
        component['shots'][name]['translation'] = translation.tolist()
    for point, xyz in points.items():
        component['points'][point]['coordinates'] = xyz.tolist()
        component['points'][point]['observations'] = [{'shot_id': shot, 'pixel': pixel.tolist()} for shot, pixel, _ in observations[point]]
    check_results = []
    refined_cameras = {name: (poses[name][0], poses[name][1], *cameras[name][2:]) for name in names}
    # Points excluded by the BA budget must follow the corrected camera geometry.
    from spherical_acceleration import retriangulate_batch
    remaining=[point for point,obs in observations.items() if point not in points and len(obs)>=2 and point in component['points']]
    for start in range(0,len(remaining),5000):
        batch=remaining[start:start+5000];xyz,valid=retriangulate_batch(batch,observations,refined_cameras)
        for i,point in enumerate(batch):
            if not valid[i]:component['points'].pop(point,None);continue
            component['points'][point]['coordinates']=xyz[i].tolist()
            component['points'][point]['observations']=[{'shot_id':shot,'pixel':pixel.tolist()} for shot,pixel,_ in observations[point]]
        notify('retriangulate',completed=min(start+5000,len(remaining)),total=len(remaining))
    for control, obs in checks:
        position = triangulate(obs, refined_cameras)
        check_results.append({'id': control['id'], 'error_units': float(np.linalg.norm(position - np.asarray(control['position'])))})
    def angle_rms(values):
        chord = residual(values, weighted=False).reshape(-1, 3)
        angles = 2 * np.arcsin(np.clip(np.linalg.norm(chord, axis=1) / 2, 0, 1))
        return float(np.degrees(np.sqrt(np.mean(angles**2))))
    report = {'backend': 'torch_cuda_sparse_lm_analytic' if use_cuda else 'scipy_sparse_cpu_analytic', 'cameras': len(names), 'optimized_points': len(point_ids),
              'observations': len(measurements), 'before_angular_rms_deg': angle_rms(initial),
              'after_angular_rms_deg': angle_rms(result.x), 'seconds': time.perf_counter() - started,
              'converged': bool(result.success), 'termination': result.message, 'checks': check_results,
              'scale': 'metric_constraints' if gcps or distances else 'original_arbitrary_scale',
              'evaluations':int(result.nfev),'device':getattr(result,'device','CPU'),
              'peak_solver_vram_mib':getattr(result,'peak_vram_mib',0.),'setup_seconds':setup_seconds,
              'solver_seconds':solve_seconds,'fallback_reason':fallback_reason}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notify('export')
    from control_points_model import atomic_json
    from refined_exports import export_refined
    atomic_json(output_path,[component],indent=None)
    report['exports']=export_refined(output_path,component)
    report['seconds']=time.perf_counter()-started
    atomic_json(output_path.with_suffix('.report.json'),report)
    notify('complete',seconds=report['seconds'])
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--controls')
    parser.add_argument('--max-points', type=int, default=10000)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--backend',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    def progress(event):print('ABU_PROGRESS '+json.dumps(event),flush=True)
    report=refine(args.input,args.output,args.controls,args.max_points,args.iterations,args.backend,progress)
    print('ABU_RESULT '+json.dumps(report),flush=True)
