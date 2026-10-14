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
from manual_camera_geometry import Camera, bearing, bearing_weight, spherical, load_alignment
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
        rays.append(rotation.T @ bearing(pixel,cameras[shot]))
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
def refine(input_path, output_path, controls_path=None, max_points=10000, max_iterations=100, backend="auto", progress=None, options=None):
    from manual_refinement_settings import validate
    settings=validate(options)
    if not 1<=max_points<=50000 or not 5<=max_iterations<=500:raise ValueError('Use 1–50000 automatic landmarks and 5–500 solver evaluations.')
    started=time.perf_counter()
    def notify(phase, **values):
        if progress:progress(dict(phase=phase,elapsed=time.perf_counter()-started,**values))
    notify("load")
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.absolute() == output_path.absolute():
        raise ValueError('Refinement output must differ from its immutable input')
    loaded,cameras=load_alignment(input_path);component=copy.deepcopy(loaded)
    controls = json.loads(Path(controls_path).read_text(encoding='utf-8')) if controls_path else {'points': []}
    from manual_image_policy import Masks,effective_controls,validate_policy
    policy=validate_policy(controls.get('image_policy'));mask_set=Masks(policy);mask_records=mask_set.validate_catalog({n:c[2:] for n,c in cameras.items()})
    controls,ignored_manual=effective_controls(controls,{n:c[2:] for n,c in cameras.items()})
    excluded=set(policy['excluded_images'])|{n for n,c in cameras.items() if mask_set.array(n,c[2:]) is not None and not mask_set.array(n,c[2:]).any()}
    for name in excluded:component['shots'].pop(name,None);cameras.pop(name,None)
    if len(cameras)<2:raise ValueError('At least two enabled registered images are required.')
    from control_points_model import validate_controls
    # Disabled incomplete legacy controls never enter the solve or validation
    # of active constraints. Preserve them in the original controls snapshot.
    disabled_ids={str(p['id']) for p in controls.get('points',[]) if not p.get('enabled',True)}
    active_controls=dict(controls,points=[p for p in controls.get('points',[]) if p.get('enabled',True)],
                         distances=[d for d in controls.get('distances',[]) if d.get('enabled',True) and str(d['a']) not in disabled_ids and str(d['b']) not in disabled_ids])
    controls=validate_controls(dict(active_controls,schema_version=controls.get('schema_version',1)))
    for control in controls['points']:
        for obs in control['observations']:
            name=obs['shot_id']
            if name not in cameras:raise ValueError(f'Control image is not registered: {name}')
            recorded=controls.get('image_sizes',{}).get(name)
            if recorded is not None and list(recorded)!=list(cameras[name][2:]):raise ValueError('Control image dimensions differ from the original calibrated images.')
            family=controls.get('image_projections',{}).get(name,controls.get('image_projection'))
            expected='spherical' if spherical(cameras[name]) else 'perspective'
            if family in {'spherical','perspective'} and family!=expected:raise ValueError('Control image projection differs from the calibrated camera; re-mark the original image.')
    if options is None:settings=validate(controls.get('refinement_settings'))
    notify("tracks", cameras=len(cameras))
    observations = {}
    ignored_automatic=0
    for shot, point, pixel in load_observations(input_path, component):
        if shot not in cameras or not mask_set.allowed(shot,pixel,cameras[shot][2:]):ignored_automatic+=1;continue
        observations.setdefault(point, []).append((shot, pixel, settings['auto_sigma_px']))
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
            if pixel.shape != (2,) or not np.isfinite(pixel).all() or not 0 <= pixel[1] < height or not np.isfinite(sigma) or sigma <= 0 or (not spherical(cameras[shot]) and not 0<=pixel[0]<width):
                raise ValueError(f'Invalid control measurement: {point_id}')
            # Longitude wraps; latitude does not.
            if spherical(cameras[shot]):pixel[0] %= width
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
    automatic = sorted(k for k, obs in observations.items() if k not in manual and len(obs) >= settings['min_track_views'])
    before_filter=len(automatic)
    if settings['min_angle_deg']>0:
        centers={n:-c[0].T@c[1] for n,c in cameras.items()}
        def enough_parallax(point):
            rays=np.array([np.asarray(component['points'][point]['coordinates'])-centers[n] for n,_,_ in observations[point]])
            norms=np.linalg.norm(rays,axis=1)
            if np.min(norms)<=1e-10:return False
            rays/=norms[:,None]
            return np.min(rays@rays.T)<=np.cos(np.radians(settings['min_angle_deg']))
        automatic=[k for k in automatic if enough_parallax(k)]
    after_parallax_filter=len(automatic)
    if max_points > 0 and len(automatic) > max_points:
        automatic = [automatic[i] for i in np.linspace(0, len(automatic) - 1, max_points, dtype=int)]
    point_ids = sorted(manual | set(automatic))
    if not point_ids:
        raise ValueError('No multi-view tracks available; rebuild/export observations first')
    names = sorted(cameras)
    involved=sorted({shot for point in point_ids for shot,_,_ in observations[point]})
    if len(involved)<2:raise ValueError('At least two translated registered cameras must have usable observations.')
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
            cameras[name] = Camera(rotation,-rotation@center,width,height,cameras[name].calibration)
        for point in component['points'].values():
            point['coordinates'] = (scale * world_rotation @ np.asarray(point['coordinates']) + world_translation).tolist()
    free_names = involved if gcps else involved[1:]
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
    measurements = [(point, shot, bearing(pixel,cameras[shot]),bearing_weight(pixel,cameras[shot],sigma))
                    for point in point_ids for shot, pixel, sigma in observations[point]]
    disabled = {str(p['id']) for p in controls.get('points', []) if not p.get('enabled', True)}
    distances = [d for d in controls.get('distances', []) if d.get('enabled', True)
                 and str(d['a']) not in disabled and str(d['b']) not in disabled]
    for distance in distances:
        if str(distance['a']) not in point_index or str(distance['b']) not in point_index or float(distance['meters']) <= 0 or float(distance.get('sigma_m', .01)) <= 0:
            raise ValueError('Invalid distance constraint or unknown endpoint')
    # Fix the arbitrary monocular scale only when no metric/world constraints exist.
    centers = {name: -r.T @ t for name, (r, t, _, _) in cameras.items()}
    scale_name = max(involved[1:], key=lambda n: np.linalg.norm(centers[n] - centers[involved[0]]))
    baseline = np.linalg.norm(centers[scale_name] - centers[involved[0]])
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
    problem=BearingProblem(cameras,free_names,point_ids,measurements,gcps,distances,scale_name,baseline,scale_anchor,involved[0])
    if 2*len(measurements)+3*len(gcps)+len(distances)+int(scale_anchor)<problem.nparams:
        raise ValueError('Too few independent constraints for camera/landmark refinement. Add distributed manual measurements or automatic multi-view tracks; a point-only cloud supplies no tracks.')
    neighbours={n:set() for n in involved}
    for point in point_ids:
        shots={n for n,_,_ in observations[point]}
        for shot in shots:neighbours[shot].update(shots)
    connected={involved[0]};pending=list(connected)
    while pending:
        for shot in neighbours[pending.pop()]-connected:connected.add(shot);pending.append(shot)
    if len(connected)!=len(involved):raise ValueError('Selected tracks form disconnected camera groups. Lower track filters, increase the landmark budget or add controls linking the groups.')
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
        try:result=solve_cuda(problem,initial,max_iterations,gpu_progress,**{k:settings[k] for k in ['loss','ftol','xtol','gtol']},loss_scale=settings['loss_scale'])
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
        result=least_squares(residual,initial,jac=jacobian,loss=settings['loss'],f_scale=settings['loss_scale'],max_nfev=max_iterations,x_scale='jac',**{k:settings[k] for k in ['ftol','xtol','gtol']})
    solve_seconds=time.perf_counter()-solve_started
    notify('retriangulate')
    from spherical_acceleration import robust_values
    objective=lambda x:robust_values(residual(x),settings['loss'],settings['loss_scale'])[0]
    if not np.isfinite(result.x).all() or objective(result.x)>objective(initial)+max(1e-10,objective(initial)*.001):
        raise RuntimeError('Refinement failed verification; input reconstruction preserved')
    poses, points = unpack(result.x)
    for name, (rotation, translation) in poses.items():
        component['shots'][name]['rotation'] = Rotation.from_matrix(rotation).as_rotvec().tolist()
        component['shots'][name]['translation'] = translation.tolist()
    for point, xyz in points.items():
        component['points'][point]['coordinates'] = xyz.tolist()
        component['points'][point]['observations'] = [{'shot_id': shot, 'pixel': pixel.tolist()} for shot, pixel, _ in observations[point]]
    check_results = []
    refined_cameras = {name: Camera(poses[name][0],poses[name][1],*cameras[name][2:],cameras[name].calibration) for name in names}
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
    # Untracked neural cloud centres cannot be corrected from changed poses.
    # Export only supported multi-view geometry, while retaining the immutable
    # source cloud at its original path. Held-out check markers live in controls.
    untracked=[p for p in component['points'] if p not in points and len(observations.get(p,[]))<2]
    for point in untracked:component['points'].pop(point)
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
              'solver_seconds':solve_seconds,'fallback_reason':fallback_reason,'settings':settings,
              'intrinsics':'fixed imported calibration','source_alignment':str(input_path.resolve()),
              'camera_families':sorted({c.calibration['projection_type'] for c in cameras.values()}),
              'optimized_camera_names':free_names,'unobserved_cameras_held_fixed':sorted(set(names)-set(involved)),
              'untracked_source_points_excluded':len(untracked),
              'automatic_tracks_after_view_filter':before_filter,'automatic_tracks_after_parallax_filter':after_parallax_filter,'automatic_tracks_after_budget':len(automatic),
              'image_policy':policy,'mask_layers':mask_records,'excluded_images':sorted(excluded),'ignored_manual_measurements':ignored_manual,'ignored_automatic_measurements':ignored_automatic}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notify('export')
    from control_points_model import atomic_json
    from refined_exports import export_refined
    atomic_json(output_path,[component],indent=None)
    report['exports']=export_refined(output_path,component)
    if all(not spherical(c) for c in cameras.values()):
        from manual_camera_geometry import to_pycolmap
        model=to_pycolmap(component);directory=output_path.with_suffix('.colmap');directory.mkdir(exist_ok=True)
        model.write(str(directory));model.write_text(str(directory));report['exports']['colmap_model']=str(directory)
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
    parser.add_argument('--settings',help='Local validated refinement settings JSON')
    args = parser.parse_args()
    def progress(event):print('ABU_PROGRESS '+json.dumps(event),flush=True)
    options=json.loads(Path(args.settings).read_text(encoding='utf-8')) if args.settings else None
    report=refine(args.input,args.output,args.controls,args.max_points,args.iterations,args.backend,progress,options)
    print('ABU_RESULT '+json.dumps(report),flush=True)
