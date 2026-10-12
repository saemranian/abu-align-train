"""Independent named-camera evaluation; no ground-truth data is used by alignment.

python benchmark_360.py --estimate reconstruction.json --reference ground_truth.json
    --alignment sim3 --output benchmark.json
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def camera_poses(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError('Evaluate disconnected components separately')
        data = data[0]
    result = {}
    for name, shot in data['shots'].items():
        rotation = Rotation.from_rotvec(shot['rotation']).as_matrix()
        result[name] = (-rotation.T @ np.asarray(shot['translation']), rotation.T)
    return result


def similarity(source, target, estimate_scale=True):
    source, target = np.asarray(source), np.asarray(target)
    a, b = source.mean(axis=0), target.mean(axis=0)
    centered = source - a
    variance = np.mean(np.sum(centered**2, axis=1))
    if variance < 1e-12:
        raise ValueError('Stationary trajectory cannot determine a similarity transform')
    u, singular, vt = np.linalg.svd((target - b).T @ centered / len(source))
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular * signs) / variance) if estimate_scale else 1.0
    return scale, rotation, b - scale * rotation @ a


def evaluate(estimate_path, reference_path, alignment='sim3', step=1):
    estimate, reference = camera_poses(estimate_path), camera_poses(reference_path)
    names = sorted(set(estimate) & set(reference))
    if len(names) < 3:
        raise ValueError('Trajectory evaluation requires three shared image names')
    if step < 1 or step >= len(names):
        raise ValueError('RPE interval must be positive and shorter than the trajectory')
    source = np.array([estimate[n][0] for n in names])
    target = np.array([reference[n][0] for n in names])
    scale, rotation, translation = (1., np.eye(3), np.zeros(3)) if alignment == 'none' else similarity(source, target, alignment == 'sim3')
    aligned = scale * (source @ rotation.T) + translation
    ate = np.linalg.norm(aligned - target, axis=1)
    errors_t, errors_r = [], []
    for i in range(len(names) - step):
        j = i + step
        r1 = rotation @ estimate[names[i]][1]
        r2 = rotation @ estimate[names[j]][1]
        q1, q2 = reference[names[i]][1], reference[names[j]][1]
        errors_t.append(np.linalg.norm(r1.T @ (aligned[j] - aligned[i]) - q1.T @ (target[j] - target[i])))
        errors_r.append(np.degrees(Rotation.from_matrix((q1.T @ q2).T @ (r1.T @ r2)).magnitude()))
    return {'shared_images': len(names), 'reference_images': len(reference),
            'registration_fraction': len(names) / len(reference), 'missing': sorted(set(reference) - set(estimate)),
            'alignment': alignment, 'fitted_scale': scale, 'units': 'reference_coordinate_units',
            'ate_rmse': float(np.sqrt(np.mean(ate**2))), 'ate_median': float(np.median(ate)),
            'ate_max': float(ate.max()), 'rpe_step': step,
            'rpe_translation_rmse': float(np.sqrt(np.mean(np.array(errors_t)**2))),
            'rpe_rotation_rmse_deg': float(np.sqrt(np.mean(np.array(errors_r)**2)))}


def spherical_psnr(prediction, reference):
    """Solid-angle weighted RGB MSE; polar ERP pixels have smaller surface area."""
    prediction, reference = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    if prediction.shape != reference.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError('Metric images must have matching HxWx3 shapes')
    h, w, _ = prediction.shape
    weights = np.sin(np.pi * (np.arange(h) + .5) / h)
    mse = np.sum((prediction - reference)**2 * weights[:, None, None]) / (weights.sum() * w * 3)
    return float(-10 * np.log10(max(mse, 1e-12)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--alignment', choices=['none', 'se3', 'sim3'], default='sim3')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = evaluate(args.estimate, args.reference, args.alignment, args.step)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
