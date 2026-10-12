"""Native equirectangular COLMAP backend with explicit CPU/CUDA capability checks.

python native_colmap_engine.py --input panoramas --output run --mapper incremental
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from run_manifest import write_manifest


def run(image_paths, output, mapper='incremental', device='auto', features=6000, width=2048, resume=False):
    import pycolmap
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((output/'run_manifest.json').read_text()) if (output/'run_manifest.json').exists() else None
    if (output/'database.db').exists() and not resume:
        raise ValueError('Use a fresh output directory to avoid stale feature/pose databases')
    if not image_paths: raise ValueError('No panorama images selected')
    directories = {str(Path(path).absolute().parent) for path in image_paths}
    if len(directories) != 1: raise ValueError('Selected images must share one folder')
    root = directories.pop()
    cuda = bool(pycolmap.has_cuda)
    if device == 'cuda' and not cuda:
        raise RuntimeError('This PyCOLMAP build lacks CUDA; install/build a CUDA backend before selecting GPU')
    gpu = cuda and device != 'cpu'
    manifest_path = output/('candidate_manifest.json' if resume else 'run_manifest.json')
    manifest = write_manifest(image_paths, manifest_path, 'native_colmap',
                              {'mapper': mapper, 'gpu': gpu, 'features': features, 'width': width, 'pycolmap': pycolmap.__version__})
    if any(item['width'] != 2*item['height'] for item in manifest['images']):
        raise ValueError('Native COLMAP input must contain full 2:1 panoramas')
    database = output/'database.db'
    if resume and (not previous or previous['fingerprint'] != manifest['fingerprint']):
        manifest_path.unlink(missing_ok=True)
        raise ValueError('Resume inputs/configuration differ from the stored manifest')
    if resume: manifest_path.unlink(missing_ok=True)
    reader = pycolmap.ImageReaderOptions(camera_model='EQUIRECTANGULAR')
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.use_gpu = gpu; extraction.max_image_size = width
    extraction.num_threads = min(8, os.cpu_count() or 4)
    extraction.sift.max_num_features = features
    matching = pycolmap.FeatureMatchingOptions()
    # PyCOLMAP 4.2.0's SIFT guided matcher aborts on spherical cameras despite
    # camera-ray verification support. Keep standard verified matching here.
    matching.use_gpu = gpu; matching.guided_matching = False
    matching.num_threads = min(8, os.cpu_count() or 4)
    pairing = pycolmap.SequentialPairingOptions(); pairing.overlap = 10
    # Retrieval vocabulary downloads must be an explicit reproducible setup step.
    pairing.loop_detection = False
    backend = pycolmap.Device.cuda if gpu else pycolmap.Device.cpu
    started = time.perf_counter()
    if not resume:
        pycolmap.extract_features(database, root, image_names=[item['name'] for item in manifest['images']],
                                 camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                                 extraction_options=extraction, device=backend)
        pycolmap.match_sequential(database, matching_options=matching, pairing_options=pairing, device=backend)
    sparse = output/'sparse'; sparse.mkdir(exist_ok=True)
    if mapper == 'global':
        options = pycolmap.GlobalPipelineOptions()
        models = pycolmap.global_mapping(database, root, sparse, options)
    else:
        options = pycolmap.IncrementalPipelineOptions()
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        # Walking panoramas have lower inter-frame parallax than object-orbit
        # captures; keep an explicit two-degree initialization threshold.
        options.mapper.init_min_tri_angle = 2.0
        options.min_model_size = 3
        models = pycolmap.incremental_mapping(database, root, sparse, options)
    if not models: raise RuntimeError('COLMAP could not reconstruct the selected panorama set')
    # Every component remains separate. Export observations for shared refinement.
    for index, reconstruction in models.items():
        component = {'cameras': {}, 'shots': {}, 'points': {}}
        for camera_id, camera in reconstruction.cameras.items():
            component['cameras'][str(camera_id)] = {'projection_type': 'spherical', 'width': camera.width, 'height': camera.height}
        for image in reconstruction.images.values():
            if not image.has_pose: continue
            pose = image.cam_from_world()
            from scipy.spatial.transform import Rotation
            component['shots'][image.name] = {'camera': str(image.camera_id),
                'rotation': Rotation.from_matrix(pose.rotation.matrix()).as_rotvec().tolist(), 'translation': pose.translation.tolist()}
        for point_id, point in reconstruction.points3D.items():
            observations = [{'shot_id': reconstruction.images[e.image_id].name,
                             'pixel': reconstruction.images[e.image_id].points2D[e.point2D_idx].xy.tolist()} for e in point.track.elements]
            component['points'][str(point_id)] = {'coordinates': point.xyz.tolist(), 'color': point.color.tolist(), 'observations': observations}
        folder = output/f'component_{index}'; folder.mkdir(exist_ok=True)
        (folder/'reconstruction.json').write_text(json.dumps([component]), encoding='utf-8')
        reconstruction.write(folder); reconstruction.write_text(folder)
    report = {'backend': 'colmap_cuda_features_matching' if gpu else 'colmap_cpu',
              'bundle_adjustment': 'Ceres; no Caspar for spherical cameras', 'mapper': mapper,
              'seconds': time.perf_counter()-started, 'components': len(models),
              'registered_images': sum(model.num_reg_images() for model in models.values()),
              'input_images': len(image_paths), 'pycolmap': pycolmap.__version__, 'manifest': manifest['fingerprint']}
    (output/'engine_provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--mapper', choices=['incremental', 'global'], default='incremental')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--features', type=int, default=6000); parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--resume', action='store_true', help='Resume mapping only after exact manifest verification')
    args = parser.parse_args()
    paths = sorted(path for path in Path(args.input).iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    print(json.dumps(run(paths, args.output, args.mapper, args.device, args.features, args.width, args.resume), indent=2))
