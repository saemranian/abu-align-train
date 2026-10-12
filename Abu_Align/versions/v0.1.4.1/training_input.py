"""Immutable derived training inputs with actual resolution and disjoint splits."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from PIL import Image
from run_manifest import write_manifest


def prepare_training_input(dataset, output, factor='1'):
    source = Path(dataset).absolute()
    image_root = source / 'images' if (source / 'images').is_dir() else source
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    paths = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise ValueError('Prepare native panorama frames before starting training')
    factor = str(factor).lower()
    with Image.open(paths[0]) as sample:
        width, height = sample.size
    if width != 2 * height:
        raise ValueError('Native spherical training requires full 2:1 equirectangular images')
    if factor == 'auto':
        divisor = max(1, (width + 1599) // 1600)
    else:
        divisor = int(factor)
    if divisor not in range(1, 65):
        raise ValueError('Invalid training downscale factor')
    new_height = max(16, height // divisor)
    new_width = new_height * 2
    target = Path(output).absolute() / 'training_input'
    target.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(paths, target / 'source_manifest.json', 'pfgs360', {'width': new_width, 'height': new_height})
    marker = target / 'prepared.json'
    # Include poses/splits as well as images in cache identity.
    candidates = [source/'transforms.json', source/'opensfm'/'reconstruction.json', source/'reconstruction.json']
    digest = hashlib.sha256(manifest['fingerprint'].encode())
    for path in candidates:
        if path.is_file(): digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    if marker.exists() and json.loads(marker.read_text()).get('fingerprint') == fingerprint:
        return str(target)
    images = target / 'images'; images.mkdir(exist_ok=True)
    for path in paths:
        with Image.open(path) as image:
            if image.size != (width, height): raise ValueError('Mixed panorama dimensions are unsupported')
        destination = images / path.name
        # Existing hard links must be unlinked before writing a derived image.
        if destination.exists(): destination.unlink()
        if divisor == 1:
            try: os.link(path, destination)
            except OSError: shutil.copy2(path, destination)
        else:
            with Image.open(path) as image:
                if image.size != (width, height): raise ValueError('Mixed panorama dimensions are unsupported')
                image.resize((new_width, new_height), Image.Resampling.LANCZOS).save(destination)
    tf = source / 'transforms.json'
    if tf.is_file():
        data = json.loads(tf.read_text(encoding='utf-8'))
        if any('mask_path' in frame for frame in data.get('frames', [])):
            raise ValueError('Derived training masks must be prepared at the training resolution first')
        data.update(w=new_width, h=new_height, fl_x=float(new_height), fl_y=float(new_height), cx=new_width/2, cy=new_height/2)
        for frame in data.get('frames', []): frame['file_path'] = 'images/' + Path(frame['file_path']).name
        (target/'transforms.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    reconstruction = next((path for path in candidates[1:] if path.is_file()), None)
    names = [p.name for p in paths]
    if reconstruction:
        data = json.loads(reconstruction.read_text(encoding='utf-8'))
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError('Train reconstruction components separately')
        if not set(data[0]['shots']).issubset(names): raise ValueError('Reconstruction refers to missing photos')
        names = sorted(data[0]['shots'])
        for camera in data[0]['cameras'].values():
            camera.update(width=new_width, height=new_height)
        for point in data[0].get('points', {}).values():
            for observation in point.get('observations', []):
                observation['pixel'] = [observation['pixel'][0]*new_width/width,
                                        observation['pixel'][1]*new_height/height]
        folder = target/'opensfm'; folder.mkdir(exist_ok=True)
        (folder/'reconstruction.json').write_text(json.dumps(data), encoding='utf-8')
    # Three-frame minimum keeps training viable; tiny datasets have no claimed evaluation.
    test = names[::10] if len(names) >= 5 else []
    train = [name for name in names if name not in set(test)]
    (target/'metadata.json').write_text(json.dumps({'train': train, 'test': test}, indent=2), encoding='utf-8')
    marker.write_text(json.dumps({'fingerprint': fingerprint, 'source': str(source), 'width': new_width,
                                 'height': new_height, 'train': len(train), 'test': len(test)}, indent=2), encoding='utf-8')
    return str(target)
