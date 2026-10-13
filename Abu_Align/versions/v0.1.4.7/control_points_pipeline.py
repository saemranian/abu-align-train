"""Early validation of manual controls against the actual selected native images."""
import json
from pathlib import Path
from PIL import Image
from control_points_model import validate_controls


def validate_selected_controls(path, image_paths):
    data=validate_controls(json.loads(Path(path).read_text(encoding='utf-8')))
    selected={Path(p).name:Path(p) for p in image_paths}
    if len(selected)!=len(image_paths):raise ValueError('Control alignment requires unique image basenames.')
    sizes={}
    active=[p for p in data['points'] if p.get('enabled',True)]
    if not active:raise ValueError('No enabled control points. Disable the controls option, or add measured points.')
    for point in active:
        if len(point['observations'])<2:raise ValueError(f"{point['id']}: add measurements in at least two images.")
        for observation in point['observations']:
            name=observation['shot_id']
            if name not in selected:raise ValueError(f'Control image {name} is not in the selected input. Check the dataset, frame stride and frame limit.')
            if name not in sizes:
                with Image.open(selected[name]) as image:sizes[name]=image.size
            width,height=sizes[name]
            if width!=2*height:raise ValueError('Manual spherical controls require original full 2:1 panoramas.')
            saved=data.get('image_sizes',{}).get(name)
            if saved is not None and list(saved)!=[width,height]:raise ValueError(f'{name}: control-file image dimensions differ from the selected originals.')
            x,y=observation['pixel']
            if not 0<=x<width or not 0<=y<height:raise ValueError(f'{name}: control measurement is outside the original image dimensions.')
    gcps=[p for p in active if p.get('type')=='gcp']
    if gcps:
        import numpy as np
        xyz=np.asarray([p['position'] for p in gcps])
        if len(gcps)<3 or np.linalg.matrix_rank(xyz-xyz.mean(0))<2:raise ValueError('Ground-control refinement requires at least three non-collinear local XYZ points.')
    return data
