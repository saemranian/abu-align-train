"""Native-frame sparse cloud and lossless camera CSV beside a reconstruction."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def export_refined(path, component=None):
    path=Path(path)
    if component is None:
        data=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data,list):
            if len(data)!=1:raise ValueError('Export each reconstruction component separately.')
            component=data[0]
        else:component=data
    cloud=path.with_suffix('.points.ply');cameras=path.with_suffix('.cameras.csv');manifest=path.with_suffix('.exports.json')
    items=list(component.get('points',{}).values())
    xyz=np.asarray([p['coordinates'] for p in items],dtype=np.float64).reshape(-1,3)
    color=np.asarray([p.get('color',[180,180,180]) for p in items]).reshape(-1,3)
    finite=np.isfinite(xyz).all(axis=1);xyz=xyz[finite];color=np.clip(color[finite],0,255).astype(np.uint8)
    vertices=np.empty(len(xyz),dtype=[('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(['x','y','z']):vertices[k]=xyz[:,i]
    for i,k in enumerate(['red','green','blue']):vertices[k]=color[:,i]
    temporary=cloud.with_name(cloud.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(('ply\nformat binary_little_endian 1.0\ncomment Abu refined sparse cloud; native alignment frame\n'
                          f'element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n'
                          'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii'))
            vertices.tofile(stream)
        temporary.replace(cloud)
    finally:temporary.unlink(missing_ok=True)
    temporary=cameras.with_name(cameras.name+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['image','center_x','center_y','center_z','camera_to_world_qw','camera_to_world_qx','camera_to_world_qy','camera_to_world_qz','world_to_camera_rx','world_to_camera_ry','world_to_camera_rz','world_to_camera_tx','world_to_camera_ty','world_to_camera_tz','projection','width','height'])
            for name,shot in sorted(component['shots'].items()):
                rotvec=np.asarray(shot['rotation']);r=Rotation.from_rotvec(rotvec);t=np.asarray(shot['translation'])
                center=-r.as_matrix().T@t;q=r.inv().as_quat();camera=component['cameras'][shot['camera']]
                writer.writerow([name,*center,q[3],*q[:3],*rotvec,*t,camera['projection_type'],camera['width'],camera['height']])
        temporary.replace(cameras)
    finally:temporary.unlink(missing_ok=True)
    result={'reconstruction':str(path),'point_cloud':str(cloud),'camera_alignment_csv':str(cameras),'points':len(vertices),'cameras':len(component['shots']),
            'coordinate_frame':'native reconstruction frame; unchanged axes and scale','quaternion':'camera-to-world, scalar first (qw,qx,qy,qz)',
            'cloud_kind':'refined sparse landmarks; not dense photogrammetry or Gaussian splats'}
    from control_points_model import atomic_json
    atomic_json(manifest,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True)
    print(json.dumps(export_refined(parser.parse_args().input),indent=2))
