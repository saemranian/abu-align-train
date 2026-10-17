"""Local calibrated-camera adapters shared by Manual control points and BA.

Pixels are original-image coordinates; OpenSfM and COLMAP conventions are kept.
No upstream code, images or calibration is changed. Intrinsics stay fixed.
"""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class Camera(tuple):
    def __new__(cls, rotation, translation, width, height, calibration=None):
        value=super().__new__(cls,(rotation,translation,int(width),int(height)))
        value.calibration=copy.deepcopy(calibration or {'projection_type':'spherical'})
        return value
    def __reduce__(self):return Camera,(*self,self.calibration)


def spherical(camera):return getattr(camera,'calibration',{}).get('projection_type','spherical') in {'spherical','equirectangular'}


def calibration_parameters(camera):
    c=camera.calibration;w,h=camera[2:];d=max(w,h);kind=c['projection_type']
    if kind=='perspective':
        fx=fy=float(c['focal'])*d;cx=w/2;cy=h/2;dist=[c.get('k1',0),c.get('k2',0),0,0,0]
    elif kind=='brown':
        fx=float(c['focal_x'])*d;fy=float(c['focal_y'])*d;cx=w/2+float(c.get('c_x',0))*d;cy=h/2+float(c.get('c_y',0))*d
        dist=[c.get('k1',0),c.get('k2',0),c.get('p1',0),c.get('p2',0),c.get('k3',0)]
    elif kind in {'pinhole','opencv','opencv_fisheye'}:
        fx,fy,cx,cy=[float(c[k]) for k in ['fx','fy','cx','cy']];dist=c.get('distortion',[0,0,0,0,0])
    else:raise ValueError(f'Unsupported calibrated camera model: {kind}. Export a supported model with original intrinsics first.')
    k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    dist=np.asarray(dist,dtype=float)
    if not np.isfinite(k).all() or not np.isfinite(dist).all() or min(fx,fy)<=0:raise ValueError('Invalid camera calibration.')
    return k,dist


def bearing(pixel,camera):
    x,y=np.asarray(pixel,dtype=float);w,h=camera[2:]
    if spherical(camera):
        lon=(x/w-.5)*2*np.pi;lat=(.5-y/h)*np.pi
        return np.array([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)])
    k,dist=calibration_parameters(camera);xy=np.array([[[x,y]]],dtype=float)
    uv=(cv2.fisheye.undistortPoints(xy,k,dist.reshape(4,1)) if camera.calibration['projection_type']=='opencv_fisheye' else cv2.undistortPoints(xy,k,dist))[0,0]
    ray=np.r_[uv,1.];return ray/np.linalg.norm(ray)


def bearings_batch(pixels,camera):
    pixels=np.asarray(pixels,dtype=float).reshape(-1,2)
    if spherical(camera):
        lon=(pixels[:,0]/camera[2]-.5)*2*np.pi;lat=(.5-pixels[:,1]/camera[3])*np.pi
        return np.column_stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)])
    k,d=calibration_parameters(camera);xy=pixels.reshape(-1,1,2)
    uv=(cv2.fisheye.undistortPoints(xy,k,d.reshape(4,1)) if camera.calibration['projection_type']=='opencv_fisheye' else cv2.undistortPoints(xy,k,d)).reshape(-1,2)
    rays=np.column_stack([uv,np.ones(len(uv))]);return rays/np.linalg.norm(rays,axis=1)[:,None]


def project(xyz,camera):
    r,t,w,h=camera;ray=r@np.asarray(xyz)+t
    if np.linalg.norm(ray)<1e-10:return None
    if spherical(camera):
        ray=ray/np.linalg.norm(ray)
        return [((.5+np.arctan2(ray[0],ray[2])/(2*np.pi))%1)*w,(.5+np.arcsin(np.clip(ray[1],-1,1))/np.pi)*h]
    if ray[2]<=1e-10:return None
    k,dist=calibration_parameters(camera)
    if camera.calibration['projection_type']=='opencv_fisheye':xy=cv2.fisheye.projectPoints(ray.reshape(1,1,3),np.zeros(3),np.zeros(3),k,dist.reshape(4,1))[0][0,0]
    else:xy=cv2.projectPoints(ray.reshape(1,1,3),np.zeros(3),np.zeros(3),k,dist)[0][0,0]
    return xy.tolist() if 0<=xy[0]<w and 0<=xy[1]<h else None


def bearing_weight(pixel,camera,sigma):
    """Local pixel-noise whitening, including the antipodal radial safeguard."""
    ray=bearing(pixel,camera)
    if spherical(camera):return np.eye(3)/max(2*np.pi/camera[2]*sigma,1e-8)
    pixel=np.asarray(pixel,dtype=float);step=.001
    jac=np.column_stack([(bearing(pixel+np.eye(2)[i]*step,camera)-bearing(pixel-np.eye(2)[i]*step,camera))/(2*step) for i in range(2)])
    tangent=np.linalg.pinv(jac)/sigma;radial=ray/max(np.linalg.norm(jac)/np.sqrt(2)*sigma,1e-8)
    return np.vstack([tangent,radial])


def camera_map(component):
    cameras={}
    for name,shot in component['shots'].items():
        if Path(name).name!=name:raise ValueError('Choose original images in one folder with unique basenames.')
        c=component['cameras'][shot['camera']];w,h=int(c['width']),int(c['height'])
        r,t=np.asarray(shot['rotation'],float),np.asarray(shot['translation'],float)
        if min(w,h)<=0 or r.shape!=(3,) or t.shape!=(3,) or not np.isfinite([r,t]).all():raise ValueError('Invalid camera dimensions or poses.')
        cam=Camera(Rotation.from_rotvec(r).as_matrix(),t,w,h,c)
        if spherical(cam):
            if w!=2*h:raise ValueError('Spherical cameras require original full 2:1 panoramas.')
        else:calibration_parameters(cam)
        cameras[name]=cam
    return cameras


def colmap_component(model):
    """Capture valid sparse observations; dense point centres are not tracks."""
    cameras={};shots={};points={}
    for ident,c in model.cameras.items():
        name=str(c.model_name);p=np.asarray(c.params);record={'projection_type':'pinhole','width':c.width,'height':c.height}
        if name=='PINHOLE':fx,fy,cx,cy=p;dist=[]
        elif name in {'SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL'}:fx,cx,cy=p[:3];fy=fx;dist=[*(p[3:]),0,0,0] if name!='SIMPLE_PINHOLE' else []
        elif name in {'OPENCV','FULL_OPENCV','OPENCV_FISHEYE'}:
            fx,fy,cx,cy=p[:4];dist=p[4:].tolist();record['projection_type']='opencv_fisheye' if name=='OPENCV_FISHEYE' else 'opencv'
        else:raise ValueError(f'Unsupported COLMAP calibration {name}; no lens convention will be guessed.')
        if name=='SIMPLE_RADIAL':dist=[p[3],0,0,0,0]
        if name=='RADIAL':dist=[p[3],p[4],0,0,0]
        record.update(fx=float(fx),fy=float(fy),cx=float(cx),cy=float(cy),distortion=list(dist) or [0,0,0,0,0],colmap_model=name)
        cameras[str(ident)]=record
    for ident,image in model.images.items():
        if hasattr(image,'has_pose') and not image.has_pose:continue
        pose=image.cam_from_world();shots[image.name]={'camera':str(image.camera_id),'rotation':Rotation.from_matrix(pose.rotation.matrix()).as_rotvec().tolist(),'translation':np.asarray(pose.translation).tolist()}
    for ident,point in model.points3D.items():
        obs=[];seen=set()
        for element in point.track.elements:
            image=model.images[element.image_id]
            if image.name in shots:
                if image.name in seen:raise ValueError('A COLMAP track has multiple measurements in one image.')
                xy=np.asarray(image.points2D[element.point2D_idx].xy);cam=model.cameras[image.camera_id]
                if not np.isfinite(xy).all() or not (0<=xy[0]<cam.width and 0<=xy[1]<cam.height):raise ValueError('COLMAP track pixels do not match original camera dimensions; correct crop/resize calibration before import.')
                seen.add(image.name);obs.append({'shot_id':image.name,'pixel':np.asarray(image.points2D[element.point2D_idx].xy).tolist()})
        points[str(ident)]={'coordinates':np.asarray(point.xyz).tolist(),'color':np.asarray(point.color).tolist(),'observations':obs}
    return {'cameras':cameras,'shots':shots,'points':points,'manual_camera_adapter':'COLMAP calibrated fixed-intrinsics'}


def load_alignment(path):
    from abu_m54 import load_calibrated_alignment
    return load_calibrated_alignment(path)


def to_pycolmap(component):
    """Export calibrated perspective results, including manual sparse tracks."""
    import pycolmap
    model=pycolmap.Reconstruction();ids={};image_ids={}
    for number,(name,c) in enumerate(component['cameras'].items(),1):
        cam=Camera(np.eye(3),np.zeros(3),c['width'],c['height'],c)
        if spherical(cam):raise ValueError('Use the existing spherical exporter for spherical models.')
        k,d=calibration_parameters(cam);params=[k[0,0],k[1,1],k[0,2],k[1,2]]
        kind='OPENCV_FISHEYE' if c['projection_type']=='opencv_fisheye' else 'FULL_OPENCV' if len(d)==8 else 'OPENCV' if np.any(d) else 'PINHOLE'
        if kind!='PINHOLE':params+=list(d) if kind in ['FULL_OPENCV','OPENCV_FISHEYE'] else list(np.pad(d,(0,max(0,5-len(d)))))[:4]
        if kind=='OPENCV' and len(d)>4 and d[4]!=0:kind='FULL_OPENCV';params=[k[0,0],k[1,1],k[0,2],k[1,2],*list(np.pad(d,(0,8-len(d))))]
        camera=pycolmap.Camera(camera_id=number,model=kind,width=c['width'],height=c['height'],params=params)
        model.add_camera_with_trivial_rig(camera);ids[name]=number
    measurements={name:[] for name in component['shots']};tracks={}
    for ident,point in component['points'].items():
        track=[]
        for obs in point.get('observations',[]):
            if obs['shot_id'] not in measurements:continue
            rows=measurements[obs['shot_id']];track.append((obs['shot_id'],len(rows)));rows.append(obs['pixel'])
        tracks[ident]=track
    for number,(name,shot) in enumerate(component['shots'].items(),1):
        image=pycolmap.Image(image_id=number,name=name,camera_id=ids[shot['camera']],points2D=[pycolmap.Point2D(xy=np.array(p),point3D_id=pycolmap.INVALID_POINT3D_ID) for p in measurements[name]])
        pose=pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec(shot['rotation']).as_matrix()),np.asarray(shot['translation']))
        model.add_image_with_trivial_frame(image,pose);image_ids[name]=number
    for ident,point in component['points'].items():
        if not tracks[ident]:continue
        track=pycolmap.Track()
        for name,index in tracks[ident]:track.add_element(image_ids[name],index)
        model.add_point3D(np.asarray(point['coordinates']),track,np.asarray(point.get('color',[180,180,180]),np.uint8))
    return model
