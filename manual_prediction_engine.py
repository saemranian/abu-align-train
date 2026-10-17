"""Local visual proposals on native originals; no cubemap dataset or auto measurements."""
import json,math,time
from collections import OrderedDict
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from manual_camera_geometry import Camera,bearing,calibration_parameters,spherical
from manual_image_policy import Masks

import os
MODEL_DIR=Path(__file__).parent/'.abu_models/manual_predictions'
USER_MODEL_DIR=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'AbuAlign'/'.abu_models'/'manual_predictions'
URLS={'aliked-n16.pth':'https://raw.githubusercontent.com/Shiaoming/ALIKED/main/models/aliked-n16.pth',
      'aliked_lightglue_v0-1_arxiv.pth':'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth'}


def setup_models():
    import hashlib,urllib.request
    target_dir=MODEL_DIR
    try:
        target_dir.mkdir(parents=True,exist_ok=True)
        probe=target_dir/'.write_probe'
        probe.write_text('',encoding='utf-8')
        probe.unlink()
    except OSError:
        target_dir=USER_MODEL_DIR
        target_dir.mkdir(parents=True,exist_ok=True)
    records={}
    for name,url in URLS.items():
        path=target_dir/name
        if not path.exists():
            temporary=path.with_suffix('.download');urllib.request.urlretrieve(url,temporary);temporary.replace(path)
        records[name]={'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    (target_dir/'models.json').write_text(json.dumps(records,indent=2));return records


def model_paths():
    import torch
    hub=Path(torch.hub.get_dir())/'checkpoints'
    folders=[MODEL_DIR,USER_MODEL_DIR,*[p/'.abu_models/manual_predictions' for p in list(Path(__file__).parents)[:5]],hub]
    return {name:next((folder/name for folder in folders if (folder/name).is_file()),None) for name in URLS}


class Matcher:
    def __init__(self,mode='auto',device='auto',keypoints=1024):
        self.kind='SIFT + mutual ratio/RANSAC';self.device='CPU';self.model=None;self.extractor=None;self.reason=''
        if mode!='sift':
            try:
                import torch
                from lightglue import ALIKED,LightGlue
                from unittest.mock import patch
                paths=model_paths()
                if not all(paths.values()):raise RuntimeError('Learned weights are not installed. Choose Set up local AI model; no images are sent.')
                selected='cuda' if device!='cpu' and torch.cuda.is_available() else 'cpu'
                if device=='cuda' and selected!='cuda':raise RuntimeError('CUDA is unavailable.')
                def offline(url,**kwargs):
                    name='aliked-n16.pth' if 'aliked-n16' in url else 'aliked_lightglue_v0-1_arxiv.pth' if 'aliked_lightglue' in url else None
                    if name is None:raise RuntimeError('Unexpected model URL blocked during offline inference.')
                    return torch.load(paths[name],map_location='cpu',weights_only=True)
                with patch('torch.hub.load_state_dict_from_url',offline):
                    self.extractor=ALIKED(max_num_keypoints=keypoints).eval().to(selected)
                    self.model=LightGlue(features='aliked',flash=True).eval().to(selected)
                self.kind='ALIKED + LightGlue (local)';self.device=selected.upper()
            except Exception as error:
                self.reason=str(error)
                self.model=self.extractor=None
                if mode=='learned':raise
        if self.model is None:self.detector=cv2.SIFT_create(nfeatures=keypoints,contrastThreshold=.015)
    def features(self,image,valid=None):
        if self.model is not None:
            import torch
            tensor=torch.from_numpy(image.copy()).permute(2,0,1).float().to(self.device.lower())/255
            with torch.inference_mode():value=self.extractor.extract(tensor,resize=None)
            if valid is not None:
                xy=value['keypoints'][0].detach().cpu().numpy().round().astype(int);keep=valid[np.clip(xy[:,1],0,len(valid)-1),np.clip(xy[:,0],0,valid.shape[1]-1)]
                index=torch.from_numpy(np.flatnonzero(keep)).to(tensor.device)
                for key in ['keypoints','descriptors','keypoint_scores']:value[key]=value[key].index_select(1,index)
            return value
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);kp,des=self.detector.detectAndCompute(gray,valid.astype('uint8')*255 if valid is not None else None)
        return (np.asarray([k.pt for k in kp],dtype='float32').reshape(-1,2),des)
    def matches(self,a,b):
        if self.model is not None:
            import torch
            if min(a['keypoints'].shape[1],b['keypoints'].shape[1])<6:return None
            with torch.inference_mode():result=self.model({'image0':a,'image1':b})
            pairs=result['matches'][0].detach().cpu().numpy();scores=result['scores'][0].detach().cpu().numpy()
            return a['keypoints'][0].detach().cpu().numpy()[pairs[:,0]],b['keypoints'][0].detach().cpu().numpy()[pairs[:,1]],scores
        xy,des=a;uv,other=b
        if des is None or other is None or min(len(des),len(other))<6:return None
        matcher=cv2.BFMatcher();forward=matcher.knnMatch(des,other,k=2);reverse=matcher.match(other,des);back={m.queryIdx:m.trainIdx for m in reverse}
        matches=[x for pair in forward if len(pair)==2 for x,y in [pair] if x.distance<.75*y.distance and back.get(x.trainIdx)==x.queryIdx]
        return xy[[m.queryIdx for m in matches]],uv[[m.trainIdx for m in matches]],np.asarray([max(0,1-m.distance/512) for m in matches])
    def close(self):
        self.model=self.extractor=None
        if self.device=='CUDA':
            import torch
            torch.cuda.empty_cache()


def ray_pixels(rays,camera):
    rays=np.asarray(rays,dtype=float);w,h=camera[2:]
    if spherical(camera):
        rays=rays/np.linalg.norm(rays,axis=-1,keepdims=True)
        return np.stack([((.5+np.arctan2(rays[...,0],rays[...,2])/(2*np.pi))%1)*w,(.5+np.arcsin(np.clip(rays[...,1],-1,1))/np.pi)*h],axis=-1)
    k,d=calibration_parameters(camera);shape=rays.shape[:-1]
    value=cv2.fisheye.projectPoints(rays.reshape(1,-1,3),np.zeros(3),np.zeros(3),k,d.reshape(4,1))[0] if camera.calibration['projection_type']=='opencv_fisheye' else cv2.projectPoints(rays.reshape(-1,3),np.zeros(3),np.zeros(3),k,d)[0]
    return value.reshape(*shape,2)


class PatchMap:
    def __init__(self,camera,center,fov=40,size=384):
        self.camera=camera;self.ray=bearing(center,camera);self.size=size;self.span=2*np.tan(np.radians(fov)/2)
        up=camera[0]@np.array([0.,1.,0.]);x=np.cross(up,self.ray)
        if np.linalg.norm(x)<1e-5:x=np.cross(np.array([1.,0.,0.]),self.ray)
        self.x=x/np.linalg.norm(x);self.y=np.cross(self.ray,self.x)
    def pixels(self,xy):
        uv=(np.asarray(xy)+.5-self.size/2)*self.span/self.size
        return ray_pixels(self.ray+uv[...,0,None]*self.x+uv[...,1,None]*self.y,self.camera)
    def extract(self,image,mask=None):
        yy,xx=np.mgrid[:self.size,:self.size];coords=self.pixels(np.stack([xx,yy],axis=-1));w,h=self.camera[2:]
        mx=((coords[...,0]+.5)*image.shape[1]/w-.5).astype('float32');my=((coords[...,1]+.5)*image.shape[0]/h-.5).astype('float32')
        valid=(coords[...,1]>=0)&(coords[...,1]<h)
        if not spherical(self.camera):valid &= (coords[...,0]>=0)&(coords[...,0]<w)
        result=cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP if spherical(self.camera) else cv2.BORDER_CONSTANT)
        if mask is not None:valid &= mask[np.clip(coords[...,1].astype(int),0,h-1),coords[...,0].astype(int)%w]
        result[~valid]=0;return result,valid


class PixelMap:
    """Unregistered perspective visual search uses pixels, never guessed optics."""
    def __init__(self,size,resolution=768):
        self.width,self.height=size;self.size=resolution;scale=resolution/max(size);self.output_size=tuple(round(v*scale) for v in size);self.scale=np.asarray(self.output_size)/size;self.pad=np.array([(resolution-v)//2 for v in self.output_size])
    def pixels(self,xy):return (np.asarray(xy)-self.pad+.5)/self.scale-.5
    def extract(self,image,mask=None):
        output=np.zeros((self.size,self.size,3),dtype='uint8');valid=np.zeros((self.size,self.size),dtype=bool);w,h=self.output_size;x,y=self.pad.astype(int)
        output[y:y+h,x:x+w]=cv2.resize(image,(w,h));valid[y:y+h,x:x+w]=True if mask is None else cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        output[~valid]=0;return output,valid


class PixelPatch:
    """Local source neighbourhood before perspective calibration is available."""
    def __init__(self,center,size):self.center=np.asarray(center);self.original_size=size;self.span=min(size)*.55
    def pixels(self,xy):return self.center+(np.asarray(xy)+.5-192)*self.span/384
    def extract(self,image,mask=None):
        yy,xx=np.mgrid[:384,:384];coords=self.pixels(np.stack([xx,yy],-1));w,h=self.original_size
        valid=(coords[...,0]>=0)&(coords[...,0]<w)&(coords[...,1]>=0)&(coords[...,1]<h)
        result=cv2.remap(image,((coords[...,0]+.5)*image.shape[1]/w-.5).astype('float32'),((coords[...,1]+.5)*image.shape[0]/h-.5).astype('float32'),cv2.INTER_LINEAR)
        if mask is not None:valid &= mask[np.clip(coords[...,1].astype(int),0,h-1),np.clip(coords[...,0].astype(int),0,w-1)]
        result[~valid]=0;return result,valid


def transfer(matches,center,size=384,target_size=None):
    if matches is None:return None
    a,b,scores=matches;distance=np.linalg.norm(a-center,axis=1);near=distance<size*.3;expanded=False
    if near.sum()<6:near=distance<size*.45;expanded=True
    a,b,scores=a[near],b[near],scores[near]
    if len(a)<6:return None
    matrix,inliers=cv2.findHomography(a,b,cv2.USAC_MAGSAC,3.,maxIters=3000,confidence=.999)
    if matrix is None or inliers is None:return None
    inliers=inliers.ravel().astype(bool)
    if inliers.sum()<6 or inliers.mean()<.45:return None
    hull=cv2.convexHull(a[inliers]);inside=cv2.pointPolygonTest(hull,tuple(map(float,center)),True)
    if inside < (0 if expanded else -size*.08):return None  # wider support must surround the query
    predicted=cv2.perspectiveTransform(np.asarray(center,dtype='float32').reshape(1,1,2),matrix)[0,0]
    reverse=cv2.perspectiveTransform(predicted.reshape(1,1,2),np.linalg.inv(matrix))[0,0]
    if not np.isfinite(predicted).all() or np.linalg.norm(reverse-center)>1. or np.any(predicted<3) or np.any(predicted>(target_size or size)-4):return None
    residual=np.linalg.norm(cv2.perspectiveTransform(a[inliers,None],matrix)[:,0]-b[inliers],axis=1)
    if np.median(residual)>2:return None
    return predicted,float(np.mean(scores[inliers])*inliers.mean()),int(inliers.sum()),float(np.median(residual))


def scan(request,emit=lambda item:None,cancelled=lambda:False):
    from control_points_model import point_geometry,projected_pixel
    from manual_camera_geometry import camera_map
    started=time.perf_counter();cameras=camera_map(request['component']) if request.get('component') else {};point=request['point'];folder=Path(request['image_dir']);masks=Masks(request.get('image_policy'))
    sizes=dict(request['image_sizes']);files=sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.webp'})
    if request.get('scope','all')=='registered':files=[p for p in files if p.name in cameras]
    files=[p for p in files if p.name not in masks.policy['excluded_images'] and p.name not in {o['shot_id'] for o in point['observations']}]
    cache=OrderedDict()
    def image(name):
        if name in cache:cache.move_to_end(name);return cache[name]
        with Image.open(folder/name) as original:
            sizes[name]=original.size;original.draft('RGB',(3072,1536));value=np.asarray(original.convert('RGB'))
        cache[name]=value
        while len(cache)>2:cache.popitem(last=False)
        return value
    def camera(name):
        if name in cameras:return cameras[name]
        w,h=sizes[name]
        if request.get('projection','spherical')=='spherical' and w==2*h:return Camera(np.eye(3),np.zeros(3),w,h)
        return None
    sources=[]
    matcher=Matcher(request.get('method','auto'),request.get('device','auto'),request.get('keypoints',1024))
    if matcher.device=='CUDA':
        import torch
        torch.cuda.reset_peak_memory_stats()
    try:
        for obs in point['observations']:
            name=obs['shot_id'];im=image(name);cam=camera(name)
            if not masks.allowed(name,obs['pixel'],sizes[name]):continue
            mapping=PatchMap(cam,obs['pixel'],32) if cam is not None else PixelPatch(obs['pixel'],sizes[name]);patch,valid=mapping.extract(im,masks.array(name,sizes[name]));features=matcher.features(patch,valid)
            count=features['keypoints'].shape[1] if matcher.model is not None else len(features[0])
            if cam is not None and count<40:
                mapping=PatchMap(cam,obs['pixel'],64);patch,valid=mapping.extract(im,masks.array(name,sizes[name]));features=matcher.features(patch,valid)
            sources.append((name,mapping,features))
            if len(sources)>=3:break
        if not sources:raise ValueError('Add a usable measured point in an original image first.')
        geometric_point=dict(point,observations=[o for o in point['observations'] if o['shot_id'] in cameras and o['shot_id'] not in masks.policy['excluded_images']]);xyz,description,errors=point_geometry(geometric_point,cameras)
        if errors and max(errors.values())>1:xyz=None
        emit(dict(event='start',images=len(files),sources=len(sources),method=matcher.kind,device=matcher.device,fallback=matcher.reason))
        results=[];failures=0
        for index,path in enumerate(files):
            if cancelled():break
            try:
                im=image(path.name);cam=camera(path.name)
                projected=projected_pixel(xyz,cam) if xyz is not None and path.name in cameras else None
                centers=[None] if cam is None else [projected] if projected is not None else ([[cam[2]*lon/6,cam[3]*lat] for lat in [.17,.5,.83] for lon in range(6)] if spherical(cam) else [[cam[2]/2,cam[3]/2]])
                proposals=[]
                for center in centers:
                    if cancelled():break
                    mapping=PixelMap(sizes[path.name]) if cam is None else PatchMap(cam,center,48 if projected is not None else 100);patch,valid=mapping.extract(im,masks.array(path.name,sizes[path.name]));features=matcher.features(patch,valid)
                    for source,source_map,source_features in sources:
                        match=transfer(matcher.matches(source_features,features),np.array([191.5,191.5]),target_size=mapping.size if isinstance(mapping,PixelMap) else 384)
                        if match is None:continue
                        pixel=mapping.pixels(match[0])
                        if cam is not None and spherical(cam):pixel[0]%=cam[2]
                        if projected is not None:
                            angle=np.degrees(np.arccos(np.clip(bearing(pixel,cam)@bearing(projected,cam),-1,1)))
                            if angle>request.get('geometry_gate_deg',5.):continue
                        if masks.allowed(path.name,pixel,sizes[path.name]):proposals.append(dict(image=path.name,pixel=pixel.tolist(),score=match[1],inliers=match[2],local_error_px=match[3],source=source,size=list(sizes[path.name])))
                if proposals:
                    proposals.sort(key=lambda v:v['score'],reverse=True);best=proposals[0]
                    def discrepancy(v):
                        return np.degrees(np.arccos(np.clip(bearing(best['pixel'],cam)@bearing(v['pixel'],cam),-1,1))) if cam is not None else np.linalg.norm(np.array(best['pixel'])-v['pixel'])/max(sizes[path.name])*180
                    agreeing=[v for v in proposals if discrepancy(v)<.35]
                    supporters={v['source'] for v in agreeing};ambiguous=any(discrepancy(v)>1 and v['score']>best['score']*.85 for v in proposals)
                    best.update(supporting_views=len(supporters),status='Ambiguous · inspect' if ambiguous else 'Multiple views agree · verify' if len(supporters)>=2 else 'One-view proposal · inspect',registered=path.name in cameras)
                    if len(sources)>1 and len(supporters)<2:best['score']*=.5
                    if ambiguous:best['score']*=.3
                    results.append(best);emit(dict(event='candidate',**best))
            except Exception as error:failures+=1;emit(dict(event='image_error',image=path.name,error=str(error)))
            emit(dict(event='progress',completed=index+1,total=len(files)))
        results.sort(key=lambda v:v['score'],reverse=True)
        peak=0.
        if matcher.device=='CUDA':peak=torch.cuda.max_memory_allocated()/1048576
        report=dict(event='complete',processed=index+1 if files else 0,total=len(files),candidates=len(results),failures=failures,seconds=time.perf_counter()-started,method=matcher.kind,device=matcher.device,peak_vram_mib=peak,cancelled=cancelled())
        emit(report);return results,report
    finally:matcher.close();cache.clear()
