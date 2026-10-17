"""Non-destructive image exclusions and original-resolution binary masks."""
import copy,hashlib,json,re
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image


def validate_policy(value=None):
    value=value or {}
    if not isinstance(value,dict):raise ValueError('Image policy must be an object.')
    names=value.get('excluded_images',[])
    if not isinstance(names,list) or any(not isinstance(n,str) or not n or Path(n).name!=n or '/' in n or '\\' in n for n in names):raise ValueError('Excluded images must be original basenames.')
    directory=value.get('mask_dir','')
    if not isinstance(directory,str):raise ValueError('Mask folder must be a path.')
    invert=value.get('invert_masks',False)
    if not isinstance(invert,bool):raise ValueError('Mask inversion must be true or false.')
    return dict(excluded_images=sorted(set(names)),mask_dir=directory,invert_masks=invert)


def runtime_path(value):
    import os
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',value):return Path('/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/'))
    return Path(value)


class Masks:
    def __init__(self,policy=None,limit_bytes=128*1024*1024):
        self.policy=validate_policy(policy);self.cache=OrderedDict();self.bytes=0;self.limit_bytes=limit_bytes;self.paths={}
    def path(self,name):
        if not self.policy['mask_dir']:return None
        directory=runtime_path(self.policy['mask_dir'])
        if not directory.is_dir():raise ValueError('Mask folder is unavailable: '+str(directory))
        candidates=list(dict.fromkeys([directory/(name+'.png'),directory/(Path(name).stem+'.mask.png'),directory/(Path(name).stem+'.png'),directory/name]))
        existing=[p for p in candidates if p.is_file()]
        if len(existing)>1:raise ValueError('Ambiguous mask names for '+name+'. Keep one matching mask.')
        return existing[0] if existing else None
    def array(self,name,size):
        path=self.path(name)
        if path is None:return None
        stamp=path.stat();key=(str(path),stamp.st_mtime_ns,stamp.st_size,tuple(size),self.policy['invert_masks'])
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        with Image.open(path) as image:
            if image.size!=tuple(size):raise ValueError(name+': mask dimensions must equal the original image; masks are never silently resized.')
            mask=np.asarray(image.convert('L'))>=128
        if self.policy['invert_masks']:mask=~mask
        mask.setflags(write=False);self.cache[key]=mask;self.bytes+=mask.nbytes
        while self.bytes>self.limit_bytes and len(self.cache)>1:self.bytes-=self.cache.popitem(last=False)[1].nbytes
        return mask
    def allowed(self,name,pixel,size):
        if name in self.policy['excluded_images']:return False
        if not np.isfinite(pixel).all():return False
        x,y=np.floor(pixel).astype(int)
        if not 0<=x<size[0] or not 0<=y<size[1]:return False
        mask=self.array(name,size)
        if mask is None:return True
        return bool(mask[y,x])
    def validate_catalog(self,sizes):
        paths={};records={}
        for name,size in sizes.items():
            path=self.path(name)
            if path is None:continue
            if path in paths:raise ValueError('One mask matches multiple original images: '+str(path))
            paths[path]=name;self.array(name,size)
            records[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return records


def effective_controls(data,sizes,strict=True):
    result=copy.deepcopy(data);masks=Masks(data.get('image_policy'));removed=0
    for point in result.get('points',[]):
        if not point.get('enabled',True):continue
        kept=[]
        for obs in point['observations']:
            name=obs['shot_id']
            if name in masks.policy['excluded_images']:removed+=1;continue
            if name not in sizes:kept.append(obs);continue
            if masks.allowed(name,obs['pixel'],sizes[name]):kept.append(obs)
            else:removed+=1
        point['observations']=kept
        if strict and point.get('enabled',True) and len(kept)<2:raise ValueError(point['id']+': fewer than two usable measurements after image exclusions/masks. Add views or disable this point.')
    return result,removed


def stage_masks(policy,sizes,output,fill_missing=False):
    masks=Masks(policy);records=masks.validate_catalog(sizes);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name in records:Image.fromarray(masks.array(name,sizes[name]).astype('uint8')*255).save(output/(name+'.png'))
    if fill_missing:
        # COLMAP treats a missing mask as an extraction error; stage explicit
        # white layers to preserve Abu's documented whole-image fallback.
        for name,size in sizes.items():
            if name not in records:Image.new('L',tuple(size),255).save(output/(name+'.png'))
    return records


def load_policy_file(path):
    return validate_policy(json.loads(Path(path).read_text(encoding='utf-8'))) if path else validate_policy()


def select_images(paths,policy=None):
    masks=Masks(policy);selected=[];omitted=[]
    for path in paths:
        name=Path(path).name
        if name in masks.policy['excluded_images']:omitted.append((name,'disabled'));continue
        if masks.path(name) is not None:
            with Image.open(path) as image:size=image.size
            mask=masks.array(name,size)
            if mask is not None and not mask.any():omitted.append((name,'fully masked'));continue
        selected.append(path)
    return selected,omitted
