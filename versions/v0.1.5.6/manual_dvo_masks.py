"""App-owned spherical source/target BA weight masking; upstream files unchanged."""
import numpy as np


class MaskAtlas:
    def __init__(self,paths,policy,height,width,resolution,device,max_bytes=128*1024*1024):
        import cv2,torch
        from PIL import Image
        from pathlib import Path
        from manual_image_policy import Masks
        h,w=height//resolution,width//resolution
        if len(paths)*h*w>max_bytes:raise ValueError('360DVO mask atlas exceeds 128 MiB. Reduce sequence/resolution; masks will not be ignored.')
        masks=Masks(policy);planes=[]
        for path in paths:
            with Image.open(path) as original:size=original.size
            mask=masks.array(Path(path).name,size)
            plane=np.ones((h,w),dtype='uint8') if mask is None else (cv2.resize(mask.astype('float32'),(w,h),interpolation=cv2.INTER_AREA)>=.999).astype('uint8')
            plane=cv2.erode(plane,np.ones((3,3),dtype='uint8'),borderType=cv2.BORDER_REPLICATE);planes.append(plane)
        self.planes=torch.from_numpy(np.stack(planes)).to(device);self.h=h;self.w=w
    def allowed(self,timestamps,coordinates):
        import torch
        finite=torch.isfinite(coordinates).all(-1);xy=torch.nan_to_num(coordinates).floor().long();t=timestamps.long()
        valid=finite & (t>=0)&(t<len(self.planes))&(xy[:,1]>=0)&(xy[:,1]<self.h)
        return valid & self.planes[t.clamp(0,len(self.planes)-1),xy[:,1].clamp(0,self.h-1),xy[:,0].remainder(self.w)].bool()
