"""Native M54 object inspection and axis/ring mouse manipulation adapters."""
import copy
import tkinter as tk
import numpy as np
from scipy.spatial.transform import Rotation
from abu_m54 import SceneTransform,rotation_numbers,gizmo_segments,screen_segment_hit


def layer_matrix(viewer,layer):
    matrix=viewer.transforms[layer].matrix()
    if layer=='B' and viewer.comparison:
        fit=viewer.comparison['report']['transform'];base=np.eye(4)
        base[:3,:3]=float(fit['scale'])*np.asarray(fit['rotation']);base[:3,3]=fit['translation'];matrix=base@matrix
    return matrix


def selection_record(viewer):
    kind=getattr(viewer,'selection_kind','');layer='A';name=''
    if kind=='null' and viewer.null_selected in viewer.nulls.items:
        name=viewer.null_selected;value=viewer.nulls.items[name];rotation=Rotation.from_euler('xyz',value['rotation_xyz_degrees'],degrees=True).as_matrix()
        return dict(kind='null',name=name,layer='World',position=value['position'],**rotation_numbers(rotation))
    if kind=='cloud' and viewer.cloud_selected is not None:layer,index=viewer.cloud_selected;name=f'Display sample {index}'
    elif kind=='camera':layer=viewer.selected_camera_layer;name=viewer.camera_selected
    elif kind=='point':name=viewer.selected
    else:return None
    point=viewer.selected_world()
    if point is None:return None
    result=dict(kind=kind,name=name,layer=layer,position=np.asarray(point).tolist(),rotation_xyz_degrees=None,quaternion_wxyz=None)
    scene=viewer.data if layer=='A' else viewer.comparison['b'] if viewer.comparison else None
    if kind=='camera' and scene and name in scene.get('camera_rotations',{}):
        r=scene['camera_rotations'][name];result.update(rotation_numbers(r))
        result['world_to_camera_rotvec_radians']=Rotation.from_matrix(r.T).as_rotvec().tolist();result['world_to_camera_translation']=(-r.T@point).tolist()
    result['model_frame_rotation_xyz_degrees']=rotation_numbers(layer_matrix(viewer,layer)[:3,:3]/np.cbrt(np.linalg.det(layer_matrix(viewer,layer)[:3,:3])))['rotation_xyz_degrees']
    return result


class NativeEditing:
    def __init__(self,viewer):
        self.viewer=viewer;self.enabled=tk.BooleanVar(viewer,False);self.mode=tk.StringVar(viewer,'Move');self.target=tk.StringVar(viewer,'Selected null');self.axes=tk.StringVar(viewer,'World')
        self.active=None;self.pending=None;self.job=None
    def spec(self):
        v=self.viewer;target=self.target.get()
        if target=='Selected null':
            item=v.nulls.items.get(v.null_selected)
            if item is None:return None
            matrix=np.eye(4);matrix[:3,:3]=Rotation.from_euler('xyz',item['rotation_xyz_degrees'],degrees=True).as_matrix();matrix[:3,3]=item['position']
            return dict(kind='null',name=v.null_selected,position=matrix[:3,3],rotation=matrix[:3,:3],matrix=matrix)
        layer='A' if target=='Layer A' else 'B';scene=v.raw_data if layer=='A' else v.compare.b
        if scene is None or layer=='B' and not v.comparison:return None
        matrix=layer_matrix(v,layer);r=matrix[:3,:3]/np.cbrt(np.linalg.det(matrix[:3,:3]));position=matrix[:3,:3]@scene['center']+matrix[:3,3]
        return dict(kind='model',layer=layer,position=position,rotation=r,matrix=matrix)
    def geometry(self):
        if not self.enabled.get():return None,[]
        spec=self.spec()
        if spec is None:return None,[]
        v=self.viewer;size=v.distance*np.tan(np.pi/8)*90/max(100,v.surface.winfo_height())*v.scale
        axes=spec['rotation'] if self.axes.get()=='Local' else np.eye(3)
        spec=dict(spec,size=size,axes=axes)
        return spec,gizmo_segments(spec['position'],axes,size,self.mode.get())
    def draw(self):
        v=self.viewer
        if v.nulls.items:
            segments=[]
            for item in v.nulls.items.values():
                r=Rotation.from_euler('xyz',item['rotation_xyz_degrees'],degrees=True).as_matrix();p=np.asarray(item['position']);size=v.scale*.008
                for axis in range(3):segments.append([p-r[:,axis]*size,p+r[:,axis]*size])
            v.draw_arrays((np.asarray(segments).reshape(-1,3)-v.origin)/v.scale,mode=1,color=(1.,.8,.4))
        spec,geometry=self.geometry()
        if spec is None:return
        v.gl.glDisable(0x0B71)
        for axis,segments in enumerate(geometry):
            v.gl.glLineWidth(4. if self.active and self.active['axis']==axis else 2.5)
            v.draw_arrays((segments.reshape(-1,3)-v.origin)/v.scale,mode=1,color=[(.95,.3,.3),(.3,.9,.4),(.3,.6,1.)][axis])
        v.gl.glLineWidth(1.);v.gl.glEnable(0x0B71)
    def press(self,event):
        spec,geometry=self.geometry()
        if spec is None:return False
        v=self.viewer;hits=[]
        for axis,segments in enumerate(geometry):
            screen,valid,_=v.project((segments.reshape(-1,3)-v.origin)/v.scale);distance=screen_segment_hit(screen,valid,event.x,event.y)
            if distance is not None:hits.append((distance,axis))
        if not hits:return False
        axis=min(hits)[1];direction=spec['axes'][:,axis]
        points=np.array([spec['position'],spec['position']+direction*spec['size']]);screen,valid,_=v.project((points-v.origin)/v.scale)
        self.active=dict(spec,axis=axis,start=np.array([event.x,event.y]),direction=direction,screen_direction=screen[1]-screen[0],changed=False,mode=self.mode.get())
        if spec['kind']=='model':self.active['original_transform']=copy.deepcopy(v.transforms[spec['layer']]);self.active['registered']=v.compare.registration.get() if spec['layer']=='B' else False
        self.active['redo']=copy.deepcopy(v.nulls.redo if spec['kind']=='null' else v.transform_redo[spec['layer']])
        self.active['start_vector']=self.plane_vector(event,spec['position'],direction)
        v.surface.grab_set();v.info.set(f"{self.mode.get()} {'XYZ'[axis]} · Esc cancels · Undo in the corresponding object/layer tools");v.request();return True
    def plane_vector(self,event,position,normal):
        v=self.viewer;view,projection,w,h=v.matrices();inverse=np.linalg.inv(projection@view)
        xy=[event.x/w*2-1,1-event.y/h*2]
        points=np.array([inverse@np.array([*xy,z,1.]) for z in [-1,1]]);points=points[:,:3]/points[:,3,None]
        origin=points[0];direction=points[1]-origin;direction/=np.linalg.norm(direction)
        pivot=(np.asarray(position)-v.origin)/v.scale;den=float(normal@direction)
        if abs(den)<1e-6:return None
        distance=float(normal@(pivot-origin))/den
        if distance<0:return None
        vector=origin+direction*distance-pivot;length=np.linalg.norm(vector)
        return vector/length if length>1e-9 else None
    def move(self,event):
        if self.active is None:return False
        a=self.active;delta=np.array([event.x,event.y])-a['start']
        if np.linalg.norm(delta)<3 and not a['changed']:return True
        matrix=a['matrix'].copy()
        if a['mode']=='Move':
            axis=a['screen_direction'];length=np.dot(axis,axis)
            if length<36:self.viewer.info.set('Axis is end-on. Orbit slightly to expose the handle.');return True
            offset=a['direction']*np.dot(delta,axis)/length*a['size'];matrix[:3,3]+=offset
        else:
            start=a['start_vector'];end=self.plane_vector(event,a['position'],a['direction'])
            if start is None or end is None:self.viewer.info.set('Rotation ring is edge-on. Orbit slightly to expose its plane.');return True
            angle=np.arctan2(a['direction']@np.cross(start,end),np.clip(start@end,-1,1));r=Rotation.from_rotvec(a['direction']*angle).as_matrix()
            matrix[:3,:3]=r@matrix[:3,:3];matrix[:3,3]=a['position']+r@(matrix[:3,3]-a['position'])
        if not a['changed']:
            a['changed']=True
            if a['kind']=='null':self.viewer.nulls.checkpoint()
            else:
                layer=a['layer'];self.viewer.transform_history[layer].append(SceneTransform.from_matrix(a['matrix']));self.viewer.transform_history[layer][:]=self.viewer.transform_history[layer][-32:];self.viewer.transform_redo[layer].clear()
        self.pending=matrix
        if self.job is None:self.job=self.viewer.after(16,self.flush)
        return True
    def flush(self):
        if self.job is not None:self.viewer.after_cancel(self.job)
        self.job=None
        if self.pending is None or self.active is None:return
        matrix,self.pending=self.pending,None;a=self.active;v=self.viewer
        if a['kind']=='null':v.nulls.set_pose(a['name'],matrix[:3,3],rotation_numbers(matrix[:3,:3])['rotation_xyz_degrees'],checkpoint=False);v.request()
        else:v.apply_transform(a['layer'],SceneTransform.from_matrix(matrix))
        v.notify_inspector()
        if v.tools_window and v.tools_window.winfo_exists():v.tools_window.read()
    def release(self,event=None):
        if self.active is None:return False
        if self.job is not None:self.viewer.after_cancel(self.job);self.job=None
        self.flush();active=self.active;self.active=None;self.viewer.surface.grab_release();self.viewer.request()
        if active['changed']:self.viewer.editor.audit.event('m54_gizmo_drag',target=active.get('layer',active.get('name')),operation=active['mode'],axis='XYZ'[active['axis']])
        return True
    def cancel(self,restore=True):
        if self.active is None:return
        if self.job is not None:self.viewer.after_cancel(self.job);self.job=None
        a=self.active;self.pending=None;v=self.viewer
        if restore and a['changed']:
            if a['kind']=='null':
                v.nulls.set_pose(a['name'],a['matrix'][:3,3],rotation_numbers(a['matrix'][:3,:3])['rotation_xyz_degrees'],checkpoint=False);v.nulls.history.pop();v.nulls.redo=a['redo']
            else:
                layer=a['layer'];v.apply_transform(layer,a['original_transform']);v.transform_history[layer].pop();v.transform_redo[layer]=a['redo']
                if layer=='B':v.compare.registration.set(a['registered']);v.compare.apply()
        self.active=None
        try:v.surface.grab_release()
        except tk.TclError:pass
        v.request();v.notify_inspector()
