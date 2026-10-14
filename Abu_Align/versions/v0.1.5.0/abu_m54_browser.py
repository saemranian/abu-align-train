"""M54 browser adapter: shared transforms, selection pivots and ground grid.

Serves only loopback. No remote service, image upload or installed-package edits.
"""
import copy
import json
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from abu_m54 import SceneTransform, transform_scene, from_records, to_records, ray_pick, grid_lines, load_scene, export_copy, NAME


def guard_server_lifecycle(server):
    """Installed 0.2.23 stop signals asynchronously; finish its owned thread.

The background thread prints after its event loop stops. Joining that exact thread
prevents Python's buffered-output shutdown race and releases the loopback port.
"""
    if getattr(server,'_m54_stop_guard',False):return
    server._m54_stop_guard=True;original=server.stop
    loop=getattr(getattr(server,'_websock_server',None),'_background_event_loop',None)
    ident=getattr(loop,'_thread_id',None);worker=next((t for t in threading.enumerate() if t.ident==ident),None)
    state={'stopped':False}
    def stop():
        if not state['stopped']:state['stopped']=True;original()
        if worker is not None and worker is not threading.current_thread():worker.join(timeout=3.)
    server.stop=stop


class BrowserTools:
    def __init__(self,server,provider,refresh,normalized=False,default_up='Z',up_control=None):
        self.server=server;self.provider=provider;self.refresh=refresh;self.normalized=normalized
        server.request_share_url=lambda verbose=True:None
        guard_server_lifecycle(server)
        self.transforms={'A':SceneTransform(),'B':SceneTransform()};self.selection=None;self.nodes=[];self.updating=False
        self.history={'A':[], 'B':[]};self.redo={'A':[], 'B':[]};self.lock=threading.RLock()
        with server.gui.add_folder('Abu M54 · Scene tools',expand_by_default=False):
            self.pick=server.gui.add_checkbox('Pick cloud sample',False,hint='Click the displayed cloud to inspect a sample. This does not create a control measurement.')
            self.auto=server.gui.add_checkbox('Orbit on selection',False)
            self.up=up_control if up_control is not None else server.gui.add_dropdown('World up',('Z','Y','X','-Z','-Y','-X'),initial_value=default_up)
            self.focus=server.gui.add_button('Orbit around selected item')
            self.selection_info=server.gui.add_markdown('Select a camera/control or enable cloud picking.')
            self.grid=server.gui.add_checkbox('Ground grid',False)
            self.plane=server.gui.add_dropdown('Grid plane',('XY','XZ','YZ'),initial_value='XY')
            self.height=server.gui.add_number('Grid elevation',0.,step=.1)
            self.layer=server.gui.add_dropdown('Transform layer',('A','B'),initial_value='A')
            self.pivot=server.gui.add_dropdown('Rotation / scale pivot',('Scene centre','Selected item','World origin'),initial_value='Scene centre')
            self.translation=server.gui.add_vector3('Move · scene units',(0.,0.,0.),step=.1)
            self.angles=server.gui.add_vector3('Rotate XYZ · degrees',(0.,0.,0.),step=1.)
            self.scale=server.gui.add_number('Uniform scale',1.,min=.000001,step=.01)
            self.apply=server.gui.add_button('Apply transform preview')
            self.actions=server.gui.add_button_group('Transform history',('Undo','Redo','Reset'))
            self.gizmo_enabled=server.gui.add_checkbox('Move / rotate gizmo',False)
            self.output=server.gui.add_text('Local output path',initial_value='',hint='A new path on this PC. Full source copies are exported, not display samples.')
            self.export=server.gui.add_button('Export transformed copy')
            self.status=server.gui.add_markdown('Cloud, cameras and triangulated controls move together. Sources stay unchanged. Uniform scale is numeric.')
        self.layer.on_update(self.read_fields);self.apply.on_click(self.apply_fields);self.actions.on_click(self.history_action)
        self.focus.on_click(self.focus_selection);self.export.on_click(self.export_copy)
        @self.up.on_update
        def up_changed(event):
            direction=self.up_vector()
            for client in self.server.get_clients().values():client.camera.up_direction=direction
        for control in [self.grid,self.plane,self.height,self.gizmo_enabled]:control.on_update(lambda e:self.render_tools())
        @server.scene.on_pointer_event('click')
        def pointer(event):
            if not self.pick.value or event.ray_origin is None:return
            scenes=self.effective();hits=[]
            for key,scene in scenes.items():
                points=self.display(scene['points'],scene)
                index=ray_pick(points,event.ray_origin,event.ray_direction)
                if index is not None:hits.append((np.linalg.norm(points[index]-event.ray_origin),key,index))
            if hits:
                _,key,index=min(hits);self.select(scenes[key]['points'][index],event.client,'cloud sample',key)
    def source_scenes(self):
        result={}
        for key,value in self.provider().items():result[key]=value if 'centers' in value else from_records(value)
        return result
    def up_vector(self):
        value=self.up.value.replace(' up','');direction=np.eye(3)[{'X':0,'Y':1,'Z':2}[value[-1]]]
        return -direction if value.startswith('-') else direction
    def effective(self):return {key:transform_scene(scene,self.transforms[key]) for key,scene in self.source_scenes().items()}
    def apply_records(self,layers):
        return {key:dict(value,**{k:v for k,v in to_records(transform_scene(from_records(value),self.transforms[key])).items() if k in ['points','colors','cameras']}) for key,value in layers.items()}
    def display(self,points,scene):
        if not self.normalized:return np.asarray(points)
        base=self.display_frame(scene)
        return (np.asarray(points)-base['center'])/base['extent']
    def display_frame(self,scene):return self.source_scenes().get('A',scene)
    def select(self,position,client=None,kind='item',layer='A'):
        self.selection=(np.asarray(position,dtype=float),layer)
        self.selection_info.content=f'**Selected {kind} · {layer}**\n\nXYZ: '+', '.join(f'{v:.8g}' for v in position)
        if self.auto.value:self.focus_selection(type('Event',(),{'client':client})())
    def focus_selection(self,event=None):
        if self.selection is None:self.status.content='Select a point, camera or cloud sample first.';return
        point,key=self.selection;scene=self.effective().get(key)
        if scene is None:return
        target=self.display(point,scene)
        clients=[event.client] if event is not None and getattr(event,'client',None) is not None else list(self.server.get_clients().values())
        for client in clients:
            # Recentring the orbit preserves direction and distance.
            offset=np.asarray(client.camera.position)-np.asarray(client.camera.look_at)
            client.camera.look_at=target;client.camera.position=target+offset
    def read_fields(self,event=None):
        value=self.transforms[self.layer.value]
        self.translation.value=tuple(value.translation);self.angles.value=tuple(value.angles);self.scale.value=value.scale
    def apply_fields(self,event=None):
        try:
            key=self.layer.value;scene=self.source_scenes().get(key)
            if scene is None:raise ValueError('Load this reconstruction layer first.')
            if self.pivot.value=='Selected item':
                if self.selection is None or self.selection[1]!=key:raise ValueError('Select an item in this layer first.')
                pivot=(np.linalg.inv(self.transforms[key].matrix())@np.r_[self.selection[0],1])[:3]
            else:pivot=scene['center'] if self.pivot.value=='Scene centre' else np.zeros(3)
            value=SceneTransform(np.array(self.translation.value),np.array(self.angles.value),float(self.scale.value),pivot);value.matrix()
            self.history[key].append(copy.deepcopy(self.transforms[key]));self.history[key]=self.history[key][-32:];self.redo[key].clear();self.transforms[key]=value
            self.selection=None;self.refresh();self.status.content='Transform applied to preview. Export a copy to retain it.'
        except (ValueError,np.linalg.LinAlgError) as error:self.status.content=str(error)
    def history_action(self,event):
        key=self.layer.value;action=self.actions.value
        if action=='Reset':self.history[key].append(copy.deepcopy(self.transforms[key]));self.transforms[key]=SceneTransform();self.redo[key].clear()
        else:
            source=self.history[key] if action=='Undo' else self.redo[key];dest=self.redo[key] if action=='Undo' else self.history[key]
            if not source:return
            dest.append(copy.deepcopy(self.transforms[key]));self.transforms[key]=source.pop()
        self.selection=None;self.read_fields();self.refresh()
    def render_tools(self):
        if self.updating:return
        with self.lock:
            for node in self.nodes:node.remove()
            self.nodes=[];scenes=self.effective();scene=scenes.get(self.layer.value) or next(iter(scenes.values()),None)
            if scene is None:return
            segments,step=grid_lines(scene['center'],scene['extent'],self.plane.value,float(self.height.value))
            self.nodes.append(self.server.scene.add_line_segments('/m54/grid',self.display(segments,scene).astype(np.float32),(60,77,84),line_width=1.,visible=self.grid.value))
            key=self.layer.value if self.layer.value in scenes else next(iter(scenes));value=self.transforms[key]
            source_center=self.source_scenes()[key]['center'];matrix=value.matrix()
            position=matrix[:3,:3]@source_center+matrix[:3,3];rotation=Rotation.from_euler('xyz',value.angles,degrees=True).as_quat()
            gizmo=self.server.scene.add_transform_controls('/m54/gizmo',scale=.15 if self.normalized else scene['extent']*.15,
                     position=tuple(self.display(position,scene)),wxyz=tuple(rotation[[3,0,1,2]]),visible=self.gizmo_enabled.value and key==self.layer.value,depth_test=False)
            self.nodes.append(gizmo)
            @gizmo.on_update
            def changed(event):
                if self.updating or getattr(event,'client',None) is None:return
                with self.lock:
                    q=np.asarray(gizmo.wxyz);r=Rotation.from_quat(q[[1,2,3,0]])
                    base=self.display_frame(scene)
                    offset=np.asarray(gizmo.position)*base['extent']+base['center'] if self.normalized else np.asarray(gizmo.position)
                    self.history[key].append(copy.deepcopy(self.transforms[key]));self.history[key]=self.history[key][-32:]
                    self.redo[key].clear();self.transforms[key]=SceneTransform(offset-source_center,r.as_euler('xyz',degrees=True),self.transforms[key].scale,source_center)
                    self.updating=True
                    try:self.read_fields();self.refresh()
                    finally:self.updating=False
    def export_copy(self,event=None):
        key=self.layer.value;scene=self.source_scenes().get(key);value=self.output.value.strip()
        if scene is None or not value:self.status.content='Load a layer and enter a new local output path.';return
        transform=copy.deepcopy(self.transforms[key])
        if 'm54_base_transform' in scene:transform=SceneTransform.from_matrix(transform.matrix()@np.asarray(scene['m54_base_transform']))
        self.status.content='Exporting full source copy…'
        def work():
            try:
                report=export_copy(scene['source'],value,transform);self.status.content='Export complete: '+json.dumps(report)
            except Exception as error:self.status.content='Export failed: '+str(error)
        threading.Thread(target=work,daemon=True).start()


class M54Browser:
    """Standalone browser surface, also used by native M54 host adapters."""
    def __init__(self,port=9015):
        import viser
        self.server=viser.ViserServer(host='127.0.0.1',port=port,label=NAME,verbose=False)
        self.server.request_share_url=lambda verbose=True:None
        self.server.gui.configure_theme(dark_mode=True,show_share_button=True)
        self.scene=None;self.nodes=[];self.closed=False;self.lock=threading.RLock()
        self.server.gui.add_markdown('## Abu 3D Viewer M54\nLocal cloud and camera inspection. Orbit/pan/zoom; source data stays unchanged.')
        self.fit=self.server.gui.add_button('Fit scene');self.info=self.server.gui.add_markdown('Open a reconstruction locally.')
        self.tools=BrowserTools(self.server,lambda:{'A':self.scene} if self.scene else {},self.render,normalized=True)
        self.fit.on_click(lambda e:self.fit_client(e.client) if e.client else [self.fit_client(c) for c in self.server.get_clients().values()])
        self.server.on_client_connect(lambda c:self.fit_client(c))
    @property
    def url(self):return f'http://127.0.0.1:{self.server.get_port()}/'
    def set_scene(self,scene):
        if scene is None:return
        if self.scene and self.scene.get('source')!=scene.get('source'):self.tools.transforms={'A':SceneTransform(),'B':SceneTransform()}
        self.scene=scene;self.render()
    def fit_client(self,client):
        client.camera.look_at=np.zeros(3);client.camera.position=np.array([1.,-1.,1.]);client.camera.up_direction=self.tools.up_vector()
        client.camera.near=.0001;client.camera.far=10000.
    def render(self):
        with self.lock:
            if self.closed or not self.scene:return
            for node in self.nodes:node.remove()
            self.nodes=[];scene=self.tools.effective()['A'];normalize=lambda p:self.tools.display(p,scene)
            if len(scene['points']):self.nodes.append(self.server.scene.add_point_cloud('/m54/scene/cloud',normalize(scene['points']).astype(np.float32),scene['colors'],point_size=.003,point_shape='circle'))
            for index,(name,position) in enumerate(scene['centers'].items()):
                q=Rotation.from_matrix(scene['camera_rotations'][name]).as_quat()
                node=self.server.scene.add_camera_frustum(f'/m54/scene/camera/{index}',fov=np.pi/3,aspect=1.5,scale=.015,wxyz=q[[3,0,1,2]],position=normalize(position),color=(112,219,192))
                node.on_click(lambda e,p=position,n=name:self.tools.select(p,e.client,'camera '+n));self.nodes.append(node)
            for index,(name,control) in enumerate(scene.get('controls',{}).items()):
                node=self.server.scene.add_icosphere(f'/m54/scene/control/{index}',radius=.008,position=normalize(control['position']),color=(255,205,115))
                node.on_click(lambda e,p=control['position'],n=name:self.tools.select(p,e.client,'control '+n));self.nodes.append(node)
            self.info.content=f"{len(scene['points']):,} display points · {len(scene['centers'])} cameras. Units are the source reconstruction's units."
            self.tools.render_tools()
    def close(self):
        self.closed=True;self.server.stop()
