"""M54 browser adapter: shared transforms, selection pivots and ground grid.

Serves only loopback. No remote service, image upload or installed-package edits.
"""
import copy
import asyncio
import json
import threading
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from abu_m54 import SceneTransform, transform_scene, from_records, to_records, ray_pick, grid_lines, load_scene, export_copy, NAME,NullObjects,rotation_numbers


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
    async def drain_connections():
        # This loop belongs exclusively to this Viser server. Its old producer
        # coroutines leave pending waiters when stopped with a browser attached.
        current=asyncio.current_task();connections={}
        for task in asyncio.all_tasks():
            coro=task.get_coro()
            # Websockets awaits ws_handler inside conn_handler, rather than
            # making ws_handler a top-level task. Inspect that owned await chain.
            for _ in range(12):
                frame=getattr(coro,'cr_frame',None)
                connection=frame.f_locals.get('connection') if frame is not None else None
                if connection is not None and hasattr(connection,'transport') and hasattr(connection,'close'):connections[id(connection)]=connection
                coro=getattr(coro,'cr_await',None)
                if coro is None:break
        if connections:
            await asyncio.gather(*(asyncio.wait_for(c.close(code=1001),timeout=1.) for c in connections.values()),return_exceptions=True)
            for connection in connections.values():
                transport=getattr(connection,'transport',None)
                if transport is not None:transport.close()
        tasks=[t for t in asyncio.all_tasks() if t is not current and not getattr(t.get_coro(),'__qualname__','').endswith('.start_server')]
        for task in tasks:task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
    def stop():
        if not state['stopped']:
            state['stopped']=True
            if loop is not None and loop.is_running() and worker is not threading.current_thread():
                try:asyncio.run_coroutine_threadsafe(drain_connections(),loop).result(timeout=2.)
                except (TimeoutError,RuntimeError):pass
            original()
        if worker is not None and worker is not threading.current_thread():worker.join(timeout=3.)
    server.stop=stop


class BrowserTools:
    def __init__(self,server,provider,refresh,normalized=False,default_up='Z',up_control=None):
        self.server=server;self.provider=provider;self.refresh=refresh;self.normalized=normalized
        server.request_share_url=lambda verbose=True:None
        guard_server_lifecycle(server)
        self.transforms={'A':SceneTransform(),'B':SceneTransform()};self.selection=None;self.nodes=[];self.updating=False
        self.history={'A':[], 'B':[]};self.redo={'A':[], 'B':[]};self.lock=threading.RLock()
        self.last_edit={'A':0.,'B':0.};self.edit_pivot={'A':None,'B':None}
        self.nulls=NullObjects();self.selected_null='';self.selection_meta=None;self.gizmo=None;self.gizmo_in_callback=False;self.null_updating=False;self.last_null_edit=0.;self.gizmo_target_key=None
        with server.gui.add_folder('Abu M54 · Scene tools',expand_by_default=False):
            self.pick=server.gui.add_checkbox('Pick one cloud sample',False,hint='Temporarily enters picking mode. Click once; orbit/pan returns automatically. Switch off to cancel. This does not create a control measurement.')
            self.auto=server.gui.add_checkbox('Orbit on selection',False)
            self.up=up_control if up_control is not None else server.gui.add_dropdown('World up',('Z','Y','X','-Z','-Y','-X'),initial_value=default_up)
            self.focus=server.gui.add_button('Orbit around selected item')
            self.grid=server.gui.add_checkbox('Ground grid',False)
            self.plane=server.gui.add_dropdown('Grid plane',('XY','XZ','YZ'),initial_value='XY')
            self.height=server.gui.add_number('Grid elevation',0.,step=.1)
            self.layer=server.gui.add_dropdown('Transform layer',('A','B'),initial_value='A')
            self.pivot=server.gui.add_dropdown('Rotation / scale pivot',('Scene centre','Selected item','World origin'),initial_value='Scene centre')
            self.translation=server.gui.add_vector3('Move · scene units',(0.,0.,0.),step=.1)
            self.angles=server.gui.add_vector3('Rotate XYZ · degrees',(0.,0.,0.),step=1.)
            self.scale=server.gui.add_number('Uniform scale',1.,min=.000001,step=.01)
            self.apply=server.gui.add_button('Validate transform values')
            self.actions=server.gui.add_button_group('Transform history',('Undo','Redo','Reset'))
            self.gizmo_enabled=server.gui.add_checkbox('Move / rotate gizmo',False)
            self.gizmo_target=server.gui.add_dropdown('Gizmo target',('Reconstruction layer','Selected null'),initial_value='Reconstruction layer')
            self.gizmo_mode=server.gui.add_dropdown('Gizmo operation',('Move + rotate','Move','Rotate'),initial_value='Move + rotate')
            self.output=server.gui.add_text('Local output path',initial_value='',hint='A new path on this PC. Full source copies are exported, not display samples.')
            self.export=server.gui.add_button('Export transformed copy')
            self.status=server.gui.add_markdown('Cloud, cameras and triangulated controls move together. Sources stay unchanged. Uniform scale is numeric.')
        with server.gui.add_folder('Abu M54 · Objects & 3D nulls',expand_by_default=False):
            self.selection_info=server.gui.add_markdown('Select a camera/control or enable cloud picking to inspect its exact world pose. Ordinary points have no stored rotation.')
            self.null_list=server.gui.add_dropdown('World null',('—',),initial_value='—')
            self.null_add=server.gui.add_button('Add null here',hint='Creates an independent oriented reference at the selected object, or at the scene centre when nothing is selected.')
            self.null_actions=server.gui.add_button_group('Null history',('Delete','Undo','Redo'))
            self.null_position=server.gui.add_vector3('Null position XYZ',(0.,0.,0.),step=.01)
            self.null_angles=server.gui.add_vector3('Null rotation XYZ · degrees',(0.,0.,0.),step=1.)
            self.null_file=server.gui.add_text('Local nulls file',initial_value='',hint='Path on this PC to a .m54nulls.json file. Sources in the file are metadata; they are not opened automatically.')
            self.null_files=server.gui.add_button_group('Null file',('Save','Load'))
            self.null_info=server.gui.add_markdown('Nulls are independent oriented world references, separate from alignment controls. Fields update live; drag RGB arrows/rings with Mouse gizmo enabled. Point samples and control points have XYZ but no stored rotation.')
        self.null_list.on_update(self.read_null_fields);self.null_add.on_click(lambda e:self.null_action(action='Add at selection'));self.null_actions.on_click(self.null_action);self.null_files.on_click(self.null_file_action)
        self.null_position.on_update(self.apply_null_fields);self.null_angles.on_update(self.apply_null_fields)
        self.layer.on_update(self.read_fields);self.apply.on_click(self.apply_fields);self.actions.on_click(self.history_action)
        for field in [self.translation,self.angles,self.scale]:field.on_update(self.apply_fields)
        self.focus.on_click(self.focus_selection);self.export.on_click(self.export_copy)
        @self.up.on_update
        def up_changed(event):
            direction=self.up_vector()
            for client in self.server.get_clients().values():client.camera.up_direction=direction
        for control in [self.grid,self.plane,self.height,self.gizmo_enabled,self.gizmo_target,self.gizmo_mode]:control.on_update(lambda e:self.render_tools())
        def pointer(event):
            if not self.pick.value or event.ray_origin is None:return
            scenes=self.effective();hits=[]
            for key,scene in scenes.items():
                points=self.display(scene['points'],scene)
                index=ray_pick(points,event.ray_origin,event.ray_direction)
                if index is not None:hits.append((np.linalg.norm(points[index]-event.ray_origin),key,index))
            if hits:
                _,key,index=min(hits);self.select(scenes[key]['points'][index],event.client,'cloud sample',key)
            self.pick.value=False
            if server.scene._scene_pointer_cb is pointer:server.scene.remove_pointer_callback()
        self.pointer=pointer
        @self.pick.on_update
        def pick_changed(event):
            if self.pick.value:
                server.scene.on_pointer_event('click')(pointer)
                self.status.content='Pick one cloud sample, then normal left-drag orbit/right-drag pan resumes. Switch picking off to cancel.'
            elif server.scene._scene_pointer_cb is pointer:server.scene.remove_pointer_callback()
        self.read_null_fields()
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
    def select(self,position,client=None,kind='item',layer='A',name=None,rotation=None):
        self.selection=(np.asarray(position,dtype=float),layer)
        name=name or (kind[7:] if kind.startswith('camera ') else kind[8:] if kind.startswith('control ') else None)
        source=(np.linalg.inv(self.transforms[layer].matrix())@np.r_[position,1])[:3] if layer in self.transforms else np.asarray(position)
        self.selection_meta=dict(kind=kind,name=name,layer=layer,source=source,rotation=rotation)
        self.update_selection_info()
        if self.auto.value:self.focus_selection(type('Event',(),{'client':client})())
    def update_selection_info(self):
        if not self.selection_meta:return
        m=self.selection_meta;key=m['layer'];scene=self.effective().get(key);p=np.asarray(m['source']);r=None
        if key=='World':
            item=self.nulls.items.get(m['name'])
            if item is None:return
            p=np.asarray(item['position']);r=Rotation.from_euler('xyz',item['rotation_xyz_degrees'],degrees=True).as_matrix()
        elif scene:
            if m['kind'].startswith('camera '):
                p=scene['centers'].get(m['name'],p);r=scene['camera_rotations'].get(m['name'],m['rotation'])
                if r is None and m['rotation'] is not None:r=m['rotation']
            elif m['kind'].startswith('control ') and m['name'] in scene.get('controls',{}):p=scene['controls'][m['name']]['position']
            else:p=self.transforms[key].matrix()[:3,:3]@p+self.transforms[key].matrix()[:3,3]
        self.selection=(np.asarray(p),key)
        text=f"**Selected {m['kind']} · {key}**\n\nWorld XYZ: "+', '.join(f'{v:.14g}' for v in p)
        if r is None:text+='\n\nRotation: **not defined** for an ordinary point. Create a null for an editable orientation.'
        else:
            values=rotation_numbers(r);text+='\n\nLocal-to-world XYZ degrees: '+', '.join(f'{v:.12g}' for v in values['rotation_xyz_degrees'])
            text+='\n\nQuaternion w, x, y, z: '+', '.join(f'{v:.12g}' for v in values['quaternion_wxyz'])
            if m['kind'].startswith('camera '):
                text+='\n\nWorld-to-camera rotvec radians: '+', '.join(f'{v:.12g}' for v in Rotation.from_matrix(r.T).as_rotvec())
                text+='\n\nWorld-to-camera translation: '+', '.join(f'{v:.12g}' for v in -r.T@p)
        if key in self.transforms:text+='\n\nModel frame XYZ degrees: '+', '.join(f'{v:.12g}' for v in self.transforms[key].angles)
        self.selection_info.content=text
    def focus_selection(self,event=None):
        if self.selection is None:self.status.content='Select a point, camera or cloud sample first.';return
        point,key=self.selection;scenes=self.effective();scene=scenes.get(key) or next(iter(scenes.values()),None)
        if scene is None:
            if key!='World':return
            scene=from_records({})
        target=self.display(point,scene)
        clients=[event.client] if event is not None and getattr(event,'client',None) is not None else list(self.server.get_clients().values())
        for client in clients:
            # Recentring the orbit preserves direction and distance.
            offset=np.asarray(client.camera.position)-np.asarray(client.camera.look_at)
            client.camera.position=target+offset;client.camera.look_at=target
    def read_fields(self,event=None):
        self.last_edit={'A':0.,'B':0.};self.edit_pivot={'A':None,'B':None}
        self.updating=True
        value=self.transforms[self.layer.value]
        self.translation.value=tuple(value.translation);self.angles.value=tuple(value.angles);self.scale.value=value.scale
        self.updating=False
    def apply_fields(self,event=None):
        if self.updating:return
        try:
            key=self.layer.value;scene=self.source_scenes().get(key)
            if scene is None:raise ValueError('Load this reconstruction layer first.')
            if self.pivot.value=='Selected item':
                if self.selection is None:
                    if self.edit_pivot[key] is None:raise ValueError('Select an item in this layer first.')
                    pivot=self.edit_pivot[key]
                else:
                    if self.selection[1]!=key:raise ValueError('Select an item in this layer first.')
                    pivot=(np.linalg.inv(self.transforms[key].matrix())@np.r_[self.selection[0],1])[:3];self.edit_pivot[key]=pivot
            else:pivot=scene['center'] if self.pivot.value=='Scene centre' else np.zeros(3)
            value=SceneTransform(np.array(self.translation.value),np.array(self.angles.value),float(self.scale.value),pivot);value.matrix()
            if np.allclose(value.matrix(),self.transforms[key].matrix(),rtol=0,atol=1e-12):return
            now=time.monotonic()
            if now-self.last_edit[key]>.5:self.history[key].append(copy.deepcopy(self.transforms[key]));self.history[key]=self.history[key][-32:];self.redo[key].clear()
            self.last_edit[key]=now;self.transforms[key]=value
            self.update_selection_info();self.refresh();self.status.content='Transform applied to preview. Export a copy to retain it.'
        except (ValueError,np.linalg.LinAlgError) as error:self.status.content=str(error)
    def history_action(self,event):
        key=self.layer.value;action=self.actions.value
        if action=='Reset':self.history[key].append(copy.deepcopy(self.transforms[key]));self.transforms[key]=SceneTransform();self.redo[key].clear()
        else:
            source=self.history[key] if action=='Undo' else self.redo[key];dest=self.redo[key] if action=='Undo' else self.history[key]
            if not source:return
            dest.append(copy.deepcopy(self.transforms[key]));self.transforms[key]=source.pop()
        self.update_selection_info();self.read_fields();self.refresh()
    def select_null(self,name,client=None):
        if name not in self.nulls.items:return
        self.selected_null=name;self.null_list.value=name;self.select(self.nulls.items[name]['position'],client,'null '+name,'World',name=name)
        self.gizmo_target.value='Selected null';self.read_null_fields();self.render_tools()
    def read_null_fields(self,event=None):
        if self.null_updating:return
        self.null_updating=True
        try:
            options=('—',)+tuple(self.nulls.items)
            if self.null_list.options!=options:self.null_list.options=options
            if self.null_list.value not in options:self.null_list.value='—'
            self.selected_null=self.null_list.value if self.null_list.value in self.nulls.items else ''
            value=self.nulls.items.get(self.selected_null)
            self.null_position.disabled=self.null_angles.disabled=value is None
            if value:self.null_position.value=tuple(value['position']);self.null_angles.value=tuple(value['rotation_xyz_degrees'])
            self.null_info.content=('**Unsaved null changes.** ' if self.nulls.dirty else '')+'Nulls are independent world references. Save/load on this PC; source geometry stays unchanged.'
        finally:self.null_updating=False
        if event is not None and self.selected_null:
            self.select(self.nulls.items[self.selected_null]['position'],kind='null '+self.selected_null,layer='World',name=self.selected_null);self.render_tools()
    def apply_null_fields(self,event=None):
        if self.null_updating or self.selected_null not in self.nulls.items:return
        try:
            now=time.monotonic();self.nulls.set_pose(self.selected_null,self.null_position.value,self.null_angles.value,checkpoint=now-self.last_null_edit>.5);self.last_null_edit=now
            self.update_selection_info();self.render_tools()
        except ValueError as error:self.status.content=str(error)
    def null_action(self,event=None,action=None):
        try:
            action=action or self.null_actions.value
            if action=='Add at selection':
                p=self.selection[0] if self.selection is not None else self.source_scenes().get('A',{}).get('center',np.zeros(3))
                angles=np.zeros(3)
                if self.selection_meta:
                    m=self.selection_meta;scene=self.effective().get(m['layer']);r=scene.get('camera_rotations',{}).get(m['name']) if scene else None
                    if r is not None:angles=rotation_numbers(r)['rotation_xyz_degrees']
                name=self.nulls.add(p,angles);self.null_list.options=('—',)+tuple(self.nulls.items);self.select_null(name);self.gizmo_enabled.value=True
            elif action=='Delete':self.nulls.delete(self.selected_null);self.selected_null='';self.null_list.value='—';self.selection_meta=None;self.selection=None
            else:self.nulls.step(action=='Redo')
            self.last_null_edit=0.;self.read_null_fields();self.render_tools()
        except (ValueError,KeyError) as error:self.status.content=str(error)
    def null_file_action(self,event=None):
        try:
            value=self.null_file.value.strip()
            if not value:raise ValueError('Enter a local .m54nulls.json path first.')
            if self.null_files.value=='Save':
                sources={k:v.get('source','') for k,v in self.source_scenes().items()}
                if any(Path(value).resolve()==Path(p).resolve() for p in sources.values() if p):raise ValueError('Nulls must not overwrite a source reconstruction.')
                self.nulls.save(value,sources)
            else:self.nulls.load(value)
            self.read_null_fields();self.render_tools();self.status.content='Nulls '+('saved' if self.null_files.value=='Save' else 'loaded')+' locally: '+value
        except (OSError,ValueError,KeyError) as error:self.status.content=str(error)
    def gizmo_spec(self,scenes):
        if self.gizmo_target.value=='Selected null':
            value=self.nulls.items.get(self.selected_null)
            if value is None:return None
            return ('null',self.selected_null,np.asarray(value['position']),Rotation.from_euler('xyz',value['rotation_xyz_degrees'],degrees=True).as_matrix())
        key=self.layer.value
        if key not in scenes:return None
        source_center=self.source_scenes()[key]['center'];matrix=self.transforms[key].matrix()
        return ('model',key,matrix[:3,:3]@source_center+matrix[:3,3],matrix[:3,:3]/self.transforms[key].scale)
    def render_tools(self):
        if self.updating:return
        with self.lock:
            for node in self.nodes:node.remove()
            self.nodes=[];scenes=self.effective();scene=scenes.get(self.layer.value) or next(iter(scenes.values()),None)
            if scene is None:
                if not self.nulls.items:return
                scene=from_records({})
            segments,step=grid_lines(scene['center'],scene['extent'],self.plane.value,float(self.height.value))
            self.nodes.append(self.server.scene.add_line_segments('/m54/grid',self.display(segments,scene).astype(np.float32),(60,77,84),line_width=1.,visible=self.grid.value))
            for index,(name,item) in enumerate(self.nulls.items.items()):
                q=Rotation.from_euler('xyz',item['rotation_xyz_degrees'],degrees=True).as_quat();p=self.display(item['position'],scene)
                self.nodes.append(self.server.scene.add_frame(f'/m54/null/{index}/axes',position=p,wxyz=q[[3,0,1,2]],axes_length=.018 if self.normalized else scene['extent']*.018,axes_radius=.0008 if self.normalized else scene['extent']*.0008))
                node=self.server.scene.add_icosphere(f'/m54/null/{index}/marker',position=p,radius=.006 if self.normalized else scene['extent']*.006,color=(255,205,115),subdivisions=1)
                node.on_click(lambda e,n=name:self.select_null(n,e.client));self.nodes.append(node)
            spec=self.gizmo_spec(scenes)
            if self.gizmo is None:
                self.gizmo=self.server.scene.add_transform_controls('/m54/gizmo',scale=.15 if self.normalized else scene['extent']*.15,depth_test=False,visible=False)
                self.gizmo.on_update(self.gizmo_changed)
            gizmo=self.gizmo;gizmo.visible=self.gizmo_enabled.value and spec is not None
            gizmo.disable_axes=self.gizmo_mode.value=='Rotate';gizmo.disable_sliders=self.gizmo_mode.value=='Rotate';gizmo.disable_rotations=self.gizmo_mode.value=='Move'
            if spec is not None:
                identity=spec[:2]
                if not self.gizmo_in_callback:
                    q=Rotation.from_matrix(spec[3]).as_quat();gizmo.position=tuple(self.display(spec[2],scene));gizmo.wxyz=tuple(q[[3,0,1,2]])
                if identity!=self.gizmo_target_key:self.last_edit={'A':0.,'B':0.};self.last_null_edit=0.;self.gizmo_target_key=identity
            self.update_selection_info()
    def gizmo_changed(self,handle):
        # Viser 0.2.23 supplies TransformControlsHandle, not a click event.
        if self.updating:return
        with self.lock:
            scenes=self.effective();spec=self.gizmo_spec(scenes)
            if spec is None:return
            scene=scenes.get(self.layer.value) or next(iter(scenes.values()),from_records({}));base=self.display_frame(scene)
            q=np.asarray(handle.wxyz);r=Rotation.from_quat(q[[1,2,3,0]]);p=np.asarray(handle.position)*base['extent']+base['center'] if self.normalized else np.asarray(handle.position)
            angles=rotation_numbers(r.as_matrix())['rotation_xyz_degrees'];kind,key=spec[:2];now=time.monotonic()
            if kind=='null':
                self.nulls.set_pose(key,p,angles,checkpoint=now-self.last_null_edit>.5);self.last_null_edit=now;self.read_null_fields()
            else:
                if now-self.last_edit[key]>.5:self.history[key].append(copy.deepcopy(self.transforms[key]));self.history[key]=self.history[key][-32:];self.redo[key].clear()
                self.last_edit[key]=now;source_center=self.source_scenes()[key]['center'];self.transforms[key]=SceneTransform(p-source_center,np.asarray(angles),self.transforms[key].scale,source_center)
                previous=self.last_edit.copy();self.read_fields();self.last_edit=previous
            self.gizmo_in_callback=True
            try:self.refresh();self.render_tools()
            finally:self.gizmo_in_callback=False
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
        # Set position/up first, then target: the initial client position can
        # arrive after connect, and position messages may recenter the target.
        scene=self.tools.effective().get('A');center=np.zeros(3);extent=1.
        if scene is not None:
            center=self.tools.display(scene['center'],scene);extent=scene['extent']/self.tools.display_frame(scene)['extent']
        if self.tools.nulls.items:
            base=scene if scene is not None else from_records({});p=self.tools.display(np.asarray([v['position'] for v in self.tools.nulls.items.values()]),base)
            p=np.vstack([p,center-extent*.5,center+extent*.5]);center=(p.max(0)+p.min(0))/2;extent=max(float(np.linalg.norm(p.max(0)-p.min(0))),.1)
        with client.atomic():
            client.camera.up_direction=self.tools.up_vector();client.camera.position=center+np.array([1.,-1.,1.])*extent;client.camera.look_at=center
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
