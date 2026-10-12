"""Loopback-only WebGL inspection of sparse geometry and triangulated controls."""
import hashlib
import threading
import numpy as np
from control_points_model import point_geometry


def scene_data(component, controls, cameras, limit=50000):
    items=list(component.get('points',{}).values())
    if len(items)>limit:items=[items[i] for i in np.linspace(0,len(items)-1,limit,dtype=int)]
    points=np.asarray([p['coordinates'] for p in items],dtype=np.float32).reshape(-1,3)
    colors=np.asarray([p.get('color',[160,175,180]) for p in items],dtype=np.uint8).reshape(-1,3)
    finite=np.isfinite(points).all(axis=1);points,colors=points[finite],colors[finite]
    centers={name:-rotation.T@translation for name,(rotation,translation,_,_) in cameras.items()}
    landmarks={};untriangulated=[]
    for point in controls['points']:
        xyz,description,errors=point_geometry(point,cameras)
        if xyz is None:untriangulated.append(point['id']);continue
        landmarks[point['id']]={'position':xyz,'kind':point.get('type','tie'),'enabled':point.get('enabled',True),
                                'description':description,'shots':[o['shot_id'] for o in point['observations']]}
    positions=np.asarray(list(centers.values()),dtype=float)
    center=np.median(positions,axis=0) if len(positions) else np.zeros(3)
    extent=max(1.,float(np.linalg.norm(np.ptp(positions,axis=0)))) if len(positions) else 1.
    return {'points':points,'colors':colors,'centers':centers,'controls':landmarks,'untriangulated':untriangulated,
            'center':center,'extent':extent,'total_points':len(component.get('points',{}))}


class ControlSceneViewer:
    def __init__(self, event_queue):
        import viser
        self.server=viser.ViserServer(host='127.0.0.1',port=9010,label='Abu · 360 control inspection',verbose=False)
        self.server.gui.configure_theme(dark_mode=True,show_share_button=False)
        self.events=event_queue;self.lock=threading.RLock();self.closed=False;self.data=None;self.nodes=[]
        self.label_nodes=[]
        @self.server.on_client_connect
        def connected(client):
            if self.data:
                client.camera.position=self.data['center']+self.data['extent']*np.array([1.,-1.,-1.])
                client.camera.look_at=self.data['center']
        self.server.gui.add_markdown('**Local 3D inspection**\n\nPoint cloud and camera centres share the alignment frame. Control markers show **triangulated positions**, not unregistered survey coordinates. Click a marker to select its point; click a camera to open its image in the editor.')
        self.selection=self.server.gui.add_dropdown('Control point',options=['—'],initial_value='—')
        self.cloud=self.server.gui.add_checkbox('Show sparse point cloud',initial_value=True)
        self.labels=self.server.gui.add_checkbox('Show point names',initial_value=True)
        self.rays=self.server.gui.add_checkbox('Selected point sight lines',initial_value=True)
        self.info=self.server.gui.add_markdown('Waiting for alignment…')
        @self.selection.on_update
        def select(event):
            if getattr(event,'client',None) is not None and self.selection.value!='—':self.events.put(('viewer_select',self.selection.value))
            self.update_selection(self.selection.value)
        @self.cloud.on_update
        def cloud(event):
            if hasattr(self,'cloud_node'):self.cloud_node.visible=self.cloud.value
        @self.labels.on_update
        def labels(event):
            for node in self.label_nodes:node.visible=self.labels.value
        @self.rays.on_update
        def rays(event):self.update_selection(self.selection.value)
    @property
    def url(self):return f'http://127.0.0.1:{self.server.get_port()}'
    def update(self, component, controls, cameras, selected=''):
        data=scene_data(component,controls,cameras)
        with self.lock:
            if self.closed:return
            self.data=data
            for node in self.nodes:node.remove()
            self.nodes=[];self.label_nodes=[]
            scale=data['extent']
            self.cloud_node=self.server.scene.add_point_cloud('/alignment/cloud',points=data['points'],colors=data['colors'],point_size=scale*.001,point_shape='circle',visible=self.cloud.value);self.nodes.append(self.cloud_node)
            for name,position in data['centers'].items():
                token=hashlib.sha1(name.encode()).hexdigest()[:12]
                node=self.server.scene.add_icosphere('/alignment/cameras/'+token,radius=scale*.004,color=(140,165,180),position=tuple(position),subdivisions=1);self.nodes.append(node)
                def click(event,image=name):self.events.put(('viewer_camera',image))
                node.on_click(click)
            palette={'tie':(112,219,192),'gcp':(120,180,255),'check':(255,125,125)}
            for name,landmark in data['controls'].items():
                token=hashlib.sha1(name.encode()).hexdigest()[:12];color=palette[landmark['kind']] if landmark['enabled'] else (130,140,145)
                node=self.server.scene.add_icosphere('/alignment/controls/'+token,radius=scale*.008,color=color,position=tuple(landmark['position']),subdivisions=2);self.nodes.append(node)
                def click(event,point=name):self.events.put(('viewer_select',point))
                node.on_click(click)
                label=self.server.scene.add_label('/alignment/labels/'+token,text=name+' · '+landmark['kind'],position=tuple(landmark['position']),visible=self.labels.value)
                self.nodes.append(label);self.label_nodes.append(label)
            options=['—']+sorted(data['controls']);self.selection.options=options
            self.selection.value=selected if selected in options else '—'
            self.info.content=f"**{len(data['points']):,} / {data['total_points']:,} cloud points displayed**\n\n{len(data['centers'])} camera centres · {len(data['controls'])} triangulated controls\n\nNot triangulated: {', '.join(data['untriangulated']) or 'none'}\n\nTie: mint · Ground control: blue · Check: red · Disabled: grey.\n\nCoordinates are the alignment's existing units. Survey targets are not overlaid until their frame is established."
            self.update_selection(self.selection.value)
    def update_selection(self,name):
        with self.lock:
            if self.closed or not self.data:return
            if hasattr(self,'ray_node'):self.ray_node.remove();del self.ray_node
            point=self.data['controls'].get(name)
            if not point:return
            if self.rays.value:
                segments=np.asarray([[point['position'],self.data['centers'][shot]] for shot in point['shots'] if shot in self.data['centers']],dtype=np.float32).reshape(-1,2,3)
                self.ray_node=self.server.scene.add_line_segments('/alignment/selected_rays',points=segments,colors=(255,205,115),line_width=2.)
            for client in self.server.get_clients().values():client.camera.look_at=point['position']
    def set_selected(self,name):
        with self.lock:
            if self.closed:return
            if name in self.selection.options:self.selection.value=name
            self.update_selection(name)
    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
        self.server.stop()
