"""Shared desktop M54 navigation/display and model transformation controls."""
import copy
import queue
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from abu_m54 import SceneTransform, export_copy, NAME


class NativeTools(tk.Toplevel):
    def __init__(self, viewer):
        super().__init__(viewer.winfo_toplevel()); self.viewer=viewer; self.editor=viewer.editor
        self.title(NAME + ' · Scene tools'); self.geometry('410x660'); self.minsize(360,540)
        from abu_m54_desktop import window_identity
        window_identity(self)
        self.configure(bg='#1a2024'); self.transient(viewer.winfo_toplevel())
        self.history=viewer.transform_history; self.redo=viewer.transform_redo; self.closed=False
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='M54 export'); self.events=queue.Queue()
        self.editor.label(self, 'Abu M54 · Scene tools', padx=12, pady=10).pack(fill='x')
        footer=tk.Frame(self,bg='#1a2024',padx=12,pady=8);footer.pack(side='bottom',fill='x')
        self.loading=False;self.live_job=None;self.end_job=None;self.edit_old=None;self.scrub=None;self.entries={}
        for title,command in [('Undo',lambda:self.step(False)),('Redo',lambda:self.step(True)),('Reset',self.reset)]:
            self.editor.button(footer,title,command).pack(side='left',padx=2)
        self.status=tk.StringVar(self,'Cloud, cameras and triangulated controls move together. Sources remain unchanged.')
        self.editor.label(self,variable=self.status,muted=True,wraplength=380,padx=12,pady=8).pack(side='bottom',fill='x')
        outer,body=self.editor.scroll_page(self);outer.pack(fill='both',expand=True)
        self.editor.button(body,'Objects, exact poses & 3D nulls…',viewer.open_inspector,primary=True).pack(fill='x',pady=8)
        for text,var in [('Ground grid',viewer.ground),('Orbit around new selection',viewer.auto_pivot)]:
            tk.Checkbutton(body,text=text,variable=var,command=viewer.request,bg='#1a2024',fg='#e7edef',selectcolor='#242d32').pack(anchor='w',pady=5)
        self.editor.label(body,'Grid plane / elevation (scene units)',muted=True).pack(fill='x',pady=(8,3))
        row=tk.Frame(body,bg='#1a2024');row.pack(fill='x')
        choice=ttk.Combobox(row,textvariable=viewer.grid_plane,values=['XY','XZ','YZ'],state='readonly',width=6);choice.pack(side='left')
        choice.bind('<<ComboboxSelected>>',lambda e:viewer.request())
        entry=self.editor.entry(row,viewer.grid_height,12);entry.pack(side='left',padx=8);entry.bind('<Return>',lambda e:self.update_grid())
        self.editor.button(row,'Set',self.update_grid).pack(side='left')
        self.editor.button(body,'Orbit around selected item',viewer.focus_selected).pack(fill='x',pady=8)
        row=tk.Frame(body,bg='#1a2024');row.pack(fill='x')
        self.editor.button(row,'Save view',viewer.save_view).pack(side='left');self.editor.button(row,'Restore view',viewer.restore_view).pack(side='left',padx=4)
        row=tk.Frame(body,bg='#1a2024');row.pack(fill='x',pady=5)
        for text,yaw,pitch in [('Front',0,0),('Top',0,89),('Right',90,0)]:self.editor.button(row,text,lambda y=yaw,p=pitch:viewer.preset(y,p)).pack(side='left',padx=2)
        tk.Checkbutton(body,text='Camera trajectory',variable=viewer.trajectory,command=viewer.request,bg='#1a2024',fg='#e7edef',selectcolor='#242d32').pack(anchor='w',pady=5)
        for title,var,choices in [('World up',viewer.world_up,['Y','Z','X']),('Point shading',viewer.shading,['RGB','Depth','Elevation','White']),('Point size · px',viewer.point_size,[1,2,3,4,6]),('Camera size',viewer.camera_size,[.5,1,2,4])]:
            self.editor.label(body,title,muted=True).pack(fill='x');field=ttk.Combobox(body,textvariable=var,values=choices,state='readonly');field.pack(fill='x',pady=3);field.bind('<<ComboboxSelected>>',lambda e:viewer.request())
        self.layer=tk.StringVar(self,'A');self.pivot=tk.StringVar(self,'Scene centre')
        self.editor.label(body,'Transform whole reconstruction',pady=8).pack(fill='x')
        for title,var,choices in [('Layer',self.layer,['A','B']),('Rotation / scale pivot',self.pivot,['Scene centre','Selected item','World origin'])]:
            self.editor.label(body,title,muted=True).pack(fill='x');field=ttk.Combobox(body,textvariable=var,values=choices,state='readonly');field.pack(fill='x',pady=4)
            if title=='Layer':field.bind('<<ComboboxSelected>>',lambda e:self.read())
        self.values={}
        for title,prefix in [('Move · scene units','t'),('Rotate · X / Y / Z degrees','r')]:
            self.editor.label(body,title,muted=True).pack(fill='x',pady=(8,3));row=tk.Frame(body,bg='#1a2024');row.pack(fill='x')
            for i in range(3):
                key=f'{prefix}{i}';self.values[key]=tk.StringVar(self,'0');entry=self.editor.entry(row,self.values[key],8);entry.pack(side='left',fill='x',expand=True,padx=2);self.bind_value(key,entry)
        self.values['s']=tk.StringVar(self,'1');self.editor.label(body,'Uniform scale',muted=True).pack(fill='x',pady=(8,3));entry=self.editor.entry(body,self.values['s']);entry.pack(fill='x');self.bind_value('s',entry)
        self.editor.guide(body,'Live model transforms','Valid numbers update the scene immediately. Type a value or drag its number left/right. Shift gives fine steps; Ctrl gives coarse steps. Scale drags proportionally and stays positive. One drag is one Undo step. Preview moves cloud/cameras/controls together; export saves a separate full-source copy.')
        self.editor.button(body,'Export transformed copy…',self.export,primary=True).pack(fill='x',pady=14)
        self.editor.label(body,'Click controls, cameras or cloud samples to inspect. Double-click or O makes the selected item the orbit pivot. Fit restores a scene-centred view. Display samples are not new control measurements.',muted=True,wraplength=350).pack(fill='x',pady=8)
        self.read();self.poll_id=self.after(60,self.poll);self.protocol('WM_DELETE_WINDOW',self.destroy)
    def read(self):
        self.loading=True
        value=self.viewer.transforms[self.layer.get()]
        for prefix,items in [('t',value.translation),('r',value.angles)]:
            for i,item in enumerate(items):self.values[f'{prefix}{i}'].set(f'{item:.8g}')
        self.values['s'].set(f'{value.scale:.8g}')
        self.loading=False
    def bind_value(self,key,entry):
        self.entries[key]=entry;self.values[key].trace_add('write',self.schedule_live)
        entry.bind('<ButtonPress-1>',lambda e,k=key:self.begin_scrub(e,k),add='+')
        entry.bind('<B1-Motion>',self.drag_value,add='+');entry.bind('<ButtonRelease-1>',self.end_scrub,add='+')
        entry.bind('<Return>',lambda e:self.finish_edit(),add='+');entry.bind('<FocusOut>',lambda e:self.finish_edit(),add='+')
        self.editor.guide(entry,'Live value · drag or type','Drag the number horizontally to change it. Shift: 10× finer; Ctrl: 10× larger. Click without dragging to type. Valid numbers update immediately; an incomplete or invalid number leaves the last valid preview intact. Undo restores the entire drag.')
    def schedule_live(self,*_):
        if self.loading or self.closed:return
        if self.live_job is None:self.live_job=self.after(16,self.live_apply)
        if self.end_job is not None:self.after_cancel(self.end_job)
        self.end_job=self.after(500,self.finish_edit) if not self.scrub or not self.scrub['dragged'] else None
    def live_apply(self):
        self.live_job=None;self.apply()
    def finish_edit(self):
        if self.live_job is not None:self.after_cancel(self.live_job);self.live_job=None;self.apply()
        if self.end_job is not None:self.after_cancel(self.end_job);self.end_job=None
        self.edit_old=None
    def begin_scrub(self,event,key):
        self.finish_edit()
        try:value=float(self.values[key].get())
        except ValueError:return
        scene=self.viewer.raw_data if self.layer.get()=='A' else self.viewer.compare.b
        self.scrub={'key':key,'x':event.x_root,'value':value,'dragged':False,'step':max(float((scene or {}).get('extent',1))*.002,1e-8)}
    def drag_value(self,event):
        if self.scrub is None:return
        dx=event.x_root-self.scrub['x']
        if abs(dx)<4 and not self.scrub['dragged']:return
        self.scrub['dragged']=True;factor=.1 if event.state&1 else 10. if event.state&4 else 1.
        key=self.scrub['key'];start=self.scrub['value']
        value=start*np.exp(np.clip(dx*.005*factor,-30,30)) if key=='s' else start+dx*factor*(.25 if key.startswith('r') else self.scrub['step'])
        self.values[key].set(f'{value:.10g}');return 'break'
    def end_scrub(self,event):
        if self.scrub and self.scrub['dragged']:self.finish_edit();self.scrub=None;return 'break'
        self.scrub=None
    def update_grid(self):
        try:
            if not np.isfinite(self.viewer.grid_height.get()):raise ValueError('Grid elevation must be finite.')
            self.viewer.request()
        except (ValueError,tk.TclError) as error:self.status.set(str(error))
    def apply(self):
        try:
            layer=self.layer.get();scene=self.viewer.data if layer=='A' else self.viewer.comparison['b'] if self.viewer.comparison else None
            if scene is None:raise ValueError('Load this reconstruction layer first.')
            old=self.viewer.transforms[layer]
            # Keep an established pivot when editing numeric values, until the
            # user explicitly chooses another pivot mode or resets the transform.
            source=self.viewer.raw_data if layer=='A' else self.viewer.compare.b
            if self.pivot.get()=='Selected item':
                pivot=self.viewer.selected_world()
                if pivot is None:raise ValueError('Select a point, camera or cloud sample first.')
                if layer=='B' and self.viewer.comparison:
                    fit=self.viewer.comparison['report']['transform'];matrix=np.eye(4);matrix[:3,:3]=fit['scale']*np.asarray(fit['rotation']);matrix[:3,3]=fit['translation']
                    pivot=(np.linalg.inv(matrix)@np.r_[pivot,1])[:3]
                pivot=(np.linalg.inv(old.matrix())@np.r_[pivot,1])[:3]
            else:pivot=source['center'] if self.pivot.get()=='Scene centre' else np.zeros(3)
            value=SceneTransform(np.array([float(self.values[f't{i}'].get()) for i in range(3)]),np.array([float(self.values[f'r{i}'].get()) for i in range(3)]),float(self.values['s'].get()),pivot)
            value.matrix()
            if np.allclose(value.matrix(),old.matrix(),rtol=0,atol=1e-12):return
            if self.edit_old is None:
                self.edit_old=copy.deepcopy(old);self.history[layer].append(copy.deepcopy(old));self.history[layer][:]=self.history[layer][-32:];self.redo[layer].clear()
            self.viewer.apply_transform(layer,value);self.status.set(f'{layer} transformed in preview. Export a copy to retain this change.')
        except (ValueError,tk.TclError,np.linalg.LinAlgError) as error:self.status.set(str(error))
    def step(self,redo):
        self.finish_edit()
        layer=self.layer.get();source=self.redo[layer] if redo else self.history[layer];destination=self.history[layer] if redo else self.redo[layer]
        if source:
            destination.append(copy.deepcopy(self.viewer.transforms[layer]));self.viewer.apply_transform(layer,source.pop());self.read()
    def reset(self):
        self.finish_edit()
        layer=self.layer.get();self.history[layer].append(copy.deepcopy(self.viewer.transforms[layer]));self.history[layer][:]=self.history[layer][-32:];self.redo[layer].clear();self.viewer.apply_transform(layer,SceneTransform());self.read()
    def export(self):
        self.finish_edit()
        layer=self.layer.get();scene=self.viewer.raw_data if layer=='A' else self.viewer.compare.b
        if not scene:self.status.set('Load this reconstruction layer first.');return
        source=scene['source'];suffix='.json' if str(source).lower().endswith('.json') else '.ply'
        value=filedialog.asksaveasfilename(parent=self,title='Export full transformed reconstruction copy',defaultextension=suffix,initialfile='m54_transformed'+suffix)
        if not value:return
        transform=copy.deepcopy(self.viewer.transforms[layer]);self.status.set('Exporting full source in background…')
        events=self.events
        def work():
            try:events.put(('done',export_copy(source,value,transform)))
            except Exception as error:events.put(('error',str(error)))
        self.executor.submit(work)
    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,value=self.events.get_nowait();self.status.set('Exported transformed copy.' if kind=='done' else 'Export failed: '+value)
                self.editor.audit.event('m54_export_'+kind,result=value)
        except queue.Empty:pass
        self.poll_id=self.after(60,self.poll)
    def destroy(self):
        if self.closed:return
        self.finish_edit();self.closed=True;self.after_cancel(self.poll_id);self.executor.shutdown(wait=False,cancel_futures=True);super().destroy()
