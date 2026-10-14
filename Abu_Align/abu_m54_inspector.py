"""Modeless M54 object browser, exact pose readout and independent null editor."""
import json
import tkinter as tk
from tkinter import ttk,filedialog
from pathlib import Path
import numpy as np
from abu_m54_editing import selection_record


class ObjectInspector(tk.Toplevel):
    def __init__(self,viewer):
        super().__init__(viewer.winfo_toplevel());self.viewer=viewer;self.editor=viewer.editor;self.closed=False;self.syncing=False;self.edit_job=None;self.end_job=None;self.editing=False;self.topology=None
        self.title('Abu M54 · Objects, poses & nulls');self.geometry('500x740');self.minsize(380,540);self.configure(bg='#1a2024');self.transient(viewer.winfo_toplevel())
        from abu_m54_desktop import window_identity
        window_identity(self)
        top=tk.Frame(self,bg='#1a2024',padx=10,pady=7);top.pack(fill='x')
        for text,command in [('Add null here',self.add_null),('Save nulls…',self.save),('Load nulls…',self.load)]:self.editor.button(top,text,command).pack(side='left',padx=2)
        self.status=tk.StringVar(self,'Select a camera, control, cloud sample or null. World positions use source scene units.')
        self.editor.label(self,variable=self.status,muted=True,wraplength=470,padx=10,pady=7).pack(side='bottom',fill='x')
        split=tk.PanedWindow(self,orient='vertical',bg='#344149',sashwidth=7,opaqueresize=True);split.pack(fill='both',expand=True,padx=8)
        list_frame=tk.Frame(split,bg='#1a2024');split.add(list_frame,minsize=75,height=170,stretch='always')
        style=ttk.Style(self);style.configure('M54Objects.Treeview',background='#242d32',fieldbackground='#242d32',foreground='#e7edef',font=('Segoe UI',10),rowheight=max(26,int(26*self.winfo_fpixels('1i')/96)))
        self.tree=ttk.Treeview(list_frame,show='tree',style='M54Objects.Treeview',selectmode='browse');self.tree.pack(side='left',fill='both',expand=True)
        scrollbar=ttk.Scrollbar(list_frame,orient='vertical',command=self.tree.yview);scrollbar.pack(side='right',fill='y');self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind('<<TreeviewSelect>>',self.choose)
        outer,body=self.editor.scroll_page(split);split.add(outer,minsize=170,stretch='always')
        self.summary=tk.StringVar(self,'Nothing selected');self.editor.label(body,variable=self.summary,pady=8).pack(fill='x')
        self.values={};self.fields={}
        for text,prefix in [('World position · X / Y / Z','p'),('Local-to-world rotation · XYZ degrees','r')]:
            self.editor.label(body,text,muted=True).pack(fill='x',pady=(6,3));row=tk.Frame(body,bg='#1a2024');row.pack(fill='x')
            for i in range(3):
                key=f'{prefix}{i}';self.values[key]=tk.StringVar(self,'—');entry=self.editor.entry(row,self.values[key],12);entry.pack(side='left',fill='x',expand=True,padx=2);self.fields[key]=entry
                self.values[key].trace_add('write',self.schedule);entry.bind('<Return>',lambda e:self.finish());entry.bind('<FocusOut>',lambda e:self.finish())
        self.detail=tk.StringVar(self);self.editor.label(body,variable=self.detail,muted=True,wraplength=455,pady=8).pack(fill='x')
        self.editor.guide(body,'Actual object data','Camera rotation is camera-to-world XYZ Euler degrees and a scalar-first quaternion. An ordinary point/control has no stored rotation; its fields show not defined. Model-frame rotation is separate. Create an oriented null from a selection to give that marker its own editable orientation. Nulls are independent world objects, not bundle-adjustment controls.')
        row=tk.Frame(body,bg='#1a2024');row.pack(fill='x',pady=5)
        self.editor.button(row,'Undo null',lambda:self.history(False)).pack(side='left');self.editor.button(row,'Redo null',lambda:self.history(True)).pack(side='left',padx=4);self.delete_button=self.editor.button(row,'Delete null',self.delete);self.delete_button.pack(side='right')
        edit=viewer.editing
        tk.Checkbutton(body,text='Mouse gizmo',variable=edit.enabled,command=viewer.request,bg='#1a2024',fg='#e7edef',selectcolor='#242d32').pack(anchor='w',pady=8)
        for title,var,choices in [('Gizmo target',edit.target,['Selected null','Layer A','Layer B']),('Operation',edit.mode,['Move','Rotate']),('Axis frame',edit.axes,['World','Local'])]:
            self.editor.label(body,title,muted=True).pack(fill='x');field=ttk.Combobox(body,textvariable=var,values=choices,state='readonly');field.pack(fill='x',pady=3);field.bind('<<ComboboxSelected>>',lambda e:viewer.request())
        self.editor.label(body,'Drag RGB arrow shafts to move; switch to Rotate and drag a coloured ring. Esc cancels. Layer A/B moves cloud, cameras and controls together. Null position/rotation fields update live; source objects remain read-only.',muted=True,wraplength=450,pady=8).pack(fill='x')
        self.editor.button(body,'Orbit around selected object',viewer.focus_selected).pack(fill='x',pady=4)
        self.protocol('WM_DELETE_WINDOW',self.destroy);self.refresh();self.poll_job=self.after(150,self.poll)
    def topology_key(self):
        v=self.viewer;result=[]
        for layer,scene in [('A',v.data),('B',v.comparison['b'] if v.comparison else None)]:
            if scene:
                result.extend(('camera',layer,n) for n in scene['centers']);result.extend(('point',layer,n) for n in scene.get('controls',{}) if layer=='A')
        result.extend(('null','World',n) for n in v.nulls.items);return tuple(result)
    def refresh(self):
        if self.closed:return
        v=self.viewer;record=selection_record(v);key=self.topology_key();self.syncing=True
        if key!=self.topology:
            self.topology=key;self.tree.delete(*self.tree.get_children());self.rows={}
            for index,item in enumerate(key):
                iid=str(index);self.rows[iid]=item;self.tree.insert('','end',iid=iid,text=f'{item[1]} · {item[0].capitalize()} · {item[2]}')
        if record:
            match=(record['kind'],record['layer'],record['name']);iid=next((i for i,row in self.rows.items() if row==match),None)
            if iid is not None and self.tree.selection()!=(iid,):self.tree.selection_set(iid);self.tree.see(iid)
            self.summary.set(f"{record['kind'].capitalize()} · {record['name']} · {record['layer']}")
            is_null=record['kind']=='null';focused=self.focus_get() in self.fields.values()
            if not (is_null and focused):
                for prefix,values in [('p',record['position']),('r',record['rotation_xyz_degrees'])]:
                    for i in range(3):self.values[f'{prefix}{i}'].set(f'{values[i]:.14g}' if values is not None else '—')
            for entry in self.fields.values():entry.configure(state='normal' if is_null else 'readonly')
            details=[]
            if record['quaternion_wxyz'] is None:details.append('Rotation: not defined for an ordinary point. No orientation is invented.')
            else:details.append('Quaternion · w, x, y, z\n'+', '.join(f'{x:.12g}' for x in record['quaternion_wxyz']))
            for title,field in [('World-to-camera rotvec · radians','world_to_camera_rotvec_radians'),('World-to-camera translation','world_to_camera_translation'),('Model frame rotation · XYZ degrees','model_frame_rotation_xyz_degrees')]:
                if field in record:details.append(title+'\n'+', '.join(f'{x:.12g}' for x in record[field]))
            self.detail.set('\n\n'.join(details));self.delete_button.configure(state='normal' if is_null else 'disabled')
        else:
            self.summary.set('Nothing selected');self.detail.set('Click a camera, control, cloud sample or null in the viewer.')
            for entry in self.fields.values():entry.configure(state='readonly')
        self.syncing=False
    def choose(self,event=None):
        if self.syncing or not self.tree.selection():return
        self.finish();kind,layer,name=self.rows[self.tree.selection()[0]]
        if kind=='null':self.viewer.select_null(name)
        elif kind=='camera':self.viewer.select_camera(name,layer)
        else:self.viewer.select_control(name)
        self.refresh()
    def add_null(self):
        self.finish();record=selection_record(self.viewer)
        position=record['position'] if record else (self.viewer.center*self.viewer.scale+self.viewer.origin)
        angles=record['rotation_xyz_degrees'] if record and record['rotation_xyz_degrees'] is not None else np.zeros(3)
        name=self.viewer.nulls.add(position,angles);self.viewer.select_null(name);self.viewer.editing.target.set('Selected null');self.viewer.editing.enabled.set(True)
        self.viewer.editor.audit.event('m54_null_add',name=name);self.viewer.request();self.status.set('Added independent world null. Edit its numbers or drag its gizmo; save nulls to retain it.');self.refresh();return name
    def schedule(self,*_):
        if self.syncing or self.closed:return
        if self.edit_job is None:self.edit_job=self.after(16,self.apply_fields)
        if self.end_job is not None:self.after_cancel(self.end_job)
        self.end_job=self.after(500,self.finish)
    def apply_fields(self):
        self.edit_job=None;v=self.viewer
        if getattr(v,'selection_kind','')!='null' or v.null_selected not in v.nulls.items:return
        try:
            p=[float(self.values[f'p{i}'].get()) for i in range(3)];r=[float(self.values[f'r{i}'].get()) for i in range(3)]
            value=v.nulls.valid_pose(p,r)
            if value!=v.nulls.items[v.null_selected]:
                if not self.editing:v.nulls.checkpoint();self.editing=True
                v.nulls.set_pose(v.null_selected,p,r,checkpoint=False);v.request();self.status.set('Null updated in world coordinates. Save nulls to retain it.')
        except (ValueError,tk.TclError) as error:self.status.set('Last valid null retained: '+str(error))
    def finish(self):
        if self.edit_job is not None:self.after_cancel(self.edit_job);self.edit_job=None;self.apply_fields()
        if self.end_job is not None:self.after_cancel(self.end_job);self.end_job=None
        if self.editing:self.viewer.editor.audit.event('m54_null_parameters',name=self.viewer.null_selected)
        self.editing=False
    def history(self,redo):
        self.finish();self.viewer.editing.cancel();self.viewer.nulls.step(redo);self.viewer.request();self.refresh()
    def delete(self):
        self.finish();self.viewer.editing.cancel();self.viewer.editor.audit.event('m54_null_delete',name=self.viewer.null_selected);self.viewer.nulls.delete(self.viewer.null_selected);self.viewer.null_selected='';self.viewer.request();self.refresh()
    def save(self):
        self.finish();v=self.viewer;value=filedialog.asksaveasfilename(parent=self,title='Save independent M54 null objects',initialfile='workspace.m54nulls.json',defaultextension='.json')
        if not value:return False
        try:
            sources={layer:scene['source'] for layer,scene in [('A',v.raw_data),('B',v.compare.b)] if scene}
            if any(Path(value).resolve()==Path(source).resolve() for source in sources.values()):raise ValueError('Nulls must not overwrite a source reconstruction.')
            v.nulls.save(value,sources);self.editor.audit.event('m54_null_save',path=value,count=len(v.nulls.items));self.status.set('Saved nulls: '+value);return True
        except Exception as error:self.editor.error('Cannot save nulls',error);return False
    def load(self):
        value=filedialog.askopenfilename(parent=self,title='Open M54 null objects',filetypes=[('M54 nulls JSON','*.json')])
        if not value:return
        try:
            self.finish();self.viewer.editing.cancel();self.viewer.nulls.load(value)
            self.editor.audit.event('m54_null_load',path=value,count=len(self.viewer.nulls.items))
            if self.viewer.nulls.items:self.viewer.select_null(next(iter(self.viewer.nulls.items)))
            self.viewer.request();self.refresh();self.status.set('Loaded nulls in their stored world frame. Use Undo null to restore the previous list.')
        except Exception as error:self.editor.error('Cannot load nulls',error)
    def poll(self):
        if self.closed:return
        self.refresh();self.poll_job=self.after(150,self.poll)
    def destroy(self):
        if self.closed:return
        self.finish();self.closed=True;self.after_cancel(self.poll_job);super().destroy()
