"""Geometric candidate images; predictions never imply visibility or a match."""
import tkinter as tk
from tkinter import ttk
import numpy as np
from control_points_model import point_geometry


def predicted_candidates(point,cameras):
    if not point or not cameras:return [],'Open an alignment and select a measured point.'
    xyz,description,errors=point_geometry(point,cameras)
    if xyz is None:return [],description
    if errors and max(errors.values())>1:return [],'Measured residual exceeds 1°. Correct observations before using predictions.'
    measured={o['shot_id'] for o in point['observations']}
    names=[n for n in cameras if n not in measured]
    if not names:return [],'This point is already measured in every registered image.'
    rotation=np.asarray([cameras[n][0] for n in names]);translation=np.asarray([cameras[n][1] for n in names])
    centers=-np.einsum('nji,nj->ni',rotation,translation)
    rays=np.einsum('nij,j->ni',rotation,xyz)+translation;length=np.linalg.norm(rays,axis=1)
    valid=length>1e-8;unit=rays/np.maximum(length[:,None],1e-8)
    pixels=np.stack([((.5+np.arctan2(unit[:,0],unit[:,2])/(2*np.pi))%1)*[cameras[n][2] for n in names],
                     (.5+np.arcsin(np.clip(unit[:,1],-1,1))/np.pi)*[cameras[n][3] for n in names]],axis=1)
    support=np.array([-cameras[n][0].T@cameras[n][1] for n in measured if n in cameras])
    separation=np.min(np.linalg.norm(centers[:,None]-support[None],axis=2),axis=1)
    result=[{'image':n,'pixel':pixels[i].tolist(),'distance':float(length[i]),'support_distance':float(separation[i])} for i,n in enumerate(names) if valid[i]]
    result.sort(key=lambda p:(p['support_distance'],p['image']))
    return result,description+' · geometric guidance; occlusion/visibility unknown'


class PredictionBrowser:
    def __init__(self,editor):
        self.editor=editor;self.closed=False;self.window=tk.Toplevel(editor.window);self.window.title('Predicted images · manual 360 points');self.window.geometry('860x620');self.window.configure(bg='#1a2024')
        top=tk.Frame(self.window,bg='#1a2024',padx=12,pady=12);top.pack(fill='x')
        self.summary=tk.StringVar(master=self.window);editor.label(top,variable=self.summary,wraplength=800).pack(fill='x')
        editor.label(top,'Predictions are locations to inspect. Check the physical feature yourself before adding a measurement.',muted=True,wraplength=800).pack(fill='x',pady=8)
        self.search=tk.StringVar(master=self.window);self.target=tk.StringVar(master=self.window,value='Left' if editor.focused==0 else 'Right')
        bar=tk.Frame(top,bg='#1a2024');bar.pack(fill='x');editor.entry(bar,self.search,30).pack(side='left')
        ttk.Combobox(bar,textvariable=self.target,values=['Left','Right'],state='readonly',style='CP.TCombobox',width=8).pack(side='left',padx=10)
        editor.button(bar,'Open & focus prediction',self.open_selected,primary=True).pack(side='right')
        self.tree=editor.tree(self.window,['Image','Predicted X / Y','Near supporting view'],[390,160,175]);self.tree.bind('<Double-1>',lambda e:self.open_selected());self.tree.bind('<Return>',lambda e:self.open_selected())
        editor.button(self.window,'Show candidates in image grid',lambda:editor.open_grid(predicted=True)).pack(side='bottom',fill='x',padx=12,pady=12,before=self.tree.master)
        self.search.trace_add('write',lambda *_:self.refresh());self.window.protocol('WM_DELETE_WINDOW',self.close);self.refresh()
    def refresh(self):
        if self.closed:return
        candidates,description=predicted_candidates(self.editor.selected(),self.editor.cameras);self.candidates={p['image']:p for p in candidates}
        self.summary.set((self.editor.active.get() or 'No point')+' · '+str(len(candidates))+' unmeasured candidates\n'+description)
        old=self.tree.selection();self.tree.delete(*self.tree.get_children())
        for p in candidates:
            if self.search.get().casefold() not in p['image'].casefold():continue
            self.tree.insert('','end',iid=p['image'],values=(p['image'],f"{p['pixel'][0]:.1f} / {p['pixel'][1]:.1f}",f"{p['support_distance']:.3g} alignment units"))
        if old and self.tree.exists(old[0]):self.tree.selection_set(old[0])
    def open_selected(self):
        values=self.tree.selection()
        if not values:return
        name=values[0];candidate=self.candidates.get(name)
        if not candidate:return
        index=0 if self.target.get()=='Left' else 1
        self.editor.choose_image(name,index);self.editor.panes[index]['focus_after_load']=(name,candidate['pixel']);self.editor.window.lift()
        self.editor.audit.event('prediction_open',point=self.editor.active.get(),image=name,pixel=candidate['pixel'])
    def close(self):self.closed=True;self.window.destroy()
