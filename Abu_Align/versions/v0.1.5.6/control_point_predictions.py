"""Geometric candidate images; predictions never imply visibility or a match."""
import copy,json,queue,subprocess,sys,threading,tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from tkinter import ttk
import numpy as np
from control_points_model import point_geometry, projected_pixel


def predicted_candidates(point,cameras,policy=None):
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
    pixels=[projected_pixel(xyz,cameras[n]) for n in names]
    support=np.array([-cameras[n][0].T@cameras[n][1] for n in measured if n in cameras])
    separation=np.min(np.linalg.norm(centers[:,None]-support[None],axis=2),axis=1)
    from manual_image_policy import Masks
    masks=Masks(policy)
    result=[{'image':n,'pixel':pixels[i],'distance':float(length[i]),'support_distance':float(separation[i])} for i,n in enumerate(names) if valid[i] and pixels[i] is not None and masks.allowed(n,pixels[i],cameras[n][2:])]
    result.sort(key=lambda p:(p['support_distance'],p['image']))
    return result,description+' · geometric guidance; occlusion/visibility unknown'


class PredictionBrowser:
    def __init__(self,editor):
        self.editor=editor;self.closed=False;self.window=tk.Toplevel(editor.window);self.window.title('Predicted images · Manual control points');self.window.geometry('860x620');self.window.configure(bg='#1a2024')
        top=tk.Frame(self.window,bg='#1a2024',padx=12,pady=12);top.pack(fill='x')
        self.summary=tk.StringVar(master=self.window);editor.label(top,variable=self.summary,wraplength=800).pack(fill='x')
        editor.label(top,'Predictions are locations to inspect. Check the physical feature yourself before adding a measurement.',muted=True,wraplength=800).pack(fill='x',pady=8)
        self.search=tk.StringVar(master=self.window);self.target=tk.StringVar(master=self.window,value='Left' if editor.focused==0 else 'Right')
        self.visual={};self.snapshot=None;self.events=queue.Queue();self.state=None;self.job=None
        controls=tk.Frame(top,bg='#1a2024');controls.pack(fill='x');self.method=tk.StringVar(self.window,'auto');self.hardware=tk.StringVar(self.window,'auto');self.scope=tk.StringVar(self.window,'all')
        for title,variable,values,tip in [('Matching',self.method,['auto','learned','sift'],'Auto uses installed local AI, otherwise computer vision. Learned requires ALIKED/LightGlue weights. SIFT is classical computer vision; no model download is needed.'),('Hardware',self.hardware,['auto','cuda','cpu'],'Auto uses the local GPU when available for learned matching. CPU is slower for learned inference. SIFT uses CPU regardless of this choice.'),('Scope',self.scope,['all','registered'],'All examines every enabled original, including images without poses. Registered restricts to known cameras. No nearest-camera count limit is used; global search takes longer.')]:
            group=tk.Frame(controls,bg='#1a2024');group.pack(side='left',padx=3);editor.label(group,title,muted=True).pack(fill='x');choice=ttk.Combobox(group,textvariable=variable,values=values,state='readonly',width=10,style='CP.TCombobox');choice.pack();editor.guide(choice,title,tip)
        editor.button(controls,'Visual search',self.scan,'Search every enabled original locally. All includes unregistered images; registered uses camera geometry. AI weights are local; geometric and visual proposals must be verified.').pack(side='left',padx=4)
        editor.button(controls,'Cancel',self.cancel).pack(side='left');editor.button(top,'Set up local AI model',lambda:self.scan(setup=True),'Download only public ALIKED/LightGlue weights. No user images leave the PC. Setup is separate from offline image inference.').pack(anchor='w',pady=4)
        bar=tk.Frame(top,bg='#1a2024');bar.pack(fill='x');editor.entry(bar,self.search,30).pack(side='left')
        ttk.Combobox(bar,textvariable=self.target,values=['Left','Right'],state='readonly',style='CP.TCombobox',width=8).pack(side='left',padx=10)
        editor.button(bar,'Open & focus prediction',self.open_selected,primary=True).pack(side='right')
        self.tree=editor.tree(self.window,['Image','Predicted X / Y','Evidence'],[290,155,340]);self.tree.bind('<Double-1>',lambda e:self.open_selected());self.tree.bind('<Return>',lambda e:self.open_selected())
        editor.button(self.window,'Show candidates in image grid',lambda:editor.open_grid(predicted=True)).pack(side='bottom',fill='x',padx=12,pady=12,before=self.tree.master)
        self.search.trace_add('write',lambda *_:self.refresh());self.window.protocol('WM_DELETE_WINDOW',self.close);self.refresh()
    def refresh(self):
        if self.closed:return
        signature=self.signature()
        if self.snapshot is not None and self.snapshot!=signature:self.cancel();self.visual.clear();self.snapshot=None
        candidates,description=predicted_candidates(self.editor.effective_point(registered_only=True),self.editor.cameras,self.editor.controls.get('image_policy'));self.candidates={p['image']:p for p in candidates};self.candidates.update(self.visual)
        candidates=sorted(self.candidates.values(),key=lambda p:(-p.get('score',-1),p.get('support_distance',0),p['image']))
        self.summary.set((self.editor.active.get() or 'No point')+' · '+str(len(candidates))+' unmeasured candidates\n'+description)
        old=self.tree.selection();self.tree.delete(*self.tree.get_children())
        for p in candidates:
            if self.search.get().casefold() not in p['image'].casefold():continue
            evidence=f"{p['status']} · score {p['score']:.2f} · {p['inliers']} inliers" if 'score' in p else f"Geometry only · {p['support_distance']:.3g} units from support"
            self.tree.insert('','end',iid=p['image'],values=(p['image'],f"{p['pixel'][0]:.1f} / {p['pixel'][1]:.1f}",evidence))
        if old and self.tree.exists(old[0]):self.tree.selection_set(old[0])
    def open_selected(self):
        values=self.tree.selection()
        if not values:return
        name=values[0];candidate=self.candidates.get(name)
        if not candidate:return
        index=0 if self.target.get()=='Left' else 1
        if name not in self.editor.image_sizes and candidate.get('size'):
            path=self.editor.image_dir/name
            if not path.is_file() or path.name!=name:return
            self.editor.image_sizes[name]=candidate['size'];self.editor.image_names.append(name)
            for pane in self.editor.panes:pane['choice']['values']=self.editor.image_names
        self.editor.choose_image(name,index);self.editor.panes[index]['focus_after_load']=(name,candidate['pixel']);self.editor.window.lift()
        self.editor.audit.event('prediction_open',point=self.editor.active.get(),image=name,pixel=candidate['pixel'])
    def signature(self):return json.dumps([self.editor.selected(),str(self.editor.reconstruction),self.editor.controls.get('image_policy')],sort_keys=True)
    def scan(self,setup=False):
        if self.state is not None and not self.state.finished:return
        point=self.editor.effective_point()
        if not setup and (not point or not point['observations']):self.summary.set('Add a usable measured point first.');return
        self.snapshot=self.signature();self.visual.clear();self.refresh();self.snapshot=self.signature()
        command=[sys.executable,'-B',str(Path(__file__).with_name('manual_prediction_worker.py'))]
        if setup:command+=['--setup']
        else:
            request_path=self.editor.audit.path.with_suffix('.prediction.request.json')
            component=self.editor.component;small=dict(cameras=component['cameras'],shots=component['shots'],points={}) if component else None
            request=dict(point=copy.deepcopy(point),component=small,image_dir=str(self.editor.image_dir),image_sizes=self.editor.image_sizes,image_policy=self.editor.controls.get('image_policy'),method=self.method.get(),device=self.hardware.get(),scope=self.scope.get(),projection='spherical' if self.editor.settings['image_projection'].get()=='360 equirectangular' else 'perspective')
            request_path.write_text(json.dumps(request),encoding='utf-8');command+=['--request',str(request_path)]
        events=self.events;state=SimpleNamespace(process=None,cancelled=False,finished=False);self.state=state;signature=self.snapshot
        def work():
            try:
                state.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if state.cancelled:state.process.terminate()
                for line in state.process.stdout:
                    if line.startswith('ABU_PREDICTION '):events.put((signature,json.loads(line[15:])))
                state.process.wait();events.put((signature,dict(event='stopped')))
            except Exception as error:events.put((signature,dict(event='error',error=str(error))))
            finally:state.finished=True
        threading.Thread(target=work,daemon=True,name='Abu local prediction monitor').start();self.summary.set('Starting local visual search…');self.poll()
    def poll(self):
        self.job=None
        if self.closed:return
        while True:
            try:signature,event=self.events.get_nowait()
            except queue.Empty:break
            if signature!=self.signature():continue
            kind=event['event'];self.editor.audit.event('prediction_'+kind,**{k:v for k,v in event.items() if k!='event'})
            if kind=='candidate':
                self.visual[event['image']]=dict(event)
                name=event['image']
                if name not in self.editor.image_sizes and (self.editor.image_dir/name).is_file():
                    self.editor.image_sizes[name]=event['size'];self.editor.image_names.append(name)
                    for pane in self.editor.panes:pane['choice']['values']=self.editor.image_names
                self.refresh();self.editor.draw()
                if self.editor.grid and not self.editor.grid.closed:self.editor.grid.refresh()
            elif kind=='progress':self.summary.set(f"Searched {event['completed']}/{event['total']} · {len(self.visual)} visual proposals · verify each feature.")
            elif kind=='start':self.summary.set(f"{event['method']} · {event['device']} · searching {event['images']} images. "+event.get('fallback',''))
            elif kind=='complete':self.summary.set(f"Searched {event['processed']}/{event['total']} · {event['candidates']} proposals · {event['seconds']:.1f}s · {event['method']} {event['device']}.")
            elif kind=='setup_complete':self.summary.set('Local AI models ready. Visual search can now run offline.')
            elif kind=='error':self.summary.set(event['error'])
        if self.state and self.state.finished:return
        self.job=self.window.after(60,self.poll)
    def cancel(self):
        if self.state:
            self.state.cancelled=True
            if self.state.process and self.state.process.poll() is None:self.state.process.terminate()
    def close(self):
        self.closed=True;self.cancel()
        if self.job is not None:self.window.after_cancel(self.job)
        self.window.destroy()
