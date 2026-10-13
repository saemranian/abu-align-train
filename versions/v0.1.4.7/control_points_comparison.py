"""Read-only bounded reconstruction comparison for the control-point navigator."""
import argparse
import json
import queue
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk,filedialog
import numpy as np
from scipy.spatial.transform import Rotation
from viewer_workspace import camera_pairs,similarity,cloud_distances

MINT=(112,219,192)
CORAL=(255,155,127)


def camera_records(scene):
    return [{'name':name,'pos':np.asarray(center)} for name,center in scene['centers'].items()]


def load_scene(source,limit=50000):
    """Executed in an isolated process; return display geometry, not source JSON."""
    source=Path(source);points=np.empty((0,3));colors=np.empty((0,3),np.uint8);records=[];component=None
    if source.is_file() and source.suffix.lower()=='.json':
        data=json.loads(source.read_text(encoding='utf-8'))
        if isinstance(data,list):
            if len(data)!=1:raise ValueError('Choose a single reconstruction component; disconnected components cannot share one comparison frame.')
            data=data[0]
        if not isinstance(data,dict):raise ValueError('Choose a reconstruction JSON, not a report or control file.')
        if 'shots' in data and 'cameras' in data:
            component=data
            entries=list(data.get('points',{}).values());total=len(entries)
            indices=np.linspace(0,max(total-1,0),min(total,limit),dtype=int)
            points=np.asarray([entries[i]['coordinates'] for i in indices],dtype=np.float64).reshape(-1,3)
            colors=np.asarray([entries[i].get('color',[180,180,180]) for i in indices]).reshape(-1,3)
            for name,shot in data['shots'].items():
                matrix=Rotation.from_rotvec(np.asarray(shot['rotation'],dtype=float)).as_matrix();translation=np.asarray(shot['translation'],dtype=float)
                records.append({'name':name,'pos':-matrix.T@translation,'matrix':matrix.T})
        elif 'frames' in data:
            total=0
            for frame in data['frames']:
                matrix=np.asarray(frame['transform_matrix'],dtype=float)
                if matrix.shape!=(4,4):raise ValueError('Invalid camera transform.')
                records.append({'name':frame['file_path'],'pos':matrix[:3,3],'matrix':matrix[:3,:3]@np.diag([1,-1,-1])})
            companions=[source.with_suffix('.points.ply'),source.parent/'points.ply',source.parent/'points3D.ply']
            cloud=next((p for p in companions if p.is_file()),None)
            if cloud:
                from fast_viewer import read_ply_data
                points,colors=read_ply_data(str(cloud));total=len(points) if points is not None else 0
        else:raise ValueError('This JSON has no camera reconstruction. Select reconstruction.json or transforms.json.')
    else:
        from fast_viewer import read_ply_data,find_run_assets,load_run_cameras
        if source.is_dir():
            # Use the main workspace's run discovery for COLMAP/engine packages.
            assets=find_run_assets(str(source));cloud=assets[0]
            if assets[5] and not any(assets[i] for i in [2,3,4]):
                scene=load_scene(assets[5],limit);scene['source']=str(source);return scene
            records,_=load_run_cameras(str(source))
        elif source.suffix.lower()=='.ply':cloud=str(source)
        else:raise ValueError('Choose a reconstruction JSON, PLY, or completed run folder.')
        if cloud:points,colors=read_ply_data(str(cloud))
        total=len(points) if points is not None else 0
        for record in records:
            q=np.asarray(record['wxyz']);record['matrix']=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    if points is None:points=np.empty((0,3))
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    if colors is None or len(colors)!=len(points):colors=np.full((len(points),3),180,np.uint8)
    else:colors=np.clip(colors,0,255).astype(np.uint8)
    mask=np.isfinite(points).all(axis=1);points,colors=points[mask],colors[mask]
    if len(points)>limit:
        indices=np.linspace(0,len(points)-1,limit,dtype=int);points,colors=points[indices],colors[indices]
    valid=[r for r in records if np.asarray(r['pos']).shape==(3,) and np.isfinite(r['pos']).all() and np.isfinite(r['matrix']).all()]
    names=[str(r['name']).replace('\\','/').removeprefix('./') for r in valid]
    if len(set(names))!=len(names):raise ValueError('Comparison has duplicate camera names; choose one component.')
    if not len(points) and not valid:raise ValueError('No finite geometry or cameras found in this reconstruction.')
    return {'source':str(source),'points':points,'colors':colors,'centers':dict(zip(names,[r['pos'] for r in valid])),
            'camera_rotations':dict(zip(names,[r['matrix'] for r in valid])),'controls':{},'total_points':total}


def save_scene(path,scene):
    names=list(scene['centers']);metadata={k:scene[k] for k in ['source','total_points']};metadata['names']=names
    np.savez_compressed(path,points=scene['points'],colors=scene['colors'],centers=np.asarray(list(scene['centers'].values())).reshape(-1,3),
                        rotations=np.asarray([scene['camera_rotations'][n] for n in names]).reshape(-1,3,3),metadata=np.array(json.dumps(metadata)))


def read_scene(path):
    with np.load(path,allow_pickle=False) as archive:
        data=json.loads(str(archive['metadata']));names=data.pop('names')
        return dict(data,points=archive['points'].copy(),colors=archive['colors'].copy(),centers=dict(zip(names,archive['centers'].copy())),
                    camera_rotations=dict(zip(names,archive['rotations'].copy())),controls={})


def compare_scenes(a,b,register=False,allow_scale=True,distances=False):
    """Transform B's preview only. Report agreement, never ground-truth error."""
    aa,bb=camera_records(a),camera_records(b);pairs=camera_pairs(aa,bb)
    left=np.array([aa[i]['pos'] for i,j in pairs]).reshape(-1,3);right=np.array([bb[j]['pos'] for i,j in pairs]).reshape(-1,3)
    scale=1.;rotation=np.eye(3);translation=np.zeros(3);error=None
    if register:
        try:
            scale,rotation,translation=similarity(right,left)
            if not allow_scale:scale=1.;translation=left.mean(0)-rotation@right.mean(0)
        except ValueError as exception:
            error=str(exception);scale=1.;rotation=np.eye(3);translation=np.zeros(3)
    preview=dict(b,points=scale*(b['points']@rotation.T)+translation,
                 centers={n:scale*(rotation@p)+translation for n,p in b['centers'].items()},
                 camera_rotations={n:rotation@r for n,r in b['camera_rotations'].items()})
    rows=[];segments=[]
    for i,j in pairs:
        na,nb=aa[i]['name'],bb[j]['name'];pa,pb=a['centers'][na],preview['centers'][nb];segments.append([pa,pb])
        angle=None
        if na in a['camera_rotations'] and nb in preview['camera_rotations']:
            angle=float(np.degrees(Rotation.from_matrix(a['camera_rotations'][na].T@preview['camera_rotations'][nb]).magnitude()))
        rows.append({'a_image':na,'b_image':nb,'position_difference':float(np.linalg.norm(pa-pb)),'orientation_difference_deg':angle})
    position=np.array([r['position_difference'] for r in rows]);angles=[r['orientation_difference_deg'] for r in rows if r['orientation_difference_deg'] is not None]
    report={'reference':a['source'],'comparison':b['source'],'frame_fit_requested':register,'frame_fit_applied':register and error is None,
            'allow_scale':allow_scale,'scale':scale,'registration_error':error,'units':'A scene units; metres only if A is metric',
            'matched_cameras':len(pairs),'a_cameras':len(aa),'b_cameras':len(bb),'camera_position_rmse':float(np.sqrt(np.mean(position**2))) if len(position) else None,
            'camera_orientation_rms_deg':float(np.sqrt(np.mean(np.asarray(angles)**2))) if angles else None,
            'transform':{'rotation':rotation.tolist(),'translation':translation.tolist(),'scale':scale},'camera_differences':rows,
            'interpretation':'Agreement between reconstructions, not independent accuracy. Fitted cameras are also evaluated.'}
    heat=None
    if distances:
        values,count=cloud_distances(a['points'],preview['points'])
        if len(values):
            high=float(np.quantile(values,.95));t=np.clip(values/max(high,1e-10),0,1)
            palette=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])
            low=np.minimum((t*4).astype(int),3);blend=t*4-low;heat=((1-blend[:,None])*palette[low]+blend[:,None]*palette[low+1]).astype(np.uint8)
            report['sampled_b_to_a']={'median':float(np.median(values)),'p95':high,'a_samples':count,'b_samples':len(values),
                                      'interpretation':'Directed distance to sampled A points; depends on point density and frame fit.'}
        else:report['distance_error']='Distance colouring needs finite points in both reconstructions; B is shown in coral.'
    return {'b':preview,'segments':np.asarray(segments,dtype=np.float64).reshape(-1,2,3),'report':report,'heat_colors':heat,'pairs':pairs}


class ComparisonPanel:
    def __init__(self,navigator,parent):
        self.nav=navigator;self.editor=navigator.editor;self.closed=False;self.a=self.b=None;self.result=None;self.generation=0;self.load_generation=0;self.process=None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Abu comparison');self.queue=queue.Queue();self.future=None
        self.frame=tk.Frame(parent,bg='#1a2024',width=340)
        self.frame.pack_propagate(False)
        header=tk.Frame(self.frame,bg='#1a2024',padx=10,pady=7);header.pack(fill='x')
        self.editor.label(header,'Reconstruction comparison').pack(side='left')
        self.editor.button(header,'Hide',self.frame.pack_forget).pack(side='right')
        sources=tk.Frame(self.frame,bg='#1a2024',padx=8);sources.pack(fill='x')
        for title,command,tip in [('Load B...',self.load_file,'Choose a single reconstruction JSON, transforms.json or sparse PLY. B is read-only.'),('Run folder...',self.load_folder,'Use the same engine/run formats as the main 3D workspace.'),('Clear B',self.clear,'Remove the comparison preview.')]:
            self.editor.button(sources,title,command,tip).pack(side='left',padx=2)
        footer=tk.Frame(self.frame,bg='#1a2024',padx=10,pady=7);footer.pack(side='bottom',fill='x')
        self.editor.button(footer,'Save comparison report',self.save_report).pack(fill='x',pady=3)
        actions=tk.Frame(footer,bg='#1a2024');actions.pack(fill='x')
        self.editor.button(actions,'Switch A / B',self.switch).pack(side='left');self.editor.button(actions,'Fit visible',self.nav.fit).pack(side='right')
        outer,body=self.editor.scroll_page(self.frame);outer.pack(fill='both',expand=True)
        self.reference=tk.StringVar(self.frame,'A: current control-point alignment');self.source=tk.StringVar(self.frame,'B: choose a reconstruction to compare')
        reference_label=self.editor.label(body,variable=self.reference,muted=True,wraplength=285);reference_label.pack(fill='x',pady=6)
        self.editor.guide(reference_label,'Editable reference A','The alignment currently open in the control-point editor.');self.reference_guide=self.editor.guides[-1]
        self.source_label=self.editor.label(body,variable=self.source,muted=True,wraplength=285);self.source_label.pack(fill='x',pady=6)
        self.editor.guide(self.source_label,'Read-only comparison B','Load a comparison model. Hover here to see its full source path.');self.source_guide=self.editor.guides[-1]
        self.mode=tk.StringVar(self.frame,'A only');self.color=tk.StringVar(self.frame,'Layer colors');self.registration=tk.BooleanVar(self.frame,False);self.allow_scale=tk.BooleanVar(self.frame,True);self.offsets=tk.BooleanVar(self.frame,True)
        for title,var,values in [('Show',self.mode,['A only','B only','Overlay']),('Point color',self.color,['Layer colors','Original RGB','B distance to A'])]:
            self.editor.label(body,title,muted=True).pack(fill='x',pady=(8,3))
            choice=ttk.Combobox(body,textvariable=var,values=values,state='readonly',style='CP.TCombobox');choice.pack(fill='x',pady=(0,5));choice.bind('<<ComboboxSelected>>',lambda e:self.apply())
            if title=='Point color':self.color_field=choice
        self.heat_legend=tk.Canvas(body,height=38,bg='#1a2024',highlightthickness=0)
        self.heat_legend.bind('<Configure>',lambda e:self.draw_legend())
        for title,var,tip in [('Fit B to A',self.registration,'Requires at least three non-collinear matched cameras. This changes the preview only.'),('Allow scale',self.allow_scale,'Switch off for rigid fitting that preserves B scale.'),('Camera offsets',self.offsets,'Connect matching camera centres; differences are not capture paths.')]:
            widget=tk.Checkbutton(body,text=title,variable=var,command=self.apply,bg='#1a2024',fg='#e7edef',selectcolor='#242d32',activebackground='#1a2024');widget.pack(anchor='w',pady=4);self.editor.guide(widget,title,tip)
        self.summary=tk.StringVar(self.frame,'A: mint / B: coral. Load B to compare without editing either reconstruction.')
        self.editor.label(body,variable=self.summary,wraplength=285).pack(fill='x',pady=10)
        self.selection_info=tk.StringVar(self.frame,'Select a camera to inspect its individual difference.')
        self.editor.label(body,variable=self.selection_info,muted=True,wraplength=285).pack(fill='x',pady=8)
        self.editor.style_dropdowns(self.frame);self.editor.bind_page_scroll(outer);self.poll_job=self.frame.after(60,self.poll)
    def show(self):self.frame.pack(side='right',fill='y',before=self.nav.surface)
    def set_reference(self,a):
        previous=self.a;changed=previous is not None and previous['source']!=a['source']
        equal=previous is not None and not changed and np.array_equal(previous['points'],a['points']) and list(previous['centers'])==list(a['centers']) and all(np.array_equal(previous['centers'][n],a['centers'][n]) and np.array_equal(previous['camera_rotations'][n],a['camera_rotations'][n]) for n in a['centers'])
        self.a=a;self.reference.set('A · '+Path(a['source']).name);self.reference_guide.text=a['source']
        if a['source']=='photos-only':self.clear();return
        if changed:self.result=None;self.nav.set_comparison(None)
        if self.b and not equal:self.apply()
    def load_file(self):
        value=filedialog.askopenfilename(parent=self.editor.window,title='Load comparison reconstruction B',filetypes=[('Reconstructions','*.json *.ply'),('All files','*.*')])
        if value:self.load(value)
    def load_folder(self):
        value=filedialog.askdirectory(parent=self.editor.window,title='Choose completed comparison run B')
        if value:self.load(value)
    def load(self,path):
        if self.process and self.process.poll() is None:self.process.terminate()
        self.load_generation+=1;generation=self.load_generation;self.summary.set('Loading B in a separate process… You can keep navigating.');self.show()
        self.editor.audit.event('comparison_load',source=path)
        def work():
            if self.closed or generation!=self.load_generation:return
            try:
                with tempfile.TemporaryDirectory(prefix='abu_compare_') as folder:
                    target=Path(folder)/'scene.npz';command=[sys.executable,'-B',str(Path(__file__).resolve()),'--source',str(path),'--output',str(target)]
                    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.process=process
                    if self.closed or generation!=self.load_generation:process.terminate()
                    output,_=process.communicate()
                    if self.process is process:self.process=None
                    if self.closed or generation!=self.load_generation:return
                    if process.returncode:raise RuntimeError(output[-3000:] or 'Unable to load comparison reconstruction.')
                    self.queue.put(('loaded',generation,read_scene(target)))
            except Exception as error:
                if not self.closed and generation==self.load_generation:self.queue.put(('load_error',generation,str(error)))
        self.executor.submit(work)
    def clear(self):
        self.load_generation+=1;self.generation+=1
        if self.process and self.process.poll() is None:self.process.terminate()
        self.b=self.result=None;self.mode.set('A only');self.source.set('B: choose a reconstruction to compare');self.summary.set('Comparison cleared. A remains editable.');self.nav.set_comparison(None)
        self.selection_info.set('Select a camera to inspect its individual difference.')
        self.heat_legend.pack_forget()
    def draw_legend(self):
        canvas=self.heat_legend;canvas.delete('all')
        if not self.result or 'sampled_b_to_a' not in self.result['report']:return
        width=max(100,canvas.winfo_width());palette=np.array([[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]])
        for i in range(40):
            t=i/39*4;lower=min(int(t),3);color=((1-(t-lower))*palette[lower]+(t-lower)*palette[lower+1]).astype(int)
            canvas.create_rectangle(i*width/40,0,(i+1)*width/40,12,fill='#%02x%02x%02x'%tuple(color),outline='')
        canvas.create_text(1,28,text='0',anchor='w',fill='#a0adb5',font=('Segoe UI',8))
        high=self.result['report']['sampled_b_to_a']['p95'];canvas.create_text(width-2,28,text=f'≥ P95: {high:.4g} A units',anchor='e',fill='#a0adb5',font=('Segoe UI',8))
    def describe_camera(self,name,layer):
        if not self.result:return
        row=next((r for r in self.result['report']['camera_differences'] if r['a_image' if layer=='A' else 'b_image']==name),None)
        message=layer+' camera · '+name
        if row:
            angle=row['orientation_difference_deg'];message+=f"\nPosition difference: {row['position_difference']:.6g} A units\nOrientation difference: "+('unavailable' if angle is None else f'{angle:.5g}°')
        else:message+='\nNo uniquely matched camera in the other reconstruction.'
        self.selection_info.set(message)
    def apply(self):
        if not self.a or not self.b or self.closed:return
        self.generation+=1;generation=self.generation
        if self.future:self.future.cancel()
        a,b=self.a,self.b;register=self.registration.get();allow_scale=self.allow_scale.get();distance=self.color.get()=='B distance to A'
        self.summary.set('Preparing comparison… viewpoint preserved.')
        def work():
            try:
                result=compare_scenes(a,b,register,allow_scale,distance)
                if not self.closed and generation==self.generation:self.queue.put(('compared',generation,result))
            except Exception as error:
                if not self.closed and generation==self.generation:self.queue.put(('compare_error',generation,str(error)))
        self.future=self.executor.submit(work)
    def switch(self):
        if not self.b:self.summary.set('Load a comparison B first.');return
        self.mode.set('B only' if self.mode.get()=='A only' else 'A only');self.apply()
    def save_report(self):
        if not self.result:self.summary.set('Load B and prepare a comparison before saving a report.');return
        value=filedialog.asksaveasfilename(parent=self.editor.window,title='Save comparison report',defaultextension='.json',initialfile='reconstruction_comparison.json')
        if not value:return
        try:
            if Path(value).resolve() in {Path(p).resolve() for p in [self.a['source'],self.b['source']]}:raise ValueError('The report must not overwrite a source reconstruction.')
            from control_points_model import atomic_json
            atomic_json(value,self.result['report']);self.editor.audit.event('comparison_report_saved',path=value)
        except Exception as error:self.editor.error('Cannot save comparison report',error)
    def poll(self):
        if self.closed:return
        while True:
            try:event=self.queue.get_nowait()
            except queue.Empty:break
            kind,generation,value=event
            if generation!=(self.load_generation if kind.startswith('load') else self.generation):continue
            if kind=='loaded':
                self.b=value;self.source.set('B · '+Path(value['source']).name);self.source_guide.text=value['source'];self.mode.set('Overlay');self.fit_pending=True;self.apply()
            elif kind=='compared':
                self.result=value;report=value['report'];rmse=report['camera_position_rmse'];angle=report['camera_orientation_rms_deg']
                frame='Raw frames' if not report['frame_fit_applied'] else f"Fitted preview · scale {report['scale']:.6g}"
                message=f"A · mint / B · coral. {frame}. {report['matched_cameras']} matching cameras; position RMS "+('—' if rmse is None else f'{rmse:.5g}')+' A units; angle RMS '+('—' if angle is None else f'{angle:.4g}°')
                if report['registration_error']:message+='\nFit unavailable: '+report['registration_error']+' Showing raw B.'
                if 'sampled_b_to_a' in report:
                    d=report['sampled_b_to_a'];message+=f"\nSampled B → A: median {d['median']:.5g}, P95 {d['p95']:.5g} A units. Depends on density."
                if 'distance_error' in report:message+='\n'+report['distance_error']
                message+='\nAgreement is not ground-truth accuracy. Controls belong to A; B is read-only.';self.summary.set(message)
                if self.color.get()=='B distance to A' and value['heat_colors'] is not None:
                    self.heat_legend.pack(fill='x',after=self.color_field,pady=3);self.draw_legend()
                else:self.heat_legend.pack_forget()
                self.nav.set_comparison(value)
                if getattr(self,'fit_pending',False):self.fit_pending=False;self.nav.fit()
                self.editor.audit.event('comparison_ready',reference=report['reference'],comparison=report['comparison'],matched=report['matched_cameras'],frame_fit=report['frame_fit_applied'])
            else:self.summary.set('Comparison unavailable: '+value);self.editor.audit.event('comparison_error',error=value)
        self.poll_job=self.frame.after(60,self.poll)
    def close(self):
        self.closed=True;self.generation+=1;self.load_generation+=1;self.frame.after_cancel(self.poll_job)
        if self.process and self.process.poll() is None:self.process.terminate()
        self.executor.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();save_scene(args.output,load_scene(args.source))
