"""Embedded Windows OpenGL navigator; no web server or extra packages required."""
import ctypes as ct
from ctypes import wintypes as wt
import math
import tkinter as tk
from tkinter import ttk
import numpy as np


class PixelFormat(ct.Structure):
    _fields_=[('nSize',wt.WORD),('nVersion',wt.WORD),('dwFlags',wt.DWORD),
              *[(n,ct.c_ubyte) for n in ['iPixelType','cColorBits','cRedBits','cRedShift','cGreenBits','cGreenShift','cBlueBits','cBlueShift','cAlphaBits','cAlphaShift','cAccumBits','cAccumRedBits','cAccumGreenBits','cAccumBlueBits','cAccumAlphaBits','cDepthBits','cStencilBits','cAuxBuffers','iLayerType','bReserved']],
              ('dwLayerMask',wt.DWORD),('dwVisibleMask',wt.DWORD),('dwDamageMask',wt.DWORD)]


class NativeControlNavigator(tk.Frame):
    def __init__(self,editor,parent):
        super().__init__(parent,bg='#111518');self.editor=editor;self.closed=False;self.context=None;self.job=None;self.data=None
        self.selected='';self.camera_selected='';self.selected_camera_layer='A';self.comparison=None;self.center=np.zeros(3);self.yaw=-25.;self.pitch=20.;self.distance=3.;self.pan=np.zeros(2);self.drag=None
        bar=tk.Frame(self,bg='#1a2024',padx=8,pady=6);bar.pack(fill='x')
        self.cloud=tk.BooleanVar(self,True);self.cameras=tk.BooleanVar(self,True);self.controls=tk.BooleanVar(self,True);self.rays=tk.BooleanVar(self,True)
        for title,var in [('Cloud',self.cloud),('Cameras',self.cameras),('Controls',self.controls),('Sight lines',self.rays)]:
            tk.Checkbutton(bar,text=title,variable=var,command=self.request,bg='#1a2024',fg='#e7edef',selectcolor='#242d32',activebackground='#1a2024').pack(side='left')
        for title,command in [('Fit',self.fit),('Focus selected',self.focus_selected),('Top',lambda:self.preset(0,89)),('Front',lambda:self.preset(0,0)),('Browser view',lambda:editor.open_3d(browser=True))]:editor.button(bar,title,command).pack(side='right',padx=3)
        selection=tk.Frame(self,bg='#1a2024',padx=8,pady=4);selection.pack(fill='x')
        self.point_var=tk.StringVar(self);self.camera_var=tk.StringVar(self)
        editor.label(selection,'Point').pack(side='left');self.point_choice=ttk.Combobox(selection,textvariable=self.point_var,state='readonly',style='CP.TCombobox',width=20);self.point_choice.pack(side='left',padx=8)
        self.point_choice.bind('<<ComboboxSelected>>',lambda e:self.select_control(self.point_var.get()))
        editor.label(selection,'Camera').pack(side='left');self.camera_choice=ttk.Combobox(selection,textvariable=self.camera_var,state='readonly',style='CP.TCombobox',width=32);self.camera_choice.pack(side='left',padx=8)
        self.camera_choice.bind('<<ComboboxSelected>>',lambda e:self.camera_option_selected())
        editor.button(selection,'2D workspace',lambda:editor.tabs.select(editor.workspace)).pack(side='right')
        editor.style_dropdowns(selection)
        self.info=tk.StringVar(self,'Preparing hardware 3D renderer…');editor.label(self,variable=self.info,muted=True,padx=10).pack(side='bottom',fill='x')
        editor.label(self,'Drag: orbit · Right/middle drag: pan · Wheel: zoom · Click: select point/camera · F: fit',muted=True,padx=10,pady=5).pack(side='bottom',fill='x')
        self.scene_body=tk.Frame(self,bg='#111518');self.scene_body.pack(fill='both',expand=True)
        self.surface=tk.Frame(self.scene_body,bg='#111518',takefocus=True);self.surface.pack(side='left',fill='both',expand=True)
        from control_points_comparison import ComparisonPanel
        self.compare=ComparisonPanel(self,self.scene_body)
        self.editor.button(selection,'Compare reconstructions',self.toggle_comparison,'Load a read-only B reconstruction, then inspect A/B or overlay from one viewpoint.').pack(side='right',padx=5)
        self.surface.bind('<Map>',self.initialize);self.surface.bind('<Configure>',lambda e:self.request());self.surface.bind('<Expose>',lambda e:self.request())
        self.surface.bind('<ButtonPress-1>',lambda e:self.press(e,'orbit'));self.surface.bind('<B1-Motion>',self.move);self.surface.bind('<ButtonRelease-1>',self.release)
        for button in [2,3]:
            self.surface.bind(f'<ButtonPress-{button}>',lambda e:self.press(e,'pan'));self.surface.bind(f'<B{button}-Motion>',self.move);self.surface.bind(f'<ButtonRelease-{button}>',self.release)
        self.surface.bind('<MouseWheel>',self.wheel);self.surface.bind('<f>',lambda e:self.fit());self.surface.bind('<Key-1>',lambda e:self.preset(0,0));self.surface.bind('<Key-7>',lambda e:self.preset(0,89))
        self.surface.bind('<b>',lambda e:self.compare.switch())
    def toggle_comparison(self):
        if self.compare.frame.winfo_manager():self.compare.frame.pack_forget()
        else:self.compare.show()
    def initialize(self,event=None):
        if self.context or self.closed:return
        try:
            self.gl=ct.WinDLL('opengl32');self.gdi=ct.WinDLL('gdi32');self.user=ct.WinDLL('user32')
            self.user.GetDC.argtypes=[wt.HWND];self.user.GetDC.restype=wt.HDC;self.user.ReleaseDC.argtypes=[wt.HWND,wt.HDC]
            self.gdi.ChoosePixelFormat.argtypes=[wt.HDC,ct.POINTER(PixelFormat)];self.gdi.SetPixelFormat.argtypes=[wt.HDC,ct.c_int,ct.POINTER(PixelFormat)];self.gdi.SwapBuffers.argtypes=[wt.HDC]
            self.gl.wglCreateContext.argtypes=[wt.HDC];self.gl.wglCreateContext.restype=ct.c_void_p
            self.gl.wglMakeCurrent.argtypes=[wt.HDC,ct.c_void_p];self.gl.wglDeleteContext.argtypes=[ct.c_void_p]
            self.hwnd=self.surface.winfo_id();self.dc=self.user.GetDC(self.hwnd)
            p=PixelFormat();p.nSize=ct.sizeof(p);p.nVersion=1;p.dwFlags=0x4|0x20|0x1;p.iPixelType=0;p.cColorBits=32;p.cDepthBits=24;p.cStencilBits=8
            pf=self.gdi.ChoosePixelFormat(self.dc,ct.byref(p))
            if not pf or not self.gdi.SetPixelFormat(self.dc,pf,ct.byref(p)):raise RuntimeError('OpenGL pixel format unavailable')
            self.context=self.gl.wglCreateContext(self.dc)
            if not self.context or not self.gl.wglMakeCurrent(self.dc,self.context):raise RuntimeError('OpenGL context unavailable')
            for name,args in [('glClearColor',[ct.c_float]*4),('glClear',[ct.c_uint]),('glEnable',[ct.c_uint]),('glDisable',[ct.c_uint]),('glEnableClientState',[ct.c_uint]),('glDisableClientState',[ct.c_uint]),('glVertexPointer',[ct.c_int,ct.c_uint,ct.c_int,ct.c_void_p]),('glColorPointer',[ct.c_int,ct.c_uint,ct.c_int,ct.c_void_p]),('glColor3f',[ct.c_float]*3),('glPointSize',[ct.c_float]),('glLineWidth',[ct.c_float]),('glDrawArrays',[ct.c_uint,ct.c_int,ct.c_int]),('glViewport',[ct.c_int]*4),('glMatrixMode',[ct.c_uint]),('glLoadMatrixd',[ct.POINTER(ct.c_double)]),('glBlendFunc',[ct.c_uint]*2)]:getattr(self.gl,name).argtypes=args
            self.gl.glGetString.argtypes=[ct.c_uint];self.gl.glGetString.restype=ct.c_char_p
            self.renderer=self.gl.glGetString(0x1F01).decode(errors='replace');self.version=self.gl.glGetString(0x1F02).decode(errors='replace')
            self.gl.glClearColor(.067,.083,.094,1);self.gl.glEnable(0x0B71);self.gl.glEnable(0x0B20);self.gl.glEnable(0x0BE2);self.gl.glBlendFunc(0x0302,0x0303)
            self.editor.audit.event('native_3d_renderer',renderer=self.renderer,version=self.version)
            self.info.set('Hardware renderer: '+self.renderer);self.request()
        except Exception as error:
            self.editor.audit.error('native_3d_error',error);self.info.set('3D renderer unavailable: '+str(error)+' · use Browser view')
    def set_data(self,data):
        changed=self.data is None or self.data.get('source')!=data.get('source')
        if changed:self.comparison=None
        self.data=data
        center=data['center'];scale=max(data['extent'],1e-8);self.origin=center;self.scale=scale
        self.points=np.ascontiguousarray((data['points']-center)/scale,dtype=np.float32);self.colors=np.ascontiguousarray(data['colors'],dtype=np.uint8)
        self.camera_names=list(data['centers']);self.camera_xyz=np.ascontiguousarray([(data['centers'][n]-center)/scale for n in self.camera_names],dtype=np.float32).reshape(-1,3)
        self.point_names=list(data['controls']);self.control_xyz=np.ascontiguousarray([(data['controls'][n]['position']-center)/scale for n in self.point_names],dtype=np.float32).reshape(-1,3)
        palette={'tie':(112,219,192),'gcp':(120,180,255),'check':(255,125,125)}
        self.control_colors=np.ascontiguousarray([palette[data['controls'][n]['kind']] if data['controls'][n]['enabled'] else (130,140,145) for n in self.point_names],dtype=np.uint8).reshape(-1,3)
        self.point_choice.configure(values=self.point_names);self.update_camera_choices()
        if self.point_var.get() not in self.point_names:self.point_var.set('')
        if self.selected_camera_layer=='A' and self.camera_selected not in self.camera_names:self.camera_var.set('');self.camera_selected=''
        self.camera_axes=np.asarray([[self.camera_xyz[i],self.camera_xyz[i]+data['camera_rotations'][name][:,axis]*.018] for i,name in enumerate(self.camera_names) for axis in range(3) if name in data.get('camera_rotations',{})],dtype=np.float32).reshape(-1,3)
        self.axis_colors=np.tile(np.repeat(np.array([[220,95,95],[100,210,130],[110,155,240]],dtype=np.uint8),2,axis=0),(len(self.camera_axes)//6,1))
        self.info.set(f"{len(self.points):,} cloud points · {len(self.camera_names)} cameras · {len(self.point_names)} triangulated controls · "+getattr(self,'renderer','starting renderer'))
        if changed:self.fit()
        self.compare.set_reference(data)
        self.request()
    def update_camera_choices(self):
        names=getattr(self,'camera_names',[])
        self.camera_options={('A · '+n if self.comparison else n):('A',n) for n in names}
        if self.comparison:self.camera_options.update({'B · '+n:('B',n) for n in self.comparison['b']['centers']})
        self.camera_choice.configure(values=list(self.camera_options))
    def camera_option_selected(self):
        layer,name=self.camera_options[self.camera_var.get()];self.select_camera(name,layer)
    def set_comparison(self,result):
        self.comparison=result
        if result and self.data:
            b=result['b'];self.b_names=list(b['centers'])
            self.b_points=np.ascontiguousarray((b['points']-self.origin)/self.scale,dtype=np.float32)
            self.b_cameras=np.ascontiguousarray([(b['centers'][n]-self.origin)/self.scale for n in self.b_names],dtype=np.float32).reshape(-1,3)
            self.b_axes=np.asarray([[self.b_cameras[i],self.b_cameras[i]+b['camera_rotations'][name][:,axis]*.018] for i,name in enumerate(self.b_names) for axis in range(3)],dtype=np.float32).reshape(-1,3)
            self.offset_segments=np.ascontiguousarray((result['segments'].reshape(-1,3)-self.origin)/self.scale,dtype=np.float32)
        elif self.selected_camera_layer=='B':self.selected_camera_layer='A';self.camera_selected='';self.camera_var.set('')
        self.update_camera_choices();self.request()
    def visible_layers(self):
        if not self.comparison:return True,False
        mode=self.compare.mode.get();return mode!='B only',mode!='A only'
    def select_control(self,name):
        self.selection_kind='point'
        if self.comparison and self.compare.mode.get()=='B only':self.compare.mode.set('Overlay');self.compare.apply()
        self.editor.select(name)
    def set_selected(self,name):
        if name!=self.selected:self.selection_kind='point'
        self.selected=name;self.point_var.set(name if name in self.point_names else '');self.request()
    def select_camera(self,name,layer='A'):
        self.selection_kind='camera';self.selected_camera_layer=layer;self.camera_selected=name;self.camera_var.set(layer+' · '+name if self.comparison else name)
        image=name if layer=='A' else next((r['a_image'] for r in self.comparison['report']['camera_differences'] if r['b_image']==name),None)
        if image:self.editor.choose_image(image)
        else:self.editor.status.set('B camera selected for inspection. No matching A image is available in the editable workspace.')
        self.compare.describe_camera(name,layer)
        self.editor.audit.event('native_3d_camera_select',image=name,layer=layer);self.request()
    def fit(self):
        self.pan[:]=0;self.center=np.zeros(3);self.distance=2.6
        if self.data:
            a,b=self.visible_layers();bounds=[]
            for active,points,cameras in [(a,self.points,self.camera_xyz),(b,getattr(self,'b_points',np.empty((0,3))),getattr(self,'b_cameras',np.empty((0,3))))]:
                if not active:continue
                if self.cloud.get() and len(points):bounds.append(np.quantile(points,[.01,.99],axis=0))
                if self.cameras.get() and len(cameras):bounds.append(cameras)
            if a and self.controls.get() and len(self.control_xyz):bounds.append(self.control_xyz)
            if bounds:
                coords=np.concatenate(bounds);self.center=(coords.min(0)+coords.max(0))/2
                span=max(float(np.linalg.norm(coords.max(0)-coords.min(0))),.02);aspect=max(.1,self.surface.winfo_width()/max(1,self.surface.winfo_height()))
                self.distance=span*1.5/max(.3,min(1.,aspect))
        self.request()
    def focus_selected(self):
        if self.data and getattr(self,'selection_kind','point')=='point' and self.selected in self.data['controls']:self.center=(self.data['controls'][self.selected]['position']-self.origin)/self.scale;self.pan[:]=0;self.request()
        elif self.data:
            scene=self.comparison['b'] if self.selected_camera_layer=='B' and self.comparison else self.data
            if self.camera_selected in scene['centers']:self.center=(scene['centers'][self.camera_selected]-self.origin)/self.scale;self.pan[:]=0;self.request()
    def preset(self,yaw,pitch):self.yaw=yaw;self.pitch=pitch;self.request()
    def press(self,e,mode):self.surface.focus_set();self.drag=(e.x,e.y,mode,e.x,e.y)
    def move(self,e):
        if not self.drag:return
        x,y,mode,sx,sy=self.drag;dx,dy=e.x-x,e.y-y
        if mode=='pan' or e.state&1:self.pan+=np.array([dx,-dy])*self.distance/max(200,self.surface.winfo_height())
        else:self.yaw+=dx*.4;self.pitch=np.clip(self.pitch+dy*.4,-89,89)
        self.drag=(e.x,e.y,mode,sx,sy);self.request()
    def release(self,e):
        if self.drag and self.drag[2]=='orbit' and math.hypot(e.x-self.drag[3],e.y-self.drag[4])<4:self.pick(e.x,e.y)
        self.drag=None
    def wheel(self,e):self.distance=float(np.clip(self.distance*(.85 if e.delta>0 else 1/.85),.02,10000));self.request();return 'break'
    def matrices(self):
        width=max(1,self.surface.winfo_width());height=max(1,self.surface.winfo_height());aspect=width/height
        y,p=np.radians([self.yaw,self.pitch]);ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
        view=np.eye(4);view[:3,:3]=rx@ry;view[:3,3]=-view[:3,:3]@self.center+np.array([*self.pan,-self.distance])
        near=max(.0001,self.distance/1000);far=max(100.,self.distance*100);f=1/math.tan(math.radians(45)/2)
        projection=np.zeros((4,4));projection[0,0]=f/aspect;projection[1,1]=f;projection[2,2]=(far+near)/(near-far);projection[2,3]=2*far*near/(near-far);projection[3,2]=-1
        return view,projection,width,height
    def project(self,points):
        v,p,w,h=self.matrices();clip=np.c_[points,np.ones(len(points))]@(p@v).T
        valid=(clip[:,3]>0);ndc=clip[:,:3]/np.maximum(clip[:,3,None],1e-12);valid&=(np.abs(ndc[:,:2])<=1).all(axis=1)&(np.abs(ndc[:,2])<=1)
        return np.c_[(ndc[:,0]+1)*w/2,(1-ndc[:,1])*h/2],valid,ndc[:,2]
    def pick(self,x,y):
        if self.data is None:return
        hits=[]
        a,b=self.visible_layers()
        for kind,names,points,enabled in [('point',self.point_names,self.control_xyz,self.controls.get() and a),('camera',self.camera_names,self.camera_xyz,self.cameras.get() and a),('b_camera',getattr(self,'b_names',[]),getattr(self,'b_cameras',np.empty((0,3))),self.cameras.get() and b)]:
            if not enabled:continue
            screen,valid,depth=self.project(points);dist=np.linalg.norm(screen-[x,y],axis=1)
            for i in np.flatnonzero(valid&(dist<12)):hits.append((dist[i],depth[i],kind,names[i]))
        if not hits:return
        _,_,kind,name=min(hits)
        if kind=='point':self.select_control(name)
        else:self.select_camera(name,'B' if kind=='b_camera' else 'A')
        self.info.set(('B camera' if kind=='b_camera' else kind.capitalize())+': '+name+' · '+getattr(self,'renderer',''));self.request()
    def request(self):
        if self.closed or self.job:return
        self.job=self.after(16,self.draw)
    def draw_arrays(self,points,colors=None,size=2.,mode=0,color=(.5,.6,.7)):
        if not len(points):return
        g=self.gl;points=np.ascontiguousarray(points,dtype=np.float32);g.glEnableClientState(0x8074);g.glVertexPointer(3,0x1406,0,points.ctypes.data)
        if colors is not None:
            colors=np.ascontiguousarray(colors,dtype=np.uint8);g.glEnableClientState(0x8076);g.glColorPointer(3,0x1401,0,colors.ctypes.data)
        else:g.glDisableClientState(0x8076);g.glColor3f(*color)
        g.glPointSize(size);g.glDrawArrays(mode,0,len(points));g.glDisableClientState(0x8076);g.glDisableClientState(0x8074)
    def draw(self):
        self.job=None
        if self.closed or not self.context or not self.winfo_ismapped():return
        g=self.gl;g.wglMakeCurrent(self.dc,self.context);view,projection,w,h=self.matrices();g.glViewport(0,0,w,h);g.glClear(0x4000|0x0100)
        for kind,matrix in [(0x1701,projection),(0x1700,view)]:g.glMatrixMode(kind);g.glLoadMatrixd(np.ascontiguousarray(matrix.T).ctypes.data_as(ct.POINTER(ct.c_double)))
        if self.data:
            a,b=self.visible_layers();color=self.compare.color.get() if self.comparison else 'Original RGB'
            if a:
                if self.cloud.get():self.draw_arrays(self.points,self.colors if color=='Original RGB' else None,size=2.,color=(.44,.86,.75))
                if self.cameras.get():self.draw_arrays(self.camera_xyz,size=7.,color=(.44,.86,.75));self.draw_arrays(self.camera_axes,self.axis_colors,mode=1)
                if self.controls.get():self.draw_arrays(self.control_xyz,self.control_colors,size=12.)
            if b:
                if self.cloud.get():
                    colors=self.comparison['b']['colors'] if color=='Original RGB' else self.comparison['heat_colors'] if color=='B distance to A' else None
                    self.draw_arrays(self.b_points,colors,size=2.,color=(1.,.61,.5))
                if self.cameras.get():self.draw_arrays(self.b_cameras,size=7.,color=(1.,.61,.5));self.draw_arrays(self.b_axes,mode=1,color=(1.,.61,.5))
            if a and b and self.compare.offsets.get():self.draw_arrays(self.offset_segments,mode=1,color=(1.,.8,.45))
            if a and self.selected in self.data['controls']:
                point=self.data['controls'][self.selected];position=(point['position']-self.origin)/self.scale
                self.draw_arrays(np.array([position]),size=18.,color=(1.,.8,.4))
                if self.rays.get():
                    segments=np.asarray([[position,(self.data['centers'][n]-self.origin)/self.scale] for n in point['shots'] if n in self.data['centers']],dtype=np.float32).reshape(-1,3)
                    self.draw_arrays(segments,mode=1,color=(1.,.8,.4))
            scene=self.comparison['b'] if self.selected_camera_layer=='B' and self.comparison else self.data
            if (a if self.selected_camera_layer=='A' else b) and self.camera_selected in scene['centers']:self.draw_arrays(np.array([(scene['centers'][self.camera_selected]-self.origin)/self.scale]),size=12.,color=(.47,.71,1.))
        self.gdi.SwapBuffers(self.dc)
    def close(self):
        if self.closed:return
        self.closed=True
        self.compare.close()
        if self.job:self.after_cancel(self.job)
        if self.context:self.gl.wglMakeCurrent(None,None);self.gl.wglDeleteContext(self.context);self.context=None
        if hasattr(self,'dc'):self.user.ReleaseDC(self.hwnd,self.dc)
