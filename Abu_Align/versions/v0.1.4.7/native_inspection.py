"""Embedded workspace controls and two-run comparison, using the native renderer."""
import os
import queue
import threading
from collections import OrderedDict
import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageTk
from desktop_ui import BG, SURFACE, FIELD, TEXT, MUTED, ACCENT, BORDER, button, label, ScrollPage
from rig_designer_ui import wrap_row, friendly_choices
from viewer_workspace import camera_pairs, similarity, transform_cameras, cloud_distances, scene_bounds


class CameraPhotoViewer(tk.Toplevel):
    """Separate inspection window displaying the high-resolution photo for the selected camera."""
    def __init__(self, master, on_step=None, on_focus=None, on_look=None):
        super().__init__(master)
        self.title("Abu Align - Camera Photo Inspection")
        self.geometry("820x660")
        self.minsize(480, 360)
        self.configure(bg=BG)

        self.on_step = on_step
        self.on_focus = on_focus
        self.on_look = on_look
        self.current_img_path = None
        self._raw_pil = None
        self._tk_img = None

        # Top Header Bar
        top_bar = tk.Frame(self, bg=SURFACE, padx=12, pady=8)
        top_bar.pack(fill="x", side="top")
        self.title_lbl = tk.Label(top_bar, text="Camera Photo", font=("Segoe UI", 10, "bold"), fg=TEXT, bg=SURFACE)
        self.title_lbl.pack(anchor="w")
        self.meta_lbl = tk.Label(top_bar, text="", font=("Segoe UI", 9), fg=MUTED, bg=SURFACE)
        self.meta_lbl.pack(side="bottom", anchor="w", pady=(4, 0))

        # Bottom Action Bar
        bottom_bar = tk.Frame(self, bg=SURFACE, padx=12, pady=8)
        bottom_bar.pack(fill="x", side="bottom")

        btn_prev = tk.Button(bottom_bar, text="◀ Previous", command=lambda: self.on_step(-1) if self.on_step else None,
                             bg=FIELD, fg=TEXT, activebackground=BORDER, relief="flat", padx=8, pady=4, cursor="hand2")
        btn_prev.pack(side="left", padx=2)

        btn_next = tk.Button(bottom_bar, text="Next ▶", command=lambda: self.on_step(1) if self.on_step else None,
                             bg=FIELD, fg=TEXT, activebackground=BORDER, relief="flat", padx=8, pady=4, cursor="hand2")
        btn_next.pack(side="left", padx=2)

        btn_focus = tk.Button(bottom_bar, text="Focus", command=lambda: self.on_focus() if self.on_focus else None,
                              bg=FIELD, fg=TEXT, activebackground=BORDER, relief="flat", padx=8, pady=4, cursor="hand2")
        btn_focus.pack(side="left", padx=(10, 2))

        btn_look = tk.Button(bottom_bar, text="Look through", command=lambda: self.on_look() if self.on_look else None,
                             bg=FIELD, fg=TEXT, activebackground=BORDER, relief="flat", padx=8, pady=4, cursor="hand2")
        btn_look.pack(side="left", padx=2)

        btn_sys = tk.Button(bottom_bar, text="Open original", command=self._open_system_viewer,
                            bg=ACCENT, fg="#111518", font=("Segoe UI", 9, "bold"), activebackground="#8ef0d8", relief="flat", padx=10, pady=4, cursor="hand2")
        btn_sys.pack(side="right", padx=4)

        # Center Canvas for photo
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)

        # Keyboard navigation
        self.bind("<Left>", lambda e: self.on_step(-1) if self.on_step else None)
        self.bind("<Right>", lambda e: self.on_step(1) if self.on_step else None)
        self.bind("<Escape>", lambda e: self.destroy())

    def show_camera(self, camera_dict, img_path, index, total):
        self.current_img_path = img_path
        name = camera_dict.get("name", f"camera_{index:04d}")
        pos = camera_dict.get("pos", np.zeros(3))
        pos_str = f"XYZ: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"

        self.title_lbl.configure(text=f"{os.path.basename(name)}  ·  {index + 1} / {total}")
        self.title(f"Photo: {name} - Camera {index + 1}/{total}")

        if img_path and os.path.isfile(img_path):
            try:
                with Image.open(img_path) as source:
                    from PIL import ImageOps
                    self._raw_pil = ImageOps.exif_transpose(source).convert("RGB")
                w, h = self._raw_pil.size
                self.meta_lbl.configure(text=f"{w}x{h} px · {pos_str}")
                self._render_photo()
            except Exception as e:
                self._raw_pil = None
                self.meta_lbl.configure(text=f"Error loading: {e}")
                self.canvas.delete("all")
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                    text=f"Cannot load image:\n{img_path}\n({e})",
                    fill=MUTED, font=("Segoe UI", 11), justify="center"
                )
        else:
            self._raw_pil = None
            self.meta_lbl.configure(text=f"No image on disk · {pos_str}")
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                text=f"Image file not found on disk for:\n{name}\n\nSearch path: {img_path or 'N/A'}",
                fill=MUTED, font=("Segoe UI", 11), justify="center"
            )

    def _on_resize(self, event=None):
        if self._raw_pil is not None:
            self._render_photo()

    def _render_photo(self):
        cw = max(50, self.canvas.winfo_width())
        ch = max(50, self.canvas.winfo_height())
        if self._raw_pil is None or cw < 10 or ch < 10:
            return

        iw, ih = self._raw_pil.size
        scale = min((cw-24) / iw, (ch-24) / ih, 1.0)
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        resized = self._raw_pil.resize((nw, nh), Image.Resampling.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        cx = (cw - nw) // 2
        cy = (ch - nh) // 2
        self.canvas.create_image(cx, cy, anchor="nw", image=self._tk_img)

    def _open_system_viewer(self):
        if self.current_img_path and os.path.isfile(self.current_img_path):
            try:
                os.startfile(self.current_img_path)
            except Exception:
                pass


class NativeInspection:
    def __init__(self, *args, **kwargs):
        self.layers = {}
        self._run_cache = OrderedDict()
        self._load_results = queue.Queue()
        self._load_generation = 0
        self._loading = False
        self._pending_load = False
        self._first_scene = True
        self.view_matrix = None
        self.selected_camera = -1
        self._camera_screen = []
        self._saved_view = None
        self.photo_viewer_window = None
        super().__init__(*args, **kwargs)
        self._poll_id = self.after(80, self._poll_load)
        self.bind('<Destroy>', self._cleanup_inspection, add='+')

    def _build_toolbar(self):
        self.compare_mode = tk.StringVar(self, 'A only')
        self.comparison_run = tk.StringVar(self, '')
        self.register_runs = tk.BooleanVar(self, False)
        self.distance_colors = tk.BooleanVar(self, False)
        self.navigation_mode = tk.StringVar(self, 'Orbit')
        self.inspection_profile = tk.StringVar(self, 'Overview')
        self.show_points_var = tk.BooleanVar(self, True)
        self.camera_choice = tk.StringVar(self, '')
        self.auto_open_photo_var = tk.BooleanVar(self, False)
        self._compare_visible = False
        self._inspector_visible = False
        self.scene_body = tk.Frame(self, bg=BG)
        self.compare_scroll = ScrollPage(self.scene_body)
        self.compare_scroll.configure(width=330)
        self.compare_scroll.pack_propagate(False)

        header = tk.Frame(self, bg=SURFACE, padx=12, pady=10)
        header.pack(fill='x')
        header.columnconfigure(1, weight=1)
        label(header, 'Reconstruction', muted=True).grid(row=0, column=0, padx=(0, 10))
        self.run_combo = ttk.Combobox(header, textvariable=self.run_choice_var, state='readonly', width=20, postcommand=self.refresh_runs)
        self.run_combo.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        self.run_combo.bind('<<ComboboxSelected>>', lambda e: self.load_selected_run())
        button(header, 'Refresh', self._reload_runs).grid(row=0, column=2, padx=3)
        self.compare_toggle = button(header, 'Compare', self.toggle_comparison)
        self.compare_toggle.grid(row=0, column=3, padx=3)
        self.inspect_toggle = button(header, 'Inspect', self.toggle_inspector)
        self.inspect_toggle.grid(row=0, column=4, padx=3)
        button(header, 'Report', self.show_reconstruction_report).grid(row=0, column=5, padx=3)

        self.comparison_panel = tk.Frame(self.compare_scroll.body, bg=SURFACE, pady=6)
        self.comparison_panel.pack(fill='x')
        row = tk.Frame(self.comparison_panel, bg=SURFACE)
        row.pack(fill='x')
        row.columnconfigure(0, weight=1)
        label(row, 'B · Compare with', muted=True).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0,6))
        self.compare_combo = ttk.Combobox(row, textvariable=self.comparison_run, state='readonly', width=20)
        self.compare_combo.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0,12))
        self.compare_combo.bind('<<ComboboxSelected>>', lambda e: self.start_comparison())
        self.mode_combo = ttk.Combobox(row, textvariable=self.compare_mode, values=('A only', 'B only', 'Overlay'), state='readonly', width=11)
        label(row, 'Show', muted=True).grid(row=2,column=0,sticky='w')
        self.mode_combo.grid(row=2, column=1)
        self.mode_combo.bind('<<ComboboxSelected>>', lambda e: self.apply_comparison())
        button(row, 'Switch A / B', self.switch_comparison).grid(row=3, column=0, columnspan=2, sticky='ew', pady=(8, 4))
        settings = tk.Frame(self.comparison_panel, bg=SURFACE)
        settings.pack(fill='x', pady=6)
        for text, variable in [('Register B to A', self.register_runs), ('Color B by distance to A', self.distance_colors)]:
            tk.Checkbutton(settings, text=text, variable=variable, command=self.comparison_settings_changed,
                bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE).pack(anchor='w', pady=4)
        self.compare_info = label(self.comparison_panel, 'A · mint / B · coral. Registration uses matching image names; distances use scene units.', muted=True, anchor='w', justify='left')
        self.compare_info.pack(fill='x')
        self.comparison_panel.bind('<Configure>', lambda e: self.compare_info.configure(wraplength=max(200, e.width-30)))

        self.navigation_bar = tk.Frame(self, bg=SURFACE, padx=12)
        self.navigation_bar.pack(fill='x')
        nav = self.navigation_bar
        profile = ttk.Combobox(nav, textvariable=self.inspection_profile, values=('Overview','Geometry','Cameras'), state='readonly', width=10)
        profile.pack(side='left', padx=(0, 8))
        profile.bind('<<ComboboxSelected>>', lambda e:self.apply_profile())
        for title, command in [('Fit', self.reset_view), ('Front', lambda:self.set_preset_view(0, 0)),
            ('Top', lambda:self.set_preset_view(0, 89.9)), ('Right', lambda:self.set_preset_view(90, 0)),
            ('−', lambda:self._zoom(1.15)), ('+', lambda:self._zoom(.85)),
            ('Save view', self.save_view), ('Restore view', self.restore_view)]:
            button(nav, title, command).pack(side='left')
        self.navigation_combo = ttk.Combobox(nav, textvariable=self.navigation_mode, values=('Orbit', 'Pan'), state='readonly', width=7)
        self.navigation_combo.pack(side='left')
        wrap_row(nav)

        self.layer_bar = tk.Frame(self, bg=SURFACE, padx=12)
        self.layer_bar.pack(fill='x')
        label(self.layer_bar, 'Point color', muted=True).pack(side='left')
        color = ttk.Combobox(self.layer_bar, textvariable=self.color_mode_var, values=('rgb', 'depth', 'elevation', 'white'), state='readonly', width=10)
        color.pack(side='left'); color.bind('<<ComboboxSelected>>', lambda e:self.render())
        friendly_choices(color,self.color_mode_var,{'rgb':'Natural','depth':'Depth','elevation':'Elevation','white':'White'})
        self.color_mode_var.trace_add('write',lambda *_:self.render())
        label(self.layer_bar, 'Size · px', muted=True).pack(side='left')
        size = ttk.Combobox(self.layer_bar, textvariable=self.point_size_var, values=(1,2,3,4), state='readonly', width=3)
        size.pack(side='left'); size.bind('<<ComboboxSelected>>', lambda e:self.render())
        for title, var in [('Points',self.show_points_var), ('Cameras',self.show_cameras_var), ('Trajectory',self.show_trajectory_var)]:
            tk.Checkbutton(self.layer_bar, text=title, variable=var, command=self.custom_layers, bg=SURFACE,
                fg=TEXT, selectcolor=FIELD, activebackground=SURFACE).pack(side='left')
        wrap_row(self.layer_bar)
        self.layer_bar.pack_forget()

        inspect = tk.Frame(self, bg=SURFACE, padx=12, pady=6)
        self.inspector_bar = inspect
        inspect.pack(fill='x')
        inspect.columnconfigure(1, weight=1)
        label(inspect, 'Camera', muted=True).grid(row=0,column=0,padx=(0,8))
        self.camera_combo = ttk.Combobox(inspect, textvariable=self.camera_choice, width=15, state='disabled')
        self.camera_combo.grid(row=0,column=1,sticky='ew',padx=(0,8))
        self.camera_combo.bind('<<ComboboxSelected>>', self._select_named_camera)
        actions = tk.Frame(inspect, bg=SURFACE)
        actions.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        for title, cmd in [('Previous',lambda:self.step_camera(-1)), ('Next',lambda:self.step_camera(1)),
                           ('Focus',self.focus_camera), ('Look through',self.look_through),
                           ('View photo',self.open_selected_camera_window)]:
            button(actions,title,cmd).pack(side='left')
        tk.Checkbutton(actions, text='Auto preview', variable=self.auto_open_photo_var,
                       bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE).pack(side='left')
        wrap_row(actions)
        inspect.pack_forget()
        self.stats_lbl = label(self, 'Choose a completed alignment to inspect.', muted=True, anchor='w')
        self.stats_lbl.pack(side='bottom',fill='x',padx=12,pady=(4,8))
        label(self, 'Drag: orbit · Right / Shift-drag: pan · Scroll: zoom · F: fit · 1 / 3 / 7: views · B: switch A/B · Click camera: inspect', muted=True, anchor='w').pack(side='bottom',fill='x',padx=12,pady=4)
        self.compare_scroll.bind_navigation()

    def _build_canvas(self):
        self.scene_body.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(self.scene_body,bg=BG,highlightthickness=0,bd=0)
        self.canvas.pack(side='left',fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.render())

    def toggle_inspector(self):
        self._inspector_visible=not self._inspector_visible
        self.inspect_toggle.configure(text='Hide inspect' if self._inspector_visible else 'Inspect')
        for panel in (self.layer_bar,self.inspector_bar):
            if self._inspector_visible:
                panel.pack(fill='x',before=self.scene_body)
            else:
                panel.pack_forget()

    def toggle_comparison(self):
        self._compare_visible = not self._compare_visible
        self.compare_toggle.configure(text='Hide compare' if self._compare_visible else 'Compare')
        if self._compare_visible:
            self.compare_scroll.pack(side='right',fill='y',before=self.canvas)
            self.refresh_runs()
            if 'B' in self.layers:
                self.compare_mode.set('Overlay')
                self.apply_comparison()
                self.reset_view()
        else:
            self.compare_scroll.pack_forget()

    def custom_layers(self):
        self.inspection_profile.set('Custom')
        self.render()

    def apply_profile(self):
        profile = self.inspection_profile.get()
        self.show_points_var.set(profile != 'Cameras')
        self.show_cameras_var.set(profile != 'Geometry')
        self.show_trajectory_var.set(profile != 'Geometry')
        self.render()
        self.stats_lbl.configure(text=f'{profile} · viewpoint preserved · Fit frames visible layers')

    def switch_comparison(self):
        if 'B' not in self.layers:
            self.compare_info.configure(text='Choose a second completed run to switch A / B.')
            return
        self.compare_mode.set('B only' if self.compare_mode.get() == 'A only' else 'A only')
        self.apply_comparison()

    def start_comparison(self):
        self.compare_mode.set('Overlay')
        self._fit_comparison = True
        self.load_selected_run()

    def comparison_settings_changed(self):
        if self.distance_colors.get():
            self.compare_mode.set('Overlay')
        self.apply_comparison()

    def refresh_runs(self):
        runs = super().refresh_runs()
        names = runs[1:]
        self.compare_combo.configure(values=names, state='readonly' if len(names)>1 else 'disabled')
        if self.comparison_run.get() not in names:
            self.comparison_run.set(names[1] if len(names)>1 else '')
        selected = self.run_choice_var.get()
        if selected == 'Latest Run (Auto)' and names:
            selected = names[0]
        if self.comparison_run.get() == selected and len(names) > 1:
            self.comparison_run.set(next(n for n in names if n != selected))
        return runs

    def _reload_runs(self):
        self._run_cache.clear()
        self.refresh_runs()
        self.load_selected_run()

    def load_selected_run(self):
        if self._loading:
            self._pending_load = True
            self.stats_lbl.configure(text='Loading… the updated selection will follow.')
            return
        names = self.refresh_runs()[1:]
        directory = self._resolve_results_dir()
        if not names or not directory:
            self.stats_lbl.configure(text='No alignment results found. Complete an alignment first.')
            return
        a = self.run_choice_var.get()
        if a == 'Latest Run (Auto)':
            a = names[0]
        b = self.comparison_run.get()
        self._load_generation += 1
        generation = self._load_generation
        self._loading = True
        self.stats_lbl.configure(text='Loading reconstruction… navigation remains available.')
        paths = {key:os.path.join(directory,name) for key,name in [('A',a),('B',b)] if name and (key=='A' or name!=a)}
        cached = {p:self._run_cache[p] for p in paths.values() if p in self._run_cache}
        def work():
            from fast_viewer import find_run_assets, read_ply_data, load_run_cameras
            try:
                layers={}
                for key,path in paths.items():
                    if path in cached:
                        layers[key]=cached[path]
                        continue
                    asset,*_=find_run_assets(path)
                    points,colors=read_ply_data(asset) if asset else (None,None)
                    cameras,source=load_run_cameras(path)
                    if points is None or not len(points):
                        points,colors=np.empty((0,3),np.float32),np.empty((0,3),np.uint8)
                    mask=np.isfinite(points).all(1)
                    points=points[mask]
                    colors=colors[mask] if colors is not None else np.full((len(points),3),180,np.uint8)
                    count=len(points)
                    indices=np.linspace(0,max(0,count-1),min(150000,count),dtype=int)
                    cameras=[c for c in cameras if np.isfinite(c['pos']).all() and np.isfinite(c['wxyz']).all() and np.linalg.norm(c['wxyz'])>0]
                    layers[key]=dict(points=points[indices],colors=colors[indices],cameras=cameras,count=count,path=path,source=source)
                self._load_results.put((generation,layers,a,None))
            except Exception as error:
                self._load_results.put((generation,None,a,str(error)))
        threading.Thread(target=work,daemon=True).start()

    def _poll_load(self):
        try:
            generation,layers,a,error=self._load_results.get_nowait()
            self._loading=False
            if generation==self._load_generation:
                if error:
                    self.stats_lbl.configure(text=f'Could not load: {error}')
                else:
                    self.layers=layers
                    for data in layers.values():
                        self._run_cache[data['path']]=data
                    while len(self._run_cache)>2:
                        self._run_cache.popitem(last=False)
                    self.active_run_name=a
                    self.active_run_dir=layers['A']['path']
                    self.apply_comparison()
                    if getattr(self, '_fit_comparison', False):
                        self.reset_view(); self._fit_comparison = False
                    if self._first_scene:
                        self.reset_view(); self._first_scene=False
            if self._pending_load:
                self._pending_load=False
                self.load_selected_run()
        except queue.Empty:
            pass
        self._poll_id=self.after(80,self._poll_load)

    def apply_comparison(self):
        if not self.layers:
            return
        try:
            a=self.layers['A']; b=dict(self.layers['B']) if 'B' in self.layers else None
            mode=self.compare_mode.get()
            if mode!='A only' and b is None:
                raise ValueError('Choose a different comparison run and load it first.')
            info='A · mint / B · coral. Raw coordinates; origins and scales may differ.'
            if b is not None:
                pairs=camera_pairs(a['cameras'],b['cameras'])
                left=np.array([a['cameras'][i]['pos'] for i,j in pairs])
                right=np.array([b['cameras'][j]['pos'] for i,j in pairs])
                if self.register_runs.get():
                    scale,rotation,translation=similarity(right,left)
                    b['points']=scale*(b['points']@rotation.T)+translation
                    b['cameras']=transform_cameras(b['cameras'],scale,rotation,translation)
                    right=scale*(right@rotation.T)+translation
                    info=f'Registered B → A · scale {scale:.4g}. '
                if pairs:
                    error=np.linalg.norm(left-right,axis=1)
                    info+=f' {len(pairs)} matched cameras · RMS {np.sqrt(np.mean(error**2)):.4g} scene units. Agreement, not accuracy.'
                else:
                    info+=' No uniquely matched image names.'
            points,colors,cameras=[],[],[]
            for key,data in [('A',a),('B',b)]:
                if data is None or (mode!='Overlay' and not mode.startswith(key)):
                    continue
                tint=(112,219,192) if key=='A' else (255,155,127)
                color=np.tile(tint,(len(data['points']),1)).astype(np.uint8) if mode=='Overlay' else data['colors']
                if key=='B' and self.distance_colors.get():
                    d,n=cloud_distances(a['points'],data['points'])
                    if len(d):
                        limit=max(float(np.percentile(d,95)),1e-9)
                        fraction=np.clip(d/limit,0,1)[:,None]
                        color=((1-fraction)*[95,190,240]+fraction*[255,115,90]).astype(np.uint8)
                        info+=f' Cloud B → A: median {np.median(d):.4g}; blue 0 → coral {limit:.4g}. Nearest samples ({n:,} reference points).'
                points.append(data['points']); colors.append(color)
                for c in data['cameras']:
                    q=np.asarray(c['wxyz']); matrix=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                    cameras.append(dict(c,forward=matrix[:,2],up=-matrix[:,1],right=matrix[:,0],layer=key,color=tint))
            self.points=np.concatenate(points) if points else None
            self.colors=np.concatenate(colors) if colors else None
            self.cameras=cameras
            self.selected_camera=min(self.selected_camera,len(cameras)-1)
            self.camera_combo.configure(values=[f'{c["layer"]} · {i+1:04d} · {c["name"]}' for i,c in enumerate(cameras)],state='readonly' if cameras else 'disabled')
            if cameras:
                self.selected_camera=max(0,self.selected_camera); self.camera_combo.current(self.selected_camera)
            else:
                self.camera_choice.set('')
            self.compare_info.configure(text=info)
            self.stats_lbl.configure(text=f'{mode} · {len(self.points) if self.points is not None else 0:,} preview points · {len(cameras)} cameras · source files unchanged')
            self.render()
        except Exception as error:
            self.compare_info.configure(text=f'{error} Previous scene retained.')

    def reset_view(self):
        points=[] if self.points is None or not self.show_points_var.get() else [self.points]
        if self.cameras and (self.show_cameras_var.get() or self.show_trajectory_var.get()):
            points.append(np.array([c['pos'] for c in self.cameras]))
        self.center,self.scale=scene_bounds(points)
        self.yaw,self.pitch,self.distance=-25.,20.,1.5
        self.pan_x=self.pan_y=0.; self.view_matrix=None
        populated=[p for p in points if len(p)]
        if populated:
            from workspace_3d import make_rotation_matrix
            rotated=((np.concatenate(populated)-self.center)/self.scale) @ make_rotation_matrix(self.yaw,self.pitch).T
            width=max(20,self.canvas.winfo_width());height=max(20,self.canvas.winfo_height())
            focal=min(width,height)*1.05
            required=np.maximum(np.abs(rotated[:,0])*focal/(width*.42),np.abs(rotated[:,1])*focal/(height*.42))-rotated[:,2]
            self.distance=max(.1,float(required.max())+.02)
        self.render()

    def set_preset_view(self,yaw,pitch):
        self.view_matrix=None
        super().set_preset_view(yaw,pitch)

    def save_view(self):
        self._saved_view=(self.center.copy(),self.scale,self.yaw,self.pitch,self.distance,self.pan_x,self.pan_y,
                          None if self.view_matrix is None else self.view_matrix.copy())
        self.stats_lbl.configure(text='View saved. Restore view returns to this position and angle.')

    def restore_view(self):
        if self._saved_view is not None:
            center,self.scale,self.yaw,self.pitch,self.distance,self.pan_x,self.pan_y,matrix=self._saved_view
            self.center=center.copy(); self.view_matrix=None if matrix is None else matrix.copy(); self.render()
        else:
            self.stats_lbl.configure(text='No saved view yet. Choose Save view first.')

    def _select_named_camera(self,event=None):
        self.selected_camera=self.camera_combo.current(); self.render()
        if getattr(self, 'photo_viewer_window', None) is not None and self.photo_viewer_window.winfo_exists():
            self.open_selected_camera_window()

    def step_camera(self,step):
        if self.cameras:
            self.selected_camera=(self.selected_camera+step)%len(self.cameras)
            self.camera_combo.current(self.selected_camera); self.render()
            if getattr(self, 'photo_viewer_window', None) is not None and self.photo_viewer_window.winfo_exists():
                self.open_selected_camera_window()

    def focus_camera(self):
        if 0<=self.selected_camera<len(self.cameras):
            self.center=self.cameras[self.selected_camera]['pos'].copy()
            self.pan_x=self.pan_y=0.; self.distance=.3; self.view_matrix=None; self.render()

    def look_through(self):
        if 0<=self.selected_camera<len(self.cameras):
            camera=self.cameras[self.selected_camera]
            self.center=camera['pos'].copy()
            self.view_matrix=np.stack([camera['right'],-camera['up'],camera['forward']])
            self.pan_x=self.pan_y=0.; self.distance=0.; self.render()
            self.stats_lbl.configure(text=f'Looking through {camera["name"]} · illustrative optics · F returns to scene')

    def open_selected_camera_window(self):
        """Opens or updates separate photo viewer window for the selected camera."""
        if not (0 <= self.selected_camera < len(self.cameras)):
            return
        cam = self.cameras[self.selected_camera]
        from workspace_3d import resolve_camera_photo_path
        run_dir = self.layers.get(cam.get('layer', 'A'), {}).get('path', getattr(self, 'active_run_dir', ''))
        dataset_dir = None
        if hasattr(self, "master") and hasattr(self.master, "dataset_var"):
            dataset_dir = self.master.dataset_var.get()
        img_path = resolve_camera_photo_path(cam.get("name", ""), run_dir, dataset_dir) or \
                   resolve_camera_photo_path(cam.get("file_path", ""), run_dir, dataset_dir)

        if getattr(self, "photo_viewer_window", None) is None or not self.photo_viewer_window.winfo_exists():
            self.photo_viewer_window = CameraPhotoViewer(
                self,
                on_step=self.step_camera,
                on_focus=self.focus_camera,
                on_look=self.look_through
            )
        self.photo_viewer_window.deiconify()
        self.photo_viewer_window.lift()
        self.photo_viewer_window.show_camera(cam, img_path, self.selected_camera, len(self.cameras))

    def _bind_mouse_events(self):
        super()._bind_mouse_events()
        self.canvas.configure(takefocus=True,highlightthickness=1,highlightbackground=BG,highlightcolor=ACCENT)
        self.canvas.bind('<ButtonRelease-1>',self._pick_camera)
        self.canvas.bind('<Shift-B1-Motion>',self._on_pan_drag)
        for key,command in {'f':self.reset_view,'1':lambda:self.set_preset_view(0,0),'3':lambda:self.set_preset_view(90,0),
            '7':lambda:self.set_preset_view(0,89.9),'plus':lambda:self._zoom(.85),'equal':lambda:self._zoom(.85),
            'minus':lambda:self._zoom(1.15),'b':self.switch_comparison,'bracketleft':lambda:self.step_camera(-1),'bracketright':lambda:self.step_camera(1)}.items():
            self.canvas.bind(f'<KeyPress-{key}>',lambda e,cmd=command:(cmd(),'break')[-1])

    def _on_mouse_down(self,event):
        self.canvas.focus_set(); self._press_xy=(event.x,event.y)
        super()._on_mouse_down(event)

    def _on_orbit_drag(self,event):
        if self.navigation_mode.get()=='Pan':
            return self._on_pan_drag(event)
        self.view_matrix=None
        super()._on_orbit_drag(event)

    def _on_pan_drag(self,event):
        focal=max(20,min(self.canvas.winfo_width(),self.canvas.winfo_height()))*1.05
        speed=max(.1,self.distance)/focal
        self.pan_x+=(event.x-self._last_mouse_x)*speed
        self.pan_y+=(event.y-self._last_mouse_y)*speed
        self._last_mouse_x,self._last_mouse_y=event.x,event.y
        self.render()

    def _on_mouse_wheel(self,event):
        if event.delta:
            self._zoom(.9**(event.delta/120))

    def _pick_camera(self,event):
        super()._on_mouse_up(event)
        if not hasattr(self,'_press_xy') or np.linalg.norm(np.array(self._press_xy)-[event.x,event.y])>5:
            return
        matches=[((x-event.x)**2+(y-event.y)**2,i) for x,y,z,i in self._camera_screen]
        if matches and min(matches)[0]<=144:
            if not self._inspector_visible:
                self.toggle_inspector()
            self.selected_camera=min(matches)[1]
            self.camera_combo.current(self.selected_camera)
            self.render()
            if getattr(self, 'auto_open_photo_var', None) and self.auto_open_photo_var.get():
                self.open_selected_camera_window()

    def _cleanup_inspection(self,event):
        if event.widget is self:
            self.after_cancel(self._poll_id)
            if hasattr(self, '_thumb_executor'):
                try:
                    self._thumb_executor.shutdown(wait=False)
                except Exception:
                    pass
