"""Native panorama control-point workspace; all files and processing stay local."""
import copy
import json
import math
import queue
import threading
import subprocess
import sys
import os
import time
import traceback
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from control_points_model import ControlDocument, validate_controls, atomic_json, point_geometry, projected_pixel
from panorama_viewport import PanoramaViewport, PanoramaAsset
from control_points_logging import ControlLog
from manual_camera_geometry import load_alignment, spherical
from manual_refinement_settings import DEFAULTS as REFINE_DEFAULTS, GUIDES as BASE_REFINE_GUIDES,guide as refine_guide, validate as validate_refine_settings
REFINE_GUIDES={k:refine_guide(k) for k in BASE_REFINE_GUIDES}
BG, PANEL, FIELD, BORDER = '#111518', '#1a2024', '#242d32', '#344149'
TEXT, MUTED, ACCENT = '#e7edef', '#a0adb5', '#70dbc0'
TYPES = {'tie': 'Tie point', 'gcp': 'Ground control', 'check': 'Check point'}
HELP = '''MANUAL CONTROL POINTS — 360° & PERSPECTIVE WORKFLOW

1. CHOOSE YOUR STARTING POINT
Before alignment: Open images loads original photos in the projection chosen in Settings. You can create, import, measure and save points without camera poses. Predictions, residuals, 3D and Refine become available after opening an alignment for those same images; prepared measurements are retained.
After alignment: Open alignment accepts native/OpenSfM reconstruction JSON or a complete COLMAP model (choose images.bin/images.txt or cameras.bin/cameras.txt). OpenSfM/360DVO panoramas retain spherical geometry; supported VGGT-Omega and other COLMAP perspective outputs retain original focal lengths, principal points and lens distortion. Intrinsics stay fixed. Work on one component and use images matching its camera dimensions exactly.
Before loading ordinary perspective photos, choose Perspective under Settings > Before-alignment image type. This avoids guessing projection from aspect ratio. Panorama seam wrapping only affects spherical images.

CONTINUE REFINING OR START AGAIN
Use the latest GOOD refined JSON to add/correct controls and refine again. It provides corrected poses and preserved sparse observations, often reducing solve time. Keep the same original images and controls file. The initial result is useful for a fresh comparison or recovery from a bad solve. Compare reports and independent check points; lower fitted residual alone is not proof of accuracy. Repeating identical constraints after convergence does not keep improving accuracy. A point-only VGGT cloud has no automatic multi-view tracks; manual measurements still work, and unobserved cameras stay fixed.

2. CREATE AND MEASURE
Add point (Ctrl+N) creates a named tie point immediately. In Place points mode, click the same physical feature in both images. Use the image selectors or Previous/Next buttons to add more views. Aim for 3–6 well-separated views with translation and clear parallax. Distribute control points throughout the scene.

3. NAVIGATE PRECISELY
Wheel: zoom around the cursor. Middle-drag or Space + left-drag: pan. Fit / F: entire panorama. 1:1 / 1: original pixel size. Drag an existing marker to adjust it. Left/Right browse images in the active pane; Alt+Left/Right also browse. Ctrl+Arrow nudges 0.1 px; Shift+Arrow nudges 10 px. Disable arrow browsing in Settings for plain-arrow 1 px nudging. Text fields retain normal typing keys. Right-click a marker removes its measurement. Ctrl+Z / Ctrl+Y undo and redo.
Ctrl+F: Focus Point in the active image; in a 3D pane, focus its selected object. Measurement rows are tinted mint for Left, blue for Right, and purple for a selected 3D camera. These are view indicators, separate from the angular error colours.

Each Left/Right view has its own Image / Grid / 3D selector. Use Grid beside Image for measuring, two Images for precise comparison, or 3D beside Image to inspect cameras and controls. The two 3D panes share Abu M54 scene data and retain independent viewpoints. Switching modes preserves image zoom and the existing viewer; a 3D pane needs an open alignment. A camera picked in 3D opens its original in the other pane when that pane is in Image mode.
Image grid shows all images on one continuous scrolling page. Local background workers prefetch thumbnails; a warm cached image appears immediately without re-decoding. A neutral image placeholder appears only on a first uncached visit, with no loading banner. Search by filename and filter images with or without a measurement of the selected point. Choose the Left or Right destination, then double-click a thumbnail or Open selected. Embedded grids default to opening the other pane; their own grid stays visible. The separate Image grid window also stays open, retaining its filter, selection and scroll position. Measured markers are yellow; predicted guidance is blue. A Predicted filter shows unmeasured candidate images. Arrow keys select thumbnails; Enter opens the selection.

Wrap panorama seam allows close-up panning across the left/right boundary. Latitude stays bounded. Pixel display uses nearest-pixel rendering for precise inspection. Marker coordinates always refer to original image pixels regardless of display zoom. Both image panes navigate independently; linked zoom synchronizes pixel scale only.

4. REVIEW POINTS AND MEASUREMENTS
Drag the divider above Measurements upward to make the table taller. The form above it scrolls independently. Measurement actions opens left/right views, predictions, uncertainty and removal; double-click still opens a measured image. Dividers resize live, and normal image quality returns after a short lightweight resize preview.
Search or filter the Points list. Yellow rows need at least two measurements. Select a point to see its type, properties, every measured image, pixel coordinates, uncertainty and angular residual. Double-click a measurement to open and focus it in the active image pane, or choose View left/right. Disabled points retain their measurements but are excluded from refinement. Distances attached to disabled points are also inactive and are labelled in the distance list.
Only the angular error numbers are coloured: green at or below 0.10°, yellow above 0.10° through 0.50°, red above 0.50°. Change both limits in Settings; valid limits are saved with the controls. Unknown errors show an uncoloured dash. These colours show consistency of measured image rays, not independent surveyed accuracy; inspect high errors before lowering uncertainty or increasing solver budgets.

5. SET POINT PROPERTIES
Tie point: the same feature in different images; no surveyed coordinates required.
Ground control: surveyed local X/Y/Z in metres. At least three non-collinear ground controls establish the world frame.
Check point: surveyed local coordinates for independent evaluation only; never optimized. Checks still require two or more image measurements to triangulate.

Apply properties validates the name, type, coordinates, enabled state and survey uncertainty. Smaller uncertainty gives a stronger constraint: enter realistic precision. Image uncertainty is in original pixels, survey uncertainty is in metres. Each measurement can have its own image uncertainty.

6. ADD A KNOWN DISTANCE (OPTIONAL)
In Distances, select two different enabled, measured tie/ground-control points. Enter the independently measured distance and uncertainty in metres, then Add distance. A distance establishes scale. Check points cannot be distance endpoints because they are not optimized.

7. IMPORT REALITYSCAN CSV (OPTIONAL)
Import CSV opens a preview with column mapping before any edits are applied. Image measurements use image path, point name, X pixel, Y pixel, optionally image uncertainty. CSV, semicolon, tab and quoted space-separated files are supported. Basenames must uniquely match the original panoramas, with the same projection and pixel dimensions. Never transfer perspective, cropped or resized image pixels onto a panorama. Use Point names only for such exports and re-mark their features on the panoramas.
Local survey XYZ uses point name, X, Y, Z and optionally survey uncertainty. Choose Ground control or Check point and the source units. Geographic latitude/longitude requires conversion outside this tool. Keep existing values is the default; Replace existing values is explicit. One import is one undo step; review unmatched rows and errors before applying.

8. SAVE AND CHOOSE AN ALIGNMENT ACTION
Ctrl+S saves portable control_points.json. Set the automatic-landmark and solver-evaluation budgets in Settings, then Refine alignment. Manual optimization points are always retained. Larger budgets need more CPU time and RAM. Refinement runs in the background with edits locked for a consistent snapshot. It writes a new reconstruction, the exact controls used and a report. Source photos and reconstruction remain unchanged.
Refine alignment operates on the alignment currently open in this editor. To run the full pipeline, Use with main Align saves and attaches controls and refinement settings to the main window. Keep Apply manual controls after initial alignment checked, choose a matching spherical or perspective engine, and press main Align. Initial registration runs first, followed by calibrated control refinement and exports using the refined poses. The CLI validates selected originals and captures an immutable control snapshot before inference. Controls are not initialization seeds and do not merge disconnected components. Attaching controls does not start alignment automatically.
Project → Images & masks disables images for subsequent Align/Refine while retaining originals and saved measurements. Mask files must match the original dimensions and names (image.jpg.png, image.mask.png or image.png); missing masks keep the image. Black ignores and white keeps, with explicit inversion. Apply then Save controls. Initial masks are supported by OpenSfM, native COLMAP and the app-owned 360DVO adapter; other initial engines reject masks explicitly. Refine masks filters measured rays with fixed original calibration. Too few retained observations require more views or disabling that control. Existing source alignments are never overwritten.

PROCESSING HARDWARE AND EXPORTS
Auto uses CUDA for larger problems when available; CUDA forces GPU processing; CPU uses vectorized analytic derivatives. Default solver budget is 100. A separate worker keeps the editor responsive. Progress shows loading, optimization, retriangulation and export. Cancel refinement stops the worker and preserves input files. An iteration limit is not convergence.
Every successful refinement writes <name>.points.ply, <name>.cameras.csv and <name>.exports.json beside the JSON. The PLY contains refined sparse landmarks. The CSV stores exact native-frame centres, camera-to-world quaternions, world-to-camera rotations/translations and spherical dimensions; it is not a RealityScan measurement CSV. Scale remains arbitrary unless established by metric constraints. To export an older result, open its refined JSON, then Results > Export cloud & camera CSV.

9. INSPECT RESULTS AND 3D
Results reports angular RMS before/after, solver convergence, independent check errors and output paths. Reaching the evaluation limit is not convergence. Reduced fitted residual is not proof of absolute accuracy. Open result inspects the new alignment with the same controls; it does not automatically replace or export the application's training package.
3D view opens Abu 3D Viewer M54, the shared GPU viewer also used by the main app and standalone Abu_M54.exe. Drag to orbit, right/middle drag to pan, wheel to zoom, F to fit. Click a control, camera or cloud sample; Orbit selected, O or double-click sets its orbit pivot. M54 tools includes world up, ground grid/plane/elevation, shading, standard views, bookmarks and reversible move/rotate/uniform-scale previews. Transform numbers update live: type or drag a number left/right; Shift is finer and Ctrl is coarser. One drag is one Undo. Browser cloud picking is a temporary single-click mode, then orbit/pan resumes. Export transformed copy reads the full source and saves a new model/cloud/camera CSV; separate surveyed control coordinates are not transformed. The Workspace tab returns to the images. Compare reconstructions opens a collapsible side panel. A is the editable alignment; Load B or Run folder loads a read-only comparison in a separate process. Show A only, B only or Overlay; Switch A/B preserves the viewpoint. Camera offsets and per-camera differences show agreement. Fit B to A optionally fits matching non-collinear cameras; Allow scale selects similarity versus rigid fitting. Source files and controls remain unchanged. B distance to A colours directed distances to sampled A points; density affects the result. These metrics are not independent accuracy. Save comparison report preserves the chosen frame transform and differences.
Browser view opens the optional local WebGL viewer for A. Public sharing is disabled. Raw survey XYZ is not guessed into an unregistered frame. The display samples at most 50,000 cloud points; optimization data remains complete. This inspects the open alignment, not a live preview of an unfinished solve.

ACCURACY SETTINGS
Start with soft_l1, robust transition 2, automatic track sigma 1 px, 100 solver evaluations and 10000 automatic landmarks. Mark crisp stable corners in 3–6 translated views and distribute points around the scene, including depth/height variation. Use realistic per-observation, survey and distance uncertainty. Increasing weight does not create accuracy.
Try 3–4 minimum automatic views and 0.5–1 degree minimum parallax only if enough good tracks remain. Increase the landmark/solver budget when coverage and convergence reports justify it. Tightening objective/step/gradient tolerances below 1e-8/1e-8/1e-7 mostly increases work; accept it only when independent checks improve. Robust loss handles outliers but cannot fix a wrong correspondence. Hover each setting for its tradeoffs.
Calibration refinement is deliberately not guessed: perspective focal/principal-point/distortion remain the imported calibration. Recalibrate a wrong lens model in the originating engine before this pose/landmark refinement. Check points remain evaluation-only. No settings guarantee higher absolute accuracy.

PREDICTIONS ARE GUIDANCE, NOT MEASUREMENTS
A dashed blue marker predicts the selected point using at least two consistent measurements and the current alignment. It is never saved automatically. Find the actual feature and click it yourself. Predictions are suppressed when triangulation is unreliable or residuals are high. Avoid repeated patterns, reflections, moving objects, sky and stitching artefacts.

PREDICTED IMAGE LIST
Properties > Measurement actions > Predicted images lists unmeasured registered images and predicted pixels, ranked near the views supporting triangulation. Select an image and Open & focus prediction, or show candidates in the grid. Occlusion and actual visibility are unknown. After measuring an image it leaves the unmeasured list. Predictions never become saved measurements automatically.

TROUBLESHOOTING
Results > Open troubleshooting log opens the local session JSONL file. It records point/measurement/property/distance edits, image and CSV selections, settings, saved paths, solver progress, device/timing/memory, outputs, cancellation and full error traces. Image contents are not logged. Logs normally live in %LOCALAPPDATA%/Abu_Align/logs, with a temporary-directory fallback if needed.

PRIVACY AND LIMITS
All images and controls remain on this PC. No uploads or cloud services are used. Image navigation uses CPU memory; joint refinement uses CUDA sparse linear solves or the analytic CPU path. Native panorama measurements require no cubemaps. Geographic latitude/longitude conversion, disconnected-component merging and automatic matching are not supplied by this editor.

SHORTCUTS
Ctrl+O  Open alignment     Ctrl+N  Add point     Ctrl+S  Save controls
Ctrl+Z  Undo               Ctrl+Y  Redo
F       Fit image         1       Original pixel scale
Space + drag / middle drag  Pan
Left/Right  Browse images    Alt+Left/Right  Browse images
Ctrl+Arrows 0.1 px nudge     Shift+Arrows 10 px nudge
Plain arrows 1 px nudge when arrow browsing is disabled
'''
HELP+='\nPARAMETER ACCURACY REFERENCE\n\n'+'\n\n'.join(k+'\n'+v for k,v in REFINE_GUIDES.items())
class HoverGuide:
    def __init__(self, widget, title, text):
        self.widget, self.title, self.text = widget, title, text
        self.job = None; self.popup = None
        widget.bind('<Enter>', self.schedule, add='+')
        for event in ['<Leave>', '<ButtonPress>', '<Destroy>']: widget.bind(event, self.hide, add='+')
    def schedule(self, event=None):
        self.hide(); self.job = self.widget.after(550, self.show)
    def show(self):
        self.job = None
        if not self.widget.winfo_exists(): return
        self.popup = tk.Toplevel(self.widget); self.popup.overrideredirect(True); self.popup.configure(bg=BORDER)
        body = tk.Frame(self.popup, bg=PANEL, padx=14, pady=12); body.pack(padx=1, pady=1)
        tk.Label(body, text=self.title, bg=PANEL, fg=ACCENT, font=('Segoe UI', 10, 'bold'), anchor='w').pack(fill='x')
        tk.Label(body, text=self.text, bg=PANEL, fg=TEXT, wraplength=300, justify='left', font=('Segoe UI', 10)).pack(fill='x', pady=(6, 0))
        self.popup.update_idletasks()
        x = min(self.widget.winfo_pointerx()+16, self.widget.winfo_screenwidth()-self.popup.winfo_reqwidth()-12)
        y = min(self.widget.winfo_pointery()+20, self.widget.winfo_screenheight()-self.popup.winfo_reqheight()-12)
        self.popup.geometry(f'+{max(0,x)}+{max(0,y)}')
    def hide(self, event=None):
        if self.job:
            try: self.widget.after_cancel(self.job)
            except tk.TclError: pass
            self.job = None
        if self.popup:
            try: self.popup.destroy()
            except tk.TclError: pass
            self.popup = None
class ControlPointEditor:
    def __init__(self, parent, on_attach=None):
        self.audit=ControlLog();self.logged_signature=None;self.worker=None;self.cancel_requested=False;self.prediction_browser=None
        self.native_3d=None;self.scene_future=None;self.scene_generation=0;self.scene_key=None;self.browser_camera_selected=None
        self.window = tk.Toplevel(parent); self.window.title('Manual control points · Abu Align & Train')
        self.window.geometry('1580x940'); self.window.minsize(1100, 720); self.window.configure(bg=BG)
        self.document = ControlDocument(); self.active = tk.StringVar(master=self.window)
        self.reconstruction = self.image_dir = self.component = self.controls_path = None
        self.cameras = {}; self.image_names = []; self.closed = self.busy = False; self.last_result = None
        self.image_sizes={};self.on_attach=on_attach;self.grid=None;self.viewer=None;self.viewer_future=None;self.thumbnail_store=None;self.scene_cache=None;self.image_policy_panel=None;self.mask_set=None
        self.viewer_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='Abu control 3D');self.viewer_generation=0;self.viewer_key=None
        self.result_queue = queue.Queue(); self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='Abu panorama')
        self.focused = 0; self.guides = []; self.edit_buttons = []; self.pending = self.populating = False
        self.status = tk.StringVar(master=self.window, value='Open an alignment, add a point, then mark the same feature in multiple images.')
        self.context = tk.StringVar(master=self.window, value='No alignment open')
        self.mode = tk.StringVar(master=self.window, value='Place points')
        self.settings = {k: tk.Variable(master=self.window, value=v) for k, v in {
            **REFINE_DEFAULTS,'image_projection':'360 equirectangular','max_points': 10000, 'iterations': 100, 'backend':'auto', 'sigma_px': 1., 'marker_size': 6,
            'wrap': False, 'preserve_zoom': True, 'linked_zoom': False, 'predictions': True, 'pixelated': False, 'other_labels': False, 'browse_arrows':True,'resize_mode':'auto','error_good_deg':.1,'error_warning_deg':.5}.items()}
        self.build(); self.window.protocol('WM_DELETE_WINDOW', self.close)
        from control_layout import LiveLayout
        self.live_layout=LiveLayout(self)
        for split in [self.workspace_split,self.image_split,self.properties_split]:self.live_layout.watch_split(split)
        self.tabs.enable_traversal()
        self.style_dropdowns(self.window)
        for sequence, command in [('<Control-o>', self.open), ('<Control-n>', self.new), ('<Control-s>', self.save), ('<Control-z>', self.undo), ('<Control-y>', self.redo)]:
            self.window.bind(sequence, lambda e, c=command: self.shortcut(e, c))
        self.window.bind('<Left>',lambda e:self.window_arrow(e,-1));self.window.bind('<Right>',lambda e:self.window_arrow(e,1))
        self.window.bind('<Control-f>',self.focus_shortcut)
        self.poll_job = self.window.after(60, self.poll)
    @property
    def controls(self): return self.document.data
    @controls.setter
    def controls(self, value): self.document.data = value
    def selected(self): return self.document.point(self.active.get())
    def label(self, parent, text='', variable=None, muted=False, **kwargs):
        widget = tk.Label(parent, text=text, textvariable=variable, bg=parent.cget('bg'), fg=MUTED if muted else TEXT,
                          font=('Segoe UI', 10), anchor='w', justify='left', **kwargs)
        return widget
    def guide(self, widget, title, text): self.guides.append(HoverGuide(widget, title, text)); return widget
    def style_dropdowns(self, parent):
        if isinstance(parent, ttk.Combobox):
            try:
                popup = self.window.tk.call('ttk::combobox::PopdownWindow', str(parent))
                self.window.tk.call(str(popup)+'.f.l', 'configure', '-background', FIELD, '-foreground', TEXT,
                                    '-selectbackground', '#31574f', '-selectforeground', 'white', '-font', '{Segoe UI} 10')
            except tk.TclError: pass
        for child in parent.winfo_children(): self.style_dropdowns(child)
    def button(self, parent, text, command, tip='', edit=False, primary=False):
        widget = tk.Button(parent, text=text, command=command, bg=ACCENT if primary else FIELD, fg=BG if primary else TEXT,
            activebackground=BORDER, activeforeground=TEXT, disabledforeground='#7a8991', relief='flat', bd=0,
            padx=9, pady=6, font=('Segoe UI', 9), cursor='hand2', takefocus=True)
        if tip: self.guide(widget, text, tip)
        if edit: self.edit_buttons.append(widget)
        return widget
    def entry(self, parent, variable, width=12):
        return tk.Entry(parent, textvariable=variable, width=width, bg=FIELD, fg=TEXT, insertbackground=TEXT, relief='flat',
                        highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT, font=('Segoe UI', 10),
                        disabledbackground=FIELD, disabledforeground=MUTED)
    def tree(self, parent, columns, widths, height=6):
        frame = tk.Frame(parent, bg=PANEL); frame.pack(fill='both', expand=True)
        tree = ttk.Treeview(frame, columns=columns, show='headings', style='CP.Treeview', selectmode='browse', height=height)
        for name, width in zip(columns, widths): tree.heading(name, text=name); tree.column(name, width=width, minwidth=35)
        scroll = ttk.Scrollbar(frame, command=tree.yview); tree.configure(yscrollcommand=scroll.set)
        horizontal = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
        tree.configure(xscrollcommand=horizontal.set); horizontal.pack(side='bottom', fill='x')
        scroll.pack(side='right', fill='y'); tree.pack(fill='both', expand=True); return tree
    def scroll_page(self, parent):
        outer = tk.Frame(parent, bg=PANEL)
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, width=260)
        scrollbar = ttk.Scrollbar(outer, command=canvas.yview); canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y'); canvas.pack(fill='both', expand=True)
        inner = tk.Frame(canvas, bg=PANEL); item = canvas.create_window(0, 0, anchor='nw', window=inner)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(item, width=e.width))
        outer.body, outer.scroll_canvas = inner, canvas
        return outer, inner
    def bind_page_scroll(self, outer):
        def wheel(event):
            outer.scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units'); return 'break'
        def bind(widget):
            if not isinstance(widget, (ttk.Treeview, tk.Text, tk.Canvas)):
                widget.bind('<MouseWheel>', wheel, add='+')
            for child in widget.winfo_children(): bind(child)
        bind(outer.body)
    def field(self, parent, title, variable, tip, values=None):
        self.label(parent, title, muted=True).pack(fill='x', pady=(9, 4))
        widget = ttk.Combobox(parent, textvariable=variable, values=values, style='CP.TCombobox', state='readonly') if values is not None else self.entry(parent, variable)
        widget.pack(fill='x'); self.guide(widget, title, tip); return widget
    def build(self):
        style = ttk.Style(self.window)
        # Scoped clam elements keep the parent app's theme unchanged while
        # removing Windows-native white fields from this dark workspace.
        elements = [('CP.Treeview.field', 'Treeview.field'), ('CP.Heading.cell', 'Treeheading.cell'),
                    ('CP.Heading.border', 'Treeheading.border'), ('CP.Notebook.tab', 'Notebook.tab'),
                    ('CP.Notebook.client', 'Notebook.client'), ('CP.Combo.field', 'Combobox.field'),
                    ('CP.Combo.arrow', 'Combobox.downarrow')]
        for target, source in elements:
            if target not in style.element_names(): style.element_create(target, 'from', 'clam', source)
        style.layout('CP.Treeview', [('CP.Treeview.field', {'sticky': 'nswe', 'children': [('Treeview.padding', {'sticky': 'nswe', 'children': [('Treeview.treearea', {'sticky': 'nswe'})]})]})])
        style.layout('CP.Treeview.Heading', [('CP.Heading.cell', {'sticky': 'nswe'}), ('CP.Heading.border', {'sticky': 'nswe', 'children': [('Treeheading.padding', {'sticky': 'nswe', 'children': [('Treeheading.text', {'sticky': 'we'})]})]})])
        style.layout('CP.TNotebook', [('CP.Notebook.client', {'sticky': 'nswe'})])
        style.layout('CP.TNotebook.Tab', [('CP.Notebook.tab', {'sticky': 'nswe', 'children': [('Notebook.padding', {'sticky': 'nswe', 'children': [('Notebook.label', {'sticky': 'nswe'})]})]})])
        style.layout('CP.TCombobox', [('CP.Combo.arrow', {'side': 'right', 'sticky': 'ns'}), ('CP.Combo.field', {'sticky': 'nswe', 'children': [('Combobox.padding', {'sticky': 'nswe', 'children': [('Combobox.textarea', {'sticky': 'nswe'})]})]})])
        style.configure('CP.TCombobox', background=FIELD, fieldbackground=FIELD, foreground=TEXT, arrowcolor=TEXT, bordercolor=BORDER, padding=4)
        style.map('CP.TCombobox', fieldbackground=[('readonly', FIELD)], foreground=[('readonly', TEXT)])
        style.configure('CP.TNotebook', background=PANEL)
        style.configure('CP.TNotebook.Tab', background=FIELD, foreground=MUTED, padding=(12, 7), font=('Segoe UI', 10))
        style.map('CP.TNotebook.Tab', background=[('selected', PANEL)], foreground=[('selected', ACCENT)])
        style.configure('CP.Treeview.Heading', background=BORDER, foreground=TEXT, font=('Segoe UI', 9, 'bold'))
        style.configure('CP.Treeview', background=FIELD, fieldbackground=FIELD, foreground=TEXT, rowheight=28, font=('Segoe UI', 9), borderwidth=0)
        style.map('CP.Treeview', background=[('selected', '#31574f')], foreground=[('selected', 'white')])
        header = tk.Frame(self.window, bg=BG, padx=18, pady=12); header.pack(fill='x')
        title = self.label(header, 'Manual control points'); title.configure(font=('Segoe UI', 20)); title.pack(side='left')
        self.label(header, 'Native panorama alignment', muted=True).pack(side='left', padx=18)
        self.button(header, 'Help', lambda: self.tabs.select(self.help_page), 'Open the complete workflow and navigation shortcuts.').pack(side='right')
        toolbar = tk.Frame(self.window, bg=BG, padx=18); toolbar.pack(fill='x', pady=(0, 10))
        project=tk.Menubutton(header,text='Project ▾',bg=FIELD,fg=TEXT,relief='flat',padx=10,pady=6,font=('Segoe UI',9))
        menu=tk.Menu(project,tearoff=False,bg=PANEL,fg=TEXT)
        for name,command in [('Open alignment',self.open),('Open panoramas (before alignment)',self.open_images),('Images & masks',self.open_image_policy),('Load controls JSON',self.load),('Import RealityScan CSV',self.import_csv),('Save controls',self.save),('Save controls as…',self.save_as)]:menu.add_command(label=name,command=command)
        menu.add_separator();menu.add_command(label='Reset pane sizes',command=lambda:self.live_layout.reset())
        project.configure(menu=menu);project.pack(side='right',padx=8)
        for text, command, tip in [('Open alignment', self.open, 'Open the latest good refined JSON to continue, or the initial model to restart/compare. Native JSON and supported calibrated COLMAP files are accepted. Use the same original images. Ctrl+O.'),
            ('Open images',self.open_images,'Prepare measurements before initial alignment. Choose the actual 360/Perspective image type in Settings first. No camera poses are invented.'),
            ('Import CSV',self.import_csv,'Import RealityScan image measurements, local XYZ coordinates, or point names with a preview and explicit column mapping.'),
            ('Save controls', self.save, 'Save portable control_points.json with measurements and uncertainties. Ctrl+S.'),
            ('Undo', self.undo, 'Undo the last point, measurement, property or distance edit. Ctrl+Z.'),
            ('Redo', self.redo, 'Restore an undone edit. Ctrl+Y.')]: self.button(toolbar, text, command, tip, edit=True).pack(side='left', padx=(0, 6))
        actions=tk.Frame(self.window,bg=BG,padx=18);actions.pack(fill='x',pady=(0,8))
        self.refine_button = self.button(actions, 'Refine alignment', self.refine, 'Jointly optimize cameras and landmarks into a new alignment and report. Source files stay unchanged.', primary=True)
        self.refine_button.pack(side='right')
        self.cancel_button=self.button(actions,'Cancel refinement',self.cancel_refinement,'Stop the separate solver process. Source files are preserved.');self.cancel_button.configure(state='disabled')
        if self.on_attach:
            self.button(actions,'Use with main Align',self.attach_to_main,'Save and attach these controls to the main application. Main Align runs initial registration, then applies control refinement.',edit=True).pack(side='right',padx=6)
        context = self.label(actions, variable=self.context, muted=True); context.pack(side='left',fill='x',expand=True)
        self.guide(context, 'Alignment input', 'Open an alignment to see its full source path.'); self.context_guide = self.guides[-1]
        self.tabs = ttk.Notebook(self.window, style='CP.TNotebook'); self.tabs.pack(fill='both', expand=True, padx=12)
        self.workspace = tk.Frame(self.tabs, bg=BG); self.help_page = tk.Frame(self.tabs, bg=PANEL)
        self.tabs.add(self.workspace, text='Workspace'); self.tabs.add(self.help_page, text='Help & shortcuts')
        from control_points_help import HelpExplorer
        self.help_view=HelpExplorer(self,self.help_page,HELP);self.help_view.pack(fill='both',expand=True)
        self.help_text=self.help_view.reference
        split = tk.PanedWindow(self.workspace, orient='horizontal', bg=BG, sashwidth=7, bd=0, opaqueresize=True); split.pack(fill='both', expand=True)
        self.workspace_split = split
        left = tk.Frame(split, bg=PANEL, padx=10, pady=10); center = tk.Frame(split, bg=BG); right = tk.Frame(split, bg=PANEL, padx=10, pady=10)
        # Panes own their allocated rectangles. Requested child widths (notably
        # stretched Treeview columns) must not push the split's geometry back.
        for frame in [left,center,right]:frame.pack_propagate(False)
        split.add(left, minsize=210, width=260, stretch='never'); split.add(center, minsize=400, stretch='always'); split.add(right, minsize=250, width=300, stretch='never')
        self.build_points(left); self.build_views(center); self.build_inspector(right)
        self.scene_page=tk.Frame(self.tabs,bg=BG);self.tabs.add(self.scene_page,text='3D navigator')
        footer = tk.Frame(self.window, bg=BG, padx=18, pady=10); footer.pack(side='bottom', fill='x', before=self.tabs)
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=100)
        self.label(footer, variable=self.status, muted=True, wraplength=1200).pack(side='left', fill='x', expand=True)
    def build_points(self, parent):
        tabs = ttk.Notebook(parent, style='CP.TNotebook'); tabs.pack(fill='both', expand=True)
        points, distances = tk.Frame(tabs, bg=PANEL), tk.Frame(tabs, bg=PANEL)
        tabs.add(points, text='Points'); tabs.add(distances, text='Distances')
        bar = tk.Frame(points, bg=PANEL); bar.pack(fill='x', pady=8)
        self.button(bar, '+ Add point', self.new, 'Create a named tie point immediately, then mark it across images. Ctrl+N.', edit=True).pack(side='left')
        self.button(bar, 'Delete', self.delete, 'Remove the point and its distances. Undo restores them.', edit=True).pack(side='right')
        self.label(points, 'Search points', muted=True).pack(fill='x', pady=(0, 4))
        self.search = tk.StringVar(master=self.window); search = self.entry(points, self.search); search.pack(fill='x', pady=(0, 6))
        self.guide(search, 'Find points', 'Type a point name to filter the list. Clear it to show all points.')
        self.search.trace_add('write', lambda *_: self.refresh_list())
        self.point_filter = tk.StringVar(master=self.window, value='All points')
        choice = ttk.Combobox(points, textvariable=self.point_filter, style='CP.TCombobox', state='readonly', values=['All points', 'Needs observations', 'Disabled', 'Ground control', 'Check points'])
        choice.pack(fill='x', pady=(0, 8)); choice.bind('<<ComboboxSelected>>', lambda e: self.refresh_list())
        self.guide(choice, 'Point filter', 'Find incomplete points, disabled constraints, ground controls or independent checks.')
        self.point_tree = self.tree(points, ['Name', 'Type', 'Views'], [80, 85, 40], 14); self.point_tree.bind('<<TreeviewSelect>>', self.tree_selection)
        for tag, color in [('disabled', '#839098'), ('incomplete', '#ffce73'), ('ready', TEXT)]: self.point_tree.tag_configure(tag, foreground=color)
        self.point_summary = tk.StringVar(master=self.window, value='0 points · 0 measurements')
        self.label(points, variable=self.point_summary, muted=True, wraplength=235).pack(side='bottom', fill='x', pady=8, before=self.point_tree.master)
        self.guide(self.point_tree, 'Point list', 'Select a point to review its measurements and properties. Yellow rows need at least two views; grey rows are disabled.')
        self.distance_a = tk.StringVar(master=self.window); self.distance_b = tk.StringVar(master=self.window)
        self.distance_m = tk.StringVar(master=self.window, value='1'); self.distance_sigma = tk.StringVar(master=self.window, value='.01')
        self.distance_choices = [self.field(distances, title, var, 'Choose an enabled tie/ground-control point with at least two measurements. Check points are excluded.', [])
                                 for title, var in [('First point', self.distance_a), ('Second point', self.distance_b)]]
        self.field(distances, 'Distance (m)', self.distance_m, 'Enter the independently measured distance between these landmarks in metres.')
        self.field(distances, 'Uncertainty (m)', self.distance_sigma, 'Expected precision of this distance. Smaller values give a stronger constraint; use realistic uncertainty.')
        self.button(distances, 'Add distance', self.distance, 'Add this independently measured distance to the refinement.', edit=True).pack(fill='x', pady=12)
        self.distance_tree = self.tree(distances, ['Points', 'Metres'], [150, 65])
        self.button(distances, 'Remove selected distance', self.remove_distance, 'Remove an incorrect distance. Undo restores it.', edit=True).pack(side='bottom', fill='x', pady=10, before=self.distance_tree.master)
    def build_views(self, parent):
        tools = tk.Frame(parent, bg=BG, padx=8, pady=5); tools.pack(fill='x')
        for i in range(3):tools.columnconfigure(i,weight=1,uniform='actions')
        modes=tk.Frame(tools,bg=BG);modes.grid(row=0,column=0,columnspan=3,sticky='w')
        for title in ['Place points', 'Pan']:
            radio = tk.Radiobutton(modes, text=title, variable=self.mode, value=title, command=self.update_view_settings,
                bg=BG, fg=TEXT, selectcolor=FIELD, activebackground=BG, activeforeground=TEXT, font=('Segoe UI', 9))
            radio.pack(side='left'); self.guide(radio, title, 'Place points: click a measurement or drag a marker. Pan: left-drag moves the image without placing points.')
        for i,(title,command,tip) in enumerate([('Fit both',self.fit_views,'Fit the current content of both views.'),('Image grid',self.open_grid,'Open the independent image browser. Each view also offers its own Grid mode.'),('3D view',self.open_3d,'Open the full M54 navigator tab. Each view can also show 3D independently.')]):self.button(tools,title,command,tip).grid(row=1,column=i,sticky='ew',padx=2,pady=3)
        frames = tk.PanedWindow(parent, orient='horizontal', bg=BG, sashwidth=6, bd=0, opaqueresize=True); frames.pack(fill='both', expand=True)
        self.image_split = frames
        self.panes = []
        for index, title in enumerate(['Left', 'Right']):
            frame = tk.Frame(frames, bg=PANEL);frame.pack_propagate(False);frames.add(frame, minsize=190, stretch='always')
            header=tk.Frame(frame,bg=PANEL,padx=6,pady=5);header.pack(fill='x');header.columnconfigure(1,weight=1)
            self.label(header,title).grid(row=0,column=0,sticky='w',padx=(0,6))
            pane = {'name': tk.StringVar(master=self.window), 'generation': 0, 'future': None, 'loading': False, 'loaded_name': None,'mode':tk.StringVar(self.window,'Image'),'frame':frame,'grid':None,'scene':None}
            self.panes.append(pane)
            selector=ttk.Combobox(header,textvariable=pane['mode'],values=['Image','Grid','3D'],state='readonly',style='CP.TCombobox',width=7);selector.grid(row=0,column=1,sticky='ew');selector.bind('<<ComboboxSelected>>',lambda e,p=pane:self.set_pane_mode(p,p['mode'].get()))
            self.guide(selector,'View content','Each view independently shows an original image, a continuous image grid, or the shared Abu M54 GPU 3D navigator. Switching retains the image zoom, grid scroll and 3D camera. Grid double-click opens in its chosen destination, defaulting to the other view.')
            container=tk.Frame(frame,bg=PANEL);container.pack(fill='both',expand=True);container.pack_propagate(False)
            pane['pages']={mode:tk.Frame(container,bg=PANEL) for mode in ['Image','Grid','3D']};image_page=pane['pages']['Image'];image_page.pack(fill='both',expand=True)
            bar = tk.Frame(image_page, bg=PANEL, padx=6); bar.pack(fill='x');bar.columnconfigure(1,weight=1)
            self.button(bar, '‹', lambda p=pane: self.navigate(p, -1), 'Previous image in sequence.').grid(row=0,column=0)
            choice = ttk.Combobox(bar, textvariable=pane['name'], style='CP.TCombobox', state='normal',width=1);choice.grid(row=0,column=1,sticky='ew',padx=4)
            pane['choice'] = choice
            self.button(bar, '›', lambda p=pane: self.navigate(p, 1), 'Next image in sequence.').grid(row=0,column=2)
            choice.bind('<<ComboboxSelected>>', lambda e, p=pane: self.show(p)); choice.bind('<Return>', lambda e, p=pane: self.show(p))
            choice.bind('<KeyRelease>', lambda e, p=pane: self.filter_images(p, e))
            self.guide(choice, 'Choose or find an image', 'Select an image, or type part of its filename to filter the dropdown. Enter opens an exact registered name.')
            nav = tk.Frame(image_page, bg=PANEL, padx=4, pady=4); nav.pack(fill='x')
            for i in range(3):nav.columnconfigure(i,weight=1,uniform='image_actions')
            pane['zoom_text'] = tk.StringVar(master=self.window, value='—');self.label(image_page,variable=pane['zoom_text'],muted=True,padx=8).pack(side='bottom',anchor='e')
            pane['info'] = tk.StringVar(master=self.window, value='No image');self.label(image_page,variable=pane['info'],muted=True,padx=8,pady=4,wraplength=220).pack(side='bottom',fill='x')
            view = PanoramaViewport(image_page, lambda pixel, remove, p=pane: self.measure(p, pixel, remove), self.select,
                                     lambda v, p=pane: self.view_navigation(p), lambda v, i=index: self.set_focus(i))
            pane['view'] = view
            view.on_image_navigation=lambda direction,p=pane:self.navigate(p,direction)
            view.canvas.bind('<Alt-Left>',lambda e,p=pane:self.navigate_key(p,-1))
            view.canvas.bind('<Alt-Right>',lambda e,p=pane:self.navigate_key(p,1))
            view.canvas.bind('<Control-f>',lambda e,i=index:(self.set_focus(i),self.focus_shortcut(e))[-1])
            for i,(label,command,tip) in enumerate([('Fit',view.fit,'Fit the entire original image. F.'),('1:1',view.actual,'Original pixel scale. 1.'),('Focus',lambda p=pane:self.focus_point(p),'Centre the selected measurement or unconfirmed prediction. Ctrl+F.')]):self.button(nav,label,command,tip).grid(row=0,column=i,sticky='ew',padx=1)
    def set_pane_mode(self,pane,mode,load=True):
        if mode not in pane['pages']:return
        pane['mode'].set(mode)
        for page in pane['pages'].values():page.pack_forget()
        page=pane['pages'][mode];page.pack(fill='both',expand=True);pane['view'].suspended=mode!='Image';self.focused=self.panes.index(pane)
        if mode=='Grid':
            if pane['grid'] is None:
                from control_images_grid import ImageGrid
                pane['grid']=ImageGrid(self,page,self.focused)
            else:pane['grid'].refresh()
        elif mode=='3D':
            if pane['scene'] is None:
                from abu_m54_native import M54NativeViewer
                from control_layout import PaneSceneHost
                pane['scene']=M54NativeViewer(PaneSceneHost(self,self.focused),page,compact=True);pane['scene'].pack(fill='both',expand=True)
                pane['scene'].surface.bind('<Control-f>',lambda e,v=pane['scene']:(v.focus_selected(),'break')[-1])
                actions=tk.Frame(page,bg=PANEL);actions.pack(side='bottom',fill='x',before=pane['scene'])
                for i,(text,command) in enumerate([('Fit',pane['scene'].fit),('Tools',pane['scene'].open_tools),('Compare',pane['scene'].toggle_comparison)]):
                    actions.columnconfigure(i,weight=1,uniform='scene_actions');self.button(actions,text,command).grid(row=0,column=i,sticky='ew',padx=1)
            self.request_scene();pane['scene'].request()
            if not self.component:self.status.set('This 3D view needs an alignment. Open an alignment to see camera poses and the cloud.')
        else:
            if load and pane['name'].get() in self.image_sizes and pane['loaded_name']!=pane['name'].get():self.show(pane)
            pane['view'].request_draw()
        self.audit.event('pane_content',pane=self.focused,mode=mode)
        self.highlight_images()
    def fit_views(self):
        for pane in self.panes:
            if pane['mode'].get()=='Image':pane['view'].fit()
            elif pane['mode'].get()=='3D' and pane['scene']:pane['scene'].fit()
            elif pane['grid']:pane['grid'].canvas.yview_moveto(0);pane['grid'].arrange()
    def native_views(self):return ([self.native_3d] if self.native_3d else [])+[p['scene'] for p in self.panes if p['scene']]
    def build_inspector(self, parent):
        self.right_tabs = ttk.Notebook(parent, style='CP.TNotebook'); self.right_tabs.pack(fill='both', expand=True)
        property_page = tk.Frame(self.right_tabs, bg=PANEL)
        self.properties_split = tk.PanedWindow(property_page, orient='vertical', bg=BORDER,
                                               sashwidth=8, sashrelief='raised', bd=0, opaqueresize=True)
        self.properties_split.pack(fill='both', expand=True)
        form_page, properties = self.scroll_page(self.properties_split)
        measurements = tk.Frame(self.properties_split, bg=PANEL, padx=4, pady=4)
        self.properties_split.add(form_page, minsize=60, height=180, stretch='never')
        self.properties_split.add(measurements, minsize=100, stretch='always')
        self.label(measurements, 'Measurements · ↕ resize above', muted=True, wraplength=270).pack(fill='x', pady=4)
        setting_page, settings = self.scroll_page(self.right_tabs)
        self.results_page = tk.Frame(self.right_tabs, bg=PANEL)
        for frame, title in [(property_page, 'Properties'), (setting_page, 'Settings'), (self.results_page, 'Results')]: self.right_tabs.add(frame, text=title)
        self.property_vars = {key: tk.StringVar(master=self.window, value=value) for key, value in {'name': '', 'type': 'Tie point', 'x': '', 'y': '', 'z': '', 'sigma_m': '.01'}.items()}
        self.enabled = tk.BooleanVar(master=self.window, value=True)
        self.field(properties, 'Point name', self.property_vars['name'], 'Unique point name. Renaming automatically updates its distance endpoints.')
        self.field(properties, 'Point type', self.property_vars['type'], 'Tie connects images; ground control constrains local XYZ; check evaluates independently.', list(TYPES.values()))
        enabled = tk.Checkbutton(properties, text='Enabled in refinement', variable=self.enabled, bg=PANEL, fg=TEXT, selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT)
        enabled.pack(anchor='w', pady=8); self.guide(enabled, 'Enable point', 'Disabled points retain their measurements but do not constrain refinement. Enabled check points are still evaluation only.')
        self.label(properties, 'Local coordinates (metres)', muted=True).pack(fill='x')
        xyz = tk.Frame(properties, bg=PANEL); xyz.pack(fill='x', pady=4); self.coordinate_entries = []
        for column, key in enumerate(['x', 'y', 'z']):
            cell = tk.Frame(xyz, bg=PANEL); cell.grid(row=0, column=column, sticky='ew', padx=(0, 4)); xyz.columnconfigure(column, weight=1)
            self.label(cell, key.upper(), muted=True).pack(fill='x'); entry = self.entry(cell, self.property_vars[key], 6); entry.pack(fill='x'); self.coordinate_entries.append(entry)
            self.guide(entry, 'Local '+key.upper(), 'Surveyed local coordinate in metres. Ground/check points require all three coordinates. Geographic latitude/longitude is not accepted.')
        self.field(properties, 'Survey uncertainty (m)', self.property_vars['sigma_m'], REFINE_GUIDES['survey'])
        self.button(properties, 'Apply properties', self.apply_properties, 'Validate and apply name, type, coordinates, enabled state and uncertainty.', edit=True).pack(fill='x', pady=8)
        self.geometry_status = tk.StringVar(master=self.window, value='Select a point'); self.label(properties, variable=self.geometry_status, muted=True, wraplength=270).pack(fill='x', pady=(0, 6))
        self.measurement_tree = self.tree(measurements, ['Image', 'X / Y px', 'σ px', '° error'], [105, 85, 40, 50], 10)
        from control_measurement_colours import ErrorCells
        self.error_cells=ErrorCells(self,self.measurement_tree)
        # Keep the explanatory legend in the scrolling form. A multiline legend
        # below the table can consume its entire height on a high-DPI monitor.
        self.error_legend=tk.StringVar(self.window);self.label(properties,variable=self.error_legend,muted=True,wraplength=270).pack(fill='x',pady=4)
        for key in ['error_good_deg','error_warning_deg']:self.settings[key].trace_add('write',self.update_error_colours)
        self.update_error_colours()
        self.guide(self.measurement_tree, 'Image measurements', 'Drag the horizontal divider above this section to make the table taller. Original coordinates and uncertainty are pixels; error is degrees. Double-click to focus. Column widths can be resized.')
        self.measurement_tree.bind('<Double-1>', lambda e: self.open_measurement(self.focused)); self.measurement_tree.bind('<<TreeviewSelect>>', self.measurement_selected)
        for tag,colour in [('shown_left','#264b4b'),('shown_right','#263f59'),('shown_3d','#46395c')]:self.measurement_tree.tag_configure(tag,background=colour)
        self.measurement_tree.tag_configure('excluded',foreground='#839098')
        actions = tk.Frame(measurements, bg=PANEL); actions.pack(side='bottom', fill='x', before=self.measurement_tree.master)
        self.measurement_sigma = tk.StringVar(master=self.window, value='1')
        menu=tk.Menu(actions,tearoff=False,bg=FIELD,fg=TEXT,activebackground=BORDER,activeforeground=TEXT)
        menu.add_command(label='Open in left view',command=lambda:self.open_measurement(0));menu.add_command(label='Open in right view',command=lambda:self.open_measurement(1))
        menu.add_command(label='Enable/disable this image',command=self.toggle_measurement_image);menu.add_command(label='Images & mask layers…',command=self.open_image_policy)
        menu.add_command(label='Predicted images…',command=self.open_predictions);menu.add_separator()
        def uncertainty():
            from tkinter import simpledialog
            value=simpledialog.askfloat('Measurement uncertainty','Expected precision of the selected observation, in original pixels:',initialvalue=float(self.measurement_sigma.get()),minvalue=.000001,parent=self.window)
            if value is not None:self.measurement_sigma.set(str(value));self.apply_measurement_sigma()
        menu.add_command(label='Set uncertainty…',command=uncertainty);menu.add_command(label='Remove observation',command=self.remove_measurement)
        def popup_actions():
            try:menu.tk_popup(action_button.winfo_rootx(),action_button.winfo_rooty()+action_button.winfo_height())
            finally:menu.grab_release()
        action_button=self.button(actions,'Measurement actions ▾',popup_actions,'Open left/right, predicted images, uncertainty and remove observation. Double-click also opens a measurement.')
        action_button.pack(fill='x',pady=5);self.measurement_menu=menu
        for variable in [*self.property_vars.values(), self.enabled]: variable.trace_add('write', self.property_changed)
        self.property_vars['type'].trace_add('write', self.coordinate_state)
        self.build_settings(settings)
        self.bind_page_scroll(form_page); self.bind_page_scroll(setting_page)
        self.results_text = ScrolledText(self.results_page, wrap='word', bg=FIELD, fg=TEXT, font=('Segoe UI', 10), relief='flat', padx=10, pady=10)
        self.results_text.pack(fill='both', expand=True, pady=(8, 0)); self.set_results('No refinement run yet.\n\nSave controls, choose a solver budget in Settings, then Refine alignment.')
        self.button(self.results_page, 'Open result', self.open_result, 'Inspect the refined alignment with the same controls. This does not replace the training package.').pack(fill='x', pady=8)
        self.button(self.results_page,'Export cloud & camera CSV',self.export_current,'Export sparse points and exact native-frame camera poses beside the JSON, including previously saved results.').pack(fill='x',pady=4)
        self.button(self.results_page,'Open troubleshooting log',self.open_log,'Open this session’s local log: edits, settings, progress, errors and outputs.').pack(fill='x',pady=4)
    def build_settings(self, parent):
        self.label(parent, 'Refinement', pady=10).pack(fill='x')
        self.button(parent,'Restore standard defaults',self.restore_refine_defaults,'Restore verified SciPy/OpenSfM numeric defaults and explicitly identified Abu choices. Existing measurement uncertainties are retained.').pack(fill='x',pady=4)
        self.field(parent,'Processing hardware',self.settings['backend'],REFINE_GUIDES['backend'],['auto','cuda','cpu'])
        self.field(parent,'Robust loss',self.settings['loss'],REFINE_GUIDES['loss'],['soft_l1','huber','linear'])
        for title, key, low, high, tip in [
            ('Automatic landmarks', 'max_points', 1, 50000, 'Maximum automatic landmarks in joint refinement. Manual optimization points are kept. More landmarks require more CPU time and RAM.'),
            ('Solver budget', 'iterations', 5, 500, 'Default 100. CUDA: LM iterations; CPU: residual evaluations. Reaching the limit is not convergence.'),
            ('New measurement σ (px)', 'sigma_px', .1, 100, 'Default uncertainty for new clicks, in original image pixels. Existing measurements retain their own uncertainty.'),
            ('Marker radius (px)', 'marker_size', 3, 16, 'Screen-pixel marker radius, independent of image zoom.')]:
            self.label(parent, title, muted=True).pack(fill='x', pady=(8, 4))
            widget = tk.Spinbox(parent, from_=low, to=high, increment=.1 if key == 'sigma_px' else 1, textvariable=self.settings[key],
                bg=FIELD, fg=TEXT, insertbackground=TEXT, buttonbackground=BORDER, relief='flat', font=('Segoe UI', 10), command=self.update_view_settings)
            widget.pack(fill='x'); self.guide(widget, title, REFINE_GUIDES.get(key,tip))
            if key == 'marker_size': widget.bind('<KeyRelease>', lambda e: self.update_view_settings())
        for title,key in [('Robust transition · σ units','loss_scale'),('Automatic track σ · px','auto_sigma_px'),('Minimum automatic views','min_track_views'),('Minimum automatic parallax · °','min_angle_deg'),('Objective tolerance','ftol'),('Step tolerance','xtol'),('Gradient tolerance','gtol')]:
            self.field(parent,title,self.settings[key],REFINE_GUIDES[key])
        self.label(parent,'Imported perspective calibration stays fixed. Refine camera poses and sparse landmarks; use distributed check points to judge accuracy.',muted=True,wraplength=260,pady=8).pack(fill='x')
        self.field(parent,'Before-alignment image type',self.settings['image_projection'],'Choose the actual image projection before Open images. Aspect ratio alone cannot identify perspective versus 360. Opening an alignment uses its calibrated camera models.',['360 equirectangular','Perspective'])
        self.label(parent, 'Image navigation', pady=12).pack(fill='x')
        self.field(parent,'Pane resizing',self.settings['resize_mode'],'Auto keeps normal live resizing in smaller windows and uses an immediate divider guide above 2 million screen pixels. Fast updates pane contents when you release the divider, avoiding expensive full-window painting during dragging. Live redraws panes while dragging and can be slower on large high-DPI windows.',['auto','fast','live'])
        self.button(parent,'Reset pane sizes',lambda:self.live_layout.reset(),'Restore point-list, image-view and properties widths, and the measurement-table height. No images, controls or settings are changed.').pack(fill='x',pady=4)
        self.field(parent,'Green error limit · degrees',self.settings['error_good_deg'],'Angular measurement errors at or below this limit are green. Default 0.10°: about 1.6 horizontal pixels on a 5760-wide panorama at the equator. Choose limits for your resolution and marking precision. This indicates measurement consistency, not surveyed accuracy.')
        self.field(parent,'Yellow error limit · degrees',self.settings['error_warning_deg'],'Errors above the green limit through this value are yellow; larger errors are red. Default 0.50°. Must exceed the positive green limit. Missing/untriangulated errors remain uncoloured. Low fitted errors do not establish absolute accuracy.')
        for title, key, tip in [('Keep zoom when changing images', 'preserve_zoom', 'Preserve pixel scale and relative view centre when stepping through panoramas.'),
            ('Link zoom between panes', 'linked_zoom', 'Synchronize original pixel scale. View centres and panning remain independent.'),
            ('Wrap panorama seam', 'wrap', 'At close zoom, pan continuously across the left/right edge. Latitude does not wrap.'),
            ('Show unmeasured predictions', 'predictions', 'Dashed guidance from triangulation. Never an automatic match or saved observation; verify and click the physical feature.'),
            ('Pixel display at close zoom', 'pixelated', 'Use crisp nearest-pixel interpolation for precise inspection. Switch off for smooth display.')]:
            widget = tk.Checkbutton(parent, text=title, variable=self.settings[key], command=self.update_view_settings, bg=PANEL, fg=TEXT,
                selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT, wraplength=260, justify='left', anchor='w', font=('Segoe UI', 9))
            widget.pack(fill='x', pady=5); self.guide(widget, title, tip)
        arrows=tk.Checkbutton(parent,text='Left/right arrows browse images',variable=self.settings['browse_arrows'],command=self.update_view_settings,
            bg=PANEL,fg=TEXT,selectcolor=FIELD,activebackground=PANEL,activeforeground=TEXT)
        arrows.pack(anchor='w',pady=5);self.guide(arrows,'Arrow navigation','Plain left/right browses the active image pane. Switch off for 1 px horizontal marker nudging. Ctrl+arrows nudge 0.1 px; Shift+arrows 10 px. Alt+left/right always browses.')
        labels = tk.Checkbutton(parent, text='Label other points', variable=self.settings['other_labels'], command=self.update_view_settings,
            bg=PANEL, fg=TEXT, selectcolor=FIELD, activebackground=PANEL, activeforeground=TEXT)
        labels.pack(anchor='w', pady=5); self.guide(labels, 'Other point labels', 'The active point is always labelled. Other names appear on hover; enable this to show every name, which can clutter dense areas.')
        self.label(parent, 'Hover over a setting for a brief guide.\nAll processing remains on this PC.', muted=True, wraplength=260, pady=12).pack(fill='x')
    def update_error_colours(self,*_):
        good,warning=self.error_cells.limits();text=f'Green ≤ {good:g}° · Yellow ≤ {warning:g}° · Red above\nMeasurement consistency; unavailable: —'
        if not self.error_cells.valid_limits:text+='\nIncomplete/invalid limits: last valid values retained.'
        self.error_legend.set(text);self.error_cells.schedule()
    def shortcut(self, event, command):
        if isinstance(event.widget, (tk.Entry, ttk.Combobox, tk.Spinbox, tk.Text)) and command in [self.new, self.undo, self.redo]: return
        command(); return 'break'
    def property_changed(self, *_):
        if not self.populating: self.pending = True
    def coordinate_state(self, *_):
        for entry in self.coordinate_entries: entry.configure(state='disabled' if self.property_vars['type'].get() == 'Tie point' else 'normal')
    def resolve_properties(self):
        if not self.pending or not self.selected(): return True
        answer = messagebox.askyesnocancel('Point properties', 'Apply pending property edits?', parent=self.window)
        if answer is None: return False
        if answer: return self.apply_properties()
        self.pending = False; return True
    def can_edit(self):
        if self.busy: self.status.set('Refinement is running. Editing resumes when it finishes.'); return False
        return True
    def unsaved_guard(self):
        if not self.resolve_properties(): return False
        if not self.document.dirty: return True
        answer = messagebox.askyesnocancel('Unsaved control points', 'Save your changes before continuing?', parent=self.window)
        return False if answer is None else self.save() if answer else True
    def error(self, title, error):
        self.audit.error(title,error);self.status.set(str(error));messagebox.showerror(title,str(error),parent=self.window)
    def open_log(self):
        self.audit.event('log_open');os.startfile(str(self.audit.path))
    @staticmethod
    def read_alignment(path):
        return load_alignment(path)
    def open(self):
        if not self.can_edit() or not self.resolve_properties(): return
        path = filedialog.askopenfilename(parent=self.window, title='Open reconstruction JSON or COLMAP model file', filetypes=[('Reconstruction / COLMAP', '*.json *.bin *.txt')])
        if not path: return
        try:
            component, cameras = self.read_alignment(path)
            candidates=[p/'images' for p in list(Path(path).parents)[:4]]
            if self.image_dir and self.reconstruction is None:
                # The intended next step from photos-only preparation is opening
                # a result for those exact originals; no new pose is invented.
                candidates.insert(0,self.image_dir)
            directory = next((p for p in candidates if p.is_dir() and all((p/n).is_file() for n in component['shots'])), None)
            if directory is None:
                value = filedialog.askdirectory(parent=self.window, title='Choose matching original images')
                if not value: return
                directory = Path(value)
            missing = [n for n in component['shots'] if not (directory/n).is_file()]
            if missing: raise ValueError(f'{len(missing)} registered images missing. First: {missing[0]}')
            keep=bool(self.controls['points']) and self.image_dir is not None and directory.absolute()==self.image_dir.absolute()
            if keep:
                for point in self.controls['points']:
                    for observation in point['observations']:
                        name=observation['shot_id']
                        old_spherical=spherical(self.cameras[name]) if name in self.cameras else self.settings['image_projection'].get()=='360 equirectangular'
                        if name not in cameras or list(cameras[name][2:])!=list(self.image_sizes.get(name,[])) or old_spherical!=spherical(cameras[name]):keep=False
            if not keep:
                if not self.unsaved_guard():return
                self.document = ControlDocument(); self.controls_path = None; self.active.set(''); self.pending = False
            self.install_alignment(path, component, cameras, directory); self.status.set('Alignment loaded. Add a point and mark the same feature in multiple views.')
            if keep:self.status.set('Alignment loaded for the same original images. Existing controls retained; inspect them in 2D or 3D.')
        except Exception as error: self.error('Cannot open alignment', error)
    def install_alignment(self, path, component, cameras, directory):
        self.viewer_generation+=1;self.viewer_key=None
        self.scene_generation+=1;self.scene_key=None
        self.reconstruction, self.component, self.cameras, self.image_dir = Path(path), component, cameras, Path(directory)
        self.image_names = sorted(component['shots']);families=sorted({'360' if spherical(c) else 'perspective' for c in cameras.values()});self.context.set(f'{Path(path).name} · {len(self.image_names)} images · {" / ".join(families)} · {Path(path).parent.name}')
        self.image_sizes={name:list(camera[2:]) for name,camera in cameras.items()}
        if families==['perspective']:self.settings['image_projection'].set('Perspective')
        elif families==['360']:self.settings['image_projection'].set('360 equirectangular')
        self.audit.event('alignment_loaded',path=path,image_dir=directory,cameras=len(cameras),points=len(component.get('points',{})))
        self.context_guide.text = str(Path(path))
        for index, pane in enumerate(self.panes):
            pane['choice']['values'] = self.image_names; pane['name'].set(self.image_names[min(index, len(self.image_names)-1)]); self.show(pane, preserve=False)
        self.refresh()
        self.update_view_settings()
        self.refine_button.configure(state='normal' if not self.busy else 'disabled')
        if self.grid and not self.grid.closed:self.grid.refresh(reset=True)
    def open_images(self):
        if not self.can_edit() or not self.unsaved_guard():return
        folder=filedialog.askdirectory(parent=self.window,title='Choose original images (image type is set in Settings)')
        if not folder:return
        try:
            from PIL import Image
            paths=sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp'})
            if not paths:raise ValueError('No panorama images found.')
            sizes={}
            for path in paths:
                with Image.open(path) as image:
                    if self.settings['image_projection'].get()=='360 equirectangular' and image.width!=2*image.height:raise ValueError(f'{path.name} is not a full 2:1 panorama. For perspective photos choose Perspective in Settings first.')
                    sizes[path.name]=list(image.size)
            self.document=ControlDocument();self.active.set('');self.controls_path=None;self.pending=False
            self.reconstruction=self.component=None;self.cameras={};self.image_dir=Path(folder);self.image_names=list(sizes);self.image_sizes=sizes
            self.scene_generation+=1;self.scene_key=None;self.scene_cache=None
            for nav in self.native_views():nav.set_data({'points':np.empty((0,3)),'colors':np.empty((0,3),dtype='uint8'),'centers':{},'camera_rotations':{},'controls':{},'center':np.zeros(3),'extent':1.,'source':'photos-only','total_points':0})
            self.audit.event('panoramas_loaded',path=folder,images=len(sizes))
            self.viewer_generation+=1;self.viewer_key=None
            self.context.set(f'{len(sizes)} images · {self.settings["image_projection"].get()} · Measurements only')
            self.context_guide.text=str(self.image_dir)
            for index,pane in enumerate(self.panes):pane['choice']['values']=self.image_names;pane['name'].set(self.image_names[min(index,len(self.image_names)-1)]);self.show(pane,preserve=False)
            self.refresh();self.refine_button.configure(state='disabled');self.status.set('Mark/import controls, save them, then Use with main Align. Initial engine registration happens before control refinement.')
            if self.grid and not self.grid.closed:self.grid.refresh(reset=True)
            if self.viewer:self.viewer.close();self.viewer=None;self.viewer_key=None
        except Exception as error:self.error('Cannot open panoramas',error)
    def import_csv(self):
        if not self.can_edit() or not self.resolve_properties():return
        if not self.image_dir:self.status.set('Open an alignment or original panoramas first.');return
        path=filedialog.askopenfilename(parent=self.window,title='Import RealityScan control points',filetypes=[('CSV or text','*.csv *.txt *.tsv'),('All files','*.*')])
        if not path:return
        try:
            from control_points_csv import CSVImportDialog
            CSVImportDialog(self,path)
        except Exception as error:self.error('Cannot open CSV',error)
    def attach_to_main(self):
        if not self.on_attach or not self.image_dir or not self.can_edit():return
        if not self.resolve_properties():return
        try:
            payload=self.refinement_payload(for_initial=True)
            if not any(p.get('enabled',True) for p in payload['points']) and not payload.get('image_policy'):raise ValueError('Add controls or configure image exclusions/masks before attaching.')
            budget,iterations=int(self.settings['max_points'].get()),int(self.settings['iterations'].get())
            if not 1<=budget<=50000 or not 5<=iterations<=500:raise ValueError('Check refinement budgets in Settings.')
            if not self.save():return
            self.on_attach(str(self.controls_path),str(self.image_dir),budget,iterations,str(self.settings['backend'].get()))
            self.status.set('Controls attached. Use the main Align button for initial registration followed by refinement. This window’s Refine button updates an existing alignment only.')
        except Exception as error:self.error('Cannot attach controls',error)
    def open_predictions(self):
        if self.prediction_browser and not self.prediction_browser.closed:self.prediction_browser.refresh();self.prediction_browser.window.lift();return
        from control_point_predictions import PredictionBrowser
        self.prediction_browser=PredictionBrowser(self);self.audit.event('predictions_open',point=self.active.get())
    def open_grid(self,predicted=False):
        if not self.image_names:self.status.set('Open panoramas or an alignment first.');return
        if self.grid and not self.grid.closed:self.grid.refresh();self.grid.window.lift()
        else:
            from control_images_grid import ImageGrid
            self.grid=ImageGrid(self)
        if predicted:self.grid.filter.set('Predicted for selected point');self.grid.refresh(reset=True)
        self.audit.event('image_grid_open',images=len(self.image_names),predicted=predicted)
    def choose_image(self,name,index=None):
        if name not in self.image_sizes:return
        index=self.focused if index is None else index;self.focused=index;pane=self.panes[index];pane['name'].set(name)
        if pane['mode'].get()!='Image':self.set_pane_mode(pane,'Image',load=False)
        self.show(pane)
        self.audit.event('image_selected',image=name,pane=index)
        self.highlight_images()
    def open_image_policy(self):
        if not self.image_names:self.status.set('Open images or an alignment first.');return
        if self.image_policy_panel and not self.image_policy_panel.closed:self.image_policy_panel.window.lift();return
        from manual_image_policy_ui import ImagePolicyPanel
        self.image_policy_panel=ImagePolicyPanel(self)
    def toggle_measurement_image(self):
        names=self.measurement_tree.selection()
        if not names or not self.can_edit():return
        from manual_image_policy import validate_policy
        policy=validate_policy(self.controls.get('image_policy'));excluded=set(policy['excluded_images']);name=names[0]
        excluded.discard(name) if name in excluded else excluded.add(name)
        self.document.checkpoint();policy['excluded_images']=sorted(excluded);self.controls['image_policy']=policy;self.audit.event('image_policy_changed',policy=policy);self.refresh()
        if self.image_policy_panel and not self.image_policy_panel.closed:self.image_policy_panel.refresh()
    def effective_point(self,registered_only=False):
        point=self.selected()
        if not point:return None
        from manual_image_policy import Masks,validate_policy
        policy=validate_policy(self.controls.get('image_policy'))
        if self.mask_set is None or self.mask_set.policy!=policy:self.mask_set=Masks(policy)
        return dict(point,observations=[o for o in point['observations'] if o['shot_id'] in self.image_sizes and (not registered_only or o['shot_id'] in self.cameras) and self.mask_set.allowed(o['shot_id'],o['pixel'],self.image_sizes[o['shot_id']])])
    def highlight_images(self):
        if not hasattr(self,'measurement_tree') or self.closed:return
        from manual_image_policy import validate_policy
        excluded=set(validate_policy(self.controls.get('image_policy'))['excluded_images']);shown={}
        for index,pane in enumerate(self.panes):
            if pane['mode'].get()=='Image' and pane['loaded_name']:shown[pane['loaded_name']]='shown_left' if index==0 else 'shown_right'
        for nav in self.native_views():
            if nav.winfo_ismapped() and nav.camera_selected:
                name=nav.camera_selected
                if nav.selected_camera_layer=='B' and nav.comparison:name=next((v['a_image'] for v in nav.comparison['report']['camera_differences'] if v['b_image']==name),None)
                if name:shown[name]='shown_3d'
        if self.viewer and self.browser_camera_selected:shown[self.browser_camera_selected]='shown_3d'
        observations={o['shot_id']:o for o in self.selected()['observations']} if self.selected() else {}
        for name in self.measurement_tree.get_children():
            inactive=name in excluded or self.mask_set is not None and name in observations and not self.mask_set.allowed(name,observations[name]['pixel'],self.image_sizes[name])
            self.measurement_tree.item(name,tags=tuple(t for t in [shown.get(name),'excluded' if inactive else None] if t))
        self.error_cells.schedule()
    def focus_shortcut(self,event=None):
        pane=self.panes[self.focused]
        if pane['mode'].get()=='Image':self.focus_point(pane)
        elif pane['mode'].get()=='3D' and pane['scene']:pane['scene'].focus_selected()
        else:self.status.set('Choose an Image or 3D view before Focus Point. Ctrl+F.')
        return 'break'
    def navigate_key(self,pane,direction):self.navigate(pane,direction);return 'break'
    def window_arrow(self,event,direction):
        if isinstance(event.widget,(tk.Entry,ttk.Combobox,tk.Spinbox,tk.Text,ttk.Treeview,tk.Canvas)) or event.state & (1|4|8):return
        if bool(self.settings['browse_arrows'].get()) and self.panes[self.focused]['mode'].get()=='Image':self.navigate(self.panes[self.focused],direction);return 'break'
    def open_3d(self,browser=False):
        if not self.component:self.status.set('3D inspection needs an aligned reconstruction; photos-only mode has no cloud or camera poses.');return
        if browser:self.request_3d(open_browser=True);return
        if self.native_3d is None:
            from control_points_native_3d import NativeControlNavigator
            self.native_3d=NativeControlNavigator(self,self.scene_page);self.native_3d.pack(fill='both',expand=True)
        self.tabs.select(self.scene_page);self.request_scene();self.audit.event('native_3d_open')
    def request_scene(self):
        views=self.native_views()
        if not views or not self.component or self.closed:return
        key=(str(self.reconstruction),self.document.signature())
        if key==self.scene_key and self.scene_cache is not None:
            for nav in views:
                if nav.data is None or nav.data.get('source')!=self.scene_cache['source']:nav.set_data(self.scene_cache)
                nav.set_selected(self.active.get())
            return
        self.scene_generation+=1;generation=self.scene_generation
        if self.scene_future:self.scene_future.cancel()
        component=self.component;controls=copy.deepcopy(self.controls);cameras=self.cameras;source=str(self.reconstruction);result_queue=self.result_queue;audit=self.audit
        def work():
            try:
                from control_points_3d import scene_data
                data=scene_data(component,controls,cameras);data['source']=source
                result_queue.put(('native_scene',generation,key,data))
            except Exception as error:audit.error('native_scene_error',error);result_queue.put(('viewer_error',str(error)))
        self.scene_future=self.viewer_executor.submit(work)
    def request_3d(self,open_browser=False):
        if not self.component or self.closed or not (self.viewer or open_browser):return
        transforms=copy.deepcopy(self.native_3d.transforms) if self.native_3d else None
        transform_key=json.dumps({k:v.report() for k,v in transforms.items()},sort_keys=True) if transforms else ''
        key=(str(self.reconstruction),self.document.signature(),transform_key)
        if self.viewer and key==self.viewer_key:
            self.viewer.set_selected(self.active.get())
            if open_browser:self.result_queue.put(('viewer_ready',self.viewer.url))
            return
        self.viewer_generation+=1;generation=self.viewer_generation
        if self.viewer_future:self.viewer_future.cancel()
        component=self.component;controls=copy.deepcopy(self.controls);cameras=self.cameras;selected=self.active.get()
        def work():
            if self.closed or generation!=self.viewer_generation:return
            try:
                if self.viewer is None:
                    from control_points_3d import ControlSceneViewer
                    self.viewer=ControlSceneViewer(self.result_queue)
                if self.closed or generation!=self.viewer_generation:self.viewer.close();self.viewer=None;return
                self.viewer.source=key[0]
                if transforms:self.viewer.m54.transforms=transforms
                self.viewer.update(component,controls,cameras,selected)
                if open_browser:self.viewer.wait_ready()
                if generation==self.viewer_generation:self.viewer_key=key
                if open_browser and not self.closed and generation==self.viewer_generation:self.result_queue.put(('viewer_ready',self.viewer.url))
            except Exception as error:
                if not self.closed and generation==self.viewer_generation:self.result_queue.put(('viewer_error',str(error)))
        self.viewer_future=self.viewer_executor.submit(work)
        if open_browser:self.status.set('Preparing local 3D view…')
    def show(self, pane, preserve=None):
        if not self.image_dir or pane['name'].get() not in self.image_sizes: self.status.set('Select an exact image from the catalog.'); return
        name = pane['name'].get(); pane['generation'] += 1; generation = pane['generation']
        if pane['future']: pane['future'].cancel()
        if pane['mode'].get()!='Image':
            if pane['loaded_name']!=name:pane['loaded_name']=None;pane['view'].asset=None
            pane['loading']=False;return
        pane['loading'] = True; pane['loaded_name'] = None
        pane['view'].overlay = 'Loading original panorama…'; pane['info'].set('Loading original image…'); self.draw()
        preserve = bool(self.settings['preserve_zoom'].get()) if preserve is None else preserve
        expected, path, index = self.image_sizes[name], self.image_dir/name, self.panes.index(pane);result_queue=self.result_queue
        def work():
            try:
                asset = PanoramaAsset(path, expected)
                result_queue.put(('image', index, generation, name, asset, preserve))
            except Exception as error: result_queue.put(('image_error', index, generation, str(error)))
        pane['future'] = self.executor.submit(work)
    def filter_images(self, pane, event):
        if event.keysym not in ['Return', 'Up', 'Down', 'Left', 'Right', 'Tab']:
            pane['choice']['values'] = [n for n in self.image_names if pane['name'].get().casefold() in n.casefold()]
    def navigate(self, pane, step):
        if not self.image_names: return
        self.focused=self.panes.index(pane)
        index = self.image_names.index(pane['name'].get()) if pane['name'].get() in self.image_names else 0
        pane['name'].set(self.image_names[(index+step) % len(self.image_names)]); self.show(pane)
    def set_focus(self, index): self.focused = index
    def view_navigation(self, pane):
        zoom = pane['view'].transform.zoom; pane['zoom_text'].set(f'{zoom*100:.0f}%')
        if bool(self.settings['linked_zoom'].get()) and pane['mode'].get()=='Image':
            for other in self.panes:
                if other is pane or other['mode'].get()!='Image': continue
                other['view'].transform.zoom = zoom; other['view'].transform.constrain(); other['view'].request_draw(); other['zoom_text'].set(f'{zoom*100:.0f}%')
    def update_view_settings(self):
        try:self.audit.event('view_settings',settings={k:v.get() for k,v in self.settings.items()},mode=self.mode.get())
        except tk.TclError:pass
        try: size = int(self.settings['marker_size'].get())
        except (ValueError, tk.TclError): size = 6
        for pane in self.panes:
            view = pane['view']; view.pan_mode = self.mode.get() == 'Pan'; view.marker_size = max(3, min(16, size))
            is_sphere=spherical(self.cameras[pane['name'].get()]) if pane['name'].get() in self.cameras else self.settings['image_projection'].get()=='360 equirectangular'
            view.pixelated = bool(self.settings['pixelated'].get()); view.transform.wrap = bool(self.settings['wrap'].get()) and is_sphere; view.transform.constrain()
            view.label_other_points = bool(self.settings['other_labels'].get())
            view.arrow_browse=bool(self.settings['browse_arrows'].get())
            view.canvas.configure(cursor='fleur' if view.pan_mode else 'crosshair')
        self.draw()
    def refresh_list(self):
        if not hasattr(self, 'point_tree'): return
        self.refreshing = True; self.point_tree.delete(*self.point_tree.get_children())
        search, filter_name = self.search.get().casefold(), self.point_filter.get()
        for p in self.controls['points']:
            enabled, count, kind = p.get('enabled', True), len(p['observations']), p.get('type', 'tie')
            if search not in p['id'].casefold(): continue
            if filter_name == 'Needs observations' and count >= 2 or filter_name == 'Disabled' and enabled or filter_name == 'Ground control' and kind != 'gcp' or filter_name == 'Check points' and kind != 'check': continue
            self.point_tree.insert('', 'end', iid=p['id'], values=(p['id'], TYPES[kind], count), tags=('disabled' if not enabled else 'incomplete' if count < 2 else 'ready',))
        if self.point_tree.exists(self.active.get()): self.point_tree.selection_set(self.active.get())
        self.refreshing = False
        self.point_summary.set(f"{len(self.controls['points'])} points · {sum(len(p['observations']) for p in self.controls['points'])} measurements" + (' · Unsaved' if self.document.dirty else ''))
    def refresh(self):
        if not self.selected(): self.active.set('')
        self.refresh_list(); self.populate_properties(); self.refresh_distances(); self.draw()
        self.window.title('Manual control points' + (' *' if self.document.dirty else '') + ' · Abu Align & Train')
        if self.grid and not self.grid.closed:self.grid.refresh()
        for pane in self.panes:
            if pane['grid'] and not pane['grid'].closed:pane['grid'].refresh()
        if self.prediction_browser and not self.prediction_browser.closed:self.prediction_browser.refresh()
        signature=self.document.signature()
        if signature!=self.logged_signature:
            self.audit.event('document_changed',controls=self.controls,active_point=self.active.get());self.logged_signature=signature
        self.request_scene()
        self.request_3d()
    def tree_selection(self, event):
        values = self.point_tree.selection()
        if not getattr(self, 'refreshing', False) and values and values[0] != self.active.get(): self.select(values[0])
    def select(self, name):
        if not self.resolve_properties(): self.refresh_list(); return False
        self.active.set(name); self.pending = False; self.refresh()
        return True
    def populate_properties(self):
        point = self.selected(); self.populating = True
        selected_images = self.measurement_tree.selection()
        values = {'name': '', 'type': 'Tie point', 'sigma_m': '.01', 'x': '', 'y': '', 'z': ''}
        if point:
            values.update(name=point['id'], type=TYPES[point.get('type', 'tie')], sigma_m=str(point.get('sigma_m', .01)))
            for k, v in zip(['x', 'y', 'z'], point.get('position', [])): values[k] = str(v)
        for key, variable in self.property_vars.items(): variable.set(values[key])
        self.enabled.set(point.get('enabled', True) if point else True); self.populating = False; self.pending = False
        self.measurement_tree.delete(*self.measurement_tree.get_children())
        _, description, errors = point_geometry(self.effective_point(registered_only=True), self.cameras) if point and self.cameras else (None, 'Measurements only · run initial alignment to triangulate' if point else 'Select a point', {})
        self.geometry_status.set(description)
        if point:
            for obs in point['observations']:
                x, y = obs['pixel']; error = errors.get(obs['shot_id'])
                self.measurement_tree.insert('', 'end', iid=obs['shot_id'], values=(obs['shot_id'], f'{x:.2f} / {y:.2f}', f"{obs.get('sigma_px',1):g}", '—' if error is None else f'{error:.3f}'))
        self.error_cells.set_errors(errors)
        if selected_images and self.measurement_tree.exists(selected_images[0]): self.measurement_tree.selection_set(selected_images[0])
        self.highlight_images()
    def draw(self):
        point = self.effective_point(registered_only=bool(self.cameras)); xyz, _, errors = point_geometry(point, self.cameras) if point and self.cameras else (None, '', {})
        if errors and max(errors.values()) > 1: xyz = None
        for pane in self.panes:
            name, view = pane['loaded_name'], pane['view']; view.active = self.active.get()
            view.markers = [(p['id'], o['pixel'], p.get('enabled', True)) for p in self.controls['points'] for o in p['observations'] if o['shot_id'] == name]
            measured = any(n == self.active.get() for n, _, _ in view.markers)
            view.prediction = projected_pixel(xyz, self.cameras[name]) if xyz is not None and name in self.cameras and not measured and bool(self.settings['predictions'].get()) else None
            browser=self.prediction_browser
            if browser and not browser.closed and browser.snapshot==browser.signature() and name in browser.visual and not measured and bool(self.settings['predictions'].get()):view.prediction=browser.visual[name]['pixel']
            view.request_draw()
        self.highlight_images()
    def focus_point(self, pane):
        point = self.selected()
        if not point or pane['loading']: return
        pixel = next((o['pixel'] for o in point['observations'] if o['shot_id'] == pane['loaded_name']), pane['view'].prediction)
        if pixel: pane['view'].focus(pixel)
        else: self.status.set('No measurement or reliable prediction for this point in this image.')
    def new(self):
        if not self.can_edit() or not self.resolve_properties(): return
        if not self.image_dir: self.status.set('Open an alignment or panoramas before adding points.'); return
        self.document.checkpoint(); name = self.document.next_name(); self.controls['points'].append({'id': name, 'type': 'tie', 'enabled': True, 'observations': []})
        self.active.set(name); self.refresh(); self.status.set(f'{name} created. Mark the same feature in multiple translated views.')
    def delete(self):
        if not self.can_edit() or not self.selected(): return
        self.document.checkpoint(); self.document.delete(self.active.get()); self.active.set(''); self.pending = False; self.refresh()
    def apply_properties(self):
        point = self.selected()
        if not self.can_edit() or not point: return False
        try:
            name = self.property_vars['name'].get().strip(); kind = next(k for k, v in TYPES.items() if v == self.property_vars['type'].get())
            if not name or name != point['id'] and self.document.point(name): raise ValueError('Choose a unique, non-empty name.')
            candidate = copy.deepcopy(self.controls); target = next(p for p in candidate['points'] if p['id'] == point['id'])
            target.update(type=kind, enabled=self.enabled.get())
            if kind != 'tie': target.update(position=[float(self.property_vars[k].get()) for k in ['x', 'y', 'z']], sigma_m=float(self.property_vars['sigma_m'].get()))
            else: target.pop('position', None); target.pop('sigma_m', None)
            validate_controls(candidate)
            if kind == 'check' and any(point['id'] in [d['a'], d['b']] for d in candidate['distances']): raise ValueError('Remove this point’s distances before changing it to a check point.')
            self.document.checkpoint(); old = point['id']; self.controls = candidate; self.document.rename(old, name)
            self.active.set(name); self.pending = False; self.refresh(); self.status.set(f'{name}: properties applied.'); return True
        except (ValueError, StopIteration) as error: self.error('Invalid properties', error); return False
    def measure(self, pane, pixel, remove=False):
        if not self.can_edit() or pane['loading'] or not pane['loaded_name']: return
        if not self.resolve_properties(): self.draw(); return
        point = self.selected()
        if not point: self.status.set('Add or select a point before placing a measurement.'); return
        try:
            sigma = float(self.settings['sigma_px'].get())
            if not math.isfinite(sigma) or sigma <= 0: raise ValueError('New measurement uncertainty must be positive.')
            existing = next((o for o in point['observations'] if o['shot_id'] == pane['loaded_name']), None)
            if existing: sigma = existing.get('sigma_px', 1)
            self.document.checkpoint(); self.document.measure(point['id'], pane['loaded_name'], None if remove else pixel, sigma)
            self.refresh(); self.status.set(f"{point['id']}: {len(point['observations'])} measurements · original pixels")
        except (ValueError, tk.TclError) as error: self.error('Invalid measurement', error)
    def undo(self):
        if self.can_edit() and self.resolve_properties() and self.document.undo(): self.refresh()
    def redo(self):
        if self.can_edit() and self.resolve_properties() and self.document.redo(): self.refresh()
    def selected_observation(self):
        point, values = self.selected(), self.measurement_tree.selection()
        return next((o for o in point['observations'] if o['shot_id'] == values[0]), None) if point and values else None
    def measurement_selected(self, event):
        obs = self.selected_observation()
        if obs: self.measurement_sigma.set(str(obs.get('sigma_px', 1)))
    def open_measurement(self, index):
        obs = self.selected_observation()
        if not obs: return
        self.focused = index; pane = self.panes[index]; pane['name'].set(obs['shot_id']); pane['focus_after_load'] = (obs['shot_id'], list(obs['pixel']))
        if pane['mode'].get()!='Image':self.set_pane_mode(pane,'Image',load=False)
        self.show(pane)
    def apply_measurement_sigma(self):
        if not self.resolve_properties(): return
        obs = self.selected_observation()
        if not self.can_edit() or not obs: return
        try:
            value = float(self.measurement_sigma.get())
            if not math.isfinite(value) or value <= 0: raise ValueError('Uncertainty must be positive.')
            self.document.checkpoint(); obs['sigma_px'] = value; self.refresh()
        except ValueError as error: self.error('Invalid uncertainty', error)
    def remove_measurement(self):
        if not self.resolve_properties(): return
        obs = self.selected_observation()
        if self.can_edit() and obs: self.document.checkpoint(); self.document.measure(self.active.get(), obs['shot_id'], None); self.refresh()
    def refresh_distances(self):
        choices = [p['id'] for p in self.controls['points'] if p.get('enabled', True) and p.get('type') != 'check' and len(p['observations']) >= 2]
        for choice in self.distance_choices: choice['values'] = choices
        self.distance_tree.delete(*self.distance_tree.get_children())
        disabled = {p['id'] for p in self.controls['points'] if not p.get('enabled', True)}
        for i, d in enumerate(self.controls['distances']):
            inactive = d['a'] in disabled or d['b'] in disabled or not d.get('enabled', True)
            self.distance_tree.insert('', 'end', iid=str(i), values=(f"{d['a']} ↔ {d['b']}" + (' (inactive)' if inactive else ''), f"{d['meters']:g}"))
    def distance(self):
        if not self.can_edit() or not self.resolve_properties(): return
        try:
            a, b = self.distance_a.get(), self.distance_b.get()
            for name in [a, b]:
                point = self.document.point(name)
                if not point or point.get('type') == 'check' or not point.get('enabled', True) or len(point['observations']) < 2: raise ValueError('Choose two enabled, measured tie/ground-control points.')
            value = {'a': a, 'b': b, 'meters': float(self.distance_m.get()), 'sigma_m': float(self.distance_sigma.get())}
            candidate = copy.deepcopy(self.controls); candidate['distances'].append(value); validate_controls(candidate)
            self.document.checkpoint(); self.controls = candidate; self.refresh()
        except ValueError as error: self.error('Invalid distance', error)
    def remove_distance(self):
        if not self.resolve_properties(): return
        if self.can_edit() and self.distance_tree.selection(): self.document.checkpoint(); del self.controls['distances'][int(self.distance_tree.selection()[0])]; self.refresh()
    def validate_for_alignment(self, payload):
        data = validate_controls(payload)
        if self.image_sizes:
            for point in data['points']:
                for obs in point['observations']:
                    if obs['shot_id'] not in self.image_sizes: raise ValueError(f"Unknown image: {obs['shot_id']}")
                    width, height = self.image_sizes[obs['shot_id']]; x, y = obs['pixel']
                    recorded=data.get('image_sizes',{}).get(obs['shot_id'])
                    expected_family=('spherical' if spherical(self.cameras[obs['shot_id']]) else 'perspective') if obs['shot_id'] in self.cameras else ('spherical' if self.settings['image_projection'].get()=='360 equirectangular' else 'perspective')
                    recorded_family=data.get('image_projections',{}).get(obs['shot_id'],data.get('image_projection'))
                    if recorded_family in {'spherical','perspective'} and recorded_family!=expected_family:raise ValueError('Saved control projection differs from the selected original image. Re-mark the feature in the correct image projection.')
                    if recorded is not None and list(recorded)!=[width,height]:raise ValueError('Saved control dimensions differ from the selected original images. Do not reuse pixels at another resolution.')
                    if not 0 <= x < width or not 0 <= y < height: raise ValueError('Measurement outside original image dimensions.')
        return data
    def load(self):
        if not self.can_edit() or not self.unsaved_guard(): return
        if not self.image_dir: self.status.set('Open the matching alignment or panoramas before loading controls.'); return
        path = filedialog.askopenfilename(parent=self.window, title='Load control points', filetypes=[('Abu controls JSON', '*.json')])
        if not path: return
        try:
            payload = self.validate_for_alignment(json.loads(Path(path).read_text(encoding='utf-8')))
            from manual_image_policy import Masks
            Masks(payload.get('image_policy')).validate_catalog(self.image_sizes)
            from control_measurement_colours import validate_limits
            display=validate_limits(payload.get('display_settings',{}))
            layout=payload.get('layout_settings',{})
            if not isinstance(layout,dict) or layout.get('resize_mode','auto') not in ['auto','fast','live']:raise ValueError('Pane resizing must be auto, fast or live.')
            refinement=validate_refine_settings(payload['refinement_settings']) if 'refinement_settings' in payload else {}
            for key,value in display.items():self.settings[key].set(value)
            self.settings['resize_mode'].set(layout.get('resize_mode','auto'))
            if 'refinement_settings' in payload:
                for key,value in refinement.items():self.settings[key].set(value)
            self.document = ControlDocument(); self.controls = payload; self.document.mark_saved(); self.controls_path = Path(path)
            self.active.set(payload['points'][0]['id'] if payload['points'] else ''); self.pending = False; self.refresh()
            self.status.set(f'Loaded {len(payload["points"])} points from {Path(path).name}.')
        except Exception as error: self.error('Cannot load controls', error)
    def save(self):
        if not self.can_edit() or not self.resolve_properties(): return False
        path = self.controls_path
        if path is None:
            value = filedialog.asksaveasfilename(parent=self.window, title='Save control points', initialfile='control_points.json', defaultextension='.json')
            if not value: return False
            path = Path(value)
        try:
            if path.suffix.lower() != '.json': raise ValueError('Save controls as a .json file.')
            if self.reconstruction and path.absolute() == self.reconstruction.absolute(): raise ValueError('Controls must not overwrite the reconstruction.')
            payload=self.validate_for_alignment(self.controls);payload['image_sizes']=copy.deepcopy(self.image_sizes);payload['refinement_settings']=self.refine_settings()
            from control_measurement_colours import validate_limits
            payload['display_settings']=validate_limits({k:self.settings[k].get() for k in ['error_good_deg','error_warning_deg']})
            if self.settings['resize_mode'].get() not in ['auto','fast','live']:raise ValueError('Pane resizing must be auto, fast or live.')
            payload['layout_settings']={'resize_mode':self.settings['resize_mode'].get()}
            payload['image_projection']=('spherical' if all(spherical(c) for c in self.cameras.values()) else 'perspective' if all(not spherical(c) for c in self.cameras.values()) else 'mixed') if self.cameras else 'spherical' if self.settings['image_projection'].get()=='360 equirectangular' else 'perspective'
            payload['image_projections']={n:('spherical' if spherical(self.cameras[n]) else 'perspective') if n in self.cameras else payload['image_projection'] for n in self.image_names} if self.cameras else {n:payload['image_projection'] for n in self.image_names}
            atomic_json(path, payload); self.controls_path = path; self.document.mark_saved(); self.refresh()
            self.audit.event('controls_saved',path=path,points=len(payload['points']),measurements=sum(len(p['observations']) for p in payload['points']))
            self.status.set(f'Controls saved: {path}'); return True
        except Exception as error: self.error('Cannot save controls', error); return False
    def save_as(self):
        previous = self.controls_path; self.controls_path = None
        if not self.save(): self.controls_path = previous
    def refinement_payload(self,for_initial=False):
        from manual_image_policy import effective_controls
        original=self.validate_for_alignment(self.controls);payload,_=effective_controls(original,self.image_sizes); active = [p for p in payload['points'] if p.get('enabled', True)]; enabled = {p['id'] for p in active}
        if self.reconstruction and not for_initial and any(o['shot_id'] not in self.cameras for p in active for o in p['observations']):raise ValueError('Some measured images have no registered camera. Use with main Align to register them before Refine.')
        for p in active:
            if len(p['observations']) < 2: raise ValueError(f"{p['id']}: mark the feature in at least two images or disable it.")
        gcps = [p for p in active if p.get('type') == 'gcp']
        if gcps and len(gcps) < 3: raise ValueError('Use at least three non-collinear ground controls, or tie points / a distance for relative refinement.')
        if gcps:
            import numpy as np
            xyz = np.asarray([p['position'] for p in gcps])
            if np.linalg.matrix_rank(xyz-xyz.mean(0)) < 2: raise ValueError('Ground controls must not all lie on a straight line.')
        checked = {p['id'] for p in active if p.get('type') == 'check'}
        if any(d.get('enabled', True) and (d['a'] in checked or d['b'] in checked) for d in payload['distances']):
            raise ValueError('Check points cannot be distance endpoints; remove that distance or use tie/ground-control points.')
        original['refinement_settings']=self.refine_settings();return original
    def restore_refine_defaults(self):
        if not self.can_edit():return
        for key,value in {**REFINE_DEFAULTS,'backend':'auto','max_points':10000,'iterations':100,'sigma_px':1.,'marker_size':6}.items():self.settings[key].set(value)
        self.audit.event('refinement_defaults_restored',settings=self.refine_settings());self.status.set('Standard defaults restored. Existing measurement σ values retained; use Help → Refine for each parameter and its effects.')
    def refine_settings(self):return validate_refine_settings({k:self.settings[k].get() for k in REFINE_DEFAULTS})
    def refine(self):
        if not self.can_edit() or not self.resolve_properties(): return
        if not self.reconstruction: self.status.set('Open an alignment first.'); return
        try:
            payload = self.refinement_payload(); budget, iterations = int(self.settings['max_points'].get()), int(self.settings['iterations'].get())
            if not 1 <= budget <= 50000 or not 5 <= iterations <= 500: raise ValueError('Use 1–50000 automatic landmarks and 5–500 solver evaluations.')
        except (ValueError, tk.TclError) as error: self.error('Check refinement settings', error); return
        value = filedialog.asksaveasfilename(parent=self.window, title='Save a new refined reconstruction', initialfile='refined_reconstruction.json', defaultextension='.json')
        if not value: return
        output = Path(value); controls = output.with_suffix('.controls.json'); report_path = output.with_suffix('.report.json');settings_path=output.with_suffix('.settings.json')
        if output.suffix.lower() != '.json': self.error('Invalid output', 'Choose a .json reconstruction filename.'); return
        if any(t.absolute() == s.absolute() for t in [output, controls, report_path,settings_path] for s in [self.reconstruction, self.controls_path] if s):
            self.error('Choose a new output', 'The output files must differ from the source alignment and saved controls.'); return
        if any(p.exists() for p in [controls, report_path,settings_path]) and not messagebox.askyesno('Existing sidecars', 'Replace existing controls/report/settings sidecars for this output?', parent=self.window): return
        try: atomic_json(controls, payload);atomic_json(settings_path,self.refine_settings())
        except Exception as error: self.error('Cannot write controls', error); return
        source=self.reconstruction;backend=str(self.settings['backend'].get())
        self.audit.event('refine_start',source=source,output=output,controls=controls,backend=backend,max_points=budget,iterations=iterations,settings=self.refine_settings())
        self.set_busy(True);self.status.set('Starting isolated refinement process…')
        command=[sys.executable,'-B',str(Path(__file__).with_name('spherical_refinement.py')),'--input',str(source),'--output',str(output),'--controls',str(controls),'--max-points',str(budget),'--iterations',str(iterations),'--backend',backend,'--settings',str(settings_path)]
        self.launch_worker(command,'refine',output)
    def launch_worker(self,command,kind,output):
        self.cancel_requested=False
        def work():
            report=None;tail=[]
            try:
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.worker=process
                if self.cancel_requested:process.terminate()
                for line in process.stdout:
                    line=line.rstrip();tail.append(line);tail=tail[-15:]
                    if line.startswith('ABU_PROGRESS '):
                        event=json.loads(line[13:]);self.audit.event('solver_progress',**event);self.result_queue.put(('solver_progress',event))
                    elif line.startswith('ABU_RESULT '):report=json.loads(line[11:])
                    else:self.audit.event('worker_output',text=line)
                code=process.wait();self.worker=None
                if self.cancel_requested:self.audit.event('worker_cancelled',kind=kind);self.result_queue.put(('worker_cancelled',));return
                if code!=0:raise RuntimeError('Worker exited '+str(code)+'\n'+'\n'.join(tail))
                if kind=='refine':
                    if report is None:raise RuntimeError('Refinement worker returned no report.')
                    self.audit.event('refine_complete',report=report);self.result_queue.put(('refined',output,report))
                else:
                    exports=json.loads(output.with_suffix('.exports.json').read_text(encoding='utf-8'));self.audit.event('export_complete',exports=exports);self.result_queue.put(('exports_ready',exports))
            except Exception as error:
                self.worker=None;self.audit.error('worker_error',error);self.result_queue.put(('refine_error',str(error)))
        threading.Thread(target=work,daemon=True,name='Abu control worker monitor').start()
    def cancel_refinement(self):
        if not self.busy:return
        self.cancel_requested=True;self.audit.event('cancel_requested')
        if self.worker and self.worker.poll() is None:self.worker.terminate()
        self.status.set('Stopping refinement… Source files are preserved.')
    def export_current(self):
        if not self.can_edit():return
        source=self.last_result or self.reconstruction
        if not source:self.status.set('Open a refined reconstruction first.');return
        self.audit.event('export_start',source=source);self.set_busy(True);self.status.set('Exporting sparse cloud and camera alignment CSV…')
        command=[sys.executable,'-B',str(Path(__file__).with_name('refined_exports.py')),'--input',str(source)]
        self.launch_worker(command,'export',source)
    def set_busy(self, value):
        self.busy = value
        for widget in self.edit_buttons:widget.configure(state='disabled' if value else 'normal')
        self.refine_button.configure(state='disabled' if value or not self.reconstruction else 'normal')
        if value:
            self.cancel_button.pack(side='right',padx=6);self.cancel_button.configure(state='normal');self.progress.pack(side='right',padx=10);self.progress.start(12)
        else:self.cancel_button.pack_forget();self.progress.stop();self.progress.pack_forget()
    def set_results(self, text):
        self.results_text.configure(state='normal'); self.results_text.delete('1.0', 'end'); self.results_text.insert('1.0', text); self.results_text.configure(state='disabled')
    def open_result(self):
        if not self.last_result or not self.can_edit() or not self.resolve_properties(): return
        try:
            component, cameras = self.read_alignment(self.last_result); self.install_alignment(self.last_result, component, cameras, self.image_dir)
            self.right_tabs.select(0); self.status.set('Inspecting refined result. Controls retained; training package unchanged.')
        except Exception as error: self.error('Cannot open result', error)
    def poll(self):
        if self.closed: return
        while True:
            try: event = self.result_queue.get_nowait()
            except queue.Empty: break
            if event[0] in ['image', 'image_error']:
                pane = self.panes[event[1]]
                if event[2] != pane['generation']: continue
                pane['loading'] = False
                if event[0] == 'image_error':
                    self.audit.event('image_error',image=pane['name'].get(),error=event[3])
                    pane['info'].set('Image unavailable'); pane['view'].overlay = 'Image unavailable · choose matching original images'
                    pane['view'].request_draw(); self.status.set(event[3]); continue
                _, _, _, name, asset, preserve = event; pane['loaded_name'] = name; pane['view'].set_asset(asset, preserve)
                pane['view'].overlay = None
                pane['info'].set(f'{self.image_names.index(name)+1} / {len(self.image_names)} · {asset.image.width} × {asset.image.height} · original pixels')
                if 'focus_after_load' in pane:
                    focus_name, pixel = pane.pop('focus_after_load')
                    if focus_name == name: pane['view'].focus(pixel)
                self.view_navigation(pane); self.draw()
            elif event[0] == 'refined':
                self.set_busy(False); _, output, report = event; self.last_result = output
                state = 'Converged' if report['converged'] else 'Evaluation budget reached — inspect before use'
                self.set_results(f"{state}\n\nBackend: {report['backend']}\nDevice: {report.get('device','CPU')}\nSolver peak VRAM: {report.get('peak_solver_vram_mib',0):.1f} MiB\n\nAngular RMS\n{report['before_angular_rms_deg']:.5f}° → {report['after_angular_rms_deg']:.5f}°\n\n{report['cameras']} cameras\n{report['optimized_points']} landmarks\n{report['observations']} observations\n{report['seconds']:.1f} seconds\n\n{report['termination']}\n\nScale: {report['scale']}\n\nIndependent checks\n{json.dumps(report['checks'], indent=2) if report['checks'] else 'No independent checks supplied'}\n\nNew alignment\n{output}\n\nReport\n{output.with_suffix('.report.json')}\n\nExports\n{json.dumps(report.get('exports',{}),indent=2)}\n\nFitted residual reduction does not prove absolute accuracy.")
                self.right_tabs.select(self.results_page); self.status.set(f'{state}. Saved: {output.name}')
            elif event[0] == 'refine_error':
                self.set_busy(False); self.set_results('Refinement failed\n\n'+event[1]); self.right_tabs.select(self.results_page); self.status.set('Refinement failed: '+event[1])
            elif event[0]=='native_scene':
                if event[1]==self.scene_generation:
                    self.scene_key=event[2];self.scene_cache=event[3]
                    for nav in self.native_views():nav.set_data(event[3]);nav.set_selected(self.active.get())
            elif event[0]=='solver_progress':
                info=event[1];phase=info['phase'];iteration=info.get('iteration',info.get('completed'));total=info.get('iterations',info.get('total',''))
                self.status.set(phase.capitalize()+(' '+str(iteration)+' / '+str(total) if iteration else '')+' · '+str(info.get('device',info.get('backend','')))+' · '+f"{info.get('elapsed',0):.1f}s")
            elif event[0]=='worker_cancelled':self.set_busy(False);self.status.set('Refinement cancelled. Source files preserved; inspect any completed outputs.')
            elif event[0]=='exports_ready':
                self.set_busy(False);self.set_results('Sparse cloud and camera alignment exported\n\n'+json.dumps(event[1],indent=2));self.right_tabs.select(self.results_page);self.status.set('Exports saved beside '+Path(event[1]['reconstruction']).name)
            elif event[0]=='viewer_ready':
                import webbrowser
                self.audit.event('browser_3d_open',url=event[1]);webbrowser.open(event[1]);self.status.set('Local 3D view: '+event[1]+' · click controls/cameras to select them here.')
            elif event[0]=='viewer_error':self.audit.event('viewer_error',error=event[1]);self.status.set('3D view unavailable: '+event[1])
            elif event[0]=='viewer_select':
                if self.document.point(event[1]):self.select(event[1])
            elif event[0]=='viewer_camera':self.browser_camera_selected=event[1];self.choose_image(event[1])
            elif event[0]=='viewer_connected':self.audit.event('browser_3d_connected',client=event[1]);self.status.set('Local browser 3D connected.')
        self.poll_job = self.window.after(60, self.poll)
    def close(self):
        if self.closed:return
        if self.busy: messagebox.showinfo('Refinement running', 'Wait for refinement to finish before closing.', parent=self.window); return
        if not self.unsaved_guard(): return
        from manual_ui_lifecycle import capture,retire
        retired=capture(self.window,[self,self.grid,self.prediction_browser,self.image_policy_panel,*[p['grid'] for p in self.panes],*[owner for nav in self.native_views() for owner in [nav,nav.compare,nav.editing]]])
        self.audit.event('editor_close');self.closed=True;self.scene_generation+=1;self.window.after_cancel(self.poll_job)
        if self.image_policy_panel and not self.image_policy_panel.closed:self.image_policy_panel.close()
        self.live_layout.close();self.error_cells.close()
        for nav in self.native_views():nav.close()
        if self.prediction_browser and not self.prediction_browser.closed:self.prediction_browser.close()
        for guide in self.guides: guide.hide()
        for pane in self.panes: pane['view'].close()
        if self.grid and not self.grid.closed:self.grid.close()
        for pane in self.panes:
            if pane['grid'] and not pane['grid'].closed:pane['grid'].close()
        if self.thumbnail_store:self.thumbnail_store.close()
        self.viewer_executor.shutdown(wait=False,cancel_futures=True)
        if self.viewer:self.viewer.close()
        self.executor.shutdown(wait=False, cancel_futures=True); self.window.destroy();retire(retired)
def open_editor(parent,on_attach=None): return ControlPointEditor(parent,on_attach)
