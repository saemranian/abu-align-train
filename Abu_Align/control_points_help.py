"""Illustrated, searchable task guide; diagrams are generated locally."""
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image,ImageDraw,ImageFont,ImageTk
from ui_icons import line_icon

TOPICS={
 'start':('Start here','folder',[
  ('1 · Open your original images','folder','Open a native reconstruction JSON or calibrated COLMAP model. For 360 use original full 2:1 images; for perspective choose Perspective in Settings before preparing points without alignment. VGGT-Omega camera calibration and tracks are imported locally.','workflow','Open alignment','open'),
  ('2 · Mark the same feature','target','Add a point, then click the same physical corner in two or more translated views. Three to six well-separated views help reveal mistaken measurements.','measure','Add a point','new'),
  ('3 · Review, save, refine','save','Check the observations and angular errors. Save controls, choose hardware and budget, then Refine the open alignment. Main Align runs registration first and applies saved controls afterward.','refine','Refinement settings','settings')]),
 'images':('Find images & predictions','camera',[
  ('Choose the content of each view','compare','The selector beside Left and Right independently switches Image, Grid or 3D. Use Grid beside Image to measure quickly, two Images to compare, or 3D beside Image to inspect a camera. Zoom and the existing M54 viewpoint are retained when switching.','views',None,None),
  ('Keep the image grid beside you','camera','Search or filter the continuous grid, choose Left or Right, and double-click a thumbnail. The grid stays open, at the same scroll position. Close it only with its own window close button.','grid','Open image grid','grid'),
  ('Jump to likely views','target','After two reliable measurements, Predicted images lists unmeasured views ranked near supporting cameras. Open & focus a prediction, inspect the actual feature, then click it yourself. Occlusion is unknown.','predictions','Predicted images','predictions'),
  ('Search visually across all originals','search','Predicted images → Visual search examines every enabled original in all scope, including unregistered images. Local ALIKED/LightGlue validates neighbourhood matches; SIFT is the fallback. Set up local AI model downloads only public weights. Inference has network access disabled. Verify every proposal; ambiguity and one-view support are labelled.','predictions','Predicted images','predictions'),
  ('Choose images and mask layers','sliders','Project → Images & masks enables/disables originals and assigns masks. Black ignores, white keeps; invert is explicit. Dimensions must match originals. Apply validates, then Save controls persists. Main Align filters images before engine input; Refine filters cameras and measured rays. Masked/disabled points need two remaining usable views.','grid','Images & masks','image_policy'),
  ('Measurements and predictions','target','Yellow thumbnail markers are saved measurements. Blue markers are geometric guidance. A predicted location is never saved automatically.','measure',None,None)]),
 'navigation':('Zoom, pan & shortcuts','search',[
  ('Resize the panels','sliders','Drag the visible divider handles to resize the point list, image views and properties/settings. The Properties divider adjusts the measurement-table height. Settings → Pane resizing offers Auto, Fast and Live. Auto uses a responsive guide in large windows; release commits the size. Project → Reset pane sizes restores the starting layout.','views','Refinement settings','settings'),
  ('Navigate the panorama','search','Wheel zooms around your cursor. Middle-drag or Space + left-drag pans. F fits the panorama; 1 restores original pixel scale. Ctrl+F focuses the selected measurement/prediction; in a 3D pane it focuses the selected object. Wrap panorama seam lets you work across the left/right boundary.','mouse',None,None),
  ('Browse or adjust precisely','sliders','Left/Right browses images in the active pane. Ctrl+Arrow nudges a measurement by 0.1 px; Shift+Arrow nudges 10 px. Disable arrow browsing for plain-arrow 1 px nudges. Drag an existing marker to correct it.','measure','Navigation settings','settings'),
  ('Undo and save','save','Ctrl+Z undoes a point, measurement, property or distance edit. Ctrl+Y redoes it. Ctrl+S saves the controls. Right-click a marker removes that measurement.','workflow',None,None)]),
 'points':('Point types & CSV','target',[
  ('Choose the right role','target','Tie: same feature in several images. Ground control: surveyed local XYZ used by refinement. Check: surveyed XYZ withheld from optimization for independent evaluation.','points','Point properties','properties'),
  ('Give measurements more room','sliders','Drag the horizontal divider above Measurements upward to make the table taller. The point properties above it remain scrollable; measurement actions stay reachable. Column widths also resize.','grid','Point properties','properties'),
  ('Read errors at a glance','activity','Angular error numbers are green through 0.10°, yellow through 0.50° and red above 0.50°. Settings changes the limits; Save controls retains them. A dash means unavailable. Colours show ray consistency, not surveyed accuracy. Check red observations before tightening uncertainty.','errors','Error colour settings','settings'),
  ('Survey and distance constraints','sliders','Use realistic uncertainty. Ground control needs at least three non-collinear measured points to establish a frame. A known distance can establish scale. Check points cannot be distance endpoints.','points',None,None),
  ('Import existing points','folder','Import CSV previews mapped image measurements or local XYZ and source units. Pixels must refer to the same original images, projection and resolution. Perspective CSV measurements can be used directly on those perspective originals; transferring them onto a panorama requires importing names and re-marking.','grid','Import CSV','csv')]),
 'refine':('Refine & export','activity',[
  ('Continue from the latest good result','compare','Open the latest GOOD refined JSON with the same original images and saved controls. Add or correct measurements, then refine again. Start from the initial model for comparison or recovery. Repeating unchanged converged constraints cannot keep improving accuracy.','compare',None,None),
  ('Tune evidence, then budgets','sliders','Use realistic pixel/survey uncertainty and distributed independent checks. Start with soft_l1, transition 1 and 10000 tracks. Default minimum parallax is 1°. Try 3–4 automatic views only if coverage remains. Increase budgets when convergence/check reports justify it. Tight tolerances do not correct wrong observations. Perspective intrinsics stay fixed.','refine','Refinement settings','settings'),
  ('Run without freezing the editor','activity','Auto uses CUDA for larger supported jobs; CPU uses analytic derivatives. The separate worker reports progress and supports cancellation. Higher budgets may improve the fit, but reaching a limit is not convergence.','refine','Refinement settings','settings'),
  ('Find the corrected outputs','save','A successful solve writes a new JSON, controls snapshot, report, sparse PLY and native camera CSV. Open result inspects the new model; Export cloud & camera CSV also works on an older saved result.','exports','Results & outputs','results'),
  ('Use the full pipeline','play','Use with main Align saves and attaches controls. The main Align button registers images, refines, and creates the prepared export/training dataset. Editor-only refinement does not replace an existing training package.','workflow',None,None)]),
 'compare':('3D & reconstruction comparison','compare',[
  ('Navigate and select in Abu M54','cube','Drag to orbit, right/middle-drag to pan, wheel to zoom. Click a control, camera or cloud sample. Double-click, press O or choose Orbit selected to orbit around it. Fit restores a scene-centred view.','mouse','Open 3D navigator','3d'),
  ('Grid and model transformations','sliders','M54 tools opens ground grid, XY/XZ/YZ plane, elevation, shading, view bookmarks and orbit-on-selection. Move, rotate XYZ and uniformly scale the whole reconstruction. Undo/Redo/Reset changes the preview; Export transformed copy preserves the source. Survey coordinates in a separate control file are not transformed by export.','compare','Open 3D navigator','3d'),
  ('Load a read-only comparison','compare','A is the alignment open in the control-point editor. Open Compare reconstructions and Load B, or choose a completed run folder. Show A only, B only or Overlay. Switch A/B keeps the viewpoint. Controls belong to A.','compare','Compare reconstructions','compare'),
  ('Choose a common frame deliberately','sliders','Raw frames are shown first. Fit B to A uses matching camera names and at least three non-collinear centres. Allow scale fits a similarity; switch it off for rigid fitting. This changes the preview only.','compare',None,None),
  ('Interpret differences carefully','activity','Camera offsets and position/orientation differences show agreement. B distance to A colours sampled nearest-point distances; point density affects them. These are not ground-truth accuracy. Save a comparison report for troubleshooting.','compare',None,None)]),
 'troubleshoot':('Troubleshooting & limits','document',[
  ('Missing or unreliable predictions','target','Add translated views with clear parallax. Correct observations with high residuals. Reflections, repeated patterns, moving objects and stitching artefacts can create misleading matches.','predictions',None,None),
  ('A comparison will not fit','compare','Check matching image names and choose a single component. Collinear camera centres leave fitting ambiguous. Keep raw coordinates to inspect a metric change without fitting it away. A cloud-only PLY has no camera matches.','compare',None,None),
  ('Find the local session log','document','Results → Open troubleshooting log records edits, imports, settings, solver progress, errors and outputs. Images stay on this PC. No cubemap conversion is used by this tool.','logs','Open troubleshooting log','log')])
}

from manual_refinement_settings import ORIGINS,guide as parameter_guide
for key,title in [('backend','Processing hardware'),('loss','Robust loss'),('max_points','Automatic landmarks'),('iterations','Solver budget'),('sigma_px','New measurement uncertainty'),('marker_size','Marker radius'),('loss_scale','Robust transition'),('auto_sigma_px','Automatic track uncertainty'),('min_track_views','Minimum automatic views'),('min_angle_deg','Minimum automatic parallax'),('ftol','Objective tolerance'),('xtol','Step tolerance'),('gtol','Gradient tolerance')]:
    TOPICS['refine'][2].append((title,'sliders',parameter_guide(key),'refine','Refinement settings','settings'))


def illustration(kind):
    """Small antialiased explanatory diagrams, not screenshots or real data."""
    s=2;image=Image.new('RGB',(668,256),'#11181c');d=ImageDraw.Draw(image)
    try:font=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',24)
    except OSError:font=ImageFont.load_default()
    def rect(bounds,color='#344149',fill='#1a242a'):d.rounded_rectangle(tuple(v*s for v in bounds),radius=8*s,outline=color,fill=fill,width=2*s)
    def line(points,color='#70dbc0',width=2):d.line([(x*s,y*s) for x,y in points],fill=color,width=width*s)
    def text(x,y,value,color='#dce8e9'):d.text((x*s,y*s),value,font=font,fill=color)
    def dot(x,y,color='#ffce73',radius=5):d.ellipse(((x-radius)*s,(y-radius)*s,(x+radius)*s,(y+radius)*s),outline=color,width=2*s)
    if kind=='measure':
        for x,label in [(12,'Image A'),(180,'Image B')]:
            rect((x,26,x+140,103));text(x+9,33,label);line([(x+20,91),(x+65,62),(x+107,94)],'#465e68');dot(x+65,62)
        line([(82,62),(246,62)],'#ffce73',1);text(105,5,'Same feature');text(82,106,'Two views · one point')
    elif kind=='views':
        for x,label in [(12,'Left · Grid'),(180,'Right · Image')]:
            rect((x,12,x+140,105));text(x+8,20,label);line([(x+8,48),(x+132,48)],'#465e68',1)
        for i in range(6):
            x=22+i%3*42;y=58+i//3*21;rect((x,y,x+34,y+16))
        line([(192,88),(222,64),(247,94),(294,70)],'#465e68');dot(247,94);text(70,108,'Image / Grid / 3D · each view')
    elif kind=='errors':
        for i,(value,color) in enumerate([('0.050°','#70dbc0'),('0.300°','#ffce73'),('1.000°','#ff7777')]):
            y=10+i*34;rect((22,y,311,y+30));text(34,y+3,'Image '+str(i+1));text(200,y+3,value,color)
        text(50,112,'Review high residuals first')
    elif kind=='grid' or kind=='predictions':
        for i in range(6):
            x=12+(i%3)*108;y=12+(i//3)*53;rect((x,y,x+96,y+43),'#70dbc0' if i==1 else '#344149')
            line([(x+8,y+33),(x+35,y+22),(x+57,y+36),(x+84,y+18)],'#465e68',1)
            if i in [1,3]:dot(x+40,y+23,'#78b4ff' if kind=='predictions' else '#ffce73',4)
        text(35,112,'Verify predictions' if kind=='predictions' else 'Double-click · grid stays open')
    elif kind=='compare':
        for i in range(18):
            x=30+(i%6)*24;y=30+(i//6)*22
            dot(x,y,'#70dbc0',2);dot(x+12,y+5,'#ff9b7f',2)
            if i%5==0:line([(x,y),(x+12,y+5)],'#ffce73',1)
        rect((204,16,321,51));text(216,25,'A · mint','#70dbc0');rect((204,62,321,97));text(216,71,'B · coral','#ff9b7f');text(31,110,'Same viewpoint · compare differences')
    elif kind=='mouse':
        rect((23,17,85,108));line([(54,19),(54,59)],'#a0adb5');line([(25,60),(83,60)],'#a0adb5');rect((46,27,62,48),'#70dbc0')
        text(105,15,'Wheel → zoom');text(105,47,'Drag → navigate');text(105,79,'F → fit the view')
    elif kind=='points':
        for x,label,role,color in [(57,'Tie','Feature','#70dbc0'),(167,'GCP','Constrain','#78b4ff'),(278,'Check','Validate','#ff7d7d')]:
            dot(x,37,color,16);text(x-18,64,label,color);text(x-26,90,role)
    elif kind=='refine':
        rect((16,26,115,93));text(33,34,'GPU / CPU');rect((137,26,317,93));text(151,34,'Separate worker');line([(115,61),(137,61)])
        for i in range(5):rect((151+i*29,67,172+i*29,78),'#70dbc0' if i<4 else '#344149')
        text(59,106,'Progress · cancel · review results')
    elif kind=='exports':
        for x,label in [(15,'JSON'),(128,'PLY'),(241,'CSV')]:rect((x,20,x+77,96));text(x+15,45,label);line([(x+13,77),(x+60,77)],'#70dbc0')
        text(34,108,'Alignment · sparse cloud · camera poses')
    elif kind=='logs':
        rect((31,10,302,113));text(49,22,'Local troubleshooting log')
        for y,color in [(55,'#70dbc0'),(76,'#78b4ff'),(97,'#ffce73')]:dot(52,y,color,3);line([(68,y),(274,y)],'#465e68')
    else:
        for x,title in [(10,'1 · Open'),(122,'2 · Mark'),(234,'3 · Refine')]:rect((x,31,x+90,92));text(x+7,51,title)
        line([(100,61),(122,61)]);line([(212,61),(234,61)]);text(67,105,'Original panoramas throughout')
    return image.resize((334,128),Image.Resampling.LANCZOS)


class HelpExplorer(tk.Frame):
    def __init__(self,editor,parent,reference):
        super().__init__(parent,bg='#1a2024');self.editor=editor;self.topic='start';self.cards=[];self.images={};self.columns=1
        notebook=ttk.Notebook(self,style='CP.TNotebook');notebook.pack(fill='both',expand=True)
        guide=tk.Frame(notebook,bg='#1a2024');reference_page=tk.Frame(notebook,bg='#1a2024');notebook.add(guide,text='Illustrated guide');notebook.add(reference_page,text='Full reference')
        self.reference=ScrolledText(reference_page,wrap='word',bg='#1a2024',fg='#e7edef',font=('Segoe UI',11),relief='flat',padx=24,pady=18,spacing3=8)
        self.reference.pack(fill='both',expand=True);self.reference.insert('1.0',reference);self.reference.configure(state='disabled')
        side=tk.Frame(guide,bg='#1a2024',width=230,padx=12,pady=14);side.pack(side='left',fill='y');side.pack_propagate(False)
        editor.label(side,'Learn by task',pady=8).pack(fill='x');self.search=tk.StringVar(self);entry=editor.entry(side,self.search);entry.pack(fill='x',pady=(0,12))
        editor.guide(entry,'Find a guide','Search topics by task: grid, prediction, survey, compare, export or log.')
        self.buttons={}
        for key,(title,icon,cards) in TOPICS.items():
            button=editor.button(side,title,lambda k=key:self.select(k));button.configure(image=line_icon(button,icon,size=24),compound='left',anchor='w',wraplength=170,justify='left');button.pack(fill='x',pady=4);self.buttons[key]=button
        self.no_results=editor.label(side,'No matching topic',muted=True)
        self.search.trace_add('write',self.filter_topics)
        holder=tk.Frame(guide,bg='#111518');holder.pack(side='left',fill='both',expand=True)
        self.canvas=tk.Canvas(holder,bg='#111518',highlightthickness=0);scroll=ttk.Scrollbar(holder,command=self.canvas.yview);self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');self.canvas.pack(fill='both',expand=True)
        self.body=tk.Frame(self.canvas,bg='#111518',padx=14,pady=14);self.item=self.canvas.create_window(0,0,anchor='nw',window=self.body)
        self.body.bind('<Configure>',lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')));self.canvas.bind('<Configure>',self.resize)
        self.select('start')
    def filter_topics(self,*_):
        query=self.search.get().casefold();matches=[]
        for key,(title,icon,cards) in TOPICS.items():
            haystack=title+' '+' '.join(c[0]+' '+c[2] for c in cards);button=self.buttons[key];button.pack_forget()
            if query in haystack.casefold():button.pack(fill='x',pady=4);matches.append(key)
        self.no_results.pack_forget()
        if not matches:self.no_results.pack(fill='x',pady=8)
        elif self.topic not in matches:self.select(matches[0])
    def select(self,key):
        self.topic=key;self.cards=[]
        for child in self.body.winfo_children():child.destroy()
        for name,button in self.buttons.items():button.configure(bg='#344149' if name==key else '#242d32')
        for index,(title,icon,body,art,action,target) in enumerate(TOPICS[key][2]):
            card=tk.Frame(self.body,bg='#1a2024',padx=14,pady=12,highlightthickness=1,highlightbackground='#344149')
            heading=self.editor.label(card,title);heading.configure(image=line_icon(heading,icon,color='#70dbc0',size=28),compound='left',font=('Segoe UI',11,'bold'));heading.pack(fill='x',pady=(0,10))
            if art not in self.images:self.images[art]=ImageTk.PhotoImage(illustration(art),master=self)
            picture=tk.Label(card,image=self.images[art],bg='#11181c',anchor='w');picture.pack(fill='x',pady=(0,10))
            text=self.editor.label(card,body,wraplength=600);text.pack(fill='x')
            if action:self.editor.button(card,action,lambda t=target:self.act(t),primary=True).pack(anchor='w',pady=(12,0))
            self.cards.append((card,heading,text))
        self.arrange();self.canvas.yview_moveto(0)
        def wheel(event):self.canvas.yview_scroll(-2 if event.delta>0 else 2,'units');return 'break'
        def bind(widget):
            widget.bind('<MouseWheel>',wheel)
            for child in widget.winfo_children():bind(child)
        bind(self.body)
    def resize(self,event):
        self.canvas.itemconfigure(self.item,width=event.width);self.columns=2 if event.width>=840 else 1;self.arrange()
    def arrange(self):
        available=max(370,self.canvas.winfo_width()-36);width=max(330,available//self.columns-40)
        for column in range(2):self.body.columnconfigure(column,weight=1 if column<self.columns else 0)
        for index,(card,heading,text) in enumerate(self.cards):
            card.grid(row=index//self.columns,column=index%self.columns,sticky='new',padx=7,pady=7);heading.configure(wraplength=width);text.configure(wraplength=width)
    def act(self,target):
        e=self.editor
        if target in ['settings','properties','results']:
            e.tabs.select(e.workspace);e.right_tabs.select({'properties':0,'settings':1,'results':2}[target])
        elif target=='compare':
            e.open_3d()
            if e.native_3d:e.native_3d.compare.show()
        else:getattr(e,{'open':'open','new':'new','grid':'open_grid','predictions':'open_predictions','image_policy':'open_image_policy','csv':'import_csv','3d':'open_3d','log':'open_log'}[target])()
    def destroy(self):
        self.images.clear();self.cards=[];super().destroy()
