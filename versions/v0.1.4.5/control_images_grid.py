"""One continuous image grid; virtualized tiles and bounded thumbnail memory."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
class ImageGrid:
    def __init__(self, editor):
        self.editor = editor; self.closed = False; self.generation = 0; self.selected = None
        self.cache = OrderedDict(); self.photos = {}; self.tiles = {}; self.results = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='Abu thumbnails'); self.futures=[]
        self.window=tk.Toplevel(editor.window);self.window.title('Panorama image browser');self.window.geometry('1050x760');self.window.minsize(650,500);self.window.configure(bg='#1a2024')
        top=tk.Frame(self.window,bg='#1a2024',padx=12,pady=10);top.pack(fill='x')
        self.search=tk.StringVar(master=self.window);self.filter=tk.StringVar(master=self.window,value='All images');self.target=tk.StringVar(master=self.window,value='Left' if editor.focused==0 else 'Right')
        editor.label(top,'Find image').pack(side='left');entry=editor.entry(top,self.search,20);entry.pack(side='left',padx=8)
        selector=ttk.Combobox(top,textvariable=self.filter,values=['All images','With selected point','Missing selected point','Predicted for selected point'],state='readonly',style='CP.TCombobox',width=22);selector.pack(side='left',padx=8)
        editor.guide(selector,'Image filter','Filter by whether the active control point has a measurement in an image. This is not an automatic visibility prediction.')
        editor.label(top,'Open in').pack(side='left',padx=6)
        target=ttk.Combobox(top,textvariable=self.target,values=['Left','Right'],state='readonly',style='CP.TCombobox',width=7);target.pack(side='left')
        self.search.trace_add('write',lambda *_:self.refresh(reset=True));selector.bind('<<ComboboxSelected>>',lambda e:self.refresh(reset=True))
        holder=tk.Frame(self.window,bg='#111518');holder.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(holder,bg='#111518',highlightthickness=0);scroll=ttk.Scrollbar(holder,command=self.scroll);self.canvas.configure(yscrollcommand=self.scrolled)
        scroll.pack(side='right',fill='y');self.canvas.pack(fill='both',expand=True)
        self.body=None;self.item=None
        self.canvas.bind('<Configure>',self.resize);self.columns=4
        self.canvas.bind('<MouseWheel>',self.wheel)
        footer=tk.Frame(self.window,bg='#1a2024',padx=12,pady=10);footer.pack(fill='x',before=holder)
        self.info=tk.StringVar(master=self.window);editor.label(footer,variable=self.info,muted=True).pack(side='left',padx=8)
        editor.button(footer,'Open selected',self.open_selected,'Double-click a tile or open the selected image in the chosen pane.',primary=True).pack(side='right')
        self.window.bind('<Left>',lambda e:self.keyboard(-1,e));self.window.bind('<Right>',lambda e:self.keyboard(1,e));self.window.bind('<Return>',lambda e:self.open_selected())
        self.window.protocol('WM_DELETE_WINDOW',self.close);editor.style_dropdowns(self.window)
        self.refresh();self.poll_job=self.window.after(60,self.poll)
    def scrolled(self,first,last):
        # Scrollbar updates can run during configure; debounce rendering.
        for widget in self.canvas.master.winfo_children():
            if isinstance(widget,ttk.Scrollbar):widget.set(first,last)
        self.schedule_visible()
    def schedule_visible(self):
        if self.closed:return
        if getattr(self,'visible_job',None):self.window.after_cancel(self.visible_job)
        self.visible_job=self.window.after(25,self.arrange)
    def scroll(self,*args):self.canvas.yview(*args);self.schedule_visible()
    def wheel(self,event):self.canvas.yview_scroll(-2 if event.delta>0 else 2,'units');self.schedule_visible();return 'break'
    def resize(self,event):
        columns=max(1,event.width//220)
        if columns!=self.columns:self.columns=columns;self.refresh()
        else:self.schedule_visible()
    def refresh(self,reset=False):
        if self.closed:return
        self.generation+=1
        for future in self.futures:future.cancel()
        self.futures=[]
        point=self.editor.selected();self.observations={o['shot_id']:o['pixel'] for o in point['observations']} if point else {}
        self.predictions={}
        names=[n for n in self.editor.image_names if self.search.get().casefold() in n.casefold()]
        if self.filter.get()=='With selected point':names=[n for n in names if n in self.observations]
        if self.filter.get()=='Missing selected point':names=[n for n in names if n not in self.observations]
        if self.filter.get()=='Predicted for selected point':
            from control_point_predictions import predicted_candidates
            candidates,_=predicted_candidates(point,self.editor.cameras);self.predictions={p['image']:p['pixel'] for p in candidates}
            allowed=set(names);names=[p['image'] for p in candidates if p['image'] in allowed]
        self.names=names
        for frame,label in self.tiles.values():frame.destroy()
        self.canvas.delete('all');self.tiles={};self.tile_items={};self.photos={}
        rows=max(1,(len(names)+self.columns-1)//self.columns)
        self.canvas.configure(scrollregion=(0,0,max(1,self.canvas.winfo_width()),rows*165),yscrollincrement=35)
        if reset:self.canvas.yview_moveto(0)
        self.info.set(f'{len(names)} images · one continuous grid · thumbnails load as you scroll')
        self.arrange()
    def arrange(self):
        self.visible_job=None
        if self.closed:return
        generation=self.generation
        start=max(0,int(self.canvas.canvasy(0)//165)-1)*self.columns
        end=min(len(self.names),(int((self.canvas.canvasy(0)+max(300,self.canvas.winfo_height()))//165)+2)*self.columns)
        visible=set(self.names[start:end])
        for name in list(self.tiles):
            if name not in visible:
                frame,label=self.tiles.pop(name);self.canvas.delete(self.tile_items.pop(name));frame.destroy();self.photos.pop(name,None)
        if not hasattr(self,'tile_items'):self.tile_items={}
        for index in range(start,end):
            name=self.names[index]
            if name in self.tiles:continue
            frame=tk.Frame(self.canvas,bg='#242d32',highlightthickness=2,highlightbackground='#70dbc0' if name==self.selected else '#344149')
            label=tk.Label(frame,text='Loading…',width=25,height=7,bg='#101619',fg='#a0adb5');label.pack()
            caption=tk.Label(frame,text=name,bg='#242d32',fg='#e7edef',font=('Segoe UI',9),wraplength=195);caption.pack(fill='x',pady=3)
            tag=tk.Label(frame,text='Measured' if name in self.observations else 'Prediction · verify' if name in self.predictions else 'Not measured',bg='#242d32',fg='#70dbc0' if name in self.observations else '#a0adb5',font=('Segoe UI',9));tag.pack(fill='x',pady=(0,3))
            self.tiles[name]=(frame,label);self.tile_items[name]=self.canvas.create_window((index%self.columns)*220+8,(index//self.columns)*165+8,anchor='nw',window=frame,width=205,height=151)
            for widget in [frame,label,caption,tag]:
                widget.bind('<Button-1>',lambda e,n=name:self.select(n));widget.bind('<Double-1>',lambda e,n=name:self.select_and_open(n));widget.bind('<MouseWheel>',self.wheel)
            path=self.editor.image_dir/name;key=str(path);measurement=self.observations.get(name,self.predictions.get(name))
            def decode(n=name,p=path,k=key,measurement=measurement):
                if self.closed or generation!=self.generation:return
                try:
                    if k in self.cache:thumb=self.cache[k].copy();self.cache.move_to_end(k)
                    else:
                        with Image.open(p) as source:
                            source.draft('RGB',(380,190));source.thumbnail((195,105),Image.Resampling.LANCZOS);thumb=source.convert('RGB')
                        self.cache[k]=thumb.copy()
                        while len(self.cache)>128:self.cache.popitem(last=False)
                    if measurement:
                        w,h=self.editor.image_sizes[n];x,y=measurement[0]*thumb.width/w,measurement[1]*thumb.height/h
                        ImageDraw.Draw(thumb).ellipse((x-4,y-4,x+4,y+4),outline='#ffce73' if n in self.observations else '#78b4ff',width=2)
                    if not self.closed and generation==self.generation:self.results.put((generation,n,thumb,None))
                except Exception as error:self.results.put((generation,n,None,str(error)))
            self.futures.append(self.executor.submit(decode))
        self.futures=[f for f in self.futures if not f.done()]
    def select(self,name):
        self.selected=name
        for n,(frame,label) in self.tiles.items():frame.configure(highlightbackground='#70dbc0' if n==name else '#344149')
    def select_and_open(self,name):self.select(name);self.open_selected()
    def open_selected(self):
        if self.selected not in self.names:
            if not self.names:return
            self.select(self.names[0])
        index=0 if self.target.get()=='Left' else 1
        self.editor.choose_image(self.selected,index)
        if self.selected in self.predictions:self.editor.panes[index]['focus_after_load']=(self.selected,self.predictions[self.selected])
        self.editor.window.lift()
    def keyboard(self,step,event=None):
        if event and isinstance(event.widget,(tk.Entry,ttk.Combobox)):return
        if not self.names:return
        index=self.names.index(self.selected) if self.selected in self.names else -1
        index=max(0,min(len(self.names)-1,index+step));name=self.names[index]
        rows=max(1,(len(self.names)+self.columns-1)//self.columns)
        self.canvas.yview_moveto((index//self.columns)/rows);self.arrange()
        self.select(name);return 'break'
    def poll(self):
        if self.closed:return
        while True:
            try:generation,name,thumb,error=self.results.get_nowait()
            except queue.Empty:break
            if generation!=self.generation or name not in self.tiles:continue
            label=self.tiles[name][1]
            if error:
                self.editor.audit.event('thumbnail_error',image=name,error=error);label.configure(text='Image unavailable');continue
            photo=ImageTk.PhotoImage(thumb,master=self.window);self.photos[name]=photo;label.configure(image=photo,text='',width=195,height=105)
        self.poll_job=self.window.after(60,self.poll)
    def close(self):
        if self.closed:return
        self.closed=True
        if getattr(self,'visible_job',None):self.window.after_cancel(self.visible_job)
        self.window.after_cancel(self.poll_job);self.executor.shutdown(wait=False,cancel_futures=True);self.photos.clear();self.window.destroy()
