"""Paged local thumbnail browser with bounded cache and asynchronous decoding."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import queue
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw


class ImageGrid:
    def __init__(self, editor):
        self.editor = editor; self.closed = False; self.generation = 0; self.page = 0; self.selected = None
        self.cache = OrderedDict(); self.photos = {}; self.tiles = {}; self.results = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='Abu thumbnails'); self.futures=[]
        self.window=tk.Toplevel(editor.window);self.window.title('Panorama image browser');self.window.geometry('1050x760');self.window.minsize(650,500);self.window.configure(bg='#1a2024')
        top=tk.Frame(self.window,bg='#1a2024',padx=12,pady=10);top.pack(fill='x')
        self.search=tk.StringVar(master=self.window);self.filter=tk.StringVar(master=self.window,value='All images');self.target=tk.StringVar(master=self.window,value='Left' if editor.focused==0 else 'Right')
        editor.label(top,'Find image').pack(side='left');entry=editor.entry(top,self.search,20);entry.pack(side='left',padx=8)
        selector=ttk.Combobox(top,textvariable=self.filter,values=['All images','With selected point','Missing selected point'],state='readonly',style='CP.TCombobox',width=22);selector.pack(side='left',padx=8)
        editor.guide(selector,'Image filter','Filter by whether the active control point has a measurement in an image. This is not an automatic visibility prediction.')
        editor.label(top,'Open in').pack(side='left',padx=6)
        target=ttk.Combobox(top,textvariable=self.target,values=['Left','Right'],state='readonly',style='CP.TCombobox',width=7);target.pack(side='left')
        self.search.trace_add('write',lambda *_:self.refresh(reset=True));selector.bind('<<ComboboxSelected>>',lambda e:self.refresh(reset=True))
        holder=tk.Frame(self.window,bg='#111518');holder.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(holder,bg='#111518',highlightthickness=0);scroll=ttk.Scrollbar(holder,command=self.canvas.yview);self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right',fill='y');self.canvas.pack(fill='both',expand=True)
        self.body=tk.Frame(self.canvas,bg='#111518');self.item=self.canvas.create_window(0,0,anchor='nw',window=self.body)
        self.body.bind('<Configure>',lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>',self.resize);self.columns=4
        self.canvas.bind('<MouseWheel>',self.wheel)
        footer=tk.Frame(self.window,bg='#1a2024',padx=12,pady=10);footer.pack(fill='x',before=holder)
        editor.button(footer,'‹ Previous page',lambda:self.change_page(-1)).pack(side='left')
        editor.button(footer,'Next page ›',lambda:self.change_page(1)).pack(side='left',padx=6)
        self.info=tk.StringVar(master=self.window);editor.label(footer,variable=self.info,muted=True).pack(side='left',padx=8)
        editor.button(footer,'Open selected',self.open_selected,'Double-click a tile or open the selected image in the chosen pane.',primary=True).pack(side='right')
        self.window.bind('<Left>',lambda e:self.keyboard(-1,e));self.window.bind('<Right>',lambda e:self.keyboard(1,e));self.window.bind('<Return>',lambda e:self.open_selected())
        self.window.protocol('WM_DELETE_WINDOW',self.close);editor.style_dropdowns(self.window)
        self.refresh();self.poll_job=self.window.after(60,self.poll)
    def wheel(self,event): self.canvas.yview_scroll(-2 if event.delta>0 else 2,'units');return 'break'
    def resize(self,event):
        self.canvas.itemconfigure(self.item,width=event.width);columns=max(1,event.width//220)
        if columns!=self.columns:self.columns=columns;self.arrange()
    def arrange(self):
        for index,(name,(frame,label)) in enumerate(self.tiles.items()):frame.grid(row=index//self.columns,column=index%self.columns,padx=8,pady=8,sticky='n')
    def refresh(self,reset=False):
        if self.closed:return
        if reset:self.page=0
        self.generation+=1;generation=self.generation
        for future in self.futures:future.cancel()
        self.futures=[]
        point=self.editor.selected();observations={o['shot_id']:o['pixel'] for o in point['observations']} if point else {}
        names=[n for n in self.editor.image_names if self.search.get().casefold() in n.casefold()]
        if self.filter.get()=='With selected point':names=[n for n in names if n in observations]
        if self.filter.get()=='Missing selected point':names=[n for n in names if n not in observations]
        self.names=names;self.pages=max(1,(len(names)+23)//24);self.page=max(0,min(self.page,self.pages-1));visible=names[self.page*24:(self.page+1)*24]
        for frame,label in self.tiles.values():frame.destroy()
        self.tiles={};self.photos={};self.canvas.yview_moveto(0)
        for name in visible:
            frame=tk.Frame(self.body,bg='#242d32',highlightthickness=2,highlightbackground='#70dbc0' if name==self.selected else '#344149')
            label=tk.Label(frame,text='Loading…',width=25,height=7,bg='#101619',fg='#a0adb5');label.pack()
            caption=tk.Label(frame,text=name,bg='#242d32',fg='#e7edef',font=('Segoe UI',9),wraplength=195);caption.pack(fill='x',pady=4)
            tag=tk.Label(frame,text='Measured' if name in observations else 'Not measured',bg='#242d32',fg='#70dbc0' if name in observations else '#a0adb5',font=('Segoe UI',9));tag.pack(fill='x',pady=(0,4))
            self.tiles[name]=(frame,label)
            for widget in [frame,label,caption,tag]:
                widget.bind('<Button-1>',lambda e,n=name:self.select(n));widget.bind('<Double-1>',lambda e,n=name:self.select_and_open(n));widget.bind('<MouseWheel>',self.wheel)
            path=self.editor.image_dir/name;key=str(path)
            def decode(n=name,p=path,k=key,measurement=observations.get(name)):
                if self.closed or generation!=self.generation:return
                try:
                    if k in self.cache:thumb=self.cache[k].copy();self.cache.move_to_end(k)
                    else:
                        with Image.open(p) as source:
                            source.draft('RGB',(380,190));source.thumbnail((195,105),Image.Resampling.LANCZOS)
                            thumb=source.convert('RGB')
                        self.cache[k]=thumb.copy()
                        while len(self.cache)>128:self.cache.popitem(last=False)
                    if measurement:
                        w,h=self.editor.image_sizes[n];x,y=measurement[0]*thumb.width/w,measurement[1]*thumb.height/h
                        d=ImageDraw.Draw(thumb);d.ellipse((x-4,y-4,x+4,y+4),outline='#ffce73',width=2)
                    if not self.closed and generation==self.generation:self.results.put((generation,n,thumb,None))
                except Exception as error:self.results.put((generation,n,None,str(error)))
            self.futures.append(self.executor.submit(decode))
        self.arrange();self.info.set(f'{len(names)} images · page {self.page+1}/{self.pages}')
    def select(self,name):
        self.selected=name
        for n,(frame,label) in self.tiles.items():frame.configure(highlightbackground='#70dbc0' if n==name else '#344149')
    def select_and_open(self,name):self.select(name);self.open_selected()
    def open_selected(self):
        if self.selected not in self.names:
            if not self.names:return
            self.select(self.names[0])
        self.editor.choose_image(self.selected,0 if self.target.get()=='Left' else 1)
        self.editor.window.lift()
    def change_page(self,step):self.page+=step;self.refresh()
    def keyboard(self,step,event=None):
        if event and isinstance(event.widget,(tk.Entry,ttk.Combobox)):return
        if not self.names:return
        index=self.names.index(self.selected) if self.selected in self.names else -1
        index=max(0,min(len(self.names)-1,index+step));name=self.names[index]
        if index//24!=self.page:self.page=index//24;self.refresh()
        self.select(name);return 'break'
    def poll(self):
        if self.closed:return
        while True:
            try:generation,name,thumb,error=self.results.get_nowait()
            except queue.Empty:break
            if generation!=self.generation or name not in self.tiles:continue
            label=self.tiles[name][1]
            if error:label.configure(text='Image unavailable');continue
            photo=ImageTk.PhotoImage(thumb,master=self.window);self.photos[name]=photo;label.configure(image=photo,text='',width=195,height=105)
        self.poll_job=self.window.after(60,self.poll)
    def close(self):
        if self.closed:return
        self.closed=True;self.window.after_cancel(self.poll_job);self.executor.shutdown(wait=False,cancel_futures=True);self.photos.clear();self.window.destroy()
