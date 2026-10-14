"""Reusable continuous image grid for a pane or popup; cached tiles never reload."""
import tkinter as tk
from tkinter import ttk
from control_thumbnail_store import ThumbnailStore


class ImageGrid:
    def __init__(self,editor,parent=None,pane_index=None):
        self.editor=editor;self.closed=False;self.generation=0;self.selected=None;self.embedded=parent is not None;self.pane_index=pane_index
        self.photos={};self.tiles={};self.tile_items={};self.pool=[];self.visible_job=None;self.columns=1;self.names=[];self.keys={};self.futures=[];self.scroll_target=None;self.layout_signature=None;self.layout_passes=0
        if getattr(editor,'thumbnail_store',None) is None:editor.thumbnail_store=ThumbnailStore(editor)
        self.store=editor.thumbnail_store;self.store.listeners.add(self);self.cache=self.store.cache
        self.tile_height=max(151,int(125+36*editor.window.winfo_fpixels('1i')/96));self.row_height=self.tile_height+14
        self.window=tk.Frame(parent,bg='#1a2024') if self.embedded else tk.Toplevel(editor.window)
        if self.embedded:self.window.pack(fill='both',expand=True)
        else:
            self.window.title('Image grid · Manual control points');self.window.geometry('1050x760');self.window.minsize(480,400);self.window.configure(bg='#1a2024');self.window.transient(editor.window)
        top=tk.Frame(self.window,bg='#1a2024',padx=8,pady=6);top.pack(fill='x');top.columnconfigure(0,weight=1)
        self.search=tk.StringVar(self.window);self.filter=tk.StringVar(self.window,'All images')
        self.target=tk.StringVar(self.window,'Right' if pane_index==0 else 'Left' if pane_index==1 or editor.focused==0 else 'Right')
        entry=editor.entry(top,self.search,14);entry.grid(row=0,column=0,sticky='ew',padx=(0,4));editor.guide(entry,'Find image','Filter filenames. Clear to browse all original images.')
        target=ttk.Combobox(top,textvariable=self.target,values=['Left','Right'],state='readonly',style='CP.TCombobox',width=6);target.grid(row=0,column=1,sticky='e');editor.guide(target,'Open selected image in','Choose the destination pane. Embedded grids default to the other pane, keeping the grid available.')
        selector=ttk.Combobox(top,textvariable=self.filter,values=['All images','With selected point','Missing selected point','Predicted for selected point'],state='readonly',style='CP.TCombobox',width=16);selector.grid(row=1,column=0,columnspan=2,sticky='ew',pady=(5,0))
        editor.guide(selector,'Image filter','Measured/missing refer to the selected point. Predicted projections remain unconfirmed until visually verified.')
        self.search.trace_add('write',lambda *_:self.refresh(reset=True));selector.bind('<<ComboboxSelected>>',lambda e:self.refresh(reset=True))
        footer=tk.Frame(self.window,bg='#1a2024',padx=8,pady=4);footer.pack(side='bottom',fill='x')
        self.info=tk.StringVar(self.window);editor.label(footer,variable=self.info,muted=True,wraplength=190).pack(fill='x')
        editor.button(footer,'Open selected',self.open_selected,'Double-click or press Enter to open in the chosen image pane.',primary=True).pack(fill='x',pady=3)
        holder=tk.Frame(self.window,bg='#111518');holder.pack(fill='both',expand=True)
        self.canvas=tk.Canvas(holder,bg='#111518',highlightthickness=0,takefocus=True);self.scrollbar=ttk.Scrollbar(holder,command=self.schedule_scroll);self.canvas.configure(yscrollcommand=self.scrolled)
        self.scrollbar.pack(side='right',fill='y');self.canvas.pack(fill='both',expand=True);self.body=self.item=None
        self.canvas.bind('<Configure>',self.resize);self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<Button-1>',lambda e:self.clicked(e));self.canvas.bind('<Double-1>',lambda e:self.clicked(e,True))
        key_target=self.canvas if self.embedded else self.window
        for key in ['Left','Right','Up','Down']:key_target.bind('<'+key+'>',lambda e,k=key:self.keyboard(-self.columns if k=='Up' else self.columns if k=='Down' else -1 if k=='Left' else 1,e))
        key_target.bind('<Return>',lambda e:self.open_selected())
        if not self.embedded:self.window.protocol('WM_DELETE_WINDOW',self.close)
        editor.style_dropdowns(self.window);self.refresh()
    def scrolled(self,first,last):self.scrollbar.set(first,last);self.schedule_visible()
    def schedule_visible(self):
        if self.closed:return
        if self.scroll_target is None and self.layout_signature==self.viewport_signature():return
        if not self.closed and self.visible_job is None:self.visible_job=self.window.after_idle(self.scroll_frame)
    def schedule_scroll(self,*args):
        if self.closed:return
        extent=max(1,((len(self.names)+self.columns-1)//self.columns)*self.row_height)
        if args[0]=='moveto':target=float(args[1])*extent
        else:
            base=self.canvas.canvasy(0) if self.scroll_target is None else self.scroll_target
            target=base+float(args[1])*(35 if args[2]=='units' else self.canvas.winfo_height()*.9)
        self.scroll_target=max(0,min(max(0,extent-self.canvas.winfo_height()),target));self.schedule_visible()
    def scroll_frame(self):
        self.visible_job=None
        if self.closed:return
        if self.scroll_target is not None:
            extent=max(1,((len(self.names)+self.columns-1)//self.columns)*self.row_height);target,self.scroll_target=self.scroll_target,None;self.canvas.yview_moveto(target/extent)
        self.arrange()
    def scroll(self,*args):self.scroll_target=None;self.canvas.yview(*args);self.arrange()
    def wheel(self,event):self.schedule_scroll('scroll',-2*event.delta/120,'units');return 'break'
    def resize(self,event):
        columns=max(1,event.width//220)
        if columns!=self.columns:self.columns=columns;self.update_region()
        self.schedule_visible()
    def update_region(self):
        rows=max(1,(len(self.names)+self.columns-1)//self.columns)
        self.canvas.configure(scrollregion=(0,0,max(1,self.canvas.winfo_width()),rows*self.row_height),yscrollincrement=1)
    def refresh(self,reset=False):
        if self.closed:return
        self.generation+=1;point=self.editor.selected();self.observations={o['shot_id']:o['pixel'] for o in point['observations']} if point else {};self.predictions={}
        names=[n for n in self.editor.image_names if self.search.get().casefold() in n.casefold()]
        if self.filter.get()=='With selected point':names=[n for n in names if n in self.observations]
        if self.filter.get()=='Missing selected point':names=[n for n in names if n not in self.observations]
        if self.filter.get()=='Predicted for selected point':
            from control_point_predictions import predicted_candidates
            candidates,_=predicted_candidates(self.editor.effective_point(registered_only=True),self.editor.cameras,self.editor.controls.get('image_policy'));self.predictions={p['image']:p['pixel'] for p in candidates};allowed=set(names);names=[p['image'] for p in candidates if p['image'] in allowed]
            browser=self.editor.prediction_browser
            if browser and not browser.closed and browser.snapshot==browser.signature():
                self.predictions.update({n:p['pixel'] for n,p in browser.visual.items()});names=list(dict.fromkeys(names+[n for n in browser.visual if n in allowed]))
        self.names=names;self.keys={n:self.store.key(self.editor.image_dir/n,self.editor.image_sizes[n]) for n in names};self.update_region()
        if reset:self.canvas.yview_moveto(0)
        self.info.set(f'{len(names)} images · continuous grid');self.arrange()
        if len(names)<=self.store.limit:
            for n in names:self.store.request(self.editor.image_dir/n,self.editor.image_sizes[n],priority=2)
    def make_tile(self):
        c=self.canvas
        tile=dict(name=None,item=c.create_rectangle(0,0,1,1,fill='#242d32',outline='#344149',width=2),image=c.create_image(0,0,image=self.store.placeholder,anchor='n'),caption=c.create_text(0,0,fill='#e7edef',font=('Segoe UI',9),anchor='n'),tag=c.create_text(0,0,font=('Segoe UI',8),anchor='s'),error=c.create_text(0,0,fill='#ff7777',font=('Segoe UI',9)),signature=None,photo=None,visible=False)
        return tile
    def clicked(self,event,open_image=False):
        index=int(self.canvas.canvasy(event.y)//self.row_height)*self.columns+min(self.columns-1,int(self.canvas.canvasx(event.x)/max(1,self.canvas.winfo_width()/self.columns)))
        if not 0<=index<len(self.names):return 'break'
        if open_image:return self.select_and_open(self.names[index])
        self.select(self.names[index]);return 'break'
    def paint(self,name,tile):
        key=self.keys[name];width=max(48,min(195,int(self.canvas.winfo_width()/self.columns)-20));photo=self.store.photo(key,self.observations.get(name,self.predictions.get(name)),name in self.observations,width);error=self.store.errors.get(key)
        if error:
            self.canvas.itemconfigure(tile['image'],image='');self.canvas.itemconfigure(tile['error'],text='Image unavailable');self.photos.pop(name,None);tile['photo']=None
        else:
            if tile['photo'] is not photo:self.canvas.itemconfigure(tile['image'],image=photo or self.store.placeholder);self.canvas.itemconfigure(tile['error'],text='');tile['photo']=photo
            if photo:self.photos[name]=photo
    def arrange(self):
        if self.visible_job is not None:self.window.after_cancel(self.visible_job);self.visible_job=None
        if self.closed:return
        signature=self.viewport_signature();start,end=signature[:2]
        if self.layout_signature==signature:return
        self.layout_signature=signature;self.layout_passes+=1
        indices=list(range(start,end));visible={self.names[i] for i in indices};existing={t['name']:t for t in self.pool if t['name'] in visible};free=[t for t in self.pool if t['name'] not in visible]
        self.tiles={};self.tile_items={};self.photos={}
        while len(self.pool)<len(indices):
            tile=self.make_tile();self.pool.append(tile);free.append(tile)
        width=max(1,self.canvas.winfo_width());cell=max(1,width/self.columns)
        for index in indices:
            name=self.names[index];tile=existing.get(name) or free.pop();tile['name']=name;self.tiles[name]=tile;self.tile_items[name]=tile['item']
            if not tile['visible']:
                for k in ['item','image','caption','tag','error']:self.canvas.itemconfigure(tile[k],state='normal')
                tile['visible']=True
            x=index%self.columns*cell+6;y=index//self.columns*self.row_height+6;w=max(1,min(205,cell-12));signature=(name,index,cell,name in self.observations,name in self.predictions,name==self.selected)
            if signature!=tile['signature']:
                tile['signature']=signature;self.canvas.coords(tile['item'],x,y,x+w,y+self.tile_height)
                self.canvas.coords(tile['image'],x+w/2,y+4);self.canvas.coords(tile['caption'],x+w/2,y+113);self.canvas.coords(tile['tag'],x+w/2,y+self.tile_height-5);self.canvas.coords(tile['error'],x+w/2,y+55)
                caption=name if len(name)<=38 else name[:20]+'…'+name[-14:];self.canvas.itemconfigure(tile['caption'],text=caption,width=max(48,w-8))
                self.canvas.itemconfigure(tile['tag'],text='Measured' if name in self.observations else 'Predicted · verify' if name in self.predictions else 'Not measured',fill='#70dbc0' if name in self.observations else '#a0adb5')
                self.canvas.itemconfigure(tile['item'],outline='#70dbc0' if name==self.selected else '#344149')
            self.paint(name,tile);self.store.request(self.editor.image_dir/name,self.editor.image_sizes[name],priority=0)
        for tile in free:
            tile['name']=None
            if tile['visible']:
                for k in ['item','image','caption','tag','error']:self.canvas.itemconfigure(tile[k],state='hidden')
                tile['visible']=False
        for index in range(max(0,start-self.columns*3),min(len(self.names),end+self.columns*3)):
            name=self.names[index];self.store.request(self.editor.image_dir/name,self.editor.image_sizes[name],priority=1)
    def viewport_signature(self):
        top=self.canvas.canvasy(0);height=max(self.row_height,self.canvas.winfo_height())
        start=max(0,int(top//self.row_height)-1)*self.columns;end=min(len(self.names),(int((top+height)//self.row_height)+2)*self.columns)
        return start,end,self.columns,self.canvas.winfo_width(),self.generation
    def thumbnail_ready(self,key):
        for name,tile in self.tiles.items():
            if self.keys[name]==key:self.paint(name,tile)
    def select(self,name):
        if name is None:return
        self.selected=name;self.canvas.focus_set()
        if self.pane_index is not None:self.editor.set_focus(self.pane_index)
        self.info.set(name)
        for n,tile in self.tiles.items():self.canvas.itemconfigure(tile['item'],outline='#70dbc0' if n==name else '#344149')
    def select_and_open(self,name):self.select(name);return self.open_selected()
    def open_selected(self):
        if self.selected not in self.names:
            if not self.names:return
            self.select(self.names[0])
        index=0 if self.target.get()=='Left' else 1;self.editor.choose_image(self.selected,index)
        if self.selected in self.predictions:self.editor.panes[index]['focus_after_load']=(self.selected,self.predictions[self.selected])
        self.editor.tabs.select(self.editor.workspace)
        if not self.embedded:self.window.lift(self.editor.window)
        self.info.set(f'Opened in {self.target.get().lower()}: {self.selected}');self.editor.audit.event('grid_image_open',image=self.selected,pane=index);return 'break'
    def keyboard(self,step,event=None):
        if event and isinstance(event.widget,(tk.Entry,ttk.Combobox)):return
        if not self.names:return
        self.scroll_target=None
        index=self.names.index(self.selected) if self.selected in self.names else -1;index=max(0,min(len(self.names)-1,index+step));row=index//self.columns;y=row*self.row_height;top=self.canvas.canvasy(0);height=self.canvas.winfo_height();rows=max(1,(len(self.names)+self.columns-1)//self.columns)
        if y<top or y+self.row_height>top+height:self.canvas.yview_moveto(max(0,y-(height-self.row_height if y>top else 0))/(rows*self.row_height))
        self.arrange();self.select(self.names[index]);return 'break'
    def close(self):
        if self.closed:return
        self.closed=True;self.store.listeners.discard(self)
        if self.visible_job is not None:self.window.after_cancel(self.visible_job);self.visible_job=None
        self.scroll_target=None;self.photos.clear();self.pool.clear();self.tiles.clear();self.tile_items.clear();self.window.destroy()
