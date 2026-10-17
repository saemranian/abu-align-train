"""Frame-coalesced sash navigation and host adapters for Manual control points."""
import time
import tkinter as tk


class LiveLayout:
    def __init__(self,editor):
        self.editor=editor;self.window=editor.window;self.flushing=False;self.closed=False;self.flushes=0;self.last_ms=0.;self.bindings=[];self.job=None;self.drag=None;self.pending={};self.preview=None;self.ghost=None;self.sizes={};self.pending_resize=set();self.preferred=(260,300);self.reset_job=None
        # A local grab can deliver motion/release to children of the grabbed
        # split. Capture at this tool's toplevel too, using screen coordinates.
        for sequence,callback in [('<B1-Motion>',self.motion),('<ButtonRelease-1>',self.release)]:
            self.bindings.append((self.window,sequence,self.window.bind(sequence,callback,add='+')))
    def watch_split(self,split):
        scale=max(1.,self.window.winfo_fpixels('1i')/96);width=max(10,round(8*scale))
        split.configure(sashwidth=width,sashrelief='raised',showhandle=True,handlesize=max(10,round(8*scale)),handlepad=48,sashcursor='sb_h_double_arrow' if str(split.cget('orient'))=='horizontal' else 'sb_v_double_arrow')
        for sequence,callback in [('<ButtonPress-1>',self.press),('<B1-Motion>',self.motion),('<ButtonRelease-1>',self.release)]:
            binding=split.bind(sequence,callback,add='+');self.bindings.append((split,sequence,binding))
        self.bindings.append((split,'<Configure>',split.bind('<Configure>',lambda event,s=split:self.resize(s,event),add='+')))
    def resize(self,split,event):
        size=(event.width,event.height)
        if split is not self.editor.workspace_split or self.sizes.get(split)==size:return
        self.sizes[split]=size;self.pending_resize.add(split);self.flush()
    def constrain(self,split):
        panes=split.panes();minimum=[int(split.panecget(p,'minsize')) for p in panes];sash=int(split.cget('sashwidth'));width=split.winfo_width()
        if len(panes)!=3 or width<sum(minimum)+2*sash:return
        available=width-minimum[1]-2*sash;left,right=self.preferred
        if left+right>available:
            extra=max(0,available-minimum[0]-minimum[2]);a=max(0,left-minimum[0]);b=max(0,right-minimum[2]);left=minimum[0]+round(extra*a/max(1,a+b));right=available-left
        left=max(minimum[0],left);right=max(minimum[2],right)
        # Tk updates sash coordinates only in its geometry pass. Two immediate
        # sash_place calls can apply deltas to stale coordinates and push the
        # last pane to its minimum. Assign desired widths in one layout pass.
        split.paneconfigure(panes[0],width=left);split.paneconfigure(panes[1],width=width-left-right-2*sash);split.paneconfigure(panes[2],width=right)
    def press(self,event):
        hit=event.widget.identify(event.x,event.y)
        if not hit:return
        index=int(hit[0]);x,y=event.widget.sash_coord(index)
        self.drag=(event.widget,index,event.x-x,event.y-y);self.preview=(index,x,y);event.widget.grab_set()
        mode=self.editor.settings['resize_mode'].get()
        self.fast=mode=='fast' or mode=='auto' and self.window.winfo_width()*self.window.winfo_height()>2_000_000
        if self.fast:self.show_guide(event.widget,x,y)
        return 'break'
    def position(self,event):
        split,index,dx,dy=self.drag;x=event.x_root-split.winfo_rootx()-dx;y=event.y_root-split.winfo_rooty()-dy
        horizontal=str(split.cget('orient'))=='horizontal';axis=0 if horizontal else 1;panes=split.panes();width=int(split.cget('sashwidth'))
        low=(split.sash_coord(index-1)[axis]+width if index else 0)+int(split.panecget(panes[index],'minsize'))
        high=(split.sash_coord(index+1)[axis] if index+2<len(panes) else split.winfo_width() if horizontal else split.winfo_height())-width-int(split.panecget(panes[index+1],'minsize'))
        value=max(low,min(high,x if horizontal else y));original=split.sash_coord(index)
        return (index,value,original[1]) if horizontal else (index,original[0],value)
    def show_guide(self,split,x,y):
        if self.ghost is None:self.ghost=tk.Frame(self.window,bg='#70dbc0',bd=0)
        horizontal=str(split.cget('orient'))=='horizontal'
        self.ghost.place(x=split.winfo_rootx()-self.window.winfo_rootx()+x,y=split.winfo_rooty()-self.window.winfo_rooty()+y,width=3 if horizontal else split.winfo_width(),height=split.winfo_height() if horizontal else 3);self.ghost.lift()
    def motion(self,event):
        if self.drag is None:return
        split=self.drag[0];self.pending[split]=self.position(event);self.flush();return 'break'
    def release(self,event):
        if self.drag is None:return
        split=self.drag[0];target=self.position(event);self.motion(event);self.settled()
        if self.fast:
            index,x,y=self.preview;split.sash_place(index,x,y)
        if split is self.editor.workspace_split:
            positions=[split.sash_coord(i)[0] for i in [0,1]];positions[target[0]]=target[1]
            self.preferred=(positions[0],split.winfo_width()-positions[1]-int(split.cget('sashwidth')))
        self.hide_guide();self.drag=None;self.preview=None;split.grab_release();return 'break'
    def hide_guide(self):
        if self.ghost is not None:self.ghost.destroy();self.ghost=None
    def flush(self):
        if self.closed or self.editor.closed:return
        if self.job is None:self.job=self.window.after(16,self.settled)
    def settled(self):
        if self.job is not None:self.window.after_cancel(self.job)
        self.job=None
        if self.closed or self.editor.closed:return
        started=time.perf_counter();self.flushes+=1
        pending=self.pending;self.pending={}
        for split,(index,x,y) in pending.items():
            if self.drag is not None and split is self.drag[0] and self.fast:self.preview=(index,x,y);self.show_guide(split,x,y)
            else:split.sash_place(index,x,y)
        resized=self.pending_resize;self.pending_resize=set()
        for split in resized:
            if self.drag is None:self.constrain(split)
        # Never re-enter Tk's event loop from a drag callback. Each fixed frame
        # consumes the latest position; motion does not restart the deadline.
        self.last_ms=(time.perf_counter()-started)*1000
    def ready(self):
        e=self.editor
        sash=int(e.image_split.cget('sashwidth'))
        self.differences=[e.tabs.winfo_width()-(self.window.winfo_width()-24),e.workspace_split.winfo_width()-e.workspace.winfo_width(),e.image_split.winfo_width()-e.image_split.master.winfo_width(),sum(p['frame'].winfo_width() for p in e.panes)+sash-e.image_split.winfo_width()]
        if abs(e.tabs.winfo_width()-(self.window.winfo_width()-24))>3:return False
        if abs(e.workspace_split.winfo_width()-e.workspace.winfo_width())>3:return False
        if abs(e.image_split.winfo_width()-e.image_split.master.winfo_width())>3:return False
        if abs(sum(p['frame'].winfo_width() for p in e.panes)+sash-e.image_split.winfo_width())>3:return False
        for pane in e.panes:
            page=pane['pages'][pane['mode'].get()]
            if abs(page.winfo_width()-pane['frame'].winfo_width())>3:return False
            if pane['mode'].get()=='Image' and abs(pane['view'].canvas.winfo_width()-page.winfo_width())>3:return False
        return True
    def reset(self):
        if self.closed:return
        e=self.editor
        if self.drag is not None:self.hide_guide();self.drag[0].grab_release();self.drag=None
        self.pending.clear();self.preferred=(260,300);self.constrain(e.workspace_split)
        if self.reset_job is not None:self.window.after_cancel(self.reset_job)
        self.reset_job=self.window.after(16,self.reset_views)
        e.status.set('Pane sizes reset. Drag the visible handles; Auto uses a fast guide for large windows.');e.audit.event('pane_layout_reset')
    def reset_views(self):
        self.reset_job=None
        if self.closed:return
        e=self.editor;e.image_split.sash_place(0,(e.image_split.winfo_width()-int(e.image_split.cget('sashwidth')))//2,0);e.properties_split.sash_place(0,0,180)
    def close(self):
        self.closed=True
        if self.job is not None:self.window.after_cancel(self.job);self.job=None
        if self.drag is not None:
            self.drag[0].grab_release();self.drag=None
        self.hide_guide()
        if self.reset_job is not None:self.window.after_cancel(self.reset_job);self.reset_job=None
        self.pending.clear();self.pending_resize.clear()
        for widget,sequence,binding in self.bindings:widget.unbind(sequence,binding)


class PaneSceneHost:
    """Host adapter for the same M54 renderer inside either workspace pane."""
    def __new__(cls,editor,index):
        from abu_m54_host import ViewerHost
        def camera(name):
            other=1-index
            if editor.panes[other]['mode'].get()=='Image':editor.choose_image(name,other)
            else:editor.status.set('Selected camera: '+name+' · change the other view to Image to inspect its original.')
            editor.highlight_images()
        host=ViewerHost(editor.window,on_camera=camera);host.audit=editor.audit;host.status=editor.status;host.select=editor.select
        return host
