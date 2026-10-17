"""Shared, bounded local thumbnails: priority prefetch and instant warm hits."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import heapq,queue,threading,time
from PIL import Image,ImageTk,ImageDraw


class ThumbnailStore:
    def __init__(self,editor,limit=128):
        self.editor=editor;self.limit=limit;self.cache=OrderedDict();self.photos=OrderedDict();self.errors=OrderedDict();self.keys={};self.listeners=set()
        self.condition=threading.Condition();self.tasks=[];self.pending={};self.running=set();self.serial=0;self.closed=False;self.results=queue.Queue()
        self.stats=dict(decodes=0,cache_hits=0,failed=0,photo_builds=0);self.executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='Abu thumbnails')
        # Decode workers own only plain queues, locks and PIL buffers. Retaining
        # this GUI owner on a worker can destroy Tk variables on that thread.
        self.state=SimpleNamespace(**{k:getattr(self,k) for k in ['condition','tasks','pending','running','cache','errors','stats','results']},limit=limit,closed=False)
        for _ in range(2):self.executor.submit(self.worker,self.state)
        placeholder=Image.new('RGB',(195,105),'#182126');draw=ImageDraw.Draw(placeholder)
        draw.rectangle((78,36,115,68),outline='#344149',width=2);draw.line((81,64,94,49,102,58,109,52,114,64),fill='#344149',width=2)
        self.placeholder=ImageTk.PhotoImage(placeholder,master=editor.window);self.poll_job=editor.window.after(16,self.poll)
    def key(self,path,expected):
        identity=(str(path),tuple(expected))
        if identity not in self.keys:
            try:stamp=path.stat().st_mtime_ns
            except OSError:stamp=0
            self.keys[identity]=(*identity,stamp)
        return self.keys[identity]
    def request(self,path,expected,priority=0):
        key=self.key(path,expected)
        with self.condition:
            if self.closed or key in self.cache or key in self.errors or key in self.running:return key
            old=self.pending.get(key)
            if old is not None and old[0]<=priority:return key
            self.serial+=1;token=(priority,self.serial);self.pending[key]=token
            heapq.heappush(self.tasks,(priority,self.serial,key,path,tuple(expected)));self.condition.notify()
        return key
    @staticmethod
    def worker(state):
        while True:
            with state.condition:
                while not state.tasks and not state.closed:state.condition.wait()
                if state.closed:return
                priority,serial,key,path,expected=heapq.heappop(state.tasks)
                if state.pending.get(key)!=(priority,serial):continue
                state.pending.pop(key);state.running.add(key)
            try:
                with Image.open(path) as source:
                    if source.size!=expected:raise ValueError('Thumbnail dimensions do not match the original-image catalog.')
                    source.draft('RGB',(390,210));source.thumbnail((195,105),Image.Resampling.LANCZOS);thumb=source.convert('RGB')
                with state.condition:
                    if not state.closed:
                        state.cache[key]=thumb;state.stats['decodes']+=1
                        while len(state.cache)>state.limit:state.cache.popitem(last=False)
            except Exception as error:
                with state.condition:
                    if not state.closed:
                        state.errors[key]=str(error);state.stats['failed']+=1
                        while len(state.errors)>state.limit:state.errors.popitem(last=False)
            finally:
                with state.condition:state.running.discard(key)
                if not state.closed:state.results.put(key)
    def photo(self,key,measurement=None,measured=False,width=195):
        width=max(48,min(195,int(width)));width=195 if width==195 else width//16*16;annotation=None if measurement is None else (float(measurement[0]),float(measurement[1]),bool(measured));pkey=(key,annotation,width)
        if pkey in self.photos:self.photos.move_to_end(pkey);self.stats['cache_hits']+=1;return self.photos[pkey]
        with self.condition:
            if key not in self.cache:return None
            self.cache.move_to_end(key);thumb=self.cache[key]
        if thumb.width>width:thumb=thumb.resize((width,max(1,round(thumb.height*width/thumb.width))),Image.Resampling.BILINEAR)
        if annotation:
            thumb=thumb.copy();w,h=key[1];x,y=annotation[0]*thumb.width/w,annotation[1]*thumb.height/h
            ImageDraw.Draw(thumb).ellipse((x-4,y-4,x+4,y+4),outline='#ffce73' if measured else '#78b4ff',width=2)
        photo=ImageTk.PhotoImage(thumb,master=self.editor.window);self.photos[pkey]=photo;self.stats['photo_builds']+=1
        while len(self.photos)>self.limit:self.photos.popitem(last=False)
        return photo
    def poll(self):
        if self.closed:return
        start=time.perf_counter();count=0
        while count<12 and time.perf_counter()-start<.006:
            try:key=self.results.get_nowait()
            except queue.Empty:break
            for grid in tuple(self.listeners):
                if not grid.closed and grid.canvas.winfo_ismapped():grid.thumbnail_ready(key)
            if key in self.errors:self.editor.audit.event('thumbnail_error',image=key[0],error=self.errors[key])
            count+=1
        self.poll_job=self.editor.window.after(16,self.poll)
    def close(self):
        if self.closed:return
        self.closed=True;self.editor.window.after_cancel(self.poll_job)
        with self.condition:self.state.closed=True;self.tasks.clear();self.pending.clear();self.condition.notify_all()
        self.executor.shutdown(wait=False,cancel_futures=True);self.listeners.clear();self.cache.clear();self.photos.clear();self.errors.clear();self.placeholder=None
