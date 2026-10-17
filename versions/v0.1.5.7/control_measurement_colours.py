"""Colour only the angular-error cells; preserve the existing native table API."""
import math
import tkinter as tk

COLOURS={'good':'#70dbc0','warning':'#ffce73','bad':'#ff7777'}


def validate_limits(settings):
    if not isinstance(settings,dict):raise ValueError('Error colour settings must be an object.')
    try:good=float(settings.get('error_good_deg',.1));warning=float(settings.get('error_warning_deg',.5))
    except (TypeError,ValueError):raise ValueError('Enter finite positive error colour limits in degrees.') from None
    if not 0<good<warning or not math.isfinite(warning):raise ValueError('Green error limit must be positive and smaller than the yellow limit.')
    return {'error_good_deg':good,'error_warning_deg':warning}


def error_band(error,good=.1,warning=.5):
    if error is None or not math.isfinite(error):return None
    return 'good' if error<=good else 'warning' if error<=warning else 'bad'


class ErrorCells:
    def __init__(self,editor,tree):
        self.editor=editor;self.tree=tree;self.labels={};self.errors={};self.job=None;self.closed=False;self.last_limits=(.1,.5);self.valid_limits=True
        old=tree.cget('yscrollcommand')
        def scrolled(first,last):
            if old:tree.tk.call(old,first,last)
            self.schedule()
        tree.configure(yscrollcommand=scrolled)
        old_x=tree.cget('xscrollcommand')
        def xscrolled(first,last):
            if old_x:tree.tk.call(old_x,first,last)
            self.schedule()
        tree.configure(xscrollcommand=xscrolled)
        for event in ['<Configure>','<<TreeviewSelect>>','<B1-Motion>','<ButtonRelease-1>']:tree.bind(event,lambda e:self.schedule(),add='+')
    def set_errors(self,errors):self.errors=errors;self.schedule()
    def schedule(self):
        if not self.closed and self.job is None:self.job=self.tree.after_idle(self.layout)
    def limits(self):
        try:
            good=float(self.editor.settings['error_good_deg'].get());warning=float(self.editor.settings['error_warning_deg'].get())
            if not 0<good<warning or not math.isfinite(warning):raise ValueError
            self.last_limits=(good,warning);self.valid_limits=True;return self.last_limits
        except (ValueError,tk.TclError):self.valid_limits=False;return self.last_limits
    def layout(self):
        if self.job is not None:self.tree.after_cancel(self.job)
        self.job=None
        if self.closed:return
        good,warning=self.limits();active=set()
        for iid in self.tree.get_children():
            band=error_band(self.errors.get(iid),good,warning);box=self.tree.bbox(iid,'° error')
            if band is None or not box:continue
            x,y,w,h=box
            if x+w<=0 or x>=self.tree.winfo_width():continue
            active.add(iid);label=self.labels.get(iid)
            if label is None:
                label=tk.Label(self.tree,bd=0,padx=2,font=('Segoe UI',10),anchor='center');self.labels[iid]=label
                label.bind('<Button-1>',lambda e,n=iid:self.select(n));label.bind('<Double-1>',lambda e,n=iid:self.open(n))
                label.bind('<MouseWheel>',self.wheel)
            tags=self.tree.item(iid,'tags');row_bg=next((self.tree.tag_configure(t,'background') for t in reversed(tags) if self.tree.tag_configure(t,'background')),'#242d32')
            signature=(self.errors[iid],band,iid in self.tree.selection(),row_bg,x,y,w,h)
            if getattr(label,'_error_signature',None)!=signature:
                label._error_signature=signature;label.configure(text=f'{self.errors[iid]:.3f}',fg=COLOURS[band],bg='#31574f' if iid in self.tree.selection() else row_bg);label.place(x=x,y=y,width=w,height=h)
        for iid in list(self.labels):
            if iid not in active:self.labels.pop(iid).destroy()
    def select(self,iid):self.tree.selection_set(iid);self.tree.focus(iid);self.tree.focus_set();self.schedule();return 'break'
    def open(self,iid):self.select(iid);self.editor.open_measurement(self.editor.focused);return 'break'
    def wheel(self,event):self.tree.yview_scroll(-1 if event.delta>0 else 1,'units');self.schedule();return 'break'
    def close(self):
        self.closed=True
        if self.job is not None:self.tree.after_cancel(self.job);self.job=None
        for label in self.labels.values():label.destroy()
        self.labels.clear()
