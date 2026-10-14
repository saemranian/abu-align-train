"""Release retired tool Tcl handles on their owner thread, before background GC."""
import threading,tkinter as tk
from PIL import ImageTk


def capture(window,owners):
    nodes=[]
    def walk(widget):
        nodes.append(widget)
        for child in list(widget.children.values()):walk(child)
    walk(window);variables={};images={}
    def values(value):
        if isinstance(value,tk.Variable):variables[id(value)]=value
        elif isinstance(value,(tk.PhotoImage,ImageTk.PhotoImage)):images[id(value)]=value
        elif isinstance(value,dict):
            for item in value.values():
                if isinstance(item,(tk.Variable,tk.PhotoImage,ImageTk.PhotoImage,list,tuple,dict)):values(item)
        elif isinstance(value,(list,tuple)):
            for item in value:values(item)
    for owner in [*nodes,*[o for o in owners if o is not None]]:
        for name,value in vars(owner).items():
            if name not in {'master','children','editor'}:values(value)
    return nodes,list(variables.values()),list(images.values())


def retire(snapshot):
    if threading.current_thread() is not threading.main_thread():raise RuntimeError('Retire tool handles on the GUI owner thread.')
    nodes,variables,images=snapshot
    for variable in variables:
        try:variable.__del__()
        except tk.TclError:pass
        # Tk Variable has no public dispose API. Mark its already-released Tcl
        # handle inert so delayed Python finalization cannot touch Tcl again.
        variable._tk=None;variable._root=None
    for image in images:
        try:image.__del__()
        except tk.TclError:pass
        if isinstance(image,ImageTk.PhotoImage):
            image._PhotoImage__photo.name=None;image._PhotoImage__photo.tk=None
        else:image.name=None
        image.tk=None
    for widget in nodes:
        widget.tk=None;widget.master=None
