"""Small host interface for M54 embedded in alignment, training or standalone UI."""
import tkinter as tk
from control_points_workspace import ControlPointEditor
from control_points_logging import ControlLog


class ViewerHost:
    label=ControlPointEditor.label
    guide=ControlPointEditor.guide
    button=ControlPointEditor.button
    entry=ControlPointEditor.entry
    scroll_page=ControlPointEditor.scroll_page
    bind_page_scroll=ControlPointEditor.bind_page_scroll
    style_dropdowns=ControlPointEditor.style_dropdowns
    def __init__(self,window,on_camera=None):
        self.window=window;self.guides=[];self.edit_buttons=[];self.audit=ControlLog()
        self.status=tk.StringVar(window);self.on_camera=on_camera
    def select(self,name):self.status.set('Selected control: '+name)
    def choose_image(self,name):
        self.status.set('Selected camera: '+name)
        if self.on_camera:self.on_camera(name)
    def error(self,title,error):
        from tkinter import messagebox
        self.audit.error('m54_error',error);messagebox.showerror(title,str(error),parent=self.window)
