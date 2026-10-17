"""Modeless image inclusion and mask-layer settings, shared by both align actions."""
import copy,tkinter as tk
from tkinter import ttk,filedialog
from manual_image_policy import Masks,validate_policy


class ImagePolicyPanel:
    def __init__(self,editor):
        self.editor=editor;self.closed=False;self.window=tk.Toplevel(editor.window);self.window.title('Images & masks · Manual control points');self.window.geometry('850x660');self.window.configure(bg='#1a2024')
        top=tk.Frame(self.window,bg='#1a2024',padx=12,pady=10);top.pack(fill='x')
        editor.label(top,'Disable images for future main Align and Refine. Originals and saved measurements remain intact.',wraplength=790).pack(fill='x')
        self.directory=tk.StringVar(self.window);self.invert=tk.BooleanVar(self.window);self.search=tk.StringVar(self.window);self.info=tk.StringVar(self.window)
        editor.label(top,'Mask folder · black = ignore, white = keep',muted=True).pack(fill='x',pady=(8,3))
        bar=tk.Frame(top,bg='#1a2024');bar.pack(fill='x');editor.entry(bar,self.directory,50).pack(side='left',fill='x',expand=True);editor.button(bar,'Browse',self.browse).pack(side='left',padx=6);editor.button(bar,'Clear',lambda:self.directory.set('')).pack(side='left')
        tk.Checkbutton(top,text='Invert mask colours',variable=self.invert,bg='#1a2024',fg='#e7edef',selectcolor='#242d32').pack(anchor='w')
        editor.guide(bar,'Original-resolution mask layers','Use image.jpg.png, image.mask.png or image.png. Sizes must equal the original. Missing masks keep the whole image. Ambiguous names are rejected. Apply validates all masks; Save controls persists the choices.')
        editor.entry(top,self.search,30).pack(fill='x',pady=6);self.search.trace_add('write',lambda *_:self.refresh())
        self.tree=editor.tree(self.window,['Image','Alignment','Mask'],[430,110,190],12);self.tree.configure(selectmode='extended');self.tree.tag_configure('disabled',foreground='#839098');self.tree.bind('<Double-1>',lambda e:self.open_selected())
        bottom=tk.Frame(self.window,bg='#1a2024',padx=12,pady=10);bottom.pack(side='bottom',fill='x',before=self.tree.master)
        editor.label(bottom,variable=self.info,muted=True,wraplength=790).pack(fill='x')
        for title,command in [('Disable selected',lambda:self.enable(False)),('Enable selected',lambda:self.enable(True)),('Apply masks',self.apply),('Open image',self.open_selected)]:editor.button(bottom,title,command).pack(side='left',padx=3)
        policy=validate_policy(editor.controls.get('image_policy'));self.directory.set(policy['mask_dir']);self.invert.set(policy['invert_masks']);self.refresh();self.window.protocol('WM_DELETE_WINDOW',self.close)
    def refresh(self):
        policy=validate_policy(self.editor.controls.get('image_policy'));self.tree.delete(*self.tree.get_children());masks=Masks(policy)
        for name in self.editor.image_names:
            if self.search.get().casefold() not in name.casefold():continue
            excluded=name in policy['excluded_images']
            try:path=masks.path(name);mask=path.name if path else 'Whole image'
            except ValueError:mask='Check mask folder/names'
            self.tree.insert('','end',iid=name,values=(name,'Disabled' if excluded else 'Enabled',mask),tags=('disabled',) if excluded else ())
        self.info.set(f"{len(policy['excluded_images'])} disabled · main Align and Refine use saved controls; Apply/Save before running.")
    def enable(self,value):
        names=self.tree.selection()
        if not names or not self.editor.can_edit():return
        policy=validate_policy(self.editor.controls.get('image_policy'));excluded=set(policy['excluded_images'])
        excluded.difference_update(names) if value else excluded.update(names);policy['excluded_images']=sorted(excluded);self.commit(policy,'images_enabled' if value else 'images_disabled');self.refresh()
    def browse(self):
        path=filedialog.askdirectory(parent=self.window,title='Choose original-resolution mask layers')
        if path:self.directory.set(path)
    def apply(self):
        if not self.editor.can_edit():return
        try:
            policy=validate_policy(self.editor.controls.get('image_policy'));policy.update(mask_dir=self.directory.get().strip(),invert_masks=self.invert.get());records=Masks(policy).validate_catalog(self.editor.image_sizes)
            from pathlib import Path
            if policy['mask_dir'] and Path(policy['mask_dir']).resolve()==self.editor.image_dir.resolve():raise ValueError('Keep mask layers in a separate folder from original images.')
            if policy['mask_dir'] and not records:raise ValueError('No mask filenames match the original images.')
            self.commit(policy,'mask_policy_applied');self.refresh();self.info.set(f'{len(records)} mask layers validated · Save controls to persist.')
        except Exception as error:self.editor.error('Check mask layers',error)
    def commit(self,policy,event):
        self.editor.document.checkpoint();self.editor.controls['image_policy']=copy.deepcopy(policy);self.editor.audit.event(event,policy=policy);self.editor.refresh()
    def open_selected(self):
        if self.tree.selection():self.editor.choose_image(self.tree.selection()[0])
    def close(self):self.closed=True;self.window.destroy()
