"""Installation capabilities and license scope, independent of model downloads."""
import json
from pathlib import Path
import platform
import sys


def catalog():
    return json.loads((Path(__file__).parent/'installer/engines.json').read_text(encoding='utf-8'))


def capabilities():
    from abu_engine_packs import catalog as packs, platform_key, executable, component_root
    installed=[]
    for pack in packs():
        if pack['platform']!=platform_key():continue
        try:
            path=executable(pack['id'],next(iter(pack['executables'])))
            state='installed' if path else 'not installed'
        except Exception as error:state=str(error)
        installed.append({'id':pack['id'],'version':pack['version'],'state':state,'capability':pack.get('capability','alignment')})
    return {'version':(Path(__file__).parent/'VERSION.txt').read_text().strip(),
            'platform':sys.platform,'architecture':platform.machine(),
            'edition':'bundled CPU core candidate' if getattr(sys,'frozen',False) else 'developer checkout',
            'engines':catalog()['engines'],
            'component_storage':str(component_root()),'components':installed,
            'notice':'Engine code, checkpoints and native libraries have separate terms. GPU availability depends on the installed component and compatible NVIDIA hardware/driver. No images are uploaded by component setup.'}


def check_bundled_engine(engine, training=False):
    """Developer installations stay intact; frozen core advertises only its payload."""
    if not getattr(sys,'frozen',False):return
    if engine=='native_colmap' and not training:return
    if training and engine.lower().strip() in {'lichtfeld','lichtfeld_studio','lfs'}:
        from abu_engine_packs import optional_executable
        if optional_executable('lichtfeld-cuda','trainer'):return
    raise RuntimeError(f'{engine} is an external engine, not part of this CPU core installer. '
                       'Use a qualified engine installation through ABU_ENGINE_PYTHON. '
                       'Research engine terms and model permissions still apply. '
                       'See Tools > Distribution and engines.')


def show_capabilities(parent):
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    window=tk.Toplevel(parent);window.title('Distribution and engines');window.geometry('860x620')
    from tkinter import ttk
    ttk.Button(window,text='GPU components — install / storage / status',command=lambda:show_component_manager(window)).pack(anchor='w',padx=12,pady=(12,0))
    text=ScrolledText(window,wrap='word',font=('TkDefaultFont',10));text.pack(fill='both',expand=True,padx=12,pady=12)
    info=capabilities()
    text.insert('end',f"Abu installation: {info['edition']}\n{info['platform']} / {info['architecture']}\n\n")
    text.insert('end','Bundled core: Manual Control Points and CPU refinement, native spherical COLMAP, '
                       'Abu M54, image tools. Windows has the native GPU viewport; other platforms use the shared browser viewport. '
                       'The GPU components button installs the official Windows COLMAP CUDA runtime for native panorama feature extraction and matching. '
                       'Use the component list below for installed training support. Learned research engines and their checkpoints have separate distribution restrictions.\n\n')
    for component in info['components']:
        text.insert('end',f"{component['id']} {component['version']}: {component['state']}\n")
    text.insert('end','\n')
    for engine in info['engines']:
        text.insert('end',f"{engine['name']}\n  Code: {engine['license']}\n  Distribution: {engine['distribution']}\n  {engine['notes']}\n\n")
    text.insert('end',info['notice']);text.configure(state='disabled')


def show_component_manager(parent):
    """All network/extraction work stays off Tk's event thread."""
    import queue
    import threading
    import tkinter as tk
    from tkinter import ttk, filedialog
    from abu_engine_packs import catalog as packs, component_root, set_component_root, platform_key, install, executable
    window=tk.Toplevel(parent);window.title('Abu GPU components');window.geometry('860x560')
    frame=ttk.Frame(window,padding=16);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='Prebuilt GPU engines',font=('TkDefaultFont',14,'bold')).pack(anchor='w')
    ttk.Label(frame,text='Downloads software only. Your images and reconstructions stay on your computer.',wraplength=800).pack(anchor='w',pady=8)
    storage=tk.StringVar(value=str(component_root()))
    row=ttk.Frame(frame);row.pack(fill='x',pady=8)
    ttk.Entry(row,textvariable=storage,state='readonly').pack(side='left',fill='x',expand=True)
    def choose():
        path=filedialog.askdirectory(parent=window,title='Choose component storage folder')
        if path:
            set_component_root(path);storage.set(str(component_root()));refresh()
    choose_button=ttk.Button(row,text='Storage folder…',command=choose);choose_button.pack(side='right')
    tree=ttk.Treeview(frame,columns=('name','version','state'),show='headings',height=5)
    for key,label,width in [('name','Component',400),('version','Version',90),('state','Status',220)]:
        tree.heading(key,text=label);tree.column(key,width=width)
    tree.pack(fill='x',pady=8)
    available={p['id']:p for p in packs() if p['platform']==platform_key()}
    detail=tk.StringVar(value='Select a component to see requirements and license information.')
    ttk.Label(frame,textvariable=detail,wraplength=800,justify='left').pack(fill='x',pady=8)
    status=tk.StringVar(value='Choose a prebuilt component. Only components published and pinned for this platform appear here. GPU alignment and training have separate qualification checks.')
    ttk.Label(frame,textvariable=status,wraplength=800).pack(fill='x',pady=8)
    bar=ttk.Progressbar(frame,maximum=100);bar.pack(fill='x',pady=8)
    events=queue.Queue();stop=threading.Event();busy=False
    def refresh():
        tree.delete(*tree.get_children())
        for key,p in available.items():
            try:state='Installed (runtime benchmark separate)' if executable(key,next(iter(p['executables']))) else 'Not installed'
            except Exception as error:state=str(error)
            tree.insert('', 'end',iid=key,values=(p['name'],p['version'],state))
    def select(event=None):
        selection=tree.selection()
        if selection:
            p=available[selection[0]];detail.set(f"{p['download_bytes']/1024**2:.0f} MB download · {p['license']}\n{p['notes']}")
    tree.bind('<<TreeviewSelect>>',select)
    def start():
        nonlocal busy
        selection=tree.selection()
        if busy or not selection:return
        key=selection[0];busy=True;stop.clear()
        button.configure(state='disabled');choose_button.configure(state='disabled');cancel.configure(state='normal')
        def work():
            last=[0.0]
            def progress(label,done,total):
                import time
                now=time.monotonic()
                if done==total or now-last[0]>.1:
                    last[0]=now;events.put(('progress',(label,done,total)))
            try:
                install(key,root=storage_path,progress=progress,cancel=stop.is_set)
                events.put(('done','Component installed. Select native COLMAP with device Auto or CUDA.'))
            except Exception as error:events.put(('done',str(error)))
        storage_path=storage.get()
        threading.Thread(target=work,daemon=True).start()
    controls=ttk.Frame(frame);controls.pack(fill='x',pady=8)
    button=ttk.Button(controls,text='Install selected component',command=start);button.pack(side='left')
    cancel=ttk.Button(controls,text='Cancel / pause',command=stop.set,state='disabled');cancel.pack(side='left',padx=8)
    def poll():
        nonlocal busy
        if not window.winfo_exists():return
        try:
            while True:
                kind,value=events.get_nowait()
                if kind=='progress':
                    label,done,total=value;status.set(label);bar['value']=100*done/max(total,1)
                else:
                    busy=False;status.set(value);button.configure(state='normal');choose_button.configure(state='normal');cancel.configure(state='disabled');refresh()
        except queue.Empty:pass
        window.after(100,poll)
    def close():
        if busy:stop.set();status.set('Cancelling safely; close this window when the operation has stopped.')
        else:window.destroy()
    window.protocol('WM_DELETE_WINDOW',close)
    refresh();poll()


def main():
    print(json.dumps(capabilities(),indent=2))
