"""Installation capabilities and license scope, independent of model downloads."""
import json
from pathlib import Path
import platform
import sys


def catalog():
    return json.loads((Path(__file__).parent/'installer/engines.json').read_text(encoding='utf-8'))


def capabilities():
    return {'version':(Path(__file__).parent/'VERSION.txt').read_text().strip(),
            'platform':sys.platform,'architecture':platform.machine(),
            'edition':'bundled CPU core candidate' if getattr(sys,'frozen',False) else 'developer checkout',
            'engines':catalog()['engines'],
            'notice':'Engine code, checkpoints and native libraries have separate terms. No engine packs are downloaded automatically.'}


def check_bundled_engine(engine, training=False):
    """Developer installations stay intact; frozen core advertises only its payload."""
    if not getattr(sys,'frozen',False):return
    if engine=='native_colmap' and not training:return
    raise RuntimeError(f'{engine} is an external engine, not part of this CPU core installer. '
                       'Use a qualified engine installation through ABU_ENGINE_PYTHON. '
                       'Research engine terms and model permissions still apply. '
                       'See Tools > Distribution and engines.')


def show_capabilities(parent):
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText
    window=tk.Toplevel(parent);window.title('Distribution and engines');window.geometry('860x620')
    text=ScrolledText(window,wrap='word',font=('TkDefaultFont',10));text.pack(fill='both',expand=True,padx=12,pady=12)
    info=capabilities()
    text.insert('end',f"Abu installation: {info['edition']}\n{info['platform']} / {info['architecture']}\n\n")
    text.insert('end','Bundled core: Manual Control Points and CPU refinement, native spherical COLMAP, '
                       'Abu M54, image tools. Windows has the native GPU viewport; other platforms use the shared browser viewport. '
                       'Video extraction needs external FFmpeg. Gaussian training and learned matching models are not bundled.\n\n')
    for engine in info['engines']:
        text.insert('end',f"{engine['name']}\n  Code: {engine['license']}\n  Distribution: {engine['distribution']}\n  {engine['notes']}\n\n")
    text.insert('end',info['notice']);text.configure(state='disabled')


def main():
    print(json.dumps(capabilities(),indent=2))
