"""Standalone desktop identity and DPI setup; call before creating Tk windows."""
import ctypes
import sys
from pathlib import Path


def prepare_desktop():
    status='platform default'
    if sys.platform=='win32':
        try:
            user=ctypes.windll.user32;user.SetProcessDpiAwarenessContext.argtypes=[ctypes.c_void_p]
            status='per-monitor v2' if user.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)) else 'existing process DPI policy'
        except (AttributeError,OSError):
            try:
                result=ctypes.windll.shcore.SetProcessDpiAwareness(2);status='per-monitor' if result==0 else 'existing process DPI policy'
            except (AttributeError,OSError):
                try:ctypes.windll.user32.SetProcessDPIAware();status='system DPI'
                except (AttributeError,OSError):pass
        try:ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Abu.M54.Viewer')
        except (AttributeError,OSError):pass
    return status


def window_identity(window):
    root=Path(__file__).parent;icon=root/'abu_icon.ico'
    if icon.is_file():
        try:window.iconbitmap(default=str(icon))
        except Exception:
            try:window.iconbitmap(str(icon))
            except Exception:pass
    png=root/'abu_icon.png'
    if png.is_file():
        try:
            from PIL import ImageTk
            window._m54_icon=ImageTk.PhotoImage(master=window,file=str(png));window.iconphoto(True,window._m54_icon)
        except Exception:pass
    # Tk's default/photo icon can remain only a class icon on Windows. Explicitly
    # assign this toplevel's multi-resolution ICO so its real titlebar gets it.
    if sys.platform=='win32' and icon.is_file():
        try:window.iconbitmap(str(icon))
        except Exception:pass
    window._m54_icon_path=str(icon)
    return float(window.winfo_fpixels('1i'))
