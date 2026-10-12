"""Compatibility entry point for the 360 control-point workspace."""
from control_points_workspace import ControlPointEditor, open_editor

if __name__ == '__main__':
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    editor = open_editor(root)
    editor.window.bind('<Destroy>', lambda event: root.destroy() if event.widget is editor.window else None)
    root.mainloop()
