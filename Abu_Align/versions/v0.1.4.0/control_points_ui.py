"""Two-pane manual panorama landmark editor; edits controls, never source photos."""
import copy
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk


class ControlPointEditor:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title('360° manual control points')
        self.window.geometry('1300x750')
        self.controls = {'schema_version': 1, 'points': [], 'distances': []}
        self.history = []
        self.reconstruction = None
        self.image_dir = None
        self.result_queue = queue.Queue()
        toolbar = ttk.Frame(self.window); toolbar.pack(fill='x')
        for text, command in [('Open reconstruction', self.open), ('Load controls', self.load),
                              ('Save controls', self.save), ('New point', self.new), ('Delete point', self.delete),
                              ('Set type / coordinates', self.set_type), ('Distance', self.distance),
                              ('Undo', self.undo), ('Refine alignment', self.refine)]:
            ttk.Button(toolbar, text=text, command=command).pack(side='left', padx=2)
        self.active = tk.StringVar()
        self.point_choice = ttk.Combobox(self.window, textvariable=self.active, state='readonly')
        self.point_choice.pack(fill='x'); self.point_choice.bind('<<ComboboxSelected>>', lambda e: self.draw())
        self.status = tk.StringVar(value='Open a spherical reconstruction. Left click places the active point; right click removes its observation. Wheel zooms; middle drag pans.')
        ttk.Label(self.window, textvariable=self.status, wraplength=1200).pack(fill='x')
        panes = ttk.Panedwindow(self.window, orient='horizontal'); panes.pack(fill='both', expand=True)
        self.panes = []
        for _ in range(2):
            frame = ttk.Frame(panes); panes.add(frame, weight=1)
            name = tk.StringVar()
            choice = ttk.Combobox(frame, textvariable=name, state='readonly'); choice.pack(fill='x')
            canvas = tk.Canvas(frame, bg='#111518', highlightthickness=0); canvas.pack(fill='both', expand=True)
            pane = {'name': name, 'choice': choice, 'canvas': canvas, 'zoom': 1., 'image': None, 'photo': None}
            self.panes.append(pane)
            choice.bind('<<ComboboxSelected>>', lambda e, p=pane: self.show(p))
            canvas.bind('<Button-1>', lambda e, p=pane: self.place(p, e))
            canvas.bind('<Button-3>', lambda e, p=pane: self.place(p, e, remove=True))
            canvas.bind('<MouseWheel>', lambda e, p=pane: self.zoom(p, e.delta))
            canvas.bind('<ButtonPress-2>', lambda e, c=canvas: c.scan_mark(e.x, e.y))
            canvas.bind('<B2-Motion>', lambda e, p=pane: self.pan(p, e))
        self.window.after(200, self.poll)

    def snapshot(self):
        self.history.append(copy.deepcopy(self.controls))
        self.history = self.history[-50:]

    def selected(self):
        return next((point for point in self.controls['points'] if point['id'] == self.active.get()), None)

    def refresh(self):
        self.point_choice['values'] = [p['id'] for p in self.controls['points']]
        self.draw()

    def open(self):
        path = filedialog.askopenfilename(parent=self.window, title='Open reconstruction.json', filetypes=[('Reconstruction', '*.json')])
        if not path: return
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError('Select one spherical reconstruction component')
            if any(c['projection_type'] != 'spherical' for c in data[0]['cameras'].values()):
                raise ValueError('Select a native spherical reconstruction')
            directory = Path(path).parent / 'images'
            if not directory.is_dir():
                selected = filedialog.askdirectory(parent=self.window, title='Choose original panorama images')
                if not selected: return
                directory = Path(selected)
            self.reconstruction = Path(path); self.image_dir = directory
            names = sorted(data[0]['shots'])
            for index, pane in enumerate(self.panes):
                pane['choice']['values'] = names
                pane['name'].set(names[min(index, len(names)-1)])
                self.show(pane)
        except Exception as error: messagebox.showerror('Cannot open', str(error), parent=self.window)

    def show(self, pane):
        if not self.image_dir: return
        try:
            with Image.open(self.image_dir / pane['name'].get()) as image:
                pane['image'] = image.convert('RGB')
            pane['zoom'] = min(1., max(pane['canvas'].winfo_width(), 600) / pane['image'].width)
            self.draw()
        except Exception as error: self.status.set(str(error))

    def draw(self):
        for pane in self.panes:
            image = pane['image']
            if image is None: continue
            canvas, zoom = pane['canvas'], pane['zoom']
            # Render only the visible image region; zooming an 8K panorama must
            # not allocate a complete 32K bitmap for either editor pane.
            left = max(0, min(image.width-1, int(canvas.canvasx(0) / zoom)))
            top = max(0, min(image.height-1, int(canvas.canvasy(0) / zoom)))
            right = min(image.width, left + max(1, int(max(canvas.winfo_width(), 600) / zoom) + 2))
            bottom = min(image.height, top + max(1, int(max(canvas.winfo_height(), 400) / zoom) + 2))
            tile = image.crop((left, top, right, bottom))
            pane['photo'] = ImageTk.PhotoImage(tile.resize((max(1, int(tile.width*zoom)), max(1, int(tile.height*zoom)))))
            canvas.delete('all'); canvas.create_image(left*zoom, top*zoom, anchor='nw', image=pane['photo'])
            canvas.configure(scrollregion=(0, 0, image.width*zoom, image.height*zoom))
            for point in self.controls['points']:
                for obs in point['observations']:
                    if obs['shot_id'] != pane['name'].get(): continue
                    x, y = [p*zoom for p in obs['pixel']]
                    color = '#ffcc55' if point['id'] == self.active.get() else '#70dbc0'
                    canvas.create_oval(x-5, y-5, x+5, y+5, outline=color, width=2)
                    canvas.create_text(x+8, y, text=point['id'], fill=color, anchor='w')

    def zoom(self, pane, delta):
        pane['zoom'] = min(4., max(.05, pane['zoom'] * (1.2 if delta > 0 else 1/1.2)))
        self.draw()

    def pan(self, pane, event):
        pane['canvas'].scan_dragto(event.x, event.y, gain=1)
        self.draw()

    def place(self, pane, event, remove=False):
        point = self.selected()
        if point is None or pane['image'] is None: return
        x = pane['canvas'].canvasx(event.x) / pane['zoom']; y = pane['canvas'].canvasy(event.y) / pane['zoom']
        if not 0 <= x < pane['image'].width or not 0 <= y < pane['image'].height: return
        self.snapshot()
        point['observations'] = [o for o in point['observations'] if o['shot_id'] != pane['name'].get()]
        if not remove: point['observations'].append({'shot_id': pane['name'].get(), 'pixel': [x, y], 'sigma_px': 1.})
        self.status.set(f"{point['id']}: {len(point['observations'])} image observations; type {point.get('type', 'tie')}")
        self.draw()

    def new(self):
        value = simpledialog.askstring('New landmark', 'Unique landmark name:', parent=self.window)
        if not value or any(p['id'] == value for p in self.controls['points']): return
        self.snapshot(); self.controls['points'].append({'id': value, 'type': 'tie', 'observations': []})
        self.active.set(value); self.refresh()

    def delete(self):
        if not self.selected(): return
        self.snapshot(); name = self.active.get()
        self.controls['points'] = [p for p in self.controls['points'] if p['id'] != name]
        self.controls['distances'] = [d for d in self.controls['distances'] if name not in [d['a'], d['b']]]
        self.active.set(''); self.refresh()

    def set_type(self):
        point = self.selected()
        if not point: return
        value = simpledialog.askstring('Landmark type', 'tie, or gcp/check followed by local X Y Z coordinates in meters:', parent=self.window)
        if not value: return
        try:
            parts = value.split()
            if parts[0] not in ['tie', 'gcp', 'check'] or (parts[0] != 'tie' and len(parts) != 4): raise ValueError('Use: gcp 1.2 3.4 5.6')
            coordinates = [float(x) for x in parts[1:]]
            self.snapshot(); point['type'] = parts[0]
            if coordinates: point['position'] = coordinates; point['sigma_m'] = .01
            else: point.pop('position', None)
            self.refresh()
        except ValueError as error: messagebox.showerror('Invalid coordinates', str(error), parent=self.window)

    def distance(self):
        value = simpledialog.askstring('Distance constraint', 'First landmark, second landmark, distance in meters:', parent=self.window)
        if not value: return
        try:
            a, b, meters = value.split()
            if float(meters) <= 0 or any(name not in [p['id'] for p in self.controls['points']] for name in [a, b]): raise ValueError('Choose existing points and a positive distance')
            self.snapshot(); self.controls['distances'].append({'a': a, 'b': b, 'meters': float(meters), 'sigma_m': .01})
        except ValueError as error: messagebox.showerror('Invalid distance', str(error), parent=self.window)

    def undo(self):
        if self.history: self.controls = self.history.pop(); self.refresh()

    def load(self):
        path = filedialog.askopenfilename(parent=self.window, filetypes=[('Controls', '*.json')])
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            if payload.get('schema_version') != 1 or not isinstance(payload.get('points'), list): raise ValueError('Unsupported control schema')
            self.snapshot(); self.controls = payload; self.controls.setdefault('distances', [])
            self.active.set(self.controls['points'][0]['id'] if self.controls['points'] else ''); self.refresh()
        except Exception as error: messagebox.showerror('Cannot load', str(error), parent=self.window)

    def save(self):
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension='.json', initialfile='control_points.json')
        if path: Path(path).write_text(json.dumps(self.controls, indent=2), encoding='utf-8')

    def refine(self):
        if not self.reconstruction: return
        output = filedialog.asksaveasfilename(parent=self.window, initialfile='refined_reconstruction.json', defaultextension='.json')
        if not output: return
        controls_path = Path(output).with_suffix('.controls.json')
        controls_path.write_text(json.dumps(self.controls, indent=2), encoding='utf-8')
        source = self.reconstruction
        self.status.set('Refining cameras and landmarks; original reconstruction is preserved…')
        def work():
            try:
                from spherical_refinement import refine
                result = refine(source, output, controls_path)
                self.result_queue.put(f"Refined: angular RMS {result['before_angular_rms_deg']:.4f}° → {result['after_angular_rms_deg']:.4f}°. Saved {output}")
            except Exception as error: self.result_queue.put(f'Refinement failed: {error}')
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while not self.result_queue.empty(): self.status.set(self.result_queue.get_nowait())
        self.window.after(200, self.poll)


def open_editor(parent):
    return ControlPointEditor(parent)


if __name__ == '__main__':
    root = tk.Tk(); root.withdraw(); editor = ControlPointEditor(root)
    editor.window.protocol('WM_DELETE_WINDOW', root.destroy); root.mainloop()
