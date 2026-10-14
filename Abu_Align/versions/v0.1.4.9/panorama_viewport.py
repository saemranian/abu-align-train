"""Cursor-anchored ERP navigation and viewport-sized rendering on Tk canvases."""
import math
from dataclasses import dataclass
import tkinter as tk
from PIL import Image, ImageTk

BG = '#101619'


@dataclass
class ViewTransform:
    image_width: int = 1
    image_height: int = 1
    width: int = 600
    height: int = 400
    zoom: float = 1
    cx: float = .5
    cy: float = .5
    wrap: bool = False

    def fit(self):
        self.zoom = min(self.width/self.image_width, self.height/self.image_height)
        self.cx, self.cy = self.image_width/2, self.image_height/2
    def image_at(self, x, y):
        return self.cx + (x-self.width/2)/self.zoom, self.cy + (y-self.height/2)/self.zoom
    def screen_at(self, x, y):
        delta = x-self.cx
        if self.wrap and self.image_width*self.zoom > self.width:
            delta = (delta + self.image_width/2) % self.image_width - self.image_width/2
        return self.width/2 + delta*self.zoom, self.height/2 + (y-self.cy)*self.zoom
    def constrain(self):
        for axis, size, view in [('cx', self.image_width, self.width), ('cy', self.image_height, self.height)]:
            half = view/(2*self.zoom)
            value = getattr(self, axis)
            if axis == 'cx' and self.wrap and half*2 < size: value %= size
            elif half*2 >= size: value = size/2
            else: value = min(size-half, max(half, value))
            setattr(self, axis, value)
    def zoom_at(self, factor, x, y):
        before = self.image_at(x, y)
        fit = min(self.width/self.image_width, self.height/self.image_height)
        self.zoom = min(16., max(fit/4, self.zoom*factor))
        self.cx = before[0] - (x-self.width/2)/self.zoom
        self.cy = before[1] - (y-self.height/2)/self.zoom
        self.constrain()
    def pan(self, dx, dy):
        self.cx -= dx/self.zoom; self.cy -= dy/self.zoom; self.constrain()


class PanoramaAsset:
    def __init__(self, path, expected=None):
        with Image.open(path) as source:
            if expected and source.size != tuple(expected):
                raise ValueError(f'Image is {source.width}×{source.height}; reconstruction expects {expected[0]}×{expected[1]}. Choose the matching original images.')
            self.image = source.convert('RGB')
        self.levels = [(1, self.image)]
        while self.levels[-1][1].width > 768:
            factor, previous = self.levels[-1]
            self.levels.append((factor*2, previous.resize((max(1, previous.width//2), max(1, previous.height//2)), Image.Resampling.BILINEAR)))
        self.bytes = sum(image.width*image.height*3 for _, image in self.levels)


class PanoramaViewport:
    def __init__(self, parent, on_measure, on_select, on_navigation, on_focus):
        self.canvas = tk.Canvas(parent, bg=BG, highlightthickness=1, highlightbackground='#344149', takefocus=True)
        self.canvas.pack(fill='both', expand=True)
        self.transform = ViewTransform()
        self.asset = None
        self.markers = []
        self.prediction = None
        self.active = None
        self.marker_size = 6
        self.pixelated = False
        self.label_other_points = False
        self.hovered = None
        self.overlay = None
        self.pan_mode = False
        self.space = False
        self.drag = None
        self.photos = []
        self.render_job = None
        self.closed = False
        self.last_render_pixels = 0
        self.resize_settle_job = None
        self.resizing = False
        self.on_measure, self.on_select = on_measure, on_select
        self.on_image_navigation=None
        self.arrow_browse=True
        self.on_navigation, self.on_focus = on_navigation, on_focus
        self.canvas.bind('<Configure>', self.resize)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<Button-4>', lambda e: self.wheel(e, 1))
        self.canvas.bind('<Button-5>', lambda e: self.wheel(e, -1))
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.canvas.bind('<ButtonPress-2>', lambda e: self.press(e, force_pan=True))
        self.canvas.bind('<B2-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-2>', self.release)
        self.canvas.bind('<Button-3>', self.remove)
        self.canvas.bind('<Motion>', self.hover)
        self.canvas.bind('<Leave>', lambda e: self.clear_hover())
        self.canvas.bind('<KeyPress-space>', self.space_down)
        self.canvas.bind('<KeyRelease-space>', self.space_up)
        self.canvas.bind('<FocusOut>', lambda e: self.space_up(e))
        for key in ['Left', 'Right', 'Up', 'Down']:
            self.canvas.bind(f'<{key}>', self.arrow)
        self.canvas.bind('<f>', lambda e: self.fit())
        self.canvas.bind('<Key-1>', lambda e: self.actual())

    def set_asset(self, asset, preserve=False):
        old = self.transform
        fx, fy = old.cx/old.image_width, old.cy/old.image_height
        self.asset = asset
        old.image_width, old.image_height = asset.image.size
        if preserve: old.cx, old.cy = fx*old.image_width, fy*old.image_height; old.constrain()
        else: old.fit()
        self.request_draw()

    def resize(self, event):
        t = self.transform
        if (t.width, t.height) == (max(1, event.width), max(1, event.height)): return
        was_fit = abs(t.zoom-min(t.width/t.image_width, t.height/t.image_height)) < 1e-6
        t.width, t.height = max(1, event.width), max(1, event.height)
        if was_fit: t.fit()
        else: t.constrain()
        self.resizing = True
        if self.resize_settle_job: self.canvas.after_cancel(self.resize_settle_job)
        self.resize_settle_job = self.canvas.after(100, self.resize_settled)
        self.request_draw()
        self.on_navigation(self)

    def resize_settled(self):
        self.resize_settle_job = None; self.resizing = False; self.request_draw()

    def request_draw(self):
        if self.closed or self.render_job: return
        self.render_job = self.canvas.after(16, self.scheduled_draw)
    def scheduled_draw(self):
        self.render_job = None
        self.draw()

    def fit(self):
        self.transform.fit(); self.request_draw(); self.on_navigation(self)
    def actual(self):
        self.transform.zoom_at(1/self.transform.zoom, self.transform.width/2, self.transform.height/2)
        self.request_draw(); self.on_navigation(self)
    def focus(self, pixel):
        self.transform.zoom = max(self.transform.zoom, 1.)
        self.transform.cx, self.transform.cy = pixel
        self.transform.constrain(); self.request_draw(); self.on_navigation(self)
    def wheel(self, event, direction=None):
        if not self.asset: return 'break'
        delta = direction if direction is not None else max(-4, min(4, event.delta/120))
        self.transform.zoom_at(1.2**delta, event.x, event.y)
        self.request_draw(); self.on_navigation(self); return 'break'
    def space_down(self, event): self.space = True; self.canvas.configure(cursor='fleur'); return 'break'
    def space_up(self, event): self.space = False; self.canvas.configure(cursor='crosshair' if not self.pan_mode else 'fleur')

    def hit(self, event):
        for name, pixel, enabled in reversed(self.markers):
            x, y = self.transform.screen_at(*pixel)
            if math.hypot(x-event.x, y-event.y) < max(10, self.marker_size+3): return name
        return None
    def hover(self, event):
        name = self.hit(event)
        if name != self.hovered: self.hovered = name; self.request_draw()
    def clear_hover(self):
        self.hovered = None; self.request_draw()
    def press(self, event, force_pan=False):
        self.canvas.focus_set(); self.on_focus(self)
        if not self.asset: return
        pan = force_pan or self.space or self.pan_mode
        name = None if pan else self.hit(event)
        if name and self.on_select(name) is False: return
        self.drag = {'last': (event.x, event.y), 'start': (event.x, event.y), 'pan': pan, 'name': name, 'moved': False}
    def motion(self, event):
        if not self.drag: return
        dx, dy = event.x-self.drag['last'][0], event.y-self.drag['last'][1]
        if math.hypot(event.x-self.drag['start'][0], event.y-self.drag['start'][1]) > 3: self.drag['moved'] = True
        if self.drag['pan']: self.transform.pan(dx, dy); self.request_draw()
        elif self.drag['name'] and self.drag['moved']:
            pixel = self.valid_pixel(event.x, event.y)
            if pixel:
                self.markers = [(name, pixel if name == self.drag['name'] else old, enabled) for name, old, enabled in self.markers]
                self.request_draw()
        self.drag['last'] = (event.x, event.y)
    def release(self, event):
        drag, self.drag = self.drag, None
        if not drag: return
        if drag['pan']: self.on_navigation(self); return
        if drag['name'] and not drag['moved']: return
        pixel = self.valid_pixel(event.x, event.y)
        if pixel: self.on_measure(pixel, False)
    def valid_pixel(self, x, y):
        t = self.transform; px, py = t.image_at(x, y)
        if t.wrap and t.image_width*t.zoom > t.width: px %= t.image_width
        if 0 <= px < t.image_width and 0 <= py < t.image_height: return [px, py]
        return None
    def remove(self, event):
        name = self.hit(event)
        if name and self.on_select(name) is not False: self.on_measure(None, True)
        return 'break'
    def nudge(self, event):
        marker = next((pixel for name, pixel, _ in self.markers if name == self.active), None)
        if not marker: return
        step = .1 if event.state & 4 else 10 if event.state & 1 else 1
        dx, dy = {'Left': (-step, 0), 'Right': (step, 0), 'Up': (0, -step), 'Down': (0, step)}[event.keysym]
        pixel = [(marker[0]+dx) % self.transform.image_width,
                 min(self.transform.image_height-1e-6, max(0, marker[1]+dy))]
        self.on_measure(pixel, False); return 'break'
    def arrow(self,event):
        # Plain arrows browse; modifier arrows retain precise marker nudging.
        if self.arrow_browse and event.keysym in ['Left','Right'] and not (event.state & (1|4|8)) and self.on_image_navigation:
            self.on_image_navigation(-1 if event.keysym=='Left' else 1);return 'break'
        return self.nudge(event)

    def draw(self):
        if self.render_job:
            self.canvas.after_cancel(self.render_job)
        self.render_job = None
        if self.closed: return
        canvas, t = self.canvas, self.transform
        canvas.delete('all'); self.photos = []; self.last_render_pixels = 0
        if not self.asset:
            canvas.create_text(t.width/2, t.height/2, text=self.overlay or 'Open an alignment to begin', fill='#a0adb5', font=('Segoe UI', 12)); return
        factor, level = self.asset.levels[0]
        for candidate_factor, candidate in self.asset.levels:
            if candidate_factor*t.zoom <= (2 if self.resizing else 1): factor, level = candidate_factor, candidate
        left, top = t.image_at(0, 0); right, bottom = t.image_at(t.width, t.height)
        y0, y1 = max(0, top), min(t.image_height, bottom)
        wrapped = t.wrap and t.image_width*t.zoom > t.width
        shifts = range(math.floor(left/t.image_width), math.floor(right/t.image_width)+1) if wrapped else [0]
        if y1 > y0:
            for shift in shifts:
                x0, x1 = max(0, left-shift*t.image_width), min(t.image_width, right-shift*t.image_width)
                if x1 <= x0: continue
                box = (max(0, int(x0/factor)), max(0, int(y0/factor)), min(level.width, math.ceil(x1/factor)), min(level.height, math.ceil(y1/factor)))
                crop = level.crop(box)
                sw = max(1, math.ceil(crop.width*factor*t.zoom)); sh = max(1, math.ceil(crop.height*factor*t.zoom))
                photo = ImageTk.PhotoImage(crop.resize((sw, sh), Image.Resampling.NEAREST if self.pixelated or self.resizing else Image.Resampling.BILINEAR), master=canvas)
                self.photos.append(photo); self.last_render_pixels += sw*sh
                sx = (box[0]*factor+shift*t.image_width-t.cx)*t.zoom+t.width/2
                sy = (box[1]*factor-t.cy)*t.zoom+t.height/2
                canvas.create_image(sx, sy, image=photo, anchor='nw')
        if self.prediction:
            x, y = t.screen_at(*self.prediction)
            canvas.create_oval(x-12, y-12, x+12, y+12, outline='#8aa7dc', dash=(3, 3), width=2)
            canvas.create_text(x+16, y-16, text='Prediction · unmeasured', fill='#b5c7ed', anchor='w', font=('Segoe UI', 9))
        for name, pixel, enabled in self.markers:
            x, y = t.screen_at(*pixel); size = self.marker_size
            if x < -size or x > t.width+size or y < -size or y > t.height+size: continue
            color = '#ffce73' if name == self.active else '#70dbc0' if enabled else '#87959d'
            canvas.create_oval(x-size, y-size, x+size, y+size, outline='#0c1114', width=4)
            canvas.create_oval(x-size, y-size, x+size, y+size, outline=color, width=2)
            canvas.create_line(x-3, y, x+3, y, fill=color); canvas.create_line(x, y-3, x, y+3, fill=color)
            if name == self.active or name == self.hovered or self.label_other_points:
                canvas.create_text(x+size+5, y-size-6, text=name, fill='#0c1114', anchor='w', font=('Segoe UI', 10, 'bold'))
                canvas.create_text(x+size+4, y-size-7, text=name, fill=color, anchor='w', font=('Segoe UI', 10, 'bold'))
        if self.overlay:
            canvas.create_rectangle(12, 12, t.width-12, 50, fill='#1a2024', outline='#344149')
            canvas.create_text(t.width/2, 31, text=self.overlay, fill='#70dbc0', font=('Segoe UI', 10))

    def close(self):
        self.closed = True
        if self.resize_settle_job:
            self.canvas.after_cancel(self.resize_settle_job); self.resize_settle_job = None
        if self.render_job:
            self.canvas.after_cancel(self.render_job); self.render_job = None
        self.asset = None; self.photos.clear()
