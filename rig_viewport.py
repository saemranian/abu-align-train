"""Antialiased, CPU-only rig preview. This does not change exported camera geometry."""
import math
from PIL import Image, ImageDraw, ImageTk


def basis(yaw, pitch, roll=0):
    """Orthonormal right/up/forward basis, with roll applied simultaneously."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    forward = (math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y))
    right = (math.cos(y), 0, -math.sin(y))
    up = (-math.sin(p) * math.sin(y), math.cos(p), -math.sin(p) * math.cos(y))
    rolled_right = tuple(a * math.cos(r) + b * math.sin(r) for a, b in zip(right, up))
    rolled_up = tuple(b * math.cos(r) - a * math.sin(r) for a, b in zip(right, up))
    return rolled_right, rolled_up, forward


def camera_corners(camera, aspect, distance=1):
    _, yaw, pitch, roll, fov = camera
    right, up, forward = basis(yaw, pitch, roll)
    half_w = distance * math.tan(math.radians(fov / 2))
    half_h = half_w / max(.1, aspect)
    return [tuple(distance * forward[i] + x * half_w * right[i] + y * half_h * up[i]
                  for i in range(3)) for x, y in [(-1, 1), (1, 1), (1, -1), (-1, -1)]]


class RigSceneRenderer:
    """Render smooth reference geometry and pickable camera markers on a Tk canvas."""
    def __init__(self, canvas):
        self.canvas = canvas
        self.selected = 0
        self.show_all = False
        self.show_labels = True
        self.show_grid = True
        self.positions = []
        self.photo = None

    def pick(self, x, y):
        candidates = [(math.hypot(px - x, py - y), depth, index)
                      for index, px, py, depth in self.positions]
        candidates = [hit for hit in candidates if hit[0] <= 20]
        if not candidates:
            return False
        self.selected = min(candidates)[2]
        return True

    def render(self, rig, orbit, aspect=1):
        canvas = self.canvas
        width, height = max(20, canvas.winfo_width()), max(20, canvas.winfo_height())
        self.selected = min(self.selected, max(0, len(rig) - 1))
        scale = 2 if width * height < 1500000 else 1
        image = Image.new("RGB", (width * scale, height * scale), "#11191e")
        draw = ImageDraw.Draw(image, "RGBA")
        vr, vu, vf = basis(orbit["yaw"], orbit["pitch"])
        # Orthographic orbit keeps glyph size stable and avoids near-plane clipping.
        zoom = 3.4 / max(1.2, orbit["dist"])
        pixels = min(width, max(100, height - 130)) * .29 * zoom
        cx, cy = width / 2 + orbit["pan_x"], height / 2 + 15 + orbit["pan_y"]

        def project(point):
            return (cx + sum(a * b for a, b in zip(point, vr)) * pixels,
                    cy - sum(a * b for a, b in zip(point, vu)) * pixels,
                    -sum(a * b for a, b in zip(point, vf)))

        def line(points, color, thickness=1):
            draw.line([(x * scale, y * scale) for x, y, _ in map(project, points)],
                      fill=color, width=max(1, int(thickness * scale)), joint="curve")

        def face(points, color):
            draw.polygon([(x * scale, y * scale) for x, y, _ in map(project, points)], fill=color)

        # A quiet ground plane and sphere provide orientation without covering the rig.
        if self.show_grid:
            for i in range(-5, 6):
                k = i * .3
                line([(k, -1.18, -1.5), (k, -1.18, 1.5)], (58, 77, 87, 70))
                line([(-1.5, -1.18, k), (1.5, -1.18, k)], (58, 77, 87, 70))
            for latitude in [-60, -30, 0, 30, 60]:
                lat = math.radians(latitude)
                points = [(math.cos(lat) * math.sin(i * math.pi / 48), math.sin(lat),
                           math.cos(lat) * math.cos(i * math.pi / 48)) for i in range(97)]
                line(points, (88, 115, 125, 90 if latitude == 0 else 38))
            for longitude in [0, 60, 120]:
                lon = math.radians(longitude)
                line([(math.sin(i * math.pi / 48) * math.sin(lon), math.cos(i * math.pi / 48),
                       math.sin(i * math.pi / 48) * math.cos(lon)) for i in range(97)], (88, 115, 125, 38))

        # Only the selected camera is filled by default. Others remain quiet markers.
        cameras = []
        for index, camera in enumerate(rig):
            right, up, forward = basis(*camera[1:4])
            cameras.append((project(forward)[2], index, camera, right, up, forward))
        cameras.sort(reverse=True)
        for depth, index, camera, right, up, forward in cameras:
            selected = index == self.selected
            if selected or self.show_all:
                # Scale the footprint onto a common sphere, preserving its FOV/aspect.
                corners = camera_corners(camera, aspect)
                length = math.sqrt(sum(x * x for x in corners[0]))
                corners = [tuple(x / length for x in corner) for corner in corners]
                face(corners, (112, 219, 192, 35 if selected else 9))
                line(corners + corners[:1], (112, 219, 192, 235 if selected else 55), 1.5 if selected else 1)
                if selected:
                    for corner in corners:
                        line([(0, 0, 0), corner], (112, 219, 192, 85))
                    line([(0, 0, 0), forward], (112, 219, 192, 155), 1.5)
            # Small oriented camera tiles show placement without the overlapping cones.
            tile = [tuple(forward[i] + x * .10 * right[i] + y * .075 * up[i]
                          for i in range(3)) for x, y in [(-1, 1), (1, 1), (1, -1), (-1, -1)]]
            face(tile, (48, 89, 82, 255) if selected else (35, 50, 58, 255))
            line(tile + tile[:1], (112, 219, 192, 255) if selected else (107, 138, 153, 150), 1.5)

        self.photo = ImageTk.PhotoImage(image.resize((width, height), Image.Resampling.LANCZOS), master=canvas)
        canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="scene")
        self.positions = []
        for depth, index, camera, right, up, forward in cameras:
            x, y, _ = project(forward)
            self.positions.append((index, x, y, depth))
            selected = index == self.selected
            radius = 5 if selected else 3
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                               fill="#70dbc0" if selected else "#93a8b2", outline="")
            if self.show_labels or selected:
                canvas.create_text(x, y - 19, text=f"{index + 1:02d}",
                                   fill="#b3f1df" if selected else "#9caeb7",
                                   font=("Segoe UI", 9, "bold" if selected else "normal"))
        ox, oy, _ = project((0, 0, 0))
        canvas.create_oval(ox-4, oy-4, ox+4, oy+4, fill="#e7edef", outline="")
        if rig:
            name, yaw, pitch, roll, fov = rig[self.selected]
            canvas.create_text(22, 22, anchor="nw", text=f"Camera {self.selected + 1:02d}  /  {len(rig):02d}",
                               fill="#e7edef", font=("Segoe UI", 12, "bold"))
            canvas.create_text(22, 49, anchor="nw", text=f"{name}   ·   FOV {fov:g}°   ·   yaw {yaw:g}° / pitch {pitch:g}°",
                               fill="#9caeb7", font=("Segoe UI", 9), width=max(100, width-44))
        canvas.create_text(22, height - 22, anchor="sw", text="Drag to orbit   ·   Scroll to zoom   ·   Click a camera to inspect",
                           fill="#9caeb7", font=("Segoe UI", 9), width=max(100, width - 105))
        # Small orientation gizmo, kept away from camera labels.
        gx, gy = width - 46, height - 48
        for name, axis, color in [("X", (1, 0, 0), "#ca9290"), ("Y", (0, 1, 0), "#70dbc0"), ("Z", (0, 0, 1), "#8eb4cf")]:
            dx = sum(a*b for a, b in zip(axis, vr)) * 23
            dy = -sum(a*b for a, b in zip(axis, vu)) * 23
            canvas.create_line(gx, gy, gx+dx, gy+dy, fill=color, width=2)
            canvas.create_text(gx+dx*1.4, gy+dy*1.4, text=name, fill=color, font=("Segoe UI", 8))
