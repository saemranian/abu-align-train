"""
Abu Align & Train - Native 3D Workspace Window
==============================================
Interactive CAD/Photogrammetry 3D viewport embedded directly inside Abu Align & Train.
Visualizes reconstructed dense/sparse 3D point clouds, camera trajectories, and frustums
with real-time orbit, pan, zoom, custom color ramps, and multi-run inspection.
"""

import os
import glob
import math
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional, Any, Callable

import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

# Styling palette matching Abu Align & Train desktop UI
BG = "#111518"
SURFACE = "#1a2024"
FIELD = "#242d32"
BORDER = "#344149"
TEXT = "#e7edef"
MUTED = "#a0adb5"
ACCENT = "#70dbc0"
HIGHLIGHT = "#38bdf8"
CAM_COLOR = "#f59e0b"
TRAJ_COLOR = "#38bdf8"


# =========================================================================
# Geometry & Color Math
# =========================================================================

def make_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Computes a 3x3 rotation matrix from Euler angles (in degrees)."""
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    # Rz(roll) * Rx(pitch) * Ry(yaw)
    Ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ], dtype=np.float32)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(p), -math.sin(p)],
        [0, math.sin(p), math.cos(p)]
    ], dtype=np.float32)

    Rz = np.array([
        [math.cos(r), -math.sin(r), 0],
        [math.sin(r), math.cos(r), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return Rz @ (Rx @ Ry)


def turbo_colormap(val: np.ndarray) -> np.ndarray:
    """Vectorized Turbo colormap approximation for depth values in [0, 1]."""
    x = np.clip(val, 0.0, 1.0)
    # Polynomial approximation of Turbo colormap
    r = 0.1357 + x * (4.6153 - x * (42.6603 - x * (132.131 - x * (152.55 - x * 59.286))))
    g = 0.0914 + x * (2.1941 + x * (4.8429 - x * (14.185 - x * (4.277 - x * 2.829))))
    b = 0.1067 + x * (12.592 - x * (60.197 - x * (109.07 - x * (88.50 - x * 26.81))))
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def viridis_colormap(val: np.ndarray) -> np.ndarray:
    """Elevation colormap from cyan/teal to yellow."""
    x = np.clip(val, 0.0, 1.0)
    r = (0.2 + 0.8 * (x ** 1.5)) * 255.0
    g = (0.3 + 0.7 * np.sin(x * math.pi * 0.8)) * 255.0
    b = (0.8 - 0.7 * x) * 255.0
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def resolve_camera_photo_path(cam_name: str, run_dir: str = "", dataset_dir: Optional[str] = None) -> Optional[str]:
    """Resolves local absolute image file path for a camera from run or dataset directories."""
    if not cam_name:
        return None
    clean_name = os.path.basename(cam_name)

    if os.path.isabs(cam_name) and os.path.isfile(cam_name):
        return cam_name

    dirs_to_check = []
    if run_dir and os.path.isdir(run_dir):
        dirs_to_check.extend([
            run_dir,
            os.path.join(run_dir, "images"),
            os.path.join(run_dir, "3dgs_dataset", "images"),
            os.path.join(run_dir, "input"),
            os.path.dirname(run_dir),
            os.path.dirname(os.path.dirname(run_dir)),
            os.path.join(os.path.dirname(os.path.dirname(run_dir)), "images"),
            os.path.join(os.path.dirname(os.path.dirname(run_dir)), "input"),
        ])
    if dataset_dir and os.path.isdir(dataset_dir):
        dirs_to_check.extend([
            dataset_dir,
            os.path.join(dataset_dir, "images"),
            os.path.join(dataset_dir, "input"),
            os.path.join(dataset_dir, "photos"),
        ])

    for d in dirs_to_check:
        if not d or not os.path.isdir(d):
            continue
        c1 = os.path.join(d, cam_name)
        if os.path.isfile(c1):
            return c1
        c2 = os.path.join(d, clean_name)
        if os.path.isfile(c2):
            return c2
        base_no_ext, _ = os.path.splitext(clean_name)
        for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp", ".BMP"]:
            c3 = os.path.join(d, base_no_ext + ext)
            if os.path.isfile(c3):
                return c3
    return None


# =========================================================================
# Reconstruction Data Loaders
# =========================================================================

def read_ply_fast(ply_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fast binaryLittleEndian and ASCII PLY reader."""
    if not os.path.isfile(ply_path):
        return None, None

    try:
        with open(ply_path, "rb") as f:
            header = []
            while True:
                line = f.readline().decode("utf-8", errors="ignore").strip()
                header.append(line)
                if line == "end_header":
                    break

            format_line = [h for h in header if h.startswith("format")][0]
            element_line = [h for h in header if h.startswith("element vertex")][0]
            num_verts = int(element_line.split()[-1])

            if "binary_little_endian" in format_line:
                props = [h.split() for h in header if h.startswith("property")]
                dt = []
                for pr in props:
                    t, name = pr[1], pr[2]
                    if t in ["float", "float32"]: dt.append((name, "<f4"))
                    elif t in ["uchar", "uint8"]: dt.append((name, "u1"))
                    elif t in ["int", "int32"]: dt.append((name, "<i4"))
                    elif t in ["double", "float64"]: dt.append((name, "<f8"))
                    else: dt.append((name, "<f4"))

                buf = f.read()
                data = np.frombuffer(buf, dtype=np.dtype(dt), count=num_verts)
                pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)

                if "red" in data.dtype.names and "green" in data.dtype.names and "blue" in data.dtype.names:
                    cols = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
                else:
                    cols = np.full((len(pts), 3), 180, dtype=np.uint8)
                return pts, cols

            elif "ascii" in format_line:
                pts, cols = [], []
                for _ in range(min(num_verts, 300000)):
                    parts = f.readline().decode("utf-8", errors="ignore").strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
                        else:
                            cols.append([180, 180, 180])
                return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)
    except Exception:
        pass

    # Trimesh fallback
    try:
        import trimesh
        mesh = trimesh.load(ply_path)
        pts = np.array(mesh.vertices, dtype=np.float32)
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            cols = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)
        return pts, cols
    except Exception:
        return None, None


def load_cameras_from_run(run_dir: str) -> List[Dict[str, Any]]:
    """Loads camera poses from transforms.json, COLMAP images.txt, or cameras.csv."""
    cameras = []
    if not os.path.isdir(run_dir):
        return cameras

    # 1. Check transforms.json (NeRF / 3DGS standard)
    cand_transforms = [
        os.path.join(run_dir, "3dgs_dataset", "transforms.json"),
        os.path.join(run_dir, "Postshot_RealityCapture", "transforms.json"),
        os.path.join(run_dir, "transforms.json"),
    ]
    for tf_path in cand_transforms:
        if os.path.isfile(tf_path):
            try:
                with open(tf_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                frames = data.get("frames", [])
                for i, frame in enumerate(frames):
                    c2w = np.array(frame.get("transform_matrix", np.eye(4)), dtype=np.float32)
                    cam_pos = c2w[:3, 3]
                    # OpenCV look vector (+Z forward in camera space)
                    forward = c2w[:3, 2]
                    up = -c2w[:3, 1]
                    right = c2w[:3, 0]
                    name = os.path.basename(frame.get("file_path", f"frame_{i:04d}"))
                    cameras.append({
                        "name": name,
                        "file_path": frame.get("file_path", ""),
                        "pos": cam_pos,
                        "forward": forward,
                        "up": up,
                        "right": right,
                        "c2w": c2w,
                    })
                if cameras:
                    return cameras
            except Exception:
                pass

    # 2. Check COLMAP images.txt
    cand_colmap = [
        os.path.join(run_dir, "COLMAP_Lichtfield", "images.txt"),
        os.path.join(run_dir, "3dgs_dataset", "sparse", "0", "images.txt"),
        os.path.join(run_dir, "sparse", "0", "images.txt"),
    ]
    for col_path in cand_colmap:
        if os.path.isfile(col_path):
            try:
                with open(col_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 10:
                        # COLMAP image line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                        qw, qx, qy, qz = map(float, parts[1:5])
                        tx, ty, tz = map(float, parts[5:8])
                        name = parts[9]

                        # Quaternion to rotation matrix
                        R_w2c = np.array([
                            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
                            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
                        ], dtype=np.float32)
                        t_w2c = np.array([tx, ty, tz], dtype=np.float32)
                        R_c2w = R_w2c.T
                        pos = -R_c2w @ t_w2c

                        cameras.append({
                            "name": name,
                            "file_path": name,
                            "pos": pos,
                            "forward": R_c2w[:, 2],
                            "up": -R_c2w[:, 1],
                            "right": R_c2w[:, 0],
                        })
                if cameras:
                    return cameras
            except Exception:
                pass

    return cameras


# =========================================================================
# Native 3D Workspace Component
# =========================================================================

class _LegacyWorkspace3DViewer(tk.Frame):
    """
    Main CAD-grade 3D photogrammetry and alignment inspection workspace.
    Renders dense point clouds and registered camera trajectories with 6-DOF controls.
    """

    def __init__(self, parent, dataset_provider: Optional[Callable[[], str]] = None, **kwargs):
        super().__init__(parent, bg=SURFACE, **kwargs)
        self.dataset_provider = dataset_provider

        # Camera & Navigation Orbit State
        self.yaw = -25.0
        self.pitch = 20.0
        self.distance = 5.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Scene Data
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.cameras: List[Dict[str, Any]] = []
        self.center = np.zeros(3, dtype=np.float32)
        self.scale = 1.0
        self.active_run_dir = ""

        # Display Toggles
        self.color_mode_var = tk.StringVar(value="rgb")
        self.point_size_var = tk.IntVar(value=1)
        self.show_cameras_var = tk.BooleanVar(value=True)
        self.show_trajectory_var = tk.BooleanVar(value=True)
        self.cam_scale_var = tk.DoubleVar(value=1.0)
        self.run_choice_var = tk.StringVar(value="Latest Run (Auto)")

        # Mouse Tracking
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._is_dragging = False
        self._thumb_cache: Dict[str, Any] = {}
        self._thumb_executor = ThreadPoolExecutor(max_workers=4)
        self._thumb_redraw_scheduled = False
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._build_toolbar()
        self._build_canvas()
        self._bind_mouse_events()

    def _schedule_thumb_redraw(self):
        if not getattr(self, "_thumb_redraw_scheduled", False):
            self._thumb_redraw_scheduled = True
            try:
                self.after(150, self._on_thumb_redraw)
            except Exception:
                self._thumb_redraw_scheduled = False

    def _on_thumb_redraw(self):
        self._thumb_redraw_scheduled = False
        if not getattr(self, "_is_dragging", False):
            self.render()

    def _get_camera_thumbnail(self, cam_name: str, file_path: Optional[str] = None) -> Optional[Image.Image]:
        """Asynchronously loads and caches downsampled thumbnail for frustum overlay."""
        key = cam_name or file_path or ""
        if not key:
            return None
        if key in self._thumb_cache:
            val = self._thumb_cache[key]
            return val if isinstance(val, Image.Image) else None

        self._thumb_cache[key] = "queued"

        dataset_dir = None
        if hasattr(self, "master") and hasattr(self.master, "dataset_var"):
            dataset_dir = self.master.dataset_var.get()
        run_dir = getattr(self, "active_run_dir", "")

        def _fetch():
            img_path = resolve_camera_photo_path(key, run_dir, dataset_dir)
            if not img_path:
                self._thumb_cache[key] = None
                return
            try:
                with Image.open(img_path) as im:
                    try:
                        im.draft("RGB", (128, 128))
                    except Exception:
                        pass
                    im = im.convert("RGB")
                    im.thumbnail((128, 128), Image.Resampling.BILINEAR)
                    thumb = im.copy()
                    self._thumb_cache[key] = thumb
                    self._schedule_thumb_redraw()
            except Exception:
                self._thumb_cache[key] = None

        if hasattr(self, "_thumb_executor"):
            try:
                self._thumb_executor.submit(_fetch)
            except Exception:
                pass
        return None

    def destroy(self):
        if hasattr(self, "_thumb_executor"):
            try:
                self._thumb_executor.shutdown(wait=False)
            except Exception:
                pass
        super().destroy()

    def _build_toolbar(self):
        """Builds top inspection and control toolbar."""
        bar = tk.Frame(self, bg=SURFACE, padx=12, pady=10)
        bar.pack(fill="x", side="top")

        # Run Selector
        tk.Label(bar, text="Alignment Run:", bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))
        self.run_combo = ttk.Combobox(
            bar, textvariable=self.run_choice_var, width=24, state="readonly", postcommand=self.refresh_runs
        )
        self.run_combo.pack(side="left", padx=(0, 6))
        self.run_combo.bind("<<ComboboxSelected>>", lambda e: self.load_selected_run())

        load_btn = tk.Button(
            bar, text="Load / Refresh", command=self.load_selected_run,
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=10, pady=4, font=("Segoe UI", 9), cursor="hand2"
        )
        load_btn.pack(side="left", padx=(0, 8))

        report_btn = tk.Button(
            bar, text="📋 Reconstruction Report", command=self.show_reconstruction_report,
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=10, pady=4, font=("Segoe UI", 9), cursor="hand2"
        )
        report_btn.pack(side="left", padx=(0, 16))

        # Color Mode Dropdown
        tk.Label(bar, text="Color:", bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        color_combo = ttk.Combobox(
            bar, textvariable=self.color_mode_var, width=12, state="readonly",
            values=["rgb", "depth", "elevation", "white"]
        )
        color_combo.pack(side="left", padx=(0, 14))
        color_combo.bind("<<ComboboxSelected>>", lambda e: self.render())

        # Point Size Control
        tk.Label(bar, text="Size:", bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        size_combo = ttk.Combobox(
            bar, textvariable=self.point_size_var, width=4, state="readonly",
            values=[1, 2, 3, 4]
        )
        size_combo.pack(side="left", padx=(0, 14))
        size_combo.bind("<<ComboboxSelected>>", lambda e: self.render())

        # Camera Toggles
        cam_chk = tk.Checkbutton(
            bar, text="Cameras", variable=self.show_cameras_var,
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE,
            activeforeground=TEXT, font=("Segoe UI", 9), command=self.render
        )
        cam_chk.pack(side="left", padx=(0, 8))

        traj_chk = tk.Checkbutton(
            bar, text="Trajectory", variable=self.show_trajectory_var,
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE,
            activeforeground=TEXT, font=("Segoe UI", 9), command=self.render
        )
        traj_chk.pack(side="left", padx=(0, 16))

        # Reset / Preset View Buttons
        reset_btn = tk.Button(
            bar, text="Reset View", command=self.reset_view,
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=8, pady=4, font=("Segoe UI", 9), cursor="hand2"
        )
        reset_btn.pack(side="left", padx=(0, 6))

        top_btn = tk.Button(
            bar, text="Top View", command=lambda: self.set_preset_view(0, 89.9),
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=8, pady=4, font=("Segoe UI", 9), cursor="hand2"
        )
        top_btn.pack(side="left", padx=(0, 14))

        # Stats Badge
        self.stats_lbl = tk.Label(
            bar, text="No model loaded", bg=FIELD, fg=ACCENT,
            font=("Segoe UI", 9, "bold"), padx=10, pady=4, relief="flat"
        )
        self.stats_lbl.pack(side="right")

    def _build_canvas(self):
        """Creates the primary 3D rendering canvas."""
        canvas_frame = tk.Frame(self, bg=BG)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())

    def _bind_mouse_events(self):
        """Attaches CAD/DCC navigation handlers."""
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_orbit_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self.canvas.bind("<ButtonPress-2>", self._on_mouse_down)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_mouse_up)

        self.canvas.bind("<ButtonPress-3>", self._on_mouse_down)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_mouse_up)

        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self._zoom(0.85))  # Linux up
        self.canvas.bind("<Button-5>", lambda e: self._zoom(1.15))  # Linux down
        self.canvas.bind("<Double-Button-1>", lambda e: self.reset_view())

    def _on_mouse_down(self, event):
        self._is_dragging = True
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

    def _on_mouse_up(self, event):
        if self._is_dragging:
            self._is_dragging = False
            self.render()

    def _on_orbit_drag(self, event):
        self._is_dragging = True
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        # Match browser orbit controls: horizontal drag rotates the view oppositely.
        self.yaw -= dx * 0.5
        self.pitch = max(-89.9, min(89.9, self.pitch + dy * 0.5))
        self.render()

    def _on_pan_drag(self, event):
        self._is_dragging = True
        dx = event.x - self._last_mouse_x
        dy = event.y - self._last_mouse_y
        self._last_mouse_x = event.x
        self._last_mouse_y = event.y

        pan_speed = max(0.001, self.distance * 0.0015)
        self.pan_x += dx * pan_speed
        self.pan_y += dy * pan_speed
        self.render()

    def _on_mouse_wheel(self, event):
        factor = 0.9 if event.delta > 0 else 1.1
        self._zoom(factor)

    def _zoom(self, factor: float):
        self.distance = max(0.1, min(500.0, self.distance * factor))
        self.render()

    def set_preset_view(self, yaw: float, pitch: float):
        self.yaw = yaw
        self.pitch = pitch
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render()

    def reset_view(self):
        self.yaw = -25.0
        self.pitch = 20.0
        self.distance = 5.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.render()

    def _resolve_results_dir(self) -> Optional[str]:
        """Resolves results/ directory for dataset or its parent directory."""
        dataset_dir = self.dataset_provider() if self.dataset_provider else ""
        if not dataset_dir or not os.path.isdir(dataset_dir):
            return None
        cand_results = os.path.join(dataset_dir, "results")
        if os.path.isdir(cand_results):
            return cand_results
        parent_results = os.path.join(os.path.dirname(dataset_dir), "results")
        if os.path.isdir(parent_results):
            return parent_results
        return None

    def refresh_runs(self) -> List[str]:
        """Discovers all available sparse_* reconstruction runs in the current dataset."""
        runs = ["Latest Run (Auto)"]
        results_dir = self._resolve_results_dir()
        if results_dir and os.path.isdir(results_dir):
            found = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("sparse_")]
            found.sort(key=lambda d: os.path.getmtime(os.path.join(results_dir, d)), reverse=True)
            runs.extend(found)
        self.run_combo["values"] = runs
        cur = self.run_choice_var.get()
        if cur not in runs:
            self.run_choice_var.set(runs[0])
        return runs

    def load_selected_run(self):
        """Loads points and cameras for the selected run."""
        results_dir = self._resolve_results_dir()
        if not results_dir or not os.path.isdir(results_dir):
            self.stats_lbl.configure(text="No results/ folder found")
            return

        choice = self.run_choice_var.get()
        if choice == "Latest Run (Auto)" or not choice:
            found = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("sparse_")]
            found.sort(key=lambda d: os.path.getmtime(os.path.join(results_dir, d)), reverse=True)
            if not found:
                self.stats_lbl.configure(text="No sparse_* runs found")
                return
            target_run_dir = os.path.join(results_dir, found[0])
            run_name = found[0]
        else:
            target_run_dir = os.path.join(results_dir, choice)
            run_name = choice

        self.active_run_name = run_name
        self.active_run_dir = target_run_dir

        # Look for point cloud
        ply_candidates = [
            os.path.join(target_run_dir, "points3D.ply"),
            os.path.join(target_run_dir, "3dgs_dataset", "points.ply"),
            os.path.join(target_run_dir, "Postshot_RealityCapture", "points.ply"),
            os.path.join(target_run_dir, "reconstruction.ply"),
        ]
        found_ply = next((p for p in ply_candidates if os.path.isfile(p)), None)

        pts, cols = (None, None)
        if found_ply:
            pts, cols = read_ply_fast(found_ply)

        cameras = load_cameras_from_run(target_run_dir)

        if pts is not None and len(pts) > 0:
            # Downsample if overly dense for smooth 60fps interaction
            if len(pts) > 300000:
                stride = int(math.ceil(len(pts) / 250000))
                pts = pts[::stride]
                if cols is not None:
                    cols = cols[::stride]

            # Compute bounding box and normalize center
            bbox_min = np.min(pts, axis=0)
            bbox_max = np.max(pts, axis=0)
            center = (bbox_min + bbox_max) * 0.5
            extent = np.linalg.norm(bbox_max - bbox_min)
            scale = max(0.001, extent)

            self.points = pts
            self.colors = cols
            self.center = center.astype(np.float32)
            self.scale = float(scale)
            self.distance = 2.5
        elif cameras:
            cam_centers = np.array([c["pos"] for c in cameras], dtype=np.float32)
            self.points = None
            self.colors = None
            self.center = np.mean(cam_centers, axis=0)
            self.scale = max(1.0, float(np.linalg.norm(np.max(cam_centers, axis=0) - np.min(cam_centers, axis=0))))
            self.distance = 2.5
        else:
            self.points = None
            self.colors = None
            self.cameras = []
            self.stats_lbl.configure(text=f"Empty run: {run_name}")
            self.render()
            return

        self.cameras = cameras
        pt_count = len(self.points) if self.points is not None else 0
        cam_count = len(self.cameras)
        self.stats_lbl.configure(text=f"{run_name} · {pt_count:,} Pts · {cam_count} Cams")
        self.render()

    def render(self):
        """Renders the 3D scene onto the Tkinter canvas."""
        w = max(20, self.canvas.winfo_width())
        h = max(20, self.canvas.winfo_height())

        # Blank background
        frame = np.full((h, w, 3), 17, dtype=np.uint8)  # #111518

        if self.points is None and not self.cameras:
            self.canvas.delete("all")
            self.canvas.create_text(
                w // 2, h // 2,
                text="No 3D Alignment Run Loaded\nSelect a run from the toolbar or complete an alignment to inspect",
                fill=MUTED, font=("Segoe UI", 12), justify="center"
            )
            return

        # View Matrix (World to View)
        R_view = self.view_matrix if self.view_matrix is not None else make_rotation_matrix(self.yaw, self.pitch, 0.0)

        # Draw Grid on ground (XZ plane at bottom of bounding box)
        grid_lines = []
        grid_size = 1.2
        steps = 10
        y_ground = 0.5  # In normalized scene space
        for gi in np.linspace(-grid_size, grid_size, steps + 1):
            p1 = np.array([gi, y_ground, -grid_size], dtype=np.float32)
            p2 = np.array([gi, y_ground, grid_size], dtype=np.float32)
            grid_lines.extend([(p1, p2), (np.array([-grid_size, y_ground, gi], dtype=np.float32), np.array([grid_size, y_ground, gi], dtype=np.float32))])

        # 1. Project Point Cloud
        is_drag = getattr(self, "_is_dragging", False)
        if self.points is not None and len(self.points) > 0 and (not hasattr(self, 'show_points_var') or self.show_points_var.get()):
            if is_drag and len(self.points) > 25000:
                stride = max(1, len(self.points) // 25000)
                pts_active = self.points[::stride]
                cols_active = self.colors[::stride] if self.colors is not None and len(self.colors) == len(self.points) else None
            else:
                pts_active = self.points
                cols_active = self.colors

            # Normalize points: centered and scaled to ~[-1, 1]
            p_norm = (pts_active - self.center) / self.scale

            # Rotate into view space
            p_rot = p_norm @ R_view.T

            # Translate by orbit camera position
            tz = self.distance
            x_cam = p_rot[:, 0] + self.pan_x
            y_cam = p_rot[:, 1] + self.pan_y
            z_cam = p_rot[:, 2] + tz

            # Perspective projection
            valid = z_cam > 0.005
            if np.any(valid):
                zc = z_cam[valid]
                focal = min(w, h) * 1.05
                px = (x_cam[valid] / zc * focal + (w * 0.5)).astype(np.int32)
                py = (y_cam[valid] / zc * focal + (h * 0.5)).astype(np.int32)

                # Bounds clipping
                in_screen = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                px = px[in_screen]
                py = py[in_screen]

                # Resolve color mode
                mode = "rgb" if self.compare_mode.get() == "Overlay" or self.distance_colors.get() else self.color_mode_var.get()
                if mode == "rgb" and cols_active is not None and len(cols_active) == len(pts_active):
                    pt_cols = cols_active[valid][in_screen]
                elif mode == "depth" and len(px):
                    depths = zc[in_screen]
                    d_norm = (depths - np.min(depths)) / max(0.001, np.max(depths) - np.min(depths))
                    pt_cols = turbo_colormap(d_norm)
                elif mode == "elevation" and len(px):
                    elevs = p_norm[valid, 1][in_screen]
                    e_norm = (elevs - np.min(elevs)) / max(0.001, np.max(elevs) - np.min(elevs))
                    pt_cols = viridis_colormap(e_norm)
                else:
                    pt_cols = np.full((len(px), 3), 230, dtype=np.uint8)

                if not is_drag:
                    order = np.argsort(-zc[in_screen])
                    px, py, pt_cols = px[order], py[order], pt_cols[order]
                    pt_size = self.point_size_var.get()
                else:
                    pt_size = 1

                # Rasterize points to frame buffer
                if pt_size <= 1:
                    frame[py, px] = pt_cols
                else:
                    # Splat slightly larger point
                    for off_y in range(-(pt_size // 2), pt_size - pt_size // 2):
                        for off_x in range(-(pt_size // 2), pt_size - pt_size // 2):
                            sy = np.clip(py + off_y, 0, h - 1)
                            sx = np.clip(px + off_x, 0, w - 1)
                            frame[sy, sx] = pt_cols

        # Blit rasterized point image to Tkinter canvas
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # 2. Render Ground Grid Lines
        focal = min(w, h) * 1.05
        tz = self.distance
        for p1, p2 in grid_lines:
            r1 = (p1 @ R_view.T) + np.array([self.pan_x, self.pan_y, tz])
            r2 = (p2 @ R_view.T) + np.array([self.pan_x, self.pan_y, tz])
            if r1[2] > 0.1 and r2[2] > 0.1:
                sx1 = int(r1[0] / r1[2] * focal + w * 0.5)
                sy1 = int(r1[1] / r1[2] * focal + h * 0.5)
                sx2 = int(r2[0] / r2[2] * focal + w * 0.5)
                sy2 = int(r2[1] / r2[2] * focal + h * 0.5)
                if -w < sx1 < 2 * w and -h < sy1 < 2 * h:
                    draw.line([(sx1, sy1), (sx2, sy2)], fill="#1f2930", width=1)

        # 3. Render Cameras & Trajectories
        cam_screen_pts = []
        self._camera_screen = []
        if self.cameras:
            cam_scale = 0.025 * self.cam_scale_var.get()
            for i, cam in enumerate(self.cameras):
                c_pos = (cam["pos"] - self.center) / self.scale
                c_rot = c_pos @ R_view.T
                c_cam = c_rot + np.array([self.pan_x, self.pan_y, tz])

                if c_cam[2] > 0.1:
                    csx = int(c_cam[0] / c_cam[2] * focal + w * 0.5)
                    csy = int(c_cam[1] / c_cam[2] * focal + h * 0.5)
                    cam_screen_pts.append((csx, csy, c_cam[2], i))

                    selected = i == self.selected_camera
                    color = "#ffdc7d" if selected else tuple(cam.get("color", (112, 219, 192)))
                    if self.show_cameras_var.get() and (i % max(1, math.ceil(len(self.cameras)/300)) == 0 or selected):
                        # Frustum pyramid corners
                        fwd = cam.get("forward", np.array([0, 0, 1], dtype=np.float32))
                        up = cam.get("up", np.array([0, -1, 0], dtype=np.float32))
                        right = cam.get("right", np.array([1, 0, 0], dtype=np.float32))

                        p_apex = c_pos
                        corners = [
                            p_apex + cam_scale * (fwd * 1.5 + right * 0.8 + up * 0.6),
                            p_apex + cam_scale * (fwd * 1.5 - right * 0.8 + up * 0.6),
                            p_apex + cam_scale * (fwd * 1.5 - right * 0.8 - up * 0.6),
                            p_apex + cam_scale * (fwd * 1.5 + right * 0.8 - up * 0.6),
                        ]

                        scr_corners = []
                        for corner in corners:
                            cr = (corner @ R_view.T) + np.array([self.pan_x, self.pan_y, tz])
                            if cr[2] > 0.1:
                                scr_corners.append((int(cr[0] / cr[2] * focal + w * 0.5), int(cr[1] / cr[2] * focal + h * 0.5)))

                        if len(scr_corners) == 4:
                            # Draw thumbnail on frustum face if available
                            if not is_drag:
                                xs = [pt[0] for pt in scr_corners]
                                ys = [pt[1] for pt in scr_corners]
                                min_x, max_x = min(xs), max(xs)
                                min_y, max_y = min(ys), max(ys)
                                bw = max_x - min_x
                                bh = max_y - min_y
                                if 14 <= bw <= 300 and 14 <= bh <= 300 and max_x > 0 and min_x < w and max_y > 0 and min_y < h:
                                    thumb = self._get_camera_thumbnail(cam.get("name", ""), cam.get("file_path"))
                                    if thumb is not None:
                                        try:
                                            resized_thumb = thumb.resize((bw, bh), Image.Resampling.BILINEAR)
                                            pil_img.paste(resized_thumb, (min_x, min_y))
                                        except Exception:
                                            pass

                            # Draw pyramid edges from apex
                            for sc in scr_corners:
                                draw.line([(csx, csy), sc], fill=color, width=2 if selected else 1)
                            # Draw base rectangle
                            draw.line([scr_corners[0], scr_corners[1], scr_corners[2], scr_corners[3], scr_corners[0]], fill=color, width=2 if selected else 1)

            # Keep A and B sequences separate; never connect independent runs.
            if self.show_trajectory_var.get():
                for layer in ('A', 'B'):
                    sequence = [(x,y) for x,y,z,i in cam_screen_pts if self.cameras[i].get('layer','A') == layer]
                    if len(sequence)>1:
                        draw.line(sequence, fill='#70dbc0' if layer=='A' else '#ff9b7f', width=1)
            self._camera_screen = cam_screen_pts if self.show_cameras_var.get() else []
            for x,y,z,i in cam_screen_pts:
                if i == self.selected_camera and self.show_cameras_var.get():
                    draw.ellipse((x-5,y-5,x+5,y+5),outline='#ffdc7d',width=2)

        # Update Tkinter Canvas Image
        self._photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

    def show_reconstruction_report(self):
        """Displays the Reconstruction Report for the active or selected alignment run."""
        results_dir = self._resolve_results_dir()
        target_dir = getattr(self, "active_run_dir", None)
        run_name = getattr(self, "active_run_name", None)

        if not target_dir or not os.path.isdir(target_dir):
            if results_dir and os.path.isdir(results_dir):
                choice = self.run_choice_var.get()
                if choice and choice != "Latest Run (Auto)":
                    cand = os.path.join(results_dir, choice)
                    if os.path.isdir(cand):
                        target_dir = cand
                        run_name = choice
                if not target_dir:
                    found = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("sparse_")]
                    found.sort(key=lambda d: os.path.getmtime(os.path.join(results_dir, d)), reverse=True)
                    if found:
                        target_dir = os.path.join(results_dir, found[0])
                        run_name = found[0]

        if not target_dir or not os.path.isdir(target_dir):
            from tkinter import messagebox
            messagebox.showinfo("Reconstruction Report", "No valid alignment run found for the current dataset.\nPlease select or run an alignment first.")
            return

        run_name = run_name or os.path.basename(target_dir)

        # 1. Search for reconstruction_report.txt
        rep_file = os.path.join(target_dir, "reconstruction_report.txt")
        report_text = ""
        if os.path.isfile(rep_file):
            try:
                with open(rep_file, "r", encoding="utf-8", errors="replace") as f:
                    report_text = f.read()
            except Exception:
                pass

        # 2. Search for any *.log file in run folder if not found
        if not report_text:
            log_files = glob.glob(os.path.join(target_dir, "*.log"))
            if log_files:
                try:
                    with open(log_files[0], "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                    start_idx = -1
                    for idx, line in enumerate(lines):
                        if "RECONSTRUCTION REPORT" in line:
                            start_idx = max(0, idx - 1)
                            break
                    if start_idx != -1:
                        report_text = "".join(lines[start_idx:])
                except Exception:
                    pass

        # 3. Dynamic synthesis if no text report file exists
        if not report_text:
            report_text = self._synthesize_report(target_dir, run_name)

        ReconstructionReportViewer(self, run_name, report_text, target_dir)

    def _synthesize_report(self, run_dir: str, run_name: str) -> str:
        """Synthesizes a reconstruction summary from disk artifacts."""
        pts_count = len(self.points) if (self.points is not None and getattr(self, "active_run_dir", None) == run_dir) else "Unknown"
        cam_count = len(self.cameras) if (self.cameras and getattr(self, "active_run_dir", None) == run_dir) else "Unknown"
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(run_dir)))
        artifacts = os.listdir(run_dir) if os.path.isdir(run_dir) else []
        text = (
            "======================================================================\n"
            f"                       RECONSTRUCTION REPORT\n"
            "======================================================================\n"
            f" Alignment Run:        {run_name}\n"
            f" Output Directory:     {run_dir}\n"
            f" Last Modified:        {mtime}\n"
            f" Registered Cameras:   {cam_count}\n"
            f" 3D Points Generated:  {pts_count if isinstance(pts_count, str) else f'{pts_count:,}'}\n"
            "----------------------------------------------------------------------\n"
            " ARTIFACTS FOUND:\n"
        )
        for a in artifacts:
            ap = os.path.join(run_dir, a)
            if os.path.isdir(ap):
                text += f"   * {a}/  (Directory)\n"
            else:
                sz = os.path.getsize(ap) / 1024.0
                text += f"   * {a}  ({sz:.1f} KB)\n"
        text += (
            "----------------------------------------------------------------------\n"
            " Status: Ready for 3DGS Training (LichtFeld Studio, Splatfacto, 3DGRUT)\n"
            "======================================================================\n"
        )
        return text


class ReconstructionReportViewer(tk.Toplevel):
    """
    Dedicated high-aesthetic dialog to display detailed Reconstruction Reports.
    Provides copying to clipboard, opening results directory, and viewing in system editor.
    """
    def __init__(self, parent, run_name: str, report_text: str, run_dir: str):
        super().__init__(parent)
        self.run_name = run_name
        self.report_text = report_text
        self.run_dir = run_dir

        self.title(f"📋 Reconstruction Report — {run_name}")
        self.geometry("780x640")
        self.minsize(600, 450)
        self.configure(bg=BG)
        self.transient(parent.winfo_toplevel() if hasattr(parent, 'winfo_toplevel') else parent)

        # Set custom window icon
        self._apply_icon()
        self._build_ui()

    def _apply_icon(self):
        try:
            ico = os.path.join(os.path.dirname(__file__), "abu_icon.ico")
            if os.path.isfile(ico):
                self.iconbitmap(ico)
            png = os.path.join(os.path.dirname(__file__), "abu_icon.png")
            if os.path.isfile(png):
                self._photo = ImageTk.PhotoImage(file=png)
                self.iconphoto(True, self._photo)
        except Exception:
            pass

    def _build_ui(self):
        # Header Bar
        hdr = tk.Frame(self, bg=SURFACE, padx=16, pady=12)
        hdr.pack(fill="x", side="top")

        tk.Label(hdr, text=f"📊 Reconstruction Report", bg=SURFACE, fg=ACCENT, font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Label(hdr, text=f"Run: {self.run_name}  ·  Folder: {self.run_dir}", bg=SURFACE, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

        # Text View Area
        txt_frame = tk.Frame(self, bg=BG, padx=12, pady=8)
        txt_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(txt_frame, bg=FIELD, troughcolor=BG)
        scrollbar.pack(side="right", fill="y")

        self.text_widget = tk.Text(
            txt_frame, wrap="word", bg="#0e1215", fg="#d1d9df",
            insertbackground=TEXT, font=("Consolas", 10),
            yscrollcommand=scrollbar.set, relief="flat", bd=0, padx=12, pady=10
        )
        self.text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.text_widget.yview)

        self.text_widget.insert("1.0", self.report_text)
        self.text_widget.configure(state="disabled")

        # Bottom Action Bar
        bar = tk.Frame(self, bg=SURFACE, padx=16, pady=10)
        bar.pack(fill="x", side="bottom")

        copy_btn = tk.Button(
            bar, text="📋 Copy Report", command=self._copy_to_clipboard,
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=12, pady=6, font=("Segoe UI", 9, "bold"), cursor="hand2"
        )
        copy_btn.pack(side="left", padx=(0, 8))

        open_folder_btn = tk.Button(
            bar, text="📂 Open Run Folder", command=self._open_folder,
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=12, pady=6, font=("Segoe UI", 9), cursor="hand2"
        )
        open_folder_btn.pack(side="left", padx=(0, 8))

        notepad_btn = tk.Button(
            bar, text="📝 Open in Notepad", command=self._open_in_notepad,
            bg=FIELD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=12, pady=6, font=("Segoe UI", 9), cursor="hand2"
        )
        notepad_btn.pack(side="left", padx=(0, 8))

        self.status_lbl = tk.Label(bar, text="", bg=SURFACE, fg=ACCENT, font=("Segoe UI", 9))
        self.status_lbl.pack(side="left", padx=10)

        close_btn = tk.Button(
            bar, text="Close", command=self.destroy,
            bg="#2a343b", fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
            relief="flat", bd=0, padx=14, pady=6, font=("Segoe UI", 9), cursor="hand2"
        )
        close_btn.pack(side="right")

    def _copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.report_text)
        self.status_lbl.configure(text="✓ Copied to clipboard!")
        self.after(2500, lambda: self.status_lbl.configure(text=""))

    def _open_folder(self):
        if os.path.isdir(self.run_dir):
            try:
                os.startfile(self.run_dir)
            except Exception:
                pass

    def _open_in_notepad(self):
        rep_file = os.path.join(self.run_dir, "reconstruction_report.txt")
        if not os.path.isfile(rep_file):
            try:
                with open(rep_file, "w", encoding="utf-8") as f:
                    f.write(self.report_text)
            except Exception:
                pass
        try:
            os.system(f'start notepad "{rep_file}"')
        except Exception:
            pass


# Keep the embedded tab API stable while separating its interaction controller.
from native_inspection import NativeInspection

class Workspace3DViewer(NativeInspection, _LegacyWorkspace3DViewer):
    """Native reconstruction workspace with stable navigation and A/B comparison."""
    pass
