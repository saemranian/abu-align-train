"""
Abu Align & Train Version 0.0.1.2
Professional Graphical User Interface for Neural Photogrammetry & 3DGS Training
Built on VGGT, VGGT-Omega, Niantic Labs ACE0, and NVIDIA Isaac PyCuSFM with LichtFeld Studio, Nerfstudio Splatfacto, and NVIDIA 3DGRUT
"""

import os
import sys
import time
import glob
import subprocess
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Ensure parent directory is on sys.path so vggt and dependencies can be imported from any subfolder
current_dir = os.path.dirname(os.path.abspath(__file__))
for parent in [current_dir, os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir))]:
    if parent and parent not in sys.path:
        sys.path.insert(0, parent)

# Enable High-DPI Awareness on Windows so the UI renders crisp and sharp without bilinear blur
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class ToolTip:
    """
    Interactive hover tooltip that displays helpful parameter explanations
    in a sleek, modern dark-themed popup.
    """
    def __init__(self, widget, text, delay=350):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.after_id = None
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        self._schedule()

    def _on_leave(self, event=None):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self.after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self):
        if self.tip_window or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4

            screen_w = self.widget.winfo_screenwidth()
            if x + 350 > screen_w:
                x = max(10, screen_w - 360)

            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tw.attributes("-topmost", True)

            frame = tk.Frame(
                tw,
                bg="#232338",
                highlightbackground="#3b82f6",
                highlightcolor="#3b82f6",
                highlightthickness=1,
                padx=9,
                pady=6
            )
            frame.pack()

            lbl = tk.Label(
                frame,
                text=self.text,
                justify=tk.LEFT,
                bg="#232338",
                fg="#f8fafc",
                font=("Segoe UI", 9),
                wraplength=340
            )
            lbl.pack()
        except Exception:
            pass

    def _hide(self):
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None


class AbuAlignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Abu Align & Train - Version 0.0.1.2 (Neural Photogrammetry & 3DGS Training Suite)")
        self.root.geometry("1060x860")
        self.root.minsize(880, 680)

        # Style configuration (Modern Dark Palette)
        self.bg_dark = "#181825"
        self.panel_bg = "#1e1e2e"
        self.accent_blue = "#3b82f6"
        self.accent_green = "#10b981"
        self.accent_red = "#ef4444"
        self.accent_purple = "#8b5cf6"
        self.accent_cyan = "#0284c7"
        self.text_primary = "#f8fafc"
        self.text_secondary = "#94a3b8"
        self.console_bg = "#11111b"

        self.root.configure(bg=self.bg_dark)

        self.conda_python = r"C:\Users\USer\miniconda3\envs\vggt\python.exe"
        cli_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "align_cli.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "align_cli.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "align_cli.py"),
        ]
        self.cli_script = cli_candidates[0]
        for c in cli_candidates:
            if os.path.isfile(c):
                self.cli_script = c
                break

        train_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_engine.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_engine.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "train_engine.py"),
        ]
        self.train_script = train_candidates[0]
        for c in train_candidates:
            if os.path.isfile(c):
                self.train_script = c
                break

        self.worker_process = None
        self.output_queue = queue.Queue()
        self.is_running = False
        self.latest_output_dir = None
        self.active_log_path = None
        self._ensure_logs_dir()

        self._build_menu()
        self._build_ui()
        self._check_queue()

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=self.panel_bg, fg=self.text_primary)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.panel_bg, fg=self.text_primary)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Dataset Folder...", command=self._browse_dataset)
        file_menu.add_command(label="Open Results Folder", command=self._open_results_folder)
        file_menu.add_command(label="📁 Open 3DGS Models Folder", command=self._open_models_folder)
        file_menu.add_separator()
        file_menu.add_command(label="📁 Open Logs Folder", command=self._open_logs_folder)
        file_menu.add_command(label="📄 View Latest Log File", command=self._open_latest_log)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.panel_bg, fg=self.text_primary)
        menubar.add_cascade(label="Help & Documentation", menu=help_menu)
        help_menu.add_command(label="📖 User Guide & Fundamentals Manual", command=self._open_user_guide)
        help_menu.add_command(label="📐 AI vs Traditional Architecture Whitepaper", command=self._open_architecture_doc)
        help_menu.add_separator()
        help_menu.add_command(label="🌐 Launch 3D Browser Viewer", command=self._launch_viewer)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ About Abu Align & Train", command=self._show_about_box)

    def _build_ui(self):
        # 1. Header Banner
        header = tk.Frame(self.root, bg=self.panel_bg, height=65)
        header.pack(fill=tk.X, padx=12, pady=(12, 6))

        title_lbl = tk.Label(
            header,
            text="ABU ALIGN & TRAIN",
            font=("Segoe UI", 16, "bold"),
            fg=self.accent_blue,
            bg=self.panel_bg
        )
        title_lbl.pack(side=tk.LEFT, padx=16, pady=8)

        version_badge = tk.Label(
            header,
            text="v0.0.1.2",
            font=("Segoe UI", 9, "bold"),
            fg="#ffffff",
            bg=self.accent_purple,
            padx=6,
            pady=2
        )
        version_badge.pack(side=tk.LEFT, pady=12)

        subtitle_lbl = tk.Label(
            header,
            text="|   Neural Photogrammetry & 3D Gaussian Splatting Training Suite",
            font=("Segoe UI", 10),
            fg=self.text_secondary,
            bg=self.panel_bg
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=10, pady=12)

        # Header Help & User Guide Button
        help_btn = tk.Button(
            header,
            text="📖 User Guide & Manual",
            font=("Segoe UI", 9, "bold"),
            bg="#29293d",
            fg="#f8fafc",
            activebackground=self.accent_blue,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=12,
            pady=4,
            command=self._open_user_guide
        )
        help_btn.pack(side=tk.RIGHT, padx=16, pady=12)
        ToolTip(help_btn, "Open Abu Align User Guide & Fundamentals Manual")

        # 2. Main Content Notebook / Grid
        content_frame = tk.Frame(self.root, bg=self.bg_dark)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Top Control Card
        control_card = tk.LabelFrame(
            content_frame,
            text=" Alignment Configuration ",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=14,
            pady=10
        )
        control_card.pack(fill=tk.X, pady=(0, 8))

        # Dataset Path Selection
        path_row = tk.Frame(control_card, bg=self.panel_bg)
        path_row.pack(fill=tk.X, pady=4)

        tk.Label(
            path_row,
            text="Dataset Folder:",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            width=14,
            anchor="w"
        ).pack(side=tk.LEFT)

        self.dataset_var = tk.StringVar(value=r"C:\Users\USer\Documents\Works\Clients\Schemata\Container_03\Sony\101MSDCF\CC\Interior\RedBox\SEL Copy")
        self.path_entry = tk.Entry(
            path_row,
            textvariable=self.dataset_var,
            font=("Segoe UI", 9),
            bg=self.bg_dark,
            fg=self.text_primary,
            insertbackground=self.text_primary,
            relief=tk.FLAT
        )
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=3)

        browse_btn = tk.Button(
            path_row,
            text="Browse...",
            font=("Segoe UI", 9),
            bg=self.accent_blue,
            fg="#ffffff",
            activebackground="#2563eb",
            relief=tk.FLAT,
            padx=12,
            command=self._browse_dataset
        )
        browse_btn.pack(side=tk.RIGHT)

        # Options Grid: 4 Clean Columns
        opts_grid = tk.Frame(control_card, bg=self.panel_bg)
        opts_grid.pack(fill=tk.X, pady=(8, 4))

        # Column 1: Engine & Canvas
        col1 = tk.Frame(opts_grid, bg=self.panel_bg)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col1, text="Engine & Canvas:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.engine_var = tk.StringVar(value="vggt_omega")
        rb_omega = tk.Radiobutton(col1, text="VGGT-Omega (Fast, Low VRAM)", variable=self.engine_var, value="vggt_omega", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_omega.pack(anchor="w")
        rb_vggt = tk.Radiobutton(col1, text="VGGT Standard 1B", variable=self.engine_var, value="vggt", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_vggt.pack(anchor="w")
        rb_ace0 = tk.Radiobutton(col1, text="ACE0 (Niantic Labs SfM)", variable=self.engine_var, value="ace0", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace0.pack(anchor="w")
        rb_cusfm = tk.Radiobutton(col1, text="NVIDIA PyCuSFM (cuSFM GPU)", variable=self.engine_var, value="pycusfm", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cusfm.pack(anchor="w")

        ToolTip(rb_omega, "VGGT-Omega: Next-Gen Visual Geometry Transformer with Register Attention. Blazing fast, low VRAM footprint.")
        ToolTip(rb_vggt, "VGGT-1B: Standard 1-Billion parameter foundation model. Full global attention cross-frame tracking.")
        ToolTip(rb_ace0, "ACE0 (ECCV 2024 Oral): Learning-based Structure-from-Motion by Niantic Labs. Trains an implicit scene coordinate regression network from scratch on unposed images. Ultra-fast, extremely low VRAM (~2GB), and highly robust.")
        ToolTip(rb_cusfm, "NVIDIA PyCuSFM (cuSFM): High-throughput CUDA Structure-from-Motion framework by NVIDIA Isaac. Combines learned ALIKED/SuperPoint keypoints, LightGlue GPU matching, and global bundle adjustment with cuDSS sparse GPU solver.")

        sub_row = tk.Frame(col1, bg=self.panel_bg)
        sub_row.pack(anchor="w", pady=(4, 0))
        tk.Label(sub_row, text="Res:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        self.res_var = tk.StringVar(value="1024")
        res_cb = ttk.Combobox(sub_row, textvariable=self.res_var, values=["512", "1024", "1536", "2048"], state="readonly", width=6)
        res_cb.pack(side=tk.LEFT, padx=(2, 6))

        tk.Label(sub_row, text="Stride:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        self.stride_var = tk.StringVar(value="1")
        stride_cb = ttk.Combobox(sub_row, textvariable=self.stride_var, values=["1", "2", "3", "4", "5"], state="readonly", width=4)
        stride_cb.pack(side=tk.LEFT, padx=2)

        # Column 2: Point Cloud Density (Quality)
        col2 = tk.Frame(opts_grid, bg=self.panel_bg)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col2, text="Point Cloud Density:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.density_var = tk.StringVar(value="high")
        rb_q1 = tk.Radiobutton(col2, text="Preview (100K pts, Conf 2.2)", variable=self.density_var, value="preview", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q1.pack(anchor="w")
        rb_q2 = tk.Radiobutton(col2, text="High Quality (300K pts, Conf 1.8)", variable=self.density_var, value="high", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q2.pack(anchor="w")
        rb_q3 = tk.Radiobutton(col2, text="Ultra Dense (800K pts, Conf 1.5)", variable=self.density_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q3.pack(anchor="w")
        rb_q4 = tk.Radiobutton(col2, text="Max Scan Grade (1.5M pts, Conf 1.2)", variable=self.density_var, value="max", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q4.pack(anchor="w")

        # Column 3: Camera Alignment Accuracy
        col3 = tk.Frame(opts_grid, bg=self.panel_bg)
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col3, text="Camera Alignment Accuracy:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.align_var = tk.StringVar(value="ultra")
        rb_a1 = tk.Radiobutton(col3, text="Feedforward (Fast, ~25s)", variable=self.align_var, value="none", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a1.pack(anchor="w")
        rb_a2 = tk.Radiobutton(col3, text="Standard Ceres BA (~1m)", variable=self.align_var, value="standard", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a2.pack(anchor="w")
        rb_a3 = tk.Radiobutton(col3, text="Ultra Sub-Pixel BA (Cauchy+Polish)", variable=self.align_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a3.pack(anchor="w")

        # Column 4: Lens Distortion Calibration
        col4 = tk.Frame(opts_grid, bg=self.panel_bg)
        col4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(col4, text="Lens Distortion Model:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.lens_var = tk.StringVar(value="brown")
        rb_l1 = tk.Radiobutton(col4, text="Pinhole (Ideal, zero distortion)", variable=self.lens_var, value="pinhole", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_l1.pack(anchor="w")
        rb_l2 = tk.Radiobutton(col4, text="Brown Calibration (k1-k2, t1-t2)", variable=self.lens_var, value="brown", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_l2.pack(anchor="w")

        tk.Label(col4, text="PyCuSFM Features:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w", pady=(6, 0))
        self.cusfm_feat_var = tk.StringVar(value="aliked")
        feat_cb = ttk.Combobox(col4, textvariable=self.cusfm_feat_var, values=["aliked", "superpoint", "sift"], state="readonly", width=12)
        feat_cb.pack(anchor="w", padx=2, pady=2)
        ToolTip(feat_cb, "Feature extractor for PyCuSFM engine: ALIKED (default, fast neural keypoints), SuperPoint (robust), or SIFT (classical CUDA).")

        # Smart Cache & Memory Controls Row
        cache_row = tk.Frame(control_card, bg=self.panel_bg)
        cache_row.pack(fill=tk.X, pady=(6, 0))

        self.cache_var = tk.BooleanVar(value=True)
        cache_chk = tk.Checkbutton(
            cache_row,
            text="⚡ Smart Persistent Cache (.abu_cache) - Instant Re-runs (< 5s)",
            variable=self.cache_var,
            font=("Segoe UI", 9, "bold"),
            fg="#60a5fa",
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        cache_chk.pack(side=tk.LEFT)
        ToolTip(cache_chk, "Smart Persistent Cache (.abu_cache):\n• Reuses neural poses & 3D points when changing Point Cloud Density, Alignment Accuracy, or Lens Distortion.\n• Reuses 2D feature tracks when re-solving Ceres BA.\n• Skips 5-15 minutes of neural inference and tracking on repeated runs.")

        clear_cache_btn = tk.Button(
            cache_row,
            text="🗑 Clear Cache",
            font=("Segoe UI", 8),
            bg="#2a2a3e",
            fg=self.text_primary,
            activebackground=self.accent_red,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=8,
            pady=1,
            command=self._clear_dataset_cache
        )
        clear_cache_btn.pack(side=tk.LEFT, padx=(8, 16))
        ToolTip(clear_cache_btn, "Purge .abu_cache for the currently selected dataset to force a fresh neural inference.")

        tk.Label(cache_row, text="Chunk:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        self.chunk_size_var = tk.StringVar(value="32")
        chunk_cb = ttk.Combobox(cache_row, textvariable=self.chunk_size_var, values=["16", "24", "32", "48", "64"], state="readonly", width=4)
        chunk_cb.pack(side=tk.LEFT, padx=(2, 8))
        ToolTip(chunk_cb, "Sliding-window chunk size for large datasets (> 48 photos). Caps VRAM usage.")

        tk.Label(cache_row, text="Anchor:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        self.anchor_var = tk.StringVar(value="8")
        anchor_cb = ttk.Combobox(cache_row, textvariable=self.anchor_var, values=["4", "6", "8", "12", "16"], state="readonly", width=4)
        anchor_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(anchor_cb, "Anchor overlap frames connecting consecutive sliding windows for seamless Umeyama trajectory alignment.")

        # 3. 3D Gaussian Splatting Training Suite Card (Abu Train)
        train_card = tk.LabelFrame(
            content_frame,
            text=" 3D Gaussian Splatting Training Suite (Abu Train) ",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=14,
            pady=8
        )
        train_card.pack(fill=tk.X, pady=(0, 8))

        train_grid = tk.Frame(train_card, bg=self.panel_bg)
        train_grid.pack(fill=tk.X, pady=(2, 4))

        # Train Col 1: Auto-Train & Engine Selection
        tcol1 = tk.Frame(train_grid, bg=self.panel_bg)
        tcol1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.auto_train_var = tk.BooleanVar(value=True)
        auto_chk = tk.Checkbutton(
            tcol1,
            text="⚡ Auto-Train 3DGS after Alignment finishes",
            variable=self.auto_train_var,
            font=("Segoe UI", 9, "bold"),
            fg=self.accent_green,
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        auto_chk.pack(anchor="w", pady=(0, 4))
        ToolTip(auto_chk, "When checked, Abu Align automatically kicks off 3DGS training immediately after photogrammetry alignment finishes.")

        tk.Label(tcol1, text="Training Engine:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.train_engine_var = tk.StringVar(value="lichtfeld")
        rb_t_lf = tk.Radiobutton(tcol1, text="LichtFeld Studio (Vulkan Real-Time GUI + GUT)", variable=self.train_engine_var, value="lichtfeld", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_t_lf.pack(anchor="w")
        rb_t_ns = tk.Radiobutton(tcol1, text="Nerfstudio Splatfacto (PyTorch SOTA + PLY)", variable=self.train_engine_var, value="splatfacto", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_t_ns.pack(anchor="w")
        rb_t_gr = tk.Radiobutton(tcol1, text="NVIDIA 3DGRUT / 3DGUT (Ray Traced & GUT)", variable=self.train_engine_var, value="3dgrut", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_t_gr.pack(anchor="w")

        ToolTip(rb_t_lf, "LichtFeld Studio: Native high-performance Vulkan/C++ 3DGS trainer with live real-time viewport (120+ FPS) and MCP bridge.")
        ToolTip(rb_t_ns, "Nerfstudio Splatfacto: State-of-the-Art PyTorch 3DGS optimizer with automatic standalone Gaussian Splatting PLY exporter.")
        ToolTip(rb_t_gr, "NVIDIA 3DGRUT / 3DGUT: NVIDIA Research framework for Gaussian ray tracing and unscented transform handling complex non-linear optics.")

        # Train Col 2: Training Iterations
        tcol2 = tk.Frame(train_grid, bg=self.panel_bg)
        tcol2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(tcol2, text="Training Iterations:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.train_iter_var = tk.StringVar(value="30,000 (Production High-Fidelity)")
        iter_cb = ttk.Combobox(
            tcol2,
            textvariable=self.train_iter_var,
            values=[
                "7,000 (Fast Preview)",
                "15,000 (Balanced / Standard)",
                "30,000 (Production High-Fidelity)"
            ],
            state="readonly",
            width=28
        )
        iter_cb.pack(anchor="w", pady=(2, 6))
        ToolTip(iter_cb, "Training duration: 7,000 steps (~1-2 min preview); 15,000 steps (~3-5 min standard); 30,000 steps (~7-10 min full production quality).")

        # Train Col 3: Viewport Mode & Optics
        tcol3 = tk.Frame(train_grid, bg=self.panel_bg)
        tcol3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(tcol3, text="Training Viewport Mode:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.train_mode_var = tk.StringVar(value="interactive")
        rb_m_gui = tk.Radiobutton(tcol3, text="Interactive Viewport (Live 3D GUI)", variable=self.train_mode_var, value="interactive", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_m_gui.pack(anchor="w")
        rb_m_cli = tk.Radiobutton(tcol3, text="Headless Background (Fast CLI)", variable=self.train_mode_var, value="headless", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_m_cli.pack(anchor="w")

        self.gut_var = tk.BooleanVar(value=True)
        gut_chk = tk.Checkbutton(
            tcol3,
            text="Enable GUT Optics (Unscented Transform)",
            variable=self.gut_var,
            font=("Segoe UI", 8),
            fg=self.text_secondary,
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        gut_chk.pack(anchor="w", pady=(4, 0))
        ToolTip(gut_chk, "Gaussian Unscented Transform (GUT): Corrects camera ray curvature and optical distortion for sharper splats.")

        # Train Col 4: Advanced Optics & Sparsity
        tcol4 = tk.Frame(train_grid, bg=self.panel_bg)
        tcol4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(tcol4, text="Advanced Opts & Compression:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        self.bilateral_var = tk.BooleanVar(value=False)
        bilat_chk = tk.Checkbutton(tcol4, text="Bilateral Grid Color Correct", variable=self.bilateral_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        bilat_chk.pack(anchor="w")
        ToolTip(bilat_chk, "Bilateral Grid: Corrects complex non-linear lighting variations, dynamic shadows, and color shifts.")

        self.exposure_var = tk.BooleanVar(value=False)
        expo_chk = tk.Checkbutton(tcol4, text="Auto Exposure / WB Align", variable=self.exposure_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        expo_chk.pack(anchor="w")
        ToolTip(expo_chk, "Per-camera auto-exposure and white-balance calibration across differing camera sensor captures.")

        self.sparsity_var = tk.BooleanVar(value=False)
        sparse_chk = tk.Checkbutton(tcol4, text="ADMM Sparsity Pruning (-60% MB)", variable=self.sparsity_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        sparse_chk.pack(anchor="w")
        ToolTip(sparse_chk, "ADMM Sparsity Pruning: Removes near-zero opacity Gaussians during training, reducing PLY file size by up to 60%.")

        self.mip_var = tk.BooleanVar(value=False)
        mip_chk = tk.Checkbutton(tcol4, text="Anti-Aliased Mip-Filtering", variable=self.mip_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        mip_chk.pack(anchor="w")
        ToolTip(mip_chk, "Mip-Splats: Anti-aliasing filter that prevents aliasing and high-frequency noise in fine textures.")

        # 4. Action Buttons
        action_bar = tk.Frame(content_frame, bg=self.bg_dark)
        action_bar.pack(fill=tk.X, pady=(0, 6))

        self.one_click_btn = tk.Button(
            action_bar,
            text="⚡  ONE-CLICK ALIGN & TRAIN",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_purple,
            fg="#ffffff",
            activebackground="#7c3aed",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            command=self._start_one_click
        )
        self.one_click_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.start_btn = tk.Button(
            action_bar,
            text="▶  ALIGNMENT ONLY",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_green,
            fg="#ffffff",
            activebackground="#059669",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            command=self._start_alignment
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.train_only_btn = tk.Button(
            action_bar,
            text="🎯  TRAIN 3DGS ONLY",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_cyan,
            fg="#ffffff",
            activebackground="#0369a1",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            command=self._start_train_only
        )
        self.train_only_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.stop_btn = tk.Button(
            action_bar,
            text="■  STOP",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_red,
            fg="#ffffff",
            activebackground="#dc2626",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            state=tk.DISABLED,
            command=self._stop_alignment
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.open_folder_btn = tk.Button(
            action_bar,
            text="📁 Results",
            font=("Segoe UI", 9),
            bg=self.panel_bg,
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=9,
            pady=6,
            command=self._open_results_folder
        )
        self.open_folder_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.models_btn = tk.Button(
            action_bar,
            text="📁 3DGS Models",
            font=("Segoe UI", 9),
            bg=self.panel_bg,
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=9,
            pady=6,
            command=self._open_models_folder
        )
        self.models_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.logs_btn = tk.Button(
            action_bar,
            text="📁 Logs",
            font=("Segoe UI", 9),
            bg=self.panel_bg,
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=9,
            pady=6,
            command=self._open_logs_folder
        )
        self.logs_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.viewer_btn = tk.Button(
            action_bar,
            text="🌐 3D Viewer",
            font=("Segoe UI", 9),
            bg=self.panel_bg,
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=9,
            pady=6,
            command=self._launch_viewer
        )
        self.viewer_btn.pack(side=tk.LEFT)

        # Attach interactive parameter tooltips
        ToolTip(self.path_entry, "Dataset Folder: Directory containing JPG/PNG photos to align.")
        ToolTip(browse_btn, "Browse: Open folder selector to pick your photoscanning dataset.")

        ToolTip(rb_omega, "VGGT-Omega Engine (2025/2026): Compressed register token architecture. Reduces VRAM usage by ~60% while maintaining wide-baseline alignment.")
        ToolTip(rb_vggt, "VGGT Standard 1B Engine: Original 1.18B-parameter vision foundation model. Predicts 3D point maps and camera poses in a single forward pass.")
        ToolTip(res_cb, "Canvas Resolution: 512 is fastest for preview/low VRAM; 1024/1536/2048 offers finer spatial detail for feature tracking on 40MP-60MP cameras.")
        ToolTip(stride_cb, "Keyframe Stride: Frame sampling interval. 1 processes every photo; 2 or 3 skips intermediate frames (ideal for dense video or drone flights).")

        ToolTip(rb_q1, "Preview (100K pts, Conf >= 2.2): High neural confidence filter. Fastest point cloud for quick visual checks.")
        ToolTip(rb_q2, "High Quality (300K pts, Conf >= 1.8): Optimal balance of density and surface cleanliness. Recommended seed for 3D Gaussian Splatting (Postshot).")
        ToolTip(rb_q3, "Ultra Dense (800K pts, Conf >= 1.5): Deep surface coverage, capturing intricate recesses, background geometry, and secondary objects.")
        ToolTip(rb_q4, "Max Scan Grade (1.5M pts, Conf >= 1.2): Maximum geometric fidelity. Retains fine edges, wires, corners, and foliage.")

        ToolTip(rb_a1, "Feedforward Pose (~25s): 100% Pure Deep Learning. Predicts camera positions and rotations in a single forward pass without bundle adjustment (~2-4px reprojection error).")
        ToolTip(rb_a2, "Standard Ceres BA (~1m): Hybrid pipeline. ALIKED neural feature tracking + Google Ceres Levenberg-Marquardt bundle adjustment.")
        ToolTip(rb_a3, "Ultra Sub-Pixel BA (Cauchy + Polish) [Recommended]: Solves 3,072 multi-view tracks with Cauchy robust loss to downweight outliers, followed by pruning and a 2nd polish pass to reach 0.35 px error.")

        ToolTip(rb_l1, "Pinhole Model: Ideal rectilinear geometry with zero distortion (k1=0, k2=0, t1=0, t2=0). Native to VGGT training prior. Recommended for pre-undistorted imagery.")
        ToolTip(rb_l2, "Brown Calibration (k1-k2, t1-t2) [Recommended]: Self-calibrates physical radial (k1, k2) and tangential (t1, t2) lens curvature in Google Ceres. Essential for wide-angle lenses.")

        ToolTip(self.one_click_btn, "One-Click Pipeline: Runs complete photogrammetry alignment and immediately launches 3DGS training automatically.")
        ToolTip(self.start_btn, "Start Alignment Only: Runs photogrammetry and exports COLMAP, RealityCapture CSV, transforms.json, and 3dgs_dataset/.")
        ToolTip(self.train_only_btn, "Train 3DGS Only: Train a 3D Gaussian Splatting model directly from an existing alignment (3dgs_dataset/ or results/).")
        ToolTip(self.stop_btn, "Stop: Safely terminate the running alignment or training process and free GPU memory.")
        ToolTip(self.open_folder_btn, "Open Results: Open output directory containing sparse reconstruction files, camera CSV, and dense PLY point cloud.")
        ToolTip(self.models_btn, "Open 3DGS Models: Open output directory containing trained Gaussian Splatting .ply / .spz models.")
        ToolTip(self.logs_btn, "Open Logs: Open the logs folder in Windows Explorer to inspect detailed diagnostic files.")
        ToolTip(self.viewer_btn, "3D Browser Viewer: Launch interactive web browser viewer (http://localhost:8080) to inspect cameras and 3D point cloud in real time.")

        # 5. Live Progress Console Card
        console_card = tk.LabelFrame(
            content_frame,
            text=" Live Telemetry & Process Logs ",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=10,
            pady=6
        )
        console_card.pack(fill=tk.BOTH, expand=True)

        # Progress Bar & Real-Time Telemetry Bar
        telemetry_frame = tk.Frame(console_card, bg=self.panel_bg)
        telemetry_frame.pack(fill=tk.X, pady=(2, 4))

        self.telemetry_lbl = tk.Label(
            telemetry_frame,
            text="Status: Ready | Pipeline Idle",
            font=("Segoe UI", 9, "bold"),
            fg=self.accent_blue,
            bg=self.panel_bg
        )
        self.telemetry_lbl.pack(side=tk.LEFT)

        self.eta_lbl = tk.Label(
            telemetry_frame,
            text="",
            font=("Segoe UI", 9),
            fg=self.accent_green,
            bg=self.panel_bg
        )
        self.eta_lbl.pack(side=tk.RIGHT)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Abu.Horizontal.TProgressbar",
            troughcolor="#11111b",
            background="#3b82f6",
            darkcolor="#2563eb",
            lightcolor="#60a5fa",
            bordercolor="#1e1e2e",
            thickness=14
        )

        self.progress_bar = ttk.Progressbar(
            console_card,
            orient="horizontal",
            mode="determinate",
            style="Abu.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))

        console_toolbar = tk.Frame(console_card, bg=self.panel_bg)
        console_toolbar.pack(fill=tk.X, pady=(0, 4))

        self.log_status_lbl = tk.Label(
            console_toolbar,
            text="Log File: Ready (All runs saved to Abu_Align\\logs\\)",
            font=("Segoe UI", 8),
            fg=self.text_secondary,
            bg=self.panel_bg
        )
        self.log_status_lbl.pack(side=tk.LEFT)

        view_log_btn = tk.Button(
            console_toolbar,
            text="📄 View Latest Log",
            font=("Segoe UI", 8),
            bg="#29293d",
            fg=self.text_primary,
            activebackground="#33334d",
            relief=tk.FLAT,
            padx=8,
            pady=2,
            command=self._open_latest_log
        )
        view_log_btn.pack(side=tk.RIGHT)
        ToolTip(view_log_btn, "Open the latest execution log in default text editor.")

        self.console = ScrolledText(
            console_card,
            bg=self.console_bg,
            fg="#a6accd",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        self._log("Abu Align & Train Version 0.0.1.2 Ready.\nQuad Alignment Engines: VGGT Foundation, VGGT-Omega, ACE0, & NVIDIA PyCuSFM.\nTriple 3DGS Training Engines: LichtFeld Studio, Nerfstudio Splatfacto, & NVIDIA 3DGRUT.\nOne-Click Align & Train: Fully Automated Handoff Active.\nSelect a photoscan directory and click 'ONE-CLICK ALIGN & TRAIN' or 'ALIGNMENT ONLY'.\n\n")

    def _get_logs_dir(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if "versions" in base_dir:
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "logs")
        else:
            logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def _ensure_logs_dir(self):
        return self._get_logs_dir()

    def _open_logs_folder(self):
        logs_dir = self._get_logs_dir()
        if os.path.isdir(logs_dir):
            os.startfile(logs_dir)
        else:
            messagebox.showinfo("Logs", f"Logs folder does not exist yet:\n{logs_dir}")

    def _open_latest_log(self):
        logs_dir = self._get_logs_dir()
        latest_file = os.path.join(logs_dir, "abu_align_latest.log")
        if os.path.isfile(latest_file) and os.path.getsize(latest_file) > 0:
            os.startfile(latest_file)
            return

        log_files = sorted(
            [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")],
            key=os.path.getmtime,
            reverse=True
        )
        if log_files:
            os.startfile(log_files[0])
        else:
            messagebox.showinfo("Logs", "No log files found yet. Run an alignment first!")

    def _browse_dataset(self):
        folder = filedialog.askdirectory(initialdir=self.dataset_var.get())
        if folder:
            self.dataset_var.set(folder)

    def _log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def _parse_train_iterations(self):
        iter_str = self.train_iter_var.get()
        if "7,000" in iter_str:
            return 7000
        elif "15,000" in iter_str:
            return 15000
        return 30000

    def _clear_dataset_cache(self):
        dataset_path = self.dataset_var.get().strip()
        if not os.path.isdir(dataset_path):
            messagebox.showinfo("Smart Cache", "Please select a valid dataset folder first.")
            return

        from cache_manager import get_cache_root, clear_cache
        cache_root = get_cache_root(dataset_path)
        if not os.path.isdir(cache_root):
            messagebox.showinfo("Smart Cache", f"No cache folder found for this dataset at:\n{cache_root}")
            return

        confirm = messagebox.askyesno(
            "Clear Cache",
            f"Are you sure you want to delete all cached neural predictions and feature tracks for this dataset?\n\nCache Location: {cache_root}\n\nFuture runs will execute full neural inference from scratch."
        )
        if confirm:
            n_cleared, b_freed = clear_cache(cache_root)
            mb = b_freed / (1024 * 1024)
            messagebox.showinfo("Smart Cache", f"Purged cache successfully!\nFreed: {mb:.1f} MB across {n_cleared} directories.")
            self._log(f"\n[SMART CACHE] Purged cache directory {cache_root} ({mb:.1f} MB freed).\n")

    def _start_alignment(self, force_auto_train=False):
        if self.is_running:
            return

        dataset_path = self.dataset_var.get().strip()
        if not os.path.isdir(dataset_path):
            messagebox.showerror("Error", f"Dataset folder not found:\n{dataset_path}")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.one_click_btn.config(state=tk.DISABLED)
        self.train_only_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)

        self.progress_bar["value"] = 0
        self.telemetry_lbl.config(text="Status: Initializing Pipeline...", fg=self.accent_blue)
        self.eta_lbl.config(text="")

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = self._get_logs_dir()
        self.active_log_path = os.path.join(logs_dir, f"abu_align_{timestamp}.log")
        self.log_status_lbl.config(
            text=f"Log: abu_align_{timestamp}.log",
            fg=self.accent_green
        )

        cmd = [
            self.conda_python,
            "-u",
            self.cli_script,
            "--input", dataset_path,
            "--engine", self.engine_var.get(),
            "--stride", str(self.stride_var.get()),
            "--load_res", str(self.res_var.get()),
            "--align_mode", self.align_var.get(),
            "--lens_model", self.lens_var.get(),
        ]

        if not self.cache_var.get():
            cmd.append("--no_cache")

        cmd.extend([
            "--chunk_size", str(self.chunk_size_var.get()),
            "--anchor_size", str(self.anchor_var.get())
        ])

        if self.engine_var.get() == "pycusfm":
            cmd.extend(["--cusfm_features", self.cusfm_feat_var.get()])

        density = self.density_var.get()
        if density == "preview":
            cmd.extend(["--conf_thresh", "2.2", "--max_points", "100000"])
        elif density == "high":
            cmd.extend(["--conf_thresh", "1.8", "--max_points", "300000"])
        elif density == "ultra":
            cmd.extend(["--conf_thresh", "1.5", "--max_points", "800000"])
        elif density == "max":
            cmd.extend(["--conf_thresh", "1.2", "--max_points", "1500000"])

        # 3DGS Training Suite settings (logged in report, executed if auto-train enabled)
        cmd.extend(["--train_engine", self.train_engine_var.get()])
        cmd.extend(["--train_iterations", str(self._parse_train_iterations())])
        cmd.extend(["--train_mode", self.train_mode_var.get()])
        if self.gut_var.get():
            cmd.append("--gut")
        if self.bilateral_var.get():
            cmd.append("--bilateral_grid")
        if self.exposure_var.get():
            cmd.append("--exposure_correction")
        if self.sparsity_var.get():
            cmd.append("--enable_sparsity")
        if self.mip_var.get():
            cmd.append("--enable_mip")

        if force_auto_train or self.auto_train_var.get():
            cmd.append("--train")

        self._log(f"Executing: {' '.join(cmd)}\n")
        self._log(f"Session Log: {self.active_log_path}\n\n")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _start_one_click(self):
        """Runs alignment and automatically chains into 3DGS training immediately upon completion."""
        self._start_alignment(force_auto_train=True)

    def _start_train_only(self):
        """Directly trains 3DGS from an existing dataset's 3dgs_dataset/ or results/ folder."""
        if self.is_running:
            return

        dataset_path = self.dataset_var.get().strip()
        if not os.path.isdir(dataset_path):
            messagebox.showerror("Error", f"Dataset folder not found:\n{dataset_path}")
            return

        if not self.train_script or not os.path.isfile(self.train_script):
            messagebox.showerror("Error", f"train_engine.py not found at:\n{self.train_script}")
            return

        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.one_click_btn.config(state=tk.DISABLED)
        self.train_only_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)

        self.progress_bar["value"] = 0
        self.telemetry_lbl.config(text="Status: Initializing 3DGS Training...", fg=self.accent_blue)
        self.eta_lbl.config(text="")

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = self._get_logs_dir()
        self.active_log_path = os.path.join(logs_dir, f"abu_train_{timestamp}.log")
        self.log_status_lbl.config(
            text=f"Log: abu_train_{timestamp}.log",
            fg=self.accent_green
        )

        cmd = [
            self.conda_python,
            "-u",
            self.train_script,
            "--dataset", dataset_path,
            "--engine", self.train_engine_var.get(),
            "--iterations", str(self._parse_train_iterations()),
            "--mode", self.train_mode_var.get(),
        ]
        if self.gut_var.get():
            cmd.append("--gut")
        if self.bilateral_var.get():
            cmd.append("--bilateral_grid")
        if self.exposure_var.get():
            cmd.append("--exposure_correction")
        if self.sparsity_var.get():
            cmd.append("--enable_sparsity")
        if self.mip_var.get():
            cmd.append("--enable_mip")

        self._log(f"Executing Standalone 3DGS Training: {' '.join(cmd)}\n")
        self._log(f"Session Log: {self.active_log_path}\n\n")

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _run_process(self, cmd):
        log_path = self.active_log_path
        logs_dir = self._get_logs_dir()
        latest_path = os.path.join(logs_dir, "abu_align_latest.log")

        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write("=" * 72 + "\n")
                lf.write("Abu Align & Train v0.0.1.2 - Run Telemetry & Diagnostic Log\n")
                lf.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                lf.write(f"Command:   {' '.join(cmd)}\n")
                lf.write("=" * 72 + "\n\n")
                lf.flush()

                self.worker_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                )

                for line in iter(self.worker_process.stdout.readline, ""):
                    if line:
                        lf.write(line)
                        lf.flush()
                        self.output_queue.put(line)
                        if "Output Directory:" in line:
                            parts = line.split("Output Directory:")
                            if len(parts) > 1:
                                self.latest_output_dir = parts[1].strip()

                self.worker_process.stdout.close()
                exit_code = self.worker_process.wait()

                lf.write("\n" + "-" * 72 + "\n")
                lf.write(f"Process Completed. Exit Code: {exit_code} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                lf.write("-" * 72 + "\n")
                lf.flush()

            try:
                import shutil
                shutil.copyfile(log_path, latest_path)
            except Exception:
                pass

        except Exception as e:
            err_msg = f"\n[ERROR] Worker process execution failed: {str(e)}\n"
            self.output_queue.put(err_msg)
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(err_msg)
            except Exception:
                pass
        finally:
            self.is_running = False
            self.output_queue.put("__PROCESS_FINISHED__")

    def _check_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg == "__PROCESS_FINISHED__":
                    self.start_btn.config(state=tk.NORMAL)
                    self.one_click_btn.config(state=tk.NORMAL)
                    self.train_only_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.progress_bar["value"] = 100
                    self.telemetry_lbl.config(text="Status: Pipeline Completed Successfully (100%)", fg=self.accent_green)
                    self.eta_lbl.config(text="Done")
                    self._log("\n[STATUS] Pipeline Job Completed.\n")
                    self.log_status_lbl.config(
                        text="Log: Saved (Abu_Align\\logs\\)",
                        fg=self.text_secondary
                    )
                else:
                    self._log(msg)
                    msg_s = msg.strip()
                    if msg_s.startswith("[TELEMETRY:STAGE3]"):
                        try:
                            parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                            pct = float(parts.get("pct", 0))
                            self.progress_bar["value"] = pct
                            if parts.get("status") == "cache_hit":
                                self.telemetry_lbl.config(text="⚡ [Smart Cache Hit] Neural Poses & 3D Points Loaded (Stage 3 & 4 Skipped)", fg="#60a5fa")
                                self.eta_lbl.config(text="Saved ~5-10 min")
                            else:
                                b = parts.get("batch", "?")
                                tot = parts.get("total", "?")
                                self.telemetry_lbl.config(text=f"Stage 3/5: Neural Batch Inference [{b}/{tot}] ({pct:.1f}%)", fg=self.accent_blue)
                        except Exception:
                            pass
                    elif msg_s.startswith("[TELEMETRY:STAGE5]"):
                        try:
                            parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                            pct = float(parts.get("pct", 0))
                            self.progress_bar["value"] = pct
                            if parts.get("status") == "cache_hit":
                                self.telemetry_lbl.config(text="⚡ [Smart Cache Hit] Multi-View Tracks Loaded (Stage 5 Skipped)", fg="#60a5fa")
                                self.eta_lbl.config(text="Saved ~10-20 min")
                            else:
                                q = parts.get("query", "?")
                                tot = parts.get("total", "?")
                                f = parts.get("frame", "?")
                                eta_s = float(parts.get("eta_sec", 0))
                                eta_str = f"{int(eta_s//60)}m {int(eta_s%60):02d}s" if eta_s > 0 else "calculating..."
                                self.telemetry_lbl.config(text=f"Stage 5/5: Multi-View Tracking [Query {q}/{tot}, Frame #{f}] ({pct:.1f}%)", fg="#a855f7")
                                self.eta_lbl.config(text=f"⏱ ETA: {eta_str}")
                        except Exception:
                            pass
                    elif "[LichtFeld]" in msg and any(k in msg.lower() for k in ["step", "iter", "export", "loss"]):
                        self.telemetry_lbl.config(text="Stage 6: LichtFeld 3DGS Training Active", fg=self.accent_cyan)
        except queue.Empty:
            pass
        self.root.after(80, self._check_queue)

    def _stop_alignment(self):
        if self.worker_process and self.is_running:
            try:
                self.worker_process.terminate()
                self._log("\n[ABORT] User cancelled process.\n")
            except Exception:
                pass
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.one_click_btn.config(state=tk.NORMAL)
            self.train_only_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _open_results_folder(self):
        target = self.latest_output_dir
        if not target or not os.path.isdir(target):
            dataset = self.dataset_var.get().strip()
            target = os.path.join(dataset, "results")
        if os.path.isdir(target):
            os.startfile(target)
        else:
            messagebox.showinfo("Info", f"Results folder does not exist yet:\n{target}")

    def _open_models_folder(self):
        target = None
        if self.latest_output_dir:
            cand = os.path.join(self.latest_output_dir, "3dgs_models")
            if os.path.isdir(cand):
                target = cand
        if not target or not os.path.isdir(target):
            dataset = self.dataset_var.get().strip()
            res_dir = os.path.join(dataset, "results")
            if os.path.isdir(res_dir):
                models_dirs = sorted(glob.glob(os.path.join(res_dir, "**", "3dgs_models"), recursive=True), key=os.path.getmtime, reverse=True)
                if models_dirs:
                    target = models_dirs[0]
            if not target:
                cand2 = os.path.join(dataset, "3dgs_models")
                if os.path.isdir(cand2):
                    target = cand2
        if target and os.path.isdir(target):
            os.startfile(target)
        else:
            messagebox.showinfo("Info", "3DGS Models folder does not exist yet. Run training first!")

    def _launch_viewer(self):
        dataset = self.dataset_var.get().strip()
        if not os.path.isdir(dataset):
            messagebox.showerror("Error", "Please select a valid dataset folder first.")
            return

        # Locate fast_viewer.py
        base_dirs = [
            os.path.dirname(self.cli_script),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ]
        fast_viewer_script = None
        for b in base_dirs:
            candidate = os.path.join(b, "fast_viewer.py")
            if os.path.isfile(candidate):
                fast_viewer_script = candidate
                break

        if not fast_viewer_script:
            messagebox.showerror("Error", "fast_viewer.py not found.")
            return

        target_arg = self.latest_output_dir if self.latest_output_dir and os.path.isdir(self.latest_output_dir) else dataset
        cmd = [self.conda_python, fast_viewer_script, "--input", target_arg]
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)

    def _open_doc_window(self, title, doc_path):
        if not os.path.isfile(doc_path):
            messagebox.showerror("Error", f"Document file not found:\n{doc_path}")
            return

        doc_win = tk.Toplevel(self.root)
        doc_win.title(f"Abu Align v0.0.0.7 - {title}")
        doc_win.geometry("960x780")
        doc_win.minsize(800, 550)
        doc_win.configure(bg=self.bg_dark)

        # Header toolbar
        tb = tk.Frame(doc_win, bg=self.panel_bg, height=48, padx=12, pady=8)
        tb.pack(fill=tk.X)

        doc_title_lbl = tk.Label(
            tb,
            text=f"📖  {title}",
            font=("Segoe UI", 11, "bold"),
            fg=self.accent_blue,
            bg=self.panel_bg
        )
        doc_title_lbl.pack(side=tk.LEFT, padx=(4, 16))

        # Search bar
        search_lbl = tk.Label(tb, text="🔍 Find:", font=("Segoe UI", 9), fg=self.text_secondary, bg=self.panel_bg)
        search_lbl.pack(side=tk.LEFT, padx=(4, 2))

        search_var = tk.StringVar()
        search_entry = tk.Entry(
            tb,
            textvariable=search_var,
            font=("Segoe UI", 9),
            bg="#2a2a3e",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            width=22
        )
        search_entry.pack(side=tk.LEFT, padx=4, ipady=3)

        match_lbl = tk.Label(tb, text="", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg)
        match_lbl.pack(side=tk.LEFT, padx=4)

        find_state = {"matches": [], "current_idx": -1}

        def do_search(forward=True):
            query = search_var.get().strip()
            text_widget.tag_remove("search_hl", "1.0", tk.END)
            text_widget.tag_remove("search_current", "1.0", tk.END)
            find_state["matches"] = []

            if not query:
                match_lbl.config(text="")
                return

            start_idx = "1.0"
            while True:
                pos = text_widget.search(query, start_idx, stopindex=tk.END, nocase=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(query)}c"
                text_widget.tag_add("search_hl", pos, end_pos)
                find_state["matches"].append((pos, end_pos))
                start_idx = end_pos

            total = len(find_state["matches"])
            if total == 0:
                match_lbl.config(text="No matches", fg=self.accent_red)
                return

            if forward:
                find_state["current_idx"] = (find_state["current_idx"] + 1) % total
            else:
                find_state["current_idx"] = (find_state["current_idx"] - 1) % total

            cur = find_state["current_idx"]
            match_lbl.config(text=f"{cur + 1}/{total}", fg=self.accent_green)

            cur_start, cur_end = find_state["matches"][cur]
            text_widget.tag_add("search_current", cur_start, cur_end)
            text_widget.see(cur_start)

        find_next_btn = tk.Button(
            tb,
            text="▼ Next",
            font=("Segoe UI", 8),
            bg="#2a2a3e",
            fg="#f8fafc",
            relief=tk.FLAT,
            padx=6,
            pady=1,
            command=lambda: do_search(True)
        )
        find_next_btn.pack(side=tk.LEFT, padx=2)

        find_prev_btn = tk.Button(
            tb,
            text="▲ Prev",
            font=("Segoe UI", 8),
            bg="#2a2a3e",
            fg="#f8fafc",
            relief=tk.FLAT,
            padx=6,
            pady=1,
            command=lambda: do_search(False)
        )
        find_prev_btn.pack(side=tk.LEFT, padx=2)

        search_entry.bind("<Return>", lambda e: do_search(True))
        search_entry.bind("<KeyRelease>", lambda e: do_search(True) if search_var.get() else match_lbl.config(text=""))
        doc_win.bind("<Control-f>", lambda e: search_entry.focus_set())
        doc_win.bind("<Escape>", lambda e: doc_win.destroy())

        close_btn = tk.Button(
            tb,
            text="✕ Close",
            font=("Segoe UI", 8, "bold"),
            bg="#374151",
            fg="#f8fafc",
            activebackground=self.accent_red,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            command=doc_win.destroy
        )
        close_btn.pack(side=tk.RIGHT, padx=4)

        # ScrolledText display
        text_frame = tk.Frame(doc_win, bg=self.bg_dark, padx=12, pady=8)
        text_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = ScrolledText(
            text_frame,
            wrap=tk.WORD,
            bg="#11111b",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            selectbackground="#3b82f6",
            selectforeground="#ffffff",
            font=("Segoe UI", 10),
            padx=18,
            pady=14,
            relief=tk.FLAT
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Configure styling tags
        text_widget.tag_configure("h1", font=("Segoe UI", 16, "bold"), foreground="#60a5fa", spacing1=12, spacing3=6)
        text_widget.tag_configure("h2", font=("Segoe UI", 13, "bold"), foreground="#38bdf8", spacing1=10, spacing3=4)
        text_widget.tag_configure("h3", font=("Segoe UI", 11, "bold"), foreground="#a5b4fc", spacing1=6, spacing3=2)
        text_widget.tag_configure("code_block", font=("Consolas", 9), background="#1e1e2e", foreground="#fdba74", spacing1=2, spacing3=2)
        text_widget.tag_configure("bullet", font=("Segoe UI", 10), foreground="#93c5fd")
        text_widget.tag_configure("alert_note", font=("Segoe UI", 9, "bold"), foreground="#38bdf8")
        text_widget.tag_configure("alert_warn", font=("Segoe UI", 9, "bold"), foreground="#f59e0b")
        text_widget.tag_configure("hr", font=("Segoe UI", 4), foreground="#374151")
        text_widget.tag_configure("search_hl", background="#854d0e", foreground="#ffffff")
        text_widget.tag_configure("search_current", background="#dc2626", foreground="#ffffff")

        # Load file and parse Markdown
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            in_code_block = False
            for line in lines:
                sline = line.strip()
                if sline.startswith("```"):
                    in_code_block = not in_code_block
                    text_widget.insert(tk.END, line, "code_block")
                elif in_code_block:
                    text_widget.insert(tk.END, line, "code_block")
                elif sline.startswith("# "):
                    text_widget.insert(tk.END, line, "h1")
                elif sline.startswith("## "):
                    text_widget.insert(tk.END, line, "h2")
                elif sline.startswith("### "):
                    text_widget.insert(tk.END, line, "h3")
                elif sline.startswith("- ") or sline.startswith("* "):
                    text_widget.insert(tk.END, "  • ", "bullet")
                    text_widget.insert(tk.END, line[2:])
                elif sline.startswith("> [!NOTE]") or sline.startswith("> [!TIP]"):
                    text_widget.insert(tk.END, "ℹ️ " + line[2:], "alert_note")
                elif sline.startswith("> [!WARNING]") or sline.startswith("> [!CAUTION]"):
                    text_widget.insert(tk.END, "⚠️ " + line[2:], "alert_warn")
                elif sline.startswith("---") or sline.startswith("==="):
                    text_widget.insert(tk.END, "─" * 70 + "\n", "hr")
                else:
                    text_widget.insert(tk.END, line)

            text_widget.config(state=tk.DISABLED)
        except Exception as e:
            text_widget.insert(tk.END, f"Error reading document: {e}")

    def _open_user_guide(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_dir, "docs", "ABU_ALIGN_USER_GUIDE.md"),
            os.path.join(os.path.dirname(base_dir), "docs", "ABU_ALIGN_USER_GUIDE.md"),
            os.path.join(base_dir, "ABU_ALIGN_USER_GUIDE.md"),
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                self._open_doc_window("User Guide & Fundamentals Manual", p)
                return
        messagebox.showinfo("User Guide", "User Guide document (ABU_ALIGN_USER_GUIDE.md) not found.")

    def _open_architecture_doc(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_dir, "docs", "AI_VS_TRADITIONAL_PIPELINE_ARCHITECTURE.md"),
            os.path.join(os.path.dirname(base_dir), "docs", "AI_VS_TRADITIONAL_PIPELINE_ARCHITECTURE.md"),
            os.path.join(base_dir, "AI_VS_TRADITIONAL_PIPELINE_ARCHITECTURE.md"),
        ]
        for p in possible_paths:
            if os.path.isfile(p):
                self._open_doc_window("AI vs Traditional Architecture Whitepaper", p)
                return
        messagebox.showinfo("Architecture Whitepaper", "Architecture document (AI_VS_TRADITIONAL_PIPELINE_ARCHITECTURE.md) not found.")

    def _show_about_box(self):
        about_text = (
            "Abu Align & Train - Version 0.0.1.2\n"
            "Unified Neural Photogrammetry & 3D Gaussian Splatting Training Suite\n\n"
            "• Photogrammetry Engines: VGGT Foundation, VGGT-Omega, Niantic Labs ACE0, & NVIDIA Isaac PyCuSFM\n"
            "• 3DGS Training Engines: LichtFeld Studio (Vulkan Real-Time GUI + GUT), Nerfstudio Splatfacto, & NVIDIA 3DGRUT\n"
            "• Automated Workflows: One-Click Align & Train, Auto-Handoff, & Standalone 3DGS Training\n"
            "• Standardized 3DGS Factory: Universal 3dgs_dataset/ (sparse/0/, images/, transforms.json, points.ply)\n"
            "• Distortion & Optics: Gaussian Unscented Transform (GUT) & Self-Calibrated Brown-Conrady\n"
            "• Telemetry: Real-time progress bars, loss curves, PSNR, and persistent logs in Abu_Align\\logs\\\n"
            "• Scalability: Smart Auto-Chunking & VRAM Guard for large multi-sensor datasets\n\n"
            "Target Integrations: LichtFeld Studio, Postshot, Nerfstudio, RealityScan, COLMAP\n"
            "Developer: saemranian"
        )
        messagebox.showinfo("About Abu Align & Train", about_text)

def main():
    root = tk.Tk()
    app = AbuAlignApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
