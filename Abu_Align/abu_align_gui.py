"""
Abu Align & Train Version 0.0.1.7
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
try:
    import psutil
except ImportError:
    psutil = None
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


class SystemHardwareMonitor:
    """
    Direct low-overhead hardware telemetry reader for CPU, RAM, and GPU.
    Uses psutil for CPU/RAM and direct ctypes calls to Windows nvml.dll for GPU telemetry
    (GPU core util %, dedicated VRAM, shared memory paging detection, and GPU temperature).
    """
    def __init__(self):
        self.nvml_available = False
        self.nvml_handle = None
        self.nvml_lib = None
        self._init_nvml()

    def _init_nvml(self):
        if sys.platform != "win32":
            return
        try:
            import ctypes
            nvml = ctypes.CDLL("nvml.dll")
            if nvml.nvmlInit_v2() == 0:
                dev = ctypes.c_void_p()
                if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) == 0:
                    self.nvml_lib = nvml
                    self.nvml_handle = dev
                    self.nvml_available = True
        except Exception:
            self.nvml_available = False

    def query(self):
        telemetry = {
            "cpu_pct": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_pct": 0.0,
            "gpu_name": "NVIDIA GPU",
            "gpu_util": 0,
            "gpu_temp": 0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 24.0,
            "vram_pct": 0.0,
            "paging_risk": False,
        }
        if psutil is not None:
            try:
                telemetry["cpu_pct"] = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                telemetry["ram_used_gb"] = vm.used / (1024**3)
                telemetry["ram_total_gb"] = vm.total / (1024**3)
                telemetry["ram_pct"] = vm.percent
            except Exception:
                pass

        if self.nvml_available and self.nvml_lib and self.nvml_handle:
            try:
                import ctypes
                temp = ctypes.c_uint()
                if self.nvml_lib.nvmlDeviceGetTemperature(self.nvml_handle, 0, ctypes.byref(temp)) == 0:
                    telemetry["gpu_temp"] = int(temp.value)

                class nvmlUtilization_t(ctypes.Structure):
                    _fields_ = [('gpu', ctypes.c_uint), ('memory', ctypes.c_uint)]
                util = nvmlUtilization_t()
                if self.nvml_lib.nvmlDeviceGetUtilizationRates(self.nvml_handle, ctypes.byref(util)) == 0:
                    telemetry["gpu_util"] = int(util.gpu)

                class nvmlMemory_t(ctypes.Structure):
                    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]
                mem = nvmlMemory_t()
                if self.nvml_lib.nvmlDeviceGetMemoryInfo(self.nvml_handle, ctypes.byref(mem)) == 0:
                    used_gb = mem.used / (1024**3)
                    total_gb = mem.total / (1024**3)
                    telemetry["vram_used_gb"] = used_gb
                    telemetry["vram_total_gb"] = total_gb
                    telemetry["vram_pct"] = (used_gb / max(total_gb, 0.1)) * 100.0
                    telemetry["paging_risk"] = telemetry["vram_pct"] > 92.0
            except Exception:
                pass
        return telemetry


class AbuAlignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Abu Align & Train - Version 0.0.1.7 (Neural Photogrammetry & 3DGS Training Suite)")
        self.root.geometry("1100x870")
        self.root.minsize(920, 700)

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
        self.is_paused = False
        self.latest_output_dir = None
        self.active_log_path = None
        self._ensure_logs_dir()

        # ----------------------------------------------------
        # Alignment Engine & Parameter Variables
        # ----------------------------------------------------
        self.engine_var = tk.StringVar(value="vggt_omega")
        self.res_var = tk.StringVar(value="1024")
        self.stride_var = tk.StringVar(value="1")

        # VGGT Specific Options
        self.density_var = tk.StringVar(value="high")
        self.align_var = tk.StringVar(value="ultra")
        self.lens_var = tk.StringVar(value="brown")

        # ACE0 Specific Options
        self.ace_iter_var = tk.StringVar(value="30")
        self.ace_refine_var = tk.StringVar(value="mlp")
        self.ace_seeds_var = tk.StringVar(value="3")
        self.ace_calib_var = tk.BooleanVar(value=True)
        self.ace_dense_var = tk.BooleanVar(value=True)

        # PyCuSFM Specific Options
        self.cusfm_feat_var = tk.StringVar(value="aliked")
        self.cusfm_matcher_var = tk.StringVar(value="lightglue")
        self.cusfm_init_var = tk.StringVar(value="omega")
        self.cusfm_ba_var = tk.StringVar(value="100")
        self.cusfm_rgb_var = tk.BooleanVar(value=True)

        # Cache & Windowing Options
        self.cache_var = tk.BooleanVar(value=True)
        self.chunk_size_var = tk.StringVar(value="96")
        self.anchor_var = tk.StringVar(value="24")

        # Hardware Resource & Telemetry Monitor
        self.hw_monitor = SystemHardwareMonitor()

        # ----------------------------------------------------
        # Training Engine & Parameter Variables
        # ----------------------------------------------------
        self.auto_train_var = tk.BooleanVar(value=True)
        self.train_engine_var = tk.StringVar(value="lichtfeld")
        self.train_iter_var = tk.StringVar(value="30,000 (Production High-Fidelity)")

        # LichtFeld Studio Specific Options
        self.lf_mode_var = tk.StringVar(value="interactive")
        self.lf_gut_var = tk.BooleanVar(value=True)
        self.lf_sh_var = tk.StringVar(value="3")
        self.lf_format_var = tk.StringVar(value="ply")

        # Nerfstudio Splatfacto Specific Options
        self.ns_model_var = tk.StringVar(value="splatfacto")
        self.ns_cam_opt_var = tk.StringVar(value="so3_x_r3")
        self.ns_bilateral_var = tk.BooleanVar(value=True)
        self.ns_mip_var = tk.BooleanVar(value=True)
        self.ns_sh_var = tk.StringVar(value="3")

        # NVIDIA 3DGRUT Specific Options
        self.grut_repr_var = tk.StringVar(value="3dgut")
        self.grut_bvh_var = tk.StringVar(value="optix")
        self.grut_exposure_var = tk.BooleanVar(value=True)
        self.grut_sparsity_var = tk.BooleanVar(value=False)
        self.grut_sh_var = tk.StringVar(value="3")

        # Legacy aliases for backward compatibility
        self.train_mode_var = self.lf_mode_var
        self.gut_var = self.lf_gut_var
        self.bilateral_var = self.ns_bilateral_var
        self.exposure_var = self.grut_exposure_var
        self.sparsity_var = self.grut_sparsity_var
        self.mip_var = self.ns_mip_var

        self._build_menu()
        self._build_ui()
        self._check_queue()
        self.root.after(500, self._update_hardware_telemetry)

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
            text="v0.0.1.7",
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

        # Options Grid: Fixed Left Column + Dynamic Right Panel
        opts_grid = tk.Frame(control_card, bg=self.panel_bg)
        opts_grid.pack(fill=tk.X, pady=(8, 4))

        # Column 1: Engine Selection & Canvas / Stride (Fixed)
        col1 = tk.Frame(opts_grid, bg=self.panel_bg)
        col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(4, 12))

        tk.Label(col1, text="Alignment Engine:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_omega = tk.Radiobutton(col1, text="VGGT-Omega (Fast, Low VRAM)", variable=self.engine_var, value="vggt_omega", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_omega.pack(anchor="w")
        rb_ace0 = tk.Radiobutton(col1, text="ACE0 (Niantic Labs SfM)", variable=self.engine_var, value="ace0", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace0.pack(anchor="w")
        rb_cusfm = tk.Radiobutton(col1, text="NVIDIA PyCuSFM (cuSFM GPU)", variable=self.engine_var, value="pycusfm", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cusfm.pack(anchor="w")

        ToolTip(rb_omega, "VGGT-Omega (2025/2026): Compressed register token architecture. Reduces VRAM usage by ~60% while maintaining wide-baseline alignment.")
        ToolTip(rb_ace0, "ACE0 (Niantic Labs ECCV 2024): Learning-based Structure-from-Motion. Trains an implicit scene coordinate regression network from scratch on unposed images. Ultra-fast, extremely low VRAM (~2GB), and highly robust.")
        ToolTip(rb_cusfm, "NVIDIA PyCuSFM (cuSFM): High-throughput CUDA SfM framework by NVIDIA Isaac. Learned ALIKED/SuperPoint keypoints, LightGlue GPU matching, and cuDSS sparse bundle adjustment.")

        sub_row = tk.Frame(col1, bg=self.panel_bg)
        sub_row.pack(anchor="w", pady=(4, 0))
        tk.Label(sub_row, text="Res:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        res_cb = ttk.Combobox(sub_row, textvariable=self.res_var, values=["512", "1024", "1536", "2048"], state="readonly", width=6)
        res_cb.pack(side=tk.LEFT, padx=(2, 6))
        ToolTip(res_cb, "Canvas Resolution: 512 is fastest for preview/low VRAM; 1024/1536/2048 offers finer spatial detail for feature tracking on 40MP-60MP cameras.")

        tk.Label(sub_row, text="Stride:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        stride_cb = ttk.Combobox(sub_row, textvariable=self.stride_var, values=["1", "2", "3", "4", "5"], state="readonly", width=4)
        stride_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(stride_cb, "Keyframe Stride: Frame sampling interval. 1 processes every photo; 2 or 3 skips intermediate frames (ideal for dense video or drone flights).")

        # Dynamic Options Container for Selected Alignment Engine
        self.align_dynamic_container = tk.Frame(opts_grid, bg=self.panel_bg)
        self.align_dynamic_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_align_panels()
        self.engine_var.trace_add("write", lambda *args: self._update_align_panel())
        self._update_align_panel()

        # Smart Cache & Memory Controls Row
        cache_row = tk.Frame(control_card, bg=self.panel_bg)
        cache_row.pack(fill=tk.X, pady=(6, 0))

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
        chunk_cb = ttk.Combobox(
            cache_row,
            textvariable=self.chunk_size_var,
            values=[
                "16", "24", "32", "48", "64", "96", "128", 
                "160", "192", "224", "256", "320", "384", "448", "512", "768", "1024"
            ],
            width=5
        )
        chunk_cb.pack(side=tk.LEFT, padx=(2, 8))
        ToolTip(chunk_cb, "Sliding-window chunk size for large datasets. Select a preset (up to 1,024) or type any custom number directly into the box.")

        tk.Label(cache_row, text="Anchor:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        anchor_cb = ttk.Combobox(
            cache_row,
            textvariable=self.anchor_var,
            values=[
                "4", "8", "12", "16", "24", "32", "48", "64", "96", "128", "192", "256"
            ],
            width=5
        )
        anchor_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(anchor_cb, "Anchor overlap frames connecting consecutive sliding windows. Select a preset or type any custom number directly into the box.")

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

        # Train Col 1: Auto-Train, Engine Selection & Common Iterations (Fixed)
        tcol1 = tk.Frame(train_grid, bg=self.panel_bg)
        tcol1.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(4, 12))

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
        rb_t_lf = tk.Radiobutton(tcol1, text="LichtFeld Studio (Vulkan Live GUI + GUT)", variable=self.train_engine_var, value="lichtfeld", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_t_lf.pack(anchor="w")
        rb_t_ns = tk.Radiobutton(tcol1, text="Nerfstudio Splatfacto (PyTorch SOTA + PLY)", variable=self.train_engine_var, value="splatfacto", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_t_ns.pack(anchor="w")
        rb_t_gr = tk.Radiobutton(tcol1, text="NVIDIA 3DGRUT / 3DGUT (Ray Traced & GUT)", variable=self.train_engine_var, value="3dgrut", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_t_gr.pack(anchor="w")

        ToolTip(rb_t_lf, "LichtFeld Studio: Native high-performance Vulkan/C++ 3DGS trainer with live real-time viewport (120+ FPS) and MCP bridge.")
        ToolTip(rb_t_ns, "Nerfstudio Splatfacto: State-of-the-Art PyTorch 3DGS optimizer with automatic standalone Gaussian Splatting PLY exporter.")
        ToolTip(rb_t_gr, "NVIDIA 3DGRUT / 3DGUT: NVIDIA Research framework for Gaussian ray tracing and unscented transform handling complex non-linear optics.")

        t_iter_row = tk.Frame(tcol1, bg=self.panel_bg)
        t_iter_row.pack(anchor="w", pady=(4, 0))
        tk.Label(t_iter_row, text="Iterations:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        iter_cb = ttk.Combobox(
            t_iter_row,
            textvariable=self.train_iter_var,
            values=[
                "7,000 (Fast Preview)",
                "15,000 (Balanced / Standard)",
                "30,000 (Production High-Fidelity)"
            ],
            state="readonly",
            width=26
        )
        iter_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(iter_cb, "Training duration: 7,000 steps (~1-2 min preview); 15,000 steps (~3-5 min standard); 30,000 steps (~7-10 min full production quality).")

        # Dynamic Options Container for Selected Training Engine
        self.train_dynamic_container = tk.Frame(train_grid, bg=self.panel_bg)
        self.train_dynamic_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_train_panels()
        self.train_engine_var.trace_add("write", lambda *args: self._update_train_panel())
        self._update_train_panel()

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

        self.pause_btn = tk.Button(
            action_bar,
            text="⏸  PAUSE",
            font=("Segoe UI", 10, "bold"),
            bg="#f59e0b",
            fg="#ffffff",
            activebackground="#d97706",
            relief=tk.FLAT,
            padx=12,
            pady=6,
            state=tk.DISABLED,
            command=self._toggle_pause
        )
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 6))

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

        ToolTip(self.one_click_btn, "One-Click Pipeline: Runs complete photogrammetry alignment and immediately launches 3DGS training automatically.")
        ToolTip(self.start_btn, "Start Alignment Only: Runs photogrammetry and exports COLMAP, RealityCapture CSV, transforms.json, and 3dgs_dataset/.")
        ToolTip(self.train_only_btn, "Train 3DGS Only: Train a 3D Gaussian Splatting model directly from an existing alignment (3dgs_dataset/ or results/).")
        ToolTip(self.pause_btn, "Pause / Resume: Freeze or thaw the current alignment or training process tree without losing VRAM or progress.")
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

        # Stage Breadcrumbs Flow
        stages_bar = tk.Frame(console_card, bg=self.panel_bg)
        stages_bar.pack(fill=tk.X, pady=(2, 6))

        stage_names = [
            "1. Scan",
            "2. GPU Init",
            "3. Preprocess",
            "4. 3D Inf",
            "5. Tracking & BA",
            "6. 3DGS Train"
        ]
        self.stage_badges = []
        for i, name in enumerate(stage_names):
            badge = tk.Label(
                stages_bar,
                text=name,
                font=("Segoe UI", 8, "bold"),
                fg="#64748b",
                bg="#232338",
                padx=8,
                pady=2,
                relief=tk.FLAT
            )
            badge.pack(side=tk.LEFT, padx=(0, 2))
            self.stage_badges.append(badge)
            if i < len(stage_names) - 1:
                sep = tk.Label(
                    stages_bar,
                    text="➔",
                    font=("Segoe UI", 8),
                    fg="#475569",
                    bg=self.panel_bg
                )
                sep.pack(side=tk.LEFT, padx=(0, 2))

        # Progress Bar & Real-Time Telemetry Bar
        telemetry_frame = tk.Frame(console_card, bg=self.panel_bg)
        telemetry_frame.pack(fill=tk.X, pady=(0, 4))

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

        # High-Contrast Progress Bar Container (Border + Gauge + Percentage Readout)
        prog_container = tk.Frame(
            console_card,
            bg="#11111b",
            highlightbackground="#3b82f6",
            highlightcolor="#3b82f6",
            highlightthickness=1,
            padx=2,
            pady=2
        )
        prog_container.pack(fill=tk.X, pady=(0, 6))

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
            bordercolor="#11111b",
            thickness=18
        )

        self.progress_bar = ttk.Progressbar(
            prog_container,
            orient="horizontal",
            mode="determinate",
            style="Abu.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 6), pady=2)

        self.pct_lbl = tk.Label(
            prog_container,
            text="  0.0% ",
            font=("Segoe UI", 9, "bold"),
            fg="#60a5fa",
            bg="#11111b",
            width=8
        )
        self.pct_lbl.pack(side=tk.RIGHT, padx=(0, 4))

        # Real-Time System Hardware Telemetry Bar (CPU, RAM, GPU Core, VRAM Dedicated, Shared Paging, GPU Temp)
        self.hw_bar = tk.Frame(
            console_card,
            bg="#161622",
            highlightbackground="#2e2e44",
            highlightcolor="#3b82f6",
            highlightthickness=1,
            padx=8,
            pady=3
        )
        self.hw_bar.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            self.hw_bar,
            text="⚡ HARDWARE:",
            font=("Segoe UI", 8, "bold"),
            fg="#38bdf8",
            bg="#161622"
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.hw_cpu_lbl = tk.Label(
            self.hw_bar,
            text="CPU: --%",
            font=("Segoe UI", 8, "bold"),
            fg="#e2e8f0",
            bg="#232338",
            padx=6,
            pady=1
        )
        self.hw_cpu_lbl.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.hw_cpu_lbl, "Total CPU Processor Utilization % across all cores.")

        self.hw_ram_lbl = tk.Label(
            self.hw_bar,
            text="RAM: --/-- GB",
            font=("Segoe UI", 8, "bold"),
            fg="#e2e8f0",
            bg="#232338",
            padx=6,
            pady=1
        )
        self.hw_ram_lbl.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.hw_ram_lbl, "System Host RAM (Used / Total in GB and percentage).")

        self.hw_gpu_lbl = tk.Label(
            self.hw_bar,
            text="GPU: --%",
            font=("Segoe UI", 8, "bold"),
            fg="#34d399",
            bg="#232338",
            padx=6,
            pady=1
        )
        self.hw_gpu_lbl.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.hw_gpu_lbl, "NVIDIA GPU Core Compute Utilization %.")

        self.hw_temp_lbl = tk.Label(
            self.hw_bar,
            text="GPU Temp: --°C",
            font=("Segoe UI", 8, "bold"),
            fg="#fbbf24",
            bg="#232338",
            padx=6,
            pady=1
        )
        self.hw_temp_lbl.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.hw_temp_lbl, "NVIDIA GPU Core Temperature (°C).")

        self.hw_vram_lbl = tk.Label(
            self.hw_bar,
            text="VRAM: --/-- GB",
            font=("Segoe UI", 8, "bold"),
            fg="#a78bfa",
            bg="#232338",
            padx=6,
            pady=1
        )
        self.hw_vram_lbl.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.hw_vram_lbl, "Dedicated GPU Video RAM (Used / Total in GB and percentage).")

        self.hw_page_lbl = tk.Label(
            self.hw_bar,
            text="Paging: OK",
            font=("Segoe UI", 8, "bold"),
            fg="#10b981",
            bg="#1e293b",
            padx=6,
            pady=1
        )
        self.hw_page_lbl.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(self.hw_page_lbl, "Windows WDDM Shared Memory Paging Guard:\n• 'Paging: OK': Dedicated VRAM is within safe margins (<92%). Full 1,008 GB/s bandwidth.\n• '⚠️ VRAM SPILL': VRAM overcommitted, spilling into Shared GPU Memory over PCIe bus (~40x slower).")

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
        self._log("Abu Align & Train Version 0.0.1.7 Ready.\nQuad Alignment Engines: VGGT Foundation, VGGT-Omega, ACE0, & NVIDIA PyCuSFM.\nTriple 3DGS Training Engines: LichtFeld Studio, Nerfstudio Splatfacto, & NVIDIA 3DGRUT.\nOne-Click Align & Train: Fully Automated Handoff Active.\nSelect a photoscan directory and click 'ONE-CLICK ALIGN & TRAIN' or 'ALIGNMENT ONLY'.\n\n")

    def _build_align_panels(self):
        """Creates the 3 engine-specific alignment panels inside self.align_dynamic_container."""
        # =========================================================================
        # 1. VGGT / VGGT-Omega Subpanel
        # =========================================================================
        self.panel_align_vggt = tk.Frame(self.align_dynamic_container, bg=self.panel_bg)

        # Col A: Point Cloud Density
        v_col1 = tk.Frame(self.panel_align_vggt, bg=self.panel_bg)
        v_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(v_col1, text="Point Cloud Density:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_q1 = tk.Radiobutton(v_col1, text="Preview (100K pts, Conf 2.2)", variable=self.density_var, value="preview", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q1.pack(anchor="w")
        rb_q2 = tk.Radiobutton(v_col1, text="High Quality (300K pts, Conf 1.8)", variable=self.density_var, value="high", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q2.pack(anchor="w")
        rb_q3 = tk.Radiobutton(v_col1, text="Ultra Dense (800K pts, Conf 1.5)", variable=self.density_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q3.pack(anchor="w")
        rb_q4 = tk.Radiobutton(v_col1, text="Max Scan Grade (1.5M pts, Conf 1.2)", variable=self.density_var, value="max", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_q4.pack(anchor="w")
        ToolTip(rb_q1, "Preview (100K pts, Conf >= 2.2):\nHigh confidence filter for fastest point cloud generation (~15-20s).\nBest for quick spatial checks.")
        ToolTip(rb_q2, "High Quality (300K pts, Conf >= 1.8) [Recommended]:\nOptimal balance of surface density and cleanliness.\nRecommended initial point cloud seed for 3DGS training.")
        ToolTip(rb_q3, "Ultra Dense (800K pts, Conf >= 1.5):\nDeep surface coverage capturing complex recesses, background geometry, and secondary objects.")
        ToolTip(rb_q4, "Max Scan Grade (1.5M pts, Conf >= 1.2):\nMaximum geometric fidelity. Preserves thin wires, sharp corners, and foliage.")

        # Col B: Camera Alignment Accuracy
        v_col2 = tk.Frame(self.panel_align_vggt, bg=self.panel_bg)
        v_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(v_col2, text="Camera Alignment Accuracy:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_a1 = tk.Radiobutton(v_col2, text="Feedforward (Fast, ~25s)", variable=self.align_var, value="none", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a1.pack(anchor="w")
        rb_a2 = tk.Radiobutton(v_col2, text="Standard Ceres BA (~1m)", variable=self.align_var, value="standard", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a2.pack(anchor="w")
        rb_a3 = tk.Radiobutton(v_col2, text="Ultra Sub-Pixel BA (Cauchy+Polish)", variable=self.align_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a3.pack(anchor="w")
        ToolTip(rb_a1, "Feedforward (~25s):\nPure deep-learning prediction directly from VGGT model forward pass.\nNo non-linear bundle adjustment (~2-4px reprojection error).")
        ToolTip(rb_a2, "Standard Ceres BA (~1m):\nALIKED neural feature tracks refined by Ceres Levenberg-Marquardt optimizer.")
        ToolTip(rb_a3, "Ultra Sub-Pixel BA (Cauchy + Polish) [Recommended]:\nSolves 3,072 multi-view tracks with Cauchy robust loss to reject outliers,\nfollowed by outlier pruning and a 2nd Polish pass reaching 0.35 px error.")

        # Col C: Lens Distortion Calibration
        v_col3 = tk.Frame(self.panel_align_vggt, bg=self.panel_bg)
        v_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(v_col3, text="Lens Distortion Model:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_l1 = tk.Radiobutton(v_col3, text="Pinhole (Ideal, zero distortion)", variable=self.lens_var, value="pinhole", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_l1.pack(anchor="w")
        rb_l2 = tk.Radiobutton(v_col3, text="Brown Calibration (k1-k2, t1-t2)", variable=self.lens_var, value="brown", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_l2.pack(anchor="w")
        ToolTip(rb_l1, "Pinhole Model:\nIdeal rectilinear geometry with zero distortion (k1=0, k2=0, t1=0, t2=0).\nNative to VGGT training prior. Recommended for pre-undistorted images.")
        ToolTip(rb_l2, "Brown Calibration (k1-k2, t1-t2) [Recommended]:\nJointly solves physical radial (k1, k2) and tangential (t1, t2) lens curvature in Ceres.\nEssential for wide-angle and smartphone photoscans.")

        # =========================================================================
        # 2. ACE0 Subpanel (Niantic Labs Scene Coordinate Regression)
        # =========================================================================
        self.panel_align_ace0 = tk.Frame(self.align_dynamic_container, bg=self.panel_bg)

        # Col A: Mapping Rounds (Iterations)
        ace_col1 = tk.Frame(self.panel_align_ace0, bg=self.panel_bg)
        ace_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ace_col1, text="ACE0 Mapping Rounds:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_ace_r1 = tk.Radiobutton(ace_col1, text="15 Rounds (Fast Draft)", variable=self.ace_iter_var, value="15", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace_r1.pack(anchor="w")
        rb_ace_r2 = tk.Radiobutton(ace_col1, text="30 Rounds (Standard Default)", variable=self.ace_iter_var, value="30", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace_r2.pack(anchor="w")
        rb_ace_r3 = tk.Radiobutton(ace_col1, text="50 Rounds (High-Fidelity)", variable=self.ace_iter_var, value="50", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace_r3.pack(anchor="w")
        ToolTip(rb_ace_r1, "15 Rounds (~30s):\nRapid training of the ACE0 scene coordinate MLP.\nBest for quick checks of camera coverage and scene geometry.")
        ToolTip(rb_ace_r2, "30 Rounds [Niantic Official Default]:\nOptimal balance between training speed and sub-millimeter pose convergence.")
        ToolTip(rb_ace_r3, "50 Rounds (High-Fidelity):\nExtended training iterations for large multi-room spaces, complex objects, or weak texture scenes.")

        # Col B: Pose Refinement Mode
        ace_col2 = tk.Frame(self.panel_align_ace0, bg=self.panel_bg)
        ace_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ace_col2, text="Pose Refinement Mode:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_ace_m1 = tk.Radiobutton(ace_col2, text="MLP Refinement (Learned)", variable=self.ace_refine_var, value="mlp", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace_m1.pack(anchor="w")
        rb_ace_m2 = tk.Radiobutton(ace_col2, text="Naive Direct Refinement", variable=self.ace_refine_var, value="naive", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace_m2.pack(anchor="w")
        rb_ace_m3 = tk.Radiobutton(ace_col2, text="None (Feedforward Regression)", variable=self.ace_refine_var, value="none", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ace_m3.pack(anchor="w")
        ToolTip(rb_ace_m1, "MLP Refinement [Recommended]:\nJointly optimizes 6DoF camera poses and ACE0 scene coordinates by propagating reprojection gradients through the MLP.")
        ToolTip(rb_ace_m2, "Naive Refinement:\nClassic Levenberg-Marquardt PnP reprojection minimization against fixed predicted coordinates.")
        ToolTip(rb_ace_m3, "None:\nUses direct RANSAC PnP poses without non-linear gradient bundle optimization.")

        # Col C: Seeds & Calibration Refinement
        ace_col3 = tk.Frame(self.panel_align_ace0, bg=self.panel_bg)
        ace_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ace_col3, text="Init Seeds & Calibration:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        seeds_row = tk.Frame(ace_col3, bg=self.panel_bg)
        seeds_row.pack(anchor="w", pady=(2, 4))
        tk.Label(seeds_row, text="Init Seeds:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        seeds_cb = ttk.Combobox(seeds_row, textvariable=self.ace_seeds_var, values=["1", "3", "5"], state="readonly", width=4)
        seeds_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(seeds_cb, "Initialization Seeds:\nNumber of relative pose seeds evaluated during mapping.\n3 seeds provides high robustness against local minima in unordered photos.")

        ace_calib_chk = tk.Checkbutton(ace_col3, text="Refine Focal Length Calib", variable=self.ace_calib_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ace_calib_chk.pack(anchor="w")
        ToolTip(ace_calib_chk, "Refine Focal Length Calibration:\nOptimizes camera focal length during mapping rounds.\nCrucial when image EXIF focal length is missing or uncalibrated.")

        ace_dense_chk = tk.Checkbutton(ace_col3, text="Export Dense Point Cloud", variable=self.ace_dense_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ace_dense_chk.pack(anchor="w")
        ToolTip(ace_dense_chk, "Export Dense Point Cloud:\nEvaluates ACE0 MLP predictions over all valid pixels to produce a dense 3D point cloud seed for 3DGS training.")

        # =========================================================================
        # 3. NVIDIA PyCuSFM Subpanel (cuSFM CUDA Framework)
        # =========================================================================
        self.panel_align_pycusfm = tk.Frame(self.align_dynamic_container, bg=self.panel_bg)

        # Col A: Features & Matcher
        cu_col1 = tk.Frame(self.panel_align_pycusfm, bg=self.panel_bg)
        cu_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(cu_col1, text="Feature Extractor & Matcher:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        f_row = tk.Frame(cu_col1, bg=self.panel_bg)
        f_row.pack(anchor="w", pady=(2, 2))
        tk.Label(f_row, text="Features:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        cu_feat_cb = ttk.Combobox(f_row, textvariable=self.cusfm_feat_var, values=["aliked", "superpoint", "sift"], state="readonly", width=10)
        cu_feat_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(cu_feat_cb, "Feature Extractor:\n• ALIKED (default): Ultra-fast neural keypoints, deformable kernel, sharp corners.\n• SuperPoint: Highly robust under dynamic lighting.\n• SIFT: Classical scale-invariant feature transform on CUDA.")

        m_row = tk.Frame(cu_col1, bg=self.panel_bg)
        m_row.pack(anchor="w", pady=(2, 2))
        tk.Label(m_row, text="Matcher:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        cu_match_cb = ttk.Combobox(m_row, textvariable=self.cusfm_matcher_var, values=["lightglue", "superglue"], state="readonly", width=10)
        cu_match_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(cu_match_cb, "Keypoint Matcher:\n• LightGlue: SOTA deep transformer matcher with adaptive depth. 3x faster than SuperGlue.\n• SuperGlue: Attentional Graph Neural Network matcher.")

        # Col B: Pose Initializer Prior
        cu_col2 = tk.Frame(self.panel_align_pycusfm, bg=self.panel_bg)
        cu_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(cu_col2, text="Pose Initialization Prior:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_cu_i1 = tk.Radiobutton(cu_col2, text="VGGT-Omega Coarse Prior", variable=self.cusfm_init_var, value="omega", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cu_i1.pack(anchor="w")
        rb_cu_i2 = tk.Radiobutton(cu_col2, text="Visual Odometry (Ego)", variable=self.cusfm_init_var, value="ego", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cu_i2.pack(anchor="w")
        ToolTip(rb_cu_i1, "VGGT-Omega Prior [Recommended]:\nSeeds PyCuSFM bundle adjustment with global foundation model poses for instant convergence and zero drift.")
        ToolTip(rb_cu_i2, "Visual Odometry (Ego):\nSequential pairwise epipolar chaining. Ideal for continuous video captures and smooth drone fly-throughs.")

        # Col C: cuDSS Sparse Solver & Colorization
        cu_col3 = tk.Frame(self.panel_align_pycusfm, bg=self.panel_bg)
        cu_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(cu_col3, text="cuDSS GPU Sparse Solver:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        ba_row = tk.Frame(cu_col3, bg=self.panel_bg)
        ba_row.pack(anchor="w", pady=(2, 4))
        tk.Label(ba_row, text="BA Iterations:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        cu_ba_cb = ttk.Combobox(ba_row, textvariable=self.cusfm_ba_var, values=["50", "100", "200"], state="readonly", width=6)
        cu_ba_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(cu_ba_cb, "Global Bundle Adjustment Iterations:\ncuDSS GPU direct sparse solver iterations for global landmark and camera pose optimization.")

        cu_rgb_chk = tk.Checkbutton(cu_col3, text="Dense RGB Point Colorization", variable=self.cusfm_rgb_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        cu_rgb_chk.pack(anchor="w")
        ToolTip(cu_rgb_chk, "Dense RGB Colorization:\nProjects aligned 3D points back onto source photos to extract accurate RGB colors for 3DGS seed initialization.")

    def _update_align_panel(self):
        engine = self.engine_var.get()
        self.panel_align_vggt.pack_forget()
        self.panel_align_ace0.pack_forget()
        self.panel_align_pycusfm.pack_forget()

        if engine in ["vggt_omega", "vggt"]:
            self.panel_align_vggt.pack(fill=tk.BOTH, expand=True)
        elif engine == "ace0":
            self.panel_align_ace0.pack(fill=tk.BOTH, expand=True)
        elif engine == "pycusfm":
            self.panel_align_pycusfm.pack(fill=tk.BOTH, expand=True)

    def _build_train_panels(self):
        """Creates the 3 engine-specific training panels inside self.train_dynamic_container."""
        # =========================================================================
        # 1. LichtFeld Studio Subpanel
        # =========================================================================
        self.panel_train_lichtfeld = tk.Frame(self.train_dynamic_container, bg=self.panel_bg)

        # Col A: Viewport & GUI Mode
        lf_col1 = tk.Frame(self.panel_train_lichtfeld, bg=self.panel_bg)
        lf_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(lf_col1, text="Training Viewport Mode:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_lf_m1 = tk.Radiobutton(lf_col1, text="Interactive Viewport (Live 3D GUI)", variable=self.lf_mode_var, value="interactive", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_lf_m1.pack(anchor="w")
        rb_lf_m2 = tk.Radiobutton(lf_col1, text="Headless Background (Fast CLI)", variable=self.lf_mode_var, value="headless", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_lf_m2.pack(anchor="w")
        ToolTip(rb_lf_m1, "Interactive Viewport [Default]:\nOpens LichtFeld Studio's Vulkan window with real-time camera controls and live splat rasterization at 120+ FPS.")
        ToolTip(rb_lf_m2, "Headless Background:\nRuns high-throughput headless CLI training in the background without launching the GUI window.")

        # Col B: Optics & Format
        lf_col2 = tk.Frame(self.panel_train_lichtfeld, bg=self.panel_bg)
        lf_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(lf_col2, text="Optics & Export Format:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        gut_chk = tk.Checkbutton(lf_col2, text="Enable GUT Optics (Unscented)", variable=self.lf_gut_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        gut_chk.pack(anchor="w")
        ToolTip(gut_chk, "Gaussian Unscented Transform (GUT):\nPropagates ray bundles through camera lens models to eliminate ray distortion and blur in peripheral splats.")

        fmt_row = tk.Frame(lf_col2, bg=self.panel_bg)
        fmt_row.pack(anchor="w", pady=(2, 2))
        tk.Label(fmt_row, text="Format:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        fmt_cb = ttk.Combobox(fmt_row, textvariable=self.lf_format_var, values=["ply", "spz"], state="readonly", width=8)
        fmt_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(fmt_cb, "Export Format:\n• PLY: Universal standard 3DGS point format.\n• SPZ: Compressed splat format (~10x smaller, FP16 packed quaternions).")

        # Col C: Spherical Harmonics
        lf_col3 = tk.Frame(self.panel_train_lichtfeld, bg=self.panel_bg)
        lf_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(lf_col3, text="Spherical Harmonics (SH):", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        sh_row = tk.Frame(lf_col3, bg=self.panel_bg)
        sh_row.pack(anchor="w", pady=(2, 4))
        tk.Label(sh_row, text="Degree:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        lf_sh_cb = ttk.Combobox(sh_row, textvariable=self.lf_sh_var, values=["0", "1", "2", "3"], state="readonly", width=4)
        lf_sh_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(lf_sh_cb, "Spherical Harmonics Degree:\n• 0: Diffuse only (fastest, flat lighting).\n• 1-2: Directional highlights.\n• 3 [Recommended]: Full view-dependent specular reflections and realistic sheen.")

        # =========================================================================
        # 2. Nerfstudio Splatfacto Subpanel
        # =========================================================================
        self.panel_train_nerfstudio = tk.Frame(self.train_dynamic_container, bg=self.panel_bg)

        # Col A: Model Architecture
        ns_col1 = tk.Frame(self.panel_train_nerfstudio, bg=self.panel_bg)
        ns_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ns_col1, text="Model Architecture:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_ns_m1 = tk.Radiobutton(ns_col1, text="Splatfacto Standard", variable=self.ns_model_var, value="splatfacto", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ns_m1.pack(anchor="w")
        rb_ns_m2 = tk.Radiobutton(ns_col1, text="Splatfacto Big (High Res)", variable=self.ns_model_var, value="splatfacto-big", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ns_m2.pack(anchor="w")
        ToolTip(rb_ns_m1, "Splatfacto Standard:\nBalanced PyTorch optimizer with fast densification and gradient thresholding. Recommended for standard scenes.")
        ToolTip(rb_ns_m2, "Splatfacto Big:\nHigh-capacity variant with larger Gaussian point budgets. Recommended for 40-60MP DSLR captures and massive scenes.")

        # Col B: Camera Optimizer
        ns_col2 = tk.Frame(self.panel_train_nerfstudio, bg=self.panel_bg)
        ns_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ns_col2, text="Camera Optimizer:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_ns_o1 = tk.Radiobutton(ns_col2, text="SO(3) x R(3) Pose Refine", variable=self.ns_cam_opt_var, value="so3_x_r3", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ns_o1.pack(anchor="w")
        rb_ns_o2 = tk.Radiobutton(ns_col2, text="SE(3) Lie Algebra", variable=self.ns_cam_opt_var, value="se3", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ns_o2.pack(anchor="w")
        rb_ns_o3 = tk.Radiobutton(ns_col2, text="None (Locked Poses)", variable=self.ns_cam_opt_var, value="none", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_ns_o3.pack(anchor="w")
        ToolTip(rb_ns_o1, "SO(3) x R(3) [Recommended]:\nRefines rotation and translation during training to absorb small SfM residual errors.")
        ToolTip(rb_ns_o2, "SE(3) Lie Algebra:\nFull manifold pose optimization for continuous trajectories.")
        ToolTip(rb_ns_o3, "None:\nFreezes camera poses strictly as produced by the alignment stage.")

        # Col C: Filters & SH Degree
        ns_col3 = tk.Frame(self.panel_train_nerfstudio, bg=self.panel_bg)
        ns_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ns_col3, text="Filters & Spherical Harmonics:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        ns_bilat_chk = tk.Checkbutton(ns_col3, text="Bilateral Grid Color Correct", variable=self.ns_bilateral_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ns_bilat_chk.pack(anchor="w")
        ToolTip(ns_bilat_chk, "Bilateral Grid:\nCorrects non-linear exposure shifts, dynamic lighting, and white balance across photos.")

        ns_mip_chk = tk.Checkbutton(ns_col3, text="Anti-Aliased Mip-Filtering", variable=self.ns_mip_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ns_mip_chk.pack(anchor="w")
        ToolTip(ns_mip_chk, "Anti-Aliased Mip-Filtering:\nMip-Splatting eliminates high-frequency moire and sparkling artifacts when viewing at lower zoom levels.")

        ns_sh_row = tk.Frame(ns_col3, bg=self.panel_bg)
        ns_sh_row.pack(anchor="w", pady=(2, 2))
        tk.Label(ns_sh_row, text="SH Degree:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        ns_sh_cb = ttk.Combobox(ns_sh_row, textvariable=self.ns_sh_var, values=["0", "1", "2", "3", "4"], state="readonly", width=4)
        ns_sh_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(ns_sh_cb, "Spherical Harmonics Degree:\nDegree 3 is standard. Degree 4 provides ultra-rich specular highlights.")

        # =========================================================================
        # 3. NVIDIA 3DGRUT / 3DGUT Subpanel
        # =========================================================================
        self.panel_train_3dgrut = tk.Frame(self.train_dynamic_container, bg=self.panel_bg)

        # Col A: Representation Type
        gr_col1 = tk.Frame(self.panel_train_3dgrut, bg=self.panel_bg)
        gr_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(gr_col1, text="Representation Model:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_gr_r1 = tk.Radiobutton(gr_col1, text="3DGUT (Unscented Transform)", variable=self.grut_repr_var, value="3dgut", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_gr_r1.pack(anchor="w")
        rb_gr_r2 = tk.Radiobutton(gr_col1, text="3DGRUT (Ray Traced & GUT)", variable=self.grut_repr_var, value="3dgrut", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_gr_r2.pack(anchor="w")
        ToolTip(rb_gr_r1, "3DGUT [Default]:\nGaussian Unscented Transform handles non-linear camera projections and lens distortion during splat rendering.")
        ToolTip(rb_gr_r2, "3DGRUT:\nNVIDIA Research Gaussian Ray Tracing formulation with BVH hardware acceleration.")

        # Col B: BVH RT Acceleration Backend
        gr_col2 = tk.Frame(self.panel_train_3dgrut, bg=self.panel_bg)
        gr_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(gr_col2, text="BVH RT Backend:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_gr_b1 = tk.Radiobutton(gr_col2, text="OptiX RT Cores (Hardware)", variable=self.grut_bvh_var, value="optix", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_gr_b1.pack(anchor="w")
        rb_gr_b2 = tk.Radiobutton(gr_col2, text="CUDA Linear BVH (Universal)", variable=self.grut_bvh_var, value="cuda", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_gr_b2.pack(anchor="w")
        ToolTip(rb_gr_b1, "OptiX RT Cores [Recommended]:\nAccelerates bounding volume hierarchy traversal using NVIDIA RTX ray-tracing hardware.")
        ToolTip(rb_gr_b2, "CUDA Linear BVH:\nSoftware BVH traversal executable on all CUDA-capable NVIDIA GPUs.")

        # Col C: Exposure, Sparsity & SH
        gr_col3 = tk.Frame(self.panel_train_3dgrut, bg=self.panel_bg)
        gr_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(gr_col3, text="Optics, Sparsity & SH:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        gr_expo_chk = tk.Checkbutton(gr_col3, text="Auto Exposure / WB Align", variable=self.grut_exposure_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        gr_expo_chk.pack(anchor="w")
        ToolTip(gr_expo_chk, "Auto Exposure / White Balance:\nNVIDIA affine color correction optimizing exposure and gain per camera view.")

        gr_sparse_chk = tk.Checkbutton(gr_col3, text="ADMM Sparsity (-60% MB)", variable=self.grut_sparsity_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        gr_sparse_chk.pack(anchor="w")
        ToolTip(gr_sparse_chk, "ADMM Sparsity Pruning:\nAlternating Direction Method of Multipliers regularizer. Removes near-zero opacity Gaussians to shrink file size by 50-60%.")

        gr_sh_row = tk.Frame(gr_col3, bg=self.panel_bg)
        gr_sh_row.pack(anchor="w", pady=(2, 2))
        tk.Label(gr_sh_row, text="SH Degree:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        gr_sh_cb = ttk.Combobox(gr_sh_row, textvariable=self.grut_sh_var, values=["0", "1", "2", "3"], state="readonly", width=4)
        gr_sh_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(gr_sh_cb, "Spherical Harmonics Degree:\nDegree 3 is standard for realistic directional shine and reflections.")

    def _update_train_panel(self):
        engine = self.train_engine_var.get()
        self.panel_train_lichtfeld.pack_forget()
        self.panel_train_nerfstudio.pack_forget()
        self.panel_train_3dgrut.pack_forget()

        if engine == "lichtfeld":
            self.panel_train_lichtfeld.pack(fill=tk.BOTH, expand=True)
        elif engine == "splatfacto":
            self.panel_train_nerfstudio.pack(fill=tk.BOTH, expand=True)
        elif engine == "3dgrut":
            self.panel_train_3dgrut.pack(fill=tk.BOTH, expand=True)

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

    def _set_stage(self, stage_idx):
        """
        Updates the 6 stage badges:
        - indices < stage_idx: completed (green bg #065f46, green fg #34d399)
        - index == stage_idx: active (blue bg #2563eb, white fg #ffffff)
        - indices > stage_idx: pending (dark bg #232338, gray fg #64748b)
        """
        if not hasattr(self, "stage_badges"):
            return
        for i, badge in enumerate(self.stage_badges):
            if i < stage_idx:
                badge.config(bg="#065f46", fg="#34d399")
            elif i == stage_idx:
                badge.config(bg="#2563eb", fg="#ffffff")
            else:
                badge.config(bg="#232338", fg="#64748b")

    def _toggle_pause(self):
        """
        Universal Pause / Resume:
        Freezes or thaws the active pipeline process tree using psutil process suspension.
        Preserves all GPU VRAM allocations and computation state in RAM.
        """
        if not self.worker_process or not self.is_running:
            return
        if psutil is None:
            messagebox.showwarning("Pause/Resume", "psutil library is required for process suspension.")
            return

        try:
            proc = psutil.Process(self.worker_process.pid)
            if not self.is_paused:
                children = proc.children(recursive=True)
                for c in children:
                    try:
                        c.suspend()
                    except Exception:
                        pass
                proc.suspend()
                self.is_paused = True
                self.pause_btn.config(text="▶  RESUME", bg="#10b981")
                self.telemetry_lbl.config(text="Status: ⏸ PAUSED (GPU VRAM & state preserved in RAM)", fg="#f59e0b")
                self._log("\n[PROCESS] ⏸ Execution paused by user. Process tree suspended.\n")
            else:
                proc.resume()
                children = proc.children(recursive=True)
                for c in children:
                    try:
                        c.resume()
                    except Exception:
                        pass
                self.is_paused = False
                self.pause_btn.config(text="⏸  PAUSE", bg="#f59e0b")
                self.telemetry_lbl.config(text="Status: ▶ Resumed and executing...", fg=self.accent_blue)
                self._log("\n[PROCESS] ▶ Execution resumed by user.\n")
        except Exception as e:
            self._log(f"\n[ERROR] Failed to toggle pause: {e}\n")

    def _start_alignment(self, force_auto_train=False):
        if self.is_running:
            return

        dataset_path = self.dataset_var.get().strip()
        if not os.path.isdir(dataset_path):
            messagebox.showerror("Error", f"Dataset folder not found:\n{dataset_path}")
            return

        self.is_running = True
        self.is_paused = False
        self.start_btn.config(state=tk.DISABLED)
        self.one_click_btn.config(state=tk.DISABLED)
        self.train_only_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="⏸  PAUSE", bg="#f59e0b")
        self.stop_btn.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)

        self._set_stage(0)
        self.progress_bar["value"] = 0
        self.pct_lbl.config(text="  0.0%")
        self.telemetry_lbl.config(text="Status: Initializing Pipeline...", fg=self.accent_blue)
        self.eta_lbl.config(text="")

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = self._get_logs_dir()
        self.active_log_path = os.path.join(logs_dir, f"abu_align_{timestamp}.log")
        self.log_status_lbl.config(
            text=f"Log: abu_align_{timestamp}.log",
            fg=self.accent_green
        )

        align_engine = self.engine_var.get()
        cmd = [
            self.conda_python,
            "-u",
            self.cli_script,
            "--input", dataset_path,
            "--engine", align_engine,
            "--stride", str(self.stride_var.get()),
            "--load_res", str(self.res_var.get()),
        ]

        if not self.cache_var.get():
            cmd.append("--no_cache")

        try:
            chunk_val = max(int(str(self.chunk_size_var.get()).strip()), 8)
        except Exception:
            chunk_val = 96
        try:
            anchor_val = max(int(str(self.anchor_var.get()).strip()), 2)
        except Exception:
            anchor_val = 24

        cmd.extend([
            "--chunk_size", str(chunk_val),
            "--anchor_size", str(anchor_val)
        ])

        # Active Alignment Engine Options
        if align_engine in ["vggt_omega", "vggt"]:
            cmd.extend([
                "--align_mode", self.align_var.get(),
                "--lens_model", self.lens_var.get(),
            ])
            density = self.density_var.get()
            if density == "preview":
                cmd.extend(["--conf_thresh", "2.2", "--max_points", "100000"])
            elif density == "high":
                cmd.extend(["--conf_thresh", "1.8", "--max_points", "300000"])
            elif density == "ultra":
                cmd.extend(["--conf_thresh", "1.5", "--max_points", "800000"])
            elif density == "max":
                cmd.extend(["--conf_thresh", "1.2", "--max_points", "1500000"])
        elif align_engine == "ace0":
            cmd.extend([
                "--ace_iter", str(self.ace_iter_var.get()),
                "--ace_refine", self.ace_refine_var.get(),
                "--ace_seeds", str(self.ace_seeds_var.get()),
            ])
            if self.ace_calib_var.get():
                cmd.append("--ace_refine_calib")
            else:
                cmd.append("--no_ace_refine_calib")
            if self.ace_dense_var.get():
                cmd.append("--ace_dense_pc")
        elif align_engine == "pycusfm":
            cmd.extend([
                "--cusfm_features", self.cusfm_feat_var.get(),
                "--cusfm_matcher", self.cusfm_matcher_var.get(),
                "--cusfm_init", self.cusfm_init_var.get(),
                "--cusfm_ba_iter", str(self.cusfm_ba_var.get()),
            ])
            if self.cusfm_rgb_var.get():
                cmd.append("--cusfm_rgb")
            else:
                cmd.append("--no_cusfm_rgb")

        # Active 3DGS Training Engine Options (logged in report, executed if auto-train enabled)
        train_engine = self.train_engine_var.get()
        cmd.extend([
            "--train_engine", train_engine,
            "--train_iterations", str(self._parse_train_iterations()),
        ])

        if train_engine == "lichtfeld":
            cmd.extend([
                "--train_mode", self.lf_mode_var.get(),
                "--sh_degree", str(self.lf_sh_var.get()),
            ])
            if self.lf_gut_var.get():
                cmd.append("--gut")
            if self.lf_format_var.get() == "spz":
                cmd.append("--export_spz")
        elif train_engine == "splatfacto":
            cmd.extend([
                "--ns_model", self.ns_model_var.get(),
                "--camera_optimizer", self.ns_cam_opt_var.get(),
                "--sh_degree", str(self.ns_sh_var.get()),
            ])
            if self.ns_bilateral_var.get():
                cmd.append("--bilateral_grid")
            if self.ns_mip_var.get():
                cmd.append("--enable_mip")
        elif train_engine == "3dgrut":
            cmd.extend([
                "--representation", self.grut_repr_var.get(),
                "--bvh_backend", self.grut_bvh_var.get(),
                "--sh_degree", str(self.grut_sh_var.get()),
            ])
            if self.grut_exposure_var.get():
                cmd.append("--exposure_correction")
            if self.grut_sparsity_var.get():
                cmd.append("--enable_sparsity")

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
        self.is_paused = False
        self.start_btn.config(state=tk.DISABLED)
        self.one_click_btn.config(state=tk.DISABLED)
        self.train_only_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="⏸  PAUSE", bg="#f59e0b")
        self.stop_btn.config(state=tk.NORMAL)
        self.console.delete(1.0, tk.END)

        self._set_stage(5)
        self.progress_bar["value"] = 0
        self.pct_lbl.config(text="  0.0%")
        self.telemetry_lbl.config(text="Status: Initializing 3DGS Training...", fg=self.accent_blue)
        self.eta_lbl.config(text="")

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = self._get_logs_dir()
        self.active_log_path = os.path.join(logs_dir, f"abu_train_{timestamp}.log")
        self.log_status_lbl.config(
            text=f"Log: abu_train_{timestamp}.log",
            fg=self.accent_green
        )

        train_engine = self.train_engine_var.get()
        cmd = [
            self.conda_python,
            "-u",
            self.train_script,
            "--dataset", dataset_path,
            "--engine", train_engine,
            "--iterations", str(self._parse_train_iterations()),
        ]

        if train_engine == "lichtfeld":
            cmd.extend([
                "--mode", self.lf_mode_var.get(),
                "--sh_degree", str(self.lf_sh_var.get()),
            ])
            if self.lf_gut_var.get():
                cmd.append("--gut")
            if self.lf_format_var.get() == "spz":
                cmd.append("--export_spz")
        elif train_engine == "splatfacto":
            cmd.extend([
                "--ns_model", self.ns_model_var.get(),
                "--camera_optimizer", self.ns_cam_opt_var.get(),
                "--sh_degree", str(self.ns_sh_var.get()),
            ])
            if self.ns_bilateral_var.get():
                cmd.append("--bilateral_grid")
            if self.ns_mip_var.get():
                cmd.append("--enable_mip")
        elif train_engine == "3dgrut":
            cmd.extend([
                "--representation", self.grut_repr_var.get(),
                "--bvh_backend", self.grut_bvh_var.get(),
                "--sh_degree", str(self.grut_sh_var.get()),
            ])
            if self.grut_exposure_var.get():
                cmd.append("--exposure_correction")
            if self.grut_sparsity_var.get():
                cmd.append("--enable_sparsity")

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
                lf.write("Abu Align & Train v0.0.1.7 - Run Telemetry & Diagnostic Log\n")
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

                last_hw_log = time.time()
                for line in iter(self.worker_process.stdout.readline, ""):
                    if line:
                        now = time.time()
                        if now - last_hw_log >= 15.0:
                            last_hw_log = now
                            try:
                                hw = self.hw_monitor.query()
                                hw_str = f"[TELEMETRY:HARDWARE] CPU: {hw['cpu_pct']:.0f}% | RAM: {hw['ram_used_gb']:.1f}/{hw['ram_total_gb']:.0f}GB | GPU: {hw['gpu_util']}% ({hw['gpu_temp']}°C) | VRAM: {hw['vram_used_gb']:.1f}/{hw['vram_total_gb']:.0f}GB\n"
                                lf.write(hw_str)
                                self.output_queue.put(hw_str)
                            except Exception:
                                pass

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
            self.output_queue.put(f"__PROCESS_FINISHED__:{exit_code if 'exit_code' in locals() else 0}")

    def _check_queue(self):
        try:
            while True:
                msg = self.output_queue.get_nowait()
                if msg.startswith("__PROCESS_FINISHED__"):
                    parts = msg.split(":")
                    code = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
                    self.start_btn.config(state=tk.NORMAL)
                    self.one_click_btn.config(state=tk.NORMAL)
                    self.train_only_btn.config(state=tk.NORMAL)
                    self.pause_btn.config(state=tk.DISABLED, text="⏸  PAUSE", bg="#f59e0b")
                    self.stop_btn.config(state=tk.DISABLED)
                    self.is_paused = False
                    if code == 0:
                        self.progress_bar["value"] = 100
                        self.pct_lbl.config(text="100.0%", fg=self.accent_green)
                        self._set_stage(6)
                        self.telemetry_lbl.config(text="Status: Pipeline Completed Successfully (100%)", fg=self.accent_green)
                        self.eta_lbl.config(text="Done")
                        self._log("\n[STATUS] Pipeline Job Completed.\n")
                    else:
                        self.pct_lbl.config(text="Failed", fg="#ff4d4d")
                        self.telemetry_lbl.config(text=f"Status: Alignment Failed (Error code {code}) - See log", fg="#ff4d4d")
                        self.eta_lbl.config(text="Error")
                        self._log(f"\n[CRITICAL ERROR] Alignment exited with error code {code}. See traceback above.\n")
                    self.log_status_lbl.config(
                        text="Log: Saved (Abu_Align\\logs\\)",
                        fg=self.text_secondary
                    )
                else:
                    self._log(msg)
                    msg_s = msg.strip()
                    if any(k in msg for k in ["Stage 1", "Found", "Scanning", "images found"]):
                        self._set_stage(0)
                    elif any(k in msg for k in ["Stage 2", "Loading model", "Initializing GPU"]):
                        self._set_stage(1)
                    elif msg_s.startswith("[TELEMETRY:STAGE3]") or "Stage 3" in msg:
                        self._set_stage(2)
                        try:
                            parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                            pct = float(parts.get("pct", 0))
                            self.progress_bar["value"] = pct
                            self.pct_lbl.config(text=f"{pct:5.1f}%")
                            if parts.get("status") == "cache_hit":
                                self.telemetry_lbl.config(text="⚡ [Smart Cache Hit] Neural Poses & 3D Points Loaded (Stage 3 & 4 Skipped)", fg="#60a5fa")
                                self.eta_lbl.config(text="Saved ~5-10 min")
                            else:
                                b = parts.get("batch", "?")
                                tot = parts.get("total", "?")
                                self.telemetry_lbl.config(text=f"Stage 3/5: Neural Batch Inference [{b}/{tot}] ({pct:.1f}%)", fg=self.accent_blue)
                        except Exception:
                            pass
                    elif any(k in msg for k in ["Stage 4", "Generating 3D points", "Predicting dense point"]):
                        self._set_stage(3)
                    elif msg_s.startswith("[TELEMETRY:STAGE5]") or "Stage 5" in msg:
                        self._set_stage(4)
                        try:
                            parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                            pct = float(parts.get("pct", 0))
                            self.progress_bar["value"] = pct
                            self.pct_lbl.config(text=f"{pct:5.1f}%")
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
                    elif any(k in msg for k in ["[LichtFeld]", "[Nerfstudio]", "[3DGRUT]", "Stage 6", "Starting 3DGS"]):
                        self._set_stage(5)
                        if any(k in msg.lower() for k in ["step", "iter", "export", "loss"]):
                            self.telemetry_lbl.config(text="Stage 6: 3DGS Training Active", fg=self.accent_cyan)
        except queue.Empty:
            pass
        self.root.after(80, self._check_queue)

    def _update_hardware_telemetry(self):
        try:
            data = self.hw_monitor.query()
            self.hw_cpu_lbl.config(text=f"CPU: {data['cpu_pct']:.0f}%")
            self.hw_ram_lbl.config(text=f"RAM: {data['ram_used_gb']:.1f}/{data['ram_total_gb']:.0f} GB ({data['ram_pct']:.0f}%)")
            self.hw_gpu_lbl.config(text=f"GPU: {data['gpu_util']}%")
            if data["gpu_temp"] > 0:
                self.hw_temp_lbl.config(text=f"GPU Temp: {data['gpu_temp']}°C")
                if data["gpu_temp"] >= 80:
                    self.hw_temp_lbl.config(fg="#ef4444")
                elif data["gpu_temp"] >= 70:
                    self.hw_temp_lbl.config(fg="#f59e0b")
                else:
                    self.hw_temp_lbl.config(fg="#fbbf24")
            else:
                self.hw_temp_lbl.config(text="GPU Temp: --")

            self.hw_vram_lbl.config(text=f"VRAM: {data['vram_used_gb']:.1f}/{data['vram_total_gb']:.0f} GB ({data['vram_pct']:.0f}%)")
            if data["paging_risk"]:
                self.hw_page_lbl.config(text="⚠️ VRAM SPILL (PCIe Paging)", fg="#ffffff", bg="#dc2626")
            else:
                self.hw_page_lbl.config(text="Paging: OK", fg="#10b981", bg="#1e293b")
        except Exception:
            pass

        # Schedule next update in 1 second
        self.root.after(1000, self._update_hardware_telemetry)

    def _toggle_pause(self):
        if not self.worker_process or not self.is_running:
            return

        try:
            if psutil is not None:
                parent = psutil.Process(self.worker_process.pid)
                children = parent.children(recursive=True)
                all_procs = [parent] + children

                if not self.is_paused:
                    for p in all_procs:
                        try:
                            p.suspend()
                        except Exception:
                            pass
                    self.is_paused = True
                    self.pause_btn.config(text="▶  RESUME", bg=self.accent_green)
                    self.telemetry_lbl.config(text="Status: Paused (Process tree suspended)", fg="#f59e0b")
                    self._log("\n[PAUSE] Alignment/Training process suspended by user.\n")
                else:
                    for p in all_procs:
                        try:
                            p.resume()
                        except Exception:
                            pass
                    self.is_paused = False
                    self.pause_btn.config(text="⏸  PAUSE", bg="#f59e0b")
                    self.telemetry_lbl.config(text="Status: Resumed", fg=self.accent_blue)
                    self._log("\n[RESUME] Alignment/Training process resumed by user.\n")
            else:
                self._log("\n[WARNING] psutil is required for pause/resume.\n")
        except Exception as e:
            self._log(f"\n[ERROR] Failed to toggle pause: {e}\n")

    def _stop_alignment(self):
        if self.worker_process and self.is_running:
            try:
                if psutil is not None:
                    proc = psutil.Process(self.worker_process.pid)
                    if self.is_paused:
                        try:
                            proc.resume()
                            for c in proc.children(recursive=True):
                                c.resume()
                        except Exception:
                            pass
                    for c in proc.children(recursive=True):
                        try:
                            c.terminate()
                        except Exception:
                            pass
                    proc.terminate()
                else:
                    self.worker_process.terminate()
                self._log("\n[ABORT] User cancelled process.\n")
            except Exception as e:
                self._log(f"\n[ABORT] Error terminating process: {e}\n")
            self.is_running = False
            self.is_paused = False
            self.start_btn.config(state=tk.NORMAL)
            self.one_click_btn.config(state=tk.NORMAL)
            self.train_only_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED, text="⏸  PAUSE", bg="#f59e0b")
            self.stop_btn.config(state=tk.DISABLED)
            self.telemetry_lbl.config(text="Status: Aborted by user", fg=self.accent_red)

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
        doc_win.title(f"Abu Align v0.0.1.7 - {title}")
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
            "Abu Align & Train - Version 0.0.1.7\n"
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
