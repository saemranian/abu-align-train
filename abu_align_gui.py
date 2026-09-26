"""
Abu Align & Train Version 0.0.8.5
Professional Graphical User Interface for Neural Photogrammetry & 3DGS Training
Built on VGGT-Omega, Niantic ACE0, NVIDIA PyCuSFM, Mapillary OpenSfM, 360DVO with LichtFeld Studio, Nerfstudio Splatfacto, NVIDIA 3DGRUT, and PFGS360
"""

import os
import sys
import time
import glob
import subprocess
import threading
import queue
import json
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
try:
    import psutil
except ImportError:
    psutil = None
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from tkinter.scrolledtext import ScrolledText
import webbrowser

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
        self.widget.bind("<FocusIn>", self._on_enter, add="+")
        self.widget.bind("<FocusOut>", self._on_leave, add="+")
        self.widget.bind("<Escape>", self._on_leave, add="+")
        self.widget.bind("<Destroy>", lambda e: self._on_leave() if e.widget is self.widget else None, add="+")

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
            # A newly opened dialog may cover a control before its hover timer fires.
            pointer = self.widget.winfo_containing(*self.widget.winfo_pointerxy())
            if pointer is not self.widget and self.widget.focus_get() is not self.widget:
                return
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
                bg="#242d32",
                highlightbackground="#70dbc0",
                highlightcolor="#70dbc0",
                highlightthickness=1,
                padx=9,
                pady=6
            )
            frame.pack()

            lbl = tk.Label(
                frame,
                text=self.text,
                justify=tk.LEFT,
                bg="#242d32",
                fg="#e7edef",
                font=("Segoe UI", 9),
                wraplength=340
            )
            lbl.pack()
            tw.update_idletasks()
            y = min(y, max(0, self.widget.winfo_screenheight() - tw.winfo_reqheight() - 12))
            tw.wm_geometry(f"+{x}+{y}")
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
        self.root.title("Abu Align & Train - Version 0.0.8.5 (Neural Photogrammetry & 3DGS Training Suite)")
        width = min(1440, self.root.winfo_screenwidth() - 80)
        height = min(1050, self.root.winfo_screenheight() - 120)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(960, 720)

        # Style configuration (Modern Dark Palette)
        self.bg_dark = "#111518"
        self.panel_bg = "#1a2024"
        self.panel_border = "#344149"
        self.accent_blue = "#70dbc0"
        self.accent_green = "#10b981"
        self.accent_red = "#ef4444"
        self.accent_purple = "#8b5cf6"
        self.accent_cyan = "#0284c7"
        self.text_primary = "#e7edef"
        self.text_secondary = "#a0adb5"
        self.console_bg = "#111518"

        self.root.configure(bg=self.bg_dark)
        self._set_window_icon()

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
        self.latest_alignment_dir = None
        self.latest_train_dir = None
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
        self.vggt_preprocess_var = tk.StringVar(value="max_size")

        # VGG-T3 (VGG-TTT) Specific Options
        self.vggttt_conf_var = tk.StringVar(value="Adaptive (20% Quantile)")
        self.vggttt_prep_var = tk.StringVar(value="Auto (Smart Center-Crop)")
        self.vggttt_mem_var = tk.BooleanVar(value=True)
        self.vggttt_steps_var = tk.StringVar(value="2 (Recommended)")
        self.vggttt_pred_var = tk.BooleanVar(value=True)

        # ACE0 Specific Options
        self.ace_iter_var = tk.StringVar(value="30")
        self.ace_refine_var = tk.StringVar(value="mlp")
        self.ace_seeds_var = tk.StringVar(value="3")
        self.ace_calib_var = tk.BooleanVar(value=True)
        self.ace_dense_var = tk.BooleanVar(value=False)
        self.ace_refit_var = tk.BooleanVar(value=True)
        self.ace_workers_var = tk.StringVar(value="2")
        self.ace_stage_res_var = tk.StringVar(value="Original Native (Full Res)")

        # PyCuSFM Specific Options
        self.cusfm_feat_var = tk.StringVar(value="aliked")
        self.cusfm_matcher_var = tk.StringVar(value="lightglue")
        self.cusfm_init_var = tk.StringVar(value="omega")
        self.cusfm_ba_var = tk.StringVar(value="100")
        self.cusfm_rgb_var = tk.BooleanVar(value=True)

        # OpenSfM 360 & Spherical Specific Options
        self.opensfm_proj_var = tk.StringVar(value="equirectangular")
        self.opensfm_feat_var = tk.StringVar(value="sift")
        self.opensfm_transcode_var = tk.BooleanVar(value=False)
        self.opensfm_matcher_var = tk.StringVar(value="words")
        self.opensfm_budget_var = tk.StringVar(value="3000")
        self.opensfm_neighbors_var = tk.StringVar(value="5")
        self.opensfm_bundle_var = tk.StringVar(value="10")
        self.opensfm_opt_cam_var = tk.BooleanVar(value=False)
        self.rig_split_mode_var = tk.StringVar(value="ring")
        self.rig_num_cams_var = tk.StringVar(value="8")
        self.rig_fov_var = tk.StringVar(value="90")
        self.rig_res_var = tk.StringVar(value="1024")

        # 360DVO Deep Visual Odometry Specific Options
        self.dvo_patches_var = tk.StringVar(value="96")
        self.dvo_polar_mask_var = tk.StringVar(value="8%")
        self.dvo_ba_window_var = tk.StringVar(value="8")
        self.dvo_scale_var = tk.StringVar(value="3.2")
        self.dvo_viz_var = tk.BooleanVar(value=True)
        self.dvo_weights_var = tk.StringVar(value="")
        self.dvo_dense_cloud_var = tk.BooleanVar(value=False)
        self.dvo_patch_lifetime_var = tk.StringVar(value="13 (Default)")
        self.dvo_final_refine_var = tk.StringVar(value="12 (Standard)")
        self.dvo_gradient_bias_var = tk.BooleanVar(value=True)

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
        self.train_target_run_var = tk.StringVar(value="Latest Run (Auto-Detect)")
        self.train_res_var = tk.StringVar(value="Original Native (100% Full Res)")
        self.train_max_gaussians_var = tk.StringVar(value="Unlimited (Engine Default)")

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

        # PFGS360 (CVPR 2026) Specific Options
        self.pfgs_depth_var = tk.StringVar(value="UniK3D")
        self.pfgs_dataparser_var = tk.StringVar(value="auto")
        self.pfgs_match_views_var = tk.StringVar(value="8 (Sliding Window - Fast & Smooth)")
        self.pfgs_densify_views_var = tk.StringVar(value="8")
        self.pfgs_spherical_var = tk.BooleanVar(value=True)
        self.pfgs_consist_var = tk.BooleanVar(value=True)
        self.pfgs_inlier_var = tk.BooleanVar(value=True)
        self.pfgs_align_depth_var = tk.BooleanVar(value=True)
        self.pfgs_filter_sky_var = tk.BooleanVar(value=True)
        self.pfgs_sky_depth_var = tk.StringVar(value="50.0")
        self.pfgs_sh_var = tk.StringVar(value="3")
        self.pfgs_downscale_var = tk.StringVar(value="1 (Full 100% Quality)")

        # Capture configuration only, before legacy aliases and UI state exist.
        self._preset_vars = {
            name.removesuffix("_var"): value for name, value in vars(self).items()
            if isinstance(value, tk.Variable)
        }
        self.presets_dir = Path(__file__).resolve().parent / "presets"
        self.preset_var = tk.StringVar(value="")

        # Legacy aliases for backward compatibility
        self.train_mode_var = self.lf_mode_var
        self.gut_var = self.lf_gut_var
        self.bilateral_var = self.ns_bilateral_var
        self.exposure_var = self.grut_exposure_var
        self.sparsity_var = self.grut_sparsity_var
        self.mip_var = self.ns_mip_var

        self._build_menu()
        self._build_ui()
        self.dataset_var.trace_add("write", self._validate_dataset)
        self._validate_dataset()
        self._check_queue()
        self.root.after(500, self._update_hardware_telemetry)

        # Background GitHub Upstream Engine Version Tracker
        self.engine_tracker_data = None
        self._checking_engine_updates = False
        self.root.after(800, self._start_background_engine_check)

    def _preset_path(self, name):
        name = name.strip()
        if (not name or len(name) > 100 or name.endswith(".")
                or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
                or name.split(".")[0].upper() in
                {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)],
                 *[f"LPT{i}" for i in range(1, 10)]}):
            raise ValueError("Use a name of 1–100 characters without Windows filename symbols or reserved names.")
        path = self.presets_dir / (name + ".json")
        if path.resolve().parent != self.presets_dir.resolve():
            raise ValueError("Presets must be stored inside the presets folder.")
        return path

    def _refresh_presets(self):
        try:
            names = sorted((p.stem for p in self.presets_dir.glob("*.json") if p.is_file()),
                           key=str.casefold)
            self.preset_combo.configure(values=names)
        except OSError as exc:
            messagebox.showerror("Presets unavailable", str(exc), parent=self.root)

    def _validate_preset(self, profile):
        if (not isinstance(profile, dict) or type(profile.get("schema_version")) is not int
                or profile["schema_version"] != 1):
            raise ValueError("Unsupported preset format. Expected schema version 1.")
        settings = profile.get("settings")
        if not isinstance(settings, dict) or not settings:
            raise ValueError("This preset has missing settings.")
        for k in set(self._preset_vars) - set(settings):
            settings[k] = self._preset_vars[k].get()
        if set(settings) != set(self._preset_vars):
            raise ValueError("This preset has unknown configuration fields.")
        # Derive allowed choices from the actual controls to avoid a second option list.
        choices = {}
        def collect(widget):
            if isinstance(widget, ttk.Combobox) and str(widget.cget("state")) == "readonly":
                choices[str(widget.cget("textvariable"))] = set(widget.cget("values"))
            elif isinstance(widget, (tk.Radiobutton, ttk.Radiobutton)):
                choices.setdefault(str(widget.cget("variable")), set()).add(str(widget.cget("value")))
            for child in widget.winfo_children():
                collect(child)
        collect(self.root)
        for key, var in self._preset_vars.items():
            value = settings[key]
            expected = bool if isinstance(var, tk.BooleanVar) else str
            if type(value) is not expected:
                raise ValueError(f"Invalid type for {key}: expected {expected.__name__}.")
            if str(var) in choices and value not in choices[str(var)]:
                raise ValueError(f"Unsupported value for {key}: {value!r}.")
        return settings

    def _save_preset(self):
        name = simpledialog.askstring("Save preset", "Profile name:",
                                      initialvalue=self.preset_var.get(), parent=self.root)
        if name is None:
            return
        temporary = None
        try:
            path = self._preset_path(name)
            profile = {"schema_version": 1,
                       "settings": {key: var.get() for key, var in self._preset_vars.items()}}
            self._validate_preset(profile)
            if path.exists() and not messagebox.askyesno(
                    "Replace preset?", f"Replace '{path.stem}'?", parent=self.root):
                return
            self.presets_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.presets_dir,
                                             suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                json.dump(profile, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
            temporary = None
            self._refresh_presets()
            self.preset_var.set(path.stem)
            self._log(f"[Preset] Saved '{path.stem}'.\n")
        except (OSError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Could not save preset", str(exc), parent=self.root)
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _load_preset(self):
        if self.is_running:
            messagebox.showinfo("Pipeline running", "Stop the current run before loading a preset.",
                                parent=self.root)
            return
        try:
            path = self._preset_path(self.preset_var.get())
            with path.open(encoding="utf-8") as stream:
                settings = self._validate_preset(json.load(stream))
            # Validate the entire document before changing any control.
            for key, value in settings.items():
                self._preset_vars[key].set(value)
            self._update_align_panel()
            self._update_train_panel()
            self._log(f"[Preset] Loaded '{path.stem}'. Dataset selection preserved.\n")
        except (OSError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Could not load preset", str(exc), parent=self.root)

    def _set_window_icon(self):
        """Sets the application titlebar and taskbar icon to custom Abu 'A' icon."""
        try:
            ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abu_icon.ico")
            if os.path.isfile(ico_path):
                try:
                    self.root.iconbitmap(default=ico_path)
                except Exception:
                    try:
                        self.root.iconbitmap(ico_path)
                    except Exception:
                        pass
            png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abu_icon.png")
            if os.path.isfile(png_path):
                try:
                    from PIL import ImageTk
                    self._app_icon_photo = ImageTk.PhotoImage(file=png_path)
                    self.root.iconphoto(True, self._app_icon_photo)
                except Exception:
                    pass
        except Exception:
            pass

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=self.panel_bg, fg=self.text_primary)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.panel_bg, fg=self.text_primary)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Dataset Folder...", command=self._browse_dataset)
        file_menu.add_command(label="Save configuration preset...", accelerator="Ctrl+Shift+S", command=self._save_preset)
        file_menu.add_command(label="Open Video File...", command=self._browse_video)
        file_menu.add_command(label="Open Results Folder", command=self._open_results_folder)
        file_menu.add_command(label="📁 Open 3DGS Models Folder", command=self._open_models_folder)
        file_menu.add_command(label="📋 View Reconstruction Report...", command=self._view_training_run_report)
        file_menu.add_separator()
        file_menu.add_command(label="📁 Open Logs Folder", command=self._open_logs_folder)
        file_menu.add_command(label="📄 View Latest Log File", command=self._open_latest_log)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=self.panel_bg, fg=self.text_primary)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="🎬 Video to Frame Sequence Extractor...", command=self._open_video_extractor_dialog)
        tools_menu.add_command(label="🌐 360° Multi-Camera Rig Projector / Splitter...", command=self._open_panoramic_splitter_dialog)
        tools_menu.add_command(label="📐 Open Native 3D Workspace", accelerator="Alt+4", command=self._show_3d_workspace)
        tools_menu.add_command(label="📋 View Reconstruction Report...", command=self._view_training_run_report)
        tools_menu.add_command(label="🌐 Launch 3D Browser Viewer", command=self._launch_viewer)
        tools_menu.add_separator()
        tools_menu.add_command(label="🔄 Check Engine Repository Updates...", command=self._open_engine_updates_dialog)

        # Engines Menu (Repository Tracker & Upstream Docs)
        engines_menu = tk.Menu(menubar, tearoff=0, bg=self.panel_bg, fg=self.text_primary)
        menubar.add_cascade(label="Engines", menu=engines_menu)
        engines_menu.add_command(label="🔄 Check GitHub Updates for All Engines...", command=self._open_engine_updates_dialog)
        engines_menu.add_separator()
        engines_menu.add_command(label="⚡ VGGT-Omega (Facebook Research)", command=lambda: self._open_engine_link("vggt_omega"))
        engines_menu.add_command(label="🚀 NVIDIA VGG-T³ (CVPR 2026 Linear SfM)", command=lambda: self._open_engine_link("vggttt"))
        engines_menu.add_command(label="🎯 Niantic Labs ACE0 (ECCV 2024 Oral)", command=lambda: self._open_engine_link("ace0"))
        engines_menu.add_command(label="✨ NVIDIA 3DGRUT (Ray Traced GS)", command=lambda: self._open_engine_link("3dgrut"))
        engines_menu.add_command(label="🌐 PFGS360 (CVPR 2026 Pose-Free)", command=lambda: self._open_engine_link("pfgs360"))
        engines_menu.add_command(label="⚡ NVIDIA Isaac pyCuSFM", command=lambda: self._open_engine_link("pycusfm"))
        engines_menu.add_command(label="🌍 Mapillary OpenSfM", command=lambda: self._open_engine_link("opensfm"))
        engines_menu.add_command(label="🔄 360DVO (IEEE RA-L 2026)", command=lambda: self._open_engine_link("360dvo"))
        engines_menu.add_command(label="💡 LichtFeld Studio (Real-Time GUT)", command=lambda: self._open_engine_link("lichtfeld"))
        engines_menu.add_command(label="🎓 Nerfstudio Splatfacto", command=lambda: self._open_engine_link("nerfstudio"))
        engines_menu.add_command(label="🔮 VGGT (Original Architecture)", command=lambda: self._open_engine_link("vggt_base"))

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.panel_bg, fg=self.text_primary)
        menubar.add_cascade(label="Help & Documentation", menu=help_menu)
        help_menu.add_command(label="📖 User Guide & Fundamentals Manual", command=self._open_user_guide)
        help_menu.add_command(label="📐 AI vs Traditional Architecture Whitepaper", command=self._open_architecture_doc)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ About Abu Align & Train", command=self._show_about_box)

    def _build_ui(self):
        from desktop_ui import build_workspace
        build_workspace(self, ToolTip)

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
        rb_a1 = tk.Radiobutton(v_col2, text="Feedforward Draft (~15s, Low VRAM)", variable=self.align_var, value="none", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a1.pack(anchor="w")
        rb_a2 = tk.Radiobutton(v_col2, text="Fast Ceres BA (~30s)", variable=self.align_var, value="fast", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a2.pack(anchor="w")
        rb_a3 = tk.Radiobutton(v_col2, text="Standard Ceres BA (~1m)", variable=self.align_var, value="standard", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a3.pack(anchor="w")
        rb_a4 = tk.Radiobutton(v_col2, text="Ultra Sub-Pixel BA (Cauchy+Polish)", variable=self.align_var, value="ultra", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_a4.pack(anchor="w")
        ToolTip(rb_a1, "Feedforward Draft (~15s):\nPure deep-learning prediction directly from VGGT-Omega foundation model forward pass.\nZero CoTracker tracking, zero Ceres BA. Lowest possible VRAM consumption.\nRecommended for large datasets (e.g. 2,500+ photos).")
        ToolTip(rb_a2, "Fast Ceres BA (~30s):\nLightweight 3-query Ceres Levenberg-Marquardt bundle adjustment (25 iterations, pinhole).\nQuick refinement of focal length and camera trajectory without heavy multi-pass overhead.")
        ToolTip(rb_a3, "Standard Ceres BA (~1m):\nALIKED neural feature tracks refined by Ceres Levenberg-Marquardt optimizer.")
        ToolTip(rb_a4, "Ultra Sub-Pixel BA (Cauchy + Polish) [Recommended]:\nSolves 3,072 multi-view tracks with Cauchy robust loss to reject outliers,\nfollowed by outlier pruning and a 2nd Polish pass reaching 0.35 px error.")

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

        # Col D: Aspect Ratio Preprocessing
        v_col4 = tk.Frame(self.panel_align_vggt, bg=self.panel_bg)
        v_col4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(v_col4, text="Aspect Preprocess Mode:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_p1 = tk.Radiobutton(v_col4, text="Max-Size (Lowest VRAM, 4K)", variable=self.vggt_preprocess_var, value="max_size", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_p1.pack(anchor="w")
        rb_p2 = tk.Radiobutton(v_col4, text="Balanced (Native Aspect)", variable=self.vggt_preprocess_var, value="balanced", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_p2.pack(anchor="w")
        rb_p3 = tk.Radiobutton(v_col4, text="Square Pad (Legacy)", variable=self.vggt_preprocess_var, value="square", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_p3.pack(anchor="w")
        ToolTip(rb_p1, "Max-Size (Lowest VRAM, 4K Friendly) [Recommended]:\nOfficial VGGT-Omega aspect-ratio preserving preprocessing.\nDownscales longest dimension without black letterbox padding.\nSaves 50% attention tokens on 16:9 photos (512 vs 1024 tokens) and runs ~2.5x faster.")
        ToolTip(rb_p2, "Balanced (Native Aspect):\nOfficial VGGT-Omega balanced resolution scaling maintaining native photo aspect ratio.")
        ToolTip(rb_p3, "Square Pad (Legacy):\nLegacy square letterbox padding (512x512). Pads non-square photos with black bars.")

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
        seeds_cb = ttk.Combobox(seeds_row, textvariable=self.ace_seeds_var, values=["1", "3", "5"], state="readonly", width=3)
        seeds_cb.pack(side=tk.LEFT, padx=(2, 6))
        ToolTip(seeds_cb, "Initialization Seeds:\nNumber of candidate seeds evaluated during mapping.\n• 3 Seeds [Recommended Default]: Robust seed selection across scan with fast early exit when a good seed (>75%) is found.\n• 5 Seeds [Official Niantic Default]: Benchmark standard for complex multi-room or unconstrained datasets.\n• 1 Seed: Fast SfM draft mode.")

        tk.Label(seeds_row, text="Workers:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        workers_cb = ttk.Combobox(seeds_row, textvariable=self.ace_workers_var, values=["1", "2", "3"], state="readonly", width=3)
        workers_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(workers_cb, "Parallel Seed Workers:\nNumber of parallel workers evaluating candidate seeds concurrently.\nOn RTX 4090 (24GB VRAM), 2-3 workers run simultaneously, cutting seed search time by 2-3x.")

        stage_res_row = tk.Frame(ace_col3, bg=self.panel_bg)
        stage_res_row.pack(anchor="w", pady=(1, 3))
        tk.Label(stage_res_row, text="Staging Res:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        stage_res_cb = ttk.Combobox(stage_res_row, textvariable=self.ace_stage_res_var, values=["Original Native (Full Res)", "2K Downscale (1920px Max)", "1K Downscale (1024px Max)", "4K Downscale (3840px Max)"], state="readonly", width=22)
        stage_res_cb.pack(side=tk.LEFT, padx=(2, 4))
        ToolTip(stage_res_cb, "ACE0 Staging Resolution:\n• Original Native (Full Res) [Default]: Zero downscaling. Preserves 100% full-resolution images and true focal lengths.\n• 2K Downscale (1920px Max): Faster feature regression for ultra-high-resolution datasets.\n• 1K Downscale (1024px Max): Maximum speed and lowest VRAM staging.")

        ace_calib_chk = tk.Checkbutton(ace_col3, text="Refine Focal Length Calib", variable=self.ace_calib_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ace_calib_chk.pack(anchor="w")
        ToolTip(ace_calib_chk, "Refine Focal Length Calibration:\nOptimizes camera focal length during mapping rounds.\nCrucial when image EXIF focal length is missing or uncalibrated.")

        ace_dense_chk = tk.Checkbutton(ace_col3, text="Unfiltered Dense Cloud", variable=self.ace_dense_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ace_dense_chk.pack(anchor="w")
        ToolTip(ace_dense_chk, "Unfiltered Dense Cloud (Noisy):\n• Unchecked [Recommended Default]: Strict 1-3px reprojection error and surface gradient filtering. Produces clean, crisp surface points.\n• Checked: Bypasses reprojection filtering. Recommended ONLY for ultra-dense scans (>2000 photos). On standard scans, produces noisy floating points.")

        ace_refit_chk = tk.Checkbutton(ace_col3, text="High-Precision 25k Refit Mode", variable=self.ace_refit_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        ace_refit_chk.pack(anchor="w")
        ToolTip(ace_refit_chk, "High-Precision 25k Refit Mode [Official Niantic Benchmark Standard]:\n• Checked [Recommended Default]: Trains a fresh 25,000-iteration MLP on 100% of registered images with camera poses frozen for the first 5,000 steps. Eliminates warped ZoeDepth depth artifacts and produces dense, pinpoint 3D scene point clouds matching published benchmarks.\n• Unchecked: Rapid SfM draft mode (stops at iteration 1).")

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
        rb_cu_i2 = tk.Radiobutton(cu_col2, text="OpenSfM Prior (reconstruction.json)", variable=self.cusfm_init_var, value="opensfm", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cu_i2.pack(anchor="w")
        rb_cu_i3 = tk.Radiobutton(cu_col2, text="Auto-Detect (Prior / Cache)", variable=self.cusfm_init_var, value="auto", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cu_i3.pack(anchor="w")
        rb_cu_i4 = tk.Radiobutton(cu_col2, text="Visual Odometry (Ego / None)", variable=self.cusfm_init_var, value="ego", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_cu_i4.pack(anchor="w")
        ToolTip(rb_cu_i1, "VGGT-Omega Prior [Recommended]:\nSeeds PyCuSFM bundle adjustment with global foundation model poses for instant convergence and zero drift.")
        ToolTip(rb_cu_i2, "OpenSfM Prior (reconstruction.json):\nIngests camera poses and intrinsics computed by OpenSfM into Isaac cuSFM frames_meta.")
        ToolTip(rb_cu_i3, "Auto-Detect Prior:\nSearches for existing OpenSfM poses, transforms.json, or VGGT cache before falling back to odometry.")
        ToolTip(rb_cu_i4, "Visual Odometry (Ego):\nSequential pairwise epipolar chaining. Ideal for continuous video captures and smooth drone fly-throughs.")

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

        # =========================================================================
        # 4. Mapillary OpenSfM Subpanel (360° Spherical & Street SfM)
        # =========================================================================
        self.panel_align_opensfm = tk.Frame(self.align_dynamic_container, bg=self.panel_bg)

        # Col A: Projection / Camera Model
        osfm_col1 = tk.Frame(self.panel_align_opensfm, bg=self.panel_bg)
        osfm_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(osfm_col1, text="Camera Projection Model:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_op_p1 = tk.Radiobutton(osfm_col1, text="Equirectangular 360° (Spherical S²)", variable=self.opensfm_proj_var, value="equirectangular", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_op_p1.pack(anchor="w")
        rb_op_p2 = tk.Radiobutton(osfm_col1, text="Perspective Pinhole (Rectilinear)", variable=self.opensfm_proj_var, value="perspective", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_op_p2.pack(anchor="w")
        rb_op_p3 = tk.Radiobutton(osfm_col1, text="Fisheye (Ultra-Wide 180°+)", variable=self.opensfm_proj_var, value="fisheye", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_op_p3.pack(anchor="w")
        ToolTip(rb_op_p1, "Equirectangular 360°:\nNative spherical bundle adjustment on unit bearing vectors (S² manifold).\nOptimizes 360 full-sphere panoramas without polar distortion.")
        ToolTip(rb_op_p2, "Perspective Pinhole:\nStandard rectilinear pinhole model for DSLR, drone, and smartphone cameras.")
        ToolTip(rb_op_p3, "Fisheye Model:\nRadial equidistant lens projection handling field-of-view up to 220°.")

        chk_opt_cam = tk.Checkbutton(
            osfm_col1,
            text="Optimize Spherical Camera",
            variable=self.opensfm_opt_cam_var,
            font=("Segoe UI", 8),
            fg=self.text_primary,
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        chk_opt_cam.pack(anchor="w", pady=(3, 0))
        ToolTip(chk_opt_cam, "Optimize Camera Calibration (optimize_camera_parameters):\nRefines spherical camera parameters (aspect ratio, sensor alignment, and field of view) in Ceres bundle adjustment.\n• Unchecked (Recommended): Strict S² spherical manifold.\n• Checked: Calibrates and refines non-ideal equirectangular stitches.")

        # Col B: Features & Matcher Strategy
        osfm_col2 = tk.Frame(self.panel_align_opensfm, bg=self.panel_bg)
        osfm_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(osfm_col2, text="Features & Matcher Strategy:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        row_b1 = tk.Frame(osfm_col2, bg=self.panel_bg)
        row_b1.pack(anchor="w", pady=(1, 1))
        tk.Label(row_b1, text="Feature:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        osfm_feat_cb = ttk.Combobox(row_b1, textvariable=self.opensfm_feat_var, values=["sift", "hahog", "orb", "akaze"], state="readonly", width=7)
        osfm_feat_cb.pack(side=tk.LEFT, padx=(2, 6))
        ToolTip(osfm_feat_cb, "Feature Extractor (feature_type):\n• SIFT (Recommended): High-precision scale invariant features with sub-pixel localization.\n• HAHOG: Hessian-Affine + Histogram of Gradients (Mapillary street standard).\n• ORB: Fast binary feature extractor for rapid drafts.\n• AKAZE: Accelerated anisotropic diffusion features.")

        tk.Label(row_b1, text="Budget:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        osfm_budget_cb = ttk.Combobox(row_b1, textvariable=self.opensfm_budget_var, values=["1500", "2500", "3000", "4000", "5000"], state="readonly", width=5)
        osfm_budget_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(osfm_budget_cb, "Feature Budget per Frame (feature_min_frames):\nTarget number of features detected and tracked per view.\n• 1500: Fast tracking for clean high-contrast video.\n• 3000 (Default): Standard balanced photogrammetric budget.\n• 4000-5000: Ultra-dense tie points for complex urban/street scenes.")

        row_b2 = tk.Frame(osfm_col2, bg=self.panel_bg)
        row_b2.pack(anchor="w", pady=(1, 1))
        tk.Label(row_b2, text="Matcher:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        osfm_match_cb = ttk.Combobox(row_b2, textvariable=self.opensfm_matcher_var, values=["words", "flann", "exhaustive"], state="readonly", width=8)
        osfm_match_cb.pack(side=tk.LEFT, padx=(2, 6))
        ToolTip(osfm_match_cb, "Matcher Strategy (matcher_type):\n• words (Default): Visual Bag-of-Words vocabulary tree indexing.\n• flann: Fast approximate nearest neighbors with reciprocal ratio test.\n• exhaustive: Full brute-force pairwise matching across candidate windows.")

        tk.Label(row_b2, text="Window:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        osfm_neighbors_cb = ttk.Combobox(row_b2, textvariable=self.opensfm_neighbors_var, values=["3", "4", "5", "8", "12"], state="readonly", width=3)
        osfm_neighbors_cb.pack(side=tk.LEFT, padx=2)
        ToolTip(osfm_neighbors_cb, "Sequential Matching Window (matching_order_neighbors):\nNumber of adjacent consecutive frames matched in temporal order.\n• 3-4: Fast linear tracks for high frame-rate video.\n• 5 (Default): Recommended window for smooth camera movement.\n• 8-12: Robust loop-closure and track recovery for rapid turns.")

        row_b3 = tk.Frame(osfm_col2, bg=self.panel_bg)
        row_b3.pack(anchor="w", pady=(1, 1))
        tk.Label(row_b3, text="BA Interval:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        osfm_bundle_cb = ttk.Combobox(row_b3, textvariable=self.opensfm_bundle_var, values=["1", "2", "5", "10", "20"], state="readonly", width=3)
        osfm_bundle_cb.pack(side=tk.LEFT, padx=(2, 4))
        tk.Label(row_b3, text="view(s)", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        ToolTip(osfm_bundle_cb, "Bundle Adjustment Frequency (bundle_interval):\nHow often Ceres bundle adjustment is triggered during incremental reconstruction.\n• 10 (Recommended): Runs full Ceres BA every 10 frames (~2-3 mins total runtime).\n• 1: Full Ceres BA after every registered frame (slowest, 30+ mins).")

        # Col C: 360° Multi-Camera Rig Projector & 3DGS Bridge
        osfm_col3 = tk.Frame(self.panel_align_opensfm, bg=self.panel_bg)
        osfm_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(osfm_col3, text="360° Multi-Camera Rig Projector:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        tc_chk = tk.Checkbutton(osfm_col3, text="⚡ Auto-Project Multi-Camera Rig", variable=self.opensfm_transcode_var, font=("Segoe UI", 8, "bold"), fg="#a855f7", bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        tc_chk.pack(anchor="w", pady=(1, 2))
        ToolTip(tc_chk, "Auto-Project Multi-Camera Rig:\nConverts 360° spherical panoramas into virtual rectilinear pinhole arrays (e.g. 8-camera ring @ 50% overlap or 6-face cubemap).\nEnables immediate Gaussian Splatting training in LichtFeld Studio & Nerfstudio Splatfacto.")

        self.osfm_rig_subframe = tk.Frame(osfm_col3, bg=self.panel_bg)

        rig_row = tk.Frame(self.osfm_rig_subframe, bg=self.panel_bg)
        rig_row.pack(anchor="w", pady=(1, 2))
        tk.Label(rig_row, text="Rig:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        rig_cb = ttk.Combobox(rig_row, textvariable=self.rig_split_mode_var, values=["ring", "cubemap", "dome"], state="readonly", width=8)
        rig_cb.pack(side=tk.LEFT, padx=(2, 6))

        tk.Label(rig_row, text="Res:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        rig_res_cb = ttk.Combobox(rig_row, textvariable=self.rig_res_var, values=["512", "1024", "1536", "2048"], state="readonly", width=5)
        rig_res_cb.pack(side=tk.LEFT, padx=(2, 4))

        rig_dlg_btn = tk.Button(
            self.osfm_rig_subframe,
            text="🌐 Open Rig Projector Dialog...",
            font=("Segoe UI", 8, "bold"),
            bg="#7c3aed",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=2,
            command=self._open_panoramic_splitter_dialog
        )
        rig_dlg_btn.pack(anchor="w", pady=(2, 0))

        def _toggle_osfm_rig(*_):
            if self.opensfm_transcode_var.get():
                self.osfm_rig_subframe.pack(anchor="w", pady=(1, 2))
            else:
                self.osfm_rig_subframe.pack_forget()

        self.opensfm_transcode_var.trace_add("write", _toggle_osfm_rig)
        _toggle_osfm_rig()

        # =========================================================================
        # 5. 360DVO Subpanel (Deep Visual Odometry for 360° Cameras)
        # =========================================================================
        self.panel_align_360dvo = tk.Frame(self.align_dynamic_container, bg=self.panel_bg)

        # Col A: Spherical Patch Tracking & Density
        dvo_col1 = tk.Frame(self.panel_align_360dvo, bg=self.panel_bg)
        dvo_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(dvo_col1, text="Patch Graph & Density:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_dvo_p1 = tk.Radiobutton(dvo_col1, text="Fast Realtime (64 Patches)", variable=self.dvo_patches_var, value="64", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg, font=("Segoe UI", 8))
        rb_dvo_p1.pack(anchor="w")
        rb_dvo_p2 = tk.Radiobutton(dvo_col1, text="Standard Odometry (96 Patches)", variable=self.dvo_patches_var, value="96", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg, font=("Segoe UI", 8))
        rb_dvo_p2.pack(anchor="w")
        rb_dvo_p3 = tk.Radiobutton(dvo_col1, text="Dense High-Precision (128 Patches)", variable=self.dvo_patches_var, value="128", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg, font=("Segoe UI", 8))
        rb_dvo_p3.pack(anchor="w")
        rb_dvo_p4 = tk.Radiobutton(dvo_col1, text="Ultra Dense (256 Patches)", variable=self.dvo_patches_var, value="256", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg, font=("Segoe UI", 8))
        rb_dvo_p4.pack(anchor="w")
        rb_dvo_p5 = tk.Radiobutton(dvo_col1, text="Extreme Density (384 Patches)", variable=self.dvo_patches_var, value="384", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg, font=("Segoe UI", 8))
        rb_dvo_p5.pack(anchor="w")
        rb_dvo_p6 = tk.Radiobutton(dvo_col1, text="Maximum Density (512 Patches)", variable=self.dvo_patches_var, value="512", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg, font=("Segoe UI", 8))
        rb_dvo_p6.pack(anchor="w")
        ToolTip(rb_dvo_p1, "64 Patches: Fastest visual odometry loop (~30 FPS). Lightweight tracking for fast previews.")
        ToolTip(rb_dvo_p2, "96 Patches [Recommended]: Balanced trajectory accuracy and factor-graph optimization.")
        ToolTip(rb_dvo_p3, "128 Patches: Maximum tracking density for complex scenes with rapid rotations.")
        ToolTip(rb_dvo_p4, "256 Patches: Ultra-dense spherical patch graph for textureless environments or long loop trajectories.")
        ToolTip(rb_dvo_p5, "384 Patches: 50% more patch coverage than 256. Ideal for large outdoor panoramas, high-motion 360 videos, or environments with uneven texture.")
        ToolTip(rb_dvo_p6, "512 Patches: Maximum patch density supported by 360DVO architecture. Tracks up to 512 concurrent surface patches across the equirectangular sphere.")

        dvo_dense_chk = tk.Checkbutton(dvo_col1, text="Dense 3x3 Micro-Cloud (9x Points)", variable=self.dvo_dense_cloud_var, font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        dvo_dense_chk.pack(anchor="w", pady=(2, 0))
        ToolTip(dvo_dense_chk, "Dense 3x3 Micro-Cloud (9x Density Multiplier):\nIn the original 360DVO repo, each tracked patch is a 3x3 grid on the sphere, but 8 out of 9 points were discarded. When checked, unprojects all 9 grid points per patch to generate a 900% denser initial point cloud (up to 700k+ points) with RGB colors for instant high-fidelity 3DGS training initialization.")

        # Col B: Factor Graph & Quality Optimization
        dvo_col2 = tk.Frame(self.panel_align_360dvo, bg=self.panel_bg)
        dvo_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(dvo_col2, text="Quality & ODBA Optimization:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        dvo_mask_frame = tk.Frame(dvo_col2, bg=self.panel_bg)
        dvo_mask_frame.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_mask = tk.Label(dvo_mask_frame, text="Polar Mask:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_mask.pack(side=tk.LEFT)
        cb_mask = ttk.Combobox(dvo_mask_frame, textvariable=self.dvo_polar_mask_var, values=["0%", "5%", "8%", "12%", "15%"], state="readonly", width=8, font=("Segoe UI", 8))
        cb_mask.pack(side=tk.LEFT)
        ToolTip(lbl_mask, "Excludes top zenith and bottom nadir poles from patch sampling to eliminate tripod shadows and spherical distortion artifacts.")
        ToolTip(cb_mask, "Excludes top zenith and bottom nadir poles from patch sampling to eliminate tripod shadows and spherical distortion artifacts (8% recommended).")

        dvo_ba_frame = tk.Frame(dvo_col2, bg=self.panel_bg)
        dvo_ba_frame.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_ba = tk.Label(dvo_ba_frame, text="ODBA Window:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_ba.pack(side=tk.LEFT)
        cb_ba = ttk.Combobox(dvo_ba_frame, textvariable=self.dvo_ba_window_var, values=["4", "8", "12", "16"], state="readonly", width=8, font=("Segoe UI", 8))
        cb_ba.pack(side=tk.LEFT)
        ToolTip(lbl_ba, "Omnidirectional Direct Bundle Adjustment (ODBA) keyframe window. Jointly refines recent camera poses on S² manifold.")
        ToolTip(cb_ba, "Omnidirectional Direct Bundle Adjustment (ODBA) keyframe window. Jointly refines recent camera poses on S² manifold.")

        dvo_life_frame = tk.Frame(dvo_col2, bg=self.panel_bg)
        dvo_life_frame.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_life = tk.Label(dvo_life_frame, text="Patch Lifetime:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_life.pack(side=tk.LEFT)
        cb_life = ttk.Combobox(dvo_life_frame, textvariable=self.dvo_patch_lifetime_var, values=["10", "13 (Default)", "16 (Extended)", "20 (Maximum)"], state="readonly", width=12, font=("Segoe UI", 8))
        cb_life.pack(side=tk.LEFT)
        ToolTip(lbl_life, "Keyframe tracking lifetime window (in frames). Controls how long patches stay active in the bundle graph.")
        ToolTip(cb_life, "Patch Lifetime:\nControls how many consecutive keyframes a patch is tracked before being retired. Higher values improve long-baseline triangulation geometry and reduce trajectory drift in expansive loops.")

        dvo_refine_frame = tk.Frame(dvo_col2, bg=self.panel_bg)
        dvo_refine_frame.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_refine = tk.Label(dvo_refine_frame, text="Final Refine:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_refine.pack(side=tk.LEFT)
        cb_refine = ttk.Combobox(dvo_refine_frame, textvariable=self.dvo_final_refine_var, values=["12 (Standard)", "18 (Deep Refine)", "24 (Production)"], state="readonly", width=12, font=("Segoe UI", 8))
        cb_refine.pack(side=tk.LEFT)
        ToolTip(lbl_refine, "Number of final global bundle adjustment iterations over all keyframes after tracking concludes.")
        ToolTip(cb_refine, "Final Refinement Iterations:\nPost-processing bundle adjustment iterations run across all keyframes and 3D tie points. Higher iterations optimize camera poses to sub-pixel accuracy before exporting poses.")

        dvo_scale_frame = tk.Frame(dvo_col2, bg=self.panel_bg)
        dvo_scale_frame.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_scale = tk.Label(dvo_scale_frame, text="Metric Scale (m):", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_scale.pack(side=tk.LEFT)
        cb_scale = ttk.Combobox(dvo_scale_frame, textvariable=self.dvo_scale_var, values=["1.0", "2.0", "3.2", "5.0"], state="readonly", width=8, font=("Segoe UI", 8))
        cb_scale.pack(side=tk.LEFT)
        ToolTip(lbl_scale, "Metric translation multiplier in meters. Calibrates monocular visual odometry bearing disparities to real-world scale.")
        ToolTip(cb_scale, "Metric translation multiplier in meters. Calibrates monocular visual odometry bearing disparities to real-world scale.")

        chk_grad = tk.Checkbutton(dvo_col2, text="Gradient-Biased Sampling", variable=self.dvo_gradient_bias_var, font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        chk_grad.pack(anchor="w", pady=(1, 0))
        ToolTip(chk_grad, "Gradient-Biased Patch Sampling:\nWhen enabled (recommended), samples patches preferentially on high-gradient edges and textured regions, significantly reducing tracking failures on smooth surfaces or skies.")

        dvo_viz_chk = tk.Checkbutton(dvo_col2, text="Real-Time 3D Visualizer (Auto-Pop)", variable=self.dvo_viz_var, font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        dvo_viz_chk.pack(anchor="w", pady=(1, 0))
        ToolTip(dvo_viz_chk, "Real-Time 3D Visualizer (Viser WebGL):\nAutomatically pops up your default web browser to http://localhost:8080 as soon as the 360DVO visualizer initializes, providing live 3D visualization of camera poses and point cloud.")

        # Col C: Universal 3DGS Bridge
        dvo_col3 = tk.Frame(self.panel_align_360dvo, bg=self.panel_bg)
        dvo_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(dvo_col3, text="Universal 3DGS Bridge:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        tk.Label(dvo_col3, text="⚡ Auto-Exports All 3DGS Formats", font=("Segoe UI", 8, "bold"), fg="#60a5fa", bg=self.panel_bg).pack(anchor="w", pady=(1, 1))
        tk.Label(dvo_col3, text="• 9x Dense Micro-Cloud (700k+ pts)\n• TUM Odometry (tx, ty, tz, qx, qy, qz, qw)\n• COLMAP cameras.txt, images.txt, points3D.txt\n• Universal transforms.json (EQUIRECTANGULAR)\n• OpenSfM reconstruction.json + metadata.json\n• 100% GPU training in PFGS360 & LichtFeld", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, justify=tk.LEFT).pack(anchor="w")

        # =========================================================================
        # 6. NVIDIA VGG-T3 (VGG-TTT) Subpanel (Linear SfM - CVPR 2026)
        # =========================================================================
        self.panel_align_vggttt = tk.Frame(self.align_dynamic_container, bg=self.panel_bg)

        # Col A: Density, Confidence & Preprocessing
        ttt_col1 = tk.Frame(self.panel_align_vggttt, bg=self.panel_bg)
        ttt_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ttt_col1, text="Point Cloud & Confidence:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        ttt_conf_frame = tk.Frame(ttt_col1, bg=self.panel_bg)
        ttt_conf_frame.pack(anchor="w", pady=(2, 2), fill=tk.X)
        lbl_ttt_conf = tk.Label(ttt_conf_frame, text="Confidence Thresh:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=15, anchor="w")
        lbl_ttt_conf.pack(side=tk.LEFT)
        cb_ttt_conf = ttk.Combobox(ttt_conf_frame, textvariable=self.vggttt_conf_var, values=["Adaptive (20% Quantile)", "1.0 (Dense)", "1.5", "2.0 (Standard)", "2.5", "3.0 (Clean)", "4.0 (Ultra)"], state="readonly", width=17, font=("Segoe UI", 8))
        cb_ttt_conf.pack(side=tk.LEFT)
        ToolTip(lbl_ttt_conf, "Minimum depth confidence threshold for 3D point cloud generation. 'Adaptive (20% Quantile)' dynamically filters outliers matching NVIDIA's official demo.")
        ToolTip(cb_ttt_conf, "Depth Confidence Filter (VGG-T3):\n• Adaptive (20% Quantile) [Recommended]: Uses NVIDIA demo quantile formula to filter floating points and preserve crisp surface geometry.\n• 2.0 (Standard): Fixed numerical threshold.\n• Lower values yield denser clouds; higher values keep strictly high-certainty tie points.")

        ttt_prep_frame = tk.Frame(ttt_col1, bg=self.panel_bg)
        ttt_prep_frame.pack(anchor="w", pady=(2, 2), fill=tk.X)
        lbl_ttt_prep = tk.Label(ttt_prep_frame, text="Preprocessing:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=15, anchor="w")
        lbl_ttt_prep.pack(side=tk.LEFT)
        cb_ttt_prep = ttk.Combobox(ttt_prep_frame, textvariable=self.vggttt_prep_var, values=["Auto (Smart Center-Crop)", "Crop (NVIDIA Demo Standard)", "Pad (Aspect Ratio Padding)"], state="readonly", width=17, font=("Segoe UI", 8))
        cb_ttt_prep.pack(side=tk.LEFT)
        ToolTip(lbl_ttt_prep, "Preprocessing mode applied before feeding frames to VGG-T3 transformer.")
        ToolTip(cb_ttt_prep, "VGG-T3 Image Preprocessing:\n• Auto (Smart Center-Crop) [Recommended]: Uses clean center-crop when mixed landscape/portrait frames exist, eliminating artificial white padding bars.\n• Crop (NVIDIA Demo Standard): Crops to 518x518 with 100% genuine photo pixels.\n• Pad: Adds white padding to preserve full sensor aspect ratio.")

        chk_pred = tk.Checkbutton(ttt_col1, text="Use Global Point Head", variable=self.vggttt_pred_var, font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        chk_pred.pack(anchor="w", pady=(2, 0))
        ToolTip(chk_pred, "Global Point Head:\nWhen enabled, extracts 3D points directly from the global transformer point prediction head (matching NVIDIA Gradio demo default). When unchecked, backprojects depth and camera intrinsics.")

        # Col B: Test-Time Training (TTT) & Attention Mode
        ttt_col2 = tk.Frame(self.panel_align_vggttt, bg=self.panel_bg)
        ttt_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ttt_col2, text="Test-Time Training (TTT):", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        ttt_steps_frame = tk.Frame(ttt_col2, bg=self.panel_bg)
        ttt_steps_frame.pack(anchor="w", pady=(2, 2), fill=tk.X)
        lbl_ttt_steps = tk.Label(ttt_steps_frame, text="TTT Inner Steps:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=15, anchor="w")
        lbl_ttt_steps.pack(side=tk.LEFT)
        cb_ttt_steps = ttk.Combobox(ttt_steps_frame, textvariable=self.vggttt_steps_var, values=["1 (Fast Preview)", "2 (Recommended)", "3 (Balanced)", "5 (Deep)"], state="readonly", width=14, font=("Segoe UI", 8))
        cb_ttt_steps.pack(side=tk.LEFT)
        ToolTip(lbl_ttt_steps, "Number of Test-Time Training optimization steps per layer during feedforward alignment.")
        ToolTip(cb_ttt_steps, "TTT Steps (NVIDIA CVPR 2026):\n• 2 (Recommended): Mathematically optimal sweet spot derived in the paper for <= 1,000 views.\n• 1 (Fast Preview): Ultra-fast single-pass update.\n• 5 (Deep): 5 gradient updates per block; can overfit on disordered mirrorless photos.")

        chk_mem = tk.Checkbutton(ttt_col2, text="Memory-Efficient (>64 views)", variable=self.vggttt_mem_var, font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        chk_mem.pack(anchor="w", pady=(2, 0))
        ToolTip(chk_mem, "Memory-Efficient Attention Mode:\nEnables chunked linear fast-weight recurrence & CPU memory offloading for massive image collections (>64 photos). Prevents VRAM overflow and keeps peak GPU memory strictly below 7 GB regardless of dataset size.")

        # Col C: Architecture & Universal 3DGS Bridge
        ttt_col3 = tk.Frame(self.panel_align_vggttt, bg=self.panel_bg)
        ttt_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(ttt_col3, text="NVIDIA VGG-T³ Architecture:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        tk.Label(ttt_col3, text="⚡ Linear O(N) SfM Transformer (CVPR 2026)", font=("Segoe UI", 8, "bold"), fg="#60a5fa", bg=self.panel_bg).pack(anchor="w", pady=(1, 1))
        tk.Label(ttt_col3, text="• Fast-Weight Attention + Muon Optimizer\n• Aligns 1,000 photos in <1 minute on GPU\n• Linear complexity without quadratic bottleneck\n• Native COLMAP, transforms.json, points.ply\n• Automatic handoff to Abu Train (LichtFeld / 3DGRUT)", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, justify=tk.LEFT).pack(anchor="w")

    def _update_align_panel(self):
        engine = self.engine_var.get()
        self.panel_align_vggt.pack_forget()
        self.panel_align_ace0.pack_forget()
        self.panel_align_pycusfm.pack_forget()
        if hasattr(self, "panel_align_opensfm"):
            self.panel_align_opensfm.pack_forget()
        if hasattr(self, "panel_align_360dvo"):
            self.panel_align_360dvo.pack_forget()
        if hasattr(self, "panel_align_vggttt"):
            self.panel_align_vggttt.pack_forget()

        if engine in ["vggt_omega", "vggt"]:
            self.panel_align_vggt.pack(fill=tk.BOTH, expand=True)
            if hasattr(self, "vggt_cache_frame"):
                self.vggt_cache_frame.pack(side=tk.LEFT)
        else:
            if hasattr(self, "vggt_cache_frame"):
                self.vggt_cache_frame.pack_forget()

            if engine in ["vggttt", "vgg-ttt", "vggt3"]:
                if hasattr(self, "panel_align_vggttt"):
                    self.panel_align_vggttt.pack(fill=tk.BOTH, expand=True)
            elif engine == "ace0":
                self.panel_align_ace0.pack(fill=tk.BOTH, expand=True)
            elif engine == "pycusfm":
                self.panel_align_pycusfm.pack(fill=tk.BOTH, expand=True)
            elif engine == "opensfm":
                self.panel_align_opensfm.pack(fill=tk.BOTH, expand=True)
            elif engine in ["360dvo", "dvo"]:
                self.panel_align_360dvo.pack(fill=tk.BOTH, expand=True)

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

        # =========================================================================
        # 4. PFGS360 (Pose-Free Omnidirectional 360° GS - CVPR 2026) Subpanel
        # =========================================================================
        self.panel_train_pfgs360 = tk.Frame(self.train_dynamic_container, bg=self.panel_bg)

        # Col A: Dataparser & Alignment Handoff
        pfgs_col1 = tk.Frame(self.panel_train_pfgs360, bg=self.panel_bg)
        pfgs_col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(pfgs_col1, text="Dataparser & Alignment Mode:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")
        rb_pfgs_d1 = tk.Radiobutton(pfgs_col1, text="Auto-Detect [100% GPU with Pre-Aligned Poses]", variable=self.pfgs_dataparser_var, value="auto", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_pfgs_d1.pack(anchor="w")
        rb_pfgs_d2 = tk.Radiobutton(pfgs_col1, text="OpenSfM / 360DVO Handoff [Instant 100% GPU]", variable=self.pfgs_dataparser_var, value="opensfm", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_pfgs_d2.pack(anchor="w")
        rb_pfgs_d3 = tk.Radiobutton(pfgs_col1, text="Pose-Free Direct [Unposed Video - CPU Matching]", variable=self.pfgs_dataparser_var, value="posefree", bg=self.panel_bg, fg=self.text_primary, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        rb_pfgs_d3.pack(anchor="w")
        tk.Label(pfgs_col1, text="⚡ Auto-Route: 100% GPU if poses exist", font=("Segoe UI", 8, "bold"), fg="#60a5fa", bg=self.panel_bg).pack(anchor="w", pady=(1, 0))
        ToolTip(rb_pfgs_d1, "Auto-Detect [Recommended Default]:\nChecks if pre-aligned camera poses exist (OpenSfM, 360DVO, or COLMAP).\n• If found: loads poses in <0.5s and engages 100% RTX 4090 CUDA GPU training immediately!\n• If unposed video: engages pose-free sliding-window matching.")
        ToolTip(rb_pfgs_d2, "OpenSfM / 360DVO Handoff:\nForces loading pre-aligned poses from reconstruction.json or transforms.json.\n• Bypasses CPU SIFT matching phase completely.\n• Maximizes RTX 4090 CUDA GPU utilization from step 0.")
        ToolTip(rb_pfgs_d3, "Pose-Free Direct [CVPR 2026 Paradigm A]:\nFor raw unaligned video clips without any prior SfM.\n• Solves relative camera poses using spherical bearing disparity.\n• Use Match Views = 8 to prevent CPU freeze.")

        # Col B: Depth Prior Model & Optimization Window
        pfgs_col2 = tk.Frame(self.panel_train_pfgs360, bg=self.panel_bg)
        pfgs_col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(pfgs_col2, text="Depth Prior & Pose Optimization:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        depth_row = tk.Frame(pfgs_col2, bg=self.panel_bg)
        depth_row.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_depth = tk.Label(depth_row, text="Depth Prior:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_depth.pack(side=tk.LEFT)
        depth_cb = ttk.Combobox(depth_row, textvariable=self.pfgs_depth_var, values=["UniK3D", "midas", "zoedepth"], state="readonly", width=9, font=("Segoe UI", 8))
        depth_cb.pack(side=tk.LEFT)
        ToolTip(lbl_depth, "Monocular depth estimation prior model for spherical backprojection.")
        ToolTip(depth_cb, "Depth Prior Model:\n• UniK3D [CVPR 2026 Recommended]: Geometrically consistent omnidirectional spherical depth prior.\n• midas: Multi-condition robust monocular depth.\n• zoedepth: Metric relative zero-shot depth estimation.")

        mv_row = tk.Frame(pfgs_col2, bg=self.panel_bg)
        mv_row.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_mv = tk.Label(mv_row, text="Match Views:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_mv.pack(side=tk.LEFT)
        mv_cb = ttk.Combobox(mv_row, textvariable=self.pfgs_match_views_var, values=["8 (Sliding Window - Fast & Smooth)", "4 (Quickest Window)", "16 (Wide Window)", "0 (All Views - Slow O(N²))"], state="readonly", width=18, font=("Segoe UI", 8))
        mv_cb.pack(side=tk.LEFT)
        ToolTip(lbl_mv, "Number of recent keyframes used for pairwise relative camera pose matching.")
        ToolTip(mv_cb, "Match Views Window Size:\n• 8 [Recommended]: Fast sliding-window matching (O(N·K)). Eliminates CPU bottleneck and keeps GPU at 100% power.\n• 4: Quickest window for fast-moving cameras.\n• 16: Wider window for high-overlap loop closure.\n• 0: Matches against ALL visited views (O(N²) CPU SIFT, causes slow training).")

        dens_row = tk.Frame(pfgs_col2, bg=self.panel_bg)
        dens_row.pack(anchor="w", pady=(1, 1), fill=tk.X)
        lbl_dens = tk.Label(dens_row, text="Densify Views:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, width=12, anchor="w")
        lbl_dens.pack(side=tk.LEFT)
        dens_cb = ttk.Combobox(dens_row, textvariable=self.pfgs_densify_views_var, values=["8", "4", "16", "0"], state="readonly", width=6, font=("Segoe UI", 8))
        dens_cb.pack(side=tk.LEFT)
        ToolTip(lbl_dens, "Number of recent frames used to compute depth inliers and outliers during DIA densification.")
        ToolTip(dens_cb, "Number of recent frames used to compute depth inliers and outliers during DIA densification (8 recommended).")

        pfgs_sph_chk = tk.Checkbutton(pfgs_col2, text="SCA-PE (Spherical Coords PE)", variable=self.pfgs_spherical_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        pfgs_sph_chk.pack(anchor="w")
        ToolTip(pfgs_sph_chk, "Spherical Coordinate Aware Positional Encoding (SCA-PE):\nMaps Gaussians directly across omnidirectional spherical coordinates to eliminate pole pinch distortion.")

        pfgs_consist_chk = tk.Checkbutton(pfgs_col2, text="Consist-Aware (Epipolar Masking)", variable=self.pfgs_consist_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        pfgs_consist_chk.pack(anchor="w")
        ToolTip(pfgs_consist_chk, "Epipolar Consistency Masks:\nDiscards occluded and dynamic moving pixels from camera pose estimation.")

        # Col C: DIA-Densify, Sky Mask & Export
        pfgs_col3 = tk.Frame(self.panel_train_pfgs360, bg=self.panel_bg)
        pfgs_col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(pfgs_col3, text="DIA-Densify, Sky Mask & SH:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg).pack(anchor="w")

        pfgs_inlier_chk = tk.Checkbutton(pfgs_col3, text="DIA-Densify (Inlier Depth Growth)", variable=self.pfgs_inlier_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        pfgs_inlier_chk.pack(anchor="w")
        ToolTip(pfgs_inlier_chk, "Depth Inlier Aware Densification (DIA-Densify):\nUses depth-consistent backprojected inliers to guide 3D Gaussian splitting and cloning.")

        pfgs_align_chk = tk.Checkbutton(pfgs_col3, text="Align Monocular Depth with Poses", variable=self.pfgs_align_depth_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        pfgs_align_chk.pack(anchor="w")
        ToolTip(pfgs_align_chk, "Align Depth:\nOptimizes affine scale and shift between monocular depth priors and rendered internal depth.")

        sky_chk = tk.Checkbutton(pfgs_col3, text="Filter Sky & Infinity Pixels (>50m)", variable=self.pfgs_filter_sky_var, font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        sky_chk.pack(anchor="w")
        ToolTip(sky_chk, "Sky Filtering:\nFilters out distant sky / infinite distance pixels (>50m) to prevent floaters in outdoor 360° panoramas.")

        pfgs_sh_row = tk.Frame(pfgs_col3, bg=self.panel_bg)
        pfgs_sh_row.pack(anchor="w", pady=(2, 2))
        tk.Label(pfgs_sh_row, text="SH Degree:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT)
        pfgs_sh_cb = ttk.Combobox(pfgs_sh_row, textvariable=self.pfgs_sh_var, values=["0", "1", "2", "3", "4"], state="readonly", width=4)
        pfgs_sh_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(pfgs_sh_cb, "Spherical Harmonics Degree:\nDegree 3 is standard. Degree 4 provides ultra-rich specular highlights.")

        pfgs_res_row = tk.Frame(pfgs_col3, bg=self.panel_bg)
        pfgs_res_row.pack(anchor="w", pady=(2, 2))
        lbl_res = tk.Label(pfgs_res_row, text="Train Res:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg)
        lbl_res.pack(side=tk.LEFT)
        pfgs_res_cb = ttk.Combobox(pfgs_res_row, textvariable=self.pfgs_downscale_var, values=["1 (Full 100% Quality)", "2 (Half Res - 2x Speed)", "4 (Quarter Res - Fast)", "Auto (<= 1600px Max)"], state="readonly", width=19, font=("Segoe UI", 8))
        pfgs_res_cb.pack(side=tk.LEFT, padx=4)
        ToolTip(lbl_res, "Training dataset image resolution / downscale factor passed to Nerfstudio dataparser.")
        ToolTip(pfgs_res_cb, "Training Image Resolution:\n• 1 (Full 100% Quality): Loads training equirectangular images at full native resolution.\n• 2 (Half Res): Downscales 2x for faster iterations and lower VRAM.\n• 4 (Quarter Res): Downscales 4x for ultra-fast previews.\n• Auto: Nerfstudio default (automatically downscales if max dimension > 1600px).\n\nNOTE: The 'Resolution: 402x512px' seen in Viser (localhost:7007) is the browser viewport preview resolution (Aspect Ratio × 'Max res: 512' slider), NOT this training dataset resolution.")
        tk.Label(pfgs_col3, text="💡 Viser 7007: adjust 'Max res' slider in browser", font=("Segoe UI", 8), fg="#94a3b8", bg=self.panel_bg).pack(anchor="w", pady=(1, 0))

    def _update_train_panel(self):
        engine = self.train_engine_var.get()
        self.panel_train_lichtfeld.pack_forget()
        self.panel_train_nerfstudio.pack_forget()
        self.panel_train_3dgrut.pack_forget()
        self.panel_train_pfgs360.pack_forget()

        if engine == "lichtfeld":
            self.panel_train_lichtfeld.pack(fill=tk.BOTH, expand=True)
        elif engine == "splatfacto":
            self.panel_train_nerfstudio.pack(fill=tk.BOTH, expand=True)
        elif engine == "3dgrut":
            self.panel_train_3dgrut.pack(fill=tk.BOTH, expand=True)
        elif engine == "pfgs360":
            self.panel_train_pfgs360.pack(fill=tk.BOTH, expand=True)

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

    def _browse_video(self):
        initial = os.path.dirname(self.dataset_var.get()) if os.path.isfile(self.dataset_var.get()) else self.dataset_var.get()
        vfile = filedialog.askopenfilename(
            initialdir=initial,
            title="Select 360° or Planar Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.insv *.avi *.mkv *.webm *.m4v"),
                ("All Files", "*.*")
            ]
        )
        if vfile:
            self.dataset_var.set(vfile)

    def _validate_dataset(self, *_):
        if not hasattr(self, "dataset_status_lbl"):
            return
        path = self.dataset_var.get().strip()
        if not path:
            self.dataset_status_lbl.config(text="Choose a folder of photos or prepared training data to begin.", fg=self.text_secondary)
            return
        if not os.path.exists(path):
            self.dataset_status_lbl.config(text="Folder not found. Check the path or choose another folder.", fg=self.accent_red)
            return

        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in [".mp4", ".mov", ".insv", ".avi", ".mkv", ".webm", ".m4v"]:
                self.dataset_status_lbl.config(
                    text=f"Video · {os.path.basename(path)} · Use Prepare input → Video frames, or choose 360DVO.",
                    fg="#38bdf8"
                )
            else:
                self.dataset_status_lbl.config(text=f"📄 Selected file: {os.path.basename(path)}", fg=self.text_secondary)
            return

        # Folder inspection
        img_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
        try:
            entries = os.listdir(path)
        except Exception as e:
            self.dataset_status_lbl.config(text=f"⚠️ Cannot read folder: {e}", fg=self.accent_red)
            return

        img_count = sum(1 for f in entries if os.path.splitext(f)[1].lower() in img_exts)
        vid_count = sum(1 for f in entries if os.path.splitext(f)[1].lower() in {".mp4", ".mov", ".insv", ".avi", ".mkv"})
        
        has_sparse = os.path.isdir(os.path.join(path, "sparse", "0")) or os.path.isdir(os.path.join(path, "3dgs_dataset", "sparse", "0"))
        has_transforms = os.path.isfile(os.path.join(path, "transforms.json")) or os.path.isfile(os.path.join(path, "3dgs_dataset", "transforms.json"))
        has_results = os.path.isdir(os.path.join(path, "results"))

        poses_str = "Camera poses found" if (has_sparse or has_transforms) else ("Alignment results found" if has_results else "No existing camera poses")
        
        if img_count > 0:
            status_text = f"{img_count:,} photos · {poses_str}"
            self.dataset_status_lbl.config(text=status_text, fg=self.accent_green)
        elif vid_count > 0:
            status_text = f"{vid_count} videos · Use Prepare input → Video frames to extract photos."
            self.dataset_status_lbl.config(text=status_text, fg="#38bdf8")
        elif has_results or has_sparse or has_transforms:
            status_text = f"{poses_str} · Choose Train only to use this reconstruction."
            self.dataset_status_lbl.config(text=status_text, fg="#a855f7")
        else:
            status_text = "No photos or videos found. Choose a folder containing your source images."
            self.dataset_status_lbl.config(text=status_text, fg="#f59e0b")

        # Discover all available alignment runs in results/
        self._refresh_train_target_runs()
        if hasattr(self, "workspace_3d"):
            try:
                self.workspace_3d.refresh_runs()
            except Exception:
                pass

    def _refresh_train_target_runs(self, select_newest: bool = False) -> List[str]:
        """Discovers all available alignment runs in results/ and dynamically updates the dropdown."""
        runs = ["Latest Run (Auto-Detect)"]
        path = self.dataset_var.get() if hasattr(self, "dataset_var") else ""
        if path and os.path.isdir(path):
            cand_results = os.path.join(path, "results")
            if not os.path.isdir(cand_results):
                cand_p_results = os.path.join(os.path.dirname(path), "results")
                if os.path.isdir(cand_p_results):
                    cand_results = cand_p_results
            if os.path.isdir(cand_results):
                subdirs = [d for d in os.listdir(cand_results) if os.path.isdir(os.path.join(cand_results, d)) and d.startswith("sparse_")]
                subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(cand_results, d)), reverse=True)
                runs.extend(subdirs)
        if hasattr(self, "train_target_run_cb"):
            self.train_target_run_cb["values"] = runs
            cur = self.train_target_run_var.get() if hasattr(self, "train_target_run_var") else ""
            if select_newest and len(runs) > 1:
                self.train_target_run_var.set(runs[1])
            elif cur not in runs:
                self.train_target_run_var.set(runs[0])
        return runs

    def _view_training_run_report(self):
        """Displays the Reconstruction Report for the currently selected alignment target in Tab 02 Training."""
        path = self.dataset_var.get() if hasattr(self, "dataset_var") else ""
        if not path or not os.path.isdir(path):
            messagebox.showinfo("Reconstruction Report", "No valid dataset path selected.")
            return

        cand_results = os.path.join(path, "results")
        if not os.path.isdir(cand_results):
            cand_p_results = os.path.join(os.path.dirname(path), "results")
            if os.path.isdir(cand_p_results):
                cand_results = cand_p_results

        if not os.path.isdir(cand_results):
            messagebox.showinfo("Reconstruction Report", "No results/ folder found for this dataset.")
            return

        target_run = self.train_target_run_var.get() if hasattr(self, "train_target_run_var") else ""
        if not target_run or target_run == "Latest Run (Auto-Detect)":
            subdirs = [d for d in os.listdir(cand_results) if os.path.isdir(os.path.join(cand_results, d)) and d.startswith("sparse_")]
            subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(cand_results, d)), reverse=True)
            if not subdirs:
                messagebox.showinfo("Reconstruction Report", "No sparse_* alignment runs found.")
                return
            target_run = subdirs[0]

        run_dir = os.path.join(cand_results, target_run)
        if not os.path.isdir(run_dir):
            messagebox.showinfo("Reconstruction Report", f"Selected alignment run directory does not exist:\n{run_dir}")
            return

        if hasattr(self, "workspace_3d") and hasattr(self.workspace_3d, "show_reconstruction_report"):
            self.workspace_3d.active_run_dir = run_dir
            self.workspace_3d.active_run_name = target_run
            self.workspace_3d.show_reconstruction_report()
        else:
            rep_path = os.path.join(run_dir, "reconstruction_report.txt")
            if os.path.isfile(rep_path):
                os.system(f'start notepad "{rep_path}"')
            else:
                os.startfile(run_dir)

    def _open_video_extractor_dialog(self):
        """
        Launches the dedicated Video to Frame Sequence Extractor dialog.
        Allows extracting any 360° or planar video into JPEG/PNG sequences with professional options.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("🎬 Video to Frame Sequence Extractor - Abu Align")
        dialog.geometry("740x630")
        dialog.minsize(680, 560)
        dialog.configure(bg=self.bg_dark)
        dialog.transient(self.root)

        # Header
        hdr = tk.Frame(dialog, bg=self.panel_bg, padx=14, pady=10)
        hdr.pack(fill=tk.X, padx=12, pady=(12, 8))
        tk.Label(
            hdr,
            text="🎬 VIDEO TO FRAME SEQUENCE EXTRACTOR",
            font=("Segoe UI", 12, "bold"),
            fg="#38bdf8",
            bg=self.panel_bg
        ).pack(anchor="w")
        tk.Label(
            hdr,
            text="Extract 360° Equirectangular or Planar video files into high-fidelity image sequences for photogrammetry & 3DGS.",
            font=("Segoe UI", 9),
            fg=self.text_secondary,
            bg=self.panel_bg
        ).pack(anchor="w", pady=(2, 0))

        from desktop_ui import ScrollPage
        video_page = ScrollPage(dialog)
        video_page.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        content = tk.Frame(video_page.body, bg=self.bg_dark)
        content.pack(fill=tk.X)

        # 1. Paths Card
        paths_card = tk.LabelFrame(
            content,
            text=" Video Input & Destination ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=12,
            pady=8
        )
        paths_card.pack(fill=tk.X, pady=(0, 8))

        vid_in_var = tk.StringVar()
        current_in = self.dataset_var.get().strip()
        if os.path.isfile(current_in) and any(current_in.lower().endswith(ext) for ext in [".mp4", ".mov", ".insv", ".avi", ".mkv"]):
            vid_in_var.set(current_in)

        out_dir_var = tk.StringVar()
        if vid_in_var.get():
            out_dir_var.set(os.path.join(os.path.dirname(vid_in_var.get()), f"{Path(vid_in_var.get()).stem}_frames"))

        # Row 1: Input Video
        r1 = tk.Frame(paths_card, bg=self.panel_bg)
        r1.pack(fill=tk.X, pady=3)
        tk.Label(r1, text="Input Video:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg, width=12, anchor="w").pack(side=tk.LEFT)
        in_entry = tk.Entry(r1, textvariable=vid_in_var, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, relief=tk.FLAT)
        in_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=2)

        def browse_in():
            f = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video Files", "*.mp4 *.mov *.insv *.avi *.mkv *.webm"), ("All Files", "*.*")]
            )
            if f:
                vid_in_var.set(f)
                if not out_dir_var.get():
                    out_dir_var.set(os.path.join(os.path.dirname(f), f"{Path(f).stem}_frames"))

        tk.Button(r1, text="Browse Video...", font=("Segoe UI", 8), bg=self.accent_blue, fg="#ffffff", relief=tk.FLAT, padx=8, command=browse_in).pack(side=tk.RIGHT)

        # Row 2: Destination Folder
        r2 = tk.Frame(paths_card, bg=self.panel_bg)
        r2.pack(fill=tk.X, pady=3)
        tk.Label(r2, text="Output Folder:", font=("Segoe UI", 9, "bold"), fg=self.text_primary, bg=self.panel_bg, width=12, anchor="w").pack(side=tk.LEFT)
        out_entry = tk.Entry(r2, textvariable=out_dir_var, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, relief=tk.FLAT)
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=2)

        def browse_out():
            d = filedialog.askdirectory(title="Select Output Directory for Frames")
            if d:
                out_dir_var.set(d)

        tk.Button(r2, text="Browse Folder...", font=("Segoe UI", 8), bg="#29293d", fg=self.text_primary, relief=tk.FLAT, padx=8, command=browse_out).pack(side=tk.RIGHT)

        # 2. Controls Card
        ctrl_card = tk.LabelFrame(
            content,
            text=" Professional Extraction Controls ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=12,
            pady=8
        )
        ctrl_card.pack(fill=tk.X, pady=(0, 8))

        # Grid of controls
        g1 = tk.Frame(ctrl_card, bg=self.panel_bg)
        g1.pack(fill=tk.X, pady=2)

        mode_var = tk.StringVar(value="fps")
        fps_var = tk.StringVar(value="5.0")
        stride_var = tk.StringVar(value="1")
        fmt_var = tk.StringVar(value="jpg")
        qual_var = tk.StringVar(value="95")
        res_var = tk.StringVar(value="native")
        hwaccel_var = tk.BooleanVar(value=True)
        blur_filter_var = tk.BooleanVar(value=False)
        blur_val_var = tk.StringVar(value="30")

        tk.Label(g1, text="Extraction Mode:", font=("Segoe UI", 9), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        mode_cb = ttk.Combobox(g1, textvariable=mode_var, values=["fps", "stride", "keyframes", "all"], state="readonly", width=10)
        mode_cb.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(g1, text="Target FPS:", font=("Segoe UI", 9), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        fps_entry = tk.Entry(g1, textvariable=fps_var, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=6, relief=tk.FLAT)
        fps_entry.pack(side=tk.LEFT, padx=(0, 16), ipady=1)

        tk.Label(g1, text="Frame Stride:", font=("Segoe UI", 9), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        stride_entry = tk.Entry(g1, textvariable=stride_var, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT)
        stride_entry.pack(side=tk.LEFT, ipady=1)

        g2 = tk.Frame(ctrl_card, bg=self.panel_bg)
        g2.pack(fill=tk.X, pady=(6, 2))

        tk.Label(g2, text="Image Format:", font=("Segoe UI", 9), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        fmt_cb = ttk.Combobox(g2, textvariable=fmt_var, values=["jpg", "png"], state="readonly", width=6)
        fmt_cb.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(g2, text="JPEG Quality:", font=("Segoe UI", 9), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        qual_entry = tk.Entry(g2, textvariable=qual_var, font=("Segoe UI", 9), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT)
        qual_entry.pack(side=tk.LEFT, padx=(0, 16), ipady=1)

        tk.Label(g2, text="Resolution:", font=("Segoe UI", 9), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        res_cb = ttk.Combobox(g2, textvariable=res_var, values=["native", "8k", "5.7k", "4k", "2k", "1k"], state="readonly", width=8)
        res_cb.pack(side=tk.LEFT)

        g3 = tk.Frame(ctrl_card, bg=self.panel_bg)
        g3.pack(fill=tk.X, pady=(6, 2))

        hw_chk = tk.Checkbutton(g3, text="⚡ NVIDIA NVDEC / CUDA Hardware Acceleration", variable=hwaccel_var, font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        hw_chk.pack(side=tk.LEFT)

        blur_chk = tk.Checkbutton(g3, text="🔍 Filter Motion Blur (Laplacian >", variable=blur_filter_var, font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg)
        blur_chk.pack(side=tk.LEFT, padx=(16, 2))
        blur_entry = tk.Entry(g3, textvariable=blur_val_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=4, relief=tk.FLAT)
        blur_entry.pack(side=tk.LEFT, ipady=1)
        tk.Label(g3, text=")", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT)

        # 3. Status & Progress Card
        prog_card = tk.LabelFrame(
            content,
            text=" Progress & Telemetry ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=12,
            pady=8
        )
        prog_card.pack(fill=tk.X, pady=(0, 8))

        dlg_pbar = ttk.Progressbar(prog_card, mode="determinate")
        dlg_pbar.pack(fill=tk.X, pady=(2, 4))

        status_lbl = tk.Label(prog_card, text="Status: Ready to extract.", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, anchor="w")
        status_lbl.pack(fill=tk.X)

        # 4. Action Buttons
        btn_row = tk.Frame(dialog, bg=self.panel_bg, padx=12, pady=10)
        btn_row.pack(fill=tk.X, side=tk.BOTTOM, before=video_page)

        extract_btn = tk.Button(
            btn_row,
            text="▶ Start Frame Extraction",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_green,
            fg="#ffffff",
            activebackground="#059669",
            relief=tk.FLAT,
            padx=16,
            pady=6
        )
        extract_btn.pack(side=tk.LEFT, padx=(0, 8))

        use_ds_btn = tk.Button(
            btn_row,
            text="✓ Set as Active Dataset in Abu Align",
            font=("Segoe UI", 9),
            bg="#29293d",
            fg=self.text_primary,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        use_ds_btn.pack(side=tk.LEFT, padx=(0, 8))

        open_folder_btn = tk.Button(
            btn_row,
            text="📁 Open Folder",
            font=("Segoe UI", 9),
            bg="#29293d",
            fg=self.text_primary,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        open_folder_btn.pack(side=tk.LEFT)

        close_btn = tk.Button(
            btn_row,
            text="Close",
            font=("Segoe UI", 9),
            bg="#1f1f2e",
            fg=self.text_secondary,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            command=dialog.destroy
        )
        close_btn.pack(side=tk.RIGHT)

        def set_as_dataset():
            d = out_dir_var.get().strip()
            if os.path.isdir(d):
                self.dataset_var.set(d)
                messagebox.showinfo("Dataset Set", f"Updated Abu Align dataset path to:\n{d}")
                dialog.destroy()

        use_ds_btn.config(command=set_as_dataset)

        def open_folder():
            d = out_dir_var.get().strip()
            if os.path.isdir(d):
                os.startfile(d)

        open_folder_btn.config(command=open_folder)

        def run_extraction_worker():
            in_v = vid_in_var.get().strip()
            out_d = out_dir_var.get().strip()
            if not in_v or not os.path.exists(in_v):
                messagebox.showerror("Error", "Please select a valid input video file or folder.")
                return
            if not out_d:
                out_d = os.path.join(os.path.dirname(in_v), f"{Path(in_v).stem}_frames")
                out_dir_var.set(out_d)

            extract_btn.config(state=tk.DISABLED, text="⏳ Extracting...")
            close_btn.config(state=tk.DISABLED)
            dlg_pbar["value"] = 0
            status_lbl.config(text="Status: Initializing extraction...", fg="#38bdf8")

            def on_progress(p):
                cur = p.get("current_frame", 0)
                tot = p.get("total_frames", 1)
                pct = p.get("percent", 0.0)
                rate = p.get("fps_rate", 0.0)
                status_txt = f"Status: Extracting frame {cur}/{tot} ({pct:.1f}%) | {rate:.1f} FPS"
                dialog.after(0, lambda: [
                    dlg_pbar.configure(value=pct),
                    status_lbl.configure(text=status_txt, fg="#38bdf8")
                ])

            def worker():
                try:
                    from video_extractor import extract_video_frames
                    mode = mode_var.get()
                    fps_val = float(fps_var.get()) if mode == "fps" else 5.0
                    stride_val = int(stride_var.get()) if mode == "stride" else 1
                    qual_val = int(qual_var.get())
                    res_val = res_var.get() if res_var.get() != "native" else None
                    blur_val = float(blur_val_var.get()) if blur_filter_var.get() else 0.0

                    res = extract_video_frames(
                        video_path=in_v,
                        output_dir=out_d,
                        mode=mode,
                        fps=fps_val,
                        stride=stride_val,
                        image_format=fmt_var.get(),
                        quality=qual_val,
                        resolution=res_val,
                        blur_threshold=blur_val,
                        use_hwaccel=hwaccel_var.get(),
                        progress_callback=on_progress,
                    )
                    tot = res.get("total_frames", 0)
                    sec = res.get("elapsed_seconds", 0.0)
                    fps_r = res.get("fps_rate", 0.0)

                    def on_done():
                        dlg_pbar["value"] = 100
                        status_lbl.config(
                            text=f"Status: Complete! Extracted {tot} frames in {sec}s ({fps_r} FPS).",
                            fg=self.accent_green
                        )
                        extract_btn.config(state=tk.NORMAL, text="Extract frames")
                        close_btn.config(state=tk.NORMAL)
                        use_ds_btn.config(state=tk.NORMAL)
                        open_folder_btn.config(state=tk.NORMAL)
                        self._log(f"\n[VIDEO EXTRACTOR] Successfully extracted {tot} frames to {out_d} in {sec}s ({fps_r} FPS).\n")

                    dialog.after(0, on_done)
                except Exception as ex:
                    def on_err(error=str(ex)):
                        status_lbl.config(text=f"Extraction failed: {error}", fg=self.accent_red)
                        extract_btn.config(state=tk.NORMAL, text="Extract frames")
                        close_btn.config(state=tk.NORMAL)
                        messagebox.showerror("Extraction Error", f"Failed to extract video frames:\n{error}")
                    dialog.after(0, on_err)

            t = threading.Thread(target=worker, daemon=True)
            t.start()

        extract_btn.config(command=run_extraction_worker)
        from desktop_ui import enhance_video_dialog
        enhance_video_dialog(self, locals())

    def _open_panoramic_splitter_dialog(self):
        """
        Launches the Advanced 360° Multi-Camera Rig Projector & Panoramic Splitter dialog.
        Features:
        - Auto-Math Solver: Bidirectional mathematical coupling between Camera Count, FOV, and Overlap.
        - Multiple Rig Architectures: Horizontal Ring, 6-Face Cubemap, Multi-Tier Dome, Fibonacci Sphere.
        - Direct 360° Video Intake (.mp4, .mov, .insv) with Target FPS & Keyframe Stride controls.
        - Interactive High-DPI Canvas Visualizer (Polar Radar View & Equirectangular Footprint Unroll).
        - Native Angular Pixel Density Matching (⚡ Auto-Res).
        - Radial Lens Distortion (k1, k2) & Exact Pinhole / OpenCV transforms.json Export.
        """
        import panoramic_splitter as ps

        dialog = tk.Toplevel(self.root)
        dialog.title("🌐 360° Multi-Camera Rig Projector & Splitter (Professional) - Abu Align")
        dialog.geometry("1160x830")
        dialog.minsize(1020, 700)
        dialog.configure(bg=self.bg_dark)
        dialog.transient(self.root)

        # Header
        hdr = tk.Frame(dialog, bg=self.panel_bg, padx=14, pady=10)
        hdr.pack(fill=tk.X, padx=12, pady=(10, 6))

        hdr_left = tk.Frame(hdr, bg=self.panel_bg)
        hdr_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            hdr_left,
            text="🌐 360° MULTI-CAMERA RIG PROJECTOR & SPLITTER",
            font=("Segoe UI", 12, "bold"),
            fg="#a855f7",
            bg=self.panel_bg
        ).pack(anchor="w")
        tk.Label(
            hdr_left,
            text="Mathematical Virtual Rig Synthesizer, Spherical Distribution Engine & Real-Time Placement Visualizer",
            font=("Segoe UI", 9),
            fg=self.text_secondary,
            bg=self.panel_bg
        ).pack(anchor="w", pady=(2, 0))

        # Main 2-Panel Content Layout
        body = tk.Frame(dialog, bg=self.bg_dark, padx=10, pady=2)
        body.pack(fill=tk.BOTH, expand=True)

        left_col = tk.Frame(body, bg=self.bg_dark)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        from desktop_ui import ScrollPage
        rig_notebook = ttk.Notebook(left_col)
        rig_notebook.pack(fill=tk.BOTH, expand=True)
        rig_pages = [ScrollPage(rig_notebook) for _ in range(3)]
        for page, title in zip(rig_pages, ("Source", "Rig", "Export")):
            rig_notebook.add(page, text=title)

        right_col = tk.Frame(body, bg=self.bg_dark, width=540)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # -----------------------------------------------------------------
        # State Variables
        # -----------------------------------------------------------------
        in_var = tk.StringVar(value=self.dataset_var.get().strip())
        out_var = tk.StringVar()
        if in_var.get():
            out_var.set(os.path.join(in_var.get(), "pinhole_rig"))

        mode_var = tk.StringVar(value="ring")           # "ring", "cubemap", "dome", "fibonacci"
        solver_mode_var = tk.StringVar(value="free")    # "free", "lock_overlap", "lock_fov"
        num_cams_var = tk.IntVar(value=8)
        fov_var = tk.DoubleVar(value=90.0)
        overlap_var = tk.DoubleVar(value=50.0)

        # Dome specific
        dome_pitch_var = tk.DoubleVar(value=35.0)
        dome_tier_cams_var = tk.IntVar(value=2)
        dome_zenith_var = tk.BooleanVar(value=True)
        dome_nadir_var = tk.BooleanVar(value=True)

        # Optics & Resolution
        aspect_ratio_var = tk.StringVar(value="1:1 (Square)")
        out_w_var = tk.IntVar(value=1024)
        out_h_var = tk.IntVar(value=1024)
        distortion_k1_var = tk.DoubleVar(value=0.0)

        # Video sampling
        video_mode_var = tk.StringVar(value="Target FPS")  # "Target FPS", "Frame Stride"
        video_fps_var = tk.DoubleVar(value=2.0)
        video_stride_var = tk.IntVar(value=15)
        video_max_frames_var = tk.IntVar(value=0)

        # Output format & manifest
        fmt_var = tk.StringVar(value="jpg")
        qual_var = tk.IntVar(value=95)
        export_tf_var = tk.BooleanVar(value=True)

        # Visualizer canvas mode
        canvas_view_mode_var = tk.StringVar(value="3d")  # "3d", "polar", "equirect"

        # 3D Orbit Viewport Camera State
        orbit_state = {
            "yaw": 35.0,
            "pitch": 22.0,
            "dist": 3.4,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "last_x": 0,
            "last_y": 0,
            "is_dragging": False,
            "drag_mode": "orbit",
            "auto_orbit": False,
            "after_id": None,
        }

        # Cached detected media info
        detected_media_info = {
            "is_video": False,
            "width": 5760,
            "height": 2880,
            "fps": 30.0,
            "frames": 0,
            "duration_sec": 0.0,
            "count": 0,
        }

        # -----------------------------------------------------------------
        # CARD 1: Input & Output (Left Column)
        # -----------------------------------------------------------------
        paths_card = tk.LabelFrame(
            rig_pages[0].body,
            text=" 1. Input 360° Source & Output Rig Folder ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=10,
            pady=6
        )
        paths_card.pack(fill=tk.X, pady=(0, 6))

        r1 = tk.Frame(paths_card, bg=self.panel_bg)
        r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="Input Source:", font=("Segoe UI", 8, "bold"), fg=self.text_primary, bg=self.panel_bg, width=11, anchor="w").pack(side=tk.LEFT)
        in_entry = tk.Entry(r1, textvariable=in_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, relief=tk.FLAT)
        in_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=2)

        def browse_in_folder():
            f = filedialog.askdirectory(initialdir=in_var.get(), title="Select Directory of 360° Panoramas")
            if f:
                in_var.set(f)
                out_var.set(os.path.join(f, "pinhole_rig"))
                inspect_input_source()

        def browse_in_video():
            f = filedialog.askopenfilename(
                initialdir=in_var.get(),
                title="Select 360° Panoramic Video File",
                filetypes=[("360 Video Files", "*.mp4 *.mov *.insv *.mkv *.avi"), ("All Files", "*.*")]
            )
            if f:
                in_var.set(f)
                base = os.path.splitext(f)[0]
                out_var.set(f"{base}_pinhole_rig")
                inspect_input_source()

        tk.Button(r1, text="📁 Folder...", font=("Segoe UI", 8), bg=self.accent_blue, fg="#ffffff", relief=tk.FLAT, padx=6, command=browse_in_folder).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(r1, text="🎬 Video...", font=("Segoe UI", 8), bg="#6366f1", fg="#ffffff", relief=tk.FLAT, padx=6, command=browse_in_video).pack(side=tk.LEFT)

        r2 = tk.Frame(paths_card, bg=self.panel_bg)
        r2.pack(fill=tk.X, pady=2)
        tk.Label(r2, text="Output Rig:", font=("Segoe UI", 8, "bold"), fg=self.text_primary, bg=self.panel_bg, width=11, anchor="w").pack(side=tk.LEFT)
        out_entry = tk.Entry(r2, textvariable=out_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, relief=tk.FLAT)
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=2)

        def browse_out_folder():
            f = filedialog.askdirectory(initialdir=out_var.get() or in_var.get(), title="Select Output Directory for Pinhole Views")
            if f:
                out_var.set(f)

        tk.Button(r2, text="Browse...", font=("Segoe UI", 8), bg=self.panel_border, fg="#ffffff", relief=tk.FLAT, padx=8, command=browse_out_folder).pack(side=tk.RIGHT)

        # Media Telemetry Badge
        media_badge_lbl = tk.Label(
            paths_card,
            text="Detected: Ready to inspect input source.",
            font=("Segoe UI", 8),
            fg="#38bdf8",
            bg=self.bg_dark,
            padx=6,
            pady=3,
            anchor="w",
            relief=tk.FLAT
        )
        media_badge_lbl.pack(fill=tk.X, pady=(3, 1))

        # -----------------------------------------------------------------
        # CARD 2: Rig Architecture & Auto-Math Solver (Left Column)
        # -----------------------------------------------------------------
        rig_card = tk.LabelFrame(
            rig_pages[1].body,
            text=" 2. Rig Architecture & Auto-Math Solver ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=10,
            pady=6
        )
        rig_card.pack(fill=tk.X, pady=(0, 6))

        # Mode row
        m_row = tk.Frame(rig_card, bg=self.panel_bg)
        m_row.pack(fill=tk.X, pady=2)
        tk.Label(m_row, text="Rig Architecture:", font=("Segoe UI", 8, "bold"), fg=self.text_primary, bg=self.panel_bg, width=14, anchor="w").pack(side=tk.LEFT)

        mode_options = [
            ("ring", "Horizontal Ring (Equatorial)"),
            ("cubemap", "6-Face Cubemap (90° FOV)"),
            ("dome", "Spherical Dome (Multi-Tier)"),
            ("fibonacci", "Fibonacci Sphere (Golden Spiral)"),
        ]
        mode_cb = ttk.Combobox(m_row, textvariable=mode_var, values=[k for k, _ in mode_options], state="readonly", width=14)
        mode_cb.pack(side=tk.LEFT, padx=(0, 10))

        # Solver Mode
        tk.Label(m_row, text="Auto-Math Solver:", font=("Segoe UI", 8, "bold"), fg="#a855f7", bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        solver_cb = ttk.Combobox(
            m_row,
            textvariable=solver_mode_var,
            values=["free", "lock_overlap", "lock_fov"],
            state="readonly",
            width=13
        )
        solver_cb.pack(side=tk.LEFT)

        # Tooltip for solver
        ToolTip(solver_cb, "Auto-Math Modes:\n- free: Adjust Cameras, FOV, and Overlap freely.\n- lock_overlap: Modifying Cameras automatically recalculates FOV to preserve exact overlap %.\n- lock_fov: Modifying Overlap calculates required minimum Camera Count.")

        # Parameters Slider Grid
        sliders_frame = tk.Frame(rig_card, bg=self.panel_bg)
        sliders_frame.pack(fill=tk.X, pady=(4, 2))

        # Row 1: Number of Cameras
        cams_row = tk.Frame(sliders_frame, bg=self.panel_bg)
        cams_row.pack(fill=tk.X, pady=1)
        cams_lbl = tk.Label(cams_row, text="Camera Count (N):", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, width=14, anchor="w")
        cams_lbl.pack(side=tk.LEFT)
        cams_scale = tk.Scale(
            cams_row,
            from_=3,
            to=32,
            orient=tk.HORIZONTAL,
            variable=num_cams_var,
            showvalue=False,
            bg=self.panel_bg,
            troughcolor=self.bg_dark,
            highlightthickness=0,
            activebackground="#a855f7"
        )
        cams_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        cams_val_lbl = tk.Label(cams_row, text="8 cams", font=("Segoe UI", 8, "bold"), fg="#a855f7", bg=self.panel_bg, width=8, anchor="e")
        cams_val_lbl.pack(side=tk.RIGHT)

        # Row 2: Horizontal Field of View (FOV)
        fov_row = tk.Frame(sliders_frame, bg=self.panel_bg)
        fov_row.pack(fill=tk.X, pady=1)
        fov_lbl = tk.Label(fov_row, text="Horizontal FOV (°):", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, width=14, anchor="w")
        fov_lbl.pack(side=tk.LEFT)
        fov_scale = tk.Scale(
            fov_row,
            from_=30.0,
            to=135.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=fov_var,
            showvalue=False,
            bg=self.panel_bg,
            troughcolor=self.bg_dark,
            highlightthickness=0,
            activebackground="#a855f7"
        )
        fov_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        fov_val_lbl = tk.Label(fov_row, text="90.0°", font=("Segoe UI", 8, "bold"), fg="#38bdf8", bg=self.panel_bg, width=8, anchor="e")
        fov_val_lbl.pack(side=tk.RIGHT)

        # Row 3: Target Overlap %
        ol_row = tk.Frame(sliders_frame, bg=self.panel_bg)
        ol_row.pack(fill=tk.X, pady=1)
        ol_lbl = tk.Label(ol_row, text="Adjacent Overlap (%):", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, width=14, anchor="w")
        ol_lbl.pack(side=tk.LEFT)
        ol_scale = tk.Scale(
            ol_row,
            from_=0.0,
            to=85.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=overlap_var,
            showvalue=False,
            bg=self.panel_bg,
            troughcolor=self.bg_dark,
            highlightthickness=0,
            activebackground="#10b981"
        )
        ol_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ol_val_lbl = tk.Label(ol_row, text="50.0%", font=("Segoe UI", 8, "bold"), fg="#10b981", bg=self.panel_bg, width=8, anchor="e")
        ol_val_lbl.pack(side=tk.RIGHT)

        # Dome extra tier options frame
        dome_frame = tk.Frame(rig_card, bg=self.panel_bg)
        dome_row1 = tk.Frame(dome_frame, bg=self.panel_bg)
        dome_row1.pack(fill=tk.X, pady=1)
        tk.Label(dome_row1, text="Pitch Tier (°):", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(dome_row1, textvariable=dome_pitch_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(dome_row1, text="Cams/Tier:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(dome_row1, textvariable=dome_tier_cams_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=4, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 12))

        tk.Checkbutton(dome_row1, text="Zenith (+90°)", variable=dome_zenith_var, font=("Segoe UI", 8), fg="#c084fc", bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(dome_row1, text="Nadir (-90°)", variable=dome_nadir_var, font=("Segoe UI", 8), fg="#c084fc", bg=self.panel_bg, selectcolor=self.bg_dark, activebackground=self.panel_bg).pack(side=tk.LEFT)

        # -----------------------------------------------------------------
        # CARD 3: Optics, Resolution & Video Sampling (Left Column)
        # -----------------------------------------------------------------
        optics_card = tk.LabelFrame(
            rig_pages[2].body,
            text=" 3. Camera Optics, Resolution & Video Sampling ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=10,
            pady=6
        )
        optics_card.pack(fill=tk.X, pady=(0, 6))

        # Resolution & Aspect Ratio Row
        o1 = tk.Frame(optics_card, bg=self.panel_bg)
        o1.pack(fill=tk.X, pady=2)
        tk.Label(o1, text="Aspect Ratio:", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        aspect_cb = ttk.Combobox(o1, textvariable=aspect_ratio_var, values=["1:1 (Square)", "4:3 Standard", "16:9 Widescreen", "Custom"], state="readonly", width=13)
        aspect_cb.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(o1, text="Size:", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 2))
        w_entry = tk.Entry(o1, textvariable=out_w_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT)
        w_entry.pack(side=tk.LEFT, padx=(0, 2), ipady=1)
        tk.Label(o1, text="×", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT, padx=1)
        h_entry = tk.Entry(o1, textvariable=out_h_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT)
        h_entry.pack(side=tk.LEFT, padx=(0, 8), ipady=1)

        auto_res_btn = tk.Button(
            o1,
            text="⚡ Auto-Res",
            font=("Segoe UI", 8, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=1
        )
        auto_res_btn.pack(side=tk.LEFT)
        ToolTip(auto_res_btn, "Auto-calculate optimal rectilinear resolution to match native equirectangular angular pixel density (PPD = W_erp / 2π).")

        # Distortion & Optics Row
        o2 = tk.Frame(optics_card, bg=self.panel_bg)
        o2.pack(fill=tk.X, pady=2)
        tk.Label(o2, text="Distortion (k1):", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, width=12, anchor="w").pack(side=tk.LEFT)
        dist_scale = tk.Scale(
            o2,
            from_=-0.20,
            to=0.20,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=distortion_k1_var,
            showvalue=False,
            bg=self.panel_bg,
            troughcolor=self.bg_dark,
            highlightthickness=0,
            activebackground="#38bdf8"
        )
        dist_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        dist_val_lbl = tk.Label(o2, text="0.00 (Pinhole)", font=("Segoe UI", 8), fg="#38bdf8", bg=self.panel_bg, width=11, anchor="e")
        dist_val_lbl.pack(side=tk.RIGHT)

        # Video Frame Extraction Row
        video_card_frame = tk.Frame(optics_card, bg=self.panel_bg)
        video_card_frame.pack(fill=tk.X, pady=(3, 1))

        tk.Label(video_card_frame, text="Video Sampling:", font=("Segoe UI", 8, "bold"), fg="#f59e0b", bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 4))
        vid_mode_cb = ttk.Combobox(video_card_frame, textvariable=video_mode_var, values=["Target FPS", "Frame Stride"], state="readonly", width=11)
        vid_mode_cb.pack(side=tk.LEFT, padx=(0, 6))

        fps_stride_val_lbl = tk.Label(video_card_frame, text="FPS:", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg)
        fps_stride_val_lbl.pack(side=tk.LEFT, padx=(0, 2))
        fps_entry = tk.Entry(video_card_frame, textvariable=video_fps_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT)
        fps_entry.pack(side=tk.LEFT, padx=(0, 8), ipady=1)

        tk.Label(video_card_frame, text="Max Frames:", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 2))
        max_f_entry = tk.Entry(video_card_frame, textvariable=video_max_frames_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=5, relief=tk.FLAT)
        max_f_entry.pack(side=tk.LEFT, padx=(0, 8), ipady=1)
        ToolTip(max_f_entry, "Maximum number of video frames to sample (0 = extract all frames across entire video duration).")

        # Format & transforms.json Row
        o3 = tk.Frame(optics_card, bg=self.panel_bg)
        o3.pack(fill=tk.X, pady=(4, 1))

        tk.Label(o3, text="Format:", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 3))
        fmt_cb = ttk.Combobox(o3, textvariable=fmt_var, values=["jpg", "png"], state="readonly", width=5)
        fmt_cb.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(o3, text="Quality:", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg).pack(side=tk.LEFT, padx=(0, 3))
        qual_entry = tk.Entry(o3, textvariable=qual_var, font=("Segoe UI", 8), bg=self.bg_dark, fg=self.text_primary, insertbackground=self.text_primary, width=4, relief=tk.FLAT)
        qual_entry.pack(side=tk.LEFT, padx=(0, 12), ipady=1)

        tf_chk = tk.Checkbutton(
            o3,
            text="Export transforms.json (COLMAP / Nerfstudio)",
            variable=export_tf_var,
            font=("Segoe UI", 8, "bold"),
            fg="#a855f7",
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        tf_chk.pack(side=tk.LEFT)

        # -----------------------------------------------------------------
        # RIGHT COLUMN: Interactive Visualizer Canvas & Telemetry Dashboard
        # -----------------------------------------------------------------
        viz_hdr = tk.Frame(right_col, bg=self.panel_bg, padx=10, pady=6)
        viz_hdr.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            viz_hdr,
            text="🎯 RIG PLACEMENT & SPHERICAL COVERAGE",
            font=("Segoe UI", 9, "bold"),
            fg="#38bdf8",
            bg=self.panel_bg
        ).pack(side=tk.LEFT)

        # View Mode Toggle
        vm_frame = tk.Frame(viz_hdr, bg=self.panel_bg)
        vm_frame.pack(side=tk.RIGHT)

        orbit3d_btn = tk.Radiobutton(
            vm_frame,
            text="🌐 3D Orbit Rig",
            variable=canvas_view_mode_var,
            value="3d",
            command=lambda: draw_visualizer(),
            font=("Segoe UI", 8, "bold"),
            fg="#38bdf8",
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        orbit3d_btn.pack(side=tk.LEFT, padx=(0, 4))

        polar_btn = tk.Radiobutton(
            vm_frame,
            text="🧭 Polar Radar",
            variable=canvas_view_mode_var,
            value="polar",
            command=lambda: draw_visualizer(),
            font=("Segoe UI", 8, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        polar_btn.pack(side=tk.LEFT, padx=(0, 4))

        equirect_btn = tk.Radiobutton(
            vm_frame,
            text="🗺️ 2D Footprint",
            variable=canvas_view_mode_var,
            value="equirect",
            command=lambda: draw_visualizer(),
            font=("Segoe UI", 8, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            selectcolor=self.bg_dark,
            activebackground=self.panel_bg
        )
        equirect_btn.pack(side=tk.LEFT, padx=(0, 6))

        def reset_3d_view():
            orbit_state["yaw"] = 35.0
            orbit_state["pitch"] = 22.0
            orbit_state["dist"] = 3.4
            orbit_state["pan_x"] = 0.0
            orbit_state["pan_y"] = 0.0
            canvas_view_mode_var.set("3d")
            draw_visualizer()

        reset_3d_btn = tk.Button(
            vm_frame,
            text="🔄 Reset 3D",
            font=("Segoe UI", 7, "bold"),
            bg=self.panel_border,
            fg="#e2e8f0",
            relief=tk.FLAT,
            padx=5,
            pady=1,
            command=reset_3d_view
        )
        reset_3d_btn.pack(side=tk.LEFT, padx=(0, 3))
        ToolTip(reset_3d_btn, "Reset 3D camera to default azimuth (35°), elevation (22°), and zoom.")

        def run_auto_orbit():
            if not orbit_state["auto_orbit"]:
                return
            try:
                if not dialog.winfo_exists():
                    return
            except Exception:
                return
            if canvas_view_mode_var.get() == "3d":
                orbit_state["yaw"] = (orbit_state["yaw"] + 0.75) % 360.0
                draw_visualizer()
            orbit_state["after_id"] = dialog.after(33, run_auto_orbit)

        def toggle_auto_orbit():
            orbit_state["auto_orbit"] = not orbit_state["auto_orbit"]
            if orbit_state["auto_orbit"]:
                auto_spin_btn.config(bg="#24483f", fg="#70dbc0", text="Stop rotation")
                if canvas_view_mode_var.get() != "3d":
                    canvas_view_mode_var.set("3d")
                run_auto_orbit()
            else:
                auto_spin_btn.config(bg="#242d32", fg="#e7edef", text="Rotate")
                if orbit_state["after_id"]:
                    try:
                        dialog.after_cancel(orbit_state["after_id"])
                    except Exception:
                        pass
                    orbit_state["after_id"] = None

        auto_spin_btn = tk.Button(
            vm_frame,
            text="Rotate",
            font=("Segoe UI", 7, "bold"),
            bg="#312e81",
            fg="#c7d2fe",
            relief=tk.FLAT,
            padx=5,
            pady=1,
            command=toggle_auto_orbit
        )
        auto_spin_btn.pack(side=tk.LEFT)
        ToolTip(auto_spin_btn, "Toggle continuous 3D turntable orbit rotation.")

        def on_close_dialog():
            if orbit_state.get("exporting", False):
                messagebox.showinfo("Export in progress", "Wait for the export to finish before closing this window.", parent=dialog)
                return
            orbit_state["auto_orbit"] = False
            if orbit_state["after_id"]:
                try:
                    dialog.after_cancel(orbit_state["after_id"])
                except Exception:
                    pass
                orbit_state["after_id"] = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close_dialog)

        # Canvas Frame
        canvas_card = tk.Frame(right_col, bg="#0d0d16", highlightthickness=1, highlightbackground=self.panel_border)
        canvas_card.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        viz_canvas = tk.Canvas(
            canvas_card,
            bg="#0d0d16",
            highlightthickness=0,
            width=520,
            height=370
        )
        viz_canvas.pack(fill=tk.BOTH, expand=True)
        from rig_viewport import RigSceneRenderer
        renderer = RigSceneRenderer(viz_canvas)

        # Mouse Interaction Handlers for 3D Orbit Viewport
        def on_canvas_b1_down(evt):
            if canvas_view_mode_var.get() == "3d":
                viz_canvas.focus_set()
                orbit_state["press_x"], orbit_state["press_y"] = evt.x, evt.y
                orbit_state["is_dragging"] = True
                orbit_state["last_x"] = evt.x
                orbit_state["last_y"] = evt.y
                orbit_state["drag_mode"] = "pan" if (evt.state & 0x0001) else "orbit"

        def on_canvas_b1_motion(evt):
            if canvas_view_mode_var.get() == "3d" and orbit_state["is_dragging"]:
                dx = evt.x - orbit_state["last_x"]
                dy = evt.y - orbit_state["last_y"]
                orbit_state["last_x"] = evt.x
                orbit_state["last_y"] = evt.y
                if orbit_state["drag_mode"] == "pan":
                    orbit_state["pan_x"] += dx
                    orbit_state["pan_y"] += dy
                else:
                    orbit_state["yaw"] = (orbit_state["yaw"] + dx * 0.5) % 360.0
                    orbit_state["pitch"] = max(-88.0, min(88.0, orbit_state["pitch"] - dy * 0.5))
                draw_visualizer()

        def on_canvas_b1_up(evt):
            if (canvas_view_mode_var.get() == "3d" and orbit_state["drag_mode"] == "orbit"
                    and math.hypot(evt.x - orbit_state.get("press_x", -999),
                                   evt.y - orbit_state.get("press_y", -999)) < 5):
                renderer.pick(evt.x, evt.y)
                draw_visualizer()
            orbit_state["is_dragging"] = False

        def on_canvas_b3_down(evt):
            if canvas_view_mode_var.get() == "3d":
                orbit_state["is_dragging"] = True
                orbit_state["drag_mode"] = "pan"
                orbit_state["last_x"] = evt.x
                orbit_state["last_y"] = evt.y

        def on_canvas_b3_motion(evt):
            if canvas_view_mode_var.get() == "3d" and orbit_state["is_dragging"]:
                dx = evt.x - orbit_state["last_x"]
                dy = evt.y - orbit_state["last_y"]
                orbit_state["last_x"] = evt.x
                orbit_state["last_y"] = evt.y
                orbit_state["pan_x"] += dx
                orbit_state["pan_y"] += dy
                draw_visualizer()

        def on_canvas_mousewheel(evt):
            if canvas_view_mode_var.get() == "3d":
                delta = evt.delta / 120.0 if evt.delta != 0 else 0.0
                orbit_state["dist"] = max(1.2, min(9.0, orbit_state["dist"] - delta * 0.3))
                draw_visualizer()

        def on_canvas_zoom_in(evt):
            if canvas_view_mode_var.get() == "3d":
                orbit_state["dist"] = max(1.2, orbit_state["dist"] - 0.3)
                draw_visualizer()

        def on_canvas_zoom_out(evt):
            if canvas_view_mode_var.get() == "3d":
                orbit_state["dist"] = min(9.0, orbit_state["dist"] + 0.3)
                draw_visualizer()

        def on_canvas_double_click(evt):
            if canvas_view_mode_var.get() == "3d":
                reset_3d_view()

        def on_canvas_enter(evt):
            if canvas_view_mode_var.get() == "3d":
                viz_canvas.config(cursor="fleur")

        def on_canvas_leave(evt):
            viz_canvas.config(cursor="")

        viz_canvas.bind("<ButtonPress-1>", on_canvas_b1_down)
        viz_canvas.bind("<B1-Motion>", on_canvas_b1_motion)
        viz_canvas.bind("<ButtonRelease-1>", on_canvas_b1_up)
        viz_canvas.bind("<ButtonPress-3>", on_canvas_b3_down)
        viz_canvas.bind("<B3-Motion>", on_canvas_b3_motion)
        viz_canvas.bind("<ButtonRelease-3>", on_canvas_b1_up)
        viz_canvas.bind("<MouseWheel>", on_canvas_mousewheel)
        viz_canvas.bind("<Button-4>", on_canvas_zoom_in)
        viz_canvas.bind("<Button-5>", on_canvas_zoom_out)
        viz_canvas.bind("<Double-Button-1>", on_canvas_double_click)
        viz_canvas.bind("<Enter>", on_canvas_enter)
        viz_canvas.bind("<Leave>", on_canvas_leave)

        # Telemetry & Optics Dashboard (Below Canvas)
        telemetry_card = tk.LabelFrame(
            right_col,
            text=" Mathematical Telemetry & Optics Solvers ",
            font=("Segoe UI", 9, "bold"),
            fg=self.text_primary,
            bg=self.panel_bg,
            padx=10,
            pady=6
        )
        telemetry_card.pack(fill=tk.X)

        t_grid = tk.Frame(telemetry_card, bg=self.panel_bg)
        t_grid.pack(fill=tk.X)

        # Left Column Stats
        t_c1 = tk.Frame(t_grid, bg=self.panel_bg)
        t_c1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        step_telemetry_lbl = tk.Label(t_c1, text="Angular Step (Δθ): 45.0°", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, anchor="w")
        step_telemetry_lbl.pack(fill=tk.X, pady=1)

        overlap_telemetry_lbl = tk.Label(t_c1, text="Rig Overlap: 50.0% [Optimal]", font=("Segoe UI", 8, "bold"), fg="#10b981", bg=self.panel_bg, anchor="w")
        overlap_telemetry_lbl.pack(fill=tk.X, pady=1)

        focal_telemetry_lbl = tk.Label(t_c1, text="Focal Length: fx=512.0px, fy=512.0px", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, anchor="w")
        focal_telemetry_lbl.pack(fill=tk.X, pady=1)

        # Right Column Stats
        t_c2 = tk.Frame(t_grid, bg=self.panel_bg)
        t_c2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        views_telemetry_lbl = tk.Label(t_c2, text="Rig Views: 8 views/pano", font=("Segoe UI", 8), fg=self.text_primary, bg=self.panel_bg, anchor="w")
        views_telemetry_lbl.pack(fill=tk.X, pady=1)

        total_telemetry_lbl = tk.Label(t_c2, text="Est. Total Views: 0 images", font=("Segoe UI", 8), fg="#38bdf8", bg=self.panel_bg, anchor="w")
        total_telemetry_lbl.pack(fill=tk.X, pady=1)

        coverage_telemetry_lbl = tk.Label(t_c2, text="Coverage: Equatorial Belt (±45°)", font=("Segoe UI", 8), fg="#a855f7", bg=self.panel_bg, anchor="w")
        coverage_telemetry_lbl.pack(fill=tk.X, pady=1)

        # -----------------------------------------------------------------
        # BOTTOM PROGRESS & ACTIONS BAR
        # -----------------------------------------------------------------
        bottom_bar = tk.Frame(dialog, bg=self.panel_bg, padx=12, pady=8)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        prog_row = tk.Frame(bottom_bar, bg=self.panel_bg)
        prog_row.pack(fill=tk.X, pady=(0, 6))

        dlg_pbar = ttk.Progressbar(prog_row, mode="determinate")
        dlg_pbar.pack(fill=tk.X, pady=(0, 2))

        status_lbl = tk.Label(prog_row, text="Status: Ready to project multi-camera rig.", font=("Segoe UI", 8), fg=self.text_secondary, bg=self.panel_bg, anchor="w")
        status_lbl.pack(fill=tk.X)

        btn_row = tk.Frame(bottom_bar, bg=self.panel_bg)
        btn_row.pack(fill=tk.X)

        split_btn = tk.Button(
            btn_row,
            text="Export views",
            font=("Segoe UI", 9, "bold"),
            bg="#7c3aed",
            fg="#ffffff",
            activebackground="#6d28d9",
            relief=tk.FLAT,
            padx=16,
            pady=6
        )
        split_btn.pack(side=tk.LEFT, padx=(0, 8))

        use_ds_btn = tk.Button(
            btn_row,
            text="✓ Set as Active Dataset in Abu Align",
            font=("Segoe UI", 9),
            bg="#29293d",
            fg=self.text_primary,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        use_ds_btn.pack(side=tk.LEFT, padx=(0, 8))

        open_folder_btn = tk.Button(
            btn_row,
            text="📁 Open Output Folder",
            font=("Segoe UI", 9),
            bg="#29293d",
            fg=self.text_primary,
            relief=tk.FLAT,
            padx=12,
            pady=6,
            state=tk.DISABLED
        )
        open_folder_btn.pack(side=tk.LEFT)

        close_btn = tk.Button(
            btn_row,
            text="Close",
            font=("Segoe UI", 9),
            bg="#1f1f2e",
            fg=self.text_secondary,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            command=on_close_dialog
        )
        close_btn.pack(side=tk.RIGHT)

        # -----------------------------------------------------------------
        # Real-time Mathematical Calculation & Canvas Synchronization
        # -----------------------------------------------------------------
        is_updating_math = False
        last_rig_mode = [mode_var.get()]

        def get_current_rig():
            """Returns the list of (name, yaw, pitch, roll, fov) for current settings."""
            m = mode_var.get()
            fov_v = fov_var.get()
            nc = max(3, num_cams_var.get())

            if m == "cubemap":
                return ps.get_cubemap_rig(fov_deg=fov_v)
            elif m == "dome":
                return ps.get_spherical_dome_rig(
                    equator_cams=nc,
                    pitch_tier=dome_pitch_var.get(),
                    tier_cams=dome_tier_cams_var.get(),
                    include_zenith=dome_zenith_var.get(),
                    include_nadir=dome_nadir_var.get(),
                    fov_deg=fov_v
                )
            elif m == "fibonacci":
                return ps.get_fibonacci_sphere_rig(num_cameras=nc, fov_deg=fov_v)
            else:  # "ring"
                return ps.get_horizontal_ring_rig(num_cameras=nc, fov_deg=fov_v)

        def sync_math_and_ui(*args):
            nonlocal is_updating_math
            if is_updating_math:
                return
            is_updating_math = True

            try:
                m = mode_var.get()
                solver = solver_mode_var.get()
                nc = max(3, num_cams_var.get())
                fov_v = max(10.0, min(150.0, fov_var.get()))
                ol_v = max(0.0, min(85.0, overlap_var.get()))

                # Visibility of mode-specific widgets
                if m == "dome":
                    dome_frame.pack(fill=tk.X, pady=(4, 2))
                    cams_lbl.config(text="Equator Cams (N):")
                    cams_scale.config(state=tk.NORMAL)
                elif m == "cubemap":
                    dome_frame.pack_forget()
                    cams_lbl.config(text="Cube Faces (Fixed):")
                    cams_scale.config(state=tk.DISABLED)
                    num_cams_var.set(6)
                    if last_rig_mode[0] != "cubemap" and (fov_v < 60.0 or fov_v > 135.0):
                        fov_var.set(90.0)
                        fov_v = 90.0
                elif m == "fibonacci":
                    dome_frame.pack_forget()
                    cams_lbl.config(text="Sphere Cams (N):")
                    cams_scale.config(state=tk.NORMAL)
                else:  # ring
                    dome_frame.pack_forget()
                    cams_lbl.config(text="Camera Count (N):")
                    cams_scale.config(state=tk.NORMAL)

                last_rig_mode[0] = m

                # Solver logic
                if m in ("ring", "dome"):
                    if solver == "lock_overlap":
                        # Recalculate FOV from locked overlap and current camera count
                        new_fov = ps.calculate_fov_from_overlap(nc, ol_v)
                        fov_var.set(round(new_fov, 1))
                        fov_v = new_fov
                    elif solver == "lock_fov":
                        # Recalculate camera count from locked FOV and target overlap
                        req_nc = ps.calculate_cameras_from_fov_and_overlap(fov_v, ol_v)
                        num_cams_var.set(req_nc)
                        nc = req_nc
                    else:
                        # Free mode: calculate actual overlap from N and FOV
                        act_ol = ps.calculate_overlap(nc, fov_v)
                        overlap_var.set(round(act_ol, 1))
                        ol_v = act_ol
                elif m == "cubemap":
                    # Cubemap faces are orthogonal (angular step Δθ = 90°)
                    # Adjacent faces overlap when FOV > 90°: Overlap = max(0, (FOV - 90) / FOV * 100)
                    if solver == "lock_overlap":
                        safe_ol = min(ol_v, 45.0)
                        new_fov = 90.0 / max(0.1, (1.0 - safe_ol / 100.0))
                        fov_var.set(round(new_fov, 1))
                        fov_v = new_fov
                    else:
                        act_ol = max(0.0, (fov_v - 90.0) / fov_v * 100.0) if fov_v >= 90.0 else 0.0
                        overlap_var.set(round(act_ol, 1))
                        ol_v = act_ol

                # Value labels
                cams_val_lbl.config(text=f"{num_cams_var.get()} cams")
                fov_val_lbl.config(text=f"{fov_var.get():.1f}°")
                ol_val_lbl.config(text=f"{overlap_var.get():.1f}%")

                k1_val = distortion_k1_var.get()
                if abs(k1_val) < 1e-4:
                    dist_val_lbl.config(text="0.00 (Pinhole)", fg="#38bdf8")
                    if hasattr(renderer, "optics_text"):
                        renderer.optics_text = "ideal pinhole optics"
                else:
                    dist_val_lbl.config(text=f"{k1_val:+.2f}", fg="#f59e0b")
                    if hasattr(renderer, "optics_text"):
                        dist_type = "barrel" if k1_val < 0 else "pincushion"
                        renderer.optics_text = f"radial {dist_type} distortion (k1={k1_val:+.2f})"

                # Aspect ratio & resolution
                ar_mode = aspect_ratio_var.get()
                w_cur = out_w_var.get()
                if "1:1" in ar_mode:
                    out_h_var.set(w_cur)
                elif "4:3" in ar_mode:
                    out_h_var.set(max(128, int(round(w_cur * 3.0 / 4.0))))
                elif "16:9" in ar_mode:
                    out_h_var.set(max(128, int(round(w_cur * 9.0 / 16.0))))

                # Calculate focal lengths and vertical FOV
                w = out_w_var.get()
                h = out_h_var.get()
                half_fov_rad = math.radians(fov_v) / 2.0
                fx = (float(w) / 2.0) / math.tan(half_fov_rad)
                fov_v_deg = ps.calculate_vertical_fov(fov_v, w, h)

                # Telemetry updates
                step_deg = ps.calculate_angular_step(nc) if m in ("ring", "dome") else (90.0 if m == "cubemap" else (360.0 / nc))
                step_telemetry_lbl.config(text=f"Angular Step (Δθ): {step_deg:.2f}°")

                rig_cams = get_current_rig()
                views_telemetry_lbl.config(text=f"Rig Views: {len(rig_cams)} views/pano")

                if ol_v >= 40.0 and ol_v <= 65.0:
                    overlap_telemetry_lbl.config(text=f"Rig Overlap: {ol_v:.1f}% [Optimal for SfM]", fg="#10b981")
                elif ol_v > 65.0:
                    overlap_telemetry_lbl.config(text=f"Rig Overlap: {ol_v:.1f}% [Dense Overlap]", fg="#38bdf8")
                elif ol_v >= 20.0:
                    overlap_telemetry_lbl.config(text=f"Rig Overlap: {ol_v:.1f}% [Moderate]", fg="#f59e0b")
                else:
                    overlap_telemetry_lbl.config(text=f"Rig Overlap: {ol_v:.1f}% [Gap Alert!]", fg="#ef4444")

                focal_telemetry_lbl.config(text=f"Focal Length: fx={fx:.1f}px, fy={fx:.1f}px (vFOV: {fov_v_deg:.1f}°)")

                if m == "fibonacci":
                    coverage_telemetry_lbl.config(text="Coverage: 4π Full Sphere (Uniform Golden Spiral)")
                elif m == "dome":
                    coverage_telemetry_lbl.config(text="Coverage: 4π Full Sphere (Multi-Tier & Poles)")
                elif m == "cubemap":
                    coverage_telemetry_lbl.config(text="Coverage: 4π Full Sphere (6 Orthogonal Faces)")
                else:
                    coverage_telemetry_lbl.config(text=f"Coverage: Equatorial Belt (±{fov_v_deg/2.0:.0f}° Lat)")

                # Estimate total views
                source_count = detected_media_info["count"]
                if detected_media_info["is_video"] and detected_media_info["fps"] > 0:
                    if video_mode_var.get() == "Target FPS":
                        fps_tgt = video_fps_var.get()
                        est_frames = int(round(detected_media_info["duration_sec"] * fps_tgt))
                    else:
                        stride = max(1, video_stride_var.get())
                        est_frames = detected_media_info["frames"] // stride
                    if video_max_frames_var.get() > 0:
                        est_frames = min(est_frames, video_max_frames_var.get())
                    source_count = est_frames

                total_est = source_count * len(rig_cams)
                total_telemetry_lbl.config(text=f"Est. Total Views: {total_est:,} pinhole images ({source_count} panos/frames)")

            except Exception:
                pass
            finally:
                is_updating_math = False

            draw_visualizer()

        # Canvas Renderer
        def draw_visualizer(*_):
            viz_canvas.delete("all")
            cw = viz_canvas.winfo_width()
            ch = viz_canvas.winfo_height()
            if cw <= 10 or ch <= 10:
                cw = 520
                ch = 370

            try:
                rig = get_current_rig()
            except (tk.TclError, ValueError, ZeroDivisionError):
                viz_canvas.create_text(24, 30, anchor="nw", text="Enter valid rig settings to update the preview.",
                                       fill=self.text_secondary, font=("Segoe UI", 10))
                return
            vmode = canvas_view_mode_var.get()

            if vmode == "polar":
                # ---------------------------------------------------------
                # Polar Compass / Radar View
                # ---------------------------------------------------------
                cx = cw / 2.0
                cy = ch / 2.0
                max_r = min(cx, cy) - 36.0

                # Concentric range rings (30°, 60°, 90°)
                for frac, deg_lbl in [(0.333, "30°"), (0.666, "60°"), (1.0, "90°")]:
                    r_i = max_r * frac
                    viz_canvas.create_oval(
                        cx - r_i, cy - r_i, cx + r_i, cy + r_i,
                        outline="#222233" if frac < 1.0 else "#36364d",
                        width=1,
                        dash=(2, 4) if frac < 1.0 else ()
                    )
                    viz_canvas.create_text(
                        cx + r_i + 2, cy - 6,
                        text=deg_lbl,
                        fill="#55556d",
                        font=("Segoe UI", 7),
                        anchor="w"
                    )

                # Cardinal Crosshairs
                viz_canvas.create_line(cx, cy - max_r - 18, cx, cy + max_r + 18, fill="#222233", width=1)
                viz_canvas.create_line(cx - max_r - 18, cy, cx + max_r + 18, cy, fill="#222233", width=1)

                # Cardinal Labels
                viz_canvas.create_text(cx, cy - max_r - 14, text="N (0°)", fill="#a0adb5", font=("Segoe UI", 8, "bold"))
                viz_canvas.create_text(cx + max_r + 16, cy, text="E (+90°)", fill="#a0adb5", font=("Segoe UI", 8, "bold"), anchor="w")
                viz_canvas.create_text(cx, cy + max_r + 14, text="S (180°)", fill="#a0adb5", font=("Segoe UI", 8, "bold"))
                viz_canvas.create_text(cx - max_r - 16, cy, text="W (-90°)", fill="#a0adb5", font=("Segoe UI", 8, "bold"), anchor="e")

                # Color palette for overlap
                ol = overlap_var.get()
                if ol >= 40.0 and ol <= 65.0:
                    wedge_color = "#10b981"  # Optimal green
                    wedge_stipple = "gray25"
                elif ol > 65.0:
                    wedge_color = "#70dbc0"  # Blue
                    wedge_stipple = "gray25"
                elif ol >= 20.0:
                    wedge_color = "#f59e0b"  # Amber
                    wedge_stipple = "gray25"
                else:
                    wedge_color = "#ef4444"  # Red
                    wedge_stipple = "gray25"

                # Draw FOV cones and direction vectors
                for idx, (cam_name, yaw_d, pitch_d, roll_d, fov_d) in enumerate(rig):
                    # In screen space, North (yaw=0) is straight UP (-Y axis)
                    # Yaw increases clockwise: +90° is East (+X axis)
                    yaw_rad = math.radians(yaw_d)
                    half_fov = math.radians(fov_d) / 2.0

                    # Adjust radial distance by pitch (projected to equatorial plane: r = max_r * cos(pitch))
                    pitch_rad = math.radians(pitch_d)
                    eff_r = max_r * max(0.15, math.cos(pitch_rad))

                    a1 = yaw_rad - half_fov
                    a2 = yaw_rad + half_fov

                    p0 = (cx, cy)
                    p1 = (cx + eff_r * math.sin(a1), cy - eff_r * math.cos(a1))
                    p2 = (cx + eff_r * math.sin(a2), cy - eff_r * math.cos(a2))

                    # Center direction ray
                    p_mid = (cx + eff_r * math.sin(yaw_rad), cy - eff_r * math.cos(yaw_rad))

                    # Draw FOV wedge polygon
                    wedge_pts = [p0[0], p0[1], p1[0], p1[1]]
                    # Interpolate arc
                    num_arc_steps = 6
                    for s in range(1, num_arc_steps):
                        t = s / num_arc_steps
                        a_t = a1 + t * (a2 - a1)
                        wedge_pts.extend([cx + eff_r * math.sin(a_t), cy - eff_r * math.cos(a_t)])
                    wedge_pts.extend([p2[0], p2[1]])

                    viz_canvas.create_polygon(
                        wedge_pts,
                        fill=wedge_color,
                        stipple=wedge_stipple,
                        outline="#4f46e5" if abs(pitch_d) < 1e-3 else "#c084fc",
                        width=1
                    )

                    # Center ray arrow
                    viz_canvas.create_line(
                        p0[0], p0[1], p_mid[0], p_mid[1],
                        fill="#ffffff" if abs(pitch_d) < 1e-3 else "#e9d5ff",
                        arrow=tk.LAST,
                        arrowshape=(6, 8, 3),
                        width=1.5
                    )

                    # Camera Label
                    lbl_x = cx + (eff_r + 14) * math.sin(yaw_rad)
                    lbl_y = cy - (eff_r + 14) * math.cos(yaw_rad)
                    tag_short = f"C{idx}" if len(rig) > 6 else cam_name.split("_")[0]
                    viz_canvas.create_text(
                        lbl_x, lbl_y,
                        text=tag_short,
                        fill="#ffffff",
                        font=("Segoe UI", 7, "bold")
                    )

                # Center Rig Origin Sphere
                viz_canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill="#a855f7", outline="#ffffff", width=2)

            elif vmode == "equirect":
                # ---------------------------------------------------------
                # Equirectangular 2D Footprint Unroll View
                # ---------------------------------------------------------
                margin_x = 44.0
                margin_y = 35.0
                box_w = cw - 2 * margin_x
                box_h = ch - 2 * margin_y

                # Background 2:1 map border
                viz_canvas.create_rectangle(
                    margin_x, margin_y, margin_x + box_w, margin_y + box_h,
                    fill="#151522",
                    outline="#38384f",
                    width=1.5
                )

                # Equator line (pitch 0°)
                y_eq = margin_y + box_h / 2.0
                viz_canvas.create_line(margin_x, y_eq, margin_x + box_w, y_eq, fill="#4f46e5", width=1.5)

                # Latitude lines (+60°, +30°, -30°, -60°)
                for p_deg in [60, 30, -30, -60]:
                    y_lat = margin_y + ((90.0 - p_deg) / 180.0) * box_h
                    viz_canvas.create_line(margin_x, y_lat, margin_x + box_w, y_lat, fill="#232336", width=1, dash=(2, 4))
                    viz_canvas.create_text(margin_x - 3, y_lat, text=f"{p_deg:+d}°", fill="#64748b", font=("Segoe UI", 7), anchor="e")

                # Longitude lines (-180, -90, 0, +90, +180)
                for y_deg in [-180, -90, 0, 90, 180]:
                    x_lon = margin_x + ((y_deg + 180.0) / 360.0) * box_w
                    viz_canvas.create_line(x_lon, margin_y, x_lon, margin_y + box_h, fill="#4f46e5" if y_deg == 0 else "#232336", width=1.5 if y_deg == 0 else 1)
                    viz_canvas.create_text(x_lon, margin_y + box_h + 10, text=f"{y_deg:+d}°", fill="#64748b", font=("Segoe UI", 7))

                # Labels for poles
                viz_canvas.create_text(margin_x - 3, margin_y, text="+90°", fill="#64748b", font=("Segoe UI", 7), anchor="e")
                viz_canvas.create_text(margin_x - 3, margin_y + box_h, text="-90°", fill="#64748b", font=("Segoe UI", 7), anchor="e")

                # Render each camera's rectilinear footprint box on the ERP
                out_w = out_w_var.get()
                out_h = out_h_var.get()

                for idx, (cam_name, yaw_d, pitch_d, roll_d, fov_d) in enumerate(rig):
                    fov_v_d = ps.calculate_vertical_fov(fov_d, out_w, out_h)
                    pw = (fov_d / 360.0) * box_w
                    ph = (fov_v_d / 180.0) * box_h

                    cx_cam = margin_x + ((yaw_d + 180.0) / 360.0) * box_w
                    cy_cam = margin_y + ((90.0 - pitch_d) / 180.0) * box_h

                    # Draw camera box
                    bx0 = cx_cam - pw / 2.0
                    by0 = cy_cam - ph / 2.0
                    bx1 = cx_cam + pw / 2.0
                    by1 = cy_cam + ph / 2.0

                    # Clamp vertical bounds
                    by0 = max(margin_y, by0)
                    by1 = min(margin_y + box_h, by1)

                    # Handle wrap around left/right boundaries
                    boxes_to_draw = []
                    if bx0 < margin_x:
                        boxes_to_draw.append((margin_x, by0, bx1, by1))
                        boxes_to_draw.append((margin_x + box_w - (margin_x - bx0), by0, margin_x + box_w, by1))
                    elif bx1 > margin_x + box_w:
                        boxes_to_draw.append((bx0, by0, margin_x + box_w, by1))
                        boxes_to_draw.append((margin_x, by0, margin_x + (bx1 - (margin_x + box_w)), by1))
                    else:
                        boxes_to_draw.append((bx0, by0, bx1, by1))

                    for x0_i, y0_i, x1_i, y1_i in boxes_to_draw:
                        viz_canvas.create_rectangle(
                            x0_i, y0_i, x1_i, y1_i,
                            fill="#7c3aed",
                            stipple="gray25",
                            outline="#38bdf8",
                            width=1
                        )

                    # Crosshair and label at center
                    viz_canvas.create_line(cx_cam - 4, cy_cam, cx_cam + 4, cy_cam, fill="#ffffff", width=1)
                    viz_canvas.create_line(cx_cam, cy_cam - 4, cx_cam, cy_cam + 4, fill="#ffffff", width=1)
                    tag_short = f"C{idx}" if len(rig) > 6 else cam_name.split("_")[0]
                    viz_canvas.create_text(cx_cam, cy_cam - 9, text=tag_short, fill="#ffffff", font=("Segoe UI", 7, "bold"))

            else:
                try:
                    renderer.render(rig, orbit_state, out_w_var.get() / max(1, out_h_var.get()))
                except (tk.TclError, ValueError, ZeroDivisionError):
                    viz_canvas.create_text(24, 30, anchor="nw", text="Enter a valid image size to preview the rig.",
                                           fill=self.text_secondary, font=("Segoe UI", 10))

        # Bind canvas resize to redraw
        viz_canvas.bind("<Configure>", draw_visualizer)

        # -----------------------------------------------------------------
        # Media Source Inspection & Auto-Res
        # -----------------------------------------------------------------
        def inspect_input_source(*_):
            p = in_var.get().strip()
            if not p or not os.path.exists(p):
                media_badge_lbl.config(text="Status: Input path not found.", fg="#f87171")
                return

            video_exts = {".mp4", ".mov", ".insv", ".mkv", ".avi"}
            if os.path.isfile(p) and os.path.splitext(p.lower())[1] in video_exts:
                # Video file
                try:
                    meta = ps.probe_video_metadata(p)
                    detected_media_info["is_video"] = True
                    detected_media_info["width"] = meta["width"]
                    detected_media_info["height"] = meta["height"]
                    detected_media_info["fps"] = meta["fps"]
                    detected_media_info["frames"] = meta["frame_count"]
                    detected_media_info["duration_sec"] = meta["duration_sec"]
                    detected_media_info["count"] = meta["frame_count"]

                    min_s = int(meta["duration_sec"] // 60)
                    sec_s = int(meta["duration_sec"] % 60)
                    media_badge_lbl.config(
                        text=f"🎬 360° Video: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} FPS, {meta['frame_count']} frames ({min_s}m {sec_s}s)",
                        fg="#38bdf8"
                    )
                except Exception as ex:
                    media_badge_lbl.config(text=f"🎬 Video detected (probe error: {ex})", fg="#f59e0b")
            elif os.path.isdir(p):
                # Directory of images
                valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
                files = [f for f in os.listdir(p) if os.path.splitext(f.lower())[1] in valid_exts]
                count = len(files)
                detected_media_info["is_video"] = False
                detected_media_info["count"] = count

                if count > 0:
                    try:
                        import cv2
                        s_img = cv2.imread(os.path.join(p, files[0]))
                        if s_img is not None:
                            h, w = s_img.shape[:2]
                            detected_media_info["width"] = w
                            detected_media_info["height"] = h
                            media_badge_lbl.config(
                                text=f"📁 Panorama Directory: {count} images ({w}x{h}, 2:1 ERP)",
                                fg="#10b981"
                            )
                        else:
                            media_badge_lbl.config(text=f"📁 Panorama Directory: {count} images", fg="#10b981")
                    except Exception:
                        media_badge_lbl.config(text=f"📁 Panorama Directory: {count} images", fg="#10b981")
                else:
                    media_badge_lbl.config(text="📁 Empty directory (no 360° images found)", fg="#f87171")
            else:
                media_badge_lbl.config(text=f"📄 Single Panorama File ({os.path.basename(p)})", fg="#10b981")

            sync_math_and_ui()

        def on_auto_res():
            """Solves optimal rectilinear resolution from input equirectangular width and FOV."""
            w_erp = detected_media_info["width"] or 5760
            fov_v = fov_var.get()
            opt_w = ps.calculate_optimal_resolution(w_erp, fov_v)
            out_w_var.set(opt_w)

            ar_mode = aspect_ratio_var.get()
            if "1:1" in ar_mode:
                out_h_var.set(opt_w)
            elif "4:3" in ar_mode:
                out_h_var.set(max(128, int(round(opt_w * 3.0 / 4.0))))
            elif "16:9" in ar_mode:
                out_h_var.set(max(128, int(round(opt_w * 9.0 / 16.0))))
            sync_math_and_ui()
            messagebox.showinfo("Auto-Resolution Solved", f"Matched Native Angular Pixel Density (PPD):\nSource ERP Width: {w_erp} px\nCamera FOV: {fov_v}°\nOptimal Pinhole Resolution: {opt_w} x {out_h_var.get()} px")

        auto_res_btn.config(command=on_auto_res)

        # Wire Traces
        mode_var.trace_add("write", sync_math_and_ui)
        solver_mode_var.trace_add("write", sync_math_and_ui)
        num_cams_var.trace_add("write", sync_math_and_ui)
        fov_var.trace_add("write", sync_math_and_ui)
        overlap_var.trace_add("write", sync_math_and_ui)
        dome_pitch_var.trace_add("write", sync_math_and_ui)
        dome_tier_cams_var.trace_add("write", sync_math_and_ui)
        dome_zenith_var.trace_add("write", sync_math_and_ui)
        dome_nadir_var.trace_add("write", sync_math_and_ui)
        aspect_ratio_var.trace_add("write", sync_math_and_ui)
        out_w_var.trace_add("write", sync_math_and_ui)
        out_h_var.trace_add("write", sync_math_and_ui)
        distortion_k1_var.trace_add("write", sync_math_and_ui)
        video_mode_var.trace_add("write", sync_math_and_ui)
        video_fps_var.trace_add("write", sync_math_and_ui)
        video_stride_var.trace_add("write", sync_math_and_ui)
        video_max_frames_var.trace_add("write", sync_math_and_ui)
        canvas_view_mode_var.trace_add("write", sync_math_and_ui)

        # Actions
        def set_as_dataset():
            d = out_var.get().strip()
            if os.path.isdir(d):
                self.dataset_var.set(d)
                messagebox.showinfo("Dataset Configured", f"Active Abu Align dataset updated to:\n{d}")
                on_close_dialog()

        use_ds_btn.config(command=set_as_dataset)

        def open_folder():
            d = out_var.get().strip()
            if os.path.isdir(d):
                os.startfile(d)

        open_folder_btn.config(command=open_folder)

        def run_split_worker():
            in_p = in_var.get().strip()
            out_d = out_var.get().strip()
            if not in_p or not os.path.exists(in_p):
                messagebox.showerror("Error", "Please select a valid input directory of 360° panoramas or video file.")
                return
            if not out_d:
                out_d = os.path.join(in_p, "pinhole_rig") if os.path.isdir(in_p) else f"{os.path.splitext(in_p)[0]}_pinhole_rig"
                out_var.set(out_d)

            orbit_state["exporting"] = True
            split_btn.config(state=tk.DISABLED, text="Exporting…")
            close_btn.config(state=tk.DISABLED)
            dlg_pbar["value"] = 0
            status_lbl.config(text="Status: Initializing virtual multi-camera rig projection...", fg="#a855f7")

            def on_progress(cur, tot, name):
                pct = (cur / max(tot, 1)) * 100.0
                status_txt = f"Status: Projecting view {cur}/{tot} ({pct:.1f}%): {name}"
                dialog.after(0, lambda: [
                    dlg_pbar.configure(value=pct),
                    status_lbl.configure(text=status_txt, fg="#a855f7")
                ])

            def worker():
                try:
                    m = mode_var.get()
                    nc = num_cams_var.get()
                    fov_v = fov_var.get()
                    ow = out_w_var.get()
                    oh = out_h_var.get()
                    k1 = distortion_k1_var.get()
                    fmt = fmt_var.get()
                    q = qual_var.get()
                    tf = export_tf_var.get()

                    v_target_fps = video_fps_var.get() if video_mode_var.get() == "Target FPS" else None
                    v_stride = video_stride_var.get() if video_mode_var.get() == "Frame Stride" else 1
                    v_max_f = video_max_frames_var.get()

                    res = ps.split_panorama_dataset(
                        input_dir=in_p,
                        output_dir=out_d,
                        mode=m,
                        num_cameras=nc,
                        fov_deg=fov_v,
                        crop_width=ow,
                        crop_height=oh,
                        k1=k1,
                        pitch_tier=dome_pitch_var.get(),
                        tier_cams=dome_tier_cams_var.get(),
                        include_zenith=dome_zenith_var.get(),
                        include_nadir=dome_nadir_var.get(),
                        image_format=fmt,
                        quality=q,
                        video_target_fps=v_target_fps,
                        video_frame_stride=v_stride,
                        video_max_frames=v_max_f,
                        export_transforms_json=tf,
                        progress_callback=on_progress,
                    )
                    tot = res.get("total_views", 0)
                    panos = res.get("total_panoramas", 0)
                    sec = res.get("elapsed_seconds", 0.0)

                    def on_done():
                        orbit_state["exporting"] = False
                        dlg_pbar["value"] = 100
                        status_lbl.config(
                            text=f"Status: Complete! Generated {tot:,} pinhole views from {panos} panoramas/frames in {sec}s.",
                            fg=self.accent_green
                        )
                        split_btn.config(state=tk.NORMAL, text="Export views")
                        close_btn.config(state=tk.NORMAL)
                        use_ds_btn.config(state=tk.NORMAL)
                        open_folder_btn.config(state=tk.NORMAL)
                        self._log(f"\n[360 RIG PROJECTOR] Successfully generated {tot:,} pinhole views ({ow}x{oh}) in {out_d} ({sec}s).\n")

                    dialog.after(0, on_done)
                except Exception as ex:
                    def on_err(error=str(ex)):
                        orbit_state["exporting"] = False
                        status_lbl.config(text=f"Status: Failed - {error}", fg=self.accent_red)
                        split_btn.config(state=tk.NORMAL, text="Export views")
                        close_btn.config(state=tk.NORMAL)
                        messagebox.showerror("Split Error", f"Failed to project multi-camera rig:\n{error}")
                    dialog.after(0, on_err)

            t = threading.Thread(target=worker, daemon=True)
            t.start()

        split_btn.config(command=run_split_worker)

        # Initial inspection and math sync
        inspect_input_source()
        sync_math_and_ui()
        from rig_designer_ui import enhance_rig_designer
        enhance_rig_designer(self, locals())

    def _clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", tk.END)
        self.console.configure(state="disabled")
        self.log_search.buffer_changed()
        self.workspace_tabs.select(self.activity_tab)

    def _log(self, text):
        self.console.configure(state="normal")
        self.console.insert(tk.END, text)
        # Bound the visible console; complete diagnostics remain in the session log.
        lines = int(self.console.index("end-1c").split(".")[0])
        if lines > 5000:
            self.console.delete("1.0", f"{lines - 5000}.0")
        self.console.configure(state="disabled")
        if hasattr(self, "log_search"):
            self.log_search.buffer_changed()
        if not hasattr(self, "follow_log_var") or self.follow_log_var.get():
            self.console.see(tk.END)
        try:
            self.root.update_idletasks()
        except Exception:
            pass

    def _parse_train_iterations(self):
        iter_str = self.train_iter_var.get()
        if "7,000" in iter_str:
            return 7000
        elif "15,000" in iter_str:
            return 15000
        return 30000

    def _parse_pfgs_match_views(self):
        raw = str(self.pfgs_match_views_var.get()).strip()
        digits = re.findall(r"\d+", raw)
        if digits:
            return int(digits[0])
        return 8

    def _parse_pfgs_downscale(self):
        raw = str(self.pfgs_downscale_var.get()).strip().lower()
        if "auto" in raw:
            return "auto"
        digits = re.findall(r"\d+", raw)
        if digits:
            return str(digits[0])
        return "1"

    def _parse_train_res(self):
        val = str(self.train_res_var.get()).lower()
        if "2k" in val or "1920" in val:
            return "2k"
        elif "1k" in val or "1024" in val:
            return "1k"
        elif "4k" in val or "3840" in val:
            return "4k"
        return "original"

    def _parse_train_max_gaussians(self):
        val = str(self.train_max_gaussians_var.get()).lower()
        if "unlimited" in val or "default" in val or "none" in val or val.strip() == "0":
            return 0
        if "500,000" in val or "500k" in val:
            return 500000
        elif "1,000,000" in val or "1m" in val:
            return 1000000
        elif "2,000,000" in val or "2m" in val:
            return 2000000
        elif "3,000,000" in val or "3m" in val:
            return 3000000
        elif "5,000,000" in val or "5m" in val:
            return 5000000
        digits = re.findall(r"\d+", val.replace(",", ""))
        if digits:
            return int(digits[0])
        return 0

    def _parse_ace_stage_res(self):
        val = str(self.ace_stage_res_var.get()).lower()
        if "2k" in val or "1920" in val:
            return "2k"
        elif "1k" in val or "1024" in val:
            return "1k"
        elif "4k" in val or "3840" in val:
            return "4k"
        return "native"

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
        self.active_stage_idx = stage_idx
        if not hasattr(self, "stage_badges"):
            return
        for i, badge in enumerate(self.stage_badges):
            if i < stage_idx:
                badge.config(bg="#24483f", fg="#70dbc0")
            elif i == stage_idx:
                badge.config(bg="#70dbc0", fg="#111518")
            else:
                badge.config(bg="#242d32", fg="#a0adb5")

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
                self.pause_btn.config(text="Pause", bg="#242d32")
                self.telemetry_lbl.config(text="Status: ▶ Resumed and executing...", fg=self.accent_blue)
                self._log("\n[PROCESS] ▶ Execution resumed by user.\n")
        except Exception as e:
            self._log(f"\n[ERROR] Failed to toggle pause: {e}\n")

    def _start_alignment(self, force_auto_train=False, skip_auto_train=False):
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
        self.pause_btn.config(state=tk.NORMAL, text="Pause", bg="#242d32")
        self.stop_btn.config(state=tk.NORMAL)
        if hasattr(self, "save_finish_btn"):
            self.save_finish_btn.config(state=tk.DISABLED)
        self._clear_console()

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

        # Active Alignment Engine Options
        if align_engine in ["vggt_omega", "vggt"]:
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
            cmd.extend([
                "--align_mode", self.align_var.get(),
                "--lens_model", self.lens_var.get(),
                "--vggt_preprocess_mode", self.vggt_preprocess_var.get(),
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
        elif align_engine in ["vggttt", "vgg-ttt", "vggt3"]:
            raw_conf = str(self.vggttt_conf_var.get()).strip()
            if "adapt" in raw_conf.lower():
                conf_str = "adaptive"
            else:
                conf_str = raw_conf.split()[0]
            steps_str = str(self.vggttt_steps_var.get()).split()[0]
            cmd.extend([
                "--vggttt_conf", conf_str,
                "--vggttt_ttt_steps", steps_str,
            ])
            prep_val = str(self.vggttt_prep_var.get()).lower()
            if "crop" in prep_val and "auto" not in prep_val:
                prep_mode = "crop"
            elif "pad" in prep_val:
                prep_mode = "pad"
            else:
                prep_mode = "auto"
            cmd.extend(["--vggttt_prep_mode", prep_mode])

            if self.vggttt_mem_var.get():
                cmd.append("--vggttt_mem_eff")
            if self.vggttt_pred_var.get():
                cmd.append("--vggttt_global_pred")
            else:
                cmd.append("--no_vggttt_global_pred")
        elif align_engine == "ace0":
            cmd.extend([
                "--ace_iterations", str(self.ace_iter_var.get()),
                "--ace_refinement", self.ace_refine_var.get(),
                "--ace_try_seeds", str(self.ace_seeds_var.get()),
            ])
            if self.ace_calib_var.get():
                cmd.append("--ace_refine_calib")
            else:
                cmd.append("--no_ace_refine_calib")
            if self.ace_dense_var.get():
                cmd.append("--ace_dense_pc")
            if self.ace_refit_var.get():
                cmd.extend(["--ace_final_refit", "--ace_final_refine"])
            else:
                cmd.extend(["--no_ace_final_refit", "--no_ace_final_refine"])
            cmd.extend(["--ace_stage_res", self._parse_ace_stage_res()])
            try:
                w_val = int(str(self.ace_workers_var.get()).strip())
                cmd.extend(["--ace_seed_workers", str(w_val)])
            except Exception:
                pass
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
        elif align_engine == "opensfm":
            cmd.extend([
                "--projection", self.opensfm_proj_var.get(),
                "--opensfm_feature", self.opensfm_feat_var.get(),
                "--opensfm_matcher", self.opensfm_matcher_var.get(),
                "--opensfm_min_features", str(self.opensfm_budget_var.get()),
                "--opensfm_neighbors", str(self.opensfm_neighbors_var.get()),
                "--opensfm_bundle_interval", str(self.opensfm_bundle_var.get()),
            ])
            if self.opensfm_opt_cam_var.get():
                cmd.append("--opensfm_optimize_camera")
            if self.opensfm_transcode_var.get():
                cmd.extend([
                    "--split_panorama",
                    "--split_mode", self.rig_split_mode_var.get(),
                    "--split_cameras", str(self.rig_num_cams_var.get()),
                    "--split_fov", str(self.rig_fov_var.get()),
                    "--split_res", str(self.rig_res_var.get()),
                ])
        elif align_engine in ["360dvo", "dvo"]:
            cmd.extend([
                "--dvo_patches", str(self.dvo_patches_var.get()),
            ])
            try:
                mask_val = float(str(self.dvo_polar_mask_var.get()).replace("%", "").strip()) / 100.0
            except Exception:
                mask_val = 0.08
            cmd.extend(["--dvo_polar_mask", str(mask_val)])

            try:
                ba_val = int(str(self.dvo_ba_window_var.get()).strip())
            except Exception:
                ba_val = 8
            cmd.extend(["--dvo_ba_window", str(ba_val)])

            try:
                scale_val = float(str(self.dvo_scale_var.get()).strip())
            except Exception:
                scale_val = 3.2
            cmd.extend(["--dvo_scale", str(scale_val)])

            if self.dvo_dense_cloud_var.get():
                cmd.append("--dvo_dense_cloud")

            try:
                lifetime_val = int(str(self.dvo_patch_lifetime_var.get()).split()[0].strip())
            except Exception:
                lifetime_val = 13
            cmd.extend(["--dvo_patch_lifetime", str(lifetime_val)])

            try:
                refine_val = int(str(self.dvo_final_refine_var.get()).split()[0].strip())
            except Exception:
                refine_val = 12
            cmd.extend(["--dvo_final_refine", str(refine_val)])

            if self.dvo_gradient_bias_var.get():
                cmd.append("--dvo_gradient_bias")
            else:
                cmd.append("--no_dvo_gradient_bias")

            if self.dvo_viz_var.get():
                cmd.append("--dvo_viz")
            else:
                cmd.append("--no_dvo_viz")
            if self.dvo_weights_var.get().strip():
                cmd.extend(["--dvo_weights", self.dvo_weights_var.get().strip()])

        # Active 3DGS Training Engine Options (logged in report, executed if auto-train enabled)
        train_engine = self.train_engine_var.get()
        cmd.extend([
            "--train_engine", train_engine,
            "--train_iterations", str(self._parse_train_iterations()),
            "--train_res", self._parse_train_res(),
        ])
        max_g = self._parse_train_max_gaussians()
        if max_g > 0:
            cmd.extend(["--train_max_gaussians", str(max_g)])

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
        elif train_engine == "pfgs360":
            cmd.extend([
                "--pfgs_depth", self.pfgs_depth_var.get().lower(),
                "--pfgs_dataparser", self.pfgs_dataparser_var.get(),
                "--pfgs_match_views", str(self._parse_pfgs_match_views()),
                "--pfgs_densify_views", str(self.pfgs_densify_views_var.get()).strip(),
                "--sh_degree", str(self.pfgs_sh_var.get()),
                "--pfgs_downscale", self._parse_pfgs_downscale(),
            ])
            if self.pfgs_filter_sky_var.get():
                cmd.append("--pfgs_filter_sky")
            else:
                cmd.append("--no_pfgs_filter_sky")
            if self.pfgs_spherical_var.get():
                cmd.append("--pfgs_spherical")
            else:
                cmd.append("--no_pfgs_spherical")
            if self.pfgs_consist_var.get():
                cmd.append("--pfgs_consist")
            else:
                cmd.append("--no_pfgs_consist")
            if self.pfgs_inlier_var.get():
                cmd.append("--pfgs_inlier")
            else:
                cmd.append("--no_pfgs_inlier")
            if self.pfgs_align_depth_var.get():
                cmd.append("--pfgs_align_depth")
            else:
                cmd.append("--no_pfgs_align_depth")

        if not skip_auto_train and (force_auto_train or self.auto_train_var.get()):
            cmd.append("--train")

        if getattr(self, "active_log_path", None):
            cmd.extend(["--log_file", self.active_log_path])

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

        train_engine = self.train_engine_var.get()
        # Smart check: if training engine requires camera poses, ensure poses exist
        if train_engine != "pfgs360":
            has_poses = False
            for cand in [
                os.path.join(dataset_path, "3dgs_dataset", "sparse", "0"),
                os.path.join(dataset_path, "sparse", "0"),
                os.path.join(dataset_path, "transforms.json"),
            ]:
                if os.path.exists(cand):
                    has_poses = True
                    break
            if not has_poses and os.path.isdir(os.path.join(dataset_path, "results")):
                for sub in os.listdir(os.path.join(dataset_path, "results")):
                    if os.path.isdir(os.path.join(dataset_path, "results", sub, "3dgs_dataset")):
                        has_poses = True
                        break

            if not has_poses:
                resp = messagebox.askyesno(
                    "Camera Poses Required",
                    f"The selected dataset folder does not contain camera poses:\n{dataset_path}\n\n"
                    f"'{train_engine}' requires camera poses (sparse/0/ or transforms.json) generated by photogrammetry alignment.\n\n"
                    f"Would you like to run 'Align & train' to align these photos and train a 3DGS model?",
                    icon="question"
                )
                if resp:
                    self._start_one_click()
                return
        elif train_engine == "pfgs360" and self.pfgs_dataparser_var.get() in ["posefree", "pose-free", "direct"]:
            has_poses = False
            for cand in [
                os.path.join(dataset_path, "transforms.json"),
                os.path.join(dataset_path, "reconstruction.json"),
                os.path.join(dataset_path, "opensfm", "reconstruction.json"),
                os.path.join(dataset_path, "3dgs_dataset", "transforms.json"),
                os.path.join(dataset_path, "3dgs_dataset", "opensfm", "reconstruction.json"),
                os.path.join(dataset_path, "sparse", "0"),
            ]:
                if os.path.exists(cand):
                    has_poses = True
                    break
            if not has_poses and os.path.isdir(os.path.join(dataset_path, "results")):
                for sub in os.listdir(os.path.join(dataset_path, "results")):
                    if os.path.isdir(os.path.join(dataset_path, "results", sub, "3dgs_dataset")):
                        has_poses = True
                        break

            if has_poses:
                resp = messagebox.askyesno(
                    "Pre-Aligned Camera Poses Detected",
                    f"Pre-aligned camera poses (from OpenSfM/360DVO) were detected for this dataset:\n{dataset_path}\n\n"
                    f"You currently have 'Pose-Free Direct [CPU Matching]' selected.\n"
                    f"Pose-free mode discards your pre-aligned poses and recalculates them from scratch on CPU (~15-30 min).\n\n"
                    f"Would you like to switch to 'Auto-Detect' to use your pre-aligned poses and train immediately at full 100% RTX 4090 GPU speed?",
                    icon="question"
                )
                if resp:
                    self.pfgs_dataparser_var.set("auto")

        self.is_running = True
        self.is_paused = False
        self.start_btn.config(state=tk.DISABLED)
        self.one_click_btn.config(state=tk.DISABLED)
        self.train_only_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause", bg="#242d32")
        self.stop_btn.config(state=tk.NORMAL)
        if hasattr(self, "save_finish_btn"):
            self.save_finish_btn.config(state=tk.NORMAL)
        self._clear_console()

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
        effective_dataset = dataset_path
        target_run = getattr(self, "train_target_run_var", None)
        selected_run = target_run.get() if target_run else "Latest Run (Auto-Detect)"
        if selected_run and selected_run != "Latest Run (Auto-Detect)":
            cand_results = os.path.join(dataset_path, "results")
            if not os.path.isdir(cand_results):
                cand_p_results = os.path.join(os.path.dirname(dataset_path), "results")
                if os.path.isdir(cand_p_results):
                    cand_results = cand_p_results
            run_path = os.path.join(cand_results, selected_run)
            if os.path.isdir(os.path.join(run_path, "3dgs_dataset")):
                effective_dataset = os.path.join(run_path, "3dgs_dataset")
            elif os.path.isdir(run_path):
                effective_dataset = run_path

        cmd = [
            self.conda_python,
            "-u",
            self.train_script,
            "--dataset", effective_dataset,
            "--engine", train_engine,
            "--iterations", str(self._parse_train_iterations()),
            "--train_res", self._parse_train_res(),
        ]
        max_g = self._parse_train_max_gaussians()
        if max_g > 0:
            cmd.extend(["--max_gaussians", str(max_g)])

        if getattr(self, "active_log_path", None):
            cmd.extend(["--log_file", self.active_log_path])

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
        elif train_engine == "pfgs360":
            cmd.extend([
                "--pfgs_depth", self.pfgs_depth_var.get().lower(),
                "--pfgs_dataparser", self.pfgs_dataparser_var.get(),
                "--pfgs_match_views", str(self._parse_pfgs_match_views()),
                "--pfgs_densify_views", str(self.pfgs_densify_views_var.get()).strip(),
                "--sh_degree", str(self.pfgs_sh_var.get()),
                "--pfgs_downscale", self._parse_pfgs_downscale(),
            ])
            if self.pfgs_filter_sky_var.get():
                cmd.append("--pfgs_filter_sky")
            else:
                cmd.append("--no_pfgs_filter_sky")
            if self.pfgs_spherical_var.get():
                cmd.append("--pfgs_spherical")
            else:
                cmd.append("--no_pfgs_spherical")
            if self.pfgs_consist_var.get():
                cmd.append("--pfgs_consist")
            else:
                cmd.append("--no_pfgs_consist")
            if self.pfgs_inlier_var.get():
                cmd.append("--pfgs_inlier")
            else:
                cmd.append("--no_pfgs_inlier")
            if self.pfgs_align_depth_var.get():
                cmd.append("--pfgs_align_depth")
            else:
                cmd.append("--no_pfgs_align_depth")

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
                lf.write("Abu Align & Train v0.0.8.5 - Run Telemetry & Diagnostic Log\n")
                lf.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                lf.write(f"Command:   {' '.join(cmd)}\n")
                lf.write("=" * 72 + "\n\n")
                lf.flush()

                sub_env = os.environ.copy()
                sub_env["PYTHONIOENCODING"] = "utf-8"
                sub_env["PYTHONUTF8"] = "1"
                sub_env["PYTHONUNBUFFERED"] = "1"
                self.worker_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                    env=sub_env,
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
                                out_p = parts[1].strip()
                                self.latest_output_dir = out_p
                                self.latest_alignment_dir = out_p
                        elif "Target Output:" in line:
                            parts = line.split("Target Output:")
                            if len(parts) > 1:
                                out_p = parts[1].strip()
                                self.latest_output_dir = out_p
                                self.latest_alignment_dir = out_p
                        elif "Output Dir:" in line:
                            parts = line.split("Output Dir:")
                            if len(parts) > 1:
                                out_p = parts[1].strip()
                                self.latest_train_dir = out_p

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
                    self.pause_btn.config(state=tk.DISABLED, text="Pause", bg="#242d32")
                    self.stop_btn.config(state=tk.DISABLED)
                    if hasattr(self, "save_finish_btn"):
                        self.save_finish_btn.config(state=tk.DISABLED)
                    self.is_paused = False
                    if code == 0:
                        self.progress_bar["value"] = 100
                        self.pct_lbl.config(text="100.0%", fg=self.accent_green)
                        self._set_stage(6)
                        self.telemetry_lbl.config(text="Status: Pipeline Completed Successfully (100%)", fg=self.accent_green)
                        self.eta_lbl.config(text="Done")
                        self._log("\n[STATUS] Pipeline Job Completed.\n")
                        # Auto-discover and select newest alignment run across all tabs
                        self._refresh_train_target_runs(select_newest=True)
                        if hasattr(self, "workspace_3d"):
                            try:
                                self.workspace_3d.refresh_runs()
                                self.workspace_3d.load_selected_run()
                            except Exception:
                                pass
                    else:
                        self.pct_lbl.config(text="Failed", fg="#ff4d4d")
                        active_task = "3DGS Training" if getattr(self, "active_stage_idx", 0) == 5 else "Alignment"
                        self.telemetry_lbl.config(text=f"Status: {active_task} Failed (Error code {code}) - See log", fg="#ff4d4d")
                        self.eta_lbl.config(text="Error")
                        self._log(f"\n[CRITICAL ERROR] {active_task} exited with error code {code}. See traceback above.\n")
                    self.log_status_lbl.config(
                        text="Log: Saved (logs/)",
                        fg=self.text_secondary
                    )
                else:
                    self._log(msg)
                    msg_s = msg.strip()
                    if any(k in msg for k in ["Stage 1", "Found", "Scanning", "images found"]):
                        self._set_stage(0)
                    elif any(k in msg for k in ["Stage 2", "Loading model", "Initializing GPU", "Initializing Neural Network", "Initializing ACE0 Engine", "ZoeDepth model ready", "Downloading ZoeDepth"]):
                        self._set_stage(1)
                        if "ACE0" in msg or "ZoeDepth" in msg:
                            self.telemetry_lbl.config(text="Stage 2/6: Initializing ACE0 & ZoeDepth Models...", fg=self.accent_blue)
                    elif msg_s.startswith("[TELEMETRY:STAGE2]") or "Processing seed" in msg or "Trying seeds:" in msg:
                        self._set_stage(2)
                        if msg_s.startswith("[TELEMETRY:STAGE2]"):
                            try:
                                parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                                s = parts.get("seed", "?")
                                tot = parts.get("total", "?")
                                self.telemetry_lbl.config(text=f"Stage 3/6: ACE0 Seed Search [{s}/{tot}] (ZoeDepth Prior)...", fg="#38bdf8")
                            except Exception:
                                pass
                        else:
                            seed_m = re.search(r"Processing seed\s+(\d+)", msg)
                            if seed_m:
                                self.telemetry_lbl.config(text=f"Stage 3/6: ACE0 Evaluating Candidate Seed #{seed_m.group(1)}...", fg="#38bdf8")
                            else:
                                self.telemetry_lbl.config(text="Stage 3/6: ACE0 Candidate Seed Evaluation...", fg="#38bdf8")
                    # Stage 3 Telemetry: Neural Batch Inference (VGGT-Omega / OpenSfM / 360DVO) OR ACE0 Scene MLP
                    elif (
                        msg_s.startswith("[TELEMETRY:STAGE3]")
                        or "Stage 3" in msg
                        or ("Iteration:" in msg and "|" in msg and ("ace_trainer" in msg or "ACE0" in msg or "Epoch" in msg))
                    ):
                        self._set_stage(3)
                        try:
                            # Disambiguate ACE0 Scene MLP vs VGGT-Omega / Foundation Inference
                            is_ace0 = (
                                "iter=" in msg_s
                                or "loss=" in msg_s
                                or ("Iteration:" in msg and "|" in msg)
                                or ((getattr(self, "engine_var", None) is not None and self.engine_var.get() in ["acezero", "ace0"]) and "batch=" not in msg_s)
                                or ((getattr(self, "align_engine_var", None) is not None and self.align_engine_var.get() in ["acezero", "ace0"]) and "batch=" not in msg_s)
                            )
                            if is_ace0:
                                if msg_s.startswith("[TELEMETRY:STAGE3]"):
                                    parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                                    pct = float(parts.get("pct", 0))
                                    cur_it = parts.get("iter", "?")
                                    tot_it = parts.get("total", "?")
                                    loss_v = parts.get("loss", "?")
                                    inl_v = parts.get("inliers", "?")
                                else:
                                    it_m = re.search(r"Iteration:\s*(\d+)\s*\|\s*(\d+).*Loss:\s*([\d\.]+).*inliers.*:\s*([\d\.]+)%", msg)
                                    if it_m:
                                        cur_it = int(it_m.group(1))
                                        tot_it = int(it_m.group(2))
                                        loss_v = it_m.group(3)
                                        inl_v = it_m.group(4)
                                        pct = (cur_it / max(1, tot_it)) * 100.0
                                    else:
                                        pct, cur_it, tot_it, loss_v, inl_v = 0, "?", "?", "?", "?"
                                self.progress_bar["value"] = pct
                                self.pct_lbl.config(text=f"{pct:5.1f}%")
                                focal_str = f" | Focal: {getattr(self, '_last_ace_focal', '')}px" if hasattr(self, '_last_ace_focal') else ""
                                self.telemetry_lbl.config(text=f"Stage 4/6: ACE0 Scene MLP [{cur_it}/{tot_it}] ({pct:.1f}%) | Loss: {loss_v} | Inliers: {inl_v}%{focal_str}", fg=self.accent_blue)
                            else:
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
                                    self.telemetry_lbl.config(text=f"Stage 4/6: Neural Batch Inference [{b}/{tot}] ({pct:.1f}%)", fg=self.accent_blue)
                        except Exception:
                            pass
                    elif msg_s.startswith("[TELEMETRY:ACE_OPTICS]") or "Current Focal Length:" in msg:
                        f_m = re.search(r"(?:Current Focal Length:|focal=)\s*([\d\.]+)", msg)
                        if f_m:
                            self._last_ace_focal = f_m.group(1)
                    elif msg_s.startswith("[TELEMETRY:STAGE4]") or ("Successfully registered" in msg and "mapping images" in msg):
                        self._set_stage(4)
                        try:
                            if msg_s.startswith("[TELEMETRY:STAGE4]"):
                                parts = dict(kv.split("=") for kv in msg_s.split()[1:] if "=" in kv)
                                reg_pct = float(parts.get("pct", 100.0))
                            else:
                                reg_m = re.search(r"Successfully registered\s*([\d\.]+)%", msg)
                                reg_pct = float(reg_m.group(1)) if reg_m else 100.0
                            self.progress_bar["value"] = reg_pct
                            self.pct_lbl.config(text=f"{reg_pct:5.1f}%")
                            self.telemetry_lbl.config(text=f"Stage 5/6: ACE0 Camera Registration [{reg_pct:.1f}% Views Aligned]", fg="#a855f7")
                        except Exception:
                            pass
                    elif msg_s.startswith("[TELEMETRY:STAGE5]") or "Stage 5" in msg or ("Deep Multi-View Tracking" in msg and "Query" in msg):
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
                                self.telemetry_lbl.config(text=f"Stage 5/6: Multi-View Tracking [Query {q}/{tot}, Frame #{f}] ({pct:.1f}%)", fg="#a855f7")
                                self.eta_lbl.config(text=f"⏱ ETA: {eta_str}")
                        except Exception:
                            pass
                    elif any(k in msg for k in ["[5/5] Building COLMAP", "Stage 4", "Generating 3D points", "Predicting dense point", "Building pycolmap"]) or ("Loaded" in msg and "3D scene coordinates" in msg):
                        self._set_stage(4)
                        self.telemetry_lbl.config(text="Stage 5/6: Building COLMAP Reconstruction & Point Cloud...", fg="#a855f7")
                    elif any(k in msg for k in ["[LichtFeld]", "[Nerfstudio]", "[3DGRUT]", "[PFGS360]", "[PFGS360-ERR]", "Stage 6", "Starting 3DGS", "pfgs360"]):
                        self._set_stage(5)
                        if hasattr(self, "save_finish_btn") and str(self.save_finish_btn["state"]) != tk.NORMAL and self.is_running:
                            self.save_finish_btn.config(state=tk.NORMAL)
                        depth_m = re.search(r"(\d+)%\|.*\|\s*(\d+)/(\d+)", msg)
                        step_m = re.search(r"(\d+)\s*\(([\d\.]+)%\)", msg)
                        lf_m = re.search(r"Iteration\s+(\d+)/(\d+)\s*\(([\d\.]+)%\)", msg)
                        pose_m = re.search(r"compute relative camera pose for views .* and (\d+):", msg)

                        if depth_m:
                            pct_val = float(depth_m.group(1))
                            cur_f = depth_m.group(2)
                            tot_f = depth_m.group(3)
                            self.progress_bar["value"] = pct_val * 0.15
                            self.pct_lbl.config(text=f"{pct_val:5.1f}%")
                            self.telemetry_lbl.config(text=f"Stage 6/6: Precomputing Depth Priors [{cur_f}/{tot_f}] ({pct_val:.0f}%)", fg=self.accent_cyan)
                        elif step_m:
                            cur_step = int(step_m.group(1))
                            step_pct = float(step_m.group(2))
                            self.progress_bar["value"] = step_pct
                            self.pct_lbl.config(text=f"{step_pct:5.1f}%")
                            engine_label = self.train_engine_var.get().upper()
                            self.telemetry_lbl.config(text=f"Stage 6/6: {engine_label} Active [Step {cur_step:,}] ({step_pct:.1f}%)", fg=self.accent_cyan)
                            parts = [p.strip() for p in re.split(r"\s{2,}", msg.strip()) if p.strip()]
                            if len(parts) >= 3:
                                self.eta_lbl.config(text=f"⏱ ETA: {parts[-1]}")
                            else:
                                eta_m = re.search(r"ETA:\s*([^\s,]+)", msg)
                                if eta_m:
                                    self.eta_lbl.config(text=f"⏱ ETA: {eta_m.group(1).strip()}")
                        elif lf_m:
                            cur_step = int(lf_m.group(1))
                            tot_step = int(lf_m.group(2))
                            lf_pct = float(lf_m.group(3))
                            self.progress_bar["value"] = lf_pct
                            self.pct_lbl.config(text=f"{lf_pct:5.1f}%")
                            self.telemetry_lbl.config(text=f"Stage 6/6: LichtFeld Active [{cur_step:,}/{tot_step:,}] ({lf_pct:.1f}%)", fg=self.accent_cyan)
                            eta_m = re.search(r"ETA:\s*([0-9a-zA-Z\s]+)", msg)
                            if eta_m:
                                self.eta_lbl.config(text=f"⏱ ETA: {eta_m.group(1).strip()}")
                        elif pose_m:
                            view_idx = pose_m.group(1)
                            self.telemetry_lbl.config(text=f"Stage 6/6: CPU Relative Pose Matching (View #{view_idx})", fg="#a855f7")
                        elif any(k in msg for k in ["Exporting", "ns-export", "export_spz"]):
                            self.telemetry_lbl.config(text="Stage 6/6: Exporting 3D Gaussian Splat (.ply)...", fg=self.accent_cyan)
                        elif "Training Finished" in msg or "Training complete!" in msg:
                            self.progress_bar["value"] = 100
                            self.pct_lbl.config(text="100.0%", fg=self.accent_green)
                            self.telemetry_lbl.config(text="Stage 6/6: 3DGS Model Generation Complete (100%)", fg=self.accent_green)
                        elif any(k in msg.lower() for k in ["step", "iter", "export", "loss", "rays / sec"]) and not (step_m or lf_m or depth_m):
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
                    self.pause_btn.config(text="Resume", bg="#24483f")
                    self.telemetry_lbl.config(text="Status: Paused (Process tree suspended)", fg="#f59e0b")
                    self._log("\n[PAUSE] Alignment/Training process suspended by user.\n")
                else:
                    for p in all_procs:
                        try:
                            p.resume()
                        except Exception:
                            pass
                    self.is_paused = False
                    self.pause_btn.config(text="Pause", bg="#242d32")
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
            self.pause_btn.config(state=tk.DISABLED, text="Pause", bg="#242d32")
            self.stop_btn.config(state=tk.DISABLED)
            if hasattr(self, "save_finish_btn"):
                self.save_finish_btn.config(state=tk.DISABLED)
            self.telemetry_lbl.config(text="Status: Aborted by user", fg=self.accent_red)

    def _save_and_finish_training(self):
        """
        Gracefully requests the active training engine (PFGS360 / Splatfacto / Nerfstudio / LichtFeld)
        to complete training at the current iteration, save the latest checkpoint, and immediately
        export the final 3DGS .ply model.
        """
        if not self.is_running:
            return

        self._log("\n" + "="*70 + "\n")
        self._log(" [SAVE & FINISH] Early stop requested by user!\n")
        self._log(" Preserving current Gaussians, concluding optimization, and exporting 3DGS model...\n")
        self._log("="*70 + "\n")

        self.telemetry_lbl.config(
            text="Status: Finalizing Training & Exporting 3DGS Model...",
            fg=self.accent_cyan
        )
        if hasattr(self, "save_finish_btn"):
            self.save_finish_btn.config(state=tk.DISABLED)

        # If training was paused, resume it so the engine can cleanly process the exit request
        if self.is_paused:
            try:
                self._toggle_pause()
            except Exception:
                pass

        # Write sentinels across Windows and WSL locations
        dataset_path = self.dataset_var.get().strip()
        sentinels = [
            os.path.join(tempfile.gettempdir(), "abu_finish_and_save.flag")
        ]
        if dataset_path and os.path.isdir(dataset_path):
            sentinels.append(os.path.join(dataset_path, ".finish_and_save"))
            res_dir = os.path.join(dataset_path, "results")
            if os.path.isdir(res_dir):
                sentinels.append(os.path.join(res_dir, ".finish_and_save"))
                try:
                    for sub in os.listdir(res_dir):
                        sub_p = os.path.join(res_dir, sub)
                        if os.path.isdir(sub_p):
                            sentinels.append(os.path.join(sub_p, ".finish_and_save"))
                            sub_models = os.path.join(sub_p, "3dgs_models")
                            if os.path.isdir(sub_models):
                                sentinels.append(os.path.join(sub_models, ".finish_and_save"))
                except Exception:
                    pass
        if self.latest_output_dir and os.path.isdir(self.latest_output_dir):
            sentinels.append(os.path.join(self.latest_output_dir, ".finish_and_save"))
            parent_dir = os.path.dirname(self.latest_output_dir)
            if parent_dir and os.path.isdir(parent_dir):
                sentinels.append(os.path.join(parent_dir, ".finish_and_save"))

        for s_path in sentinels:
            try:
                with open(s_path, "w", encoding="utf-8") as f:
                    f.write(f"finish_and_save={time.time()}\n")
            except Exception as e:
                self._log(f" [Warning] Could not create sentinel {s_path}: {e}\n")

        # Also touch flags inside WSL if WSL is available
        try:
            from wsl_bridge import run_command_in_wsl, win_to_wsl_path
            run_command_in_wsl("touch /tmp/abu_finish_and_save.flag", log_callback=lambda _: None)
            if dataset_path:
                wsl_ds = win_to_wsl_path(dataset_path)
                run_command_in_wsl(f"touch '{wsl_ds}/.finish_and_save'", log_callback=lambda _: None)
            if self.latest_output_dir:
                wsl_out = win_to_wsl_path(self.latest_output_dir)
                run_command_in_wsl(f"touch '{wsl_out}/.finish_and_save'", log_callback=lambda _: None)
        except Exception:
            pass

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

        # Terminate any existing fast_viewer instance to guarantee fresh bind on port 8080
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                cmdline = proc.info.get('cmdline') or []
                if any("fast_viewer.py" in arg for arg in cmdline):
                    proc.kill()
        except Exception:
            pass

        target_arg = self.latest_alignment_dir if (self.latest_alignment_dir and os.path.isdir(self.latest_alignment_dir)) else (
            self.latest_output_dir if (self.latest_output_dir and os.path.isdir(self.latest_output_dir)) else dataset
        )
        cmd = [self.conda_python, fast_viewer_script, "--input", target_arg]
        subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)

    def _show_3d_workspace(self):
        """Switches to the native 04 3D Workspace tab and refreshes/loads the latest reconstruction."""
        if hasattr(self, "workspace_tabs") and hasattr(self, "workspace_3d"):
            self.workspace_tabs.select(self.workspace_3d)
            self.workspace_3d.refresh_runs()
            if self.workspace_3d.points is None:
                self.workspace_3d.load_selected_run()

    def _open_doc_window(self, title, doc_path):
        if not os.path.isfile(doc_path):
            messagebox.showerror("Error", f"Document file not found:\n{doc_path}")
            return

        doc_win = tk.Toplevel(self.root)
        doc_win.title(f"Abu Align v0.0.8.5 - {title}")
        doc_win.geometry("1060x820")
        doc_win.minsize(860, 580)
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

        find_state = {"matches": [], "current_idx": -1, "query": ""}

        def do_search(forward=True):
            query = search_var.get().strip()
            if query != find_state["query"]:
                find_state["current_idx"] = -1
                find_state["query"] = query
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
            fg="#e7edef",
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
            fg="#e7edef",
            relief=tk.FLAT,
            padx=6,
            pady=1,
            command=lambda: do_search(False)
        )
        find_prev_btn.pack(side=tk.LEFT, padx=2)

        search_entry.bind("<Return>", lambda e: do_search(True))
        search_entry.bind("<Shift-Return>", lambda e: do_search(False))
        search_var.trace_add("write", lambda *_: do_search(True))
        doc_win.bind("<Control-f>", lambda e: search_entry.focus_set())
        doc_win.bind("<Escape>", lambda e: doc_win.destroy())

        close_btn = tk.Button(
            tb,
            text="✕ Close",
            font=("Segoe UI", 8, "bold"),
            bg="#374151",
            fg="#e7edef",
            activebackground=self.accent_red,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            command=doc_win.destroy
        )
        close_btn.pack(side=tk.RIGHT, padx=4)

        # Two-Panel Layout (Left Sidebar for Table of Contents, Right for Document Viewer)
        paned = tk.PanedWindow(doc_win, orient=tk.HORIZONTAL, bg=self.bg_dark, sashwidth=4, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Panel: Topics Navigation Sidebar
        nav_frame = tk.Frame(paned, bg=self.panel_bg, width=260)
        nav_frame.pack_propagate(False)
        paned.add(nav_frame, minsize=210, width=270)

        nav_hdr = tk.Frame(nav_frame, bg=self.panel_bg, padx=10, pady=8)
        nav_hdr.pack(fill=tk.X)
        tk.Label(nav_hdr, text="📑  TOPICS & CHAPTERS", font=("Segoe UI", 9, "bold"), fg=self.accent_blue, bg=self.panel_bg).pack(anchor="w")

        topic_listbox = tk.Listbox(
            nav_frame,
            font=("Segoe UI", 9),
            bg="#141420",
            fg="#cbd5e1",
            selectbackground=self.accent_blue,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            activestyle="none"
        )
        topic_scroll = ttk.Scrollbar(nav_frame, orient=tk.VERTICAL, command=topic_listbox.yview)
        topic_listbox.configure(yscrollcommand=topic_scroll.set)
        topic_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        topic_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 2), pady=(0, 8))

        # Right Panel: ScrolledText Document Reader
        content_pane = tk.Frame(paned, bg=self.bg_dark)
        paned.add(content_pane, minsize=450)

        text_widget = ScrolledText(
            content_pane,
            wrap=tk.WORD,
            bg="#111518",
            fg="#e2e8f0",
            insertbackground="#ffffff",
            selectbackground="#70dbc0",
            selectforeground="#ffffff",
            font=("Segoe UI", 10),
            padx=20,
            pady=16,
            relief=tk.FLAT
        )
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Configure styling tags
        text_widget.tag_configure("h1", font=("Segoe UI", 16, "bold"), foreground="#60a5fa", spacing1=14, spacing3=6)
        text_widget.tag_configure("h2", font=("Segoe UI", 13, "bold"), foreground="#38bdf8", spacing1=12, spacing3=4)
        text_widget.tag_configure("h3", font=("Segoe UI", 11, "bold"), foreground="#a5b4fc", spacing1=8, spacing3=2)
        text_widget.tag_configure("code_block", font=("Consolas", 9), background="#1a2024", foreground="#fdba74", spacing1=2, spacing3=2)
        text_widget.tag_configure("bullet", font=("Segoe UI", 10), foreground="#93c5fd")
        text_widget.tag_configure("alert_note", font=("Segoe UI", 9, "bold"), foreground="#38bdf8")
        text_widget.tag_configure("alert_warn", font=("Segoe UI", 9, "bold"), foreground="#f59e0b")
        text_widget.tag_configure("hr", font=("Segoe UI", 4), foreground="#374151")
        text_widget.tag_configure("search_hl", background="#854d0e", foreground="#ffffff")
        text_widget.tag_configure("search_current", background="#dc2626", foreground="#ffffff")
        text_widget.tag_configure("topic_hl", background="#1e3a8a", foreground="#ffffff")

        # Load file and parse Markdown
        topic_bookmarks = []
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            in_code_block = False
            for line in lines:
                sline = line.strip()
                cur_line_index = text_widget.index(tk.INSERT)

                if sline.startswith("```"):
                    in_code_block = not in_code_block
                    text_widget.insert(tk.END, line, "code_block")
                elif in_code_block:
                    text_widget.insert(tk.END, line, "code_block")
                elif sline.startswith("# "):
                    title_text = sline[2:].strip()
                    topic_listbox.insert(tk.END, f"📌 {title_text}")
                    topic_bookmarks.append((title_text, cur_line_index))
                    text_widget.insert(tk.END, line, "h1")
                elif sline.startswith("## "):
                    title_text = sline[3:].strip()
                    topic_listbox.insert(tk.END, f"  ▸ {title_text}")
                    topic_bookmarks.append((title_text, cur_line_index))
                    text_widget.insert(tk.END, line, "h2")
                elif sline.startswith("### "):
                    title_text = sline[4:].strip()
                    topic_bookmarks.append((title_text, cur_line_index))
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

            def on_topic_click(event=None):
                sel = topic_listbox.curselection()
                if sel:
                    idx = sel[0]
                    if idx < len(topic_bookmarks):
                        pos = topic_bookmarks[idx][1]
                        text_widget.see(pos)
                        text_widget.tag_remove("topic_hl", "1.0", tk.END)
                        end_pos = f"{pos} lineend"
                        text_widget.tag_add("topic_hl", pos, end_pos)
                        text_widget.after(2000, lambda: text_widget.tag_remove("topic_hl", "1.0", tk.END))

            topic_listbox.bind("<<ListboxSelect>>", on_topic_click)

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

    def _safe_open_url(self, url):
        try:
            webbrowser.open(url)
        except Exception as e:
            messagebox.showwarning("Open Browser", f"Could not open link:\n{e}", parent=self.root)

    def _safe_open_folder(self, folder_path):
        try:
            if os.path.isdir(folder_path):
                os.startfile(folder_path)
            elif os.path.isfile(folder_path):
                os.startfile(os.path.dirname(folder_path))
        except Exception as e:
            messagebox.showwarning("Open Folder", f"Could not open directory:\n{e}", parent=self.root)

    def _open_engine_link(self, engine_id):
        from engine_tracker import ENGINE_REGISTRY
        eng = next((e for e in ENGINE_REGISTRY if e["id"] == engine_id), None)
        if eng:
            url = eng.get("doc_url") or eng.get("repo_url")
            if url:
                self._safe_open_url(url)

    def _start_background_engine_check(self):
        """Asynchronously checks upstream GitHub repositories for engine updates."""
        if getattr(self, "_checking_engine_updates", False):
            return
        self._checking_engine_updates = True
        if hasattr(self, "engine_update_btn") and self.engine_update_btn.winfo_exists():
            self.engine_update_btn.configure(
                text="⏳ Checking engines...",
                bg="#242d32",
                fg="#a0adb5"
            )

        def worker():
            try:
                from engine_tracker import check_all_engines
                data = check_all_engines()
            except Exception as e:
                data = {
                    "has_internet": False,
                    "update_count": 0,
                    "total_engines": 10,
                    "error": str(e),
                    "engines": []
                }
            try:
                if self.root and self.root.winfo_exists():
                    self.root.after(0, lambda: self._on_engine_updates_checked(data))
            except Exception:
                pass

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_engine_updates_checked(self, data):
        self._checking_engine_updates = False
        self.engine_tracker_data = data
        if not hasattr(self, "engine_update_btn") or not self.engine_update_btn.winfo_exists():
            return

        has_net = data.get("has_internet", False)
        up_count = data.get("update_count", 0)

        if not has_net:
            self.engine_update_btn.configure(
                text="⚡ Offline (Engines)",
                bg="#242d32",
                fg="#a0adb5",
                font=("Segoe UI", 9)
            )
        elif up_count > 0:
            self.engine_update_btn.configure(
                text=f"🔔 Update Available ({up_count})",
                bg="#d97706",
                fg="#ffffff",
                font=("Segoe UI", 9, "bold")
            )
        else:
            self.engine_update_btn.configure(
                text="✓ Engines Up to Date",
                bg="#242d32",
                fg="#10b981",
                font=("Segoe UI", 9)
            )

    def _open_engine_updates_dialog(self):
        """Displays an interactive modal dialog showing all engine repository versions, commit statuses, and update actions."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Abu Align & Train - Engine Repositories & Update Monitor")
        dialog.configure(bg=self.bg_dark)
        dialog.geometry("980x680")
        dialog.minsize(860, 520)
        dialog.transient(self.root)

        # Header Frame
        hdr = tk.Frame(dialog, bg=self.panel_bg, padx=20, pady=16)
        hdr.pack(fill=tk.X)

        title_lbl = tk.Label(
            hdr, text="📦 Engine Repositories & Upstream Tracking",
            font=("Segoe UI", 16, "bold"), fg=self.text_primary, bg=self.panel_bg
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            hdr, text="Monitors upstream GitHub repositories and compares local commits across all photogrammetry, SfM, and 3DGS engines.",
            font=("Segoe UI", 9), fg=self.text_secondary, bg=self.panel_bg
        )
        sub_lbl.pack(anchor="w", pady=(3, 8))

        # Status summary banner inside header
        summary_frame = tk.Frame(hdr, bg=self.bg_dark, padx=12, pady=8)
        summary_frame.pack(fill=tk.X, pady=(4, 0))

        net_status_lbl = tk.Label(summary_frame, text="Internet: Checking...", font=("Segoe UI", 9, "bold"), fg=self.accent_blue, bg=self.bg_dark)
        net_status_lbl.pack(side=tk.LEFT, padx=(0, 16))

        up_status_lbl = tk.Label(summary_frame, text="", font=("Segoe UI", 9, "bold"), fg="#fbbf24", bg=self.bg_dark)
        up_status_lbl.pack(side=tk.LEFT, padx=(0, 16))

        time_status_lbl = tk.Label(summary_frame, text="", font=("Segoe UI", 9), fg=self.text_secondary, bg=self.bg_dark)
        time_status_lbl.pack(side=tk.RIGHT)

        # Main Table Container
        content_frame = tk.Frame(dialog, bg=self.bg_dark, padx=16, pady=12)
        content_frame.pack(fill=tk.BOTH, expand=True)

        table_frame = tk.Frame(content_frame, bg=self.panel_bg, padx=2, pady=2)
        table_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("name", "category", "applied", "remote", "status")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10, selectmode="browse")
        tree.heading("name", text="Engine Name")
        tree.heading("category", text="Category")
        tree.heading("applied", text="Local Applied Version")
        tree.heading("remote", text="GitHub Upstream (HEAD)")
        tree.heading("status", text="Status")

        tree.column("name", width=170, anchor="w")
        tree.column("category", width=150, anchor="w")
        tree.column("applied", width=150, anchor="w")
        tree.column("remote", width=180, anchor="w")
        tree.column("status", width=140, anchor="w")

        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        tree.tag_configure("update", background="#382512", foreground="#fbbf24")
        tree.tag_configure("uptodate", background="#12281e", foreground="#34d399")
        tree.tag_configure("custom", background="#142436", foreground="#38bdf8")
        tree.tag_configure("offline", background="#1a2024", foreground="#9ca3af")

        # Detail Card
        detail_card = tk.Frame(content_frame, bg=self.panel_bg, padx=16, pady=12, relief="flat", bd=1)
        detail_card.pack(fill=tk.X, pady=(12, 0))

        detail_title = tk.Label(detail_card, text="Select an engine above to inspect repository details", font=("Segoe UI", 11, "bold"), fg=self.text_primary, bg=self.panel_bg)
        detail_title.pack(anchor="w")

        detail_desc = tk.Label(detail_card, text="", font=("Segoe UI", 9), fg=self.text_secondary, bg=self.panel_bg, wraplength=850, justify="left")
        detail_desc.pack(anchor="w", pady=(3, 6))

        detail_info_lbl = tk.Label(detail_card, text="", font=("Consolas", 9), fg=self.accent_blue, bg=self.panel_bg, justify="left")
        detail_info_lbl.pack(anchor="w")

        detail_btn_row = tk.Frame(detail_card, bg=self.panel_bg)
        detail_btn_row.pack(anchor="w", pady=(8, 0))

        github_btn = tk.Button(
            detail_btn_row, text="🌐 View GitHub Repository", bg="#242d32", fg=self.text_primary,
            activebackground="#344149", relief="flat", padx=10, pady=4, font=("Segoe UI", 9), cursor="hand2",
            state=tk.DISABLED
        )
        github_btn.pack(side=tk.LEFT, padx=(0, 8))

        folder_btn = tk.Button(
            detail_btn_row, text="📁 Open Local Directory", bg="#242d32", fg=self.text_primary,
            activebackground="#344149", relief="flat", padx=10, pady=4, font=("Segoe UI", 9), cursor="hand2",
            state=tk.DISABLED
        )
        folder_btn.pack(side=tk.LEFT)

        selected_engine_ref = [None]

        def on_select_engine(event):
            sel = tree.selection()
            if not sel:
                return
            eng_id = sel[0]
            if not self.engine_tracker_data:
                return
            eng = next((e for e in self.engine_tracker_data.get("engines", []) if e["id"] == eng_id), None)
            if not eng:
                return

            selected_engine_ref[0] = eng
            detail_title.configure(text=f"{eng['name']}  ·  {eng['category']}")
            detail_desc.configure(text=eng.get("role", ""))

            loc = eng.get("local", {})
            rem = eng.get("remote")

            loc_path = loc.get("path") or "Integrated in Abu Align / Conda"
            loc_msg = loc.get("message") or ""
            rem_msg = rem.get("message") if rem else "N/A"
            rem_sha = rem.get("short_sha") if rem else "N/A"
            rem_date = rem.get("date") if rem else ""

            info_text = (
                f"• Local Path:    {loc_path}\n"
                f"• Local Commit:  {loc.get('short_sha', 'N/A')} ({loc.get('date', 'N/A')}) - {loc_msg[:65]}\n"
                f"• Remote HEAD:   {rem_sha} ({rem_date}) - {rem_msg[:65]}"
            )
            detail_info_lbl.configure(text=info_text)

            repo_url = eng.get("repo_url") or eng.get("doc_url")
            if repo_url:
                github_btn.configure(state=tk.NORMAL, command=lambda u=repo_url: self._safe_open_url(u))
            else:
                github_btn.configure(state=tk.DISABLED)

            if loc.get("found") and loc.get("path") and os.path.exists(loc["path"]):
                folder_btn.configure(state=tk.NORMAL, command=lambda p=loc["path"]: self._safe_open_folder(p))
            else:
                folder_btn.configure(state=tk.DISABLED)

        tree.bind("<<TreeviewSelect>>", on_select_engine)

        def populate_tree(data):
            for item in tree.get_children():
                tree.delete(item)

            if not data:
                return

            has_net = data.get("has_internet", False)
            up_count = data.get("update_count", 0)

            net_status_lbl.configure(
                text="Internet: Connected" if has_net else "Internet: Offline / Local Only",
                fg="#10b981" if has_net else "#ef4444"
            )
            up_status_lbl.configure(
                text=f"🔔 {up_count} Update{'s' if up_count != 1 else ''} Available!" if up_count > 0 else "✓ All Engines Up to Date",
                fg="#fbbf24" if up_count > 0 else "#10b981"
            )
            time_status_lbl.configure(text=f"Last Checked: {data.get('timestamp', 'N/A')}")

            for eng in data.get("engines", []):
                loc = eng.get("local", {})
                rem = eng.get("remote")

                if loc.get("is_git") and loc.get("short_sha"):
                    applied_str = f"{loc['short_sha']} ({loc.get('date', '')})"
                elif eng.get("pinned_version"):
                    applied_str = eng["pinned_version"]
                elif loc.get("found"):
                    applied_str = "Local Directory"
                else:
                    applied_str = "Integrated Package"

                if rem and rem.get("short_sha"):
                    remote_str = f"{rem['short_sha']} ({rem.get('date', '')})"
                elif not has_net:
                    remote_str = "Offline"
                elif not eng.get("repo_slug"):
                    remote_str = "Native Engine"
                else:
                    remote_str = "Unreachable"

                status_tag = "uptodate"
                st = eng.get("status", "")
                if st == "update_available":
                    status_tag = "update"
                elif st == "custom_ahead":
                    status_tag = "custom"
                elif "offline" in st:
                    status_tag = "offline"

                tree.insert(
                    "", tk.END, iid=eng["id"],
                    values=(eng["name"], eng["category"], applied_str, remote_str, eng.get("status_label", "Unknown")),
                    tags=(status_tag,)
                )

            children = tree.get_children()
            if children:
                update_items = [c for c in children if "update" in tree.item(c, "tags")]
                first_target = update_items[0] if update_items else children[0]
                tree.selection_set(first_target)
                tree.focus(first_target)
                on_select_engine(None)

        # Bottom Action Bar
        bottom_bar = tk.Frame(dialog, bg=self.bg_dark, padx=16, pady=12)
        bottom_bar.pack(fill=tk.X)

        def do_refresh():
            refresh_btn.configure(state=tk.DISABLED, text="⏳ Checking...")
            def bg_check():
                from engine_tracker import check_all_engines
                new_data = check_all_engines()
                def update_ui():
                    self._on_engine_updates_checked(new_data)
                    populate_tree(new_data)
                    refresh_btn.configure(state=tk.NORMAL, text="🔄 Refresh Now")
                dialog.after(0, update_ui)
            threading.Thread(target=bg_check, daemon=True).start()

        refresh_btn = tk.Button(
            bottom_bar, text="🔄 Refresh Now", bg="#242d32", fg=self.text_primary,
            activebackground="#344149", relief="flat", padx=14, pady=6,
            font=("Segoe UI", 9, "bold"), cursor="hand2", command=do_refresh
        )
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))

        def copy_report():
            if not self.engine_tracker_data:
                return
            from engine_tracker import generate_report_text
            rpt = generate_report_text(self.engine_tracker_data)
            self.root.clipboard_clear()
            self.root.clipboard_append(rpt)
            copy_btn.configure(text="✓ Copied to Clipboard!")
            dialog.after(2500, lambda: copy_btn.configure(text="📋 Copy Diagnostic Report"))

        copy_btn = tk.Button(
            bottom_bar, text="📋 Copy Diagnostic Report", bg="#242d32", fg=self.accent_blue,
            activebackground="#344149", relief="flat", padx=14, pady=6,
            font=("Segoe UI", 9), cursor="hand2", command=copy_report
        )
        copy_btn.pack(side=tk.LEFT)

        close_btn = tk.Button(
            bottom_bar, text="Close", bg="#242d32", fg=self.text_primary,
            activebackground="#344149", relief="flat", padx=18, pady=6,
            font=("Segoe UI", 9), cursor="hand2", command=dialog.destroy
        )
        close_btn.pack(side=tk.RIGHT)

        if self.engine_tracker_data:
            populate_tree(self.engine_tracker_data)
        else:
            do_refresh()

    def _show_about_box(self):
        about_text = (
            "Abu Align & Train - Version 0.0.8.5\n"
            "Unified Neural Photogrammetry & 3D Gaussian Splatting Training Suite\n\n"
            "• Photogrammetry Engines: VGGT-Omega, Niantic Labs ACE0, NVIDIA Isaac PyCuSFM, Mapillary OpenSfM (360° Spherical), & 360DVO (Deep 360° Visual Odometry)\n"
            "• 3DGS Training Engines: LichtFeld Studio (Vulkan Real-Time GUI + GUT), Nerfstudio Splatfacto, NVIDIA 3DGRUT, & PFGS360 (CVPR 2026 Pose-Free 360° GS)\n"
            "• Automated Workflows: One-Click Align & Train, Auto-Handoff, & Standalone 3DGS Training\n"
            "• Standardized 3DGS Factory: Universal 3dgs_dataset/ (sparse/0/, images/, transforms.json, points.ply)\n"
            "• Distortion & Optics: Gaussian Unscented Transform (GUT) & Self-Calibrated Brown-Conrady\n"
            "• Telemetry: Real-time progress bars, loss curves, PSNR, and persistent logs in logs/\n"
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
