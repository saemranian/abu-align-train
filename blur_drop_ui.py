"""
Abu Align & Train - BlurDrop Interactive Curation Studio
=========================================================
State-of-the-Art Graphical Studio for Dataset Sharpness Inspection,
Blur Filtering, and Clean Separation for Photogrammetry & 3DGS.
"""

import os
import sys
import time
import math
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable

import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from blur_drop_engine import BlurDropEngine, ImageQualityResult, SUPPORTED_IMAGE_EXTENSIONS

# Theme Colors (Dark Modern Studio)
BG_DARK = "#0d1117"        # GitHub Darkest Navy
SURFACE = "#161b22"        # Card background
SURFACE_LIGHT = "#21262d"  # Highlighted card background
FIELD = "#0d1117"          # Input field background
BORDER = "#30363d"         # Subtle borders
BORDER_ACTIVE = "#58a6ff"  # Focused border
TEXT = "#f0f6fc"           # High contrast text
MUTED = "#8b949e"          # Secondary description text
ACCENT = "#58a6ff"         # Primary blue accent
SUCCESS = "#3fb950"        # Green (Sharp / Keep)
WARNING = "#d29922"        # Amber (Marginal / Warning)
DANGER = "#f85149"         # Red (Blurry / Drop)
HIGHLIGHT = "#1f6feb"      # Active selection


class BlurDropStudioDialog:
    """
    Main Studio Dialog for BlurDrop in Abu Align.
    Supports real-time threshold slider adjustment, multi-metric filtering,
    thumbnail gallery preview, detail inspector loupe, and safe atomic move with undo.
    """

    def __init__(
        self,
        parent: tk.Tk,
        initial_folder: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs
    ):
        self.parent = parent
        self.log_callback = log_callback
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("💧 BlurDrop Studio · Hardware-Accelerated Blur & Sharpness Curation")
        self.dialog.geometry("1340x880")
        self.dialog.minsize(1050, 700)
        self.dialog.configure(bg=BG_DARK)

        # Center dialog
        sw = self.dialog.winfo_screenwidth()
        sh = self.dialog.winfo_screenheight()
        x = max(10, (sw - 1340) // 2)
        y = max(10, (sh - 880) // 2)
        self.dialog.geometry(f"+{x}+{y}")

        # Core Engine
        self.engine = BlurDropEngine()

        # State Variables
        self.folder_var = tk.StringVar(value=initial_folder or "")
        self.dest_folder_var = tk.StringVar(value="")
        self.threshold_var = tk.DoubleVar(value=40.0)
        self.preset_var = tk.StringVar(value="balanced")
        self.motion_sens_var = tk.DoubleVar(value=1.0)
        self.subject_aware_var = tk.BooleanVar(value=False)
        self.canvas_res_var = tk.StringVar(value="🌟 100% Native Full-Canvas (SOTA Accurate)")
        self.inspection_mode_var = tk.StringVar(value="📐 Uniform Full-Canvas (3DGS / Photogrammetry)")
        self.filter_tab_var = tk.StringVar(value="blurry")  # "blurry", "sharp", "all"

        # Selection state tracking (path -> bool checked for drop)
        self.user_selection_overrides: Dict[str, bool] = {}
        self.selected_item_index: Optional[int] = None

        # Threading / Work state
        self.is_scanning: bool = False
        self.cancel_scan_requested: bool = False
        self.thumbnail_cache: Dict[str, ImageTk.PhotoImage] = {}
        self._debounced_reclassify_timer = None

        # Auto-configure destination folder if folder is provided
        if initial_folder and os.path.isdir(initial_folder):
            self.dest_folder_var.set(os.path.join(initial_folder, "blurred_images"))

        # Build UI layout
        self._build_ui()

        # If initial folder has images, trigger scan automatically after UI is visible
        if initial_folder and os.path.isdir(initial_folder):
            self.dialog.after(200, self._start_folder_scan)

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Header Bar
        hdr = tk.Frame(self.dialog, bg=SURFACE, padx=16, pady=10)
        hdr.pack(fill=tk.X, side=tk.TOP)

        hdr_left = tk.Frame(hdr, bg=SURFACE)
        hdr_left.pack(side=tk.LEFT)
        tk.Label(hdr_left, text="💧 BlurDrop Studio", font=("Segoe UI", 14, "bold"), fg=TEXT, bg=SURFACE).pack(anchor="w")
        tk.Label(hdr_left, text="SOTA Multi-Metric Sharpness, Defocus & Motion Blur Curation",
                 font=("Segoe UI", 9), fg=MUTED, bg=SURFACE).pack(anchor="w")

        hdr_right = tk.Frame(hdr, bg=SURFACE)
        hdr_right.pack(side=tk.RIGHT)
        dev_badge = f"⚡ {self.engine.device.upper()} ACCELERATED"
        if torch.cuda.is_available() and self.engine.device.startswith("cuda"):
            dev_name = torch.cuda.get_device_name(0).split()[-2:]
            dev_badge = f"⚡ CUDA ({' '.join(dev_name)})"
        tk.Label(hdr_right, text=dev_badge, font=("Segoe UI", 9, "bold"), fg=ACCENT, bg=FIELD,
                 padx=10, pady=4, relief=tk.FLAT).pack(side=tk.RIGHT)

        # 2. Main Content Split (Paned Window: Left Controls & Gallery, Right Inspector)
        main_pane = tk.PanedWindow(self.dialog, orient=tk.HORIZONTAL, bg=BG_DARK, sashwidth=6, sashrelief=tk.FLAT)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 6))

        # Left Column Frame (Folder bar + Controls + Gallery)
        left_col = tk.Frame(main_pane, bg=BG_DARK)
        main_pane.add(left_col, minsize=650, width=900)

        # Right Column Frame (Detailed Inspector Loupe)
        self.right_col = tk.Frame(main_pane, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        main_pane.add(self.right_col, minsize=350, width=420)

        # Build Left Components
        self._build_folder_card(left_col)
        self._build_tuning_card(left_col)
        self._build_gallery_section(left_col)

        # Build Right Components (Inspector)
        self._build_inspector_panel(self.right_col)

        # 3. Bottom Action Bar (Safe Move & Undo)
        self._build_action_bar()

    def _build_folder_card(self, parent: tk.Frame):
        """Top card for folder selection, scan trigger, and progress bar."""
        card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=10)
        card.pack(fill=tk.X, pady=(0, 8))

        r1 = tk.Frame(card, bg=SURFACE)
        r1.pack(fill=tk.X)
        tk.Label(r1, text="Image Folder:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=12, anchor="w").pack(side=tk.LEFT)
        self.folder_entry = tk.Entry(r1, textvariable=self.folder_var, font=("Segoe UI", 9),
                                     bg=FIELD, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)

        self.btn_browse = tk.Button(r1, text="📁 Browse", command=self._browse_folder,
                                    font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, cursor="hand2")
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_scan = tk.Button(r1, text="⚡ Scan & Analyze", command=self._start_folder_scan,
                                  font=("Segoe UI", 9, "bold"), bg=HIGHLIGHT, fg=TEXT, activebackground=ACCENT,
                                  relief=tk.FLAT, padx=14, pady=4, cursor="hand2")
        self.btn_scan.pack(side=tk.LEFT)

        # Progress bar & live telemetry row
        r2 = tk.Frame(card, bg=SURFACE)
        r2.pack(fill=tk.X, pady=(6, 0))
        self.scan_progress = ttk.Progressbar(r2, orient="horizontal", mode="determinate")
        self.scan_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.scan_status_lbl = tk.Label(r2, text="Ready to scan folder.", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.scan_status_lbl.pack(side=tk.RIGHT)

    def _build_tuning_card(self, parent: tk.Frame):
        """Card containing the live threshold slider, sensitivity presets, and filters."""
        card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=8)
        card.pack(fill=tk.X, pady=(0, 8))

        # Row 1: Presets & Threshold Header
        r1 = tk.Frame(card, bg=SURFACE)
        r1.pack(fill=tk.X, pady=(0, 4))
        tk.Label(r1, text="Cutoff Threshold Filter:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT)

        self.thresh_val_lbl = tk.Label(r1, text="40% (Drop below 40)", font=("Segoe UI", 9, "bold"), fg=DANGER, bg=SURFACE)
        self.thresh_val_lbl.pack(side=tk.LEFT, padx=(6, 0))

        # Quick Preset Buttons
        r_presets = tk.Frame(r1, bg=SURFACE)
        r_presets.pack(side=tk.RIGHT)
        tk.Label(r_presets, text="Presets:", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(r_presets, text="🟢 Balanced (40%)", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT,
                  padx=6, pady=2, command=lambda: self._apply_preset("balanced", 40.0), cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(r_presets, text="🔴 Strict (55%)", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT,
                  padx=6, pady=2, command=lambda: self._apply_preset("strict", 55.0), cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(r_presets, text="🟡 Permissive (25%)", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT,
                  padx=6, pady=2, command=lambda: self._apply_preset("permissive", 25.0), cursor="hand2").pack(side=tk.LEFT, padx=2)

        # Row 2: Live Threshold Slider
        r2 = tk.Frame(card, bg=SURFACE)
        r2.pack(fill=tk.X, pady=(2, 6))
        self.thresh_scale = ttk.Scale(r2, from_=5, to=95, variable=self.threshold_var,
                                      command=self._on_threshold_slider_moved)
        self.thresh_scale.pack(fill=tk.X, expand=True)

        # Row 3: Canvas Scale & Inspection Mode
        r3 = tk.Frame(card, bg=SURFACE)
        r3.pack(fill=tk.X, pady=(2, 4))

        tk.Label(r3, text="Canvas Scale:", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        self.canvas_res_combo = ttk.Combobox(
            r3, textvariable=self.canvas_res_var,
            values=["🌟 100% Native Full-Canvas (SOTA Accurate)", "⚡ 2048px Fast Preview"],
            state="readonly", width=34
        )
        self.canvas_res_combo.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(r3, text="Mode:", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        def _on_inspection_mode_changed(*_):
            is_subj = "Subject" in self.inspection_mode_var.get()
            self.subject_aware_var.set(is_subj)
            self._trigger_debounced_reclassify()

        self.inspection_mode_combo = ttk.Combobox(
            r3, textvariable=self.inspection_mode_var,
            values=["📐 Uniform Full-Canvas (3DGS / Photogrammetry)", "🎯 Subject Focus (Bokeh Invariant)"],
            state="readonly", width=42
        )
        self.inspection_mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.inspection_mode_combo.bind("<<ComboboxSelected>>", _on_inspection_mode_changed)

        # Row 4: Motion Streak Sensitivity & Subject Checkbox
        r4 = tk.Frame(card, bg=SURFACE)
        r4.pack(fill=tk.X, pady=(2, 2))

        tk.Label(r4, text="Motion Streak Sensitivity:", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        self.s_motion = ttk.Scale(r4, from_=0.5, to=2.0, variable=self.motion_sens_var, length=120,
                                  command=lambda v: self._trigger_debounced_reclassify())
        self.s_motion.pack(side=tk.LEFT, padx=(0, 16))

        def _on_chk_dof_toggled():
            if self.subject_aware_var.get():
                self.inspection_mode_var.set("🎯 Subject Focus (Bokeh Invariant)")
            else:
                self.inspection_mode_var.set("📐 Uniform Full-Canvas (3DGS / Photogrammetry)")
            self._trigger_debounced_reclassify()

        self.chk_dof = tk.Checkbutton(
            r4, text="Bokeh Invariant Focus (Preserve sharp subject with blurred background)",
            variable=self.subject_aware_var, font=("Segoe UI", 8), fg=ACCENT, bg=SURFACE,
            selectcolor=FIELD, command=_on_chk_dof_toggled
        )
        self.chk_dof.pack(side=tk.LEFT)

    def _build_gallery_section(self, parent: tk.Frame):
        """Builds the tabbed gallery of image thumbnail cards."""
        gallery_frame = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        gallery_frame.pack(fill=tk.BOTH, expand=True)

        # Filter Tabs & Telemetry Row
        tab_row = tk.Frame(gallery_frame, bg=SURFACE_LIGHT, padx=10, pady=6)
        tab_row.pack(fill=tk.X)

        self.btn_tab_blurry = tk.Button(tab_row, text="🔴 Blurry to Drop (0)", font=("Segoe UI", 9, "bold"),
                                        bg=DANGER, fg=TEXT, relief=tk.FLAT, padx=10, pady=4,
                                        command=lambda: self._switch_tab("blurry"), cursor="hand2")
        self.btn_tab_blurry.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_tab_sharp = tk.Button(tab_row, text="🟢 Sharp to Keep (0)", font=("Segoe UI", 9),
                                       bg=SURFACE, fg=MUTED, relief=tk.FLAT, padx=10, pady=4,
                                       command=lambda: self._switch_tab("sharp"), cursor="hand2")
        self.btn_tab_sharp.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_tab_all = tk.Button(tab_row, text="📋 All Images (0)", font=("Segoe UI", 9),
                                     bg=SURFACE, fg=MUTED, relief=tk.FLAT, padx=10, pady=4,
                                     command=lambda: self._switch_tab("all"), cursor="hand2")
        self.btn_tab_all.pack(side=tk.LEFT)

        # Selection Quick Actions
        r_acts = tk.Frame(tab_row, bg=SURFACE_LIGHT)
        r_acts.pack(side=tk.RIGHT)
        tk.Button(r_acts, text="Select All", font=("Segoe UI", 8), bg=SURFACE, fg=TEXT, relief=tk.FLAT,
                  padx=6, pady=2, command=self._select_all_visible, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(r_acts, text="Deselect All", font=("Segoe UI", 8), bg=SURFACE, fg=TEXT, relief=tk.FLAT,
                  padx=6, pady=2, command=self._deselect_all_visible, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(r_acts, text="Invert", font=("Segoe UI", 8), bg=SURFACE, fg=TEXT, relief=tk.FLAT,
                  padx=6, pady=2, command=self._invert_selection, cursor="hand2").pack(side=tk.LEFT, padx=2)

        # Scrollable Canvas for Thumbnail Cards
        canvas_container = tk.Frame(gallery_frame, bg=BG_DARK)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.gallery_canvas = tk.Canvas(canvas_container, bg=BG_DARK, highlightthickness=0)
        self.gallery_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        self.gallery_scroll_frame = tk.Frame(self.gallery_canvas, bg=BG_DARK)

        self.gallery_scroll_frame.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        )
        self.gallery_window_id = self.gallery_canvas.create_window((0, 0), window=self.gallery_scroll_frame, anchor="nw")
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel support
        def _on_mousewheel(e):
            delta = e.delta if hasattr(e, "delta") and e.delta else (120 if getattr(e, "num", None) == 4 else -120)
            self.gallery_canvas.yview_scroll(int(-1 * (delta / 120)), "units")

        self.gallery_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.gallery_canvas.bind("<Button-4>", _on_mousewheel)
        self.gallery_canvas.bind("<Button-5>", _on_mousewheel)

        self.gallery_canvas.bind("<Configure>", lambda e: self.gallery_canvas.itemconfig(self.gallery_window_id, width=e.width))

    def _build_inspector_panel(self, parent: tk.Frame):
        """Right panel for magnified preview, sharpness gauges, and 100% crop loupe."""
        p_hdr = tk.Frame(parent, bg=SURFACE_LIGHT, padx=10, pady=8)
        p_hdr.pack(fill=tk.X)
        tk.Label(p_hdr, text="🔍 Detail Inspector Loupe", font=("Segoe UI", 10, "bold"), fg=TEXT, bg=SURFACE_LIGHT).pack(side=tk.LEFT)

        # Image Large Preview
        self.inspector_preview_lbl = tk.Label(parent, bg=BG_DARK, text="Select an image\nto inspect details",
                                              font=("Segoe UI", 9), fg=MUTED, height=12)
        self.inspector_preview_lbl.pack(fill=tk.BOTH, expand=False, padx=8, pady=8)

        # Filename & Status
        self.insp_name_lbl = tk.Label(parent, text="No image selected", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, wraplength=340)
        self.insp_name_lbl.pack(anchor="w", padx=10)

        self.insp_score_lbl = tk.Label(parent, text="", font=("Segoe UI", 12, "bold"), fg=DANGER, bg=SURFACE)
        self.insp_score_lbl.pack(anchor="w", padx=10, pady=(2, 6))

        # Metric Breakdown Gauges
        gauge_box = tk.Frame(parent, bg=SURFACE, padx=10)
        gauge_box.pack(fill=tk.X, pady=4)

        def _make_metric_row(lbl_text, col):
            r = tk.Frame(gauge_box, bg=SURFACE)
            r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=lbl_text, font=("Segoe UI", 8), fg=MUTED, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(r, text="--", font=("Segoe UI", 8, "bold"), fg=col, bg=SURFACE, width=6, anchor="e")
            val_lbl.pack(side=tk.RIGHT)
            bar = ttk.Progressbar(r, orient="horizontal", mode="determinate", length=140)
            bar.pack(side=tk.RIGHT, padx=4)
            return val_lbl, bar

        self.lbl_gauge_defocus, self.bar_gauge_defocus = _make_metric_row("Re-Blur Sharpness:", ACCENT)
        self.lbl_gauge_peak, self.bar_gauge_peak = _make_metric_row("Edge Peakiness:", SUCCESS)
        self.lbl_gauge_motion, self.bar_gauge_motion = _make_metric_row("Motion Resistance:", WARNING)
        self.lbl_gauge_uniformity, self.bar_gauge_uniformity = _make_metric_row("Canvas Uniformity:", TEXT)
        self.lbl_gauge_subject, self.bar_gauge_subject = _make_metric_row("Peak Subject Focus:", "#38bdf8")

        # 100% Crop Loupe
        tk.Label(parent, text="🔬 Sharpest 100% Region Crop:", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=SURFACE).pack(anchor="w", padx=10, pady=(10, 2))
        self.loupe_lbl = tk.Label(parent, bg=BG_DARK, text="Crop preview", font=("Segoe UI", 8), fg=MUTED, height=8)
        self.loupe_lbl.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _build_action_bar(self):
        """Bottom action bar for cutting/moving blurry images into a new folder."""
        bar = tk.Frame(self.dialog, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=10)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        r_dest = tk.Frame(bar, bg=SURFACE)
        r_dest.pack(fill=tk.X, pady=(0, 6))

        tk.Label(r_dest, text="Cut Blurry Images To:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        self.dest_entry = tk.Entry(r_dest, textvariable=self.dest_folder_var, font=("Segoe UI", 9),
                                   bg=FIELD, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT)
        self.dest_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)

        self.btn_browse_dest = tk.Button(r_dest, text="Browse...", command=self._browse_dest_folder,
                                         font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, cursor="hand2")
        self.btn_browse_dest.pack(side=tk.LEFT)

        r_btns = tk.Frame(bar, bg=SURFACE)
        r_btns.pack(fill=tk.X)

        self.btn_move = tk.Button(
            r_btns, text="✂️ Move Selected Blurry Images into New Folder",
            font=("Segoe UI", 10, "bold"), bg=DANGER, fg=TEXT, activebackground="#dc2626",
            relief=tk.FLAT, padx=16, pady=6, cursor="hand2", command=self._execute_cut_and_move
        )
        self.btn_move.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_undo = tk.Button(
            r_btns, text="↩ Undo Last Move (Restore Images)",
            font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=MUTED, relief=tk.FLAT, padx=12, pady=6,
            state=tk.DISABLED, cursor="hand2", command=self._execute_undo_move
        )
        self.btn_undo.pack(side=tk.LEFT)

        self.action_status_lbl = tk.Label(r_btns, text="", font=("Segoe UI", 9), fg=SUCCESS, bg=SURFACE)
        self.action_status_lbl.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # Core Scan & Folder Operations
    # -------------------------------------------------------------------------

    def _browse_folder(self):
        f = filedialog.askdirectory(title="Select Folder of Images to Analyze", parent=self.dialog)
        if f:
            self.folder_var.set(f)
            if not self.dest_folder_var.get().strip():
                self.dest_folder_var.set(os.path.join(f, "blurred_images"))
            self._start_folder_scan()

    def _browse_dest_folder(self):
        f = filedialog.askdirectory(title="Select Destination Folder for Blurry Images", parent=self.dialog)
        if f:
            self.dest_folder_var.set(f)

    def _start_folder_scan(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Invalid Folder", "Please select a valid folder containing images.", parent=self.dialog)
            return

        if self.is_scanning:
            return

        self.is_scanning = True
        self.cancel_scan_requested = False
        self.btn_scan.configure(text="⏳ Scanning...", state=tk.DISABLED)
        self.scan_progress["value"] = 0
        self.scan_status_lbl.configure(text="Finding images...", fg=ACCENT)
        self.user_selection_overrides.clear()

        thresh = self.threshold_var.get()
        preset = self.preset_var.get()
        motion_sens = self.motion_sens_var.get()
        subj_aware = self.subject_aware_var.get() or ("Subject" in self.inspection_mode_var.get())
        insp_mode = "subject" if subj_aware else "uniform"
        canvas_res = None if ("100%" in self.canvas_res_var.get() or self.canvas_res_var.get() == "native") else 2048

        def _prog(cur, tot, msg):
            pct = int(cur * 100 / tot)
            self.dialog.after(0, lambda: self._update_scan_progress(pct, msg))

        def _worker():
            try:
                results = self.engine.analyze_folder(
                    folder_path=folder,
                    threshold=thresh,
                    preset=preset,
                    motion_sensitivity=motion_sens,
                    subject_aware=subj_aware,
                    inspection_mode=insp_mode,
                    canvas_res=canvas_res,
                    progress_callback=_prog,
                    cancel_token=lambda: self.cancel_scan_requested
                )
                self.dialog.after(0, lambda: self._on_scan_completed(results))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_scan_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_scan_progress(self, pct: int, msg: str):
        self.scan_progress["value"] = pct
        self.scan_status_lbl.configure(text=msg)

    def _on_scan_completed(self, results: List[ImageQualityResult]):
        self.is_scanning = False
        self.btn_scan.configure(text="⚡ Scan & Analyze", state=tk.NORMAL)
        self.scan_progress["value"] = 100
        count = len(results)
        self.scan_status_lbl.configure(text=f"Scan complete: {count} images analyzed successfully.", fg=SUCCESS)
        self._refresh_gallery()

    def _on_scan_error(self, err: str):
        self.is_scanning = False
        self.btn_scan.configure(text="⚡ Scan & Analyze", state=tk.NORMAL)
        self.scan_status_lbl.configure(text=f"Scan Error: {err}", fg=DANGER)
        messagebox.showerror("Scan Error", f"Failed to analyze images:\n{err}", parent=self.dialog)

    # -------------------------------------------------------------------------
    # Threshold Tuning & Interactive Presets
    # -------------------------------------------------------------------------

    def _on_threshold_slider_moved(self, val):
        t = round(float(val), 1)
        self.thresh_val_lbl.configure(text=f"{t}% (Drop below {t})")
        self._trigger_debounced_reclassify()

    def _apply_preset(self, preset_name: str, thresh: float):
        self.preset_var.set(preset_name)
        self.threshold_var.set(thresh)
        self.thresh_val_lbl.configure(text=f"{thresh}% (Drop below {thresh})")
        self._trigger_debounced_reclassify()

    def _trigger_debounced_reclassify(self):
        if self._debounced_reclassify_timer is not None:
            self.dialog.after_cancel(self._debounced_reclassify_timer)
        self._debounced_reclassify_timer = self.dialog.after(30, self._reclassify_and_refresh)

    def _reclassify_and_refresh(self):
        self._debounced_reclassify_timer = None
        if not self.engine.cached_results:
            return

        thresh = self.threshold_var.get()
        preset = self.preset_var.get()
        motion_sens = self.motion_sens_var.get()
        subj_aware = self.subject_aware_var.get() or ("Subject" in self.inspection_mode_var.get())
        insp_mode = "subject" if subj_aware else "uniform"

        self.engine.reclassify_all(
            threshold=thresh,
            preset=preset,
            motion_sensitivity=motion_sens,
            subject_aware=subj_aware,
            inspection_mode=insp_mode
        )
        self._refresh_gallery()

    # -------------------------------------------------------------------------
    # Gallery Rendering & Inspection
    # -------------------------------------------------------------------------

    def _switch_tab(self, tab_name: str):
        self.filter_tab_var.set(tab_name)
        # Update tab button styling
        self.btn_tab_blurry.configure(bg=DANGER if tab_name == "blurry" else SURFACE, fg=TEXT if tab_name == "blurry" else MUTED)
        self.btn_tab_sharp.configure(bg=SUCCESS if tab_name == "sharp" else SURFACE, fg=TEXT if tab_name == "sharp" else MUTED)
        self.btn_tab_all.configure(bg=HIGHLIGHT if tab_name == "all" else SURFACE, fg=TEXT if tab_name == "all" else MUTED)
        self._refresh_gallery()

    def _is_image_selected_for_drop(self, r: ImageQualityResult) -> bool:
        """Returns True if this image is selected to be dropped/cut."""
        if r.path in self.user_selection_overrides:
            return self.user_selection_overrides[r.path]
        return r.is_blurry

    def _refresh_gallery(self):
        """Cleans and re-renders the thumbnail cards in the gallery."""
        # Clear existing card widgets
        for child in self.gallery_scroll_frame.winfo_children():
            child.destroy()

        results = self.engine.cached_results
        total = len(results)
        blurry_count = sum(1 for r in results if self._is_image_selected_for_drop(r))
        sharp_count = total - blurry_count

        # Update tab counts
        self.btn_tab_blurry.configure(text=f"🔴 Blurry to Drop ({blurry_count})")
        self.btn_tab_sharp.configure(text=f"🟢 Sharp to Keep ({sharp_count})")
        self.btn_tab_all.configure(text=f"📋 All Images ({total})")

        # Update bottom move button text
        self.btn_move.configure(text=f"✂️ Move {blurry_count} Blurry Images into New Folder")

        tab = self.filter_tab_var.get()
        if tab == "blurry":
            display_items = [r for r in results if self._is_image_selected_for_drop(r)]
        elif tab == "sharp":
            display_items = [r for r in results if not self._is_image_selected_for_drop(r)]
        else:
            display_items = results

        if not display_items:
            empty_lbl = tk.Label(
                self.gallery_scroll_frame,
                text="No images in this filter view.\nAdjust threshold or switch tabs to inspect photos.",
                font=("Segoe UI", 10), fg=MUTED, bg=BG_DARK, pady=60
            )
            empty_lbl.pack(fill=tk.BOTH, expand=True)
            return

        # Grid layout: 4 columns of cards
        cols = 4
        for idx, r in enumerate(display_items):
            row = idx // cols
            col = idx % cols
            self._render_thumbnail_card(r, row, col)

        # Inspect first item by default if none selected
        if display_items and self.selected_item_index is None:
            self._select_and_inspect_image(display_items[0])

    def _render_thumbnail_card(self, r: ImageQualityResult, row: int, col: int):
        """Renders an individual thumbnail card in the gallery grid."""
        is_drop = self._is_image_selected_for_drop(r)
        border_col = DANGER if is_drop else SUCCESS
        badge_bg = DANGER if is_drop else SUCCESS

        card = tk.Frame(
            self.gallery_scroll_frame, bg=SURFACE, highlightbackground=border_col,
            highlightthickness=2, padx=4, pady=4, cursor="hand2"
        )
        card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        # Top Bar: Checkbox + Score badge
        card_top = tk.Frame(card, bg=SURFACE)
        card_top.pack(fill=tk.X, pady=(0, 2))

        cb_var = tk.BooleanVar(value=is_drop)

        def _on_toggle():
            self.user_selection_overrides[r.path] = cb_var.get()
            self._refresh_gallery()

        chk = tk.Checkbutton(card_top, variable=cb_var, bg=SURFACE, selectcolor=FIELD, command=_on_toggle)
        chk.pack(side=tk.LEFT)

        score_text = f"{r.composite_score:.1f}"
        tk.Label(card_top, text=score_text, font=("Segoe UI", 8, "bold"), fg=TEXT, bg=badge_bg, padx=4).pack(side=tk.RIGHT)

        # Image Thumbnail Preview
        thumb_img = self._get_thumbnail(r.path)
        img_lbl = tk.Label(card, image=thumb_img, bg=FIELD)
        img_lbl.image = thumb_img  # Keep reference
        img_lbl.pack(pady=2)

        # Bottom Filename & Blur Classification
        class_text = r.classification.replace("_", " ")
        lbl_info = tk.Label(card, text=f"{r.filename[:16]}\n[{class_text}]",
                            font=("Segoe UI", 7), fg=TEXT if not is_drop else DANGER, bg=SURFACE, justify=tk.CENTER)
        lbl_info.pack(fill=tk.X)

        # Bind card click to inspection
        def _on_click(e):
            self._select_and_inspect_image(r)

        card.bind("<Button-1>", _on_click)
        img_lbl.bind("<Button-1>", _on_click)
        lbl_info.bind("<Button-1>", _on_click)

    def _get_thumbnail(self, path: str, size: Tuple[int, int] = (140, 105)) -> ImageTk.PhotoImage:
        """Retrieves or creates a cached PIL thumbnail for smooth scrolling."""
        if path in self.thumbnail_cache:
            return self.thumbnail_cache[path]
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail(size, Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(pil_img)
            self.thumbnail_cache[path] = photo
            return photo
        except Exception:
            # Fallback blank image
            blank = Image.new("RGB", size, (30, 30, 30))
            photo = ImageTk.PhotoImage(blank)
            self.thumbnail_cache[path] = photo
            return photo

    def _select_and_inspect_image(self, r: ImageQualityResult):
        """Displays large inspection preview, metric gauges, and 100% crop loupe."""
        self.insp_name_lbl.configure(text=f"{r.filename} ({r.width}x{r.height}, {r.file_size_mb}MB)")
        is_drop = self._is_image_selected_for_drop(r)
        status_text = f"Score: {r.composite_score:.1f} / 100 · [{r.classification.replace('_', ' ')}]"
        self.insp_score_lbl.configure(text=status_text, fg=DANGER if is_drop else SUCCESS)

        # Update Gauges
        self.lbl_gauge_defocus.configure(text=f"{r.defocus_score:.1f}")
        self.bar_gauge_defocus["value"] = r.defocus_score

        self.lbl_gauge_peak.configure(text=f"{r.edge_spread:.2f}x")
        self.bar_gauge_peak["value"] = min(100.0, max(0.0, (r.edge_spread - 1.0) / 2.5 * 100.0))

        self.lbl_gauge_motion.configure(text=f"{r.motion_score:.1f}")
        self.bar_gauge_motion["value"] = r.motion_score

        self.lbl_gauge_uniformity.configure(text=f"{r.uniformity:.1f}%")
        self.bar_gauge_uniformity["value"] = r.uniformity

        self.lbl_gauge_subject.configure(text=f"{r.subject_patch_score:.1f}")
        self.bar_gauge_subject["value"] = r.subject_patch_score

        # Render Large Preview
        try:
            pil_full = Image.open(r.path)
            orig_w, orig_h = pil_full.size
            # Scale to fit ~340x220 preview
            scale = min(340.0 / orig_w, 220.0 / orig_h)
            pw = max(32, int(round(orig_w * scale)))
            ph = max(32, int(round(orig_h * scale)))
            prev_img = pil_full.resize((pw, ph), Image.Resampling.BILINEAR)
            p_photo = ImageTk.PhotoImage(prev_img)
            self.inspector_preview_lbl.configure(image=p_photo, text="")
            self.inspector_preview_lbl.image = p_photo

            # 100% Crop Loupe of sharpest region
            # Extract a 250x150 region around image center or top patch
            cx, cy = orig_w // 2, orig_h // 2
            cw, ch = 250, 150
            x0 = max(0, cx - cw // 2)
            y0 = max(0, cy - ch // 2)
            x1 = min(orig_w, x0 + cw)
            y1 = min(orig_h, y0 + ch)
            crop_img = pil_full.crop((x0, y0, x1, y1))
            crop_photo = ImageTk.PhotoImage(crop_img)
            self.loupe_lbl.configure(image=crop_photo, text="")
            self.loupe_lbl.image = crop_photo
        except Exception as e:
            self.inspector_preview_lbl.configure(image="", text=f"Preview unavailable:\n{e}")
            self.loupe_lbl.configure(image="", text="Crop unavailable")

    # -------------------------------------------------------------------------
    # Selection Overrides (Quick Actions)
    # -------------------------------------------------------------------------

    def _select_all_visible(self):
        tab = self.filter_tab_var.get()
        for r in self.engine.cached_results:
            if tab == "blurry" and not self._is_image_selected_for_drop(r):
                continue
            if tab == "sharp" and self._is_image_selected_for_drop(r):
                continue
            self.user_selection_overrides[r.path] = True
        self._refresh_gallery()

    def _deselect_all_visible(self):
        tab = self.filter_tab_var.get()
        for r in self.engine.cached_results:
            if tab == "blurry" and not self._is_image_selected_for_drop(r):
                continue
            if tab == "sharp" and self._is_image_selected_for_drop(r):
                continue
            self.user_selection_overrides[r.path] = False
        self._refresh_gallery()

    def _invert_selection(self):
        tab = self.filter_tab_var.get()
        for r in self.engine.cached_results:
            cur = self._is_image_selected_for_drop(r)
            self.user_selection_overrides[r.path] = not cur
        self._refresh_gallery()

    # -------------------------------------------------------------------------
    # Execution: Cut / Move Blurry Images & Undo
    # -------------------------------------------------------------------------

    def _execute_cut_and_move(self):
        results = self.engine.cached_results
        if not results:
            messagebox.showinfo("No Images", "Please scan a folder first.", parent=self.dialog)
            return

        dest = self.dest_folder_var.get().strip()
        if not dest:
            messagebox.showwarning("Missing Destination", "Please specify a destination folder for blurry images.", parent=self.dialog)
            return

        to_move = [r.path for r in results if self._is_image_selected_for_drop(r)]
        count = len(to_move)

        if count == 0:
            messagebox.showinfo("Zero Blurry Images", "No images are currently selected for dropping!", parent=self.dialog)
            return

        confirm_msg = (
            f"Are you sure you want to move {count} blurry images into:\n\n"
            f"{dest}\n\n"
            "This will cleanly cut the images out of the main dataset folder,\n"
            "leaving it 100% sharp and ready for photogrammetry training.\n"
            "(You can Undo this operation at any time)."
        )
        if not messagebox.askyesno("Confirm Move", confirm_msg, parent=self.dialog):
            return

        self.btn_move.configure(state=tk.DISABLED)
        self.action_status_lbl.configure(text=f"Moving {count} images...", fg=WARNING)

        def _prog(cur, tot, msg):
            self.dialog.after(0, lambda: self.action_status_lbl.configure(text=msg))

        def _worker():
            try:
                res = self.engine.cut_and_move_blurry_images(
                    destination_dir=dest,
                    selected_paths=to_move,
                    progress_callback=_prog
                )
                self.dialog.after(0, lambda: self._on_move_completed(res))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_move_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_move_completed(self, res: Dict[str, Any]):
        self.btn_move.configure(state=tk.NORMAL)
        self.btn_undo.configure(state=tk.NORMAL, fg=TEXT)
        moved = res["moved_count"]
        rem = res["remaining_sharp_count"]
        dest = res["destination_dir"]

        success_msg = f"Cleaned! {moved} blurry images moved to:\n{dest}\n\nMain folder now contains {rem} clean, sharp images!"
        self.action_status_lbl.configure(text=f"Cleaned! {moved} images moved. {rem} sharp images remain.", fg=SUCCESS)
        messagebox.showinfo("Curation Complete", success_msg, parent=self.dialog)

        # Clear thumbnail cache for moved images and refresh
        self.thumbnail_cache.clear()
        self._refresh_gallery()

    def _on_move_error(self, err: str):
        self.btn_move.configure(state=tk.NORMAL)
        self.action_status_lbl.configure(text=f"Move Error: {err}", fg=DANGER)
        messagebox.showerror("Move Error", f"Failed to move images:\n{err}", parent=self.dialog)

    def _execute_undo_move(self):
        if not self.engine.last_move_manifest:
            messagebox.showinfo("Undo Unavailable", "No recent move operation to undo.", parent=self.dialog)
            return

        count = len(self.engine.last_move_manifest)
        if not messagebox.askyesno("Confirm Undo", f"Restore all {count} previously moved images back to their original locations?", parent=self.dialog):
            return

        self.btn_undo.configure(state=tk.DISABLED)
        self.action_status_lbl.configure(text=f"Restoring {count} images...", fg=WARNING)

        def _prog(cur, tot, msg):
            self.dialog.after(0, lambda: self.action_status_lbl.configure(text=msg))

        def _worker():
            try:
                res = self.engine.undo_last_move(progress_callback=_prog)
                self.dialog.after(0, lambda: self._on_undo_completed(res))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_undo_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_undo_completed(self, res: Dict[str, Any]):
        self.btn_undo.configure(state=tk.DISABLED, fg=MUTED)
        restored = res["restored_count"]
        self.action_status_lbl.configure(text=f"Restored {restored} images back to main dataset.", fg=SUCCESS)
        messagebox.showinfo("Undo Complete", f"Successfully restored {restored} images back to the main folder.", parent=self.dialog)
        # Re-scan to synchronize state
        self._start_folder_scan()

    def _on_undo_error(self, err: str):
        self.btn_undo.configure(state=tk.NORMAL)
        self.action_status_lbl.configure(text=f"Undo Error: {err}", fg=DANGER)
        messagebox.showerror("Undo Error", f"Failed to restore images:\n{err}", parent=self.dialog)


def open_blur_drop_dialog(
    parent: tk.Tk,
    initial_folder: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs
) -> BlurDropStudioDialog:
    """Helper function to instantiate and display the BlurDrop Studio."""
    return BlurDropStudioDialog(parent, initial_folder=initial_folder, log_callback=log_callback, *args, **kwargs)


if __name__ == "__main__":
    # Standalone execution
    root = tk.Tk()
    root.withdraw()
    init_dir = sys.argv[1] if len(sys.argv) > 1 else None
    dlg = open_blur_drop_dialog(root, initial_folder=init_dir)
    dlg.dialog.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
