"""
Abu Align - SAM 2 AI Masking Interactive UI
Professional GUI dialog for prompt-based segmentation, keyframing, timeline tracking,
mask enhancement, and RealityScan photogrammetry mask export.
"""

import os
import sys
import time
import math
import threading
from typing import Optional, List, Dict, Tuple, Any, Callable

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import torch

from sam2_engine import SAM2MaskingEngine, SAM2_MODELS

# UI Palette (consistent with Abu Align desktop theme)
BG = "#111518"
SURFACE = "#1a2024"
FIELD = "#242d32"
BORDER = "#344149"
TEXT = "#e7edef"
MUTED = "#a0adb5"
ACCENT = "#70dbc0"
ACCENT_HOVER = "#8fe5d0"
DANGER = "#f43f5e"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DISABLED = "#526068"

MASK_COLORS = {
    "Emerald": (16, 185, 129),
    "Ruby Red": (239, 68, 68),
    "Cyan": (6, 182, 212),
    "Amber Gold": (245, 158, 11),
    "Violet": (168, 85, 247)
}


class SAM2MaskerDialog:
    """
    Main dialog window providing an interactive workspace for AI masking.
    """

    def __init__(self, parent: tk.Widget, initial_path: Optional[str] = None, log_callback: Optional[Callable[[str], None]] = None):
        self.parent = parent
        self.log_callback = log_callback
        self.engine = SAM2MaskingEngine()
        if self.log_callback:
            self.engine.set_log_callback(self.log_callback)

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🎭 Abu Align · AI Subject Masker (SAM 2.1)")
        self.dialog.geometry("1400x880")
        self.dialog.minsize(1120, 720)
        self.dialog.configure(bg=BG)
        self.dialog.transient(parent)
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        # Session State
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.play_fps: int = 24
        self.is_tracking: bool = False
        self.cancel_requested: bool = False

        # Canvas Transform & Interaction
        self.zoom: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.drag_start_x: float = 0.0
        self.drag_start_y: float = 0.0
        self.is_panning: bool = False
        self.box_start: Optional[Tuple[float, float]] = None
        self.current_box_rect_id = None

        # View Mode State
        self.view_mode = tk.StringVar(value="Overlay")  # Overlay, Cutout, Matte
        self.mask_color_name = tk.StringVar(value="Ruby Red")
        self.overlay_opacity = tk.DoubleVar(value=0.55)
        self.prompt_mode = tk.StringVar(value="Foreground")  # Foreground (+), Background (-), Box

        # Post-Processing Parameters
        self.dilation_var = tk.IntVar(value=0)
        self.feather_var = tk.IntVar(value=2)
        self.threshold_var = tk.IntVar(value=128)
        self.fill_holes_var = tk.BooleanVar(value=True)
        self.min_speckle_var = tk.IntVar(value=30)
        self.invert_var = tk.BooleanVar(value=False)

        # Alt key state for Erase / Subtract mode
        self.is_alt_pressed: bool = False

        # Enhanced Export Settings & Codecs (Cutout is optional with checkmark)
        self.export_dir_var = tk.StringVar(value="")
        self.export_binary_var = tk.BooleanVar(value=True)
        self.export_cutout_var = tk.BooleanVar(value=False)
        self.export_rs_naming_var = tk.BooleanVar(value=True)
        self.mask_fmt_var = tk.StringVar(value="png")
        self.cutout_fmt_var = tk.StringVar(value="WebP (RGBA - ~1.5MB)")
        self.export_scale_var = tk.StringVar(value="2K QHD (2048px - Recommended)")
        self.export_quality_var = tk.IntVar(value=92)
        self.export_png_comp_var = tk.IntVar(value=4)
        self.export_hw_accel_var = tk.BooleanVar(value=True)

        # Composite frame cache for ultra-smooth 60+ FPS zoom, pan, and resize
        self._composite_cache_key = None
        self._composite_cache_img = None
        self._resize_timer = None

        # Cached display frames & photo reference
        self.cached_display_image: Optional[Image.Image] = None
        self.cached_photo_image: Optional[ImageTk.PhotoImage] = None

        # Precompute checkerboard pattern for alpha cutout
        self._checkerboard_cache: Optional[np.ndarray] = None

        # Setup GUI Components
        self._build_ui()
        self._bind_events()

        # Load initial model in background
        self._init_model_async("tiny")

        # Auto-load initial path if provided
        if initial_path and os.path.exists(initial_path):
            self.dialog.after(200, lambda: self._load_input_source(initial_path))

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Top Header Bar
        hdr = tk.Frame(self.dialog, bg=SURFACE, padx=16, pady=10)
        hdr.pack(fill=tk.X, side=tk.TOP)

        hdr_left = tk.Frame(hdr, bg=SURFACE)
        hdr_left.pack(side=tk.LEFT, fill=tk.Y)

        title_lbl = tk.Label(hdr_left, text="🎭 AI MASKING STUDIO", font=("Segoe UI", 13, "bold"), fg=ACCENT, bg=SURFACE)
        title_lbl.pack(anchor="w")
        sub_lbl = tk.Label(hdr_left, text="Meta SAM 2.1 Interactive Video & Sequence Segmentation · RealityScan Photogrammetry",
                           font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        sub_lbl.pack(anchor="w")

        hdr_right = tk.Frame(hdr, bg=SURFACE)
        hdr_right.pack(side=tk.RIGHT, fill=tk.Y)

        # Model Selector
        tk.Label(hdr_right, text="Model:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 6))
        self.model_combo_var = tk.StringVar(value="tiny")
        self.model_combo = ttk.Combobox(hdr_right, textvariable=self.model_combo_var,
                                        values=["tiny", "small", "base_plus", "large"],
                                        state="readonly", width=12, font=("Segoe UI", 9))
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        # Device Badge
        dev_text = f"⚡ {self.engine.device.upper()}"
        if torch.cuda.is_available():
            dev_text = f"⚡ CUDA ({torch.cuda.get_device_name(0).split()[-2:]})"
            dev_text = f"⚡ CUDA: {torch.cuda.get_device_name(0)}"
        self.device_lbl = tk.Label(hdr_right, text=dev_text, font=("Segoe UI", 8, "bold"),
                                   fg=ACCENT, bg=FIELD, padx=8, pady=4, relief=tk.FLAT)
        self.device_lbl.pack(side=tk.LEFT, padx=(0, 10))

        # Status Label
        self.status_lbl = tk.Label(hdr_right, text="Ready", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.status_lbl.pack(side=tk.LEFT)

        # 2. Main Content Area (Horizontal Split)
        main_body = tk.Frame(self.dialog, bg=BG)
        main_body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left Column: Control Sidebar
        sidebar_frame = tk.Frame(main_body, bg=SURFACE, width=390)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        sidebar_frame.pack_propagate(False)

        self._build_sidebar(sidebar_frame)

        # Right Column: Viewport & Scrubber
        viewport_frame = tk.Frame(main_body, bg=BG)
        viewport_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_viewport(viewport_frame)

    def _build_sidebar(self, parent: tk.Frame):
        # Scrollable container for sidebar
        canvas = tk.Canvas(parent, bg=SURFACE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=SURFACE, padx=10, pady=10)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(window_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # CARD 1: Input Sequence / Video
        c1 = self._create_card(scrollable_frame, "📁 SOURCE MEDIA")
        r_btns = tk.Frame(c1, bg=SURFACE)
        r_btns.pack(fill=tk.X, pady=(2, 6))

        btn_folder = tk.Button(r_btns, text="Open Image Folder", command=self._browse_folder,
                               font=("Segoe UI", 8, "bold"), bg=FIELD, fg=TEXT, activebackground=BORDER,
                               relief=tk.FLAT, padx=8, pady=5, cursor="hand2")
        btn_folder.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_video = tk.Button(r_btns, text="Open Video File", command=self._browse_video,
                              font=("Segoe UI", 8, "bold"), bg=FIELD, fg=TEXT, activebackground=BORDER,
                              relief=tk.FLAT, padx=8, pady=5, cursor="hand2")
        btn_video.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.source_info_lbl = tk.Label(c1, text="No source loaded (open folder or video)",
                                        font=("Segoe UI", 8), fg=MUTED, bg=SURFACE, justify=tk.LEFT, wraplength=350)
        self.source_info_lbl.pack(anchor="w", pady=(0, 4))

        # CARD 2: Prompt Mode & Interactive Tools
        c2 = self._create_card(scrollable_frame, "🎯 PROMPT TOOLS")
        r_prompts = tk.Frame(c2, bg=SURFACE)
        r_prompts.pack(fill=tk.X, pady=(2, 6))

        for text, val, col in [("Foreground (+)", "Foreground", SUCCESS),
                               ("Erase / Background (-) [Alt+Click]", "Background", DANGER),
                               ("Box Select", "Box", ACCENT)]:
            rb = tk.Radiobutton(r_prompts, text=text, variable=self.prompt_mode, value=val,
                                font=("Segoe UI", 8, "bold"), fg=col, bg=SURFACE, selectcolor=FIELD,
                                activebackground=SURFACE, activeforeground=col)
            rb.pack(side=tk.LEFT, padx=(0, 6))

        lbl_prompt_hint = tk.Label(c2, text="💡 Tip: Left-Click adds (+) · Hold Alt + Click (or Right-Click) to Erase (−)",
                                   font=("Segoe UI", 8, "italic"), fg=MUTED, bg=SURFACE)
        lbl_prompt_hint.pack(anchor="w", pady=(0, 4))

        r_pbtns = tk.Frame(c2, bg=SURFACE)
        r_pbtns.pack(fill=tk.X, pady=(2, 4))

        btn_undo = tk.Button(r_pbtns, text="↩ Undo Click", command=self._undo_last_prompt,
                             font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, pady=4, cursor="hand2")
        btn_undo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_clr_frame = tk.Button(r_pbtns, text="Clear Frame", command=self._clear_current_frame_prompts,
                                  font=("Segoe UI", 8), bg=FIELD, fg=WARNING, relief=tk.FLAT, padx=6, pady=4, cursor="hand2")
        btn_clr_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_clr_all = tk.Button(r_pbtns, text="Reset All", command=self._clear_all_prompts,
                                font=("Segoe UI", 8), bg=FIELD, fg=DANGER, relief=tk.FLAT, padx=6, pady=4, cursor="hand2")
        btn_clr_all.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.keyframe_info_lbl = tk.Label(c2, text="Keyframes: 0 | Current Frame: Unprompted",
                                          font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.keyframe_info_lbl.pack(anchor="w", pady=(4, 0))

        # CARD 3: Temporal Tracking / Propagation
        c3 = self._create_card(scrollable_frame, "⚡ TIMELINE PROPAGATION")
        r_track = tk.Frame(c3, bg=SURFACE)
        r_track.pack(fill=tk.X, pady=(4, 6))

        btn_track_fwd = tk.Button(r_track, text="Track Forward ▶", command=lambda: self._start_tracking("forward"),
                                  font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=8, pady=6, cursor="hand2")
        btn_track_fwd.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_track_bwd = tk.Button(r_track, text="◀ Track Backward", command=lambda: self._start_tracking("backward"),
                                  font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=8, pady=6, cursor="hand2")
        btn_track_bwd.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_track_all = tk.Button(c3, text="⚡ Track Full Sequence (Both Directions)", command=lambda: self._start_tracking("both"),
                                  font=("Segoe UI", 9, "bold"), bg=ACCENT, fg=BG, activebackground=ACCENT_HOVER,
                                  relief=tk.FLAT, padx=10, pady=7, cursor="hand2")
        btn_track_all.pack(fill=tk.X, pady=(2, 6))

        self.track_progress = ttk.Progressbar(c3, orient="horizontal", mode="determinate")
        self.track_progress.pack(fill=tk.X, pady=(2, 4))

        self.track_status_lbl = tk.Label(c3, text="Tracking idle. Place prompt clicks to track.",
                                         font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.track_status_lbl.pack(anchor="w")

        # CARD 4: Mask Customization & Refinement
        c4 = self._create_card(scrollable_frame, "✨ MASK REFINEMENT")

        # Dilation / Erosion
        r_dil = tk.Frame(c4, bg=SURFACE)
        r_dil.pack(fill=tk.X, pady=2)
        tk.Label(r_dil, text="Dilation / Margin (px):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=20, anchor="w").pack(side=tk.LEFT)
        self.dilation_lbl = tk.Label(r_dil, text="0", font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, width=4)
        self.dilation_lbl.pack(side=tk.RIGHT)
        s_dil = ttk.Scale(r_dil, from_=-25, to=25, variable=self.dilation_var, command=self._on_refine_param_changed)
        s_dil.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Feather Radius
        r_fea = tk.Frame(c4, bg=SURFACE)
        r_fea.pack(fill=tk.X, pady=2)
        tk.Label(r_fea, text="Edge Feather (Softness):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=20, anchor="w").pack(side=tk.LEFT)
        self.feather_lbl = tk.Label(r_fea, text="2", font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, width=4)
        self.feather_lbl.pack(side=tk.RIGHT)
        s_fea = ttk.Scale(r_fea, from_=0, to=20, variable=self.feather_var, command=self._on_refine_param_changed)
        s_fea.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Edge Threshold
        r_thr = tk.Frame(c4, bg=SURFACE)
        r_thr.pack(fill=tk.X, pady=2)
        tk.Label(r_thr, text="Binarize Threshold:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=20, anchor="w").pack(side=tk.LEFT)
        self.thresh_lbl = tk.Label(r_thr, text="128", font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, width=4)
        self.thresh_lbl.pack(side=tk.RIGHT)
        s_thr = ttk.Scale(r_thr, from_=1, to=254, variable=self.threshold_var, command=self._on_refine_param_changed)
        s_thr.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Speckle Removal Area
        r_spk = tk.Frame(c4, bg=SURFACE)
        r_spk.pack(fill=tk.X, pady=2)
        tk.Label(r_spk, text="Clean Speckles (min px):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=20, anchor="w").pack(side=tk.LEFT)
        self.speckle_lbl = tk.Label(r_spk, text="30", font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, width=4)
        self.speckle_lbl.pack(side=tk.RIGHT)
        s_spk = ttk.Scale(r_spk, from_=0, to=150, variable=self.min_speckle_var, command=self._on_refine_param_changed)
        s_spk.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Fill Holes & Invert
        r_chk = tk.Frame(c4, bg=SURFACE)
        r_chk.pack(fill=tk.X, pady=(4, 2))
        chk_hole = tk.Checkbutton(r_chk, text="Fill Interior Holes", variable=self.fill_holes_var,
                                  font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, selectcolor=FIELD,
                                  command=self._on_refine_param_changed)
        chk_hole.pack(side=tk.LEFT, padx=(0, 10))

        chk_inv = tk.Checkbutton(r_chk, text="Invert Mask", variable=self.invert_var,
                                 font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, selectcolor=FIELD,
                                 command=self._on_refine_param_changed)
        chk_inv.pack(side=tk.LEFT)

        # CARD 5: Export Settings & Codecs (Hardware Accelerated)
        c5 = self._create_card(scrollable_frame, "💾 EXPORT MASKS & CUTOUTS")

        # Output Directory
        r_out = tk.Frame(c5, bg=SURFACE)
        r_out.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_out, text="Output:", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE, width=6, anchor="w").pack(side=tk.LEFT)
        self.out_entry = tk.Entry(r_out, textvariable=self.export_dir_var, font=("Segoe UI", 8),
                                  bg=FIELD, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT)
        self.out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        btn_browse_out = tk.Button(r_out, text="Browse", command=self._browse_output_dir,
                                   font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_browse_out.pack(side=tk.LEFT)

        # Target Checkboxes (Cutout is Optional)
        r_targets = tk.Frame(c5, bg=SURFACE)
        r_targets.pack(fill=tk.X, pady=(2, 4))
        self.chk_bin = tk.Checkbutton(r_targets, text="RealityScan Binary Masks (<stem>.mask.<fmt>)",
                                      variable=self.export_binary_var, font=("Segoe UI", 8, "bold"),
                                      fg=ACCENT, bg=SURFACE, selectcolor=FIELD,
                                      command=self._on_export_targets_changed)
        self.chk_bin.pack(anchor="w")

        self.chk_cut = tk.Checkbutton(r_targets, text="Generate Alpha Transparent Cutouts (RGBA) [Optional]",
                                      variable=self.export_cutout_var, font=("Segoe UI", 8, "bold"),
                                      fg=TEXT, bg=SURFACE, selectcolor=FIELD,
                                      command=self._on_export_targets_changed)
        self.chk_cut.pack(anchor="w", pady=(1, 4))

        # Format & Codec Selection
        r_codecs = tk.Frame(c5, bg=SURFACE)
        r_codecs.pack(fill=tk.X, pady=(2, 4))
        self.cutout_fmt_lbl = tk.Label(r_codecs, text="Cutout Format:", font=("Segoe UI", 8), fg=DISABLED, bg=SURFACE, width=12, anchor="w")
        self.cutout_fmt_lbl.pack(side=tk.LEFT)
        self.cb_cut = ttk.Combobox(r_codecs, textvariable=self.cutout_fmt_var,
                                  values=["WebP (RGBA - ~1.5MB)", "PNG (RGBA - Lossless)"],
                                  state="disabled", width=18, font=("Segoe UI", 8))
        self.cb_cut.pack(side=tk.LEFT, fill=tk.X, expand=True)

        r_mask_fmt = tk.Frame(c5, bg=SURFACE)
        r_mask_fmt.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_mask_fmt, text="Mask Format:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=12, anchor="w").pack(side=tk.LEFT)
        cb_mask = ttk.Combobox(r_mask_fmt, textvariable=self.mask_fmt_var,
                               values=["png", "jpg", "webp"],
                               state="readonly", width=18, font=("Segoe UI", 8))
        cb_mask.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Output Resolution / Scale
        r_res = tk.Frame(c5, bg=SURFACE)
        r_res.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_res, text="Export Size:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=12, anchor="w").pack(side=tk.LEFT)
        cb_scale = ttk.Combobox(r_res, textvariable=self.export_scale_var,
                                values=["2K QHD (2048px - Recommended)", "Original (100% Native)", "50% Half Res", "4K UHD (3840px)", "1K HD (1024px)"],
                                state="readonly", width=18, font=("Segoe UI", 8))
        cb_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Quality / Compression Slider
        r_qual = tk.Frame(c5, bg=SURFACE)
        r_qual.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_qual, text="Quality (WebP/JPG):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=15, anchor="w").pack(side=tk.LEFT)
        self.qual_lbl = tk.Label(r_qual, text="92", font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, width=3)
        self.qual_lbl.pack(side=tk.RIGHT)
        s_qual = ttk.Scale(r_qual, from_=50, to=100, variable=self.export_quality_var,
                           command=lambda v: self.qual_lbl.configure(text=str(int(float(v)))))
        s_qual.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Hardware Acceleration & Size Badge
        r_hw = tk.Frame(c5, bg=SURFACE)
        r_hw.pack(fill=tk.X, pady=(2, 4))
        chk_hw = tk.Checkbutton(r_hw, text="⚡ Hardware Accelerated (CUDA GPU + Multi-Core I/O)",
                                variable=self.export_hw_accel_var, font=("Segoe UI", 8, "bold"),
                                fg=ACCENT, bg=SURFACE, selectcolor=FIELD)
        chk_hw.pack(side=tk.LEFT)

        lbl_hint = tk.Label(c5, text="💡 Tip: WebP at 2K reduces cutout file size from ~70MB down to ~1.5MB with zero visual loss!",
                            font=("Segoe UI", 8, "italic"), fg=MUTED, bg=SURFACE, justify=tk.LEFT, wraplength=350)
        lbl_hint.pack(anchor="w", pady=(0, 6))

        # Big Export Button
        self.btn_export = tk.Button(c5, text="🚀 EXPORT REALITYSCAN MASKS (BINARY ONLY)", command=self._export_masks_action,
                                    font=("Segoe UI", 9, "bold"), bg=SUCCESS, fg="#ffffff",
                                    activebackground="#059669", relief=tk.FLAT, padx=12, pady=9, cursor="hand2")
        self.btn_export.pack(fill=tk.X, pady=(2, 4))

        self.export_progress = ttk.Progressbar(c5, orient="horizontal", mode="determinate")
        self.export_progress.pack(fill=tk.X, pady=(2, 4))

        self.export_status_lbl = tk.Label(c5, text="", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.export_status_lbl.pack(anchor="w")

    def _create_card(self, parent: tk.Widget, title: str) -> tk.Frame:
        card = tk.Frame(parent, bg=SURFACE, bd=1, relief=tk.SOLID, padx=10, pady=8)
        card.pack(fill=tk.X, pady=(0, 10))
        lbl = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg=ACCENT, bg=SURFACE)
        lbl.pack(anchor="w", pady=(0, 4))
        sep = tk.Frame(card, bg=BORDER, height=1)
        sep.pack(fill=tk.X, pady=(0, 6))
        return card

    def _build_viewport(self, parent: tk.Frame):
        # 1. Viewport Toolbar
        v_toolbar = tk.Frame(parent, bg=SURFACE, padx=10, pady=6)
        v_toolbar.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))

        # View Mode Selector
        tk.Label(v_toolbar, text="Mode:", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        for mode in ["Overlay", "Cutout", "Matte"]:
            rb = tk.Radiobutton(v_toolbar, text=mode, variable=self.view_mode, value=mode,
                                font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, selectcolor=FIELD,
                                activebackground=SURFACE, command=self._render_current_frame)
            rb.pack(side=tk.LEFT, padx=(0, 6))

        # Mask Tint Color Selector
        tk.Label(v_toolbar, text="Tint:", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(10, 4))
        color_combo = ttk.Combobox(v_toolbar, textvariable=self.mask_color_name,
                                   values=list(MASK_COLORS.keys()), state="readonly", width=9, font=("Segoe UI", 8))
        color_combo.pack(side=tk.LEFT, padx=(0, 10))
        color_combo.bind("<<ComboboxSelected>>", lambda e: self._render_current_frame())

        # Opacity Slider
        tk.Label(v_toolbar, text="Opacity:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        s_op = ttk.Scale(v_toolbar, from_=0.1, to=1.0, variable=self.overlay_opacity, command=lambda v: self._render_current_frame())
        s_op.pack(side=tk.LEFT, padx=(0, 14), fill=tk.X)

        # Zoom Controls
        btn_fit = tk.Button(v_toolbar, text="Fit View [F]", command=self._zoom_to_fit,
                            font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_fit.pack(side=tk.LEFT, padx=(0, 4))

        btn_100 = tk.Button(v_toolbar, text="100%", command=self._zoom_to_100,
                            font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_100.pack(side=tk.LEFT, padx=(0, 10))

        self.zoom_lbl = tk.Label(v_toolbar, text="Zoom: 100%", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.zoom_lbl.pack(side=tk.LEFT)

        # 2. Canvas Area
        self.canvas_frame = tk.Frame(parent, bg="#0d1117")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d1117", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 3. Bottom Timeline Scrubber
        timeline_bar = tk.Frame(parent, bg=SURFACE, padx=12, pady=8)
        timeline_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

        # Controls row
        r_ctrl = tk.Frame(timeline_bar, bg=SURFACE)
        r_ctrl.pack(fill=tk.X, pady=(0, 4))

        btn_first = tk.Button(r_ctrl, text="|◀", command=lambda: self._set_frame(0),
                              font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_first.pack(side=tk.LEFT, padx=(0, 4))

        btn_prev10 = tk.Button(r_ctrl, text="◀ -10", command=lambda: self._set_frame(self.current_frame_idx - 10),
                               font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_prev10.pack(side=tk.LEFT, padx=(0, 4))

        btn_prev = tk.Button(r_ctrl, text="◀ Prev", command=lambda: self._set_frame(self.current_frame_idx - 1),
                             font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_prev.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_play = tk.Button(r_ctrl, text="▶ Play", command=self._toggle_playback,
                                  font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=10, cursor="hand2")
        self.btn_play.pack(side=tk.LEFT, padx=(0, 4))

        btn_next = tk.Button(r_ctrl, text="Next ▶", command=lambda: self._set_frame(self.current_frame_idx + 1),
                             font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_next.pack(side=tk.LEFT, padx=(0, 4))

        btn_next10 = tk.Button(r_ctrl, text="+10 ▶", command=lambda: self._set_frame(self.current_frame_idx + 10),
                               font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_next10.pack(side=tk.LEFT, padx=(0, 4))

        btn_last = tk.Button(r_ctrl, text="▶|", command=lambda: self._set_frame(self.engine.num_frames - 1),
                             font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_last.pack(side=tk.LEFT, padx=(0, 14))

        # Keyframe jump buttons
        btn_prev_kf = tk.Button(r_ctrl, text="◀ Keyframe", command=self._jump_prev_keyframe,
                                font=("Segoe UI", 8), bg=FIELD, fg=WARNING, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_prev_kf.pack(side=tk.LEFT, padx=(0, 4))

        btn_next_kf = tk.Button(r_ctrl, text="Keyframe ▶", command=self._jump_next_keyframe,
                                font=("Segoe UI", 8), bg=FIELD, fg=WARNING, relief=tk.FLAT, padx=6, cursor="hand2")
        btn_next_kf.pack(side=tk.LEFT, padx=(0, 14))

        # Frame Readout Badge
        self.frame_readout_lbl = tk.Label(r_ctrl, text="Frame: 0 / 0", font=("Segoe UI", 9, "bold"),
                                          fg=TEXT, bg=FIELD, padx=10, pady=2, relief=tk.FLAT)
        self.frame_readout_lbl.pack(side=tk.RIGHT)

        # Scrubber Slider
        self.scrubber_var = tk.IntVar(value=0)
        self.scrubber = ttk.Scale(timeline_bar, from_=0, to=0, orient="horizontal",
                                  variable=self.scrubber_var, command=self._on_scrubber_drag)
        self.scrubber.pack(fill=tk.X, pady=(2, 0))

    # -------------------------------------------------------------------------
    # Event Bindings (Canvas, Mouse, Keyboard)
    # -------------------------------------------------------------------------

    def _bind_events(self):
        # Canvas mouse interactions
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_up)

        self.canvas.bind("<ButtonPress-2>", self._on_canvas_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_canvas_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_canvas_pan_end)

        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_down)
        self.canvas.bind("<B3-Motion>", self._on_canvas_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_pan_end)

        # Zoom via Mouse Wheel & Canvas Resizing
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Alt key bindings for Erase / Negative Prompt
        self.dialog.bind("<Alt_L>", self._on_alt_down)
        self.dialog.bind("<Alt_R>", self._on_alt_down)
        self.dialog.bind("<KeyRelease-Alt_L>", self._on_alt_up)
        self.dialog.bind("<KeyRelease-Alt_R>", self._on_alt_up)
        self.canvas.bind("<Alt-ButtonPress-1>", self._on_canvas_alt_down)

        # Keyboard shortcuts
        self.dialog.bind("<space>", lambda e: self._toggle_playback())
        self.dialog.bind("<Left>", lambda e: self._set_frame(self.current_frame_idx - 1))
        self.dialog.bind("<Right>", lambda e: self._set_frame(self.current_frame_idx + 1))
        self.dialog.bind("<Control-z>", lambda e: self._undo_last_prompt())
        self.dialog.bind("<Delete>", lambda e: self._clear_current_frame_prompts())
        self.dialog.bind("<f>", lambda e: self._zoom_to_fit())
        self.dialog.bind("<F>", lambda e: self._zoom_to_fit())

    # -------------------------------------------------------------------------
    # Model Loading (Asynchronous)
    # -------------------------------------------------------------------------

    def _init_model_async(self, model_key: str):
        self.status_lbl.configure(text=f"Loading SAM 2 ({model_key})...", fg=ACCENT)

        def _worker():
            try:
                self.engine.load_model(model_key, progress_callback=lambda msg: self.dialog.after(0, lambda: self.status_lbl.configure(text=msg)))
                self.dialog.after(0, lambda: self.status_lbl.configure(text=f"SAM 2 {model_key.upper()} Ready", fg=SUCCESS))
            except Exception as e:
                self.dialog.after(0, lambda: self.status_lbl.configure(text=f"Model Error: {e}", fg=DANGER))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_changed(self, event=None):
        model_key = self.model_combo_var.get()
        self._init_model_async(model_key)

    # -------------------------------------------------------------------------
    # Source Media Ingestion
    # -------------------------------------------------------------------------

    def _browse_folder(self):
        f = filedialog.askdirectory(title="Select Image Sequence Folder", parent=self.dialog)
        if f:
            self._load_input_source(f)

    def _browse_video(self):
        f = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.webm"), ("All files", "*.*")],
            parent=self.dialog
        )
        if f:
            self._load_input_source(f)

    def _load_input_source(self, path: str):
        self.status_lbl.configure(text="Loading source frames...", fg=ACCENT)

        def _worker():
            try:
                def _prog(cur, tot, msg):
                    self.dialog.after(0, lambda: self.status_lbl.configure(text=msg))

                n_frames = self.engine.load_source(path, progress_callback=_prog)
                self.dialog.after(0, lambda: self._on_source_loaded(path, n_frames))
            except Exception as e:
                self.dialog.after(0, lambda: messagebox.showerror("Source Loading Failed", str(e), parent=self.dialog))
                self.dialog.after(0, lambda: self.status_lbl.configure(text="Loading failed", fg=DANGER))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_source_loaded(self, path: str, n_frames: int):
        self.current_frame_idx = 0
        self.scrubber.configure(from_=0, to=max(0, n_frames - 1))
        self.scrubber_var.set(0)

        # Set default export directory
        if os.path.isdir(path):
            self.export_dir_var.set(os.path.join(path, "masks"))
        else:
            self.export_dir_var.set(os.path.join(os.path.dirname(path), "masks"))

        self.source_info_lbl.configure(
            text=f"Source: {os.path.basename(path)}\n"
                 f"Frames: {n_frames} | Res: {self.engine.orig_width}x{self.engine.orig_height}"
        )
        self.status_lbl.configure(text=f"Loaded {n_frames} frames", fg=SUCCESS)

        self._zoom_to_fit()
        self._set_frame(0)

    # -------------------------------------------------------------------------
    # Frame Navigation & Scrubber
    # -------------------------------------------------------------------------

    def _set_frame(self, frame_idx: int):
        if self.engine.num_frames == 0:
            return

        clamped = max(0, min(self.engine.num_frames - 1, frame_idx))
        self.current_frame_idx = clamped
        self.scrubber_var.set(clamped)

        # Update Frame Readout
        fname = self.engine.frame_names[clamped] if clamped < len(self.engine.frame_names) else ""
        kf_star = " ⭐ Keyframe" if self.engine.has_prompts(clamped) else ""
        self.frame_readout_lbl.configure(text=f"Frame: {clamped + 1} / {self.engine.num_frames}  ({fname}){kf_star}")

        # Update Keyframe Info
        kfs = self.engine.get_keyframes()
        prompt_info = "Prompts: "
        if clamped in self.engine.prompts:
            p_data = self.engine.prompts[clamped]
            n_pts = len(p_data["points"])
            has_box = " + Box" if p_data["box"] else ""
            prompt_info += f"{n_pts} points{has_box}"
        else:
            prompt_info += "None"
        self.keyframe_info_lbl.configure(text=f"Keyframes: {len(kfs)} | Current: {prompt_info}")

        self._render_current_frame()

    def _on_scrubber_drag(self, val):
        self._set_frame(int(float(val)))

    def _toggle_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.configure(text="▶ Play", fg=ACCENT)
        else:
            if self.engine.num_frames <= 1:
                return
            self.is_playing = True
            self.btn_play.configure(text="⏸ Pause", fg=WARNING)
            self._playback_loop()

    def _playback_loop(self):
        if not self.is_playing:
            return

        next_idx = self.current_frame_idx + 1
        if next_idx >= self.engine.num_frames:
            next_idx = 0

        self._set_frame(next_idx)
        delay_ms = int(1000 / self.play_fps)
        self.dialog.after(delay_ms, self._playback_loop)

    def _jump_prev_keyframe(self):
        kfs = [k for k in self.engine.get_keyframes() if k < self.current_frame_idx]
        if kfs:
            self._set_frame(kfs[-1])

    def _jump_next_keyframe(self):
        kfs = [k for k in self.engine.get_keyframes() if k > self.current_frame_idx]
        if kfs:
            self._set_frame(kfs[0])

    # -------------------------------------------------------------------------
    # Canvas Pan & Zoom
    # -------------------------------------------------------------------------

    def _zoom_to_fit(self):
        if self.engine.orig_width == 0 or self.engine.orig_height == 0:
            return

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())

        scale_x = (canvas_w - 40) / self.engine.orig_width
        scale_y = (canvas_h - 40) / self.engine.orig_height
        self.zoom = min(scale_x, scale_y)

        # Center in canvas
        self.pan_x = (canvas_w - self.engine.orig_width * self.zoom) / 2.0
        self.pan_y = (canvas_h - self.engine.orig_height * self.zoom) / 2.0

        self.zoom_lbl.configure(text=f"Zoom: {int(self.zoom * 100)}%")
        self._render_current_frame()

    def _zoom_to_100(self):
        self.zoom = 1.0
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        self.pan_x = (canvas_w - self.engine.orig_width) / 2.0
        self.pan_y = (canvas_h - self.engine.orig_height) / 2.0
        self.zoom_lbl.configure(text="Zoom: 100%")
        self._render_current_frame()

    def _on_canvas_mousewheel(self, event):
        if self.engine.orig_width == 0:
            return

        mouse_x = event.x
        mouse_y = event.y

        # Zoom anchor relative to image
        img_x = (mouse_x - self.pan_x) / self.zoom
        img_y = (mouse_y - self.pan_y) / self.zoom

        factor = 1.15 if event.delta > 0 else 0.87
        new_zoom = max(0.05, min(10.0, self.zoom * factor))

        self.pan_x = mouse_x - img_x * new_zoom
        self.pan_y = mouse_y - img_y * new_zoom
        self.zoom = new_zoom

        self.zoom_lbl.configure(text=f"Zoom: {int(self.zoom * 100)}%")
        self._render_current_frame()

    def _on_canvas_pan_start(self, event):
        self.is_panning = True
        self.drag_start_x = event.x - self.pan_x
        self.drag_start_y = event.y - self.pan_y

    def _on_canvas_pan_drag(self, event):
        if self.is_panning:
            self.pan_x = event.x - self.drag_start_x
            self.pan_y = event.y - self.drag_start_y
            self._render_current_frame()

    def _on_canvas_pan_end(self, event):
        self.is_panning = False

    def _on_canvas_resize(self, event):
        if self._resize_timer is not None:
            self.dialog.after_cancel(self._resize_timer)
        self._resize_timer = self.dialog.after(25, self._debounced_resize)

    def _debounced_resize(self):
        self._resize_timer = None
        if self.engine.num_frames > 0:
            self._render_current_frame()

    # -------------------------------------------------------------------------
    # Interactive Prompts Handling (Left Click / Alt+Click / Right Click / Box)
    # -------------------------------------------------------------------------

    def _on_alt_down(self, event=None):
        self.is_alt_pressed = True
        self.canvas.configure(cursor="circle")
        self.status_lbl.configure(text="[ALT HELD] Click anywhere on the mask to ERASE / Subtract (−)", fg=DANGER)

    def _on_alt_up(self, event=None):
        self.is_alt_pressed = False
        self.canvas.configure(cursor="crosshair")
        self.status_lbl.configure(text="Ready. Left-Click adds (+) · Hold Alt + Click (or Right-Click) to Erase (−)", fg=MUTED)

    def _on_canvas_alt_down(self, event):
        """Dedicated handler for Alt+Left Click directly erasing mask at clicked point."""
        if self.engine.num_frames == 0:
            return
        img_x = (event.x - self.pan_x) / self.zoom
        img_y = (event.y - self.pan_y) / self.zoom
        if 0 <= img_x < self.engine.orig_width and 0 <= img_y < self.engine.orig_height:
            self._apply_point_prompt(img_x, img_y, label=0)

    def _on_canvas_left_down(self, event):
        if self.engine.num_frames == 0:
            return

        # Check Alt key or Shift key (Alt / Shift + Left Click = Erase / Negative Prompt)
        is_alt = (
            self.is_alt_pressed
            or bool(event.state & 0x20000)
            or bool(event.state & 0x0008)
            or bool(event.state & 0x0080)
        )
        is_shift = (event.state & 0x0001) != 0
        mode = self.prompt_mode.get()

        if mode == "Box":
            self.box_start = (event.x, event.y)
            return

        # Screen to Image Space
        img_x = (event.x - self.pan_x) / self.zoom
        img_y = (event.y - self.pan_y) / self.zoom

        if 0 <= img_x < self.engine.orig_width and 0 <= img_y < self.engine.orig_height:
            label = 0 if (is_alt or is_shift or mode == "Background") else 1
            self._apply_point_prompt(img_x, img_y, label)

    def _on_canvas_left_drag(self, event):
        if self.prompt_mode.get() == "Box" and self.box_start:
            x0, y0 = self.box_start
            x1, y1 = event.x, event.y
            if self.current_box_rect_id:
                self.canvas.delete(self.current_box_rect_id)
            self.current_box_rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=ACCENT, width=2, dash=(4, 4)
            )

    def _on_canvas_left_up(self, event):
        if self.prompt_mode.get() == "Box" and self.box_start:
            if self.current_box_rect_id:
                self.canvas.delete(self.current_box_rect_id)
                self.current_box_rect_id = None

            sx0, sy0 = self.box_start
            sx1, sy1 = event.x, event.y
            self.box_start = None

            # Convert to image coords
            ix0 = (min(sx0, sx1) - self.pan_x) / self.zoom
            iy0 = (min(sy0, sy1) - self.pan_y) / self.zoom
            ix1 = (max(sx0, sx1) - self.pan_x) / self.zoom
            iy1 = (max(sy0, sy1) - self.pan_y) / self.zoom

            # Ensure minimum bounding box size
            if (ix1 - ix0) > 10 and (iy1 - iy0) > 10:
                self._apply_box_prompt([ix0, iy0, ix1, iy1])

    def _on_canvas_right_down(self, event):
        # Right click adds a negative (erase) prompt directly
        if self.engine.num_frames == 0:
            return

        img_x = (event.x - self.pan_x) / self.zoom
        img_y = (event.y - self.pan_y) / self.zoom

        if 0 <= img_x < self.engine.orig_width and 0 <= img_y < self.engine.orig_height:
            self._apply_point_prompt(img_x, img_y, label=0)
        else:
            # If outside image, treat as pan start
            self._on_canvas_pan_start(event)

    def _apply_point_prompt(self, x: float, y: float, label: int):
        lbl_text = "Erase (−)" if label == 0 else "Foreground (+)"
        self.status_lbl.configure(text=f"Evaluating {lbl_text} prompt on frame {self.current_frame_idx + 1}...",
                                  fg=DANGER if label == 0 else ACCENT)
        try:
            self.engine.add_point(self.current_frame_idx, x, y, label=label)
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)
            self.status_lbl.configure(text=f"{lbl_text} prompt evaluated successfully", fg=SUCCESS)
        except Exception as e:
            self.status_lbl.configure(text=f"Prompt Error: {e}", fg=DANGER)

    def _apply_box_prompt(self, box: List[float]):
        self.status_lbl.configure(text=f"Evaluating box on frame {self.current_frame_idx + 1}...", fg=ACCENT)
        try:
            self.engine.add_box(self.current_frame_idx, box)
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)
            self.status_lbl.configure(text="Box evaluated successfully", fg=SUCCESS)
        except Exception as e:
            self.status_lbl.configure(text=f"Prompt Error: {e}", fg=DANGER)

    def _undo_last_prompt(self):
        self.engine.remove_last_prompt(self.current_frame_idx)
        self._composite_cache_key = None
        self._set_frame(self.current_frame_idx)

    def _clear_current_frame_prompts(self):
        self.engine.clear_frame_prompts(self.current_frame_idx)
        self._composite_cache_key = None
        self._set_frame(self.current_frame_idx)

    def _clear_all_prompts(self):
        if messagebox.askyesno("Reset All Prompts", "Are you sure you want to clear all prompts and tracked masks across all frames?", parent=self.dialog):
            self.engine.prompts.clear()
            self.engine.tracked_masks.clear()
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)

    # -------------------------------------------------------------------------
    # Temporal Propagation
    # -------------------------------------------------------------------------

    def _start_tracking(self, direction: str):
        if not self.engine.prompts:
            messagebox.showwarning("No Prompts", "Please click on the subject to add at least one prompt before tracking.", parent=self.dialog)
            return

        if self.is_tracking:
            return

        self.is_tracking = True
        self.cancel_requested = False
        self.track_progress["value"] = 0
        self.btn_export.configure(state=tk.DISABLED)

        # For single directional tracking, use current frame; for "both", pass None so engine starts from earliest keyframe
        start_frame = self.current_frame_idx if direction != "both" else None

        def _prog(cur, tot, msg):
            pct = int(cur * 100 / tot)
            self.dialog.after(0, lambda: self._update_tracking_ui(pct, msg))

        def _worker():
            try:
                self.engine.propagate(
                    direction=direction,
                    start_frame=start_frame,
                    progress_callback=_prog,
                    cancel_token=lambda: self.cancel_requested
                )
                self.dialog.after(0, self._on_tracking_completed)
            except Exception as e:
                self.dialog.after(0, lambda: messagebox.showerror("Tracking Failed", str(e), parent=self.dialog))
                self.dialog.after(0, self._on_tracking_completed)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_tracking_ui(self, pct: int, msg: str):
        self.track_progress["value"] = pct
        self.track_status_lbl.configure(text=msg, fg=ACCENT)

    def _on_tracking_completed(self):
        self.is_tracking = False
        self.btn_export.configure(state=tk.NORMAL)
        self.track_status_lbl.configure(text=f"Tracking complete ({len(self.engine.tracked_masks)} frames tracked)!", fg=SUCCESS)
        self._composite_cache_key = None
        self._render_current_frame()

    # -------------------------------------------------------------------------
    # Mask Refinement Callbacks
    # -------------------------------------------------------------------------

    def _on_refine_param_changed(self, *args):
        # Update numeric labels
        self.dilation_lbl.configure(text=str(self.dilation_var.get()))
        self.feather_lbl.configure(text=str(self.feather_var.get()))
        self.thresh_lbl.configure(text=str(self.threshold_var.get()))
        self.speckle_lbl.configure(text=str(self.min_speckle_var.get()))
        self._render_current_frame()

    # -------------------------------------------------------------------------
    # Viewport Rendering Engine
    # -------------------------------------------------------------------------

    def _get_checkerboard(self, width: int, height: int, tile_size: int = 16) -> np.ndarray:
        """Generates or retrieves cached checkerboard background for transparency preview."""
        if self._checkerboard_cache is not None and self._checkerboard_cache.shape[:2] == (height, width):
            return self._checkerboard_cache

        # Create tile pattern
        y, x = np.ogrid[:height, :width]
        board = ((x // tile_size) + (y // tile_size)) % 2
        bg = np.where(board[..., None] == 0, 205, 140).astype(np.uint8)
        bg = np.repeat(bg, 3, axis=-1)
        self._checkerboard_cache = bg
        return bg

    def _render_current_frame(self):
        if self.engine.num_frames == 0:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        orig_w = self.engine.orig_width
        orig_h = self.engine.orig_height
        if orig_w <= 0 or orig_h <= 0:
            return

        # 1. Composite Frame Cache: Only recompute full mask & blending when frame/params change
        p_data = self.engine.prompts.get(self.current_frame_idx, {})
        pts_tuple = tuple(p_data.get("points", []))
        lbls_tuple = tuple(p_data.get("labels", []))
        box_tuple = tuple(p_data["box"]) if p_data.get("box") is not None else None
        has_tracked = self.current_frame_idx in self.engine.tracked_masks

        cache_key = (
            self.current_frame_idx,
            self.dilation_var.get(),
            self.feather_var.get(),
            self.threshold_var.get(),
            self.fill_holes_var.get(),
            self.min_speckle_var.get(),
            self.invert_var.get(),
            self.view_mode.get(),
            self.mask_color_name.get(),
            round(self.overlay_opacity.get(), 2),
            pts_tuple,
            lbls_tuple,
            box_tuple,
            has_tracked
        )

        if self._composite_cache_key == cache_key and self._composite_cache_img is not None:
            rendered_rgb = self._composite_cache_img
        else:
            try:
                rgb = self.engine.get_frame_rgb(self.current_frame_idx)
            except Exception:
                return

            mask = self.engine.get_processed_mask(
                self.current_frame_idx,
                dilation=self.dilation_var.get(),
                feather=self.feather_var.get(),
                threshold=self.threshold_var.get(),
                fill_holes=self.fill_holes_var.get(),
                min_speckle=self.min_speckle_var.get(),
                invert=self.invert_var.get()
            )

            mode = self.view_mode.get()

            if mask is None or np.sum(mask) == 0:
                rendered_rgb = rgb
            elif mode == "Matte":
                rendered_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif mode == "Cutout":
                bg = self._get_checkerboard(orig_w, orig_h)
                alpha_f = (mask.astype(np.float32) / 255.0)[..., None]
                rendered_rgb = (rgb.astype(np.float32) * alpha_f + bg.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
            else:
                tint_color = MASK_COLORS.get(self.mask_color_name.get(), (239, 68, 68))
                opacity = self.overlay_opacity.get()
                colored_mask = np.zeros_like(rgb)
                colored_mask[mask > 0] = tint_color

                rendered_rgb = rgb.copy()
                active_pixels = (mask > 0)
                rendered_rgb[active_pixels] = cv2.addWeighted(
                    rgb[active_pixels], 1.0 - opacity,
                    colored_mask[active_pixels], opacity, 0.0
                )
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rendered_rgb, contours, -1, (255, 255, 255), 2)

            self._composite_cache_key = cache_key
            self._composite_cache_img = rendered_rgb

        # 2. Viewport Slicing: Only resize and blit the visible canvas region
        ix0 = max(0, int(math.floor(-self.pan_x / self.zoom)))
        iy0 = max(0, int(math.floor(-self.pan_y / self.zoom)))
        ix1 = min(orig_w, int(math.ceil((canvas_w - self.pan_x) / self.zoom)))
        iy1 = min(orig_h, int(math.ceil((canvas_h - self.pan_y) / self.zoom)))

        if ix1 <= ix0 or iy1 <= iy0:
            self.canvas.delete("all")
            return

        crop_rgb = rendered_rgb[iy0:iy1, ix0:ix1]
        disp_w = max(1, int(round((ix1 - ix0) * self.zoom)))
        disp_h = max(1, int(round((iy1 - iy0) * self.zoom)))
        draw_x = self.pan_x + ix0 * self.zoom
        draw_y = self.pan_y + iy0 * self.zoom

        disp_img = cv2.resize(crop_rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        self.cached_photo_image = ImageTk.PhotoImage(Image.fromarray(disp_img))

        self.canvas.delete("all")
        self.canvas.create_image(draw_x, draw_y, image=self.cached_photo_image, anchor="nw")
        self._render_prompt_markers()

    def _render_prompt_markers(self):
        if self.current_frame_idx not in self.engine.prompts:
            return

        p_data = self.engine.prompts[self.current_frame_idx]

        # Draw Points
        for (ix, iy), lbl in zip(p_data["points"], p_data["labels"]):
            sx = ix * self.zoom + self.pan_x
            sy = iy * self.zoom + self.pan_y
            r = 6
            color = SUCCESS if lbl == 1 else DANGER
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=color, outline="#ffffff", width=2)
            self.canvas.create_text(sx, sy, text="+" if lbl == 1 else "−", fill="#ffffff", font=("Segoe UI", 8, "bold"))

        # Draw Box
        if p_data["box"] is not None:
            bx0, by0, bx1, by1 = p_data["box"]
            sx0 = bx0 * self.zoom + self.pan_x
            sy0 = by0 * self.zoom + self.pan_y
            sx1 = bx1 * self.zoom + self.pan_x
            sy1 = by1 * self.zoom + self.pan_y
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline=ACCENT, width=2, dash=(6, 4))

    # -------------------------------------------------------------------------
    # Mask Export Actions
    # -------------------------------------------------------------------------

    def _on_export_targets_changed(self):
        cutout_on = self.export_cutout_var.get()
        binary_on = self.export_binary_var.get()

        if cutout_on:
            self.cb_cut.configure(state="readonly")
            self.cutout_fmt_lbl.configure(fg=TEXT)
            self.chk_cut.configure(fg=ACCENT)
        else:
            self.cb_cut.configure(state="disabled")
            self.cutout_fmt_lbl.configure(fg=DISABLED)
            self.chk_cut.configure(fg=TEXT)

        if binary_on and cutout_on:
            btn_txt = "🚀 EXPORT MASKS & RGBA CUTOUTS (HARDWARE ACCELERATED)"
        elif binary_on:
            btn_txt = "🚀 EXPORT REALITYSCAN MASKS (BINARY ONLY)"
        elif cutout_on:
            btn_txt = "🚀 EXPORT RGBA CUTOUTS ONLY (HARDWARE ACCELERATED)"
        else:
            btn_txt = "🚀 SELECT AN EXPORT TARGET OPTION"

        self.btn_export.configure(text=btn_txt)

    def _browse_output_dir(self):
        f = filedialog.askdirectory(title="Select Mask Export Directory", parent=self.dialog)
        if f:
            self.export_dir_var.set(f)

    def _export_masks_action(self):
        out_dir = self.export_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Missing Directory", "Please select an export destination folder.", parent=self.dialog)
            return

        if not self.export_binary_var.get() and not self.export_cutout_var.get():
            messagebox.showwarning("No Format Selected", "Please enable either Binary Masks or Alpha Cutouts to export.", parent=self.dialog)
            return

        self.btn_export.configure(state=tk.DISABLED)
        self.export_progress["value"] = 0
        self.export_status_lbl.configure(text="Exporting masks...", fg=ACCENT)

        params = {
            "dilation": self.dilation_var.get(),
            "feather": self.feather_var.get(),
            "threshold": self.threshold_var.get(),
            "fill_holes": self.fill_holes_var.get(),
            "min_speckle": self.min_speckle_var.get(),
            "invert": self.invert_var.get()
        }

        def _prog(cur, tot, msg):
            pct = int(cur * 100 / tot)
            self.dialog.after(0, lambda: self._update_export_ui(pct, msg))

        def _worker():
            try:
                cutout_fmt = "webp" if "webp" in self.cutout_fmt_var.get().lower() else "png"
                raw_scale = self.export_scale_var.get().split()[0].lower()
                scale_mode = "original" if "orig" in raw_scale else raw_scale

                counts = self.engine.export_masks(
                    output_dir=out_dir,
                    export_binary=self.export_binary_var.get(),
                    export_cutout=self.export_cutout_var.get(),
                    rs_naming=self.export_rs_naming_var.get(),
                    mask_format=self.mask_fmt_var.get(),
                    cutout_format=cutout_fmt,
                    scale_mode=scale_mode,
                    quality=self.export_quality_var.get(),
                    png_compression=self.export_png_comp_var.get(),
                    use_hardware_accel=self.export_hw_accel_var.get(),
                    postprocess_params=params,
                    progress_callback=_prog
                )
                self.dialog.after(0, lambda: self._on_export_success(out_dir, counts))
            except Exception as e:
                self.dialog.after(0, lambda: messagebox.showerror("Export Failed", str(e), parent=self.dialog))
                self.dialog.after(0, lambda: self.btn_export.configure(state=tk.NORMAL))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_export_ui(self, pct: int, msg: str):
        self.export_progress["value"] = pct
        self.export_status_lbl.configure(text=msg)

    def _on_export_success(self, out_dir: str, counts: Dict[str, Any]):
        self.btn_export.configure(state=tk.NORMAL)
        total_mb = counts.get('total_mb', 0.0)
        elapsed = counts.get('elapsed_sec', 0.0)
        fps = counts.get('fps', 0.0)
        avg_mb = counts.get('avg_cutout_mb', 0.0)
        saved_mb = counts.get('saved_mb', 0.0)
        res_str = counts.get('scale_res', '')

        status_text = (
            f"Export Complete: {counts.get('binary', 0)} masks, {counts.get('cutout', 0)} cutouts "
            f"in {elapsed:.1f}s ({fps:.1f} fps, {total_mb:.1f} MB)"
        )
        self.export_status_lbl.configure(text=status_text, fg=SUCCESS)

        msg = (
            f"🎉 Export Finished Successfully!\n\n"
            f"• Output Resolution: {res_str}\n"
            f"• RealityScan Masks: {counts.get('binary', 0)}\n"
            f"• Transparent Cutouts: {counts.get('cutout', 0)} (Avg: {avg_mb:.2f} MB each)\n"
            f"• Total Data Written: {total_mb:.1f} MB in {elapsed:.1f}s ({fps:.1f} frames/sec)\n"
            f"• Storage Saved: ~{saved_mb:.1f} MB vs 70MB uncompressed PNG (97%+ reduction!)\n\n"
            f"Destination Folder:\n{out_dir}"
        )
        if messagebox.askyesno("Export Complete", msg + "\n\nWould you like to open the export folder in Explorer?", parent=self.dialog):
            if sys.platform == "win32":
                os.startfile(out_dir)

    def _on_close(self):
        """Cleanly releases background threads, model buffers, and window resources."""
        self.is_playing = False
        self.cancel_requested = True
        try:
            if hasattr(self, "engine") and self.engine is not None:
                self.engine.clear_all()
        except Exception:
            pass
        self.dialog.destroy()


def open_sam2_masker_dialog(parent: tk.Widget, initial_path: Optional[str] = None, log_callback: Optional[Callable[[str], None]] = None) -> SAM2MaskerDialog:
    """Helper entry point to launch the SAM 2 Masking Studio dialog."""
    return SAM2MaskerDialog(parent, initial_path=initial_path, log_callback=log_callback)


if __name__ == "__main__":
    # Test standalone launcher
    root = tk.Tk()
    root.withdraw()
    dlg = open_sam2_masker_dialog(root)
    root.mainloop()
