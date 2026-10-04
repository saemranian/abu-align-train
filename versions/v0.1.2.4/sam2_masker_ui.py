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


class ToolTip:
    """
    Lightweight, modern floating tooltip for Tkinter controls.
    Spawns after a short hover delay; auto-destroys on mouse exit or click.
    """
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip_window = None
        self._after_id = None

        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if not self.widget.winfo_exists() or self._tip_window:
            return
        try:
            x = self.widget.winfo_rootx() + 8
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return

        self._tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.wm_geometry(f"+{x}+{y}")

        frame = tk.Frame(tw, bg="#1a2024", padx=8, pady=5, highlightthickness=1, highlightbackground=ACCENT)
        frame.pack()
        lbl = tk.Label(frame, text=self.text, font=("Segoe UI", 8), bg="#1a2024", fg=TEXT,
                       justify=tk.LEFT, wraplength=340)
        lbl.pack()

    def _hide(self):
        if self._tip_window:
            try:
                self._tip_window.destroy()
            except Exception:
                pass
            self._tip_window = None


def add_tooltip(widget: tk.Widget, text: str):
    """Helper to attach an instructive tooltip to any Tkinter widget."""
    return ToolTip(widget, text)


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

        # Tool Mode & Precision Editing State
        # Modes: "AI Point", "AI Erase", "AI Box", "Box Cut", "Brush Paint", "Brush Erase"
        self.tool_mode = tk.StringVar(value="AI Point")
        self.prompt_mode = self.tool_mode  # Backward compatibility alias
        self.brush_size_var = tk.IntVar(value=20)
        self.brush_active_mode: str = "erase"
        self.is_brushing: bool = False
        self.current_stroke: Optional[List[Tuple[float, float]]] = None
        self.box_is_cut: bool = False
        self.brush_cursor_id = None
        self.is_right_interacting: bool = False

        # Sidebar scrollable container references
        self.sidebar_frame: Optional[tk.Frame] = None
        self.sidebar_canvas: Optional[tk.Canvas] = None
        self.sidebar_scrollbar: Optional[ttk.Scrollbar] = None
        self.scrollable_frame: Optional[tk.Frame] = None

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
        self.export_scale_var = tk.StringVar(value="2K QHD (2048px - Auto Aspect)")
        self.export_quality_var = tk.IntVar(value=92)
        self.export_png_comp_var = tk.IntVar(value=1)
        self.export_hw_accel_var = tk.BooleanVar(value=True)

        # Mask & composite frame cache for ultra-smooth 60+ FPS zoom, pan, and scrubbing
        self._composite_cache_key = None
        self._composite_cache_img = None
        self._resize_timer = None
        self._refine_timer = None
        self._cached_mask_key = None
        self._cached_mask: Optional[np.ndarray] = None

        # Cached display frames & photo reference
        self.cached_display_image: Optional[Image.Image] = None
        self.cached_photo_image: Optional[ImageTk.PhotoImage] = None

        # Precompute checkerboard pattern for alpha cutout
        self._checkerboard_cache: Optional[np.ndarray] = None

        # Export state tracking (Pause / Cancel support)
        self.is_export_running: bool = False
        self.is_export_paused: bool = False
        self.export_cancel_requested: bool = False

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

        # Left Column: Control Sidebar (Generous 500px to eliminate horizontal and vertical cropping)
        sidebar_frame = tk.Frame(main_body, bg=SURFACE, width=500)
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        sidebar_frame.pack_propagate(False)

        self._build_sidebar(sidebar_frame)

        # Right Column: Viewport & Scrubber
        viewport_frame = tk.Frame(main_body, bg=BG)
        viewport_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_viewport(viewport_frame)

        # 3. Setup Instructive Hover Tooltips for All Controls
        self._setup_tooltips()

    def _build_sidebar(self, parent: tk.Frame):
        # 1. Non-Clipped Scrollable Sidebar Container with Responsive Scrollbar
        self.sidebar_frame = parent
        self.sidebar_scrollbar = ttk.Scrollbar(parent, orient="vertical")
        self.sidebar_canvas = tk.Canvas(parent, bg=SURFACE, highlightthickness=0, yscrollcommand=self.sidebar_scrollbar.set)
        self.sidebar_scrollbar.configure(command=self.sidebar_canvas.yview)

        # Dock scrollbar FIRST on the right so canvas expansion never squashes it
        self.sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollable_frame = tk.Frame(self.sidebar_canvas, bg=SURFACE, padx=8, pady=8)
        self.sidebar_window_id = self.sidebar_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def _on_sidebar_content_configure(event):
            if self.sidebar_canvas.winfo_exists():
                self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

        self.scrollable_frame.bind("<Configure>", _on_sidebar_content_configure)

        def _on_sidebar_canvas_configure(event):
            if self.sidebar_canvas.winfo_exists():
                self.sidebar_canvas.itemconfig(self.sidebar_window_id, width=event.width)
        self.sidebar_canvas.bind("<Configure>", _on_sidebar_canvas_configure)

        # CARD 1: Input Sequence / Video
        c1 = self._create_card(self.scrollable_frame, "📁 SOURCE MEDIA")
        r_btns = tk.Frame(c1, bg=SURFACE)
        r_btns.pack(fill=tk.X, pady=(2, 6))

        self.btn_folder = tk.Button(r_btns, text="Open Image Folder", command=self._browse_folder,
                                    font=("Segoe UI", 8, "bold"), bg=FIELD, fg=TEXT, activebackground=BORDER,
                                    relief=tk.FLAT, padx=8, pady=5, cursor="hand2")
        self.btn_folder.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_video = tk.Button(r_btns, text="Open Video File", command=self._browse_video,
                                   font=("Segoe UI", 8, "bold"), bg=FIELD, fg=TEXT, activebackground=BORDER,
                                   relief=tk.FLAT, padx=8, pady=5, cursor="hand2")
        self.btn_video.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.source_info_lbl = tk.Label(c1, text="No source loaded (open folder or video)",
                                        font=("Segoe UI", 8), fg=MUTED, bg=SURFACE, justify=tk.LEFT, wraplength=460)
        self.source_info_lbl.pack(anchor="w", pady=(0, 4))

        # CARD 2: Prompt Mode & Precision Interactive Tools
        c2 = self._create_card(self.scrollable_frame, "🎯 PROMPT & SELECTION TOOLS")

        # Responsive Tool Grid Container: 4 columns (Label + 3 tool options)
        tool_grid = tk.Frame(c2, bg=SURFACE)
        tool_grid.pack(fill=tk.X, pady=(2, 4))
        tool_grid.columnconfigure(0, weight=0, minsize=80)
        tool_grid.columnconfigure(1, weight=1)
        tool_grid.columnconfigure(2, weight=1)
        tool_grid.columnconfigure(3, weight=1)

        # Row 0: AI Prompt Tools
        tk.Label(tool_grid, text="AI Modes:", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=SURFACE, anchor="w").grid(
            row=0, column=0, sticky="w", padx=(0, 4), pady=2
        )
        self.rb_ai_pt = tk.Radiobutton(tool_grid, text="Point (+)", variable=self.tool_mode, value="AI Point",
                                       font=("Segoe UI", 8, "bold"), fg=SUCCESS, bg=SURFACE, selectcolor=FIELD,
                                       activebackground=SURFACE, activeforeground=SUCCESS)
        self.rb_ai_pt.grid(row=0, column=1, sticky="w", pady=2)

        self.rb_ai_er = tk.Radiobutton(tool_grid, text="Erase (−)", variable=self.tool_mode, value="AI Erase",
                                       font=("Segoe UI", 8, "bold"), fg=DANGER, bg=SURFACE, selectcolor=FIELD,
                                       activebackground=SURFACE, activeforeground=DANGER)
        self.rb_ai_er.grid(row=0, column=2, sticky="w", pady=2)

        self.rb_ai_bx = tk.Radiobutton(tool_grid, text="Box (+)", variable=self.tool_mode, value="AI Box",
                                       font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, selectcolor=FIELD,
                                       activebackground=SURFACE, activeforeground=ACCENT)
        self.rb_ai_bx.grid(row=0, column=3, sticky="w", pady=2)

        # Row 1: Direct Override Tools (Guaranteed 100% Reliable Override)
        tk.Label(tool_grid, text="Direct Edit:", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=SURFACE, anchor="w").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=2
        )
        self.rb_br_pt = tk.Radiobutton(tool_grid, text="Brush (+)", variable=self.tool_mode, value="Brush Paint",
                                       font=("Segoe UI", 8, "bold"), fg=SUCCESS, bg=SURFACE, selectcolor=FIELD,
                                       activebackground=SURFACE, activeforeground=SUCCESS)
        self.rb_br_pt.grid(row=1, column=1, sticky="w", pady=2)

        self.rb_br_er = tk.Radiobutton(tool_grid, text="Eraser (−)", variable=self.tool_mode, value="Brush Erase",
                                       font=("Segoe UI", 8, "bold"), fg=DANGER, bg=SURFACE, selectcolor=FIELD,
                                       activebackground=SURFACE, activeforeground=DANGER)
        self.rb_br_er.grid(row=1, column=2, sticky="w", pady=2)

        self.rb_bx_cut = tk.Radiobutton(tool_grid, text="Box Cut (−)", variable=self.tool_mode, value="Box Cut",
                                        font=("Segoe UI", 8, "bold"), fg=WARNING, bg=SURFACE, selectcolor=FIELD,
                                        activebackground=SURFACE, activeforeground=WARNING)
        self.rb_bx_cut.grid(row=1, column=3, sticky="w", pady=2)

        # Row 2: Brush Radius Control
        r_b_size = tk.Frame(c2, bg=SURFACE)
        r_b_size.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_b_size, text="Brush Radius (px):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=15, anchor="w").pack(side=tk.LEFT)
        self.brush_size_entry = tk.Entry(r_b_size, width=5, font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT,
                                         insertbackground=TEXT, justify="center", relief=tk.FLAT)
        self.brush_size_entry.pack(side=tk.RIGHT)
        self.brush_size_entry.insert(0, str(self.brush_size_var.get()))
        self.brush_size_entry.bind("<Return>", lambda e: self._on_brush_size_commit())
        self.brush_size_entry.bind("<FocusOut>", lambda e: self._on_brush_size_commit())
        self.s_bs = ttk.Scale(r_b_size, from_=2, to=120, variable=self.brush_size_var,
                              command=lambda v: self._on_brush_scale_moved(v))
        self.s_bs.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Row 3: AI Text Prompt Segmentation (OwLV2 + SAM 2.1 Grounded Pipeline)
        r_txt = tk.Frame(c2, bg=SURFACE)
        r_txt.pack(fill=tk.X, pady=(3, 3))
        tk.Label(r_txt, text="Text Prompt:", font=("Segoe UI", 8, "bold"), fg=MUTED, bg=SURFACE, width=11, anchor="w").pack(side=tk.LEFT)
        self.text_prompt_var = tk.StringVar(value="")
        self.text_prompt_entry = tk.Entry(r_txt, textvariable=self.text_prompt_var, font=("Segoe UI", 9),
                                          bg=FIELD, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT)
        self.text_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), ipady=2)
        self.text_prompt_entry.bind("<Return>", lambda e: self._on_generate_from_text())

        self.btn_text_prompt = tk.Button(r_txt, text="✨ AI Mask", command=self._on_generate_from_text,
                                         font=("Segoe UI", 8, "bold"), bg=ACCENT, fg=BG, activebackground=ACCENT_HOVER,
                                         relief=tk.FLAT, padx=8, pady=3, cursor="hand2")
        self.btn_text_prompt.pack(side=tk.RIGHT)

        # Row 4: Shortcuts and Guidance
        lbl_prompt_hint = tk.Label(c2, text="💡 Tip: Text prompt, click or drag box · Alt+Click erases · Brush overrides blind spots",
                                   font=("Segoe UI", 8, "italic"), fg=MUTED, bg=SURFACE, wraplength=460, justify=tk.LEFT)
        lbl_prompt_hint.pack(anchor="w", pady=(1, 4))

        # Row 5: Action Buttons (Undo, Clear, Reset)
        r_pbtns = tk.Frame(c2, bg=SURFACE)
        r_pbtns.pack(fill=tk.X, pady=(2, 3))

        self.btn_undo = tk.Button(r_pbtns, text="↩ Undo (Ctrl+Z)", command=self._undo_last_action,
                                  font=("Segoe UI", 8, "bold"), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, pady=5, cursor="hand2")
        self.btn_undo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_clr_frame = tk.Button(r_pbtns, text="Clear Frame (Del)", command=self._clear_current_frame_prompts,
                                        font=("Segoe UI", 8), bg=FIELD, fg=WARNING, relief=tk.FLAT, padx=6, pady=5, cursor="hand2")
        self.btn_clr_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_clr_all = tk.Button(r_pbtns, text="Reset All", command=self._clear_all_prompts,
                                     font=("Segoe UI", 8), bg=FIELD, fg=DANGER, relief=tk.FLAT, padx=6, pady=5, cursor="hand2")
        self.btn_clr_all.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Row 6: Fresh Re-Prompt / Clear Conditioning Button
        self.btn_fresh = tk.Button(c2, text="🔄 Fresh Re-Prompt (Reset Memory Conditioning)",
                                   command=self._reset_current_frame_conditioning,
                                   font=("Segoe UI", 8), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=6, pady=4, cursor="hand2")
        self.btn_fresh.pack(fill=tk.X, pady=(1, 3))

        self.keyframe_info_lbl = tk.Label(c2, text="Keyframes: 0 | Current Frame: Unprompted",
                                          font=("Segoe UI", 8), fg=MUTED, bg=SURFACE, wraplength=460, justify=tk.LEFT)
        self.keyframe_info_lbl.pack(anchor="w", pady=(2, 0))

        # CARD 3: Temporal Tracking / Propagation
        c3 = self._create_card(self.scrollable_frame, "⚡ TIMELINE PROPAGATION")
        r_track = tk.Frame(c3, bg=SURFACE)
        r_track.pack(fill=tk.X, pady=(4, 6))

        self.btn_track_fwd = tk.Button(r_track, text="Track Forward ▶", command=lambda: self._start_tracking("forward"),
                                       font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=8, pady=6, cursor="hand2")
        self.btn_track_fwd.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_track_bwd = tk.Button(r_track, text="◀ Track Backward", command=lambda: self._start_tracking("backward"),
                                       font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=8, pady=6, cursor="hand2")
        self.btn_track_bwd.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_track_all = tk.Button(c3, text="⚡ Track Full Sequence (Both Directions)", command=lambda: self._start_tracking("both"),
                                       font=("Segoe UI", 9, "bold"), bg=ACCENT, fg=BG, activebackground=ACCENT_HOVER,
                                       relief=tk.FLAT, padx=10, pady=7, cursor="hand2")
        self.btn_track_all.pack(fill=tk.X, pady=(2, 6))

        self.track_progress = ttk.Progressbar(c3, orient="horizontal", mode="determinate")
        self.track_progress.pack(fill=tk.X, pady=(2, 4))

        self.track_status_lbl = tk.Label(c3, text="Tracking idle. Place prompt clicks to track.",
                                         font=("Segoe UI", 8), fg=MUTED, bg=SURFACE)
        self.track_status_lbl.pack(anchor="w")

        # CARD 4: Mask Customization & Refinement (With Direct Numeric Inputs & Instant Drag Debounce)
        c4 = self._create_card(self.scrollable_frame, "✨ MASK REFINEMENT")

        # Dilation / Erosion
        r_dil = tk.Frame(c4, bg=SURFACE)
        r_dil.pack(fill=tk.X, pady=2)
        tk.Label(r_dil, text="Dilation / Margin (px):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        self.dilation_entry = tk.Entry(r_dil, width=5, font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT,
                                       insertbackground=TEXT, justify="center", relief=tk.FLAT)
        self.dilation_entry.pack(side=tk.RIGHT)
        self.dilation_entry.insert(0, str(self.dilation_var.get()))
        self.dilation_entry.bind("<Return>", lambda e: self._on_entry_param_commit("dilation"))
        self.dilation_entry.bind("<FocusOut>", lambda e: self._on_entry_param_commit("dilation"))
        self.s_dil = ttk.Scale(r_dil, from_=-25, to=25, variable=self.dilation_var,
                               command=lambda v: self._on_refine_slider_moved("dilation", v))
        self.s_dil.bind("<ButtonRelease-1>", lambda e: self._flush_refine_param())
        self.s_dil.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Feather Radius
        r_fea = tk.Frame(c4, bg=SURFACE)
        r_fea.pack(fill=tk.X, pady=2)
        tk.Label(r_fea, text="Edge Feather (Softness):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        self.feather_entry = tk.Entry(r_fea, width=5, font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT,
                                      insertbackground=TEXT, justify="center", relief=tk.FLAT)
        self.feather_entry.pack(side=tk.RIGHT)
        self.feather_entry.insert(0, str(self.feather_var.get()))
        self.feather_entry.bind("<Return>", lambda e: self._on_entry_param_commit("feather"))
        self.feather_entry.bind("<FocusOut>", lambda e: self._on_entry_param_commit("feather"))
        self.s_fea = ttk.Scale(r_fea, from_=0, to=20, variable=self.feather_var,
                               command=lambda v: self._on_refine_slider_moved("feather", v))
        self.s_fea.bind("<ButtonRelease-1>", lambda e: self._flush_refine_param())
        self.s_fea.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Edge Threshold
        r_thr = tk.Frame(c4, bg=SURFACE)
        r_thr.pack(fill=tk.X, pady=2)
        tk.Label(r_thr, text="Binarize Threshold:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        self.threshold_entry = tk.Entry(r_thr, width=5, font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT,
                                        insertbackground=TEXT, justify="center", relief=tk.FLAT)
        self.threshold_entry.pack(side=tk.RIGHT)
        self.threshold_entry.insert(0, str(self.threshold_var.get()))
        self.threshold_entry.bind("<Return>", lambda e: self._on_entry_param_commit("threshold"))
        self.threshold_entry.bind("<FocusOut>", lambda e: self._on_entry_param_commit("threshold"))
        self.s_thr = ttk.Scale(r_thr, from_=1, to=254, variable=self.threshold_var,
                               command=lambda v: self._on_refine_slider_moved("threshold", v))
        self.s_thr.bind("<ButtonRelease-1>", lambda e: self._flush_refine_param())
        self.s_thr.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Speckle Removal Area
        r_spk = tk.Frame(c4, bg=SURFACE)
        r_spk.pack(fill=tk.X, pady=2)
        tk.Label(r_spk, text="Clean Speckles (min px):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        self.speckle_entry = tk.Entry(r_spk, width=5, font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT,
                                      insertbackground=TEXT, justify="center", relief=tk.FLAT)
        self.speckle_entry.pack(side=tk.RIGHT)
        self.speckle_entry.insert(0, str(self.min_speckle_var.get()))
        self.speckle_entry.bind("<Return>", lambda e: self._on_entry_param_commit("speckle"))
        self.speckle_entry.bind("<FocusOut>", lambda e: self._on_entry_param_commit("speckle"))
        self.s_spk = ttk.Scale(r_spk, from_=0, to=150, variable=self.min_speckle_var,
                               command=lambda v: self._on_refine_slider_moved("speckle", v))
        self.s_spk.bind("<ButtonRelease-1>", lambda e: self._flush_refine_param())
        self.s_spk.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Fill Holes & Invert
        r_chk = tk.Frame(c4, bg=SURFACE)
        r_chk.pack(fill=tk.X, pady=(4, 2))
        self.chk_hole = tk.Checkbutton(r_chk, text="Fill Interior Holes", variable=self.fill_holes_var,
                                       font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, selectcolor=FIELD,
                                       command=self._on_refine_param_changed)
        self.chk_hole.pack(side=tk.LEFT, padx=(0, 10))

        self.chk_inv = tk.Checkbutton(r_chk, text="Invert Mask", variable=self.invert_var,
                                      font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, selectcolor=FIELD,
                                      command=self._on_refine_param_changed)
        self.chk_inv.pack(side=tk.LEFT)

        # CARD 5: Export Settings & Codecs (Hardware Accelerated)
        c5 = self._create_card(self.scrollable_frame, "💾 EXPORT MASKS & CUTOUTS")

        # Output Directory
        r_out = tk.Frame(c5, bg=SURFACE)
        r_out.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_out, text="Output:", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE, width=6, anchor="w").pack(side=tk.LEFT)
        self.out_entry = tk.Entry(r_out, textvariable=self.export_dir_var, font=("Segoe UI", 8),
                                  bg=FIELD, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT)
        self.out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_browse_out = tk.Button(r_out, text="Browse", command=self._browse_output_dir,
                                        font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_browse_out.pack(side=tk.LEFT)

        # Target Checkboxes (Clean layout, Cutout is Optional)
        r_targets = tk.Frame(c5, bg=SURFACE)
        r_targets.pack(fill=tk.X, pady=(2, 4))
        self.chk_bin = tk.Checkbutton(r_targets, text="RealityScan Binary Masks (.mask.png)",
                                      variable=self.export_binary_var, font=("Segoe UI", 8, "bold"),
                                      fg=ACCENT, bg=SURFACE, selectcolor=FIELD,
                                      command=self._on_export_targets_changed)
        self.chk_bin.pack(anchor="w")

        self.chk_cut = tk.Checkbutton(r_targets, text="Transparent RGBA Cutouts [Optional]",
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
        self.cb_mask = ttk.Combobox(r_mask_fmt, textvariable=self.mask_fmt_var,
                                    values=["png", "jpg", "webp"],
                                    state="readonly", width=18, font=("Segoe UI", 8))
        self.cb_mask.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Output Resolution / Scale
        r_res = tk.Frame(c5, bg=SURFACE)
        r_res.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_res, text="Export Size:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=12, anchor="w").pack(side=tk.LEFT)
        self.cb_scale = ttk.Combobox(r_res, textvariable=self.export_scale_var,
                                     values=["2K QHD (2048px - Auto Aspect)", "Original (100% Native)", "50% Half Res (Auto Aspect)", "4K UHD (3840px - Auto Aspect)", "1K HD (1024px - Auto Aspect)"],
                                     state="readonly", width=18, font=("Segoe UI", 8))
        self.cb_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Quality / Compression Slider
        r_qual = tk.Frame(c5, bg=SURFACE)
        r_qual.pack(fill=tk.X, pady=(2, 4))
        tk.Label(r_qual, text="Quality (WebP/JPG):", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE, width=15, anchor="w").pack(side=tk.LEFT)
        self.qual_lbl = tk.Label(r_qual, text="92", font=("Segoe UI", 8, "bold"), fg=ACCENT, bg=SURFACE, width=3)
        self.qual_lbl.pack(side=tk.RIGHT)
        self.s_qual = ttk.Scale(r_qual, from_=50, to=100, variable=self.export_quality_var,
                                command=lambda v: self.qual_lbl.configure(text=str(int(float(v)))))
        self.s_qual.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Hardware Acceleration & Size Badge
        r_hw = tk.Frame(c5, bg=SURFACE)
        r_hw.pack(fill=tk.X, pady=(2, 4))
        self.chk_hw = tk.Checkbutton(r_hw, text="⚡ Hardware Accelerated (CUDA GPU + I/O)",
                                     variable=self.export_hw_accel_var, font=("Segoe UI", 8, "bold"),
                                     fg=ACCENT, bg=SURFACE, selectcolor=FIELD)
        self.chk_hw.pack(side=tk.LEFT)

        lbl_hint = tk.Label(c5, text="💡 Tip: WebP at 2K reduces cutout file size from ~70MB down to ~1.5MB with zero visual loss!",
                            font=("Segoe UI", 8, "italic"), fg=MUTED, bg=SURFACE, justify=tk.LEFT, wraplength=440)
        lbl_hint.pack(anchor="w", pady=(0, 6))

        # Export Action Frame with Pause & Stop / Cancel controls
        self.export_action_frame = tk.Frame(c5, bg=SURFACE)
        self.export_action_frame.pack(fill=tk.X, pady=(2, 4))

        self.btn_export = tk.Button(self.export_action_frame, text="🚀 EXPORT REALITYSCAN MASKS (BINARY ONLY)",
                                    command=self._export_masks_action,
                                    font=("Segoe UI", 9, "bold"), bg=SUCCESS, fg="#ffffff",
                                    activebackground="#059669", relief=tk.FLAT, padx=10, pady=8, cursor="hand2",
                                    wraplength=420, justify=tk.CENTER)
        self.btn_export.pack(fill=tk.X)

        # Active Pause & Cancel buttons (visible while export is running)
        self.export_active_bar = tk.Frame(self.export_action_frame, bg=SURFACE)

        self.btn_pause_export = tk.Button(self.export_active_bar, text="⏸ Pause Export",
                                          command=self._toggle_export_pause,
                                          font=("Segoe UI", 8, "bold"), bg=WARNING, fg="#ffffff",
                                          activebackground="#d97706", relief=tk.FLAT, padx=8, pady=7, cursor="hand2")
        self.btn_pause_export.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_stop_export = tk.Button(self.export_active_bar, text="⏹ Stop / Cancel",
                                         command=self._cancel_export_action,
                                         font=("Segoe UI", 8, "bold"), bg=DANGER, fg="#ffffff",
                                         activebackground="#dc2626", relief=tk.FLAT, padx=8, pady=7, cursor="hand2")
        self.btn_stop_export.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.export_progress = ttk.Progressbar(c5, orient="horizontal", mode="determinate")
        self.export_progress.pack(fill=tk.X, pady=(2, 4))

        self.export_status_lbl = tk.Label(c5, text="", font=("Segoe UI", 8), fg=MUTED, bg=SURFACE,
                                          wraplength=440, justify=tk.LEFT, anchor="w")
        self.export_status_lbl.pack(fill=tk.X)

    def _create_card(self, parent: tk.Widget, title: str) -> tk.Frame:
        card = tk.Frame(parent, bg=SURFACE, bd=1, relief=tk.SOLID, padx=8, pady=6)
        card.pack(fill=tk.X, pady=(0, 6))
        lbl = tk.Label(card, text=title, font=("Segoe UI", 9, "bold"), fg=ACCENT, bg=SURFACE)
        lbl.pack(anchor="w", pady=(0, 3))
        sep = tk.Frame(card, bg=BORDER, height=1)
        sep.pack(fill=tk.X, pady=(0, 5))
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
        self.color_combo = ttk.Combobox(v_toolbar, textvariable=self.mask_color_name,
                                        values=list(MASK_COLORS.keys()), state="readonly", width=9, font=("Segoe UI", 8))
        self.color_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.color_combo.bind("<<ComboboxSelected>>", lambda e: self._render_current_frame())

        # Opacity Slider
        tk.Label(v_toolbar, text="Opacity:", font=("Segoe UI", 8), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        self.s_op = ttk.Scale(v_toolbar, from_=0.1, to=1.0, variable=self.overlay_opacity, command=lambda v: self._render_current_frame())
        self.s_op.pack(side=tk.LEFT, padx=(0, 14), fill=tk.X)

        # Zoom Controls
        self.btn_fit = tk.Button(v_toolbar, text="Fit View [F]", command=self._zoom_to_fit,
                                 font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_fit.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_100 = tk.Button(v_toolbar, text="100%", command=self._zoom_to_100,
                                 font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_100.pack(side=tk.LEFT, padx=(0, 10))

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

        self.btn_first = tk.Button(r_ctrl, text="|◀", command=lambda: self._set_frame(0),
                                   font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_first.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_prev10 = tk.Button(r_ctrl, text="◀ -10", command=lambda: self._set_frame(self.current_frame_idx - 10),
                                    font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_prev10.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_prev = tk.Button(r_ctrl, text="◀ Prev", command=lambda: self._set_frame(self.current_frame_idx - 1),
                                  font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_play = tk.Button(r_ctrl, text="▶ Play", command=self._toggle_playback,
                                  font=("Segoe UI", 8, "bold"), bg=FIELD, fg=ACCENT, relief=tk.FLAT, padx=10, cursor="hand2")
        self.btn_play.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_next = tk.Button(r_ctrl, text="Next ▶", command=lambda: self._set_frame(self.current_frame_idx + 1),
                                  font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_next.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_next10 = tk.Button(r_ctrl, text="+10 ▶", command=lambda: self._set_frame(self.current_frame_idx + 10),
                                    font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_next10.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_last = tk.Button(r_ctrl, text="▶|", command=lambda: self._set_frame(self.engine.num_frames - 1),
                                  font=("Segoe UI", 8), bg=FIELD, fg=TEXT, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_last.pack(side=tk.LEFT, padx=(0, 14))

        # Keyframe jump buttons
        self.btn_prev_kf = tk.Button(r_ctrl, text="◀ Keyframe", command=self._jump_prev_keyframe,
                                     font=("Segoe UI", 8), bg=FIELD, fg=WARNING, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_prev_kf.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_next_kf = tk.Button(r_ctrl, text="Keyframe ▶", command=self._jump_next_keyframe,
                                     font=("Segoe UI", 8), bg=FIELD, fg=WARNING, relief=tk.FLAT, padx=6, cursor="hand2")
        self.btn_next_kf.pack(side=tk.LEFT, padx=(0, 14))

        # Frame Readout Badge
        self.frame_readout_lbl = tk.Label(r_ctrl, text="Frame: 0 / 0", font=("Segoe UI", 9, "bold"),
                                          fg=TEXT, bg=FIELD, padx=10, pady=2, relief=tk.FLAT)
        self.frame_readout_lbl.pack(side=tk.RIGHT)

        # Scrubber Slider
        self.scrubber_var = tk.IntVar(value=0)
        self.scrubber = ttk.Scale(timeline_bar, from_=0, to=0, orient="horizontal",
                                  variable=self.scrubber_var, command=self._on_scrubber_drag)
        self.scrubber.pack(fill=tk.X, pady=(2, 0))

    def _setup_tooltips(self):
        """Attaches instructive tooltips to all interactive controls and sliders across the UI."""
        tips = {
            # Header
            "model_combo": "Select SAM 2.1 neural backbone:\n• tiny: Fastest, lowest VRAM (~38MB weights)\n• small: Balanced speed & precision (~92MB)\n• base_plus: High detail (~201MB)\n• large: Maximum segmentation accuracy (~898MB)",
            "device_lbl": "Hardware acceleration device currently powering SAM 2.1 inference & memory tracking.",

            # Source Media
            "btn_folder": "Open a folder containing an ordered sequence of images (.png, .jpg, .tiff, etc.)",
            "btn_video": "Open a video file (.mp4, .mov, .avi, etc.) to extract and segment frames.",

            # Prompt & Selection Tools
            "rb_ai_pt": "AI Foreground Point (+):\nClick inside target subject to prompt SAM 2.1 to select it.",
            "rb_ai_er": "AI Background Point (−):\nClick on non-target or background areas to subtract them. (Shortcut: Alt+Click)",
            "rb_ai_bx": "AI Bounding Box (+):\nDrag a rectangular marquee around the subject. SAM 2.1 segments within the box boundary.",
            "rb_br_pt": "Direct Brush Paint (+):\nDraw directly on the mask to manually force-include pixels. Overrides model ambiguity.",
            "rb_br_er": "Direct Brush Eraser (−):\nErase directly on the mask to manually carve away unwanted parts. 100% reliable direct override.",
            "rb_bx_cut": "Direct Box Cut (−):\nDrag a rectangle to instantly carve out and delete everything within it from the mask.",
            "s_bs": "Adjust the radius (in pixels) of the direct Brush Paint and Eraser tools.",
            "brush_size_entry": "Type exact brush radius in pixels and press Enter.",
            "text_prompt_entry": "Type natural language description (e.g., 'white turntable', 'bracket', 'screw', 'person') for automated zero-shot object detection.",
            "btn_text_prompt": "Run OwLV2 zero-shot detection to find bounding boxes matching the text prompt, then feed directly into SAM 2.1 to mask the subject.",
            "btn_undo": "Undo the last prompt point, box, or brush stroke (Ctrl+Z).",
            "btn_clr_frame": "Remove all prompts and generated mask from the active frame (Delete key).",
            "btn_clr_all": "Completely reset and clear all prompts, keyframes, and masks across all frames.",
            "btn_fresh": "Fresh Re-Prompt:\nPurges cached temporal conditioning and memory features for this frame so you can start clean.",

            # Timeline Propagation
            "btn_track_fwd": "Propagate and track the mask forward from the current keyframe through remaining frames.",
            "btn_track_bwd": "Propagate and track the mask backward from the current keyframe towards the beginning.",
            "btn_track_all": "Propagate and track the mask across the entire sequence in both directions.",

            # Mask Refinement
            "s_dil": "Dilation / Margin:\nExpand (positive) or contract (negative) mask edges by pixels to eliminate background fringes.",
            "dilation_entry": "Exact pixel dilation/margin value (-25 to +25). Press Enter to apply.",
            "s_fea": "Edge Feather:\nApplies Gaussian softening to mask borders for smooth, natural subject cutouts.",
            "feather_entry": "Exact feather radius in pixels (0 to 20). Press Enter to apply.",
            "s_thr": "Binarize Threshold:\nCutoff probability (1–254) for converting model logits into binary foreground/background.",
            "threshold_entry": "Exact binarization cutoff threshold (1–254). Press Enter to apply.",
            "s_spk": "Clean Speckles:\nAutomatically removes floating noise islands smaller than the specified pixel area.",
            "speckle_entry": "Minimum island area in pixels (0 to 150) below which noise is erased. Press Enter to apply.",
            "chk_hole": "Fill Interior Holes:\nAutomatically detects and fills enclosed unselected holes inside the segmented subject.",
            "chk_inv": "Invert Mask:\nSwaps foreground and background selections.",

            # Export Settings
            "out_entry": "Folder path where masks and cutout images will be saved.",
            "btn_browse_out": "Browse file system to choose output export folder.",
            "chk_bin": "Generate 8-bit black-and-white mask files named <image>.mask.png, ready for RealityScan photogrammetry.",
            "chk_cut": "Generate transparent RGBA images with background removed.",
            "cb_cut": "Cutout file format:\n• WebP: Highly compressed (~1.5MB vs 70MB PNG) with lossless alpha channel.\n• PNG: Standard lossless RGBA.",
            "cb_mask": "File format for binary mask images (PNG, JPG, or WebP).",
            "cb_scale": "Target resolution scaling for export:\n• Native: Full original camera resolution\n• 2K QHD / 4K UHD: Standard downscaled targets",
            "s_qual": "Compression quality (50–100) for WebP and JPG exports.",
            "chk_hw": "Enable GPU-accelerated batch image processing and multithreaded file I/O for faster exports.",
            "btn_export": "Start exporting masks/cutouts for all frames in the sequence.",
            "btn_pause_export": "Pause the running export process. Click again to resume.",
            "btn_stop_export": "Cancel and abort the active export immediately.",

            # Viewport & Scrubber
            "color_combo": "Change the tinted overlay color used to visualize the mask on the canvas.",
            "s_op": "Adjust transparency of the mask overlay on top of the original image.",
            "btn_fit": "Fit the image to the viewport window (Hot-key: F).",
            "btn_100": "Zoom image to 100% native pixel resolution (Hot-key: 1).",
            "btn_first": "Jump to the first frame (Frame 0).",
            "btn_prev10": "Jump backward 10 frames.",
            "btn_prev": "Step to previous frame (Left Arrow).",
            "btn_play": "Toggle real-time sequence playback (Spacebar).",
            "btn_next": "Step to next frame (Right Arrow).",
            "btn_next10": "Jump forward 10 frames.",
            "btn_last": "Jump to the last frame in the sequence.",
            "btn_prev_kf": "Jump to the previous annotated keyframe.",
            "btn_next_kf": "Jump to the next annotated keyframe.",
            "scrubber": "Drag or click to scrub through frames across the timeline.",
        }

        for attr_name, text in tips.items():
            widget = getattr(self, attr_name, None)
            if widget is not None:
                add_tooltip(widget, text)

    # -------------------------------------------------------------------------
    # Event Bindings (Canvas, Mouse, Keyboard)
    # -------------------------------------------------------------------------

    def _is_typing_focused(self, event=None) -> bool:
        """
        Returns True if keyboard focus is currently inside any editable text widget
        (such as the Text Prompt entry, Brush radius entry, refinement entries, or export path).
        Prevents global viewport shortcuts (Spacebar, Arrows, Delete, F, etc.) from interfering
        with text input.
        """
        try:
            focused = self.dialog.focus_get()
            if focused is not None:
                w_class = focused.winfo_class()
                if w_class in ("Entry", "TEntry", "Text", "Combobox", "TCombobox"):
                    return True
                if isinstance(focused, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
                    return True

            if event is not None and hasattr(event, "widget") and event.widget is not None:
                w = event.widget
                w_class = getattr(w, "winfo_class", lambda: "")()
                if w_class in ("Entry", "TEntry", "Text", "Combobox", "TCombobox"):
                    return True
                if isinstance(w, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox)):
                    return True
        except Exception:
            pass
        return False

    def _bind_events(self):
        # Canvas mouse interactions
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_down)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_up)

        self.canvas.bind("<ButtonPress-2>", self._on_canvas_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_canvas_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_canvas_pan_end)

        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_down)
        self.canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_right_up)

        self.canvas.bind("<Motion>", self._on_canvas_motion)

        # Centralized MouseWheel Routing (Smooth Sidebar Scroll & Viewport Zoom)
        def _on_window_mousewheel(event):
            rx = event.x_root
            ry = event.y_root

            # 1. Check if pointer is over sidebar area
            if hasattr(self, "sidebar_frame") and self.sidebar_frame is not None and self.sidebar_frame.winfo_exists():
                try:
                    sx = self.sidebar_frame.winfo_rootx()
                    sy = self.sidebar_frame.winfo_rooty()
                    sw = self.sidebar_frame.winfo_width()
                    sh = self.sidebar_frame.winfo_height()
                    if sx <= rx <= sx + sw and sy <= ry <= sy + sh:
                        if hasattr(self, "sidebar_canvas") and self.sidebar_canvas is not None and self.sidebar_canvas.winfo_exists():
                            delta = event.delta if hasattr(event, "delta") and event.delta else (120 if getattr(event, "num", None) == 4 else -120)
                            self.sidebar_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
                            return "break"
                except Exception:
                    pass

            # 2. Check if pointer is over main viewport canvas
            if hasattr(self, "canvas") and self.canvas is not None and self.canvas.winfo_exists():
                try:
                    cx = self.canvas.winfo_rootx()
                    cy = self.canvas.winfo_rooty()
                    cw = self.canvas.winfo_width()
                    ch = self.canvas.winfo_height()
                    if cx <= rx <= cx + cw and cy <= ry <= cy + ch:
                        self._on_canvas_mousewheel(event)
                        return "break"
                except Exception:
                    pass

        self.dialog.bind("<MouseWheel>", _on_window_mousewheel)
        self.dialog.bind("<Button-4>", _on_window_mousewheel)
        self.dialog.bind("<Button-5>", _on_window_mousewheel)

        def _on_canvas_enter(event):
            self._sync_alt_state(event)

        def _on_canvas_leave(event):
            if self.brush_cursor_id:
                self.canvas.delete(self.brush_cursor_id)
                self.brush_cursor_id = None
            self._on_alt_up()

        self.canvas.bind("<Enter>", _on_canvas_enter)
        self.canvas.bind("<Leave>", _on_canvas_leave)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Alt key bindings for Erase / Negative Prompt
        self.dialog.bind("<Alt_L>", self._on_alt_down)
        self.dialog.bind("<Alt_R>", self._on_alt_down)
        self.dialog.bind("<KeyRelease-Alt_L>", self._on_alt_up)
        self.dialog.bind("<KeyRelease-Alt_R>", self._on_alt_up)
        self.dialog.bind("<FocusOut>", lambda e: self._on_alt_up())
        self.dialog.bind("<FocusIn>", lambda e: self._on_alt_up())
        self.canvas.bind("<Alt-ButtonPress-1>", self._on_canvas_alt_down)

        # Keyboard shortcuts (Guarded against active text entry focus)
        def _on_key_space(e):
            if self._is_typing_focused(e):
                return
            self._toggle_playback()
            return "break"

        def _on_key_left(e):
            if self._is_typing_focused(e):
                return
            self._set_frame(self.current_frame_idx - 1)
            return "break"

        def _on_key_right(e):
            if self._is_typing_focused(e):
                return
            self._set_frame(self.current_frame_idx + 1)
            return "break"

        def _on_key_undo(e):
            if self._is_typing_focused(e):
                return
            self._undo_last_action()
            return "break"

        def _on_key_delete(e):
            if self._is_typing_focused(e):
                return
            self._clear_current_frame_prompts()
            return "break"

        def _on_key_fit(e):
            if self._is_typing_focused(e):
                return
            self._zoom_to_fit()
            return "break"

        def _on_key_brush_dec(e):
            if self._is_typing_focused(e):
                return
            self._adjust_brush_size(-5)
            return "break"

        def _on_key_brush_inc(e):
            if self._is_typing_focused(e):
                return
            self._adjust_brush_size(5)
            return "break"

        self.dialog.bind("<space>", _on_key_space)
        self.dialog.bind("<Left>", _on_key_left)
        self.dialog.bind("<Right>", _on_key_right)
        self.dialog.bind("<Control-z>", _on_key_undo)
        self.dialog.bind("<Control-Z>", _on_key_undo)
        self.dialog.bind("<Delete>", _on_key_delete)
        self.dialog.bind("<f>", _on_key_fit)
        self.dialog.bind("<F>", _on_key_fit)
        self.dialog.bind("[", _on_key_brush_dec)
        self.dialog.bind("]", _on_key_brush_inc)

    # -------------------------------------------------------------------------
    # Model Loading (Asynchronous)
    # -------------------------------------------------------------------------

    def _init_model_async(self, model_key: str):
        self.status_lbl.configure(text=f"Loading SAM 2.1 ({model_key})...", fg=ACCENT)
        self.model_combo.configure(state=tk.DISABLED)

        def _worker():
            try:
                self.engine.load_model(model_key, progress_callback=lambda msg: self.dialog.after(0, lambda: self.status_lbl.configure(text=msg)))
                def _on_success():
                    self.model_combo.configure(state="readonly")
                    self.status_lbl.configure(text=f"SAM 2.1 {model_key.upper()} Ready", fg=SUCCESS)
                    # Refresh device badge
                    dev_text = f"⚡ {self.engine.device.upper()}"
                    if torch.cuda.is_available():
                        dev_text = f"⚡ CUDA: {torch.cuda.get_device_name(0)}"
                    self.device_lbl.configure(text=dev_text)
                    # If sequence is loaded, re-render current frame with new model's output
                    if self.engine.num_frames > 0:
                        self._cached_mask_key = None
                        self._cached_mask = None
                        self._composite_cache_key = None
                        self._set_frame(self.current_frame_idx)
                self.dialog.after(0, _on_success)
            except Exception as e:
                def _on_error():
                    self.model_combo.configure(state="readonly")
                    self.status_lbl.configure(text=f"Model Error: {e}", fg=DANGER)
                self.dialog.after(0, _on_error)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_changed(self, event=None):
        model_key = self.model_combo_var.get()
        self._init_model_async(model_key)

    # -------------------------------------------------------------------------
    # Text-Prompt AI Masking Callbacks (Grounded-SAM 2)
    # -------------------------------------------------------------------------

    def _on_generate_from_text(self):
        prompt_text = self.text_prompt_var.get().strip()
        if not prompt_text:
            messagebox.showinfo("Text Prompt", "Please enter a descriptive object name (e.g. 'turntable', 'bracket', 'screw', 'camera').", parent=self.dialog)
            return

        if self.engine.num_frames == 0:
            messagebox.showwarning("No Source", "Please load an image sequence or video first.", parent=self.dialog)
            return

        self.status_lbl.configure(text=f"🔍 Grounding '{prompt_text}' with OwLV2 & SAM 2.1...", fg=ACCENT)
        self.btn_text_prompt.configure(state=tk.DISABLED)

        def _worker():
            try:
                def _prog(msg):
                    self.dialog.after(0, lambda: self.status_lbl.configure(text=msg))

                mask_np, box, score = self.engine.generate_mask_from_text(
                    self.current_frame_idx,
                    prompt_text,
                    threshold=0.10,
                    progress_callback=_prog
                )
                if mask_np is not None:
                    self.dialog.after(0, lambda: self._on_text_mask_success(prompt_text, box, score))
                else:
                    self.dialog.after(0, lambda: self._on_text_mask_not_found(prompt_text))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_text_mask_error(prompt_text, str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_text_mask_success(self, prompt_text: str, box: List[float], score: float):
        self.btn_text_prompt.configure(state=tk.NORMAL)
        self._cached_mask_key = None
        self._cached_mask = None
        self._composite_cache_key = None
        self._set_frame(self.current_frame_idx)
        score_pct = int(round(score * 100))
        self.status_lbl.configure(text=f"✨ AI Mask created for '{prompt_text}' ({score_pct}% confidence)", fg=SUCCESS)

    def _on_text_mask_not_found(self, prompt_text: str):
        self.btn_text_prompt.configure(state=tk.NORMAL)
        self.status_lbl.configure(text=f"No matching '{prompt_text}' detected. Try different phrasing.", fg=WARNING)

    def _on_text_mask_error(self, prompt_text: str, err: str):
        self.btn_text_prompt.configure(state=tk.NORMAL)
        self.status_lbl.configure(text=f"Text Prompt Error: {err}", fg=DANGER)

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

        has_mixed = len(set(self.engine.frame_dims)) > 1
        res_summary = "Mixed Aspect (Auto H/V)" if has_mixed else f"{self.engine.orig_width}x{self.engine.orig_height}"
        self.source_info_lbl.configure(
            text=f"Source: {os.path.basename(path)}\n"
                 f"Frames: {n_frames} | Res: {res_summary}"
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

        # Update Frame Readout with per-frame dimensions and orientation
        fname = self.engine.frame_names[clamped] if clamped < len(self.engine.frame_names) else ""
        cur_w, cur_h = self.engine.get_frame_dims(clamped)
        ratio_tag = "Portrait" if cur_h > cur_w else ("Landscape" if cur_w > cur_h else "Square")
        kf_star = " ⭐ Keyframe" if self.engine.has_prompts(clamped) else ""
        self.frame_readout_lbl.configure(
            text=f"Frame: {clamped + 1} / {self.engine.num_frames}  ({fname}) [{cur_w}x{cur_h} {ratio_tag}]{kf_star}"
        )

        # Check if frame dimensions or orientation changed, auto-fit view cleanly
        prev_dims = getattr(self, "_last_rendered_dims", None)
        self._last_rendered_dims = (cur_w, cur_h)
        if prev_dims is not None and prev_dims != (cur_w, cur_h):
            self._zoom_to_fit()

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
        cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
        if cur_w == 0 or cur_h == 0:
            return

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())

        scale_x = (canvas_w - 40) / cur_w
        scale_y = (canvas_h - 40) / cur_h
        self.zoom = min(scale_x, scale_y)

        # Center in canvas
        self.pan_x = (canvas_w - cur_w * self.zoom) / 2.0
        self.pan_y = (canvas_h - cur_h * self.zoom) / 2.0

        self.zoom_lbl.configure(text=f"Zoom: {int(self.zoom * 100)}%")
        self._render_current_frame()

    def _zoom_to_100(self):
        cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
        if cur_w == 0 or cur_h == 0:
            return
        self.zoom = 1.0
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        self.pan_x = (canvas_w - cur_w) / 2.0
        self.pan_y = (canvas_h - cur_h) / 2.0
        self.zoom_lbl.configure(text="Zoom: 100%")
        self._render_current_frame()

    def _on_canvas_mousewheel(self, event):
        cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
        if cur_w == 0:
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
        if self._is_typing_focused(event):
            return
        self.is_alt_pressed = True
        self.canvas.configure(cursor="circle")
        mode = self.tool_mode.get()
        if mode == "AI Box":
            self.status_lbl.configure(text="[ALT HELD] Box Marquee will CUT / ERASE (−)", fg=DANGER)
        elif mode == "Brush Paint":
            self.status_lbl.configure(text="[ALT HELD] Brush will ERASE (−)", fg=DANGER)
        else:
            self.status_lbl.configure(text="[ALT HELD] Click / Drag to ERASE / Subtract (−)", fg=DANGER)

    def _on_alt_up(self, event=None):
        self.is_alt_pressed = False
        self.canvas.configure(cursor="crosshair")
        self.status_lbl.configure(text="Ready. Select tool mode: AI Point/Box or Direct Brush/Cut", fg=MUTED)

    def _sync_alt_state(self, event=None):
        """Ensures is_alt_pressed is cleared if the physical Alt key is not depressed."""
        if event and hasattr(event, "state"):
            if not bool(event.state & 0x20000):
                self._on_alt_up()

    def _on_canvas_alt_down(self, event):
        """Dedicated handler for Alt+Left Click directly erasing mask at clicked point."""
        if self.engine.num_frames == 0:
            return
        cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
        img_x = (event.x - self.pan_x) / self.zoom
        img_y = (event.y - self.pan_y) / self.zoom
        if 0 <= img_x < cur_w and 0 <= img_y < cur_h:
            self._apply_point_prompt(img_x, img_y, label=0)

    def _on_canvas_motion(self, event):
        mode = self.tool_mode.get()
        if mode in ("Brush Paint", "Brush Erase"):
            r_img = self.brush_size_var.get()
            r_canvas = max(2, int(round(r_img * self.zoom)))
            x, y = event.x, event.y
            color = DANGER if (mode == "Brush Erase" or self.is_alt_pressed) else SUCCESS
            if self.brush_cursor_id:
                self.canvas.coords(self.brush_cursor_id, x - r_canvas, y - r_canvas, x + r_canvas, y + r_canvas)
                self.canvas.itemconfig(self.brush_cursor_id, outline=color)
            else:
                self.brush_cursor_id = self.canvas.create_oval(
                    x - r_canvas, y - r_canvas, x + r_canvas, y + r_canvas,
                    outline=color, width=1, dash=(3, 3), tags="brush_cursor"
                )
        else:
            if self.brush_cursor_id:
                self.canvas.delete(self.brush_cursor_id)
                self.brush_cursor_id = None

    def _on_canvas_left_down(self, event):
        if self.engine.num_frames == 0:
            return

        is_alt = bool(event.state & 0x20000) or self.is_alt_pressed
        is_shift = bool(event.state & 0x0001)
        mode = self.tool_mode.get()

        cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
        img_x = (event.x - self.pan_x) / self.zoom
        img_y = (event.y - self.pan_y) / self.zoom

        if mode in ("AI Box", "Box Cut"):
            self.box_start = (event.x, event.y)
            self.box_is_cut = (mode == "Box Cut" or is_alt or is_shift)
            return

        if mode in ("Brush Paint", "Brush Erase"):
            if 0 <= img_x < cur_w and 0 <= img_y < cur_h:
                self.is_brushing = True
                self.brush_active_mode = "erase" if (mode == "Brush Erase" or is_alt or is_shift) else "paint"
                self.engine.push_undo_state(self.current_frame_idx)
                self.current_stroke = [(img_x, img_y)]
                self.engine.brush_stroke_mask(
                    self.current_frame_idx,
                    self.current_stroke,
                    radius=self.brush_size_var.get(),
                    mode=self.brush_active_mode,
                    sync_to_sam2=False
                )
                self._render_current_frame(fast_brush=True)
            return

        # AI Point mode (AI Point or AI Erase)
        if 0 <= img_x < cur_w and 0 <= img_y < cur_h:
            label = 0 if (mode == "AI Erase" or is_alt or is_shift) else 1
            self._apply_point_prompt(img_x, img_y, label)

    def _on_canvas_left_drag(self, event):
        if self.box_start:
            x0, y0 = self.box_start
            x1, y1 = event.x, event.y
            outline_col = DANGER if self.box_is_cut else ACCENT
            if self.current_box_rect_id:
                self.canvas.coords(self.current_box_rect_id, x0, y0, x1, y1)
                self.canvas.itemconfig(self.current_box_rect_id, outline=outline_col)
            else:
                self.current_box_rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline=outline_col, width=2, dash=(4, 4)
                )
            return

        if self.is_brushing:
            cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
            img_x = (event.x - self.pan_x) / self.zoom
            img_y = (event.y - self.pan_y) / self.zoom
            img_x = max(0.0, min(float(cur_w - 1), img_x))
            img_y = max(0.0, min(float(cur_h - 1), img_y))
            if self.current_stroke:
                self.current_stroke.append((img_x, img_y))
                self.engine.brush_stroke_mask(
                    self.current_frame_idx,
                    self.current_stroke[-2:],
                    radius=self.brush_size_var.get(),
                    mode=self.brush_active_mode,
                    sync_to_sam2=False
                )
                self._render_current_frame(fast_brush=True)

    def _on_canvas_left_up(self, event):
        if self.box_start:
            if self.current_box_rect_id:
                self.canvas.delete(self.current_box_rect_id)
                self.current_box_rect_id = None

            sx0, sy0 = self.box_start
            sx1, sy1 = event.x, event.y
            self.box_start = None

            cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
            ix0 = (min(sx0, sx1) - self.pan_x) / self.zoom
            iy0 = (min(sy0, sy1) - self.pan_y) / self.zoom
            ix1 = (max(sx0, sx1) - self.pan_x) / self.zoom
            iy1 = (max(sy0, sy1) - self.pan_y) / self.zoom

            ix0 = max(0.0, min(float(cur_w - 1), ix0))
            iy0 = max(0.0, min(float(cur_h - 1), iy0))
            ix1 = max(0.0, min(float(cur_w - 1), ix1))
            iy1 = max(0.0, min(float(cur_h - 1), iy1))

            if (ix1 - ix0) > 5 and (iy1 - iy0) > 5:
                if self.box_is_cut:
                    self._apply_box_cut([ix0, iy0, ix1, iy1])
                else:
                    self._apply_box_prompt([ix0, iy0, ix1, iy1])
            self.box_is_cut = False
            return

        if self.is_brushing:
            self.is_brushing = False
            self.current_stroke = None
            self.status_lbl.configure(text=f"Brush {self.brush_active_mode} applied to frame {self.current_frame_idx + 1}", fg=SUCCESS)
            self.engine.sync_frame_mask_to_sam2(self.current_frame_idx, async_exec=True)
            self._cached_mask_key = None
            self._cached_mask = None
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)

    def _on_canvas_right_down(self, event):
        if self.engine.num_frames == 0:
            return

        cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
        img_x = (event.x - self.pan_x) / self.zoom
        img_y = (event.y - self.pan_y) / self.zoom

        mode = self.tool_mode.get()
        if 0 <= img_x < cur_w and 0 <= img_y < cur_h:
            if mode in ("Brush Paint", "Brush Erase"):
                self.is_right_interacting = True
                self.brush_active_mode = "erase"
                self.engine.push_undo_state(self.current_frame_idx)
                self.current_stroke = [(img_x, img_y)]
                self.engine.brush_stroke_mask(
                    self.current_frame_idx,
                    self.current_stroke,
                    radius=self.brush_size_var.get(),
                    mode="erase",
                    sync_to_sam2=False
                )
                self._render_current_frame(fast_brush=True)
            elif mode in ("AI Box", "Box Cut"):
                self.is_right_interacting = True
                self.box_start = (event.x, event.y)
                self.box_is_cut = True
            else:
                # Right click in point mode = immediate AI Erase click
                self._apply_point_prompt(img_x, img_y, label=0)
        else:
            # Outside image: pan canvas
            self._on_canvas_pan_start(event)

    def _on_canvas_right_drag(self, event):
        if self.is_right_interacting:
            mode = self.tool_mode.get()
            if mode in ("Brush Paint", "Brush Erase"):
                cur_w, cur_h = self.engine.get_frame_dims(self.current_frame_idx)
                img_x = max(0.0, min(float(cur_w - 1), (event.x - self.pan_x) / self.zoom))
                img_y = max(0.0, min(float(cur_h - 1), (event.y - self.pan_y) / self.zoom))
                if self.current_stroke:
                    self.current_stroke.append((img_x, img_y))
                    self.engine.brush_stroke_mask(
                        self.current_frame_idx,
                        self.current_stroke[-2:],
                        radius=self.brush_size_var.get(),
                        mode="erase",
                        sync_to_sam2=False
                    )
                    self._render_current_frame(fast_brush=True)
            elif mode in ("AI Box", "Box Cut"):
                self._on_canvas_left_drag(event)
        else:
            self._on_canvas_pan_drag(event)

    def _on_canvas_right_up(self, event):
        if self.is_right_interacting:
            mode = self.tool_mode.get()
            if mode in ("Brush Paint", "Brush Erase"):
                self.current_stroke = None
                self.engine.sync_frame_mask_to_sam2(self.current_frame_idx, async_exec=True)
                self._cached_mask_key = None
                self._cached_mask = None
                self._composite_cache_key = None
                self._set_frame(self.current_frame_idx)
            elif mode in ("AI Box", "Box Cut"):
                self._on_canvas_left_up(event)
            self.is_right_interacting = False
        else:
            self._on_canvas_pan_end(event)

    def _adjust_brush_size(self, delta: int):
        cur = self.brush_size_var.get()
        new_val = max(2, min(200, cur + delta))
        self.brush_size_var.set(new_val)
        self.brush_size_entry.delete(0, tk.END)
        self.brush_size_entry.insert(0, str(new_val))
        self.status_lbl.configure(text=f"Brush size: {new_val}px", fg=ACCENT)

    def _on_brush_scale_moved(self, value):
        val = int(round(float(value)))
        self.brush_size_entry.delete(0, tk.END)
        self.brush_size_entry.insert(0, str(val))

    def _on_brush_size_commit(self):
        val = self._parse_and_clamp_entry(self.brush_size_entry, self.brush_size_var.get(), 2, 200)
        self.brush_size_var.set(val)

    def _apply_point_prompt(self, x: float, y: float, label: int):
        lbl_text = "Erase (−)" if label == 0 else "Foreground (+)"
        self.status_lbl.configure(text=f"Evaluating {lbl_text} prompt on frame {self.current_frame_idx + 1}...",
                                  fg=DANGER if label == 0 else ACCENT)
        try:
            self.engine.add_point(self.current_frame_idx, x, y, label=label)
            self._cached_mask_key = None
            self._cached_mask = None
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)
            self.status_lbl.configure(text=f"{lbl_text} evaluated successfully on frame {self.current_frame_idx + 1}", fg=SUCCESS)
        except Exception as e:
            self.status_lbl.configure(text=f"Prompt Error: {e}", fg=DANGER)

    def _apply_box_prompt(self, box: List[float]):
        self.status_lbl.configure(text=f"Evaluating AI Box on frame {self.current_frame_idx + 1}...", fg=ACCENT)
        try:
            self.engine.add_box(self.current_frame_idx, box, clear_points=True)
            self._cached_mask_key = None
            self._cached_mask = None
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)
            self.status_lbl.configure(text=f"AI Box evaluated successfully on frame {self.current_frame_idx + 1}", fg=SUCCESS)
        except Exception as e:
            self.status_lbl.configure(text=f"Box Error: {e}", fg=DANGER)

    def _apply_box_cut(self, box: List[float]):
        self.status_lbl.configure(text=f"Applying Box Cut on frame {self.current_frame_idx + 1}...", fg=DANGER)
        try:
            self.engine.box_cut_mask(self.current_frame_idx, box, sync_to_sam2=False)
            self.engine.sync_frame_mask_to_sam2(self.current_frame_idx, async_exec=True)
            self._cached_mask_key = None
            self._cached_mask = None
            self._composite_cache_key = None
            self._set_frame(self.current_frame_idx)
            self.status_lbl.configure(text=f"Box Cut applied on frame {self.current_frame_idx + 1}", fg=SUCCESS)
        except Exception as e:
            self.status_lbl.configure(text=f"Box Cut Error: {e}", fg=DANGER)

    def _undo_last_action(self):
        res = self.engine.undo(self.current_frame_idx)
        self._cached_mask_key = None
        self._cached_mask = None
        self._composite_cache_key = None
        self._set_frame(self.current_frame_idx)
        self.status_lbl.configure(text=f"Reverted last edit on frame {self.current_frame_idx + 1}", fg=WARNING)

    def _undo_last_prompt(self):
        self._undo_last_action()

    def _clear_current_frame_prompts(self):
        self.engine.clear_frame_prompts(self.current_frame_idx)
        self._cached_mask_key = None
        self._cached_mask = None
        self._composite_cache_key = None
        self._set_frame(self.current_frame_idx)
        self.status_lbl.configure(text=f"Cleared prompts & mask on frame {self.current_frame_idx + 1}", fg=WARNING)

    def _reset_current_frame_conditioning(self):
        if self.engine.num_frames == 0:
            return
        self.engine.reset_frame_conditioning(self.current_frame_idx)
        self._cached_mask_key = None
        self._cached_mask = None
        self._composite_cache_key = None
        self._set_frame(self.current_frame_idx)
        self.status_lbl.configure(text=f"Reset frame {self.current_frame_idx + 1} memory conditioning: ready for clean prompt!", fg=SUCCESS)

    def _clear_all_prompts(self):
        if messagebox.askyesno("Reset All Prompts", "Are you sure you want to clear all prompts and tracked masks across all frames?", parent=self.dialog):
            self.engine.prompts.clear()
            self.engine.tracked_masks.clear()
            self._cached_mask_key = None
            self._cached_mask = None
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
    # Mask Refinement Callbacks (Immediate Slider Sync, Debounced Render & Entry Commits)
    # -------------------------------------------------------------------------

    def _on_refine_slider_moved(self, param_name: str, value: str):
        int_val = int(round(float(value)))
        if param_name == "dilation":
            self.dilation_entry.delete(0, tk.END)
            self.dilation_entry.insert(0, str(int_val))
        elif param_name == "feather":
            self.feather_entry.delete(0, tk.END)
            self.feather_entry.insert(0, str(int_val))
        elif param_name == "threshold":
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, str(int_val))
        elif param_name == "speckle":
            self.speckle_entry.delete(0, tk.END)
            self.speckle_entry.insert(0, str(int_val))

        # Debounce the mask recompute & render (45ms)
        if self._refine_timer is not None:
            self.dialog.after_cancel(self._refine_timer)
        self._refine_timer = self.dialog.after(45, self._debounced_refine_render)

    def _flush_refine_param(self):
        """Immediately flushes debounced parameter changes upon slider mouse release or entry commit."""
        if self._refine_timer is not None:
            self.dialog.after_cancel(self._refine_timer)
            self._refine_timer = None
        self._debounced_refine_render()

    def _debounced_refine_render(self):
        self._refine_timer = None
        self._composite_cache_key = None
        self._render_current_frame()

    def _on_entry_param_commit(self, param_name: str):
        if param_name == "dilation":
            val = self._parse_and_clamp_entry(self.dilation_entry, self.dilation_var.get(), -100, 100)
            self.dilation_var.set(val)
        elif param_name == "feather":
            val = self._parse_and_clamp_entry(self.feather_entry, self.feather_var.get(), 0, 100)
            self.feather_var.set(val)
        elif param_name == "threshold":
            val = self._parse_and_clamp_entry(self.threshold_entry, self.threshold_var.get(), 1, 254)
            self.threshold_var.set(val)
        elif param_name == "speckle":
            val = self._parse_and_clamp_entry(self.speckle_entry, self.min_speckle_var.get(), 0, 100000)
            self.min_speckle_var.set(val)

        self._flush_refine_param()

    def _parse_and_clamp_entry(self, entry: tk.Entry, current_val: int, min_val: int, max_val: int) -> int:
        raw_text = entry.get().strip()
        try:
            val = int(raw_text)
            clamped = max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            clamped = current_val

        entry.delete(0, tk.END)
        entry.insert(0, str(clamped))
        return clamped

    def _on_refine_param_changed(self, *args):
        self._on_refine_slider_moved("dilation", str(self.dilation_var.get()))
        self._on_refine_slider_moved("feather", str(self.feather_var.get()))
        self._on_refine_slider_moved("threshold", str(self.threshold_var.get()))
        self._on_refine_slider_moved("speckle", str(self.min_speckle_var.get()))
        self._flush_refine_param()

    # -------------------------------------------------------------------------
    # Viewport Rendering Engine (Ultra-Fast Visible Viewport Slicing)
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

    def _render_current_frame(self, fast_brush: bool = False):
        if self.engine.num_frames == 0:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        orig_w, orig_h = self.engine.get_frame_dims(self.current_frame_idx)
        if orig_w <= 0 or orig_h <= 0:
            return

        if fast_brush:
            mask = self.engine.tracked_masks.get(self.current_frame_idx, None)
        else:
            # 1. Mask Processing & Cache: Only recompute mask when frame or refine params change
            p_data = self.engine.prompts.get(self.current_frame_idx, {})
            pts_tuple = tuple(p_data.get("points", []))
            lbls_tuple = tuple(p_data.get("labels", []))
            box_tuple = tuple(p_data["box"]) if p_data.get("box") is not None else None
            has_tracked = self.current_frame_idx in self.engine.tracked_masks
            mask_rev = self.engine.get_mask_revision(self.current_frame_idx)

            mask_cache_key = (
                self.current_frame_idx,
                mask_rev,
                self.dilation_var.get(),
                self.feather_var.get(),
                self.threshold_var.get(),
                self.fill_holes_var.get(),
                self.min_speckle_var.get(),
                self.invert_var.get(),
                pts_tuple,
                lbls_tuple,
                box_tuple,
                has_tracked
            )

            if self._cached_mask_key == mask_cache_key and self._cached_mask is not None:
                mask = self._cached_mask
            else:
                mask = self.engine.get_processed_mask(
                    self.current_frame_idx,
                    dilation=self.dilation_var.get(),
                    feather=self.feather_var.get(),
                    threshold=self.threshold_var.get(),
                    fill_holes=self.fill_holes_var.get(),
                    min_speckle=self.min_speckle_var.get(),
                    invert=self.invert_var.get()
                )
                self._cached_mask_key = mask_cache_key
                self._cached_mask = mask

        # 2. Viewport Slicing: Only slice and blit the visible canvas region (Instant sub-10ms blitting)
        ix0 = max(0, int(math.floor(-self.pan_x / self.zoom)))
        iy0 = max(0, int(math.floor(-self.pan_y / self.zoom)))
        ix1 = min(orig_w, int(math.ceil((canvas_w - self.pan_x) / self.zoom)))
        iy1 = min(orig_h, int(math.ceil((canvas_h - self.pan_y) / self.zoom)))

        if ix1 <= ix0 or iy1 <= iy0:
            self.canvas.delete("all")
            return

        disp_w = max(1, int(round((ix1 - ix0) * self.zoom)))
        disp_h = max(1, int(round((iy1 - iy0) * self.zoom)))
        draw_x = self.pan_x + ix0 * self.zoom
        draw_y = self.pan_y + iy0 * self.zoom

        try:
            rgb = self.engine.get_frame_rgb(self.current_frame_idx)
        except Exception:
            return

        if rgb is not None and rgb.shape[:2] != (orig_h, orig_w):
            orig_h, orig_w = rgb.shape[:2]

        if mask is not None and mask.shape[:2] != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        crop_rgb = rgb[iy0:iy1, ix0:ix1]
        mode = self.view_mode.get()

        if mask is None or np.sum(mask) == 0:
            disp_img = cv2.resize(crop_rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        elif mode == "Matte":
            crop_mask = mask[iy0:iy1, ix0:ix1]
            disp_mask = cv2.resize(crop_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST if self.zoom > 1.0 else cv2.INTER_LINEAR)
            disp_img = cv2.cvtColor(disp_mask, cv2.COLOR_GRAY2RGB)
        elif mode == "Cutout":
            crop_mask = mask[iy0:iy1, ix0:ix1]
            disp_rgb = cv2.resize(crop_rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            disp_mask = cv2.resize(crop_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST if self.zoom > 1.0 else cv2.INTER_LINEAR)
            bg = self._get_checkerboard(disp_w, disp_h)
            alpha_f = (disp_mask.astype(np.float32) / 255.0)[..., None]
            disp_img = (disp_rgb.astype(np.float32) * alpha_f + bg.astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
        else:  # Overlay mode
            crop_mask = mask[iy0:iy1, ix0:ix1]
            disp_rgb = cv2.resize(crop_rgb, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            disp_mask = cv2.resize(crop_mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST if self.zoom > 1.0 else cv2.INTER_LINEAR)

            tint_color = MASK_COLORS.get(self.mask_color_name.get(), (239, 68, 68))
            opacity = self.overlay_opacity.get()
            active_pixels = (disp_mask > 0)

            colored_overlay = np.zeros_like(disp_rgb)
            colored_overlay[active_pixels] = tint_color

            disp_img = disp_rgb.copy()
            disp_img[active_pixels] = cv2.addWeighted(
                disp_rgb[active_pixels], 1.0 - opacity,
                colored_overlay[active_pixels], opacity, 0.0
            )
            # Find and draw crisp contours on the visible display resolution
            contours, _ = cv2.findContours(disp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(disp_img, contours, -1, (255, 255, 255), 2)

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
            btn_txt = "🚀 EXPORT REALITYSCAN MASKS & RGBA CUTOUTS\n(HARDWARE ACCELERATED)"
        elif binary_on:
            btn_txt = "🚀 EXPORT REALITYSCAN MASKS\n(BINARY ONLY)"
        elif cutout_on:
            btn_txt = "🚀 EXPORT RGBA CUTOUTS ONLY\n(HARDWARE ACCELERATED)"
        else:
            btn_txt = "🚀 SELECT AN EXPORT TARGET OPTION"

        self.btn_export.configure(text=btn_txt)

    def _browse_output_dir(self):
        f = filedialog.askdirectory(title="Select Mask Export Directory", parent=self.dialog)
        if f:
            self.export_dir_var.set(f)

    def _toggle_export_pause(self):
        if not self.is_export_running:
            return
        self.is_export_paused = not self.is_export_paused
        if self.is_export_paused:
            self.btn_pause_export.configure(text="▶ Resume Export", bg=SUCCESS)
            self.export_status_lbl.configure(text="Export paused by user. Click Resume to continue.", fg=WARNING)
        else:
            self.btn_pause_export.configure(text="⏸ Pause Export", bg=WARNING)
            self.export_status_lbl.configure(text="Resuming export...", fg=ACCENT)

    def _cancel_export_action(self):
        if not self.is_export_running:
            return
        self.export_cancel_requested = True
        self.btn_stop_export.configure(text="Stopping...", state=tk.DISABLED)
        self.export_status_lbl.configure(text="Cancelling export, finishing pending disk writes...", fg=DANGER)

    def _restore_export_ui(self):
        self.is_export_running = False
        self.is_export_paused = False
        self.export_cancel_requested = False
        self.export_active_bar.pack_forget()
        self.btn_export.pack(fill=tk.X)
        self.btn_export.configure(state=tk.NORMAL)
        self.btn_stop_export.configure(state=tk.NORMAL, text="⏹ Stop / Cancel")

    def _export_masks_action(self):
        out_dir = self.export_dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Missing Directory", "Please select an export destination folder.", parent=self.dialog)
            return

        if not self.export_binary_var.get() and not self.export_cutout_var.get():
            messagebox.showwarning("No Format Selected", "Please enable either Binary Masks or Alpha Cutouts to export.", parent=self.dialog)
            return

        self.is_export_running = True
        self.is_export_paused = False
        self.export_cancel_requested = False

        self.btn_export.pack_forget()
        self.export_active_bar.pack(fill=tk.X)
        self.btn_pause_export.configure(text="⏸ Pause Export", bg=WARNING)
        self.btn_stop_export.configure(state=tk.NORMAL, text="⏹ Stop / Cancel")

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
                    progress_callback=_prog,
                    cancel_token=lambda: self.export_cancel_requested,
                    pause_token=lambda: self.is_export_paused
                )
                self.dialog.after(0, lambda: self._on_export_finished(out_dir, counts))
            except Exception as e:
                self.dialog.after(0, lambda: messagebox.showerror("Export Failed", str(e), parent=self.dialog))
                self.dialog.after(0, self._restore_export_ui)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_export_ui(self, pct: int, msg: str):
        self.export_progress["value"] = pct
        self.export_status_lbl.configure(text=msg)

    def _on_export_finished(self, out_dir: str, counts: Dict[str, Any]):
        self._restore_export_ui()
        if counts.get("cancelled", False):
            self.export_status_lbl.configure(
                text=f"Export cancelled: {counts.get('binary', 0)} masks saved before stop.",
                fg=WARNING
            )
            messagebox.showinfo(
                "Export Cancelled",
                f"Export process was stopped by user.\n\n"
                f"• Processed masks: {counts.get('binary', 0)}\n"
                f"• Processed cutouts: {counts.get('cutout', 0)}\n"
                f"Output location:\n{out_dir}",
                parent=self.dialog
            )
            return

        self._on_export_success(out_dir, counts)

    def _on_export_success(self, out_dir: str, counts: Dict[str, Any]):
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
        self.export_cancel_requested = True
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
