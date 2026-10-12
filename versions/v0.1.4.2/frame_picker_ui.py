"""
Abu Align & Train - Frame Picker Studio GUI
===========================================
Ultra-fast, hardware-accelerated frame decimation and cadence picker studio.
Provides live real-time preview of picked vs skipped frames, visual filmstrip timeline,
natural sequence sorting, multi-threaded parallel Copy/Cut transfers, and one-click undo.
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable, Set

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from frame_picker_engine import (
    FramePickerEngine,
    FrameItem,
    FramePickerStats,
    FramePickerManifest,
    SUPPORTED_IMAGE_EXTENSIONS
)

# Theme Colors (Modern Studio Dark Theme matching Abu Align)
BG_DARK = "#0d1117"        # Darkest navy
SURFACE = "#161b22"        # Panel background
SURFACE_LIGHT = "#21262d"  # Highlight background
FIELD = "#0d1117"          # Input field background
BORDER = "#30363d"         # Subtle borders
BORDER_ACTIVE = "#70dbc0"  # Teal highlight
TEXT = "#f0f6fc"           # Crisp white text
TEXT_MUTED = "#8b949e"     # Secondary gray text
ACCENT = "#70dbc0"         # Cyan/Teal accent
ACCENT_BLUE = "#38bdf8"    # Sky blue
SUCCESS = "#10b981"        # Emerald green
WARNING = "#f59e0b"        # Amber
DANGER = "#ef4444"         # Crimson red


def format_size(bytes_val: int) -> str:
    """Formats bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"


class FramePickerStudioDialog:
    """
    Interactive Studio Dialog for Frame Picker in Abu Align.
    Supports sub-millisecond cadence changes, visual filmstrip timeline,
    natural sequence ordering, parallel copy/cut transfers, and one-click undo.
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
        self.dialog.title("🎞️ Frame Picker Studio · Light-Speed Frame Cadence Decimator")
        self.dialog.geometry("1240x840")
        self.dialog.minsize(980, 660)
        self.dialog.configure(bg=BG_DARK)

        # Center dialog
        sw = self.dialog.winfo_screenwidth()
        sh = self.dialog.winfo_screenheight()
        x = max(10, (sw - 1240) // 2)
        y = max(10, (sh - 840) // 2)
        self.dialog.geometry(f"+{x}+{y}")

        # Core Engine
        self.engine = FramePickerEngine()

        # State Variables
        self.folder_var = tk.StringVar(value=initial_folder or "")
        self.dest_folder_var = tk.StringVar(value="")
        self.cadence_var = tk.IntVar(value=5)      # Stride N (e.g. 5 means every 5th frame)
        self.offset_var = tk.IntVar(value=1)       # 1-based start frame offset
        self.invert_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=False)
        self.sort_by_var = tk.StringVar(value="natural")
        self.range_start_var = tk.StringVar(value="")
        self.range_end_var = tk.StringVar(value="")
        self.filter_mode_var = tk.StringVar(value="all")  # "all", "picked", "skipped"
        self.search_filter_var = tk.StringVar(value="")

        # Action: "copy" or "move"
        self.action_var = tk.StringVar(value="copy")

        self.is_scanning: bool = False
        self.is_transferring: bool = False
        self.selected_item: Optional[FrameItem] = None
        self._thumb_img_ref = None  # Prevent GC for Tkinter PhotoImage

        if initial_folder and os.path.isdir(initial_folder):
            self.dest_folder_var.set(os.path.join(initial_folder, "picked_cadence_5"))

        self._build_ui()

        # Auto-scan if folder is valid
        if initial_folder and os.path.isdir(initial_folder):
            self.dialog.after(100, self._start_folder_scan)

    def _log(self, msg: str):
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass
        print(f"[FramePicker] {msg}")

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.dialog, bg=SURFACE, padx=18, pady=12, bd=0, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🎞️ Frame Picker Studio  ·  Light-Speed Cadence Decimator",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT,
            bg=SURFACE
        )
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(
            header,
            text="⚡ Pick every N-th image sequence frame · Instant live preview & parallel I/O",
            font=("Segoe UI", 9),
            fg=ACCENT,
            bg=SURFACE
        )
        sub_lbl.pack(side=tk.RIGHT, padx=10)

        # Main Container
        main_box = tk.Frame(self.dialog, bg=BG_DARK, padx=14, pady=10)
        main_box.pack(fill=tk.BOTH, expand=True)

        # 2. Configuration Panel (Top Card)
        config_card = tk.LabelFrame(
            main_box,
            text="  Configuration: Image Folder, Sampling Cadence & Decimation Settings  ",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT,
            bg=SURFACE,
            padx=12,
            pady=8,
            bd=1,
            relief=tk.SOLID
        )
        config_card.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Source Folder
        r1 = tk.Frame(config_card, bg=SURFACE)
        r1.pack(fill=tk.X, pady=3)

        tk.Label(r1, text="Source Folder:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=16, anchor="w").pack(side=tk.LEFT)
        e_fld = tk.Entry(r1, textvariable=self.folder_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        e_fld.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)
        self.folder_var.trace_add("write", lambda *_: self._on_source_folder_changed())

        tk.Button(r1, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_source).pack(side=tk.LEFT, padx=(0, 6))
        tk.Checkbutton(r1, text="Subfolders", variable=self.recursive_var, bg=SURFACE, fg=TEXT_MUTED, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT, command=self._start_folder_scan).pack(side=tk.LEFT)

        # Row 2: Destination Folder
        r2 = tk.Frame(config_card, bg=SURFACE)
        r2.pack(fill=tk.X, pady=3)

        tk.Label(r2, text="Destination Folder:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=16, anchor="w").pack(side=tk.LEFT)
        e_dst = tk.Entry(r2, textvariable=self.dest_folder_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        e_dst.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)
        tk.Button(r2, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_dest).pack(side=tk.LEFT)

        # Row 3: Cadence & Decimation Controls
        r3 = tk.Frame(config_card, bg=SURFACE)
        r3.pack(fill=tk.X, pady=(6, 2))

        tk.Label(r3, text="Pick Every N Frames:", font=("Segoe UI", 9, "bold"), fg=ACCENT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)

        self.slider_cadence = tk.Scale(
            r3, from_=1, to=60, orient=tk.HORIZONTAL, variable=self.cadence_var,
            bg=SURFACE, fg=TEXT, troughcolor=FIELD, highlightthickness=0,
            activebackground=ACCENT, font=("Segoe UI", 8), length=220,
            command=lambda val: self._on_cadence_changed()
        )
        self.slider_cadence.pack(side=tk.LEFT, padx=(0, 8))

        sp_cad = tk.Spinbox(
            r3, textvariable=self.cadence_var, from_=1, to=9999, width=5,
            bg=FIELD, fg=TEXT, insertbackground=TEXT, font=("Segoe UI", 9, "bold"),
            command=self._on_cadence_changed
        )
        sp_cad.pack(side=tk.LEFT, padx=(0, 14))
        sp_cad.bind("<KeyRelease>", lambda e: self._on_cadence_changed())

        # Preset Quick Buttons
        presets_box = tk.Frame(r3, bg=SURFACE)
        presets_box.pack(side=tk.LEFT, padx=(0, 14))

        def _make_preset_btn(text, n_val):
            return tk.Button(
                presets_box, text=text, font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT,
                relief=tk.FLAT, padx=6, pady=2,
                command=lambda: self._set_cadence(n_val)
            )

        _make_preset_btn("Every 2nd (50%)", 2).pack(side=tk.LEFT, padx=2)
        _make_preset_btn("Every 3rd (33%)", 3).pack(side=tk.LEFT, padx=2)
        _make_preset_btn("Every 5th (20%)", 5).pack(side=tk.LEFT, padx=2)
        _make_preset_btn("Every 10th (10%)", 10).pack(side=tk.LEFT, padx=2)
        _make_preset_btn("Every 30th (3.3%)", 30).pack(side=tk.LEFT, padx=2)

        # Start Offset / Phase
        tk.Label(r3, text="Start Offset:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(10, 4))
        sp_off = tk.Spinbox(
            r3, textvariable=self.offset_var, from_=1, to=9999, width=4,
            bg=FIELD, fg=TEXT, insertbackground=TEXT, font=("Segoe UI", 9),
            command=self._on_cadence_changed
        )
        sp_off.pack(side=tk.LEFT, padx=(0, 10))
        sp_off.bind("<KeyRelease>", lambda e: self._on_cadence_changed())

        # Invert Selection
        tk.Checkbutton(
            r3, text="🔄 Invert (Drop every Nth, keep rest)", variable=self.invert_var,
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT,
            font=("Segoe UI", 9), command=self._on_cadence_changed
        ).pack(side=tk.LEFT, padx=10)

        # 3. Middle Section: Panes (Left: Visual Timeline + Table, Right: Inspector Card)
        middle_panes = tk.PanedWindow(main_box, orient=tk.HORIZONTAL, bg=BG_DARK, bd=0, sashwidth=4, sashrelief=tk.FLAT)
        middle_panes.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # --- Left Panel: Visual Filmstrip Timeline & Selection Table ---
        left_frame = tk.Frame(middle_panes, bg=SURFACE, bd=1, relief=tk.SOLID, highlightbackground=BORDER)
        middle_panes.add(left_frame, minsize=580, stretch="always")

        # Visual Filmstrip Timeline Canvas Card
        timeline_bar = tk.Frame(left_frame, bg=SURFACE_LIGHT, padx=12, pady=6)
        timeline_bar.pack(fill=tk.X)

        tk.Label(
            timeline_bar, text="🎞️ Cadence Sequence Timeline (Filmstrip):",
            font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=SURFACE_LIGHT
        ).pack(side=tk.LEFT)

        self.lbl_timeline_summary = tk.Label(
            timeline_bar, text="No frames loaded",
            font=("Segoe UI", 8), fg=ACCENT, bg=SURFACE_LIGHT
        )
        self.lbl_timeline_summary.pack(side=tk.RIGHT)

        self.timeline_canvas = tk.Canvas(left_frame, bg="#10141a", height=24, highlightthickness=0)
        self.timeline_canvas.pack(fill=tk.X, padx=10, pady=(4, 6))

        # Table Control Bar
        table_top_bar = tk.Frame(left_frame, bg=SURFACE, padx=10, pady=4)
        table_top_bar.pack(fill=tk.X)

        # Filter mode radio / toggles
        tk.Label(table_top_bar, text="View:", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        for mode_key, mode_title in [("all", "All Frames"), ("picked", "✅ Picked Only"), ("skipped", "⏭️ Skipped Only")]:
            tk.Radiobutton(
                table_top_bar, text=mode_title, variable=self.filter_mode_var, value=mode_key,
                bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT,
                font=("Segoe UI", 8), command=self._render_table
            ).pack(side=tk.LEFT, padx=3)

        # Search box
        search_box = tk.Frame(table_top_bar, bg=SURFACE)
        search_box.pack(side=tk.RIGHT)
        tk.Label(search_box, text="🔍 Search:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        e_filt = tk.Entry(search_box, textvariable=self.search_filter_var, font=("Segoe UI", 8), bg=FIELD, fg=TEXT, insertbackground=TEXT, width=18, bd=1, relief=tk.SOLID)
        e_filt.pack(side=tk.LEFT, ipady=2)
        self.search_filter_var.trace_add("write", lambda *_: self._render_table())

        # Treeview Table
        columns = ("seq", "status", "filename", "size", "phase")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("seq", text="# Index", anchor="w")
        self.tree.heading("status", text="Cadence Status", anchor="w")
        self.tree.heading("filename", text="File Name", anchor="w")
        self.tree.heading("size", text="File Size", anchor="e")
        self.tree.heading("phase", text="Cadence Phase", anchor="center")

        self.tree.column("seq", width=65, stretch=False, anchor="w")
        self.tree.column("status", width=120, stretch=False, anchor="w")
        self.tree.column("filename", width=260, stretch=True, anchor="w")
        self.tree.column("size", width=90, stretch=False, anchor="e")
        self.tree.column("phase", width=95, stretch=False, anchor="center")

        tree_scroll = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))

        # Tag Styles for Treeview
        self.tree.tag_configure("picked", foreground=SUCCESS, font=("Segoe UI", 9, "bold"))
        self.tree.tag_configure("skipped", foreground=TEXT_MUTED, font=("Segoe UI", 9))

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", lambda e: self._open_selected_in_viewer())

        # --- Right Panel: Frame Inspector & Fast Thumbnail ---
        right_frame = tk.Frame(middle_panes, bg=SURFACE, bd=1, relief=tk.SOLID, highlightbackground=BORDER, padx=12, pady=10)
        middle_panes.add(right_frame, minsize=320, stretch="never")

        tk.Label(right_frame, text="🔍 Frame Inspector", font=("Segoe UI", 11, "bold"), fg=TEXT, bg=SURFACE).pack(anchor="w", pady=(0, 8))

        # Thumbnail Box
        self.thumb_container = tk.Frame(right_frame, bg=FIELD, bd=1, relief=tk.SOLID, highlightbackground=BORDER, height=220)
        self.thumb_container.pack(fill=tk.X, pady=(0, 10))
        self.thumb_container.pack_propagate(False)

        self.lbl_thumbnail = tk.Label(self.thumb_container, bg=FIELD, fg=TEXT_MUTED, text="Select a frame\nto inspect", font=("Segoe UI", 9))
        self.lbl_thumbnail.pack(expand=True)

        # Status Badge
        self.badge_status = tk.Label(right_frame, text="No Frame Selected", font=("Segoe UI", 10, "bold"), fg=TEXT_MUTED, bg=SURFACE_LIGHT, padx=8, pady=4)
        self.badge_status.pack(fill=tk.X, pady=(0, 8))

        # Metadata Card
        meta_box = tk.Frame(right_frame, bg=SURFACE)
        meta_box.pack(fill=tk.X, pady=(0, 12))

        def _make_meta_row(label_text):
            r = tk.Frame(meta_box, bg=SURFACE)
            r.pack(fill=tk.X, pady=2)
            tk.Label(r, text=label_text, font=("Segoe UI", 8), fg=TEXT_MUTED, bg=SURFACE, width=14, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(r, text="--", font=("Segoe UI", 8, "bold"), fg=TEXT, bg=SURFACE, anchor="w")
            val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            return val_lbl

        self.lbl_meta_fname = _make_meta_row("File Name:")
        self.lbl_meta_seq = _make_meta_row("Sequence #:")
        self.lbl_meta_size = _make_meta_row("File Size:")
        self.lbl_meta_phase = _make_meta_row("Cadence Cycle:")
        self.lbl_meta_path = _make_meta_row("Relative Path:")

        # Quick Viewer Buttons
        btn_box = tk.Frame(right_frame, bg=SURFACE)
        btn_box.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            btn_box, text="👁️ View Full Size", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT,
            relief=tk.FLAT, padx=8, pady=4, command=self._open_selected_in_viewer
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_box, text="📂 Reveal in Explorer", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT,
            relief=tk.FLAT, padx=8, pady=4, command=self._reveal_selected_in_explorer
        ).pack(side=tk.LEFT)

        # 4. Bottom Action & Telemetry Bar
        action_bar = tk.Frame(self.dialog, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=10)
        action_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Top row: Statistics Telemetry Badge
        self.lbl_telemetry = tk.Label(
            action_bar,
            text="📊 Total: 0 frames | ✅ Picked: 0 (0.0%) | ⏭️ Skipped: 0 (0.0%)",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT,
            bg=SURFACE
        )
        self.lbl_telemetry.pack(anchor="w", pady=(0, 8))

        # Bottom row: Transfer Action Controls
        act_row = tk.Frame(action_bar, bg=SURFACE)
        act_row.pack(fill=tk.X)

        # Operation Mode Radios (Copy vs Cut)
        tk.Label(act_row, text="Action:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 6))
        tk.Radiobutton(
            act_row, text="📋 Copy Picked Frames", variable=self.action_var, value="copy",
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT,
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Radiobutton(
            act_row, text="✂️ Cut (Move) Picked Frames", variable=self.action_var, value="move",
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT,
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=(0, 16))

        # Execution Button
        self.btn_execute = tk.Button(
            act_row, text="🚀 Execute Frame Pick Transfer",
            font=("Segoe UI", 10, "bold"), bg="#0284c7", fg="#ffffff", activebackground="#0369a1",
            relief=tk.FLAT, padx=18, pady=6, cursor="hand2", command=self._execute_transfer
        )
        self.btn_execute.pack(side=tk.LEFT, padx=(0, 10))

        # Undo Button
        self.btn_undo = tk.Button(
            act_row, text="↩ Undo Last Move",
            font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT_MUTED, relief=tk.FLAT,
            padx=12, pady=6, state=tk.DISABLED, cursor="hand2", command=self._execute_undo
        )
        self.btn_undo.pack(side=tk.LEFT)

        # Progress bar
        self.progress_bar = ttk.Progressbar(act_row, orient="horizontal", mode="determinate", length=220)
        self.progress_bar.pack(side=tk.RIGHT, padx=(8, 0))

        self.lbl_status = tk.Label(act_row, text="Ready", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=SURFACE)
        self.lbl_status.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # Event Handlers & Dynamic Recalculation
    # -------------------------------------------------------------------------

    def _browse_source(self):
        f = filedialog.askdirectory(title="Select Source Folder with Image Sequence", parent=self.dialog)
        if f:
            self.folder_var.set(f)
            self._on_source_folder_changed()

    def _browse_dest(self):
        f = filedialog.askdirectory(title="Select Destination Folder for Picked Frames", parent=self.dialog)
        if f:
            self.dest_folder_var.set(f)

    def _on_source_folder_changed(self):
        src = self.folder_var.get().strip()
        if src and os.path.isdir(src):
            cad = self.cadence_var.get()
            self.dest_folder_var.set(os.path.join(src, f"picked_cadence_{cad}"))
            self._start_folder_scan()

    def _set_cadence(self, n_val: int):
        self.cadence_var.set(n_val)
        self._on_cadence_changed()

    def _on_cadence_changed(self):
        """Zero-latency (0.0 ms) recalculation of selection without re-reading files."""
        if not self.engine.raw_frames:
            return

        cad = max(1, self.cadence_var.get())
        off = max(1, self.offset_var.get())
        inv = self.invert_var.get()

        # Update destination folder suggestion if still following standard format
        src = self.folder_var.get().strip()
        curr_dst = self.dest_folder_var.get().strip()
        if src and ("picked_cadence_" in curr_dst or not curr_dst):
            inv_suffix = "_inverted" if inv else ""
            self.dest_folder_var.set(os.path.join(src, f"picked_cadence_{cad}{inv_suffix}"))

        self.engine.compute_selection(cadence=cad, offset=off, invert=inv)
        self._render_timeline()
        self._render_table()
        self._update_telemetry()

    def _start_folder_scan(self):
        """Scans folder in background thread."""
        src = self.folder_var.get().strip()
        if not src or not os.path.isdir(src) or self.is_scanning:
            return

        self.is_scanning = True
        self.lbl_status.config(text="Scanning folder and sorting sequence...", fg=ACCENT)
        self.btn_execute.config(state=tk.DISABLED)

        def worker():
            t0 = time.time()
            try:
                items = self.engine.scan_folder(
                    folder_path=src,
                    recursive=self.recursive_var.get(),
                    sort_by=self.sort_by_var.get()
                )
                dur = time.time() - t0
                cad = max(1, self.cadence_var.get())
                off = max(1, self.offset_var.get())
                inv = self.invert_var.get()
                self.engine.compute_selection(cadence=cad, offset=off, invert=inv)

                self.dialog.after(0, lambda: self._on_scan_completed(len(items), dur))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_scan_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_completed(self, total_count: int, duration_sec: float):
        self.is_scanning = False
        self.btn_execute.config(state=tk.NORMAL if total_count > 0 else tk.DISABLED)
        self.lbl_status.config(text=f"Loaded {total_count:,} frames in {duration_sec * 1000:.1f}ms", fg=SUCCESS)
        self._render_timeline()
        self._render_table()
        self._update_telemetry()

        # Select first item if available
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._on_tree_select(None)

    def _on_scan_error(self, err_msg: str):
        self.is_scanning = False
        self.lbl_status.config(text=f"Scan error: {err_msg}", fg=DANGER)
        messagebox.showerror("Scan Error", f"Failed to scan folder:\n{err_msg}", parent=self.dialog)

    # -------------------------------------------------------------------------
    # Visual Timeline Filmstrip Rendering
    # -------------------------------------------------------------------------

    def _render_timeline(self):
        """Draws sequence distribution ribbon on the timeline canvas."""
        canvas = self.timeline_canvas
        canvas.delete("all")

        frames = self.engine.raw_frames
        total = len(frames)
        if total == 0:
            self.lbl_timeline_summary.config(text="No frames loaded")
            return

        w = canvas.winfo_width() or 800
        h = 24

        stats = self.engine.cached_stats
        inv_str = " (Inverted)" if self.invert_var.get() else ""
        self.lbl_timeline_summary.config(
            text=f"Every {stats.cadence} frames{inv_str} · Picked {stats.picked_frames}/{total} ({stats.decimation_ratio_pct:.1f}%)"
        )

        step_px = max(1.0, w / total)
        for idx, item in enumerate(frames):
            x1 = int(idx * step_px)
            x2 = max(x1 + 1, int((idx + 1) * step_px))
            fill_col = SUCCESS if item.selected else "#1e293b"
            canvas.create_rectangle(x1, 0, x2, h, fill=fill_col, outline="")

    # -------------------------------------------------------------------------
    # Table Rendering & Telemetry
    # -------------------------------------------------------------------------

    def _render_table(self):
        """Populates Treeview with current filtered frames."""
        self.tree.delete(*self.tree.get_children())
        frames = self.engine.raw_frames
        if not frames:
            return

        filter_mode = self.filter_mode_var.get()
        query = self.search_filter_var.get().strip().lower()

        for item in frames:
            if filter_mode == "picked" and not item.selected:
                continue
            if filter_mode == "skipped" and item.selected:
                continue
            if query and (query not in item.filename.lower()):
                continue

            status_text = "✅ PICKED" if item.selected else "⏭️ SKIPPED"
            tag = "picked" if item.selected else "skipped"
            phase_str = f"Phase {item.phase_offset}" if item.phase_offset >= 0 else "--"

            self.tree.insert(
                "",
                tk.END,
                iid=str(item.sequence_index),
                values=(
                    f"#{item.sequence_index}",
                    status_text,
                    item.filename,
                    format_size(item.size_bytes),
                    phase_str
                ),
                tags=(tag,)
            )

    def _update_telemetry(self):
        stats = self.engine.cached_stats
        self.lbl_telemetry.config(
            text=(
                f"📊 Total: {stats.total_frames:,} frames ({format_size(stats.picked_size_bytes + stats.skipped_size_bytes)})  |  "
                f"✅ Picked: {stats.picked_frames:,} ({format_size(stats.picked_size_bytes)}, {stats.decimation_ratio_pct:.1f}%)  |  "
                f"⏭️ Skipped: {stats.skipped_frames:,} ({format_size(stats.skipped_size_bytes)})"
            )
        )

    # -------------------------------------------------------------------------
    # Frame Selection & Thumbnail Inspector
    # -------------------------------------------------------------------------

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        seq_idx = int(sel[0])
        # Find item
        item = next((f for f in self.engine.raw_frames if f.sequence_index == seq_idx), None)
        if not item:
            return

        self.selected_item = item
        self.lbl_meta_fname.config(text=item.filename)
        self.lbl_meta_seq.config(text=f"#{item.sequence_index} of {len(self.engine.raw_frames):,}")
        self.lbl_meta_size.config(text=format_size(item.size_bytes))
        self.lbl_meta_phase.config(text=f"Offset {item.phase_offset} (Every {self.cadence_var.get()})")
        self.lbl_meta_path.config(text=item.relative_path or item.filename)

        if item.selected:
            self.badge_status.config(text="✅ PICKED FOR TRANSFER", fg=SUCCESS, bg="#064e3b")
        else:
            self.badge_status.config(text="⏭️ SKIPPED (DECIMATED)", fg=TEXT_MUTED, bg="#1e293b")

        # Load thumbnail in background to keep GUI buttery smooth
        threading.Thread(target=self._load_thumbnail_async, args=(item.full_path,), daemon=True).start()

    def _load_thumbnail_async(self, fpath: str):
        try:
            from PIL import ImageTk
            pil_thumb = self.engine.get_thumbnail(fpath, max_dim=210)
            photo = ImageTk.PhotoImage(pil_thumb)
            self._thumb_img_ref = photo
            self.dialog.after(0, lambda: self.lbl_thumbnail.config(image=photo, text=""))
        except Exception:
            self.dialog.after(0, lambda: self.lbl_thumbnail.config(image="", text="Preview unavailable"))

    def _open_selected_in_viewer(self):
        if self.selected_item and os.path.isfile(self.selected_item.full_path):
            try:
                os.startfile(self.selected_item.full_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image:\n{e}", parent=self.dialog)

    def _reveal_selected_in_explorer(self):
        if self.selected_item and os.path.isfile(self.selected_item.full_path):
            try:
                subprocess.Popen(f'explorer /select,"{os.path.normpath(self.selected_item.full_path)}"')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reveal in explorer:\n{e}", parent=self.dialog)

    # -------------------------------------------------------------------------
    # Transfer Execution & Undo Restoration
    # -------------------------------------------------------------------------

    def _execute_transfer(self):
        if self.is_transferring:
            return

        picked = [it for it in self.engine.raw_frames if it.selected]
        if not picked:
            messagebox.showwarning("No Frames Picked", "No frames are currently picked for transfer.\nAdjust the cadence slider.", parent=self.dialog)
            return

        dest_folder = self.dest_folder_var.get().strip()
        if not dest_folder:
            messagebox.showwarning("Destination Missing", "Please select a Destination Folder for the picked frames.", parent=self.dialog)
            return

        action = self.action_var.get()
        action_title = "Copy" if action == "copy" else "Cut (Move)"
        
        # Confirm for move
        if action == "move":
            confirm = messagebox.askyesno(
                "Confirm Cut & Move",
                f"You are about to CUT (MOVE) {len(picked):,} picked frames into:\n{dest_folder}\n\n"
                f"This will remove them from the source folder.\n"
                f"(You can use 'Undo Last Move' to restore them if needed).\n\nProceed?",
                parent=self.dialog
            )
            if not confirm:
                return

        self.is_transferring = True
        self.btn_execute.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = len(picked)
        self.lbl_status.config(text=f"Starting parallel {action_title}...", fg=ACCENT)

        def progress_cb(cur, tot, fname):
            self.dialog.after(0, lambda: self._update_transfer_progress(cur, tot, fname))

        def worker():
            t0 = time.time()
            ok_cnt, err_cnt, manifest = self.engine.execute_transfer(
                items_to_transfer=picked,
                destination_folder=dest_folder,
                action=action,
                preserve_subdirs=True,
                progress_callback=progress_cb
            )
            elapsed = time.time() - t0
            self.dialog.after(0, lambda: self._on_transfer_finished(ok_cnt, err_cnt, elapsed, action, dest_folder))

        threading.Thread(target=worker, daemon=True).start()

    def _update_transfer_progress(self, current: int, total: int, fname: str):
        self.progress_bar["value"] = current
        pct = (current / total * 100.0) if total > 0 else 0.0
        self.lbl_status.config(text=f"Transferring {current}/{total} ({pct:.0f}%): {fname[:24]}")

    def _on_transfer_finished(self, ok_cnt: int, err_cnt: int, elapsed_sec: float, action: str, dest_folder: str):
        self.is_transferring = False
        self.btn_execute.config(state=tk.NORMAL)
        self.progress_bar["value"] = self.progress_bar["maximum"]

        if action == "move" and self.engine.last_manifest:
            self.btn_undo.config(state=tk.NORMAL)

        rate_msg = f"{ok_cnt / max(0.01, elapsed_sec):.1f} files/sec"
        self.lbl_status.config(text=f"Completed {ok_cnt:,} {action}s in {elapsed_sec:.2f}s ({rate_msg})", fg=SUCCESS)

        msg = f"Successfully {action}ed {ok_cnt:,} frames into:\n{dest_folder}\n\nTime: {elapsed_sec:.2f}s ({rate_msg})"
        if err_cnt > 0:
            msg += f"\n\nWarnings: {err_cnt} files could not be transferred."

        messagebox.showinfo("Transfer Complete", msg, parent=self.dialog)

        # If move was executed, refresh source folder
        if action == "move":
            self._start_folder_scan()

    def _execute_undo(self):
        if not self.engine.last_manifest:
            return

        cnt = len(self.engine.last_manifest.mappings)
        confirm = messagebox.askyesno(
            "Confirm Undo Move",
            f"Restore {cnt:,} files back to their original folder?",
            parent=self.dialog
        )
        if not confirm:
            return

        self.btn_undo.config(state=tk.DISABLED)
        self.lbl_status.config(text="Restoring files...", fg=WARNING)

        def worker():
            t0 = time.time()
            restored, errs = self.engine.undo_last_operation()
            elapsed = time.time() - t0
            self.dialog.after(0, lambda: self._on_undo_finished(restored, errs, elapsed))

        threading.Thread(target=worker, daemon=True).start()

    def _on_undo_finished(self, restored: int, errs: int, elapsed: float):
        self.lbl_status.config(text=f"Restored {restored:,} files in {elapsed:.2f}s", fg=SUCCESS)
        messagebox.showinfo("Undo Complete", f"Restored {restored:,} files back to their original locations.", parent=self.dialog)
        self._start_folder_scan()


def open_frame_picker_dialog(parent: tk.Tk, initial_folder: Optional[str] = None, log_callback: Optional[Callable[[str], None]] = None):
    """Factory helper to open FramePickerStudioDialog."""
    return FramePickerStudioDialog(parent, initial_folder=initial_folder, log_callback=log_callback)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    init_dir = sys.argv[1] if len(sys.argv) > 1 else None
    dlg = FramePickerStudioDialog(root, initial_folder=init_dir)
    dlg.dialog.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
