"""
Abu Align & Train - FileSift Interactive Studio GUI
===================================================
Ultra-fast, hardware-accelerated cross-folder file comparison, stem matching,
and parallel copy/cut curation studio for photogrammetry, 3DGS, and VFX.
"""

import os
import sys
import time
import math
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from file_sift_engine import FileSiftEngine, SiftMatchRecord

# Theme Colors (Dark Modern Studio matching Abu Align)
BG_DARK = "#0d1117"        # Darkest navy
SURFACE = "#161b22"        # Card background
SURFACE_LIGHT = "#21262d"  # Highlight card
FIELD = "#0d1117"          # Input fields
BORDER = "#30363d"         # Subtle borders
BORDER_ACTIVE = "#58a6ff"  # Accent blue
TEXT = "#f0f6fc"           # Crisp white
TEXT_MUTED = "#8b949e"     # Subtle gray
ACCENT = "#58a6ff"         # Electric blue
SUCCESS = "#10b981"        # Emerald green
WARNING = "#f59e0b"        # Amber
DANGER = "#ef4444"         # Crimson red
PURPLE = "#a855f7"         # Violet


class FileSiftStudioDialog:
    """
    Main Studio Dialog for FileSift in Abu Align.
    Supports sub-millisecond matching between source lists/folders and target directories,
    interactive preview, selective checkboxes, multi-threaded copy/cut, and one-click undo.
    """

    def __init__(
        self,
        parent: tk.Tk,
        initial_source: Optional[str] = None,
        initial_target: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs
    ):
        self.parent = parent
        self.log_callback = log_callback
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("📂 FileSift Studio · Ultra-Fast Cross-Folder File Matcher & Sifter")
        self.dialog.geometry("1300x880")
        self.dialog.minsize(1000, 680)
        self.dialog.configure(bg=BG_DARK)

        # Center dialog
        sw = self.dialog.winfo_screenwidth()
        sh = self.dialog.winfo_screenheight()
        x = max(10, (sw - 1300) // 2)
        y = max(10, (sh - 880) // 2)
        self.dialog.geometry(f"+{x}+{y}")

        # Core Engine
        self.engine = FileSiftEngine()

        # State Variables
        self.source_mode_var = tk.StringVar(value="folder")  # "folder", "list_file", "paste"
        self.source_path_var = tk.StringVar(value=initial_source or "")
        self.target_path_var = tk.StringVar(value=initial_target or "")
        self.dest_path_var = tk.StringVar(value="")

        # Matching configuration
        self.match_mode_var = tk.StringVar(value="stem")     # "stem", "exact", "prefix", "suffix"
        self.recursive_var = tk.BooleanVar(value=True)
        self.case_sensitive_var = tk.BooleanVar(value=False)
        self.invert_var = tk.BooleanVar(value=False)
        self.preserve_hierarchy_var = tk.BooleanVar(value=True)
        self.ext_filter_var = tk.StringVar(value="All Files (*.*)")

        # Review & View state
        self.view_tab_var = tk.StringVar(value="matched")    # "matched", "missing", "unmatched_target"
        self.search_filter_var = tk.StringVar(value="")
        self.is_processing: bool = False
        self.cancel_requested: bool = False

        # Auto-configure destination if target is given
        if initial_target and os.path.isdir(initial_target):
            self.dest_path_var.set(os.path.join(initial_target, "sifted_files"))

        self._build_ui()

    def _log(self, msg: str):
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass
        print(f"[FileSift] {msg}")

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.dialog, bg=SURFACE, padx=18, pady=12, bd=0, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="📂 FileSift Studio  ·  High-Speed File Matching & Cross-Folder Sifter",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT,
            bg=SURFACE
        )
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(
            header,
            text="⚡ Sub-millisecond O(1) set matching across folders and file lists",
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
            text="  Configuration: Source Reference, Target Search Folder & Destination  ",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT,
            bg=SURFACE,
            padx=12,
            pady=8,
            bd=1,
            relief=tk.SOLID
        )
        config_card.pack(fill=tk.X, pady=(0, 10))

        # Row A: Source Reference
        r_src = tk.Frame(config_card, bg=SURFACE)
        r_src.pack(fill=tk.X, pady=3)

        tk.Label(r_src, text="1. Source Reference:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)

        # Source mode radios
        rb_f = tk.Radiobutton(r_src, text="Folder", variable=self.source_mode_var, value="folder", bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT, command=self._on_source_mode_changed)
        rb_l = tk.Radiobutton(r_src, text="File List (.txt/.csv/.json)", variable=self.source_mode_var, value="list_file", bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT, command=self._on_source_mode_changed)
        rb_p = tk.Radiobutton(r_src, text="Clipboard / Text Paste", variable=self.source_mode_var, value="paste", bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT, command=self._on_source_mode_changed)
        rb_f.pack(side=tk.LEFT, padx=(0, 6))
        rb_l.pack(side=tk.LEFT, padx=(0, 6))
        rb_p.pack(side=tk.LEFT, padx=(0, 12))

        self.src_entry = tk.Entry(r_src, textvariable=self.source_path_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        self.src_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)

        self.btn_browse_src = tk.Button(r_src, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_source)
        self.btn_browse_src.pack(side=tk.LEFT, padx=(0, 4))

        btn_paste = tk.Button(r_src, text="📋 Paste Text", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=6, command=self._paste_source_from_clipboard)
        btn_paste.pack(side=tk.LEFT)

        # Row B: Target Folder (Folder 2 to search inside)
        r_tgt = tk.Frame(config_card, bg=SURFACE)
        r_tgt.pack(fill=tk.X, pady=3)

        tk.Label(r_tgt, text="2. Target Folder (2nd):", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        e_tgt = tk.Entry(r_tgt, textvariable=self.target_path_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        e_tgt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)

        tk.Button(r_tgt, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_target).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(r_tgt, text="Recursive Subfolders", variable=self.recursive_var, bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT).pack(side=tk.LEFT, padx=(6, 0))

        # Row C: Destination Folder (Folder 3 where matching files go)
        r_dst = tk.Frame(config_card, bg=SURFACE)
        r_dst.pack(fill=tk.X, pady=3)

        tk.Label(r_dst, text="3. Destination (3rd):", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=18, anchor="w").pack(side=tk.LEFT)
        e_dst = tk.Entry(r_dst, textvariable=self.dest_path_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        e_dst.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)

        tk.Button(r_dst, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_dest).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(r_dst, text="⚡ Auto-suggest", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=ACCENT, relief=tk.FLAT, padx=6, command=self._auto_suggest_dest).pack(side=tk.LEFT)

        # Row D: Matching Parameters & Big Sift Button
        r_opts = tk.Frame(config_card, bg=SURFACE)
        r_opts.pack(fill=tk.X, pady=(6, 2))

        tk.Label(r_opts, text="Matching Mode:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 6))
        mode_cb = ttk.Combobox(r_opts, textvariable=self.match_mode_var, values=["stem", "exact", "prefix", "suffix"], state="readonly", width=10, font=("Segoe UI", 9))
        mode_cb.pack(side=tk.LEFT, padx=(0, 10))

        tk.Checkbutton(r_opts, text="Case-Insensitive", variable=self.case_sensitive_var, onvalue=False, offvalue=True, bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(r_opts, text="Invert Selection", variable=self.invert_var, bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT).pack(side=tk.LEFT, padx=(0, 10))
        tk.Checkbutton(r_opts, text="Preserve Subfolder Hierarchy", variable=self.preserve_hierarchy_var, bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT).pack(side=tk.LEFT, padx=(0, 14))

        # Big SIFT Button
        self.btn_sift = tk.Button(
            r_opts,
            text="⚡ SIFT & MATCH FILES",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            cursor="hand2",
            command=self._start_sift_match
        )
        self.btn_sift.pack(side=tk.RIGHT)

        # 3. Middle Section: Telemetry Counters Bar & Results Gallery
        results_card = tk.Frame(main_box, bg=SURFACE, bd=1, relief=tk.SOLID, highlightbackground=BORDER)
        results_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Status and Tabs Bar
        top_res_bar = tk.Frame(results_card, bg=SURFACE_LIGHT, padx=10, pady=8)
        top_res_bar.pack(fill=tk.X)

        self.lbl_stats = tk.Label(
            top_res_bar,
            text="Ready. Select source and target folders, then click '⚡ SIFT & MATCH FILES'.",
            font=("Segoe UI", 9, "bold"),
            fg=TEXT,
            bg=SURFACE_LIGHT
        )
        self.lbl_stats.pack(side=tk.LEFT)

        # Filter tabs
        tab_box = tk.Frame(top_res_bar, bg=SURFACE_LIGHT)
        tab_box.pack(side=tk.RIGHT)

        self.btn_tab_matched = tk.Radiobutton(
            tab_box, text="🟢 Matched in Target (0)", variable=self.view_tab_var, value="matched",
            indicatoron=False, bg=SURFACE, fg=TEXT, selectcolor="#064e3b", activebackground=SURFACE_LIGHT,
            padx=10, pady=4, relief=tk.FLAT, command=self._refresh_table_view
        )
        self.btn_tab_missing = tk.Radiobutton(
            tab_box, text="⚠️ Missing from Target (0)", variable=self.view_tab_var, value="missing",
            indicatoron=False, bg=SURFACE, fg=TEXT, selectcolor="#78350f", activebackground=SURFACE_LIGHT,
            padx=10, pady=4, relief=tk.FLAT, command=self._refresh_table_view
        )
        self.btn_tab_all = tk.Radiobutton(
            tab_box, text="📋 All Target Files (0)", variable=self.view_tab_var, value="unmatched_target",
            indicatoron=False, bg=SURFACE, fg=TEXT, selectcolor="#1e293b", activebackground=SURFACE_LIGHT,
            padx=10, pady=4, relief=tk.FLAT, command=self._refresh_table_view
        )
        self.btn_tab_matched.pack(side=tk.LEFT, padx=2)
        self.btn_tab_missing.pack(side=tk.LEFT, padx=2)
        self.btn_tab_all.pack(side=tk.LEFT, padx=2)

        # Search Bar & Batch Checkbox Controls
        search_bar = tk.Frame(results_card, bg=SURFACE, padx=10, pady=6)
        search_bar.pack(fill=tk.X)

        tk.Label(search_bar, text="🔍 Filter:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 4))
        self.e_search = tk.Entry(search_bar, textvariable=self.search_filter_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, width=30, bd=1, relief=tk.SOLID)
        self.e_search.pack(side=tk.LEFT, padx=(0, 10), ipady=2)
        self.search_filter_var.trace_add("write", lambda *_: self._refresh_table_view())

        tk.Button(search_bar, text="Select All", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=6, command=lambda: self._toggle_all_selection(True)).pack(side=tk.LEFT, padx=2)
        tk.Button(search_bar, text="Deselect All", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=6, command=lambda: self._toggle_all_selection(False)).pack(side=tk.LEFT, padx=2)
        tk.Button(search_bar, text="Invert Checks", font=("Segoe UI", 8), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=6, command=self._invert_checkboxes).pack(side=tk.LEFT, padx=2)

        self.lbl_selected_tally = tk.Label(search_bar, text="Selected: 0 files (0.0 MB)", font=("Segoe UI", 9), fg=ACCENT, bg=SURFACE)
        self.lbl_selected_tally.pack(side=tk.RIGHT)

        # Treeview / Table
        table_frame = tk.Frame(results_card, bg=SURFACE)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "FileSift.Treeview",
            background=FIELD,
            foreground=TEXT,
            fieldbackground=FIELD,
            rowheight=24,
            font=("Segoe UI", 9)
        )
        style.map("FileSift.Treeview", background=[("selected", "#1e3a5f")], foreground=[("selected", "#ffffff")])
        style.configure("FileSift.Treeview.Heading", background=SURFACE_LIGHT, foreground=TEXT, font=("Segoe UI", 9, "bold"))

        cols = ("check", "target_name", "source_key", "size", "rel_path")
        self.tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            style="FileSift.Treeview",
            selectmode="extended"
        )
        self.tree.heading("check", text="Transfer?", anchor="center")
        self.tree.heading("target_name", text="Target Filename (Folder 2)", anchor="w")
        self.tree.heading("source_key", text="Matched Source Key", anchor="w")
        self.tree.heading("size", text="Size", anchor="e")
        self.tree.heading("rel_path", text="Relative Directory Path", anchor="w")

        self.tree.column("check", width=70, anchor="center")
        self.tree.column("target_name", width=340, anchor="w")
        self.tree.column("source_key", width=260, anchor="w")
        self.tree.column("size", width=100, anchor="e")
        self.tree.column("rel_path", width=400, anchor="w")

        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Button-1>", self._on_tree_click)
        self.tree.bind("<space>", self._on_tree_space_toggle)

        # 4. Action Execution & Safety Footer
        footer = tk.Frame(main_box, bg=SURFACE, padx=14, pady=10, bd=1, relief=tk.SOLID, highlightbackground=BORDER)
        footer.pack(fill=tk.X)

        # Progress bar
        self.pbar = ttk.Progressbar(footer, orient=tk.HORIZONTAL, mode="determinate")
        self.pbar.pack(fill=tk.X, pady=(0, 8))

        # Bottom row with action controls
        bot_row = tk.Frame(footer, bg=SURFACE)
        bot_row.pack(fill=tk.X)

        self.lbl_action_status = tk.Label(
            bot_row,
            text="Ready to transfer matched files.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=SURFACE
        )
        self.lbl_action_status.pack(side=tk.LEFT)

        # Action Buttons
        self.btn_undo = tk.Button(
            bot_row,
            text="↩️ Undo Last Transfer",
            font=("Segoe UI", 9),
            bg=SURFACE_LIGHT,
            fg=WARNING,
            activebackground="#451a03",
            activeforeground="#fef08a",
            relief=tk.FLAT,
            padx=12,
            pady=5,
            state=tk.DISABLED,
            command=self._execute_undo
        )
        self.btn_undo.pack(side=tk.RIGHT, padx=(8, 0))

        self.btn_cut = tk.Button(
            bot_row,
            text="✂️ Cut / Move to Destination",
            font=("Segoe UI", 10, "bold"),
            bg="#b91c1c",
            fg="#ffffff",
            activebackground="#991b1b",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED,
            command=lambda: self._execute_transfer_flow("cut")
        )
        self.btn_cut.pack(side=tk.RIGHT, padx=(8, 0))

        self.btn_copy = tk.Button(
            bot_row,
            text="📋 Copy to Destination",
            font=("Segoe UI", 10, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=5,
            cursor="hand2",
            state=tk.DISABLED,
            command=lambda: self._execute_transfer_flow("copy")
        )
        self.btn_copy.pack(side=tk.RIGHT)

    # -------------------------------------------------------------------------
    # Event Handlers & Browsing
    # -------------------------------------------------------------------------

    def _on_source_mode_changed(self):
        mode = self.source_mode_var.get()
        if mode == "paste":
            self.src_entry.config(state=tk.NORMAL)
            self.btn_browse_src.config(state=tk.DISABLED)
        else:
            self.src_entry.config(state=tk.NORMAL)
            self.btn_browse_src.config(state=tk.NORMAL)

    def _browse_source(self):
        mode = self.source_mode_var.get()
        if mode == "folder":
            d = filedialog.askdirectory(title="Select Source Reference Folder", parent=self.dialog)
            if d:
                self.source_path_var.set(d)
        else:
            f = filedialog.askopenfilename(
                title="Select File List (.txt, .csv, .json, .lst)",
                filetypes=[
                    ("Text / List Files", "*.txt *.csv *.json *.lst"),
                    ("All Files", "*.*")
                ],
                parent=self.dialog
            )
            if f:
                self.source_path_var.set(f)

    def _browse_target(self):
        d = filedialog.askdirectory(title="Select Target Search Folder (Folder 2)", parent=self.dialog)
        if d:
            self.target_path_var.set(d)
            if not self.dest_path_var.get().strip():
                self._auto_suggest_dest()

    def _browse_dest(self):
        d = filedialog.askdirectory(title="Select Destination Folder (Folder 3)", parent=self.dialog)
        if d:
            self.dest_path_var.set(d)

    def _auto_suggest_dest(self):
        tgt = self.target_path_var.get().strip()
        if tgt and os.path.isdir(tgt):
            self.dest_path_var.set(os.path.join(tgt, "sifted_files"))

    def _paste_source_from_clipboard(self):
        try:
            cb_text = self.dialog.clipboard_get()
            if cb_text:
                self.source_mode_var.set("paste")
                self.source_path_var.set(cb_text.strip().replace("\n", ", "))
                self._log("Pasted file list from clipboard.")
        except Exception:
            messagebox.showinfo("Clipboard", "Clipboard is empty or does not contain text.", parent=self.dialog)

    # -------------------------------------------------------------------------
    # Matching Logic
    # -------------------------------------------------------------------------

    def _start_sift_match(self):
        src_val = self.source_path_var.get().strip()
        tgt_val = self.target_path_var.get().strip()
        mode = self.source_mode_var.get()

        if not src_val:
            messagebox.showerror("Source Required", "Please specify a Source Folder, File List, or paste filenames.", parent=self.dialog)
            return
        if not tgt_val or not os.path.isdir(tgt_val):
            messagebox.showerror("Target Required", "Please select a valid Target Search Folder (Folder 2).", parent=self.dialog)
            return

        self.btn_sift.config(state=tk.DISABLED, text="⚡ Sifting...")
        self.lbl_stats.config(text="Scanning target files and performing O(1) set matching...", fg=ACCENT)
        self.dialog.update_idletasks()

        def worker():
            try:
                # 1. Ingest source
                if mode == "folder":
                    self.engine.load_source_from_folder(src_val, recursive=self.recursive_var.get())
                elif mode == "list_file":
                    self.engine.load_source_from_list_file(src_val)
                else:
                    self.engine.load_source_from_text(src_val)

                # 2. Scan target
                self.engine.scan_target_folder(tgt_val, recursive=self.recursive_var.get())

                # 3. Match
                m_mode = self.match_mode_var.get()
                case_sens = self.case_sensitive_var.get()
                inv = self.invert_var.get()
                matches, unmatched_src, unmatched_tgt = self.engine.match_files(
                    mode=m_mode,
                    case_sensitive=case_sens,
                    invert_selection=inv
                )

                self.dialog.after(0, lambda: self._on_match_finished(matches, unmatched_src, unmatched_tgt))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_match_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_match_finished(
        self,
        matches: List[SiftMatchRecord],
        unmatched_src: List[str],
        unmatched_tgt: List[Tuple[str, str, int]]
    ):
        self.btn_sift.config(state=tk.NORMAL, text="⚡ SIFT & MATCH FILES")

        # Update tab counters
        self.btn_tab_matched.config(text=f"🟢 Matched in Target ({len(matches):,})")
        self.btn_tab_missing.config(text=f"⚠️ Missing from Target ({len(unmatched_src):,})")
        self.btn_tab_all.config(text=f"📋 All Target Files ({len(self.engine.target_files_cache):,})")

        total_mb = sum(r.file_size_bytes for r in matches) / (1024 * 1024)
        stat_txt = f"Found {len(matches):,} matches ({total_mb:.1f} MB) out of {len(self.engine.source_reference_files):,} source items in {len(self.engine.target_files_cache):,} target files."
        self.lbl_stats.config(text=stat_txt, fg=SUCCESS)

        # Enable action buttons if matches exist
        has_matches = len(matches) > 0
        self.btn_copy.config(state=tk.NORMAL if has_matches else tk.DISABLED)
        self.btn_cut.config(state=tk.NORMAL if has_matches else tk.DISABLED)

        self._refresh_table_view()

    def _on_match_error(self, err_msg: str):
        self.btn_sift.config(state=tk.NORMAL, text="⚡ SIFT & MATCH FILES")
        self.lbl_stats.config(text=f"Error: {err_msg}", fg=DANGER)
        messagebox.showerror("Sift Error", f"Failed to match files:\n{err_msg}", parent=self.dialog)

    # -------------------------------------------------------------------------
    # Results Table Rendering
    # -------------------------------------------------------------------------

    def _refresh_table_view(self):
        # Clear table
        for item in self.tree.get_children():
            self.tree.delete(item)

        tab = self.view_tab_var.get()
        search_query = self.search_filter_var.get().strip().lower()

        if tab == "matched":
            items = self.engine.cached_matches
            for idx, r in enumerate(items):
                if search_query and (search_query not in r.target_filename.lower() and search_query not in r.source_key.lower()):
                    continue
                sz_str = f"{r.file_size_bytes / (1024 * 1024):.1f} MB" if r.file_size_bytes > (1024 * 1024) else f"{r.file_size_bytes / 1024:.0f} KB"
                chk_icon = "☑️" if r.selected else "⬜"
                self.tree.insert(
                    "",
                    tk.END,
                    iid=f"match_{idx}",
                    values=(chk_icon, r.target_filename, r.source_key, sz_str, r.relative_path),
                    tags=("matched",)
                )
        elif tab == "missing":
            items_missing = self.engine.cached_unmatched_source
            for idx, name in enumerate(items_missing):
                if search_query and search_query not in name.lower():
                    continue
                self.tree.insert(
                    "",
                    tk.END,
                    iid=f"miss_{idx}",
                    values=("-", "[NOT FOUND IN TARGET]", name, "-", "-"),
                    tags=("missing",)
                )
        else:
            items_target = self.engine.target_files_cache
            for idx, (full_p, rel_p, sz) in enumerate(items_target):
                fname = os.path.basename(full_p)
                if search_query and search_query not in fname.lower():
                    continue
                sz_str = f"{sz / (1024 * 1024):.1f} MB" if sz > (1024 * 1024) else f"{sz / 1024:.0f} KB"
                self.tree.insert(
                    "",
                    tk.END,
                    iid=f"tgt_{idx}",
                    values=("-", fname, "-", sz_str, rel_p),
                    tags=("target",)
                )

        self._update_selected_tally()

    def _on_tree_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree.identify_column(event.x)
        item_id = self.tree.identify_row(event.y)
        if not item_id or not item_id.startswith("match_"):
            return

        idx = int(item_id.split("_")[1])
        if idx < len(self.engine.cached_matches):
            record = self.engine.cached_matches[idx]
            # Toggle selection if first column clicked or on row toggle
            if col == "#1":
                record.selected = not record.selected
                chk_icon = "☑️" if record.selected else "⬜"
                vals = list(self.tree.item(item_id, "values"))
                vals[0] = chk_icon
                self.tree.item(item_id, values=vals)
                self._update_selected_tally()

    def _on_tree_space_toggle(self, event):
        selected_iids = self.tree.selection()
        for iid in selected_iids:
            if iid.startswith("match_"):
                idx = int(iid.split("_")[1])
                record = self.engine.cached_matches[idx]
                record.selected = not record.selected
                chk_icon = "☑️" if record.selected else "⬜"
                vals = list(self.tree.item(iid, "values"))
                vals[0] = chk_icon
                self.tree.item(iid, values=vals)
        self._update_selected_tally()

    def _toggle_all_selection(self, selected: bool):
        for r in self.engine.cached_matches:
            r.selected = selected
        self._refresh_table_view()

    def _invert_checkboxes(self):
        for r in self.engine.cached_matches:
            r.selected = not r.selected
        self._refresh_table_view()

    def _update_selected_tally(self):
        sel_records = [r for r in self.engine.cached_matches if r.selected]
        total_mb = sum(r.file_size_bytes for r in sel_records) / (1024 * 1024)
        self.lbl_selected_tally.config(
            text=f"Selected: {len(sel_records):,} files ({total_mb:.1f} MB)"
        )

    # -------------------------------------------------------------------------
    # Transfer Execution & Undo
    # -------------------------------------------------------------------------

    def _execute_transfer_flow(self, action: str):
        dest = self.dest_path_var.get().strip()
        if not dest:
            messagebox.showerror("Destination Required", "Please specify a Destination Folder (Folder 3).", parent=self.dialog)
            return

        selected_count = sum(1 for r in self.engine.cached_matches if r.selected)
        if selected_count == 0:
            messagebox.showinfo("No Files Selected", "Please select at least one matched file to transfer.", parent=self.dialog)
            return

        action_name = "COPY" if action == "copy" else "CUT (MOVE)"
        verb = "copied" if action == "copy" else "moved"
        msg = (
            f"Are you sure you want to {action_name} {selected_count:,} matched files to:\n\n"
            f"{dest}\n\n"
            f"Files will be safely {verb} using multi-threaded I/O."
        )
        if action == "cut":
            msg += "\n\n⚠️ NOTE: Files will be removed from Target Folder (Folder 2) and placed in Folder 3! (Full Undo is available)"

        if not messagebox.askyesno(f"Confirm {action_name}", msg, icon="warning" if action == "cut" else "question", parent=self.dialog):
            return

        # Disable controls during transfer
        self.btn_copy.config(state=tk.DISABLED)
        self.btn_cut.config(state=tk.DISABLED)
        self.btn_sift.config(state=tk.DISABLED)
        self.pbar["value"] = 0
        self.lbl_action_status.config(text=f"Initializing {action} transfer...", fg=ACCENT)

        preserve = self.preserve_hierarchy_var.get()

        def worker():
            t0 = time.time()

            def on_progress(cur, tot, fname, mb_done):
                pct = (cur / max(tot, 1)) * 100.0
                elapsed = max(0.001, time.time() - t0)
                rate = mb_done / elapsed
                txt = f"{action.capitalize()}ing {cur:,}/{tot:,} ({pct:.1f}%): {fname} · {rate:.1f} MB/s"
                self.dialog.after(0, lambda: [
                    self.pbar.configure(value=pct),
                    self.lbl_action_status.configure(text=txt, fg=ACCENT)
                ])

            try:
                succ, fail, mb = self.engine.execute_transfer(
                    destination_folder=dest,
                    action=action,
                    preserve_hierarchy=preserve,
                    progress_callback=on_progress
                )
                self.dialog.after(0, lambda: self._on_transfer_finished(action, succ, fail, mb, dest))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_transfer_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_transfer_finished(self, action: str, succ: int, fail: int, mb: float, dest: str):
        self.btn_copy.config(state=tk.NORMAL)
        self.btn_cut.config(state=tk.NORMAL)
        self.btn_sift.config(state=tk.NORMAL)
        self.btn_undo.config(state=tk.NORMAL)
        self.pbar["value"] = 100

        action_name = "Copied" if action == "copy" else "Moved"
        status_txt = f"✓ {action_name} {succ:,} files ({mb:.1f} MB) successfully to destination."
        self.lbl_action_status.config(text=status_txt, fg=SUCCESS)

        # Notify user
        msg = f"Successfully {action_name.lower()} {succ:,} files ({mb:.1f} MB) to:\n{dest}"
        if fail > 0:
            msg += f"\n\n⚠️ Encountered errors on {fail:,} files."
        messagebox.showinfo(f"{action_name} Complete", msg, parent=self.dialog)

        # If Cut was executed, re-sift to update the live list cleanly
        if action == "cut":
            self._start_sift_match()

    def _on_transfer_error(self, err: str):
        self.btn_copy.config(state=tk.NORMAL)
        self.btn_cut.config(state=tk.NORMAL)
        self.btn_sift.config(state=tk.NORMAL)
        self.lbl_action_status.config(text=f"Transfer Error: {err}", fg=DANGER)
        messagebox.showerror("Transfer Error", f"Failed to transfer files:\n{err}", parent=self.dialog)

    def _execute_undo(self):
        if not self.engine.last_manifest:
            messagebox.showinfo("Undo", "No previous transfer found to undo.", parent=self.dialog)
            return

        manifest = self.engine.last_manifest
        action = manifest.action
        cnt = len(manifest.transferred_files)
        msg = (
            f"Are you sure you want to UNDO the previous {action.upper()} operation?\n\n"
            f"• Action: {action.upper()}\n"
            f"• Files: {cnt:,}\n"
            f"• From: {manifest.destination_folder}\n"
            f"• Back to: {manifest.target_folder}"
        )
        if not messagebox.askyesno("Confirm Undo", msg, icon="warning", parent=self.dialog):
            return

        self.btn_undo.config(state=tk.DISABLED)
        self.lbl_action_status.config(text="Reverting transfer...", fg=WARNING)

        def worker():
            try:
                restored, errors = self.engine.undo_last_transfer()
                self.dialog.after(0, lambda: self._on_undo_finished(restored, errors))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_undo_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_undo_finished(self, restored: int, errors: int):
        self.lbl_action_status.config(text=f"✓ Reverted {restored:,} files successfully.", fg=SUCCESS)
        messagebox.showinfo("Undo Successful", f"Successfully restored {restored:,} files back to their original locations.", parent=self.dialog)
        self._start_sift_match()

    def _on_undo_error(self, err: str):
        self.lbl_action_status.config(text=f"Undo Error: {err}", fg=DANGER)
        messagebox.showerror("Undo Error", f"Failed to undo transfer:\n{err}", parent=self.dialog)


# -----------------------------------------------------------------------------
# Module Entrypoint Helper Function
# -----------------------------------------------------------------------------

def open_file_sift_dialog(
    parent: tk.Tk,
    initial_source: Optional[str] = None,
    initial_target: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs
) -> FileSiftStudioDialog:
    """Helper function to instantiate and display the FileSift Studio dialog."""
    return FileSiftStudioDialog(
        parent=parent,
        initial_source=initial_source,
        initial_target=initial_target,
        log_callback=log_callback,
        *args,
        **kwargs
    )


if __name__ == "__main__":
    # Standalone execution
    root = tk.Tk()
    root.withdraw()
    src_arg = sys.argv[1] if len(sys.argv) > 1 else None
    tgt_arg = sys.argv[2] if len(sys.argv) > 2 else None
    dlg = open_file_sift_dialog(root, initial_source=src_arg, initial_target=tgt_arg)
    dlg.dialog.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
