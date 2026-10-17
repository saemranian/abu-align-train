"""
Abu Align & Train - Renamer Interactive Studio GUI
==================================================
Ultra-fast, collision-safe sequence and batch renaming studio with real-time live preview,
natural/chronological sorting, in-place renaming, destination copying, and one-click undo.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from renamer_engine import RenamerEngine, RenameItem

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


class RenamerStudioDialog:
    """
    Main Studio Dialog for Renamer in Abu Align.
    Provides live real-time preview of sequenced names before execution,
    in-place overwriting vs destination copying, and one-click undo restoration.
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
        self.dialog.title("🏷️ Renamer Studio · Sequence & Batch File Renamer")
        self.dialog.geometry("1180x820")
        self.dialog.minsize(920, 620)
        self.dialog.configure(bg=BG_DARK)

        # Center dialog
        sw = self.dialog.winfo_screenwidth()
        sh = self.dialog.winfo_screenheight()
        x = max(10, (sw - 1180) // 2)
        y = max(10, (sh - 820) // 2)
        self.dialog.geometry(f"+{x}+{y}")

        # Core Engine
        self.engine = RenamerEngine()

        # State Variables
        self.folder_var = tk.StringVar(value=initial_folder or "")
        self.base_name_var = tk.StringVar(value="frame")
        self.start_index_var = tk.IntVar(value=1)
        self.padding_var = tk.IntVar(value=4)
        self.delimiter_var = tk.StringVar(value="_")
        self.sort_by_var = tk.StringVar(value="natural")  # "natural", "name", "date_asc", "date_desc"
        self.lowercase_ext_var = tk.BooleanVar(value=True)

        # Operation Mode ("inplace" vs "copy")
        self.mode_var = tk.StringVar(value="inplace")     # "inplace" (overwrite in input) or "copy" (new folder)
        self.dest_folder_var = tk.StringVar(value="")

        self.search_filter_var = tk.StringVar(value="")
        self.is_renaming: bool = False

        if initial_folder and os.path.isdir(initial_folder):
            self.dest_folder_var.set(os.path.join(initial_folder, "renamed_sequence"))

        self._build_ui()

        # Auto-generate initial preview if folder is valid
        if initial_folder and os.path.isdir(initial_folder):
            self.dialog.after(150, self._update_preview)

    def _log(self, msg: str):
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass
        print(f"[Renamer] {msg}")

    # -------------------------------------------------------------------------
    # UI Layout Construction
    # -------------------------------------------------------------------------

    def _build_ui(self):
        # 1. Header Bar
        header = tk.Frame(self.dialog, bg=SURFACE, padx=18, pady=12, bd=0, highlightthickness=1, highlightbackground=BORDER)
        header.pack(fill=tk.X)

        title_lbl = tk.Label(
            header,
            text="🏷️ Renamer Studio  ·  Sequence & Batch File Renamer",
            font=("Segoe UI", 13, "bold"),
            fg=TEXT,
            bg=SURFACE
        )
        title_lbl.pack(side=tk.LEFT)

        sub_lbl = tk.Label(
            header,
            text="⚡ Live before/after preview · Collision-safe in-place overwrite or destination copy",
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
            text="  Configuration: Input Folder, Prompted Name & Sequence Settings  ",
            font=("Segoe UI", 9, "bold"),
            fg=ACCENT,
            bg=SURFACE,
            padx=12,
            pady=8,
            bd=1,
            relief=tk.SOLID
        )
        config_card.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Input Folder
        r1 = tk.Frame(config_card, bg=SURFACE)
        r1.pack(fill=tk.X, pady=3)

        tk.Label(r1, text="Input Folder:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=16, anchor="w").pack(side=tk.LEFT)
        e_fld = tk.Entry(r1, textvariable=self.folder_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        e_fld.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)
        self.folder_var.trace_add("write", lambda *_: self._on_folder_changed())

        tk.Button(r1, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_folder).pack(side=tk.LEFT)

        # Row 2: Prompted Desired Name + Delimiter + Start Index + Padding
        r2 = tk.Frame(config_card, bg=SURFACE)
        r2.pack(fill=tk.X, pady=4)

        tk.Label(r2, text="Desired Base Name:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE, width=16, anchor="w").pack(side=tk.LEFT)
        e_name = tk.Entry(r2, textvariable=self.base_name_var, font=("Segoe UI", 9, "bold"), bg=FIELD, fg=ACCENT, insertbackground=TEXT, width=22, bd=1, relief=tk.SOLID)
        e_name.pack(side=tk.LEFT, padx=(0, 12), ipady=3)
        self.base_name_var.trace_add("write", lambda *_: self._update_preview())

        tk.Label(r2, text="Delimiter:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 3))
        cb_delim = ttk.Combobox(r2, textvariable=self.delimiter_var, values=["_", "-", " ", ""], width=4, font=("Segoe UI", 9))
        cb_delim.pack(side=tk.LEFT, padx=(0, 12))
        self.delimiter_var.trace_add("write", lambda *_: self._update_preview())

        tk.Label(r2, text="Start Index:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 3))
        sp_start = tk.Spinbox(r2, textvariable=self.start_index_var, from_=0, to=999999, width=6, bg=FIELD, fg=TEXT, insertbackground=TEXT, font=("Segoe UI", 9))
        sp_start.pack(side=tk.LEFT, padx=(0, 12))
        self.start_index_var.trace_add("write", lambda *_: self._update_preview())

        tk.Label(r2, text="Digits Padding:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=SURFACE).pack(side=tk.LEFT, padx=(0, 3))
        sp_pad = tk.Spinbox(r2, textvariable=self.padding_var, from_=1, to=10, width=4, bg=FIELD, fg=TEXT, insertbackground=TEXT, font=("Segoe UI", 9))
        sp_pad.pack(side=tk.LEFT, padx=(0, 12))
        self.padding_var.trace_add("write", lambda *_: self._update_preview())

        tk.Checkbutton(r2, text="Lowercase Extensions (.jpg)", variable=self.lowercase_ext_var, bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT, command=self._update_preview).pack(side=tk.LEFT)

        # Row 3: Sort Order
        r3 = tk.Frame(config_card, bg=SURFACE)
        r3.pack(fill=tk.X, pady=4)

        tk.Label(r3, text="Sort Order:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=SURFACE, width=16, anchor="w").pack(side=tk.LEFT)
        sort_opts = [
            ("natural", "Natural Alphanumeric (frame_1, frame_2, ... frame_10)"),
            ("name", "Alphabetical (A - Z)"),
            ("date_asc", "Chronological (Date Modified / Taken - Earliest First)"),
            ("date_desc", "Reverse Chronological (Latest First)")
        ]
        cb_sort = ttk.Combobox(r3, textvariable=self.sort_by_var, values=[k for k, _ in sort_opts], state="readonly", width=14, font=("Segoe UI", 9))
        cb_sort.pack(side=tk.LEFT, padx=(0, 16))
        self.sort_by_var.trace_add("write", lambda *_: self._update_preview())

        # Operation Mode Radios
        tk.Label(r3, text="Mode:", font=("Segoe UI", 9, "bold"), fg=TEXT, bg=SURFACE).pack(side=tk.LEFT, padx=(10, 6))
        rb_in = tk.Radiobutton(
            r3, text="✏️ Overwrite in Input Folder (In-Place)", variable=self.mode_var, value="inplace",
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT,
            command=self._on_mode_changed
        )
        rb_cp = tk.Radiobutton(
            r3, text="📋 Copy with New Names to Another Folder", variable=self.mode_var, value="copy",
            bg=SURFACE, fg=TEXT, selectcolor=FIELD, activebackground=SURFACE, activeforeground=TEXT,
            command=self._on_mode_changed
        )
        rb_in.pack(side=tk.LEFT, padx=(0, 10))
        rb_cp.pack(side=tk.LEFT)

        # Row 4: Destination Folder (Only active when Copy mode is chosen)
        self.r_dest = tk.Frame(config_card, bg=SURFACE)
        self.r_dest.pack(fill=tk.X, pady=(4, 2))

        tk.Label(self.r_dest, text="Destination Folder:", font=("Segoe UI", 9, "bold"), fg=TEXT_MUTED, bg=SURFACE, width=16, anchor="w").pack(side=tk.LEFT)
        self.e_dest = tk.Entry(self.r_dest, textvariable=self.dest_folder_var, font=("Segoe UI", 9), bg=FIELD, fg=TEXT, insertbackground=TEXT, bd=1, relief=tk.SOLID)
        self.e_dest.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)
        self.dest_folder_var.trace_add("write", lambda *_: self._update_preview())

        self.btn_browse_dest = tk.Button(self.r_dest, text="Browse...", font=("Segoe UI", 9), bg=SURFACE_LIGHT, fg=TEXT, relief=tk.FLAT, padx=10, command=self._browse_dest)
        self.btn_browse_dest.pack(side=tk.LEFT)

        # 3. Middle Section: Live Before / After Preview Table
        preview_card = tk.Frame(main_box, bg=SURFACE, bd=1, relief=tk.SOLID, highlightbackground=BORDER)
        preview_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Preview Banner
        top_prev_bar = tk.Frame(preview_card, bg=SURFACE_LIGHT, padx=12, pady=8)
        top_prev_bar.pack(fill=tk.X)

        self.lbl_preview_summary = tk.Label(
            top_prev_bar,
            text="👁️ Live Preview: Select a folder to view sequenced naming before applying.",
            font=("Segoe UI", 9, "bold"),
            fg=TEXT,
            bg=SURFACE_LIGHT
        )
        self.lbl_preview_summary.pack(side=tk.LEFT)

        # Search filter
        search_box = tk.Frame(top_prev_bar, bg=SURFACE_LIGHT)
        search_box.pack(side=tk.RIGHT)
        tk.Label(search_box, text="🔍 Filter:", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=SURFACE_LIGHT).pack(side=tk.LEFT, padx=(0, 4))
        e_filt = tk.Entry(search_box, textvariable=self.search_filter_var, font=("Segoe UI", 8), bg=FIELD, fg=TEXT, insertbackground=TEXT, width=22, bd=1, relief=tk.SOLID)
        e_filt.pack(side=tk.LEFT, ipady=2)
        self.search_filter_var.trace_add("write", lambda *_: self._render_table())

        # Table Treeview
        table_frame = tk.Frame(preview_card, bg=SURFACE)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Renamer.Treeview",
            background=FIELD,
            foreground=TEXT,
            fieldbackground=FIELD,
            rowheight=24,
            font=("Segoe UI", 9)
        )
        style.map("Renamer.Treeview", background=[("selected", "#1e3a5f")], foreground=[("selected", "#ffffff")])
        style.configure("Renamer.Treeview.Heading", background=SURFACE_LIGHT, foreground=TEXT, font=("Segoe UI", 9, "bold"))

        cols = ("idx", "orig_name", "arrow", "new_name", "size", "status")
        self.tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            style="Renamer.Treeview",
            selectmode="browse"
        )
        self.tree.heading("idx", text="#", anchor="center")
        self.tree.heading("orig_name", text="Original Filename", anchor="w")
        self.tree.heading("arrow", text="➔", anchor="center")
        self.tree.heading("new_name", text="New Sequenced Name (Preview)", anchor="w")
        self.tree.heading("size", text="Size", anchor="e")
        self.tree.heading("status", text="Action Status", anchor="center")

        self.tree.column("idx", width=50, anchor="center")
        self.tree.column("orig_name", width=360, anchor="w")
        self.tree.column("arrow", width=40, anchor="center")
        self.tree.column("new_name", width=360, anchor="w")
        self.tree.column("size", width=90, anchor="e")
        self.tree.column("status", width=120, anchor="center")

        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        # 4. Action Execution & Safety Footer
        footer = tk.Frame(main_box, bg=SURFACE, padx=14, pady=10, bd=1, relief=tk.SOLID, highlightbackground=BORDER)
        footer.pack(fill=tk.X)

        self.pbar = ttk.Progressbar(footer, orient=tk.HORIZONTAL, mode="determinate")
        self.pbar.pack(fill=tk.X, pady=(0, 8))

        bot_row = tk.Frame(footer, bg=SURFACE)
        bot_row.pack(fill=tk.X)

        self.lbl_action_status = tk.Label(
            bot_row,
            text="Verify preview above, then execute renaming.",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=SURFACE
        )
        self.lbl_action_status.pack(side=tk.LEFT)

        # Action Buttons
        self.btn_undo = tk.Button(
            bot_row,
            text="↩️ Undo Last Rename",
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

        self.btn_execute = tk.Button(
            bot_row,
            text="⚡ RENAME ALL IN-PLACE",
            font=("Segoe UI", 10, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=18,
            pady=5,
            cursor="hand2",
            command=self._start_execution
        )
        self.btn_execute.pack(side=tk.RIGHT)
        self._on_mode_changed()

    # -------------------------------------------------------------------------
    # Mode & Input Event Handlers
    # -------------------------------------------------------------------------

    def _on_folder_changed(self):
        f = self.folder_var.get().strip()
        if f and os.path.isdir(f):
            if not self.dest_folder_var.get().strip():
                self.dest_folder_var.set(os.path.join(f, "renamed_sequence"))
            self._update_preview()

    def _on_mode_changed(self):
        mode = self.mode_var.get()
        if mode == "copy":
            self.e_dest.config(state=tk.NORMAL)
            self.btn_browse_dest.config(state=tk.NORMAL)
            self.btn_execute.config(
                text="📋 COPY RENAMED SEQUENCE",
                bg="#059669",
                activebackground="#047857"
            )
        else:
            self.e_dest.config(state=tk.DISABLED)
            self.btn_browse_dest.config(state=tk.DISABLED)
            self.btn_execute.config(
                text="⚡ RENAME ALL IN-PLACE (OVERWRITE)",
                bg="#0284c7",
                activebackground="#0369a1"
            )
        self._update_preview()

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select Folder to Rename", parent=self.dialog)
        if d:
            self.folder_var.set(d)

    def _browse_dest(self):
        d = filedialog.askdirectory(title="Select Destination Folder for Renamed Copies", parent=self.dialog)
        if d:
            self.dest_folder_var.set(d)

    # -------------------------------------------------------------------------
    # Live Preview Computation & Rendering
    # -------------------------------------------------------------------------

    def _update_preview(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            self._clear_table()
            self.lbl_preview_summary.config(text="Select a valid folder to view sequenced naming preview.", fg=TEXT_MUTED)
            return

        base = self.base_name_var.get()
        delim = self.delimiter_var.get()
        try:
            start_idx = int(self.start_index_var.get())
        except Exception:
            start_idx = 1
        try:
            pad = int(self.padding_var.get())
        except Exception:
            pad = 4

        sort_by = self.sort_by_var.get()
        low_ext = self.lowercase_ext_var.get()
        mode = self.mode_var.get()
        dest = self.dest_folder_var.get().strip() if mode == "copy" else None

        items = self.engine.generate_preview(
            folder_path=folder,
            base_name=base,
            start_index=start_idx,
            padding=pad,
            delimiter=delim,
            sort_by=sort_by,
            lowercase_ext=low_ext,
            dest_folder=dest
        )

        total = len(items)
        if total == 0:
            self.lbl_preview_summary.config(text="No files found in folder.", fg=WARNING)
            self._clear_table()
            return

        first_eg = f"{items[0].original_filename} ➔ {items[0].new_filename}"
        last_eg = f"{items[-1].original_filename} ➔ {items[-1].new_filename}" if total > 1 else ""
        summary_txt = f"👁️ Live Preview: {total:,} files will be sequenced | E.g., {first_eg}"
        if last_eg:
            summary_txt += f" ... {last_eg}"

        self.lbl_preview_summary.config(text=summary_txt, fg=SUCCESS)
        self._render_table()

    def _clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _render_table(self):
        self._clear_table()
        query = self.search_filter_var.get().strip().lower()
        mode = self.mode_var.get()

        for idx, item in enumerate(self.engine.cached_items):
            if query and (query not in item.original_filename.lower() and query not in item.new_filename.lower()):
                continue

            sz_str = f"{item.size_bytes / (1024 * 1024):.1f} MB" if item.size_bytes > 1024*1024 else f"{item.size_bytes / 1024:.0f} KB"
            status_txt = "WILL COPY" if mode == "copy" else ("WILL RENAME" if item.is_changed else "UNCHANGED")

            self.tree.insert(
                "",
                tk.END,
                iid=f"item_{idx}",
                values=(idx + 1, item.original_filename, "➔", item.new_filename, sz_str, status_txt)
            )

    # -------------------------------------------------------------------------
    # Execution & Undo
    # -------------------------------------------------------------------------

    def _start_execution(self):
        items = self.engine.cached_items
        if not items:
            messagebox.showinfo("No Files", "No files found to rename.", parent=self.dialog)
            return

        mode = self.mode_var.get()
        dest = self.dest_folder_var.get().strip() if mode == "copy" else None

        if mode == "copy" and not dest:
            messagebox.showerror("Destination Required", "Please specify a destination folder for copied files.", parent=self.dialog)
            return

        total = len(items)
        if mode == "inplace":
            msg = (
                f"Are you sure you want to RENAME {total:,} files IN-PLACE inside:\n\n"
                f"{self.folder_var.get()}\n\n"
                f"Using prompted base name '{self.base_name_var.get()}'?\n"
                f"(Safe two-phase renaming ensures zero collisions, and full One-Click Undo is available!)"
            )
            if not messagebox.askyesno("Confirm In-Place Rename", msg, icon="warning", parent=self.dialog):
                return
        else:
            msg = (
                f"Are you sure you want to COPY {total:,} files with new sequenced names into:\n\n"
                f"{dest}\n\n"
                f"Original files in the input folder will remain 100% untouched."
            )
            if not messagebox.askyesno("Confirm Copy Rename", msg, icon="question", parent=self.dialog):
                return

        self.btn_execute.config(state=tk.DISABLED)
        self.pbar["value"] = 0
        self.lbl_action_status.config(text=f"Executing {mode} rename...", fg=ACCENT)

        def worker():
            t0 = time.time()

            def on_progress(cur, tot, fname):
                pct = (cur / max(tot, 1)) * 100.0
                txt = f"Processing {cur:,}/{tot:,} ({pct:.1f}%): {fname}"
                self.dialog.after(0, lambda: [
                    self.pbar.configure(value=pct),
                    self.lbl_action_status.configure(text=txt, fg=ACCENT)
                ])

            try:
                succ, err = self.engine.execute_rename(
                    items=items,
                    mode=mode,
                    dest_folder=dest,
                    progress_callback=on_progress
                )
                self.dialog.after(0, lambda: self._on_rename_finished(mode, succ, err))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_rename_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_rename_finished(self, mode: str, succ: int, err: int):
        self.btn_execute.config(state=tk.NORMAL)
        self.btn_undo.config(state=tk.NORMAL)
        self.pbar["value"] = 100

        action_name = "Copied & Sequenced" if mode == "copy" else "Renamed In-Place"
        self.lbl_action_status.config(text=f"✓ Successfully {action_name.lower()} {succ:,} files.", fg=SUCCESS)

        msg = f"✓ {action_name} {succ:,} files successfully!"
        if err > 0:
            msg += f"\n\n⚠️ Encountered errors on {err} files."
        messagebox.showinfo("Renamer Complete", msg, parent=self.dialog)

        # Refresh preview
        self._update_preview()

    def _on_rename_error(self, err_msg: str):
        self.btn_execute.config(state=tk.NORMAL)
        self.lbl_action_status.config(text=f"Rename Error: {err_msg}", fg=DANGER)
        messagebox.showerror("Renamer Error", f"Operation failed:\n{err_msg}", parent=self.dialog)

    def _execute_undo(self):
        if not self.engine.last_manifest:
            messagebox.showinfo("Undo", "No previous rename operation found to undo.", parent=self.dialog)
            return

        manifest = self.engine.last_manifest
        cnt = len(manifest.mappings)
        msg = f"Revert {cnt:,} files back to their original names?"
        if not messagebox.askyesno("Confirm Undo", msg, icon="warning", parent=self.dialog):
            return

        self.btn_undo.config(state=tk.DISABLED)
        self.lbl_action_status.config(text="Reverting rename operation...", fg=WARNING)

        def worker():
            try:
                restored, errors = self.engine.undo_last_rename()
                self.dialog.after(0, lambda: self._on_undo_finished(restored, errors))
            except Exception as e:
                self.dialog.after(0, lambda: self._on_undo_error(str(e)))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_undo_finished(self, restored: int, errors: int):
        self.lbl_action_status.config(text=f"✓ Restored {restored:,} files back to original names.", fg=SUCCESS)
        messagebox.showinfo("Undo Successful", f"Restored {restored:,} files back to original names.", parent=self.dialog)
        self._update_preview()

    def _on_undo_error(self, err_msg: str):
        self.lbl_action_status.config(text=f"Undo Error: {err_msg}", fg=DANGER)
        messagebox.showerror("Undo Error", f"Failed to undo rename:\n{err_msg}", parent=self.dialog)


# -----------------------------------------------------------------------------
# Module Entrypoint Helper Function
# -----------------------------------------------------------------------------

def open_renamer_dialog(
    parent: tk.Tk,
    initial_folder: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs
) -> RenamerStudioDialog:
    """Helper function to instantiate and display the Renamer Studio dialog."""
    return RenamerStudioDialog(
        parent=parent,
        initial_folder=initial_folder,
        log_callback=log_callback,
        *args,
        **kwargs
    )


if __name__ == "__main__":
    # Standalone execution
    root = tk.Tk()
    root.withdraw()
    init_dir = sys.argv[1] if len(sys.argv) > 1 else None
    dlg = open_renamer_dialog(root, initial_folder=init_dir)
    dlg.dialog.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
