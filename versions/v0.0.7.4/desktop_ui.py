"""Desktop presentation layer. Pipeline state and engine callbacks stay in the app."""
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


BG = "#111518"
SURFACE = "#1a2024"
FIELD = "#242d32"
BORDER = "#344149"
TEXT = "#e7edef"
MUTED = "#a0adb5"
ACCENT = "#70dbc0"
DISABLED = "#75838b"


def style_entry(widget):
    widget.configure(bg=FIELD, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0,
                     font=("Segoe UI", 10),
                     disabledbackground=SURFACE, disabledforeground=DISABLED,
                     selectbackground=ACCENT, selectforeground=BG,
                     highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)


def wrap_choices(frame, minimum=170):
    """Keep selection cards readable at narrow window widths."""
    children = frame.winfo_children()
    columns = [None]
    def arrange(event):
        count = max(1, min(len(children), event.width // minimum))
        if columns[0] == count:
            return
        columns[0] = count
        for i in range(len(children)):
            frame.columnconfigure(i, weight=1 if i < count else 0, uniform="choices" if i < count else "")
        for i, child in enumerate(children):
            child.grid(row=i//count, column=i%count, sticky="ew", padx=(0, 8), pady=(0, 8))
    frame.bind("<Configure>", arrange)


def label(parent, text, size=10, muted=False, **kwargs):
    return tk.Label(parent, text=text, bg=parent.cget("bg"),
                    fg=MUTED if muted else TEXT, font=("Segoe UI", size), **kwargs)


class StateAwareButton(tk.Button):
    """Let related presentation controls follow the existing button lifecycle."""
    def configure(self, cnf=None, **kwargs):
        result = super().configure(cnf, **kwargs)
        listener = getattr(self, "state_listener", None)
        if listener is not None:
            listener(str(self.cget("state")))
        return result

    config = configure


def button(parent, text, command, primary=False, **kwargs):
    widget = StateAwareButton(parent, text=text, command=command,
                       bg=ACCENT if primary else FIELD,
                       fg=BG if primary else TEXT,
                       activebackground="#93e8d2" if primary else BORDER,
                       activeforeground=BG if primary else TEXT,
                       disabledforeground="#738087", relief="flat", bd=0,
                       highlightthickness=1, highlightbackground=parent.cget("bg"),
                       highlightcolor=ACCENT, padx=14, pady=8,
                       font=("Segoe UI", 10, "bold" if primary else "normal"),
                       cursor="hand2", takefocus=True, **kwargs)
    widget.bind("<Return>", lambda e: widget.invoke())
    widget.bind("<Enter>", lambda e: widget.configure(
        highlightbackground=ACCENT if str(widget.cget("state")) != "disabled" else parent.cget("bg")))
    widget.bind("<Leave>", lambda e: widget.configure(highlightbackground=parent.cget("bg")))
    return widget


def configure_theme(app):
    root = app.root
    root.option_add("*Font", "{Segoe UI} 10")
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", BG)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("TCombobox", fieldbackground=FIELD, background=FIELD,
                    foreground=TEXT, arrowcolor=MUTED, bordercolor=BORDER,
                    lightcolor=FIELD, darkcolor=FIELD, padding=6)
    style.map("TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", SURFACE)],
              foreground=[("disabled", "#738087"), ("readonly", TEXT)],
              selectbackground=[("!disabled", FIELD)], selectforeground=[("!disabled", TEXT)],
              bordercolor=[("focus", ACCENT)])
    style.configure("TNotebook", background=BG, borderwidth=0, bordercolor=BG,
                    lightcolor=BG, darkcolor=BG, tabmargins=(0, 0, 0, 8))
    style.configure("TNotebook.Tab", background=BG, foreground=MUTED,
                    padding=(22, 12), borderwidth=0, bordercolor=BG,
                    lightcolor=BG, darkcolor=BG)
    style.map("TNotebook.Tab", background=[("selected", SURFACE), ("active", FIELD)],
              foreground=[("selected", ACCENT), ("active", TEXT)],
              bordercolor=[("selected", SURFACE), ("!selected", BG)],
              lightcolor=[("selected", SURFACE), ("!selected", BG)],
              darkcolor=[("selected", SURFACE), ("!selected", BG)])
    style.configure("TNotebook.Tab", focuscolor=ACCENT)
    style.configure("Vertical.TScrollbar", background=BORDER, troughcolor=SURFACE,
                    borderwidth=0, arrowsize=12, arrowcolor=MUTED,
                    bordercolor=SURFACE, lightcolor=BORDER, darkcolor=BORDER)
    style.configure("Abu.Horizontal.TProgressbar", background=ACCENT,
                    troughcolor=FIELD, borderwidth=0, lightcolor=ACCENT,
                    darkcolor=ACCENT, bordercolor=FIELD, thickness=5)
    style.configure("TScale", background=SURFACE, troughcolor=FIELD)


def polish_controls(parent):
    """Bring existing engine/tool widgets into the shared theme, retaining callbacks."""
    for widget in parent.winfo_children():
        if isinstance(widget, (tk.Radiobutton, tk.Checkbutton)):
            widget.configure(font=("Segoe UI", 10), bg=SURFACE, fg=TEXT,
                             activebackground=SURFACE, activeforeground=ACCENT,
                             selectcolor=BG, highlightthickness=1,
                             highlightbackground=SURFACE, highlightcolor=ACCENT,
                             bd=0, pady=5, cursor="hand2", takefocus=True)
        elif isinstance(widget, ttk.Combobox):
            widget.configure(font=("Segoe UI", 10))
        elif isinstance(widget, tk.Entry) and not isinstance(widget, ttk.Entry):
            style_entry(widget)
        elif isinstance(widget, tk.Label):
            # Keep semantic warning/status colors, but make small parameter labels readable.
            font = widget.tk.call("font", "actual", widget.cget("font"))
            values = dict(zip(font[::2], font[1::2]))
            if abs(int(values.get("-size", 10))) < 10:
                widget.configure(font=("Segoe UI", 10, values.get("-weight", "normal")))
        polish_controls(widget)


def polish_dialog(dialog):
    """Unify tool dialogs without recoloring their interactive preview canvases."""
    dialog.configure(bg=BG)
    def walk(parent):
        for widget in parent.winfo_children():
            if isinstance(widget, (tk.Frame, tk.LabelFrame)):
                widget.configure(bg=SURFACE, bd=0)
                if isinstance(widget, tk.LabelFrame):
                    widget.configure(fg=TEXT, padx=12, pady=10)
            elif isinstance(widget, tk.Button):
                widget.configure(bg=FIELD, fg=TEXT, activebackground=BORDER,
                                 activeforeground=TEXT, relief="flat", bd=0,
                                 highlightthickness=1, highlightbackground=FIELD,
                                 highlightcolor=ACCENT, disabledforeground="#738087",
                                 cursor="hand2")
            elif isinstance(widget, (tk.Label, tk.Checkbutton, tk.Radiobutton)):
                widget.configure(bg=SURFACE)
            elif isinstance(widget, tk.Entry) and not isinstance(widget, ttk.Entry):
                style_entry(widget)
            elif isinstance(widget, tk.Listbox):
                widget.configure(bg=BG, fg=TEXT, selectbackground=FIELD,
                                 selectforeground=ACCENT, highlightthickness=0)
            elif isinstance(widget, tk.Text):
                widget.configure(bg=BG, fg=TEXT, insertbackground=TEXT,
                                 selectbackground=FIELD, selectforeground=TEXT)
            walk(widget)
    walk(dialog)


class ScrollPage(tk.Frame):
    """A settings page with local wheel scrolling and keyboard focus reveal."""
    def __init__(self, parent):
        super().__init__(parent, bg=SURFACE)
        self.canvas = tk.Canvas(self, bg=SURFACE, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.bar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = tk.Frame(self.canvas, bg=SURFACE, padx=22, pady=18)
        self.window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self.canvas.bind("<Configure>", self._resize)
        self.body.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))

    def _resize(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def bind_navigation(self, subtree=None):
        def wheel(event):
            if self.canvas.yview() != (0.0, 1.0):
                self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        def reveal(event):
            widget = event.widget
            y = widget.winfo_rooty() - self.body.winfo_rooty()
            top = self.canvas.canvasy(0)
            bottom = top + self.canvas.winfo_height()
            height = max(1, self.body.winfo_height())
            if y < top:
                self.canvas.yview_moveto(max(0, y - 12) / height)
            elif y + widget.winfo_height() > bottom:
                self.canvas.yview_moveto((y + widget.winfo_height() - self.canvas.winfo_height() + 12) / height)

        def walk(widget):
            widget.bind("<MouseWheel>", wheel, add="+")
            widget.bind("<FocusIn>", reveal, add="+")
            for child in widget.winfo_children():
                walk(child)
        walk(subtree if subtree is not None else self.body)
        self.canvas.bind("<MouseWheel>", wheel)


def responsive_groups(panel):
    """Wrap whole parameter groups without changing their internal controls."""
    groups = list(panel.winfo_children())
    for group in groups:
        group.pack_forget()
        if isinstance(group, tk.Frame):
            group.configure(padx=12, pady=12, highlightthickness=1, highlightbackground=BORDER)
    last = [None]

    def arrange(event):
        minimum = max((g.winfo_reqwidth() for g in groups), default=280) + 28
        columns = max(1, min(len(groups), event.width // minimum))
        if columns == last[0]:
            return
        last[0] = columns
        for i in range(len(groups)):
            panel.columnconfigure(i, weight=1 if i < columns else 0, minsize=0)
        for i, group in enumerate(groups):
            group.grid(row=i // columns, column=i % columns, sticky="new",
                       padx=(0, 16), pady=(6, 16))
    panel.bind("<Configure>", arrange)


def section(parent, title, subtitle=""):
    label(parent, title, 15).pack(anchor="w", pady=(0, 5))
    if subtitle:
        label(parent, subtitle, muted=True).pack(anchor="w", pady=(0, 12))


def field(parent, title, variable, values, width=12):
    frame = tk.Frame(parent, bg=SURFACE)
    frame.pack(side="left", padx=(0, 22), pady=6)
    label(frame, title, muted=True).pack(anchor="w", pady=(0, 5))
    control = ttk.Combobox(frame, textvariable=variable, values=values,
                           state="readonly", width=width)
    control.pack(anchor="w")
    return control


def open_monitor(app):
    existing = getattr(app, "monitor_window", None)
    if existing is not None and existing.winfo_exists():
        existing.lift()
        return
    dialog = app.monitor_window = tk.Toplevel(app.root)
    dialog.title("Live hardware monitor")
    dialog.configure(bg=SURFACE)
    dialog.geometry("600x340")
    dialog.minsize(500, 300)
    dialog._custom_theme = True
    dialog.transient(app.root)
    label(dialog, "Live hardware", 18).pack(anchor="w", padx=22, pady=18)
    sources = [app.hw_cpu_lbl, app.hw_ram_lbl, app.hw_gpu_lbl,
               app.hw_temp_lbl, app.hw_vram_lbl, app.hw_page_lbl]
    labels = []
    grid = tk.Frame(dialog, bg=SURFACE)
    grid.pack(fill="both", expand=True, padx=22, pady=(0, 18))
    for index, source in enumerate(sources):
        item = label(grid, "", anchor="w", padx=14, pady=14)
        item.configure(bg=FIELD)
        item.grid(row=index//2, column=index%2, sticky="nsew", padx=4, pady=4)
        grid.columnconfigure(index%2, weight=1)
        grid.rowconfigure(index//2, weight=1)
        labels.append(item)
    timer = [None]
    def refresh():
        for source, item in zip(sources, labels):
            item.configure(text=source.cget("text"), fg=source.cget("fg"))
        timer[0] = dialog.after(1000, refresh)
    def close():
        if timer[0] is not None:
            dialog.after_cancel(timer[0])
        dialog.destroy()
    dialog.protocol("WM_DELETE_WINDOW", close)
    dialog.bind("<Escape>", lambda e: close())
    refresh()


def enhance_video_dialog(app, ctx):
    """A scrollable, task-focused presentation over the existing extractor."""
    from types import SimpleNamespace
    dialog = ctx["dialog"]
    dialog._custom_theme = True
    dialog.title("Video frames · Abu Align")
    dialog.geometry("820x760")
    dialog.minsize(680, 560)
    polish_dialog(dialog)
    header = ctx["hdr"]
    for child in header.winfo_children():
        child.destroy()
    label(header, "Extract video frames", 20).pack(anchor="w")
    label(header, "Turn a video into a photo sequence for alignment.", muted=True,
          wraplength=590, justify="left").pack(anchor="w", pady=(6, 0))
    page = ctx["video_page"]
    for name, title in [("paths_card", "Source & destination"), ("ctrl_card", "Sampling & quality"),
                        ("prog_card", "Progress")]:
        ctx[name].configure(text=title, font=("Segoe UI", 11, "bold"), padx=12, pady=12)
        ctx[name].pack_configure(pady=(0, 18))
    # Pair every label with its input instead of squeezing six controls across a row.
    for name in ("g1", "g2"):
        row = ctx[name]
        children = row.winfo_children()
        for child in children:
            child.pack_forget()
        for index, child in enumerate(children):
            child.grid(row=index//2, column=index%2, sticky="w", padx=(0, 18), pady=6)
    ctx["hw_chk"].configure(text="Use NVIDIA hardware acceleration", fg=TEXT)
    ctx["blur_chk"].configure(text="Skip blurry frames · threshold", fg=TEXT)
    for child in ctx["g3"].winfo_children():
        child.pack_forget()
    ctx["hw_chk"].grid(row=0, column=0, columnspan=2, sticky="w", pady=6)
    ctx["blur_chk"].grid(row=1, column=0, sticky="w", pady=6)
    ctx["blur_entry"].grid(row=1, column=1, sticky="w")
    ctx["extract_btn"].configure(text="Extract frames", bg=ACCENT, fg=BG, activebackground="#93e8d2")
    ctx["use_ds_btn"].configure(text="Use dataset")
    ctx["open_folder_btn"].configure(text="Open output")
    ctx["dlg_pbar"].configure(style="Abu.Horizontal.TProgressbar")
    ctx["status_lbl"].configure(text="Choose a video to begin.", wraplength=530, justify="left")
    # Keep internal engine values stable while presenting readable choices.
    from rig_designer_ui import friendly_choices
    friendly_choices(ctx["mode_cb"], ctx["mode_var"],
                     {"fps": "Frames per second", "stride": "Every Nth frame", "keyframes": "Keyframes only", "all": "Every frame"})
    ctx["mode_cb"].configure(width=22)
    def sync(*_):
        for widget, enabled in [(ctx["fps_entry"], ctx["mode_var"].get() == "fps"),
                                (ctx["stride_entry"], ctx["mode_var"].get() == "stride"),
                                (ctx["qual_entry"], ctx["fmt_var"].get() == "jpg"),
                                (ctx["blur_entry"], ctx["blur_filter_var"].get())]:
            widget.configure(state="normal" if enabled else "disabled")
    for name in ("mode_var", "fmt_var", "blur_filter_var"):
        ctx[name].trace_add("write", sync)
    polish_controls(ctx["content"])
    sync()
    page.bind_navigation(ctx["content"])
    def close():
        if str(ctx["extract_btn"].cget("state")) != "disabled":
            dialog.destroy()
    ctx["close_btn"].configure(command=close)
    dialog.protocol("WM_DELETE_WINDOW", close)
    dialog.bind("<Escape>", lambda e: close())
    app.video_designer = SimpleNamespace(dialog=dialog, page=page, mode=ctx["mode_var"],
        format=ctx["fmt_var"], blur=ctx["blur_filter_var"], fps=ctx["fps_entry"],
        stride=ctx["stride_entry"], quality=ctx["qual_entry"], threshold=ctx["blur_entry"], close=close)
    dialog.after_idle(ctx["in_entry"].focus_set)


class LogSearch:
    """On-demand, non-destructive inspection of the retained log buffer."""
    def __init__(self, parent, console, follow):
        self.console, self.follow = console, follow
        self.frame = tk.Frame(parent, bg=SURFACE)
        self.query = tk.StringVar(master=parent)
        self.entry = tk.Entry(self.frame, textvariable=self.query)
        style_entry(self.entry)
        self.entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.count = label(self.frame, "", muted=True)
        self.count.pack(side="left", padx=8)
        for title, command in [("Previous", lambda: self.find(-1)), ("Next", self.find),
                               ("Resume live", self.resume_live), ("Close", self.hide)]:
            button(self.frame, title, command).pack(side="left", padx=3)
        self.current = -1
        self.query.trace_add("write", self.changed)
        self.entry.bind("<Return>", lambda e: (self.find(), "break")[-1])
        self.entry.bind("<Shift-Return>", lambda e: (self.find(-1), "break")[-1])
        self.entry.bind("<Escape>", lambda e: (self.hide(), "break")[-1])
        console.tag_configure("log_match", background=ACCENT, foreground=BG)

    def show(self):
        self.frame.pack(fill="x", pady=(0, 8), before=self.console.frame)
        self.entry.focus_set()
        self.entry.selection_range(0, "end")
        if self.query.get():
            self.find(0)

    def hide(self):
        self.frame.pack_forget()
        self.console.tag_remove("log_match", "1.0", "end")
        self.console.focus_set()

    def resume_live(self):
        self.hide()
        self.follow.set(True)
        self.console.see("end")

    def changed(self, *_):
        self.current = -1
        self.find()

    def find(self, direction=1):
        query = self.query.get()
        self.console.tag_remove("log_match", "1.0", "end")
        if not query:
            self.count.configure(text="Find in log")
            return
        matches, start = [], "1.0"
        length = int(self.console.tk.call("string", "length", query))
        while True:
            match = self.console.search(query, start, stopindex="end", nocase=True)
            if not match:
                break
            matches.append(match)
            start = f"{match}+{length}c"
        if not matches:
            self.count.configure(text="No matches")
            self.current = -1
            return
        self.current = (self.current + direction) % len(matches)
        match = matches[self.current]
        self.follow.set(False)
        self.console.tag_add("log_match", match, f"{match}+{length}c")
        self.console.see(match)
        self.count.configure(text=f"{self.current+1} / {len(matches)}")


def build_workspace(app, tooltip):
    configure_theme(app)
    root = app.root
    app.run_mode_var = tk.StringVar(master=root, value="Align & train" if app.auto_train_var.get() else "Align only")
    # Header: identity on the left, secondary destinations on the right.
    header = tk.Frame(root, bg=BG, padx=24, pady=12)
    header.pack(fill="x")
    mark = tk.Label(header, text="A", bg=ACCENT, fg=BG, font=("Segoe UI", 19, "bold"), padx=10)
    mark.pack(side="left", padx=(0, 12))
    label(header, "Abu Align & Train", 19).pack(side="left")
    label(header, "v0.0.7.4", 9, muted=True).pack(side="left", padx=12)
    button(header, "User guide", app._open_user_guide).pack(side="right")
    prepare = tk.Menubutton(header, text="Prepare input ▾", bg=BG, fg=MUTED,
        activebackground=FIELD, activeforeground=TEXT, relief="flat", padx=14, pady=8,
        font=("Segoe UI", 10), takefocus=True, highlightthickness=1,
        highlightbackground=BG, highlightcolor=ACCENT)
    prepare_menu = tk.Menu(prepare, tearoff=False, bg=SURFACE, fg=TEXT,
                           activebackground=FIELD, activeforeground=ACCENT, relief="flat", bd=0)
    prepare_menu.add_command(label="360° rig designer", command=app._open_panoramic_splitter_dialog)
    prepare_menu.add_command(label="Video frames", command=app._open_video_extractor_dialog)
    prepare.configure(menu=prepare_menu)
    prepare.pack(side="right", padx=8)

    # Pin run controls to the bottom before allocating expandable notebook space.
    footer = tk.Frame(root, bg=BG, padx=24, pady=12)
    footer.pack(side="bottom", fill="x")
    app.workflow_summary = label(footer, "", muted=True, anchor="w")
    app.workflow_summary.pack(fill="x", pady=(0, 10))
    def summarize_workflow(*_):
        alignment = {"vggt_omega": "VGGT-Omega", "ace0": "ACE0", "pycusfm": "PyCuSFM",
                     "opensfm": "OpenSfM", "360dvo": "360DVO"}.get(app.engine_var.get(), app.engine_var.get())
        training = {"lichtfeld": "LichtFeld Studio", "splatfacto": "Splatfacto",
                    "3dgrut": "3DGRUT", "pfgs360": "PFGS360"}.get(app.train_engine_var.get(), app.train_engine_var.get())
        mode = app.run_mode_var.get()
        app.workflow_summary.configure(text=(f"WORKFLOW   {training}  ·  Training only" if mode == "Train only"
            else f"WORKFLOW   {alignment}  →  {training}" if mode == "Align & train"
            else f"WORKFLOW   {alignment}  ·  Alignment only"))
    for variable in (app.engine_var, app.train_engine_var, app.run_mode_var):
        variable.trace_add("write", summarize_workflow)
    summarize_workflow()
    status = tk.Frame(footer, bg=BG)
    status.pack(fill="x", pady=(0, 8))
    def compact_footer(event):
        if event.widget is not root or not app.workflow_summary.winfo_exists():
            return
        if event.height < 820:
            app.workflow_summary.pack_forget()
        elif not app.workflow_summary.winfo_manager():
            app.workflow_summary.pack(fill="x", pady=(0, 10), before=status)
    root.bind("<Configure>", compact_footer, add="+")
    app.telemetry_lbl = label(status, "Ready to start", muted=True, anchor="w")
    app.telemetry_lbl.pack(side="left", fill="x", expand=True)
    app.eta_lbl = label(status, "", muted=True)
    app.eta_lbl.pack(side="right", padx=12)
    app.pct_lbl = label(status, "0.0%", muted=True)
    app.pct_lbl.pack(side="right")
    status.bind("<Configure>", lambda e: app.telemetry_lbl.configure(wraplength=max(200, e.width - 230)))
    app.progress_bar = ttk.Progressbar(footer, style="Abu.Horizontal.TProgressbar")
    app.progress_bar.pack(fill="x", pady=(0, 12))
    actions = tk.Frame(footer, bg=BG)
    actions.pack(fill="x")
    app.run_mode_combo = ttk.Combobox(actions, textvariable=app.run_mode_var,
        values=("Align & train", "Align only", "Train only"), state="readonly", width=16)
    app.run_mode_combo.pack(side="left", padx=(0, 8))
    def start_selected_run():
        if app.is_running:
            return
        mode = app.run_mode_var.get()
        if mode == "Align & train":
            app._start_alignment(force_auto_train=True)
        elif mode == "Align only":
            app._start_alignment(skip_auto_train=True)
        elif mode == "Train only":
            app._start_train_only()
    app.one_click_btn = button(actions, "Start run", start_selected_run, primary=True)
    app.one_click_btn.pack(side="left", padx=(0, 8))
    app.one_click_btn.state_listener = lambda state: app.run_mode_combo.configure(
        state="disabled" if state == "disabled" else "readonly")
    syncing_mode = [False]
    def mode_changed(*_):
        if syncing_mode[0]:
            return
        syncing_mode[0] = True
        try:
            mode = app.run_mode_var.get()
            if mode != "Train only":
                app.auto_train_var.set(mode == "Align & train")
            app.one_click_btn.configure(text={"Align only": "Start alignment", "Train only": "Start training"}.get(mode, "Start run"))
        finally:
            syncing_mode[0] = False
    def auto_training_changed(*_):
        if not syncing_mode[0] and app.run_mode_var.get() != "Train only":
            app.run_mode_var.set("Align & train" if app.auto_train_var.get() else "Align only")
    app.run_mode_var.trace_add("write", mode_changed)
    app.auto_train_var.trace_add("write", auto_training_changed)
    mode_changed()
    # Existing engine lifecycle code still enables/disables these internal handles.
    app.start_btn = button(actions, "Align only", lambda: app._start_alignment(skip_auto_train=True))
    app.train_only_btn = button(actions, "Train only", app._start_train_only)
    app.stop_btn = button(actions, "Stop", app._stop_alignment, state="disabled")
    app.stop_btn.pack(side="right")
    app.save_finish_btn = button(actions, "Save & finish", app._save_and_finish_training, state="disabled")
    app.save_finish_btn.pack(side="right", padx=8)
    app.pause_btn = button(actions, "Pause", app._toggle_pause, state="disabled")
    app.pause_btn.pack(side="right", padx=(0, 8))

    # Dataset and profiles are shared across both workflow stages.
    project = tk.Frame(root, bg=SURFACE, padx=20, pady=10)
    project.pack(fill="x", padx=24, pady=(0, 12))
    row = tk.Frame(project, bg=SURFACE)
    row.pack(fill="x")
    label(row, "Dataset", 10, muted=True).pack(side="left", padx=(0, 16))
    app.dataset_var = tk.StringVar(value="")
    app.path_entry = tk.Entry(row, textvariable=app.dataset_var, font=("Segoe UI", 10))
    app.path_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
    button(row, "Browse folder", app._browse_dataset).pack(side="left")
    app.profile_panel = tk.Frame(project, bg=SURFACE)
    def toggle_profiles():
        shown = bool(app.profile_panel.winfo_manager())
        if shown:
            app.profile_panel.pack_forget()
        else:
            app.profile_panel.pack(fill="x", pady=(6, 0))
        app.profiles_button.configure(text="Hide profiles" if not shown else "Profiles ▾")
    app.toggle_profiles = toggle_profiles
    app.profiles_button = button(row, "Profiles ▾", toggle_profiles)
    app.profiles_button.pack(side="left", padx=(8, 0))
    app.dataset_status_lbl = label(project, "Choose a folder of photos or prepared training data.", muted=True, anchor="w")
    app.dataset_status_lbl.pack(fill="x", pady=(6, 8))
    project.bind("<Configure>", lambda e: app.dataset_status_lbl.configure(wraplength=max(200, e.width - 40)))
    row = tk.Frame(app.profile_panel, bg=SURFACE)
    row.pack(fill="x")
    label(row, "Profile", 10, muted=True).pack(side="left", padx=(0, 24))
    app.preset_combo = ttk.Combobox(row, textvariable=app.preset_var, width=23,
                                    state="readonly", postcommand=app._refresh_presets)
    app.preset_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
    app.preset_load_btn = button(row, "Load", app._load_preset)
    app.preset_load_btn.pack(side="left", padx=(0, 8))
    button(row, "Save as…", app._save_preset).pack(side="left")
    label(app.profile_panel, "Save a profile to reuse engine settings. Your dataset stays separate.",
          muted=True, anchor="w", wraplength=650).pack(fill="x", pady=(8, 0))
    app._refresh_presets()
    app.preset_var.trace_add("write", lambda *_: app.preset_load_btn.configure(
        state="normal" if app.preset_var.get() else "disabled"))
    app.preset_load_btn.configure(state="disabled")

    app.workspace_tabs = ttk.Notebook(root, takefocus=True)
    app.workspace_tabs.pack(fill="both", expand=True, padx=24)
    align_page = ScrollPage(app.workspace_tabs)
    train_page = ScrollPage(app.workspace_tabs)
    activity = tk.Frame(app.workspace_tabs, bg=SURFACE, padx=20, pady=16)
    app.workspace_tabs.add(align_page, text="01  Alignment")
    app.workspace_tabs.add(train_page, text="02  Training")
    app.workspace_tabs.add(activity, text="03  Activity")
    app.activity_tab = activity
    app.settings_pages = (align_page, train_page)
    app.workspace_tabs.enable_traversal()

    align = align_page.body
    section(align, "Alignment settings")
    engines = tk.Frame(align, bg=SURFACE)
    engines.pack(fill="x", pady=(0, 14))
    engine_controls = []
    for i, (name, value) in enumerate([("VGGT-Omega", "vggt_omega"), ("ACE0", "ace0"),
            ("PyCuSFM", "pycusfm"), ("OpenSfM", "opensfm"), ("360DVO", "360dvo")]):
        detail = {"vggt_omega": "Neural geometry", "ace0": "Pose refinement", "pycusfm": "GPU matching",
                  "opensfm": "Photos & panoramas", "360dvo": "360° sequences"}[value]
        control = tk.Radiobutton(engines, text=f"{name}\n{detail}", variable=app.engine_var, value=value,
                                  indicatoron=False, padx=14, pady=8, justify="left", anchor="w")
        control.grid(row=0, column=i, sticky="ew", padx=(0, 8))
        engines.columnconfigure(i, weight=1)
        engine_controls.append((control, app.engine_var))
    wrap_choices(engines)
    hints = {"vggt_omega": "General photos & drones · neural geometry prediction",
             "ace0": "Scene coordinate regression · camera pose refinement",
             "pycusfm": "Perspective photos · GPU feature matching",
             "opensfm": "360° panoramas & perspective photos · feature-based reconstruction",
             "360dvo": "Sequential 360° frames & video · visual odometry"}
    for control, variable in engine_controls:
        tooltip(control, hints.get(control.cget("value"), ""))
    general = tk.Frame(align, bg=SURFACE)
    general.pack(fill="x", pady=(0, 14))
    field(general, "Image resolution", app.res_var, ["512", "1024", "1536", "2048"])
    field(general, "Frame stride", app.stride_var, ["1", "2", "3", "4", "5"])
    app.align_dynamic_container = tk.Frame(align, bg=SURFACE)
    app.align_dynamic_container.pack(fill="both", expand=True)
    app._build_align_panels()
    cache = tk.Frame(align, bg=SURFACE)
    cache.pack(fill="x", pady=(10, 0))
    tk.Checkbutton(cache, text="Reuse cached results", variable=app.cache_var).pack(side="left")
    button(cache, "Clear cache", app._clear_dataset_cache).pack(side="left", padx=14)
    app.vggt_cache_frame = tk.Frame(cache, bg=SURFACE)
    for title, var, options in [("Chunk", app.chunk_size_var, ["16", "24", "32", "48", "64", "96", "128", "160", "192", "224", "256", "320", "384", "448", "512", "768", "1024"]),
                                ("Overlap", app.anchor_var, ["4", "8", "12", "16", "24", "32", "48", "64", "96", "128", "192", "256"])]:
        combo = field(app.vggt_cache_frame, title, var, options, width=6)
        combo.configure(state="normal")
    app.engine_var.trace_add("write", lambda *_: app._update_align_panel())
    app._update_align_panel()

    train = train_page.body
    section(train, "Training settings")
    engines = tk.Frame(train, bg=SURFACE)
    engines.pack(fill="x", pady=(0, 14))
    for i, (name, value) in enumerate([("LichtFeld Studio", "lichtfeld"), ("Splatfacto", "splatfacto"),
                                      ("3DGRUT", "3dgrut"), ("PFGS360", "pfgs360")]):
        detail = {"lichtfeld": "Live Vulkan viewer", "splatfacto": "Nerfstudio workflow",
                  "3dgrut": "Ray-traced optics", "pfgs360": "Panoramic scenes"}[value]
        control = tk.Radiobutton(engines, text=f"{name}\n{detail}", variable=app.train_engine_var, value=value,
                                  indicatoron=False, padx=14, pady=8, justify="left", anchor="w")
        control.grid(row=0, column=i, sticky="ew", padx=(0, 8))
        engines.columnconfigure(i, weight=1)
        engine_controls.append((control, app.train_engine_var))
    wrap_choices(engines, 190)
    general = tk.Frame(train, bg=SURFACE)
    general.pack(fill="x", pady=(0, 16))
    field(general, "Training iterations", app.train_iter_var,
          ["7,000 (Fast Preview)", "15,000 (Balanced / Standard)", "30,000 (Production High-Fidelity)"], width=32)
    app.train_res_cb = field(general, "Training image resolution", app.train_res_var,
          ["Original Native (100% Full Res)", "2K Downscale (1920px Max)", "1K Downscale (1024px Max)", "4K Downscale (3840px Max)"], width=28)
    tooltip(app.train_res_cb, "Training Image Resolution:\n• Original Native (100% Full Res) [Default]: Preserves source photo resolution with zero downscaling.\n• 2K Downscale (1920px Max): Non-destructive runtime resizing for fast VRAM-efficient training.\n• 1K Downscale (1024px Max): Maximum training speed and low VRAM footprint.\n• 4K Downscale (3840px Max): High resolution cap.")
    app.train_max_gaussians_cb = field(general, "Max Gaussians", app.train_max_gaussians_var,
          ["Unlimited (Engine Default)", "500,000 (Fast / Low VRAM)", "1,000,000 (Standard)", "2,000,000 (Detailed)", "3,000,000 (High Quality)", "5,000,000 (Maximum Fidelity)"], width=26)
    tooltip(app.train_max_gaussians_cb, "Max Gaussians Cap:\nSet the maximum number of Gaussians allowed during training and model export across all 4 training engines.\n• Unlimited (Engine Default): Uses each engine's default capacity.\n• 500,000: Fast training, ultra-compact export, minimal VRAM usage.\n• 1,000,000: Standard balance for most objects and scenes.\n• 2,000,000 - 5,000,000: High fidelity for large environments.")
    app.train_target_run_cb = field(general, "Alignment run target", app.train_target_run_var,
          ["Latest Run (Auto-Detect)"], width=26)
    tooltip(app.train_target_run_cb, "Alignment Output Run Target:\nSelect which alignment run in results/ to use for 3DGS training.\n• Latest Run (Auto-Detect): Automatically detects and uses the newest sparse_* alignment.\n• Specific run: Explicitly targets that alignment run.")
    app.training_summary = general
    for child in general.winfo_children():
        child.pack_forget()
    wrap_choices(general, max(child.winfo_reqwidth() for child in general.winfo_children()) + 40)
    app.train_dynamic_container = tk.Frame(train, bg=SURFACE)
    app.train_dynamic_container.pack(fill="both", expand=True)
    app._build_train_panels()
    app.train_engine_var.trace_add("write", lambda *_: app._update_train_panel())
    app._update_train_panel()

    toolbar = tk.Frame(activity, bg=SURFACE)
    toolbar.pack(fill="x", pady=(0, 14))
    label(toolbar, "Activity", 16).pack(side="left", padx=(0, 6))
    for name, attr, callback in [("Results", "open_folder_btn", app._open_results_folder),
            ("Models", "models_btn", app._open_models_folder), ("Logs", "logs_btn", app._open_logs_folder),
            ("3D viewer", "viewer_btn", app._launch_viewer)]:
        control = button(toolbar, name, callback)
        control.pack(side="left", padx=(6, 0))
        setattr(app, attr, control)
    button(toolbar, "Monitor", lambda: open_monitor(app)).pack(side="left", padx=(6, 0))
    app.follow_log_var = tk.BooleanVar(value=True)
    tk.Checkbutton(toolbar, text="Follow log", variable=app.follow_log_var).pack(side="right")
    stages = tk.Frame(activity, bg=SURFACE)
    stages.pack(fill="x", pady=(0, 14))
    app.stage_badges = []
    for i, name in enumerate(["Scan", "Initialize", "Preprocess", "Infer", "Track & refine", "Train"]):
        badge = label(stages, f"{i + 1}  {name}", 9, muted=True, padx=8, pady=6)
        badge.grid(row=0, column=i, sticky="ew", padx=(0, 4))
        stages.columnconfigure(i, weight=1)
        app.stage_badges.append(badge)
    app.hw_bar = tk.Frame(activity, bg=SURFACE)
    app.hw_bar.pack(fill="x", pady=(0, 12))
    for i, (attr, text) in enumerate([("hw_cpu_lbl", "CPU: —"), ("hw_ram_lbl", "RAM: —"),
            ("hw_gpu_lbl", "GPU: —"), ("hw_temp_lbl", "GPU Temp: —"),
            ("hw_vram_lbl", "VRAM: —"), ("hw_page_lbl", "Paging: —")]):
        item = label(app.hw_bar, text, 9, muted=True, anchor="w", padx=10, pady=6)
        item.grid(row=i // 3, column=i % 3, sticky="ew", padx=(0, 6), pady=2)
        app.hw_bar.columnconfigure(i % 3, weight=1)
        setattr(app, attr, item)
    logbar = tk.Frame(activity, bg=SURFACE)
    logbar.pack(fill="x", pady=(0, 8))
    app.log_status_lbl = label(logbar, "Session log · saved automatically when a run starts", 9, muted=True)
    app.log_status_lbl.pack(side="left", fill="x", expand=True)
    button(logbar, "Open latest log", app._open_latest_log).pack(side="right")
    app.console = ScrolledText(activity, bg=BG, fg=MUTED, insertbackground=TEXT,
                               font=("Consolas", 10), relief="flat", bd=0,
                               padx=14, pady=12, height=8, wrap="word", state="disabled")
    app.console.pack(fill="both", expand=True)
    app.console.vbar.configure(bg=BORDER, troughcolor=BG, activebackground=MUTED, relief="flat", bd=0)
    app.log_search = LogSearch(activity, app.console, app.follow_log_var)
    button(toolbar, "Find", app.log_search.show).pack(side="left", padx=(6, 0))
    for child in toolbar.winfo_children():
        child.pack_forget()
    wrap_choices(toolbar, max(child.winfo_reqwidth() for child in toolbar.winfo_children()) + 16)
    tooltip(app.log_search.entry, "Search retained output. Enter: next; Shift+Enter: previous; Escape: close. Enable Follow log to resume live scrolling.")
    def show_log_search(event=None):
        app.workspace_tabs.select(app.activity_tab)
        app.log_search.show()
        return "break"
    root.bind("<Control-f>", show_log_search)
    app._log("Ready. Choose a dataset, configure alignment and training, then start a run.\n")

    compact = [None]
    def activity_layout(event):
        small = event.height < 520
        if small == compact[0]:
            return
        compact[0] = small
        for widget in (stages, app.hw_bar, logbar):
            widget.pack_forget()
        if not small:
            for widget in (stages, app.hw_bar, logbar):
                widget.pack(fill="x", pady=(0, 12), before=app.console.frame)
    activity.bind("<Configure>", activity_layout)

    polish_controls(root)
    def update_selection(*_):
        for control, variable in engine_controls:
            selected = variable.get() == control.cget("value")
            control.configure(selectcolor="#24483f", fg=ACCENT if selected else MUTED,
                              highlightbackground=ACCENT if selected else BORDER,
                              bg=FIELD, activebackground=BORDER, pady=10)
    app.engine_var.trace_add("write", update_selection)
    app.train_engine_var.trace_add("write", update_selection)
    update_selection()
    for container in (app.align_dynamic_container, app.train_dynamic_container):
        for panel in container.winfo_children():
            responsive_groups(panel)
    for page in app.settings_pages:
        page.bind_navigation()
    # Tooltips remain available on engine parameters; keyboard users can reveal them too.
    root.bind("<Control-o>", lambda e: app._browse_dataset())
    root.bind("<Control-Shift-S>", lambda e: app._save_preset())
    for i in range(3):
        root.bind(f"<Alt-Key-{i + 1}>", lambda e, index=i: app.workspace_tabs.select(index))
    def on_map(event):
        if (isinstance(event.widget, tk.Toplevel) and not event.widget.overrideredirect()
                and not getattr(event.widget, "_custom_theme", False)):
            polish_dialog(event.widget)
    root.bind_class("Toplevel", "<Map>", on_map, add="+")
    for widget, text in [
        (app.one_click_btn, "Start the mode selected beside this button."),
        (app.run_mode_combo, "Choose alignment plus training, alignment alone, or training alone. Locked while a run is active."),
        (app.start_btn, "Run alignment without training, regardless of the automatic training setting."),
        (app.train_only_btn, "Train from an existing reconstruction or a supported pose-free input."),
        (app.preset_combo, "No saved profiles yet? Choose Save as. Profiles preserve engine settings, not the dataset."),
        (app.path_entry, "Choose a dataset folder. Shortcut: Ctrl+O."),
        (app.pause_btn, "Pause or resume the current process. Its GPU memory remains allocated."),
        (app.save_finish_btn, "Stop training anytime and preserve current Gaussians, exporting the final 3DGS .ply model immediately."),
        (app.stop_btn, "Stop and abort the active process immediately."),
    ]:
        tooltip(widget, text)
