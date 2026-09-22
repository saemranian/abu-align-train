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


def label(parent, text, size=10, muted=False, **kwargs):
    return tk.Label(parent, text=text, bg=parent.cget("bg"),
                    fg=MUTED if muted else TEXT, font=("Segoe UI", size), **kwargs)


def button(parent, text, command, primary=False, **kwargs):
    widget = tk.Button(parent, text=text, command=command,
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
              foreground=[("selected", ACCENT), ("active", TEXT)])
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
            widget.configure(bg=FIELD, fg=TEXT, insertbackground=TEXT,
                             relief="flat", highlightthickness=1,
                             highlightbackground=BORDER, highlightcolor=ACCENT)
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
                widget.configure(bg=FIELD, fg=TEXT, insertbackground=TEXT,
                                 relief="flat", highlightthickness=1,
                                 highlightbackground=BORDER, highlightcolor=ACCENT)
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

    def bind_navigation(self):
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
        walk(self.body)
        self.canvas.bind("<MouseWheel>", wheel)


def responsive_groups(panel):
    """Wrap whole parameter groups without changing their internal controls."""
    groups = list(panel.winfo_children())
    for group in groups:
        group.pack_forget()
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
            group.grid(row=i // columns, column=i % columns, sticky="nw",
                       padx=(0, 24), pady=(6, 22))
    panel.bind("<Configure>", arrange)


def section(parent, title, subtitle=""):
    label(parent, title, 16).pack(anchor="w", pady=(0, 5))
    if subtitle:
        label(parent, subtitle, muted=True).pack(anchor="w", pady=(0, 18))


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
    dialog.geometry("560x360")
    dialog.transient(app.root)
    label(dialog, "Live hardware", 18).pack(anchor="w", padx=22, pady=18)
    sources = [app.hw_cpu_lbl, app.hw_ram_lbl, app.hw_gpu_lbl,
               app.hw_temp_lbl, app.hw_vram_lbl, app.hw_page_lbl]
    labels = []
    for source in sources:
        item = label(dialog, "", anchor="w")
        item.pack(fill="x", padx=22, pady=4)
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


def build_workspace(app, tooltip):
    configure_theme(app)
    root = app.root
    # Header: identity on the left, secondary destinations on the right.
    header = tk.Frame(root, bg=BG, padx=24, pady=18)
    header.pack(fill="x")
    mark = tk.Label(header, text="A", bg=ACCENT, fg=BG, font=("Segoe UI", 19, "bold"), padx=10)
    mark.pack(side="left", padx=(0, 12))
    label(header, "Abu Align & Train", 19).pack(side="left")
    label(header, "v0.0.3.7", 9, muted=True).pack(side="left", padx=12)
    button(header, "User guide", app._open_user_guide).pack(side="right")

    # Pin run controls to the bottom before allocating expandable notebook space.
    footer = tk.Frame(root, bg=BG, padx=24, pady=12)
    footer.pack(side="bottom", fill="x")
    status = tk.Frame(footer, bg=BG)
    status.pack(fill="x", pady=(0, 8))
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
    app.one_click_btn = button(actions, "Align & train", app._start_alignment, primary=True)
    app.auto_train_var.trace_add("write", lambda *_: app.one_click_btn.configure(
        text="Align & train" if app.auto_train_var.get() else "Start alignment"))
    app.start_btn = button(actions, "Align only", lambda: app._start_alignment(skip_auto_train=True))
    app.train_only_btn = button(actions, "Train only", app._start_train_only)
    for widget in (app.one_click_btn, app.start_btn, app.train_only_btn):
        widget.pack(side="left", padx=(0, 8))
    app.stop_btn = button(actions, "Stop", app._stop_alignment, state="disabled")
    app.stop_btn.pack(side="right")
    app.pause_btn = button(actions, "Pause", app._toggle_pause, state="disabled")
    app.pause_btn.pack(side="right", padx=8)

    # Dataset and profiles are shared across both workflow stages.
    project = tk.Frame(root, bg=SURFACE, padx=20, pady=14)
    project.pack(fill="x", padx=24, pady=(0, 12))
    row = tk.Frame(project, bg=SURFACE)
    row.pack(fill="x")
    label(row, "DATASET", 9, muted=True).pack(side="left", padx=(0, 16))
    app.dataset_var = tk.StringVar(value="")
    app.path_entry = tk.Entry(row, textvariable=app.dataset_var, font=("Segoe UI", 10))
    app.path_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
    button(row, "Browse folder", app._browse_dataset).pack(side="left")
    app.dataset_status_lbl = label(project, "Choose a folder of photos or prepared training data.", muted=True, anchor="w")
    app.dataset_status_lbl.pack(fill="x", pady=(8, 12))
    project.bind("<Configure>", lambda e: app.dataset_status_lbl.configure(wraplength=max(200, e.width - 40)))
    row = tk.Frame(project, bg=SURFACE)
    row.pack(fill="x")
    label(row, "PRESET", 9, muted=True).pack(side="left", padx=(0, 24))
    app.preset_combo = ttk.Combobox(row, textvariable=app.preset_var, width=23,
                                    state="readonly", postcommand=app._refresh_presets)
    app.preset_combo.pack(side="left", fill="x", expand=True, padx=(0, 8))
    app.preset_load_btn = button(row, "Load", app._load_preset)
    app.preset_load_btn.pack(side="left", padx=(0, 8))
    button(row, "Save as…", app._save_preset).pack(side="left")
    button(row, "360° splitter", app._open_panoramic_splitter_dialog).pack(side="right", padx=(12, 0))
    button(row, "Video frames", app._open_video_extractor_dialog).pack(side="right", padx=(12, 0))
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
    section(align, "Reconstruct your scene", "Choose an alignment engine, then refine its settings below.")
    engines = tk.Frame(align, bg=SURFACE)
    engines.pack(fill="x", pady=(0, 14))
    engine_controls = []
    for i, (name, value) in enumerate([("VGGT-Omega", "vggt_omega"), ("ACE0", "ace0"),
            ("PyCuSFM", "pycusfm"), ("OpenSfM", "opensfm"), ("360DVO", "360dvo")]):
        control = tk.Radiobutton(engines, text=name, variable=app.engine_var, value=value,
                                  indicatoron=False, padx=14, pady=8)
        control.grid(row=0, column=i, sticky="ew", padx=(0, 8))
        engines.columnconfigure(i, weight=1)
        engine_controls.append((control, app.engine_var))
    alignment_hint = label(align, "", muted=True, anchor="w")
    alignment_hint.pack(fill="x", pady=(0, 10))
    alignment_hint.bind("<Configure>", lambda e: alignment_hint.configure(wraplength=e.width))
    hints = {"vggt_omega": "General photos & drones · neural geometry prediction",
             "ace0": "Scene coordinate regression · camera pose refinement",
             "pycusfm": "Perspective photos · GPU feature matching",
             "opensfm": "360° panoramas & perspective photos · feature-based reconstruction",
             "360dvo": "Sequential 360° frames & video · visual odometry"}
    def update_hint(*_):
        alignment_hint.configure(text=hints.get(app.engine_var.get(), ""))
    app.engine_var.trace_add("write", update_hint)
    update_hint()
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
    section(train, "Train a Gaussian scene", "Use aligned camera poses or a supported pose-free workflow.")
    engines = tk.Frame(train, bg=SURFACE)
    engines.pack(fill="x", pady=(0, 14))
    for i, (name, value) in enumerate([("LichtFeld Studio", "lichtfeld"), ("Splatfacto", "splatfacto"),
                                      ("3DGRUT", "3dgrut"), ("PFGS360", "pfgs360")]):
        control = tk.Radiobutton(engines, text=name, variable=app.train_engine_var, value=value,
                                  indicatoron=False, padx=14, pady=8)
        control.grid(row=0, column=i, sticky="ew", padx=(0, 8))
        engines.columnconfigure(i, weight=1)
        engine_controls.append((control, app.train_engine_var))
    general = tk.Frame(train, bg=SURFACE)
    general.pack(fill="x", pady=(0, 16))
    field(general, "Training iterations", app.train_iter_var,
          ["7,000 (Fast Preview)", "15,000 (Balanced / Standard)", "30,000 (Production High-Fidelity)"], width=34)
    tk.Checkbutton(general, text="Train after alignment", variable=app.auto_train_var).pack(side="left", padx=14)
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
    app._log("Ready. Choose a dataset, configure alignment and training, then start a run.\n")

    compact = [None]
    def activity_layout(event):
        small = event.height < 480
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
        if isinstance(event.widget, tk.Toplevel) and not event.widget.overrideredirect():
            polish_dialog(event.widget)
    root.bind_class("Toplevel", "<Map>", on_map, add="+")
    for widget, text in [
        (app.one_click_btn, "Run the configured workflow. The Training tab controls automatic training."),
        (app.start_btn, "Run alignment without training, regardless of the automatic training setting."),
        (app.train_only_btn, "Train from an existing reconstruction or a supported pose-free input."),
        (app.preset_combo, "No saved profiles yet? Choose Save as. Profiles preserve engine settings, not the dataset."),
        (app.path_entry, "Choose a dataset folder. Shortcut: Ctrl+O."),
        (app.pause_btn, "Pause or resume the current process. Its GPU memory remains allocated."),
        (app.stop_btn, "Stop the active process."),
    ]:
        tooltip(widget, text)
