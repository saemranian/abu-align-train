"""Progressive disclosure and viewport controls for the existing rig workflow."""
import math
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from desktop_ui import BG, SURFACE, FIELD, TEXT, MUTED, ACCENT, BORDER, button, label, polish_controls


def wrap_row(row):
    children = list(row.winfo_children())
    for widget in children:
        widget.pack_forget()
    previous = [None]
    def arrange(event=None):
        available = max(1, row.winfo_width() - 2 * row.winfo_pixels(row.cget("padx")))
        signature = (available, tuple((w.winfo_reqwidth(), w.winfo_reqheight()) for w in children))
        if signature == previous[0]:
            return
        previous[0] = signature
        x, line = 0, 0
        for widget in children:
            width = widget.winfo_reqwidth() + 8
            if x and x + width > available:
                line, x = line + 1, 0
            # Insets keep each line independent while grid tracks requested sizes.
            widget.grid(row=line, column=0, sticky="w", padx=(x, 8), pady=5)
            x += width
    row.bind("<Configure>", arrange, add="+")
    for widget in children:
        widget.bind("<Configure>", arrange, add="+")


def friendly_choices(combo, variable, options):
    display = tk.StringVar(value=options[variable.get()])
    combo.configure(textvariable=display, values=list(options.values()))
    reverse = {label: value for value, label in options.items()}
    combo.bind("<<ComboboxSelected>>", lambda e: variable.set(reverse[display.get()]))
    variable.trace_add("write", lambda *_: display.set(options[variable.get()]))
    combo.display_variable = display


def enhance_rig_designer(app, ctx):
    dialog, body, left, right = [ctx[k] for k in ("dialog", "body", "left_col", "right_col")]
    dialog.title("360° Rig Designer · Abu Align")
    dialog.geometry(f"{min(1500, dialog.winfo_screenwidth()-80)}x{min(1040, dialog.winfo_screenheight()-120)}")
    dialog.minsize(1120, 760)
    dialog._custom_theme = True
    # Keep actions pinned even when a smaller viewport requires scrolling.
    body.pack_forget()
    ctx["bottom_bar"].pack_forget()
    ctx["bottom_bar"].pack(side="bottom", fill="x", padx=16, pady=(0, 12))
    body.pack(fill="both", expand=True, padx=8, pady=(4, 10))
    left.pack_propagate(False)
    left.configure(width=620, bg=BG)
    left.pack_configure(expand=False, padx=(0, 14))
    right.configure(bg=BG)
    right.pack_configure(expand=True, padx=0)

    for child in ctx["hdr_left"].winfo_children():
        child.destroy()
    label(ctx["hdr_left"], "360° Rig Designer", 20).pack(anchor="w")
    label(ctx["hdr_left"], "Compose your cameras. Inspect coverage. Export ready-to-use views.", muted=True).pack(anchor="w", pady=(5, 0))

    notebook = ctx["rig_notebook"]
    pages = ctx["rig_pages"]
    for page, title, card in zip(pages, ["Source", "Rig", "Export"],
                                 [ctx["paths_card"], ctx["rig_card"], ctx["optics_card"]]):
        card.pack_forget()
        card.configure(text="", bd=0, padx=0, pady=0)
        card.pack(fill="x")
    notebook.select(1 if ctx["in_var"].get() else 0)
    notebook.enable_traversal()
    # Reserve browse actions before allowing long source paths to fill the row.
    for row_name in ("r1", "r2", "r3", "r4"):
        if row_name in ctx:
            row = ctx[row_name]
            row.columnconfigure(1, weight=1)
            children = row.winfo_children()
            for widget in children:
                widget.pack_forget()
            for column, widget in enumerate(children):
                if isinstance(widget, tk.Entry):
                    widget.configure(width=1)
                widget.grid(row=0, column=column, sticky="ew", padx=(0, 6), pady=3)
    friendly_choices(ctx["mode_cb"], ctx["mode_var"],
                     {"ring": "Ring", "cubemap": "Cubemap", "dome": "Dome", "fibonacci": "Fibonacci sphere"})
    friendly_choices(ctx["solver_cb"], ctx["solver_mode_var"],
                     {"free": "Manual", "lock_overlap": "Keep overlap", "lock_fov": "Keep field of view"})
    ctx["solver_cb"].configure(width=17)
    for widget in ctx["m_row"].winfo_children():
        if isinstance(widget, tk.Label):
            widget.configure(text="Layout" if "Architecture" in widget.cget("text") else "Adjustments",
                             width=11, anchor="w")
    for widget in ctx["m_row"].winfo_children():
        widget.pack_forget()
    for i, widget in enumerate(ctx["m_row"].winfo_children()):
        widget.grid(row=i//2, column=i%2, sticky="w", pady=7, padx=(0, 10))
    for name in ("o1", "o3", "o4", "dome_row1", "video_card_frame"):
        if name in ctx and name != "video_card_frame":  # This row's entry changes between FPS/stride via pack.
            wrap_row(ctx[name])
    for widget in ctx["video_card_frame"].winfo_children():
        if isinstance(widget, tk.Label):
            if widget.cget("text") == "Video Sampling:":
                widget.configure(text="Video")
            elif widget.cget("text") == "Max Frames:":
                widget.configure(text="Limit")
    # Long source fields stretch; vertical space belongs to the currently relevant step.
    for page, title, detail in zip(pages, ["Choose your panoramas", "Camera arrangement", "Output & sampling"],
                                  ["Start with 360° images or a panoramic video.",
                                   "Adjust the rig while watching the live preview.",
                                   "Set image size, optics, and video sampling."]):
        heading = label(page.body, title, 15)
        heading.pack(before=page.body.pack_slaves()[0], anchor="w", pady=(0, 7))
        description = label(page.body, detail, muted=True, wraplength=460, justify="left")
        description.pack(after=heading, anchor="w", pady=(0, 18))
    for row_name in ("cams_row", "fov_row", "ol_row", "o2"):
        if row_name not in ctx:
            continue
        row = ctx[row_name]
        row.pack_configure(pady=6)
        for widget in row.winfo_children():
            if isinstance(widget, tk.Scale):
                widget.configure(bd=0, sliderrelief="flat", sliderlength=20, width=8,
                                 bg=ACCENT, activebackground="#9ce8d3", troughcolor=FIELD)
            elif isinstance(widget, tk.Checkbutton):
                widget.configure(bg=BG, fg=MUTED, selectcolor=FIELD, activebackground=BG,
                                 activeforeground=ACCENT, bd=0, highlightthickness=0)
            elif isinstance(widget, tk.Frame):
                widget.configure(bg=row.cget("bg"))
                for child in widget.winfo_children():
                    if isinstance(child, tk.Entry):
                        child.configure(bg=FIELD, fg=TEXT, insertbackground=ACCENT, selectbackground=BORDER, relief="flat", bd=0)
                    elif isinstance(child, tk.Label):
                        if child == ctx.get("fov_val_lbl"):
                            child.configure(bg=FIELD, fg=ACCENT, padx=6, pady=2)
                        elif child == ctx.get("dist_val_lbl"):
                            child.configure(fg=ACCENT)
                        else:
                            child.configure(fg=MUTED)
    for name, title in [("cams_lbl", "Cameras"), ("fov_lbl", "Field of view"), ("ol_lbl", "Overlap"), ("dist_lbl", "Distortion")]:
        if name in ctx:
            ctx[name].configure(text=title, width=11, fg=MUTED)

    # A dedicated view strip stops controls disappearing beyond the viewport edge.
    for child in ctx["viz_hdr"].winfo_children():
        if child is not ctx["vm_frame"]:
            child.destroy()
    modes = ctx["vm_frame"]
    modes.pack_configure(side="left", fill="x", expand=True)
    for control, title in [(ctx["orbit3d_btn"], "Orbit"), (ctx["polar_btn"], "Polar"), (ctx["equirect_btn"], "Map")]:
        control.configure(text=title, indicatoron=False, selectcolor="#24483f", fg=ACCENT,
                          bg=FIELD, activebackground=BORDER, bd=0, padx=14, pady=8)
    ctx["reset_3d_btn"].pack_forget()
    ctx["auto_spin_btn"].pack_forget()
    view_tools = tk.Frame(right, bg=SURFACE, padx=10, pady=6)
    view_tools.pack(fill="x", before=ctx["canvas_card"])
    renderer = ctx["renderer"]
    orbit, redraw = ctx["orbit_state"], ctx["draw_visualizer"]
    focused = tk.BooleanVar(value=False)
    def focus_view(restore=False):
        focused.set(False if restore else not focused.get())
        if focused.get():
            left.pack_forget()
        else:
            left.pack(side="left", fill="both", expand=False, padx=(0, 14), before=right)
        focus_button.configure(text="Show settings" if focused.get() else "Focus view")
        dialog.after_idle(redraw)
    focus_button = button(view_tools, "Focus view", focus_view)
    focus_button.pack(side="right")
    focus_button.configure(pady=4)
    def view(yaw, pitch):
        ctx["canvas_view_mode_var"].set("3d")
        orbit.update(yaw=yaw, pitch=pitch, pan_x=0, pan_y=0, dist=3.4)
        redraw()
    for title, command in [("Fit", ctx["reset_3d_view"]), ("Top", lambda: view(0, 88)),
                            ("Front", lambda: view(0, 0)), ("Rotate", ctx["toggle_auto_orbit"])]:
        if title == "Rotate":
            ctx["auto_spin_btn"].configure(text="Rotate")
            ctx["auto_spin_btn"].pack(side="right", padx=(6, 0))
        else:
            control = button(view_tools, title, command)
            control.configure(pady=4)
            control.pack(side="left", padx=(0, 6))
    # Use compact controls in a separate row, with no effect on exported camera settings.
    toggles = tk.Frame(right, bg=SURFACE, padx=10, pady=6)
    display_open = tk.BooleanVar(value=False)
    def toggle_display():
        display_open.set(not display_open.get())
        if display_open.get():
            toggles.pack(fill="x", before=ctx["canvas_card"])
        else:
            toggles.pack_forget()
        display_button.configure(text="Hide display" if display_open.get() else "Display")
    display_button = button(view_tools, "Display", toggle_display)
    display_button.configure(pady=4, padx=8)
    display_button.pack(side="left")
    controls_guide = tk.Frame(right, bg=SURFACE, padx=12, pady=10,
                             highlightthickness=1, highlightbackground=BORDER)
    guide_text = label(controls_guide,
        "Orbit: drag or arrow keys · Pan: Shift-drag or right-drag\n"
        "Zoom: scroll or + / − · Reset: F or double-click\n"
        "Inspect a camera: click its marker, use Previous / Next, or [ / ]",
        muted=True, justify="left", anchor="w")
    guide_text.pack(fill="x")
    controls_guide.bind("<Configure>", lambda e: guide_text.configure(wraplength=max(180, e.width-24)))
    def toggle_guide():
        shown = bool(controls_guide.winfo_manager())
        if shown:
            controls_guide.place_forget()
        else:
            # Help is a temporary overlay; it must never collapse the 3D scene.
            controls_guide.place(in_=ctx["canvas_card"], x=8, y=8, relwidth=1, width=-16)
            controls_guide.lift()
        guide_button.configure(text="Hide controls" if not shown else "Controls")
    guide_button = button(view_tools, "Controls", toggle_guide)
    guide_button.configure(pady=4, padx=8)
    guide_button.pack(side="left")

    if "toggle_log_monitor" in ctx:
        orig_toggle_log = ctx["toggle_log_monitor"]
        log_button = button(view_tools, "Logs", None)
        log_button.configure(pady=4, padx=8)
        log_button.pack(side="left")

        def sync_log_button():
            t_btn = ctx.get("toggle_log_btn")
            is_open = "▲" in (t_btn.cget("text") if t_btn else "")
            log_button.configure(text="Hide logs" if is_open else "Logs",
                                 fg=ACCENT if is_open else TEXT)

        def handle_toggle_log():
            orig_toggle_log()
            sync_log_button()

        log_button.configure(command=handle_toggle_log)
        if "toggle_log_btn" in ctx and ctx["toggle_log_btn"]:
            ctx["toggle_log_btn"].configure(command=handle_toggle_log)
        ctx["log_button"] = log_button
    def dismiss_view_help(event):
        if controls_guide.winfo_manager():
            toggle_guide()
        elif focused.get():
            focus_view(True)
        return "break"
    dialog.bind("<Escape>", dismiss_view_help, add="+")
    wrap_row(view_tools)
    controls = {}
    for title, attribute, initial in [("All coverage", "show_all", False), ("Labels", "show_labels", True),
                                      ("Grid", "show_grid", True)]:
        var = tk.BooleanVar(value=initial)
        controls[attribute] = var
        def toggle(attr=attribute, value=var):
            setattr(renderer, attr, value.get())
            redraw()
        tk.Checkbutton(toggles, text=title, variable=var, command=toggle,
                       bg=SURFACE, fg=MUTED, selectcolor=BG, activebackground=SURFACE,
                       activeforeground=ACCENT, bd=0, highlightthickness=0).pack(side="left", padx=(0, 14))

    show_35mm_var = ctx.get("show_35mm_eq_var")
    if show_35mm_var is not None:
        def toggle_35mm():
            if "sync_math_and_ui" in ctx:
                ctx["sync_math_and_ui"]()
            inspect_camera()
        tk.Checkbutton(toggles, text="35mm Eq", variable=show_35mm_var, command=toggle_35mm,
                       bg=SURFACE, fg=MUTED, selectcolor=BG, activebackground=SURFACE,
                       activeforeground=ACCENT, bd=0, highlightthickness=0).pack(side="left", padx=(0, 14))

    ctx["viz_canvas"].configure(bg="#11191e", takefocus=True)
    ctx["viz_canvas"].configure(highlightthickness=1, highlightbackground="#11191e", highlightcolor=BORDER)
    ctx["canvas_card"].configure(bg="#11191e", highlightthickness=0)
    inspector = tk.Frame(right, bg=SURFACE, padx=12, pady=10)
    inspector.pack(fill="x", after=ctx["canvas_card"])
    camera_text = label(inspector, "", muted=True, anchor="w", justify="left")
    camera_text.pack(side="left", fill="x", expand=True)
    def select_camera(step):
        rig = ctx["get_current_rig"]()
        renderer.selected = (renderer.selected + step) % max(1, len(rig))
        ctx["canvas_view_mode_var"].set("3d")
        redraw()
    next_button = button(inspector, "Next →", lambda: select_camera(1))
    next_button.pack(side="right")
    camera_number = tk.StringVar(master=dialog)
    camera_picker = ttk.Combobox(inspector, textvariable=camera_number, state="readonly", width=4)
    camera_picker.pack(side="right", padx=6)
    def choose_camera(event=None):
        rig = ctx["get_current_rig"]()
        try:
            index = int(camera_number.get()) - 1
        except ValueError:
            return
        if rig:
            renderer.selected = max(0, min(index, len(rig)-1))
            redraw()
    camera_picker.bind("<<ComboboxSelected>>", choose_camera)
    previous_button = button(inspector, "← Previous", lambda: select_camera(-1))
    previous_button.pack(side="right", padx=6)
    for control in (next_button, previous_button):
        control.configure(padx=8, pady=4)
    def inspect_camera(event=None):
        rig = ctx["get_current_rig"]()
        choices = tuple(f"{i+1:02d}" for i in range(len(rig)))
        if tuple(camera_picker.cget("values")) != choices:
            camera_picker.configure(values=choices)
        camera_picker.configure(state="readonly" if rig else "disabled")
        if rig:
            index = min(renderer.selected, len(rig)-1)
            camera_number.set(choices[index])
            name, yaw, pitch, roll, fov = rig[index]
            show_eq = ctx.get("show_35mm_eq_var")
            eq_text = ""
            if show_eq is None or show_eq.get():
                try:
                    f_eq = 18.0 / math.tan(math.radians(max(1.0, min(170.0, float(fov)))) / 2.0)
                    eq_text = f" ({f_eq:.1f}mm)"
                except Exception:
                    pass
            camera_text.configure(text=f"CAMERA {index+1:02d} / {len(rig):02d}\nYaw {yaw:g}° · Pitch {pitch:g}° · FOV {fov:g}°{eq_text}")
    ctx["viz_canvas"].bind("<<RigRendered>>", inspect_camera)
    def sync_view(*_):
        is_orbit = ctx["canvas_view_mode_var"].get() == "3d"
        for control in toggles.winfo_children():
            control.configure(state="normal" if is_orbit else "disabled")
        if is_orbit:
            inspector.pack(fill="x", after=ctx["canvas_card"])
        else:
            inspector.pack_forget()
        for control, mode in [(ctx["orbit3d_btn"], "3d"), (ctx["polar_btn"], "polar"),
                              (ctx["equirect_btn"], "equirect")]:
            control.configure(fg=ACCENT if ctx["canvas_view_mode_var"].get() == mode else MUTED)
    ctx["canvas_view_mode_var"].trace_add("write", sync_view)
    sync_view()
    inspector.bind("<Configure>", lambda e: camera_text.configure(wraplength=max(100,
        e.width-next_button.winfo_reqwidth()-previous_button.winfo_reqwidth()-camera_picker.winfo_reqwidth()-60)))
    ctx["telemetry_card"].configure(text="Rig summary", fg=MUTED, bd=0)
    ctx["t_c1"].pack_configure(side="top", fill="x", expand=False)
    ctx["t_c2"].pack_configure(side="top", fill="x", expand=False)
    # Dense diagnostics are available when needed, leaving the scene room to breathe.
    summary_visible = tk.BooleanVar(value=False)
    summary_button = button(modes, "Details", lambda: toggle_summary())
    summary_button.pack(side="right")
    ctx["telemetry_card"].pack_forget()
    def toggle_summary():
        summary_visible.set(not summary_visible.get())
        if summary_visible.get():
            ctx["telemetry_card"].pack(fill="x")
        else:
            ctx["telemetry_card"].pack_forget()
        summary_button.configure(text="Hide details" if summary_visible.get() else "Details")

    ctx["split_btn"].configure(text="Export views", bg=ACCENT, fg=BG, activebackground="#9ce8d3")
    ctx["use_ds_btn"].configure(text="Use dataset")
    ctx["open_folder_btn"].configure(text="Open output")
    ctx["dlg_pbar"].configure(style="Abu.Horizontal.TProgressbar")
    ctx["status_lbl"].configure(text="Ready to export. Choose a source in the Source tab.")
    if not ctx["in_var"].get():
        ctx["media_badge_lbl"].configure(text="No source selected yet.", fg=MUTED)
    ctx["tf_chk"].configure(text="Export camera transforms", fg=MUTED)
    if "rc_csv_chk" in ctx:
        ctx["rc_csv_chk"].configure(text="Export RealityScan cameras.csv", fg=MUTED)
    if "subfolder_chk" in ctx:
        ctx["subfolder_chk"].configure(text="Folder per perspective (with cameras.csv)", fg=MUTED)
    ctx["auto_res_btn"].configure(text="Match source")

    def theme(widget):
        if widget == ctx.get("log_monitor_frame"):
            return
        for child in widget.winfo_children():
            if child == ctx.get("log_monitor_frame"):
                continue
            if isinstance(child, tk.Button) and child is not ctx["split_btn"]:
                child.configure(bg=FIELD, fg=TEXT, activebackground=BORDER, relief="flat", bd=0,
                                highlightthickness=1, highlightcolor=ACCENT, highlightbackground=FIELD,
                                disabledforeground="#697980", cursor="hand2")
                child.bind("<Return>", lambda e, w=child: w.invoke())
            elif isinstance(child, tk.Label):
                child.configure(fg=MUTED if child.cget("fg") not in (TEXT, ACCENT) else child.cget("fg"))
            elif isinstance(child, tk.Entry) and not isinstance(child, ttk.Entry):
                child.configure(bg=FIELD, fg=TEXT, insertbackground=TEXT, relief="flat", bd=0,
                                highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)
            theme(child)
    theme(dialog)
    for page in pages:
        polish_controls(page.body)
        page.bind_navigation()
    canvas = ctx["viz_canvas"]
    def key_view(event):
        if ctx["canvas_view_mode_var"].get() != "3d":
            return
        key = event.keysym
        if key in ("Left", "Right"):
            orbit["yaw"] += -5 if key == "Left" else 5
        elif key in ("Up", "Down"):
            orbit["pitch"] = max(-88, min(88, orbit["pitch"] + (5 if key == "Up" else -5)))
        elif key in ("plus", "equal", "minus"):
            orbit["dist"] = max(1.2, min(9, orbit["dist"] + (.25 if key == "minus" else -.25)))
        elif key.lower() == "f":
            ctx["reset_3d_view"]()
        elif key in ("bracketleft", "bracketright"):
            renderer.selected = (renderer.selected + (1 if key == "bracketright" else -1)) % max(1, len(ctx["get_current_rig"]()))
        else:
            return
        redraw()
        return "break"
    canvas.bind("<KeyPress>", key_view)
    app.rig_designer = SimpleNamespace(dialog=dialog, tabs=notebook, pages=pages, canvas=canvas,
        renderer=renderer, orbit=orbit, mode=ctx["mode_var"], view=ctx["canvas_view_mode_var"],
        cameras=ctx["num_cams_var"], fov=ctx["fov_var"], width=ctx["out_w_var"], height=ctx["out_h_var"],
        redraw=redraw, get_rig=ctx["get_current_rig"], close=ctx["on_close_dialog"], controls=controls,
        focus_view=focus_view, focused=focused, select_camera=select_camera, camera_text=camera_text,
        inspector=inspector, toggle_display=toggle_display, display_open=display_open,
        controls_guide=controls_guide, toggle_guide=toggle_guide, camera_picker=camera_picker)
    dialog.after_idle(redraw)
