"""Progressive disclosure and viewport controls for the existing rig workflow."""
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from desktop_ui import BG, SURFACE, FIELD, TEXT, MUTED, ACCENT, BORDER, ScrollPage, button, label


def wrap_row(row):
    children = list(row.winfo_children())
    for widget in children:
        widget.pack_forget()
    row.pack_propagate(False)
    previous = [None]
    def arrange(event):
        if event.width == previous[0]:
            return
        previous[0] = event.width
        x, y, line_height = 0, 0, 0
        for widget in children:
            width = widget.winfo_reqwidth() + 8
            if x and x + width > event.width:
                y, x, line_height = y + line_height + 10, 0, 0
            height = widget.winfo_reqheight()
            widget.place(x=x, y=y+5, width=min(width-8, event.width), height=height)
            x = x + width
            line_height = max(line_height, height)
        row.configure(height=y+line_height+10)
    row.bind("<Configure>", arrange)


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
    left.configure(width=560, bg=BG)
    left.pack_configure(expand=False, padx=(0, 14))
    right.configure(bg=BG)
    right.pack_configure(expand=True, padx=0)

    for child in ctx["hdr_left"].winfo_children():
        child.destroy()
    label(ctx["hdr_left"], "360° Rig Designer", 20).pack(anchor="w")
    label(ctx["hdr_left"], "Compose your cameras. Inspect coverage. Export ready-to-use views.", muted=True).pack(anchor="w", pady=(5, 0))

    notebook = ttk.Notebook(left)
    notebook.pack(fill="both", expand=True)
    pages = [ScrollPage(notebook) for _ in range(3)]
    for page, title, card in zip(pages, ["Source", "Rig", "Export"],
                                 [ctx["paths_card"], ctx["rig_card"], ctx["optics_card"]]):
        notebook.add(page, text=title)
        card.pack_forget()
        card.configure(text="", bd=0, padx=0, pady=0)
        card.pack(in_=page.body, fill="x")
        card.lift()
    notebook.select(1 if ctx["in_var"].get() else 0)
    notebook.enable_traversal()
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
    for name in ("o1", "o3", "dome_row1", "video_card_frame"):
        if name != "video_card_frame":  # This row's entry changes between FPS/stride via pack.
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
        row = ctx[row_name]
        row.pack_configure(pady=9)
        for widget in row.winfo_children():
            if isinstance(widget, tk.Scale):
                widget.configure(bd=0, sliderrelief="flat", sliderlength=24, width=7,
                                 bg=ACCENT, activebackground="#9ce8d3", troughcolor=FIELD)
    for name, title in [("cams_lbl", "Cameras"), ("fov_lbl", "Field of view"), ("ol_lbl", "Overlap")]:
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
    dialog.bind("<Escape>", lambda e: focus_view(True) if focused.get() else None, add="+")
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
    toggles.pack(fill="x", before=ctx["canvas_card"])
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
    previous_button = button(inspector, "← Previous", lambda: select_camera(-1))
    previous_button.pack(side="right", padx=6)
    for control in (next_button, previous_button):
        control.configure(padx=8, pady=4)
    def inspect_camera(event=None):
        rig = ctx["get_current_rig"]()
        if rig:
            index = min(renderer.selected, len(rig)-1)
            name, yaw, pitch, roll, fov = rig[index]
            camera_text.configure(text=f"CAMERA {index+1:02d} / {len(rig):02d}\nYaw {yaw:g}° · Pitch {pitch:g}° · FOV {fov:g}°")
    ctx["viz_canvas"].bind("<<RigRendered>>", inspect_camera)
    inspector.bind("<Configure>", lambda e: camera_text.configure(wraplength=max(120, e.width-230)))
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
    ctx["auto_res_btn"].configure(text="Match source")

    def theme(widget):
        for child in widget.winfo_children():
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
        page.bind_navigation()
    # Reparented geometry still belongs to left_col in Tk; bind wheel events there too.
    for card, page in zip([ctx["paths_card"], ctx["rig_card"], ctx["optics_card"]], pages):
        def bind_card(widget, target=page):
            widget.bind("<MouseWheel>", lambda e: (target.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"), "break")[-1])
            def reveal(event):
                y = event.widget.winfo_rooty() - target.body.winfo_rooty()
                top = target.canvas.canvasy(0)
                visible = target.canvas.winfo_height()
                total = max(1, target.body.winfo_height())
                if y < top:
                    target.canvas.yview_moveto(max(0, y-12) / total)
                elif y + event.widget.winfo_height() > top + visible:
                    target.canvas.yview_moveto((y + event.widget.winfo_height()-visible+12) / total)
            widget.bind("<FocusIn>", reveal, add="+")
            for child in widget.winfo_children():
                bind_card(child, target)
        bind_card(card)
    canvas = ctx["viz_canvas"]
    def key_view(event):
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
        focus_view=focus_view, focused=focused, select_camera=select_camera, camera_text=camera_text)
    dialog.after_idle(redraw)
