import os
import sys
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))
import test_gui_presets


class DesktopTests(unittest.TestCase):
    setUp = test_gui_presets.PresetTests.setUp
    close_root = test_gui_presets.PresetTests.close_root

    def test_manual_controls_menu_opens_new_workspace(self):
        from tkinter import ttk
        from unittest.mock import patch
        def descendants(widget):
            yield widget
            for child in widget.winfo_children(): yield from descendants(child)
        prepare = next(w for w in descendants(self.root) if isinstance(w, tk.Menubutton) and w.cget('text') == 'Prepare input ▾')
        menu = self.root.nametowidget(prepare.cget('menu'))
        index = next(i for i in range(menu.index('end')+1) if menu.type(i) == 'command' and menu.entrycget(i, 'label') == '360° manual control points')
        theme = ttk.Style(self.root).theme_use()
        from control_points_ui import open_editor
        opened = []
        def factory(parent):
            editor = open_editor(parent); opened.append(editor); return editor
        with patch('control_points_ui.open_editor', side_effect=factory):
            menu.invoke(index); self.root.update()
            dialogs = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
            self.assertTrue(any('Control Points' in w.title() for w in dialogs))
        self.assertEqual(ttk.Style(self.root).theme_use(), theme)
        for editor in opened: editor.close()

    def test_shared_icons_keep_labels_and_follow_disabled_state(self):
        from desktop_ui import button, DISABLED
        from ui_icons import line_icon
        control = button(self.root, 'Save profile', lambda: None)
        control.pack()
        self.assertEqual(control.cget('text'), 'Save profile')
        enabled = str(control.cget('image'))
        self.assertTrue(enabled)
        control.configure(state='disabled')
        self.assertNotEqual(str(control.cget('image')), enabled)
        self.assertEqual(control.cget('cursor'), 'arrow')
        control.configure(state='normal', text='Refresh')
        self.assertEqual(control.cget('text'), 'Refresh')
        self.assertEqual(control.cget('cursor'), 'hand2')
        self.assertIs(line_icon(control, 'refresh'), line_icon(control, 'refresh'))
        for tab in self.app.workspace_tabs.tabs():
            self.assertTrue(self.app.workspace_tabs.tab(tab, 'image'))
        control.destroy()

    def test_all_engine_groups_fit_at_minimum_and_default_size(self):
        errors = []
        self.root.report_callback_exception = lambda *error: errors.append(error)
        self.root.deiconify()
        for size in ["960x720", "1240x900", "1600x1000"]:
            self.root.geometry(size)
            for tab, variable, engines, container in [
                (0, self.app.engine_var, ["vggt_omega", "vggttt", "ace0", "pycusfm", "opensfm", "360dvo"], self.app.align_dynamic_container),
                (1, self.app.train_engine_var, ["lichtfeld", "splatfacto", "3dgrut", "pfgs360"], self.app.train_dynamic_container),
            ]:
                self.app.workspace_tabs.select(tab)
                for engine in engines:
                    variable.set(engine)
                    self.root.update()
                    visible = [panel for panel in container.winfo_children() if panel.winfo_ismapped()]
                    self.assertEqual(len(visible), 1)
                    panel = visible[0]
                    for group in panel.winfo_children():
                        self.assertLessEqual(group.winfo_x() + group.winfo_reqwidth(), panel.winfo_width(),
                                             (size, engine, group.winfo_reqwidth(), panel.winfo_width()))
                    self.assertTrue(self.app.one_click_btn.winfo_ismapped())
                    self.assertLess(self.app.one_click_btn.winfo_rooty(), self.root.winfo_rooty() + self.root.winfo_height())
        self.assertEqual(errors, [])

    def test_activity_remains_readable_at_minimum_size(self):
        self.root.deiconify()
        self.app.workspace_tabs.select(2)
        for size in ["960x720", "1240x900", "1440x1050"]:
            self.root.geometry(size)
            self.root.update()
            self.assertGreaterEqual(self.app.console.winfo_height(), 80, size)

    def test_tool_dialogs_open_without_callback_errors(self):
        errors = []
        self.root.report_callback_exception = lambda *error: errors.append(error)
        self.root.deiconify()
        for command in (self.app._open_video_extractor_dialog,
                        self.app._open_panoramic_splitter_dialog):
            command()
            self.root.update()
            dialogs = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
            self.assertTrue(dialogs)
            for dialog in dialogs:
                callback = dialog.protocol("WM_DELETE_WINDOW")
                if callback:
                    self.root.tk.call(callback)
                else:
                    dialog.destroy()
        self.assertEqual(errors, [])

    def test_readonly_log_clear_and_follow(self):
        self.app._log("A diagnostic message\n")
        self.assertEqual(self.app.console.cget("state"), "disabled")
        self.assertIn("diagnostic", self.app.console.get("1.0", "end"))
        self.app.follow_log_var.set(False)
        with patch.object(self.app.console, "see") as see:
            self.app._log("Another line\n")
            see.assert_not_called()
        self.app._clear_console()
        self.assertEqual(self.app.console.get("1.0", "end"), "\n")
        self.assertEqual(self.app.workspace_tabs.select(), str(self.app.activity_tab))
        self.app._log("line\n" * 6000)
        self.assertLessEqual(int(self.app.console.index("end-1c").split(".")[0]), 5001)

    def test_alignment_only_omits_training_and_workflow_respects_toggle(self):
        self.app.dataset_var.set(self.temp.name)
        with patch("abu_align_gui.threading.Thread") as thread:
            for skip, auto, expected in [(True, True, False), (False, True, True), (False, False, False)]:
                self.app.is_running = False
                self.app.auto_train_var.set(auto)
                self.app._start_alignment(skip_auto_train=skip)
                command = thread.call_args.kwargs["args"][0]
                self.assertEqual("--train" in command, expected)
                self.assertEqual(self.app.workspace_tabs.select(), str(self.app.activity_tab))
        self.app.is_running = False

    def test_empty_state_and_preset_load_affordance(self):
        self.assertEqual(self.app.dataset_var.get(), "")
        self.assertEqual(self.app.preset_load_btn.cget("state"), "disabled")
        self.app.preset_var.set("Example")
        self.assertEqual(self.app.preset_load_btn.cget("state"), "normal")
        self.app.auto_train_var.set(False)
        self.assertEqual(self.app.one_click_btn.cget("text"), "Start alignment")

    def test_profiles_disclosure_and_training_summary_bounds(self):
        self.root.deiconify()
        self.root.geometry("960x720")
        self.root.update()
        self.assertFalse(self.app.profile_panel.winfo_ismapped())
        self.app.profiles_button.invoke()
        self.root.update()
        self.assertTrue(self.app.profile_panel.winfo_ismapped())
        self.assertGreaterEqual(self.app.preset_combo.winfo_width(), 100)
        self.app.workspace_tabs.select(1)
        self.root.update()
        for child in self.app.training_summary.winfo_children():
            self.assertLessEqual(child.winfo_x()+child.winfo_width(), self.app.training_summary.winfo_width())

    def test_log_search_wraps_and_never_changes_output(self):
        self.app._clear_console()
        self.app._log("ERROR first\nnormal\nerror second\n")
        original = self.app.console.get("1.0", "end")
        search = self.app.log_search
        search.show()
        search.query.set("error")
        self.assertEqual(search.count.cget("text"), "1 / 2")
        self.assertFalse(self.app.follow_log_var.get())
        search.find(-1)
        self.assertEqual(search.count.cget("text"), "2 / 2")
        search.find()
        self.assertEqual(search.count.cget("text"), "1 / 2")
        search.query.set("missing")
        self.assertEqual(search.count.cget("text"), "No matches")
        search.query.set("")
        self.assertFalse(self.app.console.tag_ranges("log_match"))
        search.hide()
        self.assertEqual(self.app.console.get("1.0", "end"), original)

    def test_run_mode_dispatch_and_lifecycle_lock(self):
        with patch.object(self.app, "_start_alignment") as align, patch.object(self.app, "_start_train_only") as train:
            self.app.run_mode_var.set("Align & train")
            self.app.one_click_btn.invoke()
            align.assert_called_once_with(force_auto_train=True)
            train.assert_not_called()
            align.reset_mock()
            self.app.run_mode_var.set("Align only")
            self.app.one_click_btn.invoke()
            align.assert_called_once_with(skip_auto_train=True)
            self.assertFalse(self.app.auto_train_var.get())
            align.reset_mock()
            self.app.run_mode_var.set("Train only")
            self.app.one_click_btn.invoke()
            train.assert_called_once_with()
            align.assert_not_called()
            self.assertIn("Training only", self.app.workflow_summary.cget("text"))
            self.app.one_click_btn.config(state="disabled")
            self.assertEqual(str(self.app.run_mode_combo.cget("state")), "disabled")
            self.app.one_click_btn.configure(state="normal")
            self.assertEqual(str(self.app.run_mode_combo.cget("state")), "readonly")

    def test_log_resume_is_explicit_and_preserves_query(self):
        self.app._log("warning here\nlatest line\n")
        search = self.app.log_search
        search.query.set("warning")
        search.hide()
        self.assertFalse(self.app.follow_log_var.get())
        search.show()
        self.assertTrue(self.app.console.tag_ranges("log_match"))
        original = self.app.console.get("1.0", "end")
        with patch.object(self.app.console, "see") as see:
            search.resume_live()
            see.assert_called_once_with("end")
        self.assertTrue(self.app.follow_log_var.get())
        self.assertEqual(search.query.get(), "warning")
        self.assertFalse(search.frame.winfo_manager())
        self.assertEqual(self.app.console.get("1.0", "end"), original)

    def test_log_search_refreshes_without_moving_reading_position(self):
        self.app._clear_console()
        self.app._log("warning first\nwarning second\n")
        search = self.app.log_search
        search.show()
        search.query.set("warning")
        search.find()
        self.app._log("warning third\n")
        with patch.object(self.app.console, "see") as see:
            search.refresh()
            see.assert_not_called()
        self.assertEqual(search.count.cget("text"), "2 / 3")
        self.assertFalse(self.app.follow_log_var.get())
        self.app._clear_console()
        search.refresh()
        self.assertEqual(search.count.cget("text"), "No matches")
        self.app._log("warning new\n")
        self.assertIsNotNone(search.pending_refresh)
        search.hide()
        self.assertIsNone(search.pending_refresh)

    def test_precision_wheel_and_option_protection(self):
        from desktop_ui import ScrollPage
        from tkinter import ttk
        page = ScrollPage(self.root)
        combo = ttk.Combobox(page.body, values=("First", "Second"), state="readonly")
        combo.set("First")
        combo.pack()
        page.bind_navigation()
        with patch.object(page.canvas, "yview", return_value=(0.2, 0.5)), patch.object(page.canvas, "yview_scroll") as scroll:
            for delta in (0, -30, -30, -30):
                combo.event_generate("<MouseWheel>", delta=delta)
            scroll.assert_not_called()
            combo.event_generate("<MouseWheel>", delta=-30)
            scroll.assert_called_once_with(1, "units")
            scroll.reset_mock()
            combo.event_generate("<MouseWheel>", delta=360)
            scroll.assert_called_once_with(-3, "units")
        self.assertEqual(combo.get(), "First")
        page.destroy()

    def test_video_controls_follow_sampling_and_format(self):
        self.root.deiconify()
        self.app._open_video_extractor_dialog()
        ui = self.app.video_designer
        self.root.update()
        self.assertEqual(ui.fps.cget("state"), "normal")
        self.assertEqual(ui.stride.cget("state"), "disabled")
        ui.mode.set("stride")
        self.assertEqual(ui.fps.cget("state"), "disabled")
        self.assertEqual(ui.stride.cget("state"), "normal")
        ui.format.set("png")
        self.assertEqual(ui.quality.cget("state"), "disabled")
        ui.blur.set(True)
        self.assertEqual(ui.threshold.cget("state"), "normal")
        ui.dialog.geometry("680x560")
        self.root.update()
        self.assertGreater(ui.page.canvas.winfo_height(), 150)
        ui.threshold.event_generate("<FocusIn>")
        self.root.update()
        top = ui.page.canvas.winfo_rooty()
        bottom = top + ui.page.canvas.winfo_height()
        self.assertGreaterEqual(ui.threshold.winfo_rooty(), top)
        self.assertLessEqual(ui.threshold.winfo_rooty() + ui.threshold.winfo_height(), bottom)
        ui.close()


    def test_train_only_pfgs360_workflow_and_telemetry_parsing(self):
        self.app.dataset_var.set(self.temp.name)
        self.app.train_engine_var.set("pfgs360")
        self.app.pfgs_dataparser_var.set("auto")
        with patch("abu_align_gui.threading.Thread") as thread:
            self.app.is_running = False
            self.app._start_train_only()
            self.assertTrue(thread.called)
            command = thread.call_args.kwargs["args"][0]
            self.assertIn("--engine", command)
            self.assertIn("pfgs360", command)
            self.assertIn("--pfgs_downscale", command)
            self.assertEqual(self.app.workspace_tabs.select(), str(self.app.activity_tab))
        self.app.is_running = False

        # Verify queue telemetry parser for PFGS360 depth priors
        self.app.output_queue.put("[PFGS360] 25%|██▌       | 43/169 [00:04<00:09, 12.74it/s]")
        self.app._check_queue()
        self.assertIn("Depth Priors", self.app.telemetry_lbl.cget("text"))
        self.assertIn("25.0%", self.app.pct_lbl.cget("text"))

        # Verify queue telemetry parser for 3DGS training step and ETA
        self.app.output_queue.put("[PFGS360] 1000 (3.33%)           12 s, 11.424 ms      1 m, 0 s")
        self.app._check_queue()
        self.assertIn("PFGS360 Active", self.app.telemetry_lbl.cget("text"))
        self.assertIn("3.3%", self.app.pct_lbl.cget("text"))
        self.assertIn("1 m, 0 s", self.app.eta_lbl.cget("text"))

        # Verify successful completion
        self.app.output_queue.put("__PROCESS_FINISHED__:0")
        self.app._check_queue()
        self.assertEqual(self.app.progress_bar["value"], 100)
        self.assertEqual(self.app.pct_lbl.cget("text"), "100.0%")

    def test_vggt_omega_vs_ace0_telemetry_disambiguation(self):
        """Verify VGGT-Omega neural batch inference is never mislabeled as ACE0 Scene MLP."""
        # 1. VGGT-Omega Batch Inference
        self.app.engine_var.set("vggt_omega")
        self.app.output_queue.put("[TELEMETRY:STAGE3] batch=1 total=12 pct=4.2")
        self.app._check_queue()
        text = self.app.telemetry_lbl.cget("text")
        self.assertIn("Neural Batch Inference", text)
        self.assertIn("[1/12]", text)
        self.assertNotIn("ACE0", text)
        self.assertNotIn("Loss:", text)
        self.assertEqual(self.app.active_stage_idx, 3)

        # 2. VGGT-Omega Multi-View Tracking
        self.app.output_queue.put("[TELEMETRY:STAGE5] query=1 total=3 frame=0 pct=33.3 eta_sec=45")
        self.app._check_queue()
        text5 = self.app.telemetry_lbl.cget("text")
        self.assertIn("Multi-View Tracking", text5)
        self.assertIn("[Query 1/3, Frame #0]", text5)
        self.assertIn("0m 45s", self.app.eta_lbl.cget("text"))
        self.assertEqual(self.app.active_stage_idx, 4)

        # 3. ACE0 Structured Telemetry
        self.app.engine_var.set("acezero")
        self.app.output_queue.put("[TELEMETRY:STAGE3] pct=12.5 iter=1000 total=8000 loss=0.123 inliers=85.0")
        self.app._check_queue()
        ace_text = self.app.telemetry_lbl.cget("text")
        self.assertIn("ACE0 Scene MLP", ace_text)
        self.assertIn("[1000/8000]", ace_text)
        self.assertIn("Loss: 0.123", ace_text)
        self.assertIn("Inliers: 85.0%", ace_text)

        # 4. ACE0 Raw Iteration Telemetry
        self.app.output_queue.put("Iteration: 500 | 8000 / Epoch 001 ... Loss: 0.234 ... inliers: 75.0%")
        self.app._check_queue()
        raw_text = self.app.telemetry_lbl.cget("text")
        self.assertIn("ACE0 Scene MLP", raw_text)
        self.assertIn("[500/8000]", raw_text)

    def test_native_3d_workspace_tab_integration(self):
        """Verify the 04 3D Workspace tab is mounted, selectable, and responsive."""
        self.root.deiconify()
        self.assertTrue(hasattr(self.app, "workspace_3d"))
        
        # Test tab navigation
        self.app.workspace_tabs.select(self.app.workspace_3d)
        self.assertEqual(self.app.workspace_tabs.select(), str(self.app.workspace_3d))
        
        # Test show_3d_workspace callback
        self.app.workspace_tabs.select(0)
        self.assertEqual(self.app.workspace_tabs.index(self.app.workspace_tabs.select()), 0)
        self.app._show_3d_workspace()
        self.assertEqual(self.app.workspace_tabs.select(), str(self.app.workspace_3d))
        
        # Test 3D view controls and presets
        ws = self.app.workspace_3d
        ws.reset_view()
        self.assertEqual(ws.yaw, -25.0)
        self.assertEqual(ws.pitch, 20.0)
        self.assertEqual(ws.distance, 1.5)  # Fit now uses normalized scene bounds.
        
        ws.set_preset_view(0, 89.9)
        self.assertEqual(ws.yaw, 0.0)
        self.assertEqual(ws.pitch, 89.9)
        
        ws.reset_view()
        self.assertEqual(ws.yaw, -25.0)
        self.root.geometry('960x720')
        ws.toggle_comparison()
        self.root.update()
        self.assertGreater(ws.canvas.winfo_height(), 120)
        self.assertGreater(ws.canvas.winfo_width(), 400)
        self.assertTrue(self.app.stop_btn.winfo_ismapped())
        self.assertFalse(self.app.path_entry.winfo_ismapped())
        self.app.workspace_tabs.select(0)
        self.root.update()
        self.assertTrue(self.app.path_entry.winfo_ismapped())


if __name__ == "__main__":
    unittest.main()
