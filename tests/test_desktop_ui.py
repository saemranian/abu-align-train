"""Layout and workflow interaction regressions; no GPU processes are launched."""
import tkinter as tk
import unittest
from unittest.mock import patch

import test_gui_presets


class DesktopTests(unittest.TestCase):
    setUp = test_gui_presets.PresetTests.setUp
    close_root = test_gui_presets.PresetTests.close_root

    def test_all_engine_groups_fit_at_minimum_and_default_size(self):
        errors = []
        self.root.report_callback_exception = lambda *error: errors.append(error)
        self.root.deiconify()
        for size in ["960x720", "1240x900", "1600x1000"]:
            self.root.geometry(size)
            for tab, variable, engines, container in [
                (0, self.app.engine_var, ["vggt_omega", "ace0", "pycusfm", "opensfm", "360dvo"], self.app.align_dynamic_container),
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


if __name__ == "__main__":
    unittest.main()
