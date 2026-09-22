"""Preset integration checks using real Tk controls and isolated profile files."""
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from abu_align_gui import AbuAlignApp


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        with patch("abu_align_gui.SystemHardwareMonitor"), patch.object(
                AbuAlignApp, "_ensure_logs_dir"), patch.object(AbuAlignApp, "_check_queue"):
            self.app = AbuAlignApp(self.root)
        self.app.presets_dir = Path(self.temp.name)
        self.errors = patch("abu_align_gui.messagebox.showerror").start()
        self.info = patch("abu_align_gui.messagebox.showinfo").start()
        self.addCleanup(patch.stopall)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.close_root)

    def close_root(self):
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        self.root.destroy()

    def save(self, name="Tour – Fast"):
        with patch("abu_align_gui.simpledialog.askstring", return_value=name):
            self.app._save_preset()
        self.errors.assert_not_called()
        return self.app._preset_path(name)

    def settings(self):
        return {k: v.get() for k, v in self.app._preset_vars.items()}

    def test_round_trip_all_engines_and_dataset_preserved(self):
        for align in ["vggt_omega", "ace0", "pycusfm", "opensfm", "360dvo"]:
            for train in ["lichtfeld", "splatfacto", "3dgrut", "pfgs360"]:
                self.app.engine_var.set(align)
                self.app.train_engine_var.set(train)
                self.app.ns_bilateral_var.set(False)
                self.app.dvo_weights_var.set("C:/weights/custom.pth")
                expected = self.settings()
                path = self.save(f"{align}-{train}")
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn("dataset", saved["settings"])
                self.assertNotIn("train_mode", saved["settings"])
                for var in self.app._preset_vars.values():
                    if isinstance(var, tk.BooleanVar):
                        var.set(not var.get())
                self.app.engine_var.set("ace0")
                self.app.train_engine_var.set("3dgrut")
                self.app.dataset_var.set("C:/different-dataset")
                self.app._load_preset()
                self.assertEqual(self.settings(), expected)
                self.assertEqual(self.app.dataset_var.get(), "C:/different-dataset")
                self.assertEqual(self.app.train_mode_var.get(), self.app.lf_mode_var.get())
        self.errors.assert_not_called()
        self.app._refresh_presets()
        self.assertEqual(len(self.app.preset_combo.cget("values")), 20)

    def test_invalid_documents_leave_every_setting_unchanged(self):
        path = self.save()
        original = json.loads(path.read_text(encoding="utf-8"))
        before = self.settings()
        documents = ["{broken", "[]"]
        for field, value in [("engine", "unknown"), ("auto_train", "false"), ("res", "42")]:
            profile = json.loads(json.dumps(original))
            profile["settings"][field] = value
            documents.append(json.dumps(profile))
        documents.append(json.dumps({"schema_version": 99, "settings": before}))
        documents.append(json.dumps({"schema_version": 1, "settings": {}}))
        for document in documents:
            path.write_text(document, encoding="utf-8")
            self.app._load_preset()
            self.assertEqual(self.settings(), before)
        self.assertEqual(self.errors.call_count, len(documents))

    def test_cancel_overwrite_and_atomic_write_failure(self):
        path = self.save()
        before = path.read_bytes()
        self.app.cache_var.set(False)
        with patch("abu_align_gui.simpledialog.askstring", return_value=path.stem):
            with patch("abu_align_gui.messagebox.askyesno", return_value=False):
                self.app._save_preset()
            self.assertEqual(path.read_bytes(), before)
            with patch("abu_align_gui.messagebox.askyesno", return_value=True), patch(
                    "abu_align_gui.os.replace", side_effect=OSError("Disk unavailable")):
                self.app._save_preset()
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(self.app.presets_dir.glob("*.tmp")), [])
            with patch("abu_align_gui.messagebox.askyesno", return_value=True):
                self.app._save_preset()
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["settings"]["cache"])

    def test_paths_and_running_guard(self):
        for name in ["", "../outside", "CON", "LPT1.txt", "bad:name", "trailing."]:
            with self.assertRaises(ValueError):
                self.app._preset_path(name)
        self.save()
        self.app.is_running = True
        self.app.cache_var.set(False)
        self.app._load_preset()
        self.assertFalse(self.app.cache_var.get())
        self.info.assert_called_once()
        self.app.is_running = False
        self.app.preset_var.set("missing")
        self.app._load_preset()
        self.errors.assert_called_once()


if __name__ == "__main__":
    unittest.main()
