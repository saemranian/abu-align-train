"""Rig preview geometry, selection, and responsive desktop regressions."""
import math
import unittest
from unittest.mock import patch
import tkinter as tk
import test_gui_presets
from rig_viewport import basis, camera_corners


class GeometryTests(unittest.TestCase):
    def test_camera_basis_is_orthonormal_with_roll_and_poles(self):
        for yaw in [0, 45, 180, 359]:
            for pitch in [-90, -35, 0, 60, 90]:
                axes = basis(yaw, pitch, 37)
                for a in range(3):
                    for b in range(3):
                        dot = sum(x*y for x,y in zip(axes[a], axes[b]))
                        self.assertAlmostEqual(dot, 1 if a == b else 0)

    def test_frustum_preserves_fov_and_aspect(self):
        corners = camera_corners(("camera", 0, 0, 0, 90), 2)
        self.assertAlmostEqual(corners[0][0], -1)
        self.assertAlmostEqual(corners[0][1], .5)
        self.assertAlmostEqual(corners[0][2], 1)
        self.assertAlmostEqual(math.degrees(math.atan2(abs(corners[0][0]), corners[0][2])) * 2, 90)


class RigDesignerTests(unittest.TestCase):
    close_root = test_gui_presets.PresetTests.close_root

    def setUp(self):
        test_gui_presets.PresetTests.setUp(self)
        self.errors_seen = []
        self.root.report_callback_exception = lambda *e: self.errors_seen.append(e)
        self.root.deiconify()
        self.app._open_panoramic_splitter_dialog()
        self.ui = self.app.rig_designer
        self.root.update()

    def test_tabs_and_controls_visible_at_minimum_size(self):
        self.ui.dialog.geometry("1120x760")
        for index, page in enumerate(self.ui.pages):
            self.ui.tabs.select(index)
            self.root.update()
            cards = [w for w in page.body.pack_slaves() if isinstance(w, tk.LabelFrame)]
            self.assertEqual(len(cards), 1)
            self.assertTrue(cards[0].winfo_ismapped())
            self.assertGreater(cards[0].winfo_height(), 50)
            # A sibling-hosted card must be above the Notebook, not hidden behind it.
            x, y = cards[0].winfo_rootx()+5, cards[0].winfo_rooty()+5
            hit = self.root.winfo_containing(x, y)
            self.assertTrue(str(hit).startswith(str(cards[0])), (index, hit))
        self.assertGreater(self.ui.canvas.winfo_width(), 400)
        self.assertGreater(self.ui.canvas.winfo_height(), 250)
        self.assertEqual(self.errors_seen, [])

    def test_modes_views_selection_and_invalid_edit(self):
        for mode in ["ring", "cubemap", "dome", "fibonacci"]:
            self.ui.mode.set(mode)
            self.root.update()
            for view in ["3d", "polar", "equirect"]:
                self.ui.view.set(view)
                self.root.update()
                self.assertGreater(len(self.ui.canvas.find_all()), 0)
        self.ui.view.set("3d")
        self.ui.redraw()
        index, x, y, _ = self.ui.renderer.positions[-1]
        self.assertTrue(self.ui.renderer.pick(x, y))
        self.assertEqual(self.ui.renderer.selected, index)
        self.ui.width.set("")
        self.root.update()
        self.ui.width.set(1024)
        self.ui.redraw()
        self.assertEqual(self.errors_seen, [])

    def test_selected_camera_clamps_after_rig_change(self):
        self.ui.mode.set("fibonacci")
        self.ui.cameras.set(32)
        self.ui.renderer.selected = 31
        self.ui.redraw()
        self.ui.mode.set("cubemap")
        self.root.update()
        self.assertLess(self.ui.renderer.selected, len(self.ui.get_rig()))
        self.assertEqual(self.errors_seen, [])


if __name__ == "__main__":
    unittest.main()
