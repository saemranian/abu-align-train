"""Control workflow, image-coordinate invariants and asynchronous UI regression."""
import copy
import json
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from control_points_model import ControlDocument, validate_controls
from panorama_viewport import ViewTransform, PanoramaAsset
from control_points_ui import open_editor


class DocumentTests(unittest.TestCase):
    def test_rename_distances_and_reversible_measurements(self):
        doc = ControlDocument()
        doc.data['points'] = [{'id': 'A', 'type': 'tie', 'observations': []}, {'id': 'B', 'type': 'tie', 'observations': []}]
        doc.data['distances'] = [{'a': 'A', 'b': 'B', 'meters': 2, 'sigma_m': .01}]
        doc.mark_saved(); doc.checkpoint(); doc.rename('A', 'Corner A'); doc.measure('Corner A', 'first.jpg', [10.5, 20.5], .2)
        self.assertTrue(doc.dirty); self.assertEqual(doc.data['distances'][0]['a'], 'Corner A')
        doc.undo(); self.assertFalse(doc.dirty); doc.redo(); self.assertEqual(doc.point('Corner A')['observations'][0]['sigma_px'], .2)
        doc.checkpoint(); doc.delete('Corner A'); self.assertEqual(doc.data['distances'], [])
        doc.undo(); self.assertEqual(len(doc.data['distances']), 1)

    def test_validation_rejects_nan_duplicates_and_invalid_distance(self):
        data = {'schema_version': 1, 'points': [{'id': 'A', 'type': 'tie', 'observations': [{'shot_id': 'x', 'pixel': [1, float('nan')]}]}]}
        with self.assertRaises(ValueError): validate_controls(data)
        data['points'][0]['observations'] = []
        data['points'].append(copy.deepcopy(data['points'][0]))
        with self.assertRaises(ValueError): validate_controls(data)
        with self.assertRaises(ValueError): validate_controls({'schema_version': 1, 'points': [], 'distances': [{'a': 'X', 'b': 'X', 'meters': 1}]})


class NavigationTests(unittest.TestCase):
    def test_cursor_anchor_and_coordinate_roundtrip(self):
        t = ViewTransform(7680, 3840, 800, 500, 1., 3000, 1800)
        before = t.image_at(630, 170); t.zoom_at(1.2, 630, 170)
        self.assertAlmostEqual(before[0], t.image_at(630, 170)[0]); self.assertAlmostEqual(before[1], t.image_at(630, 170)[1])
        x, y = t.screen_at(*before); self.assertAlmostEqual(x, 630); self.assertAlmostEqual(y, 170)
    def test_seam_wrap_and_latitude_bounds(self):
        t = ViewTransform(7680, 3840, 800, 500, 2., 5, 50, True)
        t.pan(100, 10000)
        self.assertTrue(0 <= t.cx < 7680); self.assertEqual(t.cy, 125)
        x, _ = t.screen_at(7670, 125); self.assertLess(abs(x-t.width/2), 800)
        t.fit(); self.assertAlmostEqual(t.cx, 3840); self.assertAlmostEqual(t.cy, 1920)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.folder = Path(self.temp.name)
        for name, color in [('a.jpg', 'red'), ('b.jpg', 'blue'), ('c.jpg', 'green')]: Image.new('RGB', (256, 128), color).save(self.folder/name)
        self.component = {'cameras': {'s': {'projection_type': 'spherical', 'width': 256, 'height': 128}},
                          'shots': {n: {'camera': 's', 'rotation': [0,0,0], 'translation': [-i,0,0]} for i,n in enumerate(['a.jpg','b.jpg','c.jpg'])}, 'points': {}}
        self.rec = self.folder/'reconstruction.json'; self.rec.write_text(json.dumps([self.component]))
        self.root = tk.Tk(); self.root.withdraw(); self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.editor = open_editor(self.root); self.editor.window.geometry('1100x720'); self.root.update()
        component, cameras = self.editor.read_alignment(self.rec)
        self.editor.install_alignment(self.rec, component, cameras, self.folder)
        self.wait(lambda: all(p['loaded_name'] for p in self.editor.panes))

    def wait(self, condition):
        deadline = time.monotonic()+5
        while not condition() and time.monotonic() < deadline:
            self.root.update(); time.sleep(.01)
        self.assertTrue(condition()); self.root.update()
    def tearDown(self):
        self.editor.pending = False; self.editor.document.mark_saved(); self.editor.busy = False
        self.editor.close(); self.root.destroy(); self.temp.cleanup()
        self.assertEqual(self.errors, [])

    def test_click_drag_nudge_undo_and_measurement_focus(self):
        e = self.editor; e.new()
        for pane in e.panes: e.measure(pane, [125., 60.])
        self.assertEqual(len(e.selected()['observations']), 2)
        view = e.panes[0]['view']; view.nudge(SimpleNamespace(keysym='Right', state=4))
        pixel = lambda: next(o['pixel'][0] for o in e.selected()['observations'] if o['shot_id'] == 'a.jpg')
        self.assertAlmostEqual(pixel(), 125.1)
        e.undo(); self.assertEqual(pixel(), 125)
        e.redo(); self.assertAlmostEqual(pixel(), 125.1)
        e.measurement_tree.selection_set('a.jpg'); e.open_measurement(1)
        self.wait(lambda: e.panes[1]['loaded_name'] == 'a.jpg')
        self.assertEqual(e.panes[1]['view'].transform.zoom, 1.)
        self.assertEqual(len(e.selected()['observations']), 2)

    def test_rapid_navigation_and_bounded_tile_rendering(self):
        e = self.editor; pane = e.panes[0]
        for name in ['b.jpg', 'a.jpg', 'c.jpg']:
            pane['name'].set(name); e.show(pane)
        self.wait(lambda: pane['loaded_name'] == 'c.jpg')
        self.assertGreater(pane['view'].asset.image.getpixel((0,0))[1], 100)
        view = pane['view']; view.transform.zoom = 16; view.transform.constrain(); view.draw()
        t = view.transform
        self.assertLessEqual(view.last_render_pixels, (t.width+32)*(t.height+32))
        self.assertLess(len(view.photos), 3)

    def test_properties_distance_save_and_disabled_preflight(self):
        e = self.editor
        for _ in range(2):
            e.new()
            for pane in e.panes: e.measure(pane, [125., 60.])
        e.distance_a.set('CP001'); e.distance_b.set('CP002'); e.distance_m.set('2'); e.distance()
        e.property_vars['name'].set('Second corner'); self.assertTrue(e.apply_properties())
        self.assertEqual(e.controls['distances'][0]['b'], 'Second corner')
        e.controls_path = self.folder/'controls.json'; self.assertTrue(e.save()); self.assertFalse(e.document.dirty)
        loaded = validate_controls(json.loads(e.controls_path.read_text())); self.assertEqual(len(loaded['points']), 2)
        e.enabled.set(False); e.apply_properties()
        self.assertFalse(e.refinement_payload()['points'][1]['enabled'])
        e.controls['distances'].clear(); payload = e.refinement_payload()
        self.assertFalse(payload['points'][1]['enabled'])
        e.property_vars['type'].set('Ground control'); e.property_vars['x'].set('1'); e.property_vars['y'].set('2'); e.property_vars['z'].set('3')
        self.assertTrue(e.apply_properties())

    def test_help_tooltips_and_dirty_close_cancel(self):
        e = self.editor
        self.assertIn('PREDICTIONS ARE GUIDANCE', e.help_text.get('1.0', 'end'))
        self.assertGreaterEqual(len(e.guides), 40)
        guide = e.guides[0]; guide.show(); self.root.update(); self.assertIsNotNone(guide.popup); guide.hide(); self.assertIsNone(guide.popup)
        e.new()
        with patch('control_points_workspace.messagebox.askyesnocancel', return_value=None): e.close()
        self.assertFalse(e.closed)

    def test_minimum_layout_and_parent_theme_stays_unchanged(self):
        from tkinter import ttk
        e = self.editor
        self.assertTrue(e.point_summary.get())
        self.assertGreater(e.point_tree.winfo_height(), 50)
        footer = next(w for w in e.window.winfo_children() if isinstance(w, tk.Frame) and e.progress.master == w)
        self.assertLessEqual(footer.winfo_y()+footer.winfo_height(), e.window.winfo_height())
        page = e.right_tabs.nametowidget(e.right_tabs.tabs()[1]); e.right_tabs.select(page); self.root.update()
        page.scroll_canvas.yview_moveto(1); self.root.update()
        self.assertGreater(page.scroll_canvas.yview()[0], 0)
        theme = ttk.Style(self.root).theme_use()
        second = open_editor(self.root); self.assertEqual(ttk.Style(self.root).theme_use(), theme); second.close()

    def test_refinement_passes_settings_and_immutable_snapshot(self):
        e = self.editor; e.new()
        for pane in e.panes: e.measure(pane, [125., 60.])
        output = self.folder/'refined.json'; source = self.rec.read_bytes()
        report = {'converged': False, 'before_angular_rms_deg': 1., 'after_angular_rms_deg': .5, 'cameras': 3,
                  'optimized_points': 1, 'observations': 2, 'seconds': .1, 'termination': 'Budget reached', 'scale': 'original_arbitrary_scale', 'checks': []}
        e.settings['max_points'].set(250); e.settings['iterations'].set(15)
        with patch('control_points_workspace.filedialog.asksaveasfilename', return_value=str(output)), patch.object(e,'launch_worker') as run:
            e.refine();self.assertTrue(e.busy)
            command,kind,target=run.call_args.args
            self.assertEqual(kind,'refine');self.assertEqual(target,output)
            self.assertEqual(command[command.index('--max-points')+1],'250')
            self.assertEqual(command[command.index('--iterations')+1],'15')
            self.assertEqual(command[command.index('--backend')+1],'auto')
            report['backend']='test';e.result_queue.put(('refined',output,report));self.wait(lambda:not e.busy)
        self.assertEqual(self.rec.read_bytes(), source)
        self.assertTrue(output.with_suffix('.controls.json').exists())
        self.assertIn('Evaluation budget reached', e.results_text.get('1.0', 'end'))


if __name__ == '__main__': unittest.main()
