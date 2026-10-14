"""Continuous resize, warm scrolling, mixed view modes and precise error-cell regressions."""
import copy,json,time,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from control_measurement_colours import error_band,COLOURS,validate_limits
import test_control_points_workspace as fixtures


class ColourTests(unittest.TestCase):
    def test_threshold_boundaries_and_unavailable(self):
        self.assertEqual([error_band(x) for x in [0,.1,.10001,.5,.50001]],['good','good','warning','warning','bad'])
        self.assertIsNone(error_band(None));self.assertIsNone(error_band(float('nan')))
        for invalid in [None,[],{'error_good_deg':-1},{'error_warning_deg':.05},{'error_good_deg':'nan'}]:
            with self.assertRaises(ValueError):validate_limits(invalid)


class ViewTests(unittest.TestCase):
    setUp=fixtures.WorkspaceTests.setUp
    tearDown=fixtures.WorkspaceTests.tearDown
    wait=fixtures.WorkspaceTests.wait
    def catalog48(self):
        e=self.editor
        for i in range(45):
            name=f'synthetic_{i:03}.jpg';Image.new('RGB',(256,128),(20+i*4,80,160)).save(self.folder/name);e.image_sizes[name]=[256,128]
        e.image_names=list(e.image_sizes)
    def test_sash_layout_updates_without_waiting_for_release(self):
        e=self.editor;e.settings['resize_mode'].set('live');e.window.geometry('2200x1000');self.root.update()
        split=e.workspace_split;x,y=split.sash_coord(0)
        split.event_generate('<ButtonPress-1>',x=x+2,y=y+30)
        positions=[]
        for delta in [100,220,310,180,20]:
            split.event_generate('<B1-Motion>',x=x+delta,y=y+30)
            # Let one normal GUI frame render, keeping the mouse held down.
            # Layout must follow each motion before release, without a debounce
            # that restarts its delay on every movement.
            self.wait(lambda:e.live_layout.job is None);positions.append(split.sash_coord(0)[0])
            for pane in e.panes:
                page=pane['pages']['Image'];canvas=pane['view'].canvas
                self.assertLessEqual(abs(canvas.winfo_width()-page.winfo_width()),3)
            self.assertLessEqual(abs(e.image_split.winfo_width()-e.image_split.master.winfo_width()),3)
        self.assertGreater(len(set(positions)),3)
        split.event_generate('<ButtonRelease-1>',x=x+20,y=y+30)
        e.window.geometry('1100x720');self.root.update()
        for pane in e.panes:self.assertLessEqual(pane['choice'].winfo_x()+pane['choice'].winfo_width(),pane['choice'].master.winfo_width())
        self.assertEqual(e.live_layout.flushing,False)
    def test_drag_events_coalesce_to_latest_position_without_debounce(self):
        e=self.editor;e.window.geometry('1580x940');self.root.update();self.wait(lambda:e.live_layout.job is None);split=e.workspace_split;x,y=split.sash_coord(0)
        split.event_generate('<ButtonPress-1>',x=x+2,y=y+30)
        for delta in range(40,140):
            split.event_generate('<B1-Motion>',x=x+delta,y=y+30)
            job=e.live_layout.job
            if delta==40:first_job=job
            self.assertEqual(job,first_job)
        self.assertEqual(len(e.live_layout.pending),1);self.wait(lambda:e.live_layout.job is None)
        self.assertEqual(split.sash_coord(0)[0],x+137)
        split.event_generate('<ButtonRelease-1>',x=x+120,y=y+30);self.root.update()
        self.assertIsNone(e.live_layout.drag);self.assertEqual(split.sash_coord(0)[0],x+118)
    def test_empty_workspace_resize_has_no_central_dead_space(self):
        e=self.editor
        for pane in e.panes:pane['view'].asset=None;pane['view'].draw()
        for width,height in [(3300,1700),(1450,820),(2500,1400),(1100,720)]:
            e.window.geometry(f'{width}x{height}');e.live_layout.flush();self.root.update_idletasks()
            self.assertLessEqual(abs(e.image_split.winfo_width()-e.image_split.master.winfo_width()),3)
            for pane in e.panes:self.assertLessEqual(abs(pane['view'].canvas.winfo_width()-pane['pages']['Image'].winfo_width()),3)
    def test_grid_warm_scroll_is_immediate_reuses_tiles_and_never_redecodes(self):
        self.catalog48();e=self.editor;e.open_grid();g=e.grid
        self.wait(lambda:len(g.store.cache)==48 and not g.store.pending and not g.store.running)
        for position in [.5,1.,0.]:g.scroll('moveto',position);self.root.update()
        pool={t['item'] for t in g.pool};decoded=g.store.stats['decodes']
        for position in [1.,0.,.5,0.,1.,0.]:
            g.scroll('moveto',position)
            self.assertEqual(len(g.photos),len(g.tiles));self.assertTrue(all(g.canvas.itemcget(t['error'],'text')=='' for t in g.tiles.values()))
            self.assertEqual({t['item'] for t in g.pool},pool)
        self.assertEqual(g.store.stats['decodes'],decoded);self.assertEqual(len(g.names),48);self.assertLessEqual(len(g.store.photos),128)
    def test_embedded_grids_share_cache_and_open_other_image_view(self):
        e=self.editor;e.set_pane_mode(e.panes[0],'Grid');left=e.panes[0]['grid'];e.set_pane_mode(e.panes[1],'Grid');right=e.panes[1]['grid']
        self.assertIs(left.store,right.store);self.wait(lambda:len(left.store.cache)==3)
        left.target.set('Right');left.select_and_open('a.jpg');self.wait(lambda:e.panes[1]['loaded_name']=='a.jpg')
        self.assertEqual(e.panes[0]['mode'].get(),'Grid');self.assertEqual(e.panes[1]['mode'].get(),'Image');self.assertFalse(left.closed);self.assertTrue(left.canvas.winfo_ismapped())
        scroll=left.canvas.yview();self.assertEqual(e.focused,1);e.set_pane_mode(e.panes[0],'Image');e.set_pane_mode(e.panes[0],'Grid');self.root.update()
        self.assertIs(e.panes[0]['grid'],left);self.assertEqual(left.canvas.yview()[0],scroll[0])
    def test_native_M54_modes_share_source_keep_viewpoints_and_select_controls(self):
        e=self.editor;e.new();e.measure(e.panes[0],[125,60]);e.measure(e.panes[1],[120,61]);before=copy.deepcopy(e.controls)
        e.set_pane_mode(e.panes[0],'3D');e.set_pane_mode(e.panes[1],'3D');self.wait(lambda:all(v.data is not None and v.context for v in e.native_views()))
        a,b=e.panes[0]['scene'],e.panes[1]['scene'];self.assertIs(a.raw_data,b.raw_data);self.assertIn('NVIDIA',a.renderer)
        a.yaw=63.;b.yaw=-41.;e.set_pane_mode(e.panes[0],'Grid');e.set_pane_mode(e.panes[0],'3D');self.root.update()
        self.assertIs(e.panes[0]['scene'],a);self.assertEqual(a.yaw,63.);self.assertEqual(b.yaw,-41.);self.assertEqual(e.controls,before)
        a.select_camera('a.jpg');self.assertEqual(e.panes[1]['mode'].get(),'3D')
        e.set_pane_mode(e.panes[1],'Image');a.select_camera('b.jpg');self.wait(lambda:e.panes[1]['loaded_name']=='b.jpg');self.assertEqual(e.panes[0]['mode'].get(),'3D')
    def test_image_zoom_survives_grid_switch_and_hidden_render_is_suspended(self):
        e=self.editor;pane=e.panes[0];v=pane['view'];v.transform.zoom=2.75;v.transform.cx=100;before=v.transform.zoom
        e.set_pane_mode(pane,'Grid');v.request_draw();self.root.update();self.assertTrue(v.suspended)
        e.set_pane_mode(pane,'Image');self.root.update();self.assertFalse(v.suspended);self.assertEqual(v.transform.zoom,before)
    def test_error_numbers_only_coloured_live_and_double_click_opens_measurement(self):
        e=self.editor;e.window.geometry('1580x940');self.root.update();e.new();point=e.selected();point['observations']=[{'shot_id':n,'pixel':[120,60],'sigma_px':1} for n in ['a.jpg','b.jpg','c.jpg']]
        errors={'a.jpg':.05,'b.jpg':.3,'c.jpg':1.}
        with patch('control_points_workspace.point_geometry',return_value=(np.array([0,0,3]),'Synthetic consistency',errors)):e.populate_properties()
        self.wait(lambda:e.live_layout.job is None);self.assertEqual({n:l.cget('fg') for n,l in e.error_cells.labels.items()},{'a.jpg':COLOURS['good'],'b.jpg':COLOURS['warning'],'c.jpg':COLOURS['bad']})
        values=e.measurement_tree.item('b.jpg','values');e.settings['error_good_deg'].set(.4);e.settings['error_warning_deg'].set(2);self.root.update()
        self.assertEqual(e.error_cells.labels['b.jpg'].cget('fg'),COLOURS['good']);self.assertEqual(e.measurement_tree.item('b.jpg','values'),values)
        e.settings['error_good_deg'].set('nan');self.root.update();self.assertIn('last valid',e.error_legend.get());self.assertEqual(e.error_cells.limits(),(.4,2.))
        e.focused=1;e.error_cells.open('c.jpg');self.wait(lambda:e.panes[1]['loaded_name']=='c.jpg');self.assertEqual(e.measurement_tree.selection(),('c.jpg',))
        e.error_cells.set_errors({});self.root.update();self.assertEqual(e.error_cells.labels,{})
    def test_close_with_two_GPU_views_and_shared_grid_workers(self):
        e=self.editor;e.set_pane_mode(e.panes[0],'Grid');store=e.thumbnail_store;e.set_pane_mode(e.panes[0],'3D');e.set_pane_mode(e.panes[1],'3D')
        self.wait(lambda:all(v.context and v.data is not None for v in e.native_views()));views=e.native_views();e.document.mark_saved();e.close()
        self.assertTrue(store.closed);self.assertTrue(all(v.closed and v.context is None for v in views));self.assertTrue(e.error_cells.closed);self.assertTrue(e.live_layout.closed)
    def test_error_display_limits_roundtrip_without_solver_changes(self):
        e=self.editor;e.settings['error_good_deg'].set(.2);e.settings['error_warning_deg'].set(.8);e.controls_path=self.folder/'controls.json'
        solver=e.refine_settings();self.assertTrue(e.save());payload=json.loads(e.controls_path.read_text())
        self.assertEqual(payload['display_settings'],{'error_good_deg':.2,'error_warning_deg':.8});self.assertEqual(payload['refinement_settings'],solver)
        e.settings['error_good_deg'].set(.1);e.settings['error_warning_deg'].set(.5)
        with patch('control_points_workspace.filedialog.askopenfilename',return_value=str(e.controls_path)):e.load()
        self.assertEqual(e.error_cells.limits(),(.2,.8));self.assertEqual(e.refine_settings(),solver)
        before=copy.deepcopy(e.controls);payload['display_settings']['error_good_deg']=-1;e.controls_path.write_text(json.dumps(payload))
        with patch('control_points_workspace.filedialog.askopenfilename',return_value=str(e.controls_path)),patch.object(e,'error') as error:e.load()
        self.assertTrue(error.called);self.assertEqual(e.controls,before);self.assertEqual(e.error_cells.limits(),(.2,.8))
    def test_vertical_measurements_drag_and_plain_worker_ownership(self):
        e=self.editor;split=e.properties_split;x,y=split.sash_coord(0);before=e.measurement_tree.winfo_height()
        split.event_generate('<ButtonPress-1>',x=x+20,y=y+2);split.event_generate('<B1-Motion>',x=x+20,y=max(110,y-100));self.wait(lambda:e.live_layout.job is None)
        self.assertGreater(e.measurement_tree.winfo_height(),before)
        split.event_generate('<ButtonRelease-1>',x=x+20,y=max(110,y-100));self.root.update()
        e.set_pane_mode(e.panes[0],'Grid');state=e.thumbnail_store.state
        self.assertNotIn('editor',vars(state));self.assertNotIn('photos',vars(state));self.assertIsNone(getattr(e.thumbnail_store.worker,'__self__',None))


if __name__=='__main__':unittest.main()
