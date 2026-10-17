"""Actual child-target pointer events, scroll bursts and canonical version identity."""
import copy,json,tempfile,time,tkinter as tk,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import test_control_points_workspace as fixtures


class InputTests(unittest.TestCase):
    setUp=fixtures.WorkspaceTests.setUp;tearDown=fixtures.WorkspaceTests.tearDown;wait=fixtures.WorkspaceTests.wait
    def drag(self,split,index,child,delta,mode):
        e=self.editor;e.settings['resize_mode'].set(mode);axis=0 if str(split.cget('orient'))=='horizontal' else 1;x,y=split.sash_coord(index);px,py=(x+2,y+60) if axis==0 else (x+30,y+2)
        split.event_generate('<ButtonPress-1>',x=px,y=py);gx=split.winfo_rootx()+px+(delta if axis==0 else 0);gy=split.winfo_rooty()+py+(delta if axis==1 else 0)
        child.event_generate('<B1-Motion>',x=gx-child.winfo_rootx(),y=gy-child.winfo_rooty());self.wait(lambda:e.live_layout.job is None)
        if mode=='fast':
            self.assertEqual(split.sash_coord(index)[axis],(x,y)[axis]);self.assertIsNotNone(e.live_layout.ghost);self.assertEqual(e.live_layout.preview[axis+1]-(x,y)[axis],delta)
        child.event_generate('<ButtonRelease-1>',x=gx-child.winfo_rootx(),y=gy-child.winfo_rooty());self.root.update()
        self.assertEqual(split.sash_coord(index)[axis]-(x,y)[axis],delta);self.assertIsNone(e.live_layout.drag);self.assertIsNone(e.live_layout.ghost);self.assertIsNone(e.window.grab_current())
        if split is e.workspace_split:self.assertEqual(e.live_layout.preferred,(split.sash_coord(0)[0],split.winfo_width()-split.sash_coord(1)[0]-int(split.cget('sashwidth'))))
    def test_panels_resize_across_child_widgets_in_live_and_fast_modes(self):
        e=self.editor;e.window.geometry('2500x1400');self.root.update();self.wait(lambda:e.live_layout.job is None);before=copy.deepcopy(e.controls)
        for mode in ['live','fast']:
            self.drag(e.workspace_split,0,e.panes[0]['view'].canvas,75,mode)
            e.right_tabs.select(1);self.root.update();self.drag(e.workspace_split,1,e.right_tabs,-75,mode)
            e.right_tabs.select(0);self.root.update();self.drag(e.properties_split,0,e.measurement_tree,-30,mode)
        self.assertEqual(e.controls,before);self.assertGreaterEqual(int(e.workspace_split.cget('sashwidth')),10)
    def test_large_side_panels_fit_when_window_shrinks_and_restore_when_enlarged(self):
        e=self.editor;e.window.geometry('2500x1400');self.root.update();self.wait(lambda:e.live_layout.job is None);split=e.workspace_split
        x,y=split.sash_coord(0);split.event_generate('<ButtonPress-1>',x=x+2,y=y+60);split.event_generate('<B1-Motion>',x=9000,y=y+60);split.event_generate('<ButtonRelease-1>',x=9000,y=y+60);self.root.update();preferred=e.live_layout.preferred
        for size in ['1100x720','2500x1400']:
            e.window.geometry(size);self.root.update();self.wait(lambda:e.live_layout.job is None)
            widths=[e.window.nametowidget(p).winfo_width() for p in split.panes()];self.assertGreaterEqual(widths[0],210);self.assertGreaterEqual(widths[1],400);self.assertGreaterEqual(widths[2],250)
            self.assertLessEqual(abs(sum(widths)+2*int(split.cget('sashwidth'))-split.winfo_width()),3)
        self.assertEqual(e.live_layout.preferred,preferred)
        e.live_layout.reset();self.wait(lambda:e.live_layout.reset_job is None);self.assertEqual(e.live_layout.preferred,(260,300));self.assertAlmostEqual(split.sash_coord(0)[0],260,delta=1)
    def test_grid_wheel_bursts_coalesce_and_small_deltas_keep_their_magnitude(self):
        e=self.editor
        for i in range(45):
            n=f'synthetic_{i:02}.jpg';Image.new('RGB',(256,128),(60+i,100,160)).save(self.folder/n);e.image_sizes[n]=[256,128]
        e.image_names=list(e.image_sizes);e.open_grid();g=e.grid;self.wait(lambda:len(g.store.cache)==48 and not g.store.pending and not g.store.running);self.root.update()
        g.scroll('moveto',0);self.root.update();before=g.canvas.canvasy(0);passes=g.layout_passes;decoded=g.store.stats['decodes']
        job=None
        for _ in range(40):
            g.canvas.event_generate('<MouseWheel>',delta=-15,x=20,y=20)
            if job is None:job=g.visible_job
            self.assertEqual(g.visible_job,job)
        self.assertEqual(g.canvas.canvasy(0),before);self.wait(lambda:g.visible_job is None)
        self.assertAlmostEqual(g.canvas.canvasy(0)-before,350,delta=2);self.assertLessEqual(g.layout_passes-passes,1);self.assertEqual(g.store.stats['decodes'],decoded)
        passes=g.layout_passes;g.arrange();g.arrange();self.assertEqual(g.layout_passes,passes)
    def test_resize_choice_roundtrips_and_invalid_file_does_not_change_controls(self):
        e=self.editor;e.settings['resize_mode'].set('fast');e.controls_path=self.folder/'controls.json';self.assertTrue(e.save());solver=e.refine_settings();payload=json.loads(e.controls_path.read_text());self.assertEqual(payload['layout_settings'],{'resize_mode':'fast'})
        e.settings['resize_mode'].set('live')
        with patch('control_points_workspace.filedialog.askopenfilename',return_value=str(e.controls_path)):e.load()
        self.assertEqual(e.settings['resize_mode'].get(),'fast');self.assertEqual(e.refine_settings(),solver)
        before=copy.deepcopy(e.controls);payload['layout_settings']['resize_mode']='wrong';e.controls_path.write_text(json.dumps(payload))
        with patch('control_points_workspace.filedialog.askopenfilename',return_value=str(e.controls_path)),patch.object(e,'error') as error:e.load()
        self.assertTrue(error.called);self.assertEqual(e.controls,before);self.assertEqual(e.settings['resize_mode'].get(),'fast')


class VersionTests(unittest.TestCase):
    def test_main_title_badge_and_batch_launchers_use_the_version_file(self):
        import version_manager
        from abu_align_gui import AbuAlignApp
        from manual_ui_lifecycle import capture,retire
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'VERSION.txt';source.write_text('0.1.5.8');root=tk.Tk();root.withdraw()
            with patch.object(version_manager,'VERSION_FILE',str(source)),patch('abu_align_gui.SystemHardwareMonitor'),patch.object(AbuAlignApp,'_ensure_logs_dir'),patch.object(AbuAlignApp,'_check_queue'):
                app=AbuAlignApp(root);self.assertIn('Version 0.1.5.8',root.title());self.assertEqual(app.version_label.cget('text'),'v0.1.5.8')
            retired=capture(root,[app])
            for job in root.tk.call('after','info'):root.tk.call('after','cancel',job)
            root.destroy();retire(retired)
        for name in ['Abu_Align.bat','Abu_Align_and_Train.bat']:
            text=(Path(version_manager.ROOT_DIR)/name).read_text();self.assertIn('%~dp0VERSION.txt',text);self.assertIn('Version %ABU_VERSION%',text)


if __name__=='__main__':unittest.main()
