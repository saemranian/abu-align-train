"""Read-only comparison, persistent grid and illustrated-guide regressions."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from control_points_comparison import compare_scenes,load_scene,save_scene,read_scene
from control_points_help import TOPICS,illustration
import test_control_points_workspace as workspace_tests


class ComparisonMathTests(unittest.TestCase):
    def scene(self):
        positions=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
        return {'source':'A.json','points':np.array([[.2,.3,2.],[.5,.1,3.],[.8,.4,4.]]),'colors':np.full((3,3),150,np.uint8),
                'centers':dict(zip(['a','b','c','d'],positions)),'camera_rotations':{n:np.eye(3) for n in ['a','b','c','d']},'controls':{},'total_points':3}
    def test_similarity_preview_pose_agreement_and_input_immutability(self):
        a=self.scene();rotation=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();scale=2.3;translation=np.array([5.,-2.,3.])
        b=copy.deepcopy(a);b['source']='B.json';b['points']=(a['points']-translation)@rotation/scale
        b['centers']={n:rotation.T@(p-translation)/scale for n,p in a['centers'].items()}
        b['camera_rotations']={n:rotation.T@r for n,r in a['camera_rotations'].items()};before=copy.deepcopy(b)
        raw=compare_scenes(a,b);self.assertGreater(raw['report']['camera_position_rmse'],1.)
        fitted=compare_scenes(a,b,True,True,True)
        self.assertAlmostEqual(fitted['report']['scale'],scale);self.assertLess(fitted['report']['camera_position_rmse'],1e-12)
        self.assertLess(fitted['report']['camera_orientation_rms_deg'],1e-12);self.assertLess(fitted['report']['sampled_b_to_a']['p95'],1e-10)
        np.testing.assert_allclose(fitted['b']['points'],a['points'],atol=1e-12);np.testing.assert_array_equal(b['points'],before['points'])
        for n in b['centers']:np.testing.assert_array_equal(b['centers'][n],before['centers'][n])
        rigid=compare_scenes(a,b,True,False);self.assertEqual(rigid['report']['scale'],1.);self.assertGreater(rigid['report']['camera_position_rmse'],.1)
    def test_ambiguous_or_missing_registration_returns_explicit_raw_preview(self):
        a=self.scene();b=copy.deepcopy(a)
        for i,n in enumerate(a['centers']):a['centers'][n]=np.array([i,0.,0.]);b['centers'][n]=np.array([i+3,0.,0.])
        result=compare_scenes(a,b,True);self.assertIn('collinear',result['report']['registration_error']);self.assertFalse(result['report']['frame_fit_applied'])
        np.testing.assert_array_equal(result['b']['points'],b['points'])
        b['centers']={};b['camera_rotations']={};result=compare_scenes(a,b,True)
        self.assertEqual(result['report']['matched_cameras'],0);self.assertIsNone(result['report']['camera_position_rmse'])
    def test_bounded_loader_archive_and_invalid_model(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);component={'shots':{'a.jpg':{'rotation':[0,0,0],'translation':[-1,0,0]}},'cameras':{},
                                    'points':{str(i):{'coordinates':[i,0,1],'color':[10,20,30]} for i in range(100)}}
            source=p/'model.json';source.write_text(json.dumps([component]));before=source.read_bytes();scene=load_scene(source,limit=17)
            self.assertEqual(len(scene['points']),17);self.assertEqual(scene['total_points'],100);self.assertEqual(source.read_bytes(),before)
            save_scene(p/'scene.npz',scene);loaded=read_scene(p/'scene.npz');np.testing.assert_array_equal(loaded['points'],scene['points']);np.testing.assert_array_equal(loaded['centers']['a.jpg'],[1,0,0])
            source.write_text(json.dumps([component,component]))
            with self.assertRaisesRegex(ValueError,'single reconstruction component'):load_scene(source)
            source.write_text('{"schema_version":1,"points":[]}')
            with self.assertRaisesRegex(ValueError,'no camera reconstruction'):load_scene(source)
    def test_other_formats_preserve_camera_frame_and_reject_disconnected_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);cloud=p/'points.ply'
            cloud.write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n1 2 3 10 20 30\n2 3 4 40 50 60\n')
            ply_scene=load_scene(cloud);self.assertEqual(ply_scene['centers'],{});self.assertEqual(len(ply_scene['points']),2)
            matrix=np.diag([1.,-1.,-1.,1.]);matrix[:3,3]=[1,2,3]
            transforms=p/'transforms.json';transforms.write_text(json.dumps({'frames':[{'file_path':'./a.jpg','transform_matrix':matrix.tolist()}]}))
            tf_scene=load_scene(transforms);np.testing.assert_array_equal(tf_scene['camera_rotations']['a.jpg'],np.eye(3))
            np.testing.assert_array_equal(tf_scene['centers']['a.jpg'],[1,2,3]);self.assertEqual(len(tf_scene['points']),2)
            run=p/'run';(run/'COLMAP_Lichtfield').mkdir(parents=True)
            (run/'COLMAP_Lichtfield'/'images.txt').write_text('1 1 0 0 0 -1 -2 -3 1 a.jpg\n\n')
            colmap_scene=load_scene(run);np.testing.assert_array_equal(colmap_scene['centers']['a.jpg'],[1,2,3])
            np.testing.assert_array_equal(colmap_scene['camera_rotations']['a.jpg'],np.eye(3))
            osfm=p/'opensfm_only';osfm.mkdir();component={'shots':{'a.jpg':{'rotation':[0,0,0],'translation':[-1,-2,-3]}},'cameras':{},'points':{}}
            reconstruction=osfm/'reconstruction.json';reconstruction.write_text(json.dumps([component,component]))
            with self.assertRaisesRegex(ValueError,'single reconstruction component'):load_scene(osfm)
            reconstruction.write_text(json.dumps([component]));self.assertEqual(load_scene(osfm)['source'],str(osfm))
            result=compare_scenes(self.scene(),colmap_scene,distances=True)
            self.assertIn('distance_error',result['report']);self.assertIsNone(result['heat_colors'])


class InterfaceTests(unittest.TestCase):
    setUp=workspace_tests.WorkspaceTests.setUp
    tearDown=workspace_tests.WorkspaceTests.tearDown
    def wait(self,condition):
        deadline=time.monotonic()+30
        while not condition() and time.monotonic()<deadline:self.root.update();time.sleep(.01)
        self.assertTrue(condition());self.root.update()
    def test_grid_double_click_stays_above_editor_and_preserves_state(self):
        e=self.editor;e.open_grid();g=e.grid;self.wait(lambda:len(g.photos)==3)
        geometry=g.window.geometry();g.target.set('Right');g.select('c.jpg');position=g.canvas.yview()
        result=g.select_and_open('c.jpg');self.wait(lambda:e.panes[1]['loaded_name']=='c.jpg')
        self.assertEqual(result,'break');self.assertFalse(g.closed);self.assertEqual(g.window.state(),'normal');self.assertEqual(g.window.geometry(),geometry)
        self.assertEqual(g.target.get(),'Right');self.assertEqual(g.selected,'c.jpg');self.assertEqual(g.canvas.yview(),position)
        self.assertEqual(str(g.window.transient()),str(e.window))
        self.assertEqual(int(self.root.tk.call('wm','stackorder',g.window,'isabove',e.window)),1)
    def test_comparison_load_switch_report_and_control_document_unchanged(self):
        e=self.editor;component=copy.deepcopy(self.component);component['shots']['c.jpg']['translation']=[0,-1,0]
        component['points']={'p':{'coordinates':[.3,.2,3.],'color':[170,180,190]}}
        self.rec.write_text(json.dumps([component]));data,cameras=e.read_alignment(self.rec);e.install_alignment(self.rec,data,cameras,self.folder)
        before_controls=copy.deepcopy(e.controls);before_source=self.rec.read_bytes();b=copy.deepcopy(component)
        for shot in b['shots'].values():shot['translation'][0]-=.05
        other=self.folder/'comparison.json';other.write_text(json.dumps([b]));before_b=other.read_bytes()
        e.open_3d();self.wait(lambda:e.native_3d.data is not None);nav=e.native_3d;panel=nav.compare;panel.load(other)
        self.wait(lambda:panel.result is not None);self.assertEqual(panel.mode.get(),'Overlay');self.assertEqual(panel.result['report']['matched_cameras'],3)
        self.assertEqual(nav.surface.winfo_height(),nav.scene_body.winfo_height());self.assertGreater(nav.surface.winfo_height(),180);self.assertGreater(nav.surface.winfo_width(),450)
        self.assertFalse(panel.result['report']['frame_fit_applied']);self.assertTrue(panel.frame.winfo_ismapped())
        view=(nav.yaw,nav.pitch,nav.distance,nav.center.copy());old=panel.result;panel.switch();self.wait(lambda:panel.result is not old)
        self.assertEqual(nav.visible_layers(),(True,False));self.assertEqual((nav.yaw,nav.pitch,nav.distance),view[:3]);np.testing.assert_array_equal(nav.center,view[3])
        old=panel.result;panel.registration.set(True);panel.allow_scale.set(False);panel.color.set('B distance to A');panel.apply();self.wait(lambda:panel.result is not old)
        self.assertTrue(panel.result['report']['frame_fit_applied']);self.assertLess(panel.result['report']['camera_position_rmse'],1e-10)
        nav.select_camera('b.jpg','B');self.assertEqual(nav.selected_camera_layer,'B')
        self.assertEqual(e.controls,before_controls);self.assertEqual(self.rec.read_bytes(),before_source);self.assertEqual(other.read_bytes(),before_b)
        report=self.folder/'comparison.report.json'
        with patch('control_points_comparison.filedialog.asksaveasfilename',return_value=str(report)):panel.save_report()
        self.assertEqual(json.loads(report.read_text())['matched_cameras'],3)
        panel.clear();self.assertIsNone(nav.comparison);self.assertIsNone(panel.b)
    def test_illustrated_help_topics_search_actions_and_reference(self):
        e=self.editor;guide=e.help_view;e.tabs.select(e.help_page);self.root.update()
        self.assertEqual(len(guide.buttons),len(TOPICS));self.assertGreaterEqual(len(guide.cards),3)
        self.assertTrue(guide.images);self.assertIn('PREDICTIONS ARE GUIDANCE',guide.reference.get('1.0','end'))
        guide.search.set('ground-truth');self.root.update();self.assertEqual(guide.topic,'compare')
        guide.search.set('');guide.select('navigation');self.root.update();guide.act('settings')
        self.assertEqual(e.tabs.select(),str(e.workspace));self.assertEqual(e.right_tabs.index(e.right_tabs.select()),1)
        for kind in ['measure','compare','grid','points','mouse','refine','exports','logs']:
            image=illustration(kind);self.assertEqual(image.size,(334,128));self.assertGreater(len(image.getcolors(334*128)),5)


if __name__=='__main__':unittest.main()
