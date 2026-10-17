"""Independent M54 geometry, file, GUI and real loopback browser checks."""
import copy
import csv
import json
import queue
import struct
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from abu_m54 import SceneTransform, transform_scene, transformed_component, load_scene, export_copy, export_geometry, ray_pick, screen_pick, grid_lines
import test_control_points_workspace as workspace_tests


def fixture():
    p=np.array([[0.,0,4],[1,0,5],[0,2,3],[1,2,4.]])
    return dict(source='synthetic.ply',points=p,colors=np.full((4,3),170,np.uint8),centers={'a.jpg':np.zeros(3),'b.jpg':np.array([1.,0,0])},
                camera_rotations={'a.jpg':np.eye(3),'b.jpg':np.eye(3)},controls={'CP':dict(position=p[0],kind='tie',enabled=True,shots=['a.jpg','b.jpg'])},center=np.array([.5,1,2.5]),extent=5.,total_points=4)


class GeometryTests(unittest.TestCase):
    def test_transform_preserves_bearings_and_inputs(self):
        scene=fixture();before=copy.deepcopy(scene);t=SceneTransform(np.array([7.,-2,3]),np.array([30.,40,50]),2.5,np.array([1.,2,3]))
        preview=transform_scene(scene,t)
        for name in scene['centers']:
            original=(scene['points']-scene['centers'][name])@scene['camera_rotations'][name]
            changed=(preview['points']-preview['centers'][name])@preview['camera_rotations'][name]
            np.testing.assert_allclose(changed,original*2.5,atol=1e-12)
        np.testing.assert_array_equal(scene['points'],before['points']);np.testing.assert_array_equal(scene['controls']['CP']['position'],before['controls']['CP']['position'])
        np.testing.assert_allclose(preview['controls']['CP']['position'],preview['points'][0])
        for scale in [0,-1,np.nan]:
            with self.assertRaises(ValueError):SceneTransform(scale=scale).matrix()
    def test_full_json_copy_exports_all_and_roundtrips_poses(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);source=p/'source.json';component={'cameras':{'s':{'projection_type':'spherical','width':256,'height':128}},
                    'shots':{'a.jpg':{'camera':'s','rotation':[.1,.2,.3],'translation':[1.,2,3]}},
                    'points':{str(i):{'coordinates':[i*.1,.2,4.],'color':[100,150,200]} for i in range(31)}}
            source.write_text(json.dumps([component]));before=source.read_bytes();t=SceneTransform(np.array([2.,3,4]),np.array([10.,20,30]),1.7)
            report=export_copy(source,p/'copy.json',t);self.assertEqual(report['points'],31);self.assertEqual(source.read_bytes(),before)
            expected=transform_scene(load_scene(source),t);actual=load_scene(p/'copy.json')
            np.testing.assert_allclose(actual['points'],expected['points'],atol=1e-12);np.testing.assert_allclose(actual['centers']['a.jpg'],expected['centers']['a.jpg'],atol=1e-12)
            np.testing.assert_allclose(load_scene(p/'copy.points.ply')['points'],expected['points'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual['camera_rotations']['a.jpg'],expected['camera_rotations']['a.jpg'],atol=1e-12)
            csv_scene=load_scene(p/'copy.cameras.csv');np.testing.assert_allclose(csv_scene['centers']['a.jpg'],expected['centers']['a.jpg'])
            self.assertEqual(len(csv_scene['points']),31)
            with self.assertRaisesRegex(ValueError,'source'):export_copy(source,source,t)
            with self.assertRaisesRegex(ValueError,'already exists'):export_copy(source,p/'copy.json',t)
    def test_double_precision_ply_and_camera_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);scene=fixture();scene['points']+=1e9;scene['points'][1,0]+=.012345
            report=export_geometry(scene,p/'precision.ply');cloud=load_scene(report['point_cloud'],limit=3)
            self.assertEqual(len(cloud['points']),3);self.assertEqual(len(cloud['centers']),2)
            full=load_scene(report['point_cloud']);np.testing.assert_array_equal(full['points'],scene['points'])
            csv_scene=load_scene(report['camera_alignment_csv']);self.assertEqual(len(csv_scene['points']),4)
    def test_realityscan_angle_csv_is_not_identity_or_measurement_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cameras.csv';p.write_text('#name,x,y,alt,yaw,pitch,roll\na.jpg,1,2,3,0,90,0\n')
            scene=load_scene(p);r=scene['camera_rotations']['a.jpg'];np.testing.assert_allclose(r[:,2],[0,1,0],atol=1e-12)
            np.testing.assert_allclose(r[:,1],[0,0,-1],atol=1e-12);self.assertAlmostEqual(np.linalg.det(r),1.)
            p.write_text('image,point,x,y\na.jpg,CP,10,20\n')
            with self.assertRaisesRegex(ValueError,'not a camera alignment'):load_scene(p)
    def test_cloud_selection_and_grid_planes(self):
        points=np.array([[0,0,5.],[.01,0,2.],[4,0,1]])
        self.assertEqual(ray_pick(points,[0,0,0],[0,0,1]),1)
        self.assertIsNone(ray_pick(points,[0,0,0],[0,0,0]))
        self.assertEqual(screen_pick(points,np.array([[50,50],[50,50],[10,10]]),np.array([True,True,True]),np.array([.8,.2,.1]),50,50),1)
        for plane,axis in [('XY',2),('XZ',1),('YZ',0)]:
            lines,step=grid_lines([1.,2,3],8.,plane,7.);self.assertTrue(step>0);np.testing.assert_allclose(lines[:,:,axis],7.)
    def test_colmap_points_files_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'images.txt').write_text('1 1 0 0 0 -1 -2 -3 1 a.jpg\n\n')
            cloud=p/'points3D.bin'
            with cloud.open('wb') as stream:
                stream.write(struct.pack('<Q',2))
                for i in range(2):stream.write(struct.pack('<QdddBBBdQ',100+i,1e9+i*.0123,2.,3.,10,20,30,.1,1));stream.write(struct.pack('<II',1,i))
            scene=load_scene(cloud);self.assertEqual(len(scene['points']),2);self.assertEqual(len(scene['centers']),1)
            self.assertAlmostEqual(scene['points'][1,0]-scene['points'][0,0],.0123,places=6)
            folder=load_scene(p);np.testing.assert_array_equal(folder['points'],scene['points'])
            cloud.write_bytes(cloud.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError,'Truncated'):load_scene(cloud)
            text=p/'points3D.txt';text.write_text('# sparse points\n5 1 2 3 10 20 30 .1 1 0\n')
            self.assertEqual(load_scene(text)['points'].shape,(1,3))
    def test_integer_transforms_export_preserves_fractional_camera_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);source=p/'transforms.json';source.write_text(json.dumps({'frames':[{'file_path':'a.jpg','transform_matrix':np.eye(4,dtype=int).tolist()}]}))
            t=SceneTransform(np.array([.12,.23,.34]),np.array([12.,23.,34.]),1.5)
            export_copy(source,p/'new.json',t);expected=transform_scene(load_scene(source),t);actual=load_scene(p/'new.json')
            np.testing.assert_allclose(actual['centers']['a.jpg'],expected['centers']['a.jpg'],atol=1e-12)
            np.testing.assert_allclose(actual['camera_rotations']['a.jpg'],expected['camera_rotations']['a.jpg'],atol=1e-12)


class InterfaceTests(unittest.TestCase):
    setUp=workspace_tests.WorkspaceTests.setUp
    tearDown=workspace_tests.WorkspaceTests.tearDown
    wait=workspace_tests.WorkspaceTests.wait
    def test_live_split_and_adjustable_measurement_height(self):
        e=self.editor;self.assertTrue(e.workspace_split.cget('opaqueresize'));self.assertTrue(e.image_split.cget('opaqueresize'))
        split=e.properties_split;self.root.update();before=e.measurement_tree.winfo_height();x,y=split.sash_coord(0)
        split.sash_place(0,x,max(110,y-100));self.root.update();self.assertGreater(e.measurement_tree.winfo_height(),before+60)
        view=e.panes[0]['view']
        for i in range(30):view.resize(SimpleNamespace(width=350+i,height=280))
        self.assertIsNotNone(view.render_job);self.assertTrue(view.resizing)
        self.wait(lambda:not view.resizing);self.assertIsNone(view.resize_settle_job)
    def test_native_control_camera_cloud_pivots_transform_undo_and_grid(self):
        e=self.editor;e.component['points']={'p':{'coordinates':[.4,.2,3.],'color':[100,150,200]}}
        before=copy.deepcopy(e.controls);e.open_3d();self.wait(lambda:e.native_3d.data is not None);nav=e.native_3d
        nav.select_camera('b.jpg');nav.focus_selected();np.testing.assert_allclose(nav.center*nav.scale+nav.origin,nav.data['centers']['b.jpg'])
        nav.cloud_selected=('A',0);nav.selection_kind='cloud';nav.focus_selected();np.testing.assert_allclose(nav.center*nav.scale+nav.origin,nav.data['points'][0])
        nav.open_tools();tools=nav.tools_window;tools.values['t0'].set('2');tools.values['r2'].set('20');tools.values['s'].set('1.5');tools.apply()
        self.assertAlmostEqual(nav.transforms['A'].scale,1.5);tools.step(False);self.assertEqual(nav.transforms['A'].scale,1.);tools.step(True);self.assertEqual(nav.transforms['A'].scale,1.5)
        nav.ground.set(True);nav.grid_plane.set('XZ');nav.grid_height.set('invalid');nav.request();self.root.update();nav.draw()
        self.assertEqual(e.controls,before);self.assertTrue(nav.context);self.assertIn('NVIDIA',nav.renderer)


class BrowserTests(unittest.TestCase):
    def test_real_browser_controls_transforms_preserve_world_motion_and_pivot(self):
        from abu_m54_browser import M54Browser
        with patch('viser._client_autobuild._check_viser_yarn_running',return_value=False):viewer=M54Browser(port=9028)
        self.addCleanup(viewer.close);scene=fixture();viewer.set_scene(scene);tools=viewer.tools
        before=tools.display(scene['points'],scene).copy();tools.translation.value=(4.,0.,0.);tools.scale.value=2.;tools.apply_fields()
        changed=tools.effective()['A'];after=tools.display(changed['points'],changed);self.assertGreater(np.linalg.norm(after-before),1.)
        camera=SimpleNamespace(position=np.array([3.,4.,5.]),look_at=np.array([.5,.2,.3]));client=SimpleNamespace(camera=camera)
        offset=camera.position-camera.look_at;tools.select(changed['points'][0],client,'cloud sample');tools.focus_selection(SimpleNamespace(client=client))
        np.testing.assert_allclose(camera.position-camera.look_at,offset);np.testing.assert_allclose(camera.look_at,tools.display(changed['points'][0],changed))
        tools.grid.value=True;tools.plane.value='XZ';tools.gizmo_enabled.value=True;tools.render_tools();handle=tools.gizmo
        tools.render_tools();self.assertIs(tools.gizmo,handle);self.assertTrue(handle.visible)
        tools.layer.value='B';tools.render_tools();tools.apply_fields();self.assertIn('Load this reconstruction layer',tools.status.content);tools.layer.value='A'
        np.testing.assert_array_equal(scene['points'],fixture()['points']);self.assertIsNone(viewer.server.request_share_url())
    def test_control_browser_uses_shared_engine_and_keeps_control_events(self):
        from control_points_3d import ControlSceneViewer
        with patch('viser._client_autobuild._check_viser_yarn_running',return_value=False):viewer=ControlSceneViewer(queue.Queue())
        self.addCleanup(viewer.close);component={'points':{'p':{'coordinates':[0,0,3],'color':[180,180,180]}}};cameras={'a':(np.eye(3),np.zeros(3),256,128)}
        viewer.update(component,{'points':[]},cameras);viewer.m54.translation.value=(2.,0.,0.);viewer.m54.apply_fields()
        np.testing.assert_allclose(viewer.data['centers']['a'],[2,0,0]);self.assertTrue(viewer.m54.normalized)
        viewer.wait_ready(timeout=3)


class StandaloneTests(unittest.TestCase):
    def test_standalone_app_loads_space_path_in_isolated_worker(self):
        import tkinter as tk
        from abu_m54_app import M54Application
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'cloud with spaces.ply';export_geometry(fixture(),source)
            root=tk.Tk();errors=[];root.report_callback_exception=lambda *e:errors.append(e);app=M54Application(root)
            try:
                app.load(source);deadline=time.monotonic()+30
                while app.viewer.data is None and time.monotonic()<deadline:root.update();time.sleep(.02)
                self.assertIsNotNone(app.viewer.data);self.assertEqual(len(app.viewer.data['centers']),2)
                self.assertIn('Abu 3D Viewer M54',root.title());self.assertEqual(errors,[])
            finally:app.close()


if __name__=='__main__':unittest.main()
