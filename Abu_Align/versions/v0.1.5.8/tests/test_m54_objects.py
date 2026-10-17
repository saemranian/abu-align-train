"""Synthetic M54 pose, persistence, real GPU gesture and Viser protocol regressions."""
import copy
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from abu_m54 import NullObjects,SceneTransform,rotation_numbers,transform_scene,load_scene
from test_abu_m54 import fixture


class NullAndPoseTests(unittest.TestCase):
    def test_transactional_save_load_and_independent_undo(self):
        nulls=NullObjects();name=nulls.add([1e9+.012345,2,3],[10,90,20]);self.assertTrue(nulls.dirty)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'nulls.json';nulls.save(p,{'A':'nonexistent local model.json'});self.assertFalse(nulls.dirty)
            original=copy.deepcopy(nulls.items);nulls.set_pose(name,[4,5,6],[1,2,3]);self.assertTrue(nulls.dirty)
            metadata=nulls.load(p);self.assertEqual(nulls.items,original);self.assertFalse(nulls.dirty);self.assertIn('A',metadata)
            nulls.step();self.assertEqual(nulls.items[name]['position'],[4.,5.,6.]);nulls.step(True);self.assertEqual(nulls.items,original)
            before=copy.deepcopy(nulls.items);p.write_text(json.dumps({'schema':'abu_m54_nulls','schema_version':1,'nulls':{'bad':{'position':[1,2,3],'rotation_xyz_degrees':[0,float('nan'),0]}}}))
            with self.assertRaises(ValueError):nulls.load(p)
            self.assertEqual(nulls.items,before)
    def test_pose_quaternion_is_authoritative_at_euler_singularity(self):
        matrix=Rotation.from_euler('xyz',[10,90,20],degrees=True).as_matrix();values=rotation_numbers(matrix)
        q=np.asarray(values['quaternion_wxyz']);np.testing.assert_allclose(Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),matrix,atol=1e-14)
        np.testing.assert_allclose(Rotation.from_euler('xyz',values['rotation_xyz_degrees'],degrees=True).as_matrix(),matrix,atol=1e-14)
    def test_null_validation_does_not_consume_history(self):
        nulls=NullObjects()
        for position,angles in [([0,0],[0,0,0]),([0,float('inf'),0],[0,0,0]),([0,0,0],[0,0])]:
            with self.assertRaises(ValueError):nulls.add(position,angles)
        self.assertEqual(nulls.items,{});self.assertEqual(nulls.history,[])
        for payload in [[],{'x':{}},{'x':{'position':{},'rotation_xyz_degrees':[0,0,0]}},{'x':{'position':[1,2,3],'rotation_xyz_degrees':[0,0,float('nan')]}}]:
            with self.assertRaises(ValueError):nulls.load_data(payload)
        self.assertEqual(nulls.history,[])
    def test_control_sidecar_survives_worker_archive_without_inventing_orientation(self):
        from manual_camera_geometry import camera_map,project
        from control_points_comparison import save_scene,read_scene
        point=np.array([.2,.4,4]);component={'cameras':{'s':{'projection_type':'spherical','width':1024,'height':512}},'shots':{'a':{'camera':'s','rotation':[0,0,0],'translation':[0,0,0]},'b':{'camera':'s','rotation':[0,0,0],'translation':[-1,0,0]}},'points':{}}
        cameras=camera_map(component);controls={'schema_version':1,'points':[{'id':'CP','type':'tie','observations':[{'shot_id':n,'pixel':project(point,c)} for n,c in cameras.items()]}]}
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'refined.json';p.write_text(json.dumps([component]));p.with_suffix('.controls.json').write_text(json.dumps(controls))
            scene=load_scene(p);self.assertIn('CP',scene['controls']);np.testing.assert_allclose(scene['controls']['CP']['position'],point,atol=1e-10)
            save_scene(Path(directory)/'scene.npz',scene);loaded=read_scene(Path(directory)/'scene.npz');np.testing.assert_array_equal(loaded['controls']['CP']['position'],scene['controls']['CP']['position'])
            self.assertNotIn('rotation',loaded['controls']['CP']);self.assertEqual(loaded['controls']['CP']['shots'],['a','b'])


class DesktopTests(unittest.TestCase):
    def test_browser_shutdown_with_a_connected_websocket_drains_owned_tasks(self):
        code="""import time
from unittest.mock import patch
from abu_m54_browser import M54Browser
from websockets.sync.client import connect
with patch('viser._client_autobuild._check_viser_yarn_running',return_value=False):viewer=M54Browser(port=9039)
with connect('ws://127.0.0.1:9039',open_timeout=3) as client:
    client.recv(timeout=3);start=time.monotonic();viewer.close();assert time.monotonic()-start<2.5
print('CONNECTED_SHUTDOWN_OK')
"""
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('CONNECTED_SHUTDOWN_OK',result.stdout)
        for warning in ['Task was destroyed','never retrieved','Fatal Python error']:self.assertNotIn(warning,result.stdout+result.stderr)
    def test_fresh_process_dpi_policy_and_real_window_icon(self):
        if sys.platform!='win32':self.skipTest('Windows desktop identity')
        code="""import ctypes,json,tkinter as tk
from abu_m54_desktop import prepare_desktop,window_identity
policy=prepare_desktop();root=tk.Tk();dpi=window_identity(root);root.update()
awareness=ctypes.c_int();ctypes.windll.shcore.GetProcessDpiAwareness(None,ctypes.byref(awareness))
ctypes.windll.user32.GetAncestor.argtypes=[ctypes.c_void_p,ctypes.c_uint];ctypes.windll.user32.GetAncestor.restype=ctypes.c_void_p
hwnd=ctypes.windll.user32.GetAncestor(root.winfo_id(),2)
ctypes.windll.user32.SendMessageW.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t,ctypes.c_ssize_t];ctypes.windll.user32.SendMessageW.restype=ctypes.c_void_p
icon=ctypes.windll.user32.SendMessageW(hwnd,0x7f,1,0)
print(json.dumps(dict(policy=policy,awareness=awareness.value,dpi=dpi,icon=bool(icon),path=root._m54_icon_path)));root.destroy()
"""
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(result.returncode,0,result.stderr);value=json.loads(result.stdout.strip().splitlines()[-1]);self.assertEqual(value['awareness'],2);self.assertTrue(value['icon']);self.assertGreaterEqual(value['dpi'],96);self.assertEqual(Path(value['path']).name,'abu_icon.ico')


class NativeGestureTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from abu_m54_app import M54Application
        self.root=tk.Tk();self.errors=[];self.root.report_callback_exception=lambda *e:self.errors.append(e)
        self.app=M54Application(self.root);self.v=self.app.viewer;self.v.set_data(fixture());self.root.update();self.assertTrue(self.v.context)
        self.v.open_inspector();self.inspector=self.v.inspector_window;self.root.update()
    def tearDown(self):
        self.v.nulls.saved=self.v.nulls.signature();self.app.close();self.assertEqual(self.errors,[])
    def event(self,p):return SimpleNamespace(x=float(p[0]),y=float(p[1]),state=0)
    def move_gesture(self,pixels=40):
        edit=self.v.editing;spec,geometry=edit.geometry();screen,valid,_=self.v.project((np.array([spec['position'],spec['position']+spec['axes'][:,0]*spec['size']])-self.v.origin)/self.v.scale)
        self.assertTrue(valid.all());delta=screen[1]-screen[0];start=screen[0]+delta*.65
        self.assertTrue(edit.press(self.event(start)));self.assertEqual(edit.active['axis'],0);edit.move(self.event(start+delta/np.linalg.norm(delta)*pixels));edit.flush();edit.release()
        return spec,delta
    def test_inspector_camera_point_cloud_and_oriented_null(self):
        from abu_m54_editing import selection_record
        self.v.select_camera('a.jpg');self.inspector.refresh();record=selection_record(self.v)
        np.testing.assert_allclose(record['quaternion_wxyz'],[1,0,0,0]);self.assertEqual(self.inspector.fields['p0'].cget('state'),'readonly')
        self.v.select_control('CP');self.assertIsNone(selection_record(self.v)['rotation_xyz_degrees'])
        self.v.selection_kind='cloud';self.v.cloud_selected=('A',2);self.inspector.refresh();self.assertIsNone(selection_record(self.v)['quaternion_wxyz']);self.assertIn('not defined',self.inspector.detail.get())
        name=self.inspector.add_null();np.testing.assert_allclose(self.v.nulls.items[name]['position'],self.v.data['points'][2]);self.assertEqual(self.inspector.fields['p0'].cget('state'),'normal')
        self.inspector.values['p0'].set('123.123456789012');self.inspector.values['r2'].set('37');self.inspector.finish()
        self.assertAlmostEqual(self.v.nulls.items[name]['position'][0],123.123456789012,places=12);self.assertEqual(self.v.nulls.items[name]['rotation_xyz_degrees'][2],37)
        self.inspector.values['p0'].set('nan');self.inspector.finish();self.assertAlmostEqual(self.v.nulls.items[name]['position'][0],123.123456789012)
    def test_null_move_and_rotate_are_one_undo_each(self):
        self.v.world_up.set('Y');self.v.preset(0,0);name=self.inspector.add_null();before=copy.deepcopy(self.v.nulls.items[name]);count=len(self.v.nulls.history)
        spec,delta=self.move_gesture();expected=np.asarray(before['position'])+np.array([40/np.linalg.norm(delta)*spec['size'],0,0])
        np.testing.assert_allclose(self.v.nulls.items[name]['position'],expected,atol=1e-12);self.assertEqual(len(self.v.nulls.history),count+1);self.inspector.history(False);self.assertEqual(self.v.nulls.items[name],before)
        self.v.editing.mode.set('Rotate');spec,geometry=self.v.editing.geometry();start=geometry[2][8,0];angle=np.pi/6
        end=spec['position']+Rotation.from_rotvec([0,0,angle]).as_matrix()@(start-spec['position']);screen,valid,_=self.v.project((np.array([start,end])-self.v.origin)/self.v.scale)
        self.assertTrue(valid.all());self.assertTrue(self.v.editing.press(self.event(screen[0])));self.assertEqual(self.v.editing.active['axis'],2)
        self.v.editing.move(self.event(screen[1]));self.v.editing.release();self.assertAlmostEqual(self.v.nulls.items[name]['rotation_xyz_degrees'][2],30,places=8)
        self.inspector.history(False);self.assertEqual(self.v.nulls.items[name],before)
    def test_gizmo_cancel_restores_null_and_does_not_leave_grab(self):
        name=self.inspector.add_null();before=copy.deepcopy(self.v.nulls.items);count=len(self.v.nulls.history);edit=self.v.editing;self.v.preset(0,0)
        spec,_=edit.geometry();screen,_,_=self.v.project((np.array([spec['position'],spec['position']+[spec['size'],0,0]])-self.v.origin)/self.v.scale);start=screen.mean(0)
        self.assertTrue(edit.press(self.event(start)));edit.move(self.event(start+[40,0]));edit.flush();edit.cancel()
        self.assertEqual(self.v.nulls.items,before);self.assertEqual(len(self.v.nulls.history),count);self.assertIsNone(self.v.surface.grab_current())
    def test_model_mouse_move_keeps_geometry_and_optics_linked(self):
        before=copy.deepcopy(self.v.raw_data);self.v.preset(0,0);edit=self.v.editing;edit.enabled.set(True);edit.target.set('Layer A');self.move_gesture()
        for name in before['centers']:
            a=(before['points']-before['centers'][name])@before['camera_rotations'][name];b=(self.v.data['points']-self.v.data['centers'][name])@self.v.data['camera_rotations'][name]
            np.testing.assert_allclose(a,b,atol=1e-12)
        np.testing.assert_array_equal(self.v.raw_data['points'],before['points']);np.testing.assert_allclose(self.v.data['controls']['CP']['position'],self.v.data['points'][0])
        self.v.open_tools();self.v.tools_window.step(False);np.testing.assert_allclose(self.v.data['points'],before['points'])
    def test_fitted_B_gizmo_has_no_jump_and_cancel_restores_registration(self):
        from control_points_comparison import compare_scenes
        scene=fixture();scene['centers']['c.jpg']=np.array([0.,1,0]);scene['camera_rotations']['c.jpg']=np.eye(3);self.v.set_data(scene)
        b=transform_scene(scene,SceneTransform(np.array([3.,4,5]),np.array([10.,20,30]),2.));b['source']='synthetic-B.ply';self.v.compare.b=b;self.v.compare.registration.set(True)
        result=compare_scenes(self.v.data,b,True,True,False);self.v.set_comparison(result);before=self.v.comparison['b']['points'].copy()
        self.v.preset(0,0);edit=self.v.editing;edit.enabled.set(True);edit.target.set('Layer B');spec,_=edit.geometry()
        screen,_,_=self.v.project((np.array([spec['position'],spec['position']+[spec['size'],0,0]])-self.v.origin)/self.v.scale);start=screen.mean(0);self.assertTrue(edit.press(self.event(start)));edit.move(self.event(start+[25,0]));edit.flush()
        deadline=time.monotonic()+5
        while self.v.compare.registration.get() or self.v.comparison['report']['frame_fit_applied']:
            self.root.update();time.sleep(.01)
            if time.monotonic()>deadline:self.fail('B transform did not finish')
        delta=self.v.comparison['b']['points']-before;np.testing.assert_allclose(delta, np.broadcast_to(delta[0],delta.shape),atol=1e-10)
        edit.cancel();self.assertTrue(self.v.compare.registration.get());self.assertIsNone(self.v.surface.grab_current())
    def test_unsaved_null_close_can_cancel(self):
        self.inspector.add_null()
        with patch('abu_m54_app.messagebox.askyesnocancel',return_value=None):self.app.close()
        self.assertFalse(self.app.closed);self.assertTrue(self.root.winfo_exists())
    def test_open_browser_receives_an_independent_null_snapshot(self):
        name=self.inspector.add_null();before=copy.deepcopy(self.v.nulls.items)
        with patch('viser._client_autobuild._check_viser_yarn_running',return_value=False),patch('webbrowser.open',return_value=True):self.v.open_browser()
        browser=self.v.external_browser;self.assertEqual(browser.tools.nulls.items,before)
        browser.tools.nulls.set_pose(name,[20,30,40],[10,20,30]);self.assertEqual(self.v.nulls.items,before)
    def test_close_mid_gesture_releases_gpu_and_pending_jobs(self):
        self.inspector.add_null();self.v.preset(0,0);edit=self.v.editing;spec,_=edit.geometry()
        screen,_,_=self.v.project((np.array([spec['position'],spec['position']+[spec['size'],0,0]])-self.v.origin)/self.v.scale);start=screen.mean(0)
        self.assertTrue(edit.press(self.event(start)));edit.move(self.event(start+[40,0]));self.assertIsNotNone(edit.job)
        with patch('abu_m54_app.messagebox.askyesnocancel',return_value=False):self.app.close()
        self.assertIsNone(edit.active);self.assertIsNone(edit.job);self.assertIsNone(self.v.context);self.assertTrue(self.app.closed)


class BrowserProtocolTests(unittest.TestCase):
    def setUp(self):
        from abu_m54_browser import M54Browser
        with patch('viser._client_autobuild._check_viser_yarn_running',return_value=False):self.viewer=M54Browser(port=9037)
        self.viewer.set_scene(fixture());self.t=self.viewer.tools
    def tearDown(self):self.viewer.close()
    def send_pose(self,p,angles):
        from viser import _messages
        import asyncio
        q=Rotation.from_euler('xyz',angles,degrees=True).as_quat();message=_messages.TransformControlsUpdateMessage('/m54/gizmo',tuple(q[[3,0,1,2]]),tuple(p))
        asyncio.run(self.viewer.server.scene._handle_transform_controls_updates(0,message))
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            target=self.t.nulls.items.get(self.t.selected_null) if self.t.gizmo_target.value=='Selected null' else None
            expected=np.asarray(p)*fixture()['extent']+fixture()['center']
            actual=np.asarray(target['position']) if target else self.t.transforms['A'].matrix()[:3,:3]@fixture()['center']+self.t.transforms['A'].matrix()[:3,3]
            if np.allclose(actual,expected,atol=1e-10):return
            time.sleep(.01)
        self.fail('Viser gizmo callback did not apply the delivered pose')
    def test_actual_viser_handle_callback_and_persistent_mouse_model_transform(self):
        t=self.t;t.gizmo_enabled.value=True;t.render_tools();handle=t.gizmo;self.send_pose([.2,.3,.4],[10,20,30]);t.render_tools()
        self.assertIs(t.gizmo,handle);expected=np.array([.2,.3,.4])*fixture()['extent']+fixture()['center'];np.testing.assert_allclose(t.transforms['A'].matrix()[:3,:3]@fixture()['center']+t.transforms['A'].matrix()[:3,3],expected)
        np.testing.assert_allclose(t.effective()['A']['camera_rotations']['a.jpg'],Rotation.from_euler('xyz',[10,20,30],degrees=True).as_matrix())
        count=len(t.history['A']);self.send_pose([.21,.3,.4],[10,20,31]);self.assertEqual(len(t.history['A']),count);np.testing.assert_array_equal(self.viewer.scene['points'],fixture()['points'])
        t.select(t.effective()['A']['centers']['a.jpg'],kind='camera a.jpg');self.assertIn('Quaternion',t.selection_info.content);self.assertIn('World-to-camera translation',t.selection_info.content)
        self.assertIsNone(self.viewer.server.scene._scene_pointer_cb)
    def test_browser_null_fields_mouse_persistence_and_point_readout(self):
        t=self.t;t.select(fixture()['points'][0],kind='cloud sample');self.assertIn('not defined',t.selection_info.content)
        self.assertTrue(t.null_position.disabled);t.null_action(action='Add at selection');name=t.selected_null;self.assertTrue(name);self.assertFalse(t.null_position.disabled)
        t.null_position.value=(5.,6.,7.);t.null_angles.value=(11.,22.,33.);t.apply_null_fields();np.testing.assert_allclose(t.nulls.items[name]['position'],[5,6,7])
        t.gizmo_target.value='Selected null';t.render_tools();self.send_pose([.1,.2,.3],[12,24,36]);expected=np.array([.1,.2,.3])*fixture()['extent']+fixture()['center'];np.testing.assert_allclose(t.nulls.items[name]['position'],expected)
        np.testing.assert_allclose(t.nulls.items[name]['rotation_xyz_degrees'],[12,24,36]);self.assertIn('Quaternion',t.selection_info.content)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'local.m54nulls.json';t.null_file.value=str(p);t.null_files.value='Save';t.null_file_action();self.assertTrue(p.is_file());self.assertFalse(t.nulls.dirty)
            t.nulls.delete(name);t.null_files.value='Load';t.null_file_action();self.assertIn(name,t.nulls.items)


if __name__=='__main__':unittest.main()
