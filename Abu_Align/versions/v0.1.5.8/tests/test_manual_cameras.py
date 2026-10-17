"""Independent calibrated-pixel, solver, exports and live interaction checks."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from manual_camera_geometry import Camera,bearing,bearings_batch,bearing_weight,project,load_alignment,to_pycolmap,colmap_component
from manual_refinement_settings import validate,DEFAULTS
from spherical_refinement import refine
from spherical_acceleration import BearingProblem,robust_values
import test_control_points_workspace as workspace_tests


def perspective_fixture():
    rng=np.random.default_rng(451);c={'projection_type':'opencv','width':800,'height':600,'fx':700.,'fy':710.,'cx':400.,'cy':300.,'distortion':[.04,-.01,.001,-.002,0.]}
    comp={'cameras':{'cam':c},'shots':{},'points':{}}
    k=np.array([[700,0,400],[0,710,300],[0,0,1.]])
    for i in range(4):
        comp['shots'][f'{i}.jpg']={'camera':'cam','rotation':[0.,.015*i,0.],'translation':[-.6*i,0.,0.]}
    for i in range(24):
        xyz=np.r_[rng.uniform(-.8,.8,2),rng.uniform(4,7)];obs=[]
        for name,shot in comp['shots'].items():
            xy=cv2.projectPoints(xyz.reshape(1,1,3),np.array(shot['rotation']),np.array(shot['translation']),k,np.array(c['distortion']))[0][0,0]
            obs.append({'shot_id':name,'pixel':xy.tolist()})
        comp['points'][str(i)]={'coordinates':xyz.tolist(),'color':[100,150,200],'observations':obs}
    return comp


class CameraTests(unittest.TestCase):
    def test_distorted_bearing_projection_batch_and_pixel_uncertainty(self):
        comp=perspective_fixture();c=comp['cameras']['cam'];cam=Camera(np.eye(3),np.zeros(3),800,600,c)
        for xyz in [[.8,.3,5],[-.7,-.4,4.]]:
            pixel=project(xyz,cam);np.testing.assert_allclose(bearing(pixel,cam),np.array(xyz)/np.linalg.norm(xyz),atol=1e-9)
            weight=bearing_weight(pixel,cam,2);moved=bearing(np.array(pixel)+[.0001,0],cam)
            np.testing.assert_allclose((weight@(moved-bearing(pixel,cam)))[:2],[.00005,0],atol=1e-10)
        pixels=np.array([[100,200],[700,300.]])
        np.testing.assert_allclose(bearings_batch(pixels,cam),np.array([bearing(p,cam) for p in pixels]))
        self.assertIsNone(project([0,0,-5],cam));self.assertIsNone(project([10,0,1],cam))
        np.testing.assert_allclose(copy.deepcopy(cam).calibration['fx'],700)
    def test_colmap_roundtrip_full_calibration_tracks_and_empty_dense_tracks(self):
        comp=perspective_fixture();model=to_pycolmap(comp);back=colmap_component(model)
        self.assertEqual(len(back['points']),24);self.assertEqual(len(back['shots']),4)
        self.assertEqual(sum(len(p['observations']) for p in back['points'].values()),96)
        with tempfile.TemporaryDirectory() as d:
            model.write(d);component,cameras=load_alignment(Path(d)/'images.bin')
            for n in cameras:np.testing.assert_allclose(cameras[n][1],comp['shots'][n]['translation'])
            np.testing.assert_allclose(cameras['0.jpg'].calibration['distortion'][:4],comp['cameras']['cam']['distortion'][:4])
            self.assertEqual(len(component['points']),24)
    def test_matrix_whitening_analytic_jacobian_against_independent_difference(self):
        comp=perspective_fixture();from manual_camera_geometry import camera_map
        cams=camera_map(comp);ids=['0','1'];free=['1.jpg','2.jpg','3.jpg'];measurements=[]
        for ident in ids:
            for obs in comp['points'][ident]['observations']:
                name=obs['shot_id'];p=obs['pixel'];measurements.append((ident,name,bearing(p,cams[name]),bearing_weight(p,cams[name],1)))
        problem=BearingProblem(cams,free,ids,measurements,{},[],'3.jpg',1.8,True)
        x=np.r_[*[np.r_[comp['shots'][n]['rotation'],comp['shots'][n]['translation']] for n in free],*[comp['points'][p]['coordinates'] for p in ids]]
        _,jac=problem.evaluate(x,jac=True);numeric=np.column_stack([(problem.evaluate(x+np.eye(len(x))[i]*1e-6)-problem.evaluate(x-np.eye(len(x))[i]*1e-6))/2e-6 for i in range(len(x))])
        np.testing.assert_allclose(jac.toarray(),numeric,rtol=1e-5,atol=1e-6)
    def test_refinement_improves_independent_check_and_can_refine_again(self):
        truth=perspective_fixture();noisy=copy.deepcopy(truth);rng=np.random.default_rng(121)
        for shot in noisy['shots'].values():
            shot['rotation']=(np.array(shot['rotation'])+rng.normal(0,.007,3)).tolist();shot['translation']=(np.array(shot['translation'])+rng.normal(0,.025,3)).tolist()
        for p in noisy['points'].values():p['coordinates']=(np.array(p['coordinates'])+rng.normal(0,.015,3)).tolist()
        controls={'schema_version':1,'points':[]}
        for ident in ['0','1','2','3']:
            p=truth['points'][ident];controls['points'].append({'id':ident,'type':'check' if ident=='3' else 'gcp','position':p['coordinates'],'sigma_m':.002,'observations':p['observations']})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'initial.json';source.write_text(json.dumps([noisy]));cp=root/'controls.json';cp.write_text(json.dumps(controls));before=source.read_bytes()
            result=refine(source,root/'refined.json',cp,max_iterations=200,backend='cpu',options={'min_track_views':3,'loss':'huber'})
            self.assertLess(result['after_angular_rms_deg'],result['before_angular_rms_deg']*.1)
            self.assertLess(result['checks'][0]['error_units'],.002);self.assertEqual(source.read_bytes(),before)
            self.assertEqual(result['intrinsics'],'fixed imported calibration');self.assertTrue(Path(result['exports']['colmap_model']).is_dir())
            final=json.loads((root/'refined.json').read_text())[0];self.assertEqual(final['cameras'],truth['cameras'])
            rerun=refine(root/'refined.json',root/'second.json',cp,max_iterations=100,backend='cpu')
            self.assertLess(rerun['checks'][0]['error_units'],.002);self.assertEqual(rerun['observations'],result['observations'])
    def test_settings_reject_invalid_and_robust_cost_matches_definition(self):
        for options in [{'min_track_views':2.2},{'loss_scale':0},{'ftol':float('nan')},{'min_angle_deg':-1},{'loss':'magic'}]:
            with self.assertRaises(ValueError):validate(options)
        r=np.array([0.,1.,3.]);cost,weight=robust_values(r,'huber',2)
        self.assertAlmostEqual(cost,.5*(0+1+8));np.testing.assert_allclose(weight,[1,1,np.sqrt(2/3)])
    def test_perspective_cuda_preserves_calibration_and_improves_checks(self):
        from spherical_acceleration import cuda_available
        if not cuda_available():self.skipTest('Local CUDA unavailable')
        truth=perspective_fixture();noisy=copy.deepcopy(truth)
        for shot in noisy['shots'].values():shot['rotation'][0]+=.006;shot['translation'][1]+=.02
        controls={'schema_version':1,'points':[]}
        for ident in ['0','1','2','3']:
            p=truth['points'][ident];controls['points'].append(dict(id=ident,type='check' if ident=='3' else 'gcp',position=p['coordinates'],sigma_m=.002,observations=p['observations']))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'original.json';source.write_text(json.dumps([noisy]));cp=root/'controls.json';cp.write_text(json.dumps(controls))
            report=refine(source,root/'gpu.json',cp,max_iterations=200,backend='cuda',options={'loss':'huber'})
            self.assertEqual(report['backend'],'torch_cuda_sparse_lm_analytic');self.assertLess(report['checks'][0]['error_units'],.005)
            self.assertLess(report['after_angular_rms_deg'],report['before_angular_rms_deg']*.05)
            self.assertEqual(json.loads((root/'gpu.json').read_text())[0]['cameras'],truth['cameras'])
    def test_fisheye_known_projection_and_unsupported_model_rejection(self):
        c={'projection_type':'opencv_fisheye','fx':420.,'fy':425.,'cx':400.,'cy':300.,'distortion':[.01,-.005,.001,0.]}
        cam=Camera(np.eye(3),np.zeros(3),800,600,c);xyz=np.array([1.,.2,2.])
        np.testing.assert_allclose(bearing(project(xyz,cam),cam),xyz/np.linalg.norm(xyz),atol=1e-10)
        comp=perspective_fixture();comp['cameras']['cam']['projection_type']='unknown_lens'
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json';path.write_text(json.dumps([comp]))
            with self.assertRaisesRegex(ValueError,'Unsupported calibrated'):load_alignment(path)
    def test_perspective_pre_alignment_controls_validate_original_dimensions(self):
        from control_points_pipeline import validate_selected_controls
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);images=[]
            for name in ['a.jpg','b.jpg']:Image.new('RGB',(800,600),'grey').save(root/name);images.append(root/name)
            payload={'schema_version':1,'image_projection':'perspective','points':[{'id':'p','observations':[{'shot_id':n.name,'pixel':[200,300]} for n in images]}]}
            path=root/'controls.json';path.write_text(json.dumps(payload))
            self.assertEqual(validate_selected_controls(path,images,False)['image_projection'],'perspective')
            with self.assertRaises(ValueError):validate_selected_controls(path,images,True)
    def test_neural_cloud_without_tracks_does_not_become_fake_refined_geometry(self):
        comp=perspective_fixture();untracked=copy.deepcopy(comp['points']['0']);untracked['observations']=[];comp['points']['dense_untracked']=untracked
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'input.json';path.write_text(json.dumps([comp]));report=refine(path,root/'result.json',backend='cpu')
            result=json.loads((root/'result.json').read_text())[0]
            self.assertNotIn('dense_untracked',result['points']);self.assertEqual(report['untracked_source_points_excluded'],1)
            self.assertIn('dense_untracked',json.loads(path.read_text())[0]['points'])
    def test_underconstrained_manual_only_model_fails_without_writing_output(self):
        comp=perspective_fixture();comp['points']={'p':comp['points']['0']};comp['points']['p']['observations']=comp['points']['p']['observations'][:2]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.json';source.write_text(json.dumps([comp]))
            with self.assertRaisesRegex(ValueError,'Too few independent'):refine(source,root/'result.json',backend='cpu')
            self.assertFalse((root/'result.json').exists())
    def test_solver_rejects_wrong_control_projection_and_dimensions(self):
        comp=perspective_fixture();payload={'schema_version':1,'image_projection':'spherical','points':[{'id':'p','observations':comp['points']['0']['observations']}]}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.json';source.write_text(json.dumps([comp]));cp=root/'controls.json';cp.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'projection differs'):refine(source,root/'result.json',cp)
            payload['image_projection']='perspective';payload['image_sizes']={'0.jpg':[400,300]};cp.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'dimensions differ'):refine(source,root/'result.json',cp)


class LiveTests(unittest.TestCase):
    setUp=workspace_tests.WorkspaceTests.setUp
    tearDown=workspace_tests.WorkspaceTests.tearDown
    wait=workspace_tests.WorkspaceTests.wait
    def test_numbers_update_without_apply_drag_undo_and_invalid_value(self):
        e=self.editor;e.component['points']={'p':{'coordinates':[.4,.2,3.],'color':[100,150,200]}}
        e.open_3d();self.wait(lambda:e.native_3d.data is not None);nav=e.native_3d;nav.open_tools();tools=nav.tools_window
        tools.values['t0'].set('2');self.wait(lambda:nav.transforms['A'].translation[0]==2)
        tools.finish_edit();tools.begin_scrub(SimpleNamespace(x_root=100),'r2')
        tools.drag_value(SimpleNamespace(x_root=140,state=0));self.wait(lambda:nav.transforms['A'].angles[2]==10)
        tools.drag_value(SimpleNamespace(x_root=180,state=0));tools.end_scrub(SimpleNamespace())
        self.assertEqual(nav.transforms['A'].angles[2],20);tools.step(False);self.assertEqual(nav.transforms['A'].angles[2],0)
        self.assertEqual(nav.transforms['A'].translation[0],2)
        tools.values['s'].set('-');self.root.update();time.sleep(.03);self.root.update();self.assertEqual(nav.transforms['A'].scale,1.)
        self.assertIn('Manual control points',e.window.title());self.assertTrue(all(k in e.settings for k in DEFAULTS))
    def test_perspective_originals_no_seam_wrap_and_frustum_predictions(self):
        e=self.editor;comp=perspective_fixture()
        from manual_camera_geometry import camera_map
        from PIL import Image
        for n in comp['shots']:Image.new('RGB',(800,600),'grey').save(self.folder/n)
        path=self.folder/'perspective.json';path.write_text(json.dumps([comp]));e.install_alignment(path,comp,camera_map(comp),self.folder)
        self.wait(lambda:all(p['loaded_name'] in comp['shots'] for p in e.panes));e.settings['wrap'].set(True);e.update_view_settings()
        self.assertFalse(e.panes[0]['view'].transform.wrap)
        from control_point_predictions import predicted_candidates
        point={'id':'p','observations':comp['points']['0']['observations'][:2]}
        rows,_=predicted_candidates(point,e.cameras);self.assertEqual(len(rows),2)
        np.testing.assert_allclose(rows[0]['pixel'],comp['points']['0']['observations'][2]['pixel'],atol=1e-7)
    def test_browser_orbit_is_default_and_pick_is_temporary(self):
        from abu_m54_browser import M54Browser
        from test_abu_m54 import fixture
        with patch('viser._client_autobuild._check_viser_yarn_running',return_value=False):browser=M54Browser(port=9032)
        try:
            browser.set_scene(fixture());tools=browser.tools
            self.assertIsNone(browser.server.scene._scene_pointer_cb)
            tools.pick.value=True
            for callback in tools.pick._impl.update_cb:callback(SimpleNamespace(client=None))
            self.assertIs(browser.server.scene._scene_pointer_cb,tools.pointer)
            tools.pointer(SimpleNamespace(ray_origin=np.array([10,10,10]),ray_direction=np.array([1,0,0]),client=None))
            self.assertFalse(tools.pick.value);self.assertIsNone(browser.server.scene._scene_pointer_cb)
        finally:browser.close()


if __name__=='__main__':unittest.main()
