"""Derivative, CUDA/CPU accuracy, exports, responsive workers and local 3D."""
import copy
import csv
import functools
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize._numdiff import approx_derivative
from spherical_acceleration import BearingProblem,cuda_available,retriangulate_batch
from spherical_refinement import refine,pixel_bearing
from refined_exports import export_refined
from control_point_predictions import predicted_candidates
import test_control_points_workspace as workspace_tests


class DerivativeTests(unittest.TestCase):
    def test_batched_retriangulation_matches_scalar_and_rejects_parallel_rays(self):
        from spherical_refinement import triangulate
        from control_points_model import projected_pixel
        rng=np.random.default_rng(99);cameras={str(i):(np.eye(3),np.array([-i*.3,0.,0.]),1024,512) for i in range(4)}
        observations={str(i):[(n,projected_pixel(xyz,c),1.) for n,c in cameras.items()] for i,xyz in enumerate(rng.uniform([-1,-1,3],[1,1,5],(20,3)))}
        observations['parallel']=[(n,[512,256],1.) for n in cameras]
        names=list(observations);xyz,valid=retriangulate_batch(names,observations,cameras)
        self.assertTrue(valid[:-1].all());self.assertFalse(valid[-1])
        for i,name in enumerate(names[:-1]):np.testing.assert_allclose(xyz[i],triangulate(observations[name],cameras),atol=1e-10)
    def test_bearing_and_metric_jacobians_against_independent_differences(self):
        rng=np.random.default_rng(123);names=['a','b','c'];points=['0','1','2']
        cameras={n:(Rotation.from_rotvec(rng.normal(0,.4,3)).as_matrix(),np.array([-i,.1,0]),1024,512) for i,n in enumerate(names)}
        xyz=rng.uniform([-1,-1,3],[1,1,5],(3,3));measurements=[(p,n,pixel_bearing([510,255],1024,512),.01) for p in points for n in names]
        for gcp,anchor in [(False,True),(True,False)]:
            free=names if gcp else names[1:]
            gcps={'0':{'position':xyz[0].tolist(),'sigma_m':.02}} if gcp else {}
            distances=[{'a':'1','b':'2','meters':2.,'sigma_m':.03}]
            baseline=np.linalg.norm(-cameras['c'][0].T@cameras['c'][1]+cameras['a'][0].T@cameras['a'][1])
            problem=BearingProblem(cameras,free,points,measurements,gcps,distances,'c',baseline,anchor)
            x=np.r_[np.concatenate([np.r_[Rotation.from_matrix(cameras[n][0]).as_rotvec(),cameras[n][1]] for n in free]),xyz.ravel()]
            numerical=approx_derivative(problem.evaluate,x,method='3-point')
            _,analytic=problem.evaluate(x,jac=True)
            np.testing.assert_allclose(analytic.toarray(),numerical,atol=2e-6,rtol=2e-6)

    @unittest.skipUnless(cuda_available(),'CUDA unavailable')
    def test_cuda_accuracy_with_ground_controls_and_independent_checks(self):
        import test_spherical_refinement as original
        with patch.object(original,'refine',functools.partial(refine,backend='cuda')):
            original.TestSphericalRefinement('test_joint_refinement_reduces_independent_error').test_joint_refinement_reduces_independent_error()


class InteractionTests(unittest.TestCase):
    setUp=workspace_tests.WorkspaceTests.setUp
    tearDown=workspace_tests.WorkspaceTests.tearDown
    wait=workspace_tests.WorkspaceTests.wait
    def measure_geometry(self):
        e=self.editor;e.new();xyz=np.array([.3,.2,3.])
        from control_points_model import projected_pixel
        for pane in e.panes:e.measure(pane,projected_pixel(xyz,e.cameras[pane['loaded_name']]))
    def test_prediction_list_focus_and_logs(self):
        self.measure_geometry();e=self.editor;rows,description=predicted_candidates(e.selected(),e.cameras)
        self.assertEqual([r['image'] for r in rows],['c.jpg'])
        e.open_predictions();browser=e.prediction_browser;browser.tree.selection_set('c.jpg');browser.open_selected()
        self.wait(lambda:e.panes[0]['loaded_name']=='c.jpg');self.assertIsNotNone(e.panes[0]['view'].prediction)
        self.assertEqual(len(e.selected()['observations']),2)
        log=[json.loads(line) for line in e.audit.path.read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(row['event']=='document_changed' for row in log));self.assertTrue(any(row['event']=='prediction_open' for row in log))
    def test_continuous_grid_has_all_images_but_bounded_live_tiles(self):
        e=self.editor
        # Hundreds of logical names exercise virtualization without decoding a dataset.
        e.image_names=[f'{i:04}.jpg' for i in range(600)];e.image_sizes={n:[256,128] for n in e.image_names}
        e.open_grid();g=e.grid;self.root.update();self.assertEqual(len(g.names),600);self.assertLess(len(g.tiles),80)
        g.canvas.yview_moveto(1);g.arrange();self.root.update();self.assertIn('0599.jpg',g.tiles);self.assertLess(len(g.tiles),80)
        g.selected='0599.jpg';g.keyboard(-1);self.assertEqual(g.selected,'0598.jpg');self.assertLessEqual(len(g.cache),128)
    def test_real_refinement_worker_keeps_ui_polling_and_exports(self):
        self.measure_geometry();e=self.editor;out=self.folder/'worker.json';e.settings['backend'].set('cpu');e.settings['iterations'].set(5)
        ticks=[]
        def tick():
            ticks.append(time.monotonic())
            if e.busy:self.root.after(25,tick)
        with patch('control_points_workspace.filedialog.asksaveasfilename',return_value=str(out)):
            e.refine();tick();self.wait(lambda:not e.busy)
        self.assertEqual(e.last_result,out);self.assertGreater(len(ticks),3);self.assertLess(max(np.diff(ticks)),.5)
        self.assertTrue(out.with_suffix('.points.ply').exists());self.assertTrue(out.with_suffix('.cameras.csv').exists())
        self.assertEqual(json.loads(out.with_suffix('.controls.json').read_text())['points'][0]['observations'],e.controls['points'][0]['observations'])
    def test_native_gpu_navigator_selects_controls_and_cameras(self):
        self.measure_geometry();e=self.editor;e.component['points']={'demo':{'coordinates':[.3,.2,3.],'color':[200,120,90]}}
        e.open_3d();self.wait(lambda:e.native_3d.data is not None);self.wait(lambda:e.native_3d.context is not None)
        nav=e.native_3d;self.assertNotIn('GDI Generic',nav.renderer)
        e.active.set('');nav.pick(*nav.project(nav.control_xyz)[0][0]);self.assertEqual(e.active.get(),'CP001')
        nav.controls.set(False);nav.pick(*nav.project(nav.camera_xyz)[0][0]);self.assertEqual(nav.camera_selected,'a.jpg')
    def test_cancel_stops_isolated_worker_and_preserves_source(self):
        import sys
        e=self.editor;before=self.rec.read_bytes();e.set_busy(True)
        e.launch_worker([sys.executable,'-B','-c','import time; print("started",flush=True); time.sleep(30)'],'export',self.rec)
        self.wait(lambda:e.worker is not None);e.cancel_refinement();self.wait(lambda:not e.busy)
        self.assertEqual(self.rec.read_bytes(),before)
        events=[json.loads(line)['event'] for line in e.audit.path.read_text(encoding='utf-8').splitlines()]
        self.assertIn('worker_cancelled',events)
    def test_export_preserves_pose_and_point_frame(self):
        self.component['points']={'1':{'coordinates':[1,2,3],'color':[120,130,140]}};self.rec.write_text(json.dumps([self.component]))
        before=self.rec.read_bytes();result=export_refined(self.rec)
        self.assertEqual(self.rec.read_bytes(),before);self.assertEqual(result['points'],1)
        with Path(result['camera_alignment_csv']).open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),3);self.assertEqual(float(rows[1]['center_x']),1.);self.assertEqual(float(rows[1]['camera_to_world_qw']),1.)
        from plyfile import PlyData
        vertices=PlyData.read(result['point_cloud'])['vertex'].data
        self.assertEqual(tuple(vertices[0])[:3],(1,2,3))


if __name__=='__main__':unittest.main()
