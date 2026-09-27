"""Embedded 3D workspace layout, navigation, and comparison regressions."""
import tkinter as tk
import tempfile
import time
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from workspace_3d import Workspace3DViewer


def layer(offset=0):
    points=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,1]])+offset
    return dict(points=points,colors=np.full((4,3),180,np.uint8),
        cameras=[dict(name=f'{i}.jpg',pos=p.copy(),wxyz=np.array([1.,0,0,0])) for i,p in enumerate(points)],
        count=4,path='fixture',source='Synthetic')


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.root.geometry('960x720')
        from desktop_ui import configure_theme
        configure_theme(SimpleNamespace(root=self.root))
        self.ui=Workspace3DViewer(self.root)
        self.ui.pack(fill='both',expand=True)
        self.ui.layers={'A':layer(),'B':layer(2)}
        self.errors=[]
        self.root.report_callback_exception=lambda *e:self.errors.append(e)
        self.ui.apply_comparison();self.ui.reset_view();self.root.update()
        self.addCleanup(self.root.destroy)

    def test_profiles_fit_visible_content_and_disable_hidden_camera_picking(self):
        self.ui.inspection_profile.set('Geometry')
        self.ui.apply_profile()
        self.root.update()
        self.assertEqual(self.ui._camera_screen, [])
        self.assertTrue(self.ui.show_points_var.get())
        self.ui.inspection_profile.set('Cameras')
        self.ui.apply_profile()
        self.assertFalse(self.ui.show_points_var.get())
        self.assertTrue(self.ui.show_trajectory_var.get())
        self.ui.points = np.array([[1000.,1000,1000]])
        self.ui.reset_view()
        self.assertLess(np.linalg.norm(self.ui.center), 5)
        self.assertEqual(self.errors, [])

    def test_quiet_defaults_and_fixed_view_switch(self):
        center = self.ui.center.copy()
        self.ui.yaw = 42
        self.ui.switch_comparison()
        self.assertEqual(self.ui.compare_mode.get(), 'B only')
        self.ui.switch_comparison()
        self.assertEqual(self.ui.compare_mode.get(), 'A only')
        np.testing.assert_equal(self.ui.center, center)
        self.assertEqual(self.ui.yaw, 42)
        self.assertFalse(self.ui.auto_open_photo_var.get())
        self.ui.toggle_inspector()
        self.root.update()
        self.assertEqual(self.ui.inspect_toggle.cget('text'), 'Hide inspect')
        for child in self.ui.inspector_bar.winfo_children():
            self.assertLessEqual(child.winfo_x()+child.winfo_width(), self.ui.inspector_bar.winfo_width())

    def test_photo_dialog_and_comparison_activation(self):
        from native_inspection import CameraPhotoViewer
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            photo = Path(directory)/'test.jpg'
            Image.new('RGB', (100, 50)).save(photo)
            window = CameraPhotoViewer(self.root)
            window.show_camera(self.ui.cameras[0], str(photo), 0, 4)
            self.root.update()
            self.assertEqual(window._raw_pil.size, (100, 50))
            window.destroy()
        self.ui.toggle_comparison()
        self.assertEqual(self.ui.compare_mode.get(), 'Overlay')
        self.assertEqual(len(self.ui.points), 8)
        self.assertEqual(self.errors, [])

    def test_minimum_layout_with_comparison_expanded(self):
        self.ui.toggle_comparison();self.root.update()
        self.assertGreater(self.ui.canvas.winfo_height(),200)
        self.ui.toggle_inspector();self.root.update()
        self.assertGreater(self.ui.canvas.winfo_height(),150)
        for bar in (self.ui.navigation_bar,self.ui.layer_bar):
            for child in bar.winfo_children():
                self.assertLessEqual(child.winfo_x()+child.winfo_width(),bar.winfo_width())
        self.assertEqual(self.errors,[])

    def test_overlay_registration_and_viewpoint_preservation(self):
        original=self.ui.layers['B']['points'].copy()
        self.ui.yaw=73.;center=self.ui.center.copy()
        self.ui.compare_mode.set('Overlay');self.ui.register_runs.set(True)
        self.ui.apply_comparison();self.root.update()
        self.assertEqual(len(self.ui.points),8)
        np.testing.assert_allclose(self.ui.points[:4],self.ui.points[4:],atol=1e-10)
        np.testing.assert_equal(self.ui.layers['B']['points'],original)
        np.testing.assert_equal(self.ui.center,center)
        self.assertEqual(self.ui.yaw,73)
        self.assertIn('Registered',self.ui.compare_info.cget('text'))
        self.assertEqual(self.errors,[])

    def test_bad_registration_retains_scene(self):
        before=self.ui.points.copy()
        self.ui.layers['B']['cameras']=[]
        self.ui.register_runs.set(True);self.ui.apply_comparison()
        np.testing.assert_equal(self.ui.points,before)
        self.assertIn('Previous scene retained',self.ui.compare_info.cget('text'))

    def test_camera_selection_lookthrough_and_view_restore(self):
        self.ui.save_view();center=self.ui.center.copy()
        self.ui.step_camera(1)
        self.ui.look_through();self.root.update()
        self.assertIsNotNone(self.ui.view_matrix)
        self.ui.restore_view()
        np.testing.assert_equal(self.ui.center,center)
        self.assertIsNone(self.ui.view_matrix)
        self.ui._on_mouse_wheel(SimpleNamespace(delta=0))
        self.assertEqual(self.errors,[])

    def test_offscreen_depth_render_is_safe(self):
        self.ui.color_mode_var.set('depth');self.ui.pan_x=10000
        self.ui.render();self.root.update()
        self.assertEqual(self.errors,[])

    def test_fit_places_geometry_inside_viewport_margins(self):
        from workspace_3d import make_rotation_matrix
        self.ui.reset_view()
        p=(self.ui.points-self.ui.center)/self.ui.scale
        p=p@make_rotation_matrix(self.ui.yaw,self.ui.pitch).T
        width,height=self.ui.canvas.winfo_width(),self.ui.canvas.winfo_height()
        focal=min(width,height)*1.05
        x=p[:,0]/(p[:,2]+self.ui.distance)*focal
        y=p[:,1]/(p[:,2]+self.ui.distance)*focal
        self.assertLessEqual(np.abs(x).max(),width*.43)
        self.assertLessEqual(np.abs(y).max(),height*.43)

    def test_point_cloud_distance_legend(self):
        self.ui.compare_mode.set('B only');self.ui.distance_colors.set(True)
        self.ui.apply_comparison()
        self.assertIn('Cloud B → A',self.ui.compare_info.cget('text'))
        self.assertEqual(len(self.ui.colors),len(self.ui.points))

    def test_background_loader_discovers_cloud_without_blocking_tk(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'results'/'sparse_fixture';run.mkdir(parents=True)
            (run/'points3D.ply').write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n')
            self.ui.dataset_provider=lambda:folder
            self.ui.load_selected_run()
            deadline=time.monotonic()+5
            while self.ui._loading and time.monotonic()<deadline:
                self.root.update()
                time.sleep(.01)
            self.assertFalse(self.ui._loading)
            self.assertEqual(len(self.ui.points),3)
            self.assertEqual(self.ui.active_run_name,'sparse_fixture')
            self.assertEqual(self.errors,[])


if __name__=='__main__':
    unittest.main()
