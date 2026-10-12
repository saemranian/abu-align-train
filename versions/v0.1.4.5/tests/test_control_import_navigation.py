"""CSV interoperability, pre-alignment preparation, browsing and 3D data checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from control_points_csv import read_table, import_table, CSVImportDialog
from control_points_pipeline import validate_selected_controls
from control_points_3d import scene_data
import test_control_points_workspace as workspace_tests


class CSVTests(unittest.TestCase):
    def test_real_format_paths_quotes_bom_and_merge_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'RS.csv'
            path.write_text('"C:\\Folder, name\\a.JPG", "Door corner",12.5,20\nC:\\Elsewhere\\b.jpg,Door corner,10,21\n',encoding='utf-8-sig')
            table=read_table(path);base={'schema_version':1,'points':[],'distances':[]}
            data,stats=import_table(table,base,{'a.jpg':[100,50],'b.jpg':[100,50]})
            self.assertEqual(stats['added_measurements'],2);self.assertEqual(data['points'][0]['id'],'Door corner')
            self.assertEqual(data['points'][0]['observations'][0]['pixel'],[12.5,20.])
            original=copy.deepcopy(data);data['points'][0]['observations'][0]['pixel']=[13,20]
            kept,stats=import_table(table,data,{'a.jpg':[100,50],'b.jpg':[100,50]},kind='xyz',names_only=True)
            self.assertEqual(kept['points'][0]['observations'][0]['pixel'],[13,20])
            replaced,stats=import_table(table,data,{'a.jpg':[100,50],'b.jpg':[100,50]},replace=True)
            self.assertEqual(replaced['points'][0]['observations'][0]['pixel'],[12.5,20.])
            self.assertEqual(base['points'],[])
    def test_header_xyz_units_and_named_ground_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'xyz.csv';path.write_text('Name;X;Y;Alt;Accuracy\nGCP A;1000;2000;3000;20\n')
            data,_=import_table(read_table(path),{'schema_version':1,'points':[],'distances':[]},{},kind='xyz',default_type='check',unit_scale=.001)
            self.assertEqual(data['points'][0]['position'],[1,2,3]);self.assertEqual(data['points'][0]['type'],'check');self.assertEqual(data['points'][0]['sigma_m'],.02)
    def test_space_quoted_names_and_tab_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'space.csv';path.write_text('"C:\\Folder with spaces\\a.jpg" "Point one" 12 20\n"C:\\Folder with spaces\\b.jpg" "Point one" 13 20\n')
            data,_=import_table(read_table(path),{'schema_version':1,'points':[],'distances':[]},{'a.jpg':[100,50],'b.jpg':[100,50]})
            self.assertEqual(len(data['points'][0]['observations']),2)
            path.write_text('Image\tPoint\tX\tY\na.jpg\t门角\t12\t20\n',encoding='utf-16')
            data,_=import_table(read_table(path),{'schema_version':1,'points':[],'distances':[]},{'a.jpg':[100,50]})
            self.assertEqual(data['points'][0]['id'],'门角')
    def test_perspective_pixels_unknown_images_and_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);photo=root/'a.jpg';Image.new('RGB',(50,100)).save(photo)
            csv=root/'rs.csv';csv.write_text(f'{photo},CP,12,20\n')
            table=read_table(csv);self.assertEqual(table['source_shape'],[50,100])
            with self.assertRaisesRegex(ValueError,'perspective'):import_table(table,{'schema_version':1,'points':[],'distances':[]},{'a.jpg':[100,50]})
            names,_=import_table(table,{'schema_version':1,'points':[],'distances':[]},{},names_only=True);self.assertEqual(names['points'][0]['observations'],[])
            csv.write_text('missing.jpg,CP,12,20\n')
            with self.assertRaisesRegex(ValueError,'not uniquely'):import_table(read_table(csv),{'schema_version':1,'points':[],'distances':[]},{'a.jpg':[100,50]})
            csv.write_text('a.jpg,CP,12,20\na.jpg,CP,13,20\n')
            with self.assertRaisesRegex(ValueError,'Conflicting'):import_table(read_table(csv),{'schema_version':1,'points':[],'distances':[]},{'a.jpg':[100,50]})


class WorkflowTests(unittest.TestCase):
    setUp=workspace_tests.WorkspaceTests.setUp
    tearDown=workspace_tests.WorkspaceTests.tearDown
    wait=workspace_tests.WorkspaceTests.wait
    def test_prepare_before_alignment_arrow_navigation_and_attach(self):
        e=self.editor
        with patch('control_points_workspace.filedialog.askdirectory',return_value=str(self.folder)):e.open_images()
        self.assertIsNone(e.reconstruction);self.assertEqual(e.cameras,{})
        self.wait(lambda:all(p['loaded_name'] for p in e.panes));e.new()
        for pane in e.panes:e.measure(pane,[125,60])
        old=e.panes[0]['name'].get();e.panes[0]['view'].arrow(SimpleNamespace(keysym='Right',state=0))
        self.assertNotEqual(old,e.panes[0]['name'].get());self.wait(lambda:e.panes[0]['loaded_name']==e.panes[0]['name'].get())
        calls=[];e.on_attach=lambda *args:calls.append(args);e.controls_path=self.folder/'prepared_controls.json';e.attach_to_main()
        self.assertEqual(calls[0][:2],(str(e.controls_path),str(self.folder)));self.assertTrue(e.controls_path.exists())
        validate_selected_controls(e.controls_path,[self.folder/n for n in ['a.jpg','b.jpg','c.jpg']])
        with self.assertRaisesRegex(ValueError,'not in'):validate_selected_controls(e.controls_path,[self.folder/'b.jpg'])
    def test_grid_filters_selection_and_paged_thumbnail_cache(self):
        e=self.editor;e.new();e.measure(e.panes[0],[125,60]);e.open_grid();g=e.grid
        self.wait(lambda:len(g.photos)==3);g.filter.set('With selected point');g.refresh(reset=True);self.wait(lambda:len(g.photos)==1)
        self.assertEqual(g.names,['a.jpg']);g.target.set('Right');g.select_and_open('a.jpg');self.wait(lambda:e.panes[1]['loaded_name']=='a.jpg')
        g.filter.set('Missing selected point');g.refresh(reset=True);self.assertEqual(g.names,['b.jpg','c.jpg'])
        g.keyboard(1);self.assertIn(g.selected,g.names);self.assertLessEqual(len(g.cache),128);g.close()
    def test_open_alignment_retains_prepared_measurements_for_same_originals(self):
        e=self.editor
        with patch('control_points_workspace.filedialog.askdirectory',return_value=str(self.folder)):e.open_images()
        self.wait(lambda:all(p['loaded_name'] for p in e.panes));e.new()
        for pane in e.panes:e.measure(pane,[125,60])
        before=copy.deepcopy(e.controls);e.controls_path=self.folder/'prepared.json'
        with patch('control_points_workspace.filedialog.askopenfilename',return_value=str(self.rec)):
            e.open()
        self.assertEqual(e.controls,before);self.assertEqual(e.controls_path,self.folder/'prepared.json')
        self.assertEqual(len(e.cameras),3);self.assertTrue(e.document.dirty)
    def test_main_align_actions_fit_minimum_window_width(self):
        from control_points_ui import open_editor
        self.editor.document.mark_saved();self.editor.close()
        self.editor=open_editor(self.root,on_attach=lambda *args:None)
        self.editor.window.geometry('1100x720');self.root.update()
        actions=self.editor.refine_button.master
        for button in actions.winfo_children():
            if button.winfo_class()=='Button' and button!=self.editor.cancel_button:
                self.assertTrue(button.winfo_ismapped())
                self.assertGreaterEqual(button.winfo_width(),button.winfo_reqwidth())
    def test_csv_dialog_preview_is_atomic_and_undoable(self):
        e=self.editor;path=self.folder/'measurements.csv';path.write_text('a.jpg,Door corner,125,60\nb.jpg,Door corner,120,61\n')
        dialog=CSVImportDialog(e,path);self.assertTrue(dialog.preview_import());self.assertEqual(e.controls['points'],[])
        dialog.apply();self.assertEqual(e.selected()['id'],'Door corner');self.assertEqual(len(e.selected()['observations']),2)
        e.undo();self.assertEqual(e.controls['points'],[])
    def test_scene_uses_triangulated_controls_not_survey_frame_guess(self):
        e=self.editor;e.new();xyz=np.array([.3,.2,3.])
        for index,pane in enumerate(e.panes):
            ray=xyz-np.array([index,0,0]);ray/=np.linalg.norm(ray)
            pixel=[(.5+np.arctan2(ray[0],ray[2])/(2*np.pi))*256,(.5+np.arcsin(ray[1])/np.pi)*128]
            e.measure(pane,pixel)
        e.controls['points'][0].update(type='gcp',position=[500000,600000,700000],sigma_m=.01)
        data=scene_data(e.component,e.controls,e.cameras)
        np.testing.assert_allclose(data['controls']['CP001']['position'],xyz,atol=1e-10)
        self.assertEqual(len(data['centers']),3);self.assertEqual(data['untriangulated'],[])


if __name__=='__main__':unittest.main()
