import copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from PIL import Image
from manual_camera_geometry import Camera,project,bearings_batch
from manual_image_policy import Masks,effective_controls,stage_masks,validate_policy,select_images
from manual_prediction_engine import PatchMap,PixelMap,transfer,scan
import test_control_points_workspace as fixtures


def planar_scene(folder,count=6):
    w,h=1440,720;rng=np.random.default_rng(54);texture=rng.integers(20,235,(1024,1024,3),dtype='uint8');texture=cv2.GaussianBlur(texture,(5,5),0)
    for i in range(90):
        x,y=rng.integers(30,995,2);cv2.circle(texture,(int(x),int(y)),int(rng.integers(3,18)),tuple(map(int,rng.integers(0,255,3))),-1)
    component=dict(cameras={'s':dict(projection_type='spherical',width=w,height=h)},shots={},points={});cameras={};xyz=np.array([.15,.2,4.])
    yy,xx=np.mgrid[:h,:w];rays=bearings_batch(np.stack([xx,yy],-1),Camera(np.eye(3),np.zeros(3),w,h)).reshape(h,w,3)
    for i in range(count):
        name=f'frame{i}.jpg';center=np.array([i*.28,0.,0.]);cam=Camera(np.eye(3),-center,w,h);cameras[name]=cam;component['shots'][name]=dict(camera='s',rotation=[0,0,0],translation=(-center).tolist())
        depth=np.where(rays[...,2]>.1,4/np.maximum(rays[...,2],.1),-1);world=center+depth[...,None]*rays
        mx=((world[...,0]+4)/8*1024).astype('float32');my=((world[...,1]+3)/6*1024).astype('float32');image=cv2.remap(texture,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(30,30,30));image[depth<0]=30
        Image.fromarray(image).save(folder/name,quality=95)
    point=dict(id='Physical corner',type='tie',observations=[dict(shot_id=n,pixel=project(xyz,cameras[n]),sigma_px=1.) for n in ['frame0.jpg','frame1.jpg']])
    return component,cameras,point,xyz


class PolicyTests(unittest.TestCase):
    def test_wholly_masked_originals_are_omitted_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);masks=folder/'masks';masks.mkdir();paths=[]
            for name in ['empty.jpg','partial.jpg','disabled.jpg','plain.jpg']:
                path=folder/name;Image.new('RGB',(200,100),'orange').save(path);paths.append(path)
            Image.new('L',(200,100),0).save(masks/'empty.mask.png');partial=np.full((100,200),255,dtype='uint8');partial[:,:100]=0;Image.fromarray(partial).save(masks/'partial.mask.png')
            before={p:p.read_bytes() for p in paths};chosen,omitted=select_images(paths,dict(mask_dir=str(masks),excluded_images=['disabled.jpg']))
            self.assertEqual([p.name for p in chosen],['partial.jpg','plain.jpg']);self.assertEqual(omitted,[('empty.jpg','fully masked'),('disabled.jpg','disabled')]);self.assertEqual(before,{p:p.read_bytes() for p in paths})
            records=stage_masks(dict(mask_dir=str(masks)),{p.name:(200,100) for p in chosen},folder/'staged',fill_missing=True);self.assertEqual(list(records),['partial.jpg'])
            with Image.open(folder/'staged/plain.jpg.png') as image:self.assertTrue((np.asarray(image)==255).all())
    def test_uncalibrated_perspective_query_uses_original_pixel_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);rng=np.random.default_rng(55);base=cv2.GaussianBlur(rng.integers(0,255,(720,1200,3),dtype='uint8'),(5,5),0)
            for i in range(3):Image.fromarray(cv2.warpAffine(base,np.array([[1.,0,i*25.],[0,1.,i*10.]]),(1200,720))).save(folder/f'p{i}.jpg',quality=95)
            request=dict(component=None,point=dict(id='Query',type='tie',observations=[dict(shot_id='p0.jpg',pixel=[600.,360.],sigma_px=1.)]),image_dir=str(folder),image_sizes={f'p{i}.jpg':[1200,720] for i in range(3)},projection='perspective',method='sift',scope='all')
            results,report=scan(request);self.assertEqual(report['processed'],2);self.assertEqual(len(results),2)
            for value in results:self.assertLess(np.linalg.norm(np.array(value['pixel'])-[600+int(value['image'][1])*25,360+int(value['image'][1])*10]),1.)
    def test_mask_names_dimensions_inversion_filtering_and_immutable_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);mask=np.full((100,200),255,dtype='uint8');mask[:,90:]=0;Image.fromarray(mask).save(folder/'photo.mask.png')
            policy=dict(mask_dir=str(folder),excluded_images=['disabled.jpg']);m=Masks(policy);self.assertTrue(m.allowed('photo.jpg',[5,20],[200,100]));self.assertFalse(m.allowed('photo.jpg',[110,20],[200,100]));self.assertFalse(m.allowed('disabled.jpg',[5,5],[200,100]))
            with self.assertRaises(ValueError):m.array('photo.jpg',[201,100])
            data=dict(schema_version=1,points=[dict(id='A',observations=[dict(shot_id=n,pixel=[5,20]) for n in ['photo.jpg','other.jpg','disabled.jpg']])],image_policy=policy);before=copy.deepcopy(data)
            result,removed=effective_controls(data,{n:[200,100] for n in ['photo.jpg','other.jpg','disabled.jpg']});self.assertEqual(removed,1);self.assertEqual(len(result['points'][0]['observations']),2);self.assertEqual(data,before)
            records=stage_masks(policy,{'photo.jpg':[200,100]},folder/'staged');self.assertEqual(list(records),['photo.jpg']);self.assertEqual(Image.open(folder/'staged/photo.jpg.png').size,(200,100))
            self.assertFalse(Masks(dict(policy,invert_masks=True)).allowed('photo.jpg',[5,20],[200,100]))
            Image.fromarray(mask).save(folder/'photo.png')
            with self.assertRaises(ValueError):Masks(policy).path('photo.jpg')
            with self.assertRaises(ValueError):validate_policy(dict(excluded_images=['../photo.jpg']))
    def test_DVO_source_target_mask_and_seam_sampling_on_CPU_and_GPU(self):
        import torch
        from manual_dvo_masks import MaskAtlas
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);mask_dir=folder/'masks';mask_dir.mkdir();Image.new('RGB',(256,128)).save(folder/'a.jpg');mask=np.full((128,256),255,dtype='uint8');mask[:,100:150]=0;Image.fromarray(mask).save(mask_dir/'a.mask.png')
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                atlas=MaskAtlas([folder/'a.jpg'],dict(mask_dir=str(mask_dir)),128,256,4,device);points=torch.tensor([[2.,10.],[30.,10.],[66.,10.],[2.,-1.]],device=device)
                self.assertEqual(atlas.allowed(torch.zeros(4,device=device),points).cpu().tolist(),[True,False,True,False])
            with self.assertRaises(ValueError):MaskAtlas([folder/'a.jpg'],dict(mask_dir=str(mask_dir)),128,256,4,'cpu',max_bytes=1)
    def test_defaults_match_verified_numeric_sources_with_explicit_Abu_origins(self):
        import inspect
        from scipy.optimize import least_squares
        from manual_refinement_settings import DEFAULTS,ORIGINS,guide
        signature=inspect.signature(least_squares)
        for key in ['ftol','xtol','gtol']:self.assertEqual(DEFAULTS[key],signature.parameters[key].default)
        self.assertEqual(DEFAULTS['loss_scale'],signature.parameters['f_scale'].default);self.assertEqual(DEFAULTS['min_angle_deg'],1.)
        for key in ORIGINS:self.assertIn('Default source:',guide(key))
    def test_patch_centres_and_transfer_rejects_extrapolation(self):
        camera=Camera(np.eye(3),np.zeros(3),5760,2880)
        for pixel in [[2870,1400],[2,1400],[2900,30]]:np.testing.assert_allclose(PatchMap(camera,pixel).pixels([191.5,191.5]),pixel,atol=1e-8)
        points=np.array([[x,y] for x in [120,160,200,240] for y in [120,160,200,240]],dtype='float32');result=transfer((points,points+[8.,-4.],np.ones(len(points))),np.array([191.5,191.5]));self.assertIsNotNone(result);np.testing.assert_allclose(result[0],[199.5,187.5],atol=.01)
        self.assertIsNone(transfer((points+300,points,np.ones(len(points))),np.array([191.5,191.5])))
    def test_visual_search_all_originals_recovers_held_out_native_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);component,cameras,point,xyz=planar_scene(folder);component['shots'].pop('frame4.jpg');events=[]
            request=dict(component=component,point=point,image_dir=str(folder),image_sizes={n:list(c[2:]) for n,c in cameras.items()},image_policy={'excluded_images':['frame5.jpg']},method='sift',scope='all')
            results,report=scan(request,events.append);self.assertEqual(report['total'],3);self.assertEqual(report['processed'],3);self.assertEqual(report['failures'],0)
            by_name={r['image']:r for r in results};self.assertIn('frame2.jpg',by_name);self.assertIn('frame4.jpg',by_name)
            for name in ['frame2.jpg','frame4.jpg']:self.assertLess(np.linalg.norm(np.array(by_name[name]['pixel'])-project(xyz,cameras[name])),3.)
            self.assertNotIn('frame5.jpg',by_name);self.assertEqual(len(point['observations']),2)
    def test_geometric_predictions_respect_target_exclusions_and_masks(self):
        from control_point_predictions import predicted_candidates
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);component,cameras,point,xyz=planar_scene(folder,4);masks=folder/'masks';masks.mkdir();Image.new('L',(1440,720),0).save(masks/'frame2.mask.png')
            candidates,_=predicted_candidates(point,cameras,dict(mask_dir=str(masks),excluded_images=['frame3.jpg']));self.assertEqual(candidates,[])
    def test_initial_preflight_preserves_raw_controls_and_staged_mask_evidence(self):
        from control_points_pipeline import validate_selected_controls
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);component,cameras,point,xyz=planar_scene(folder,3);point['observations'].append(dict(shot_id='frame2.jpg',pixel=project(xyz,cameras['frame2.jpg']),sigma_px=1.));masks=folder/'masks';masks.mkdir()
            Image.new('L',(1440,720),0).save(masks/'frame0.mask.png');Image.new('L',(1440,720),255).save(masks/'frame1.mask.png')
            policy=dict(mask_dir=str(masks));data=dict(schema_version=1,points=[point],image_policy=policy,image_sizes={n:list(c[2:]) for n,c in cameras.items()});source=folder/'controls.json';source.write_text(json.dumps(data));before=source.read_bytes();selected,omitted=select_images(list(folder.glob('*.jpg')),policy)
            validated=validate_selected_controls(source,selected);self.assertEqual(len(validated['points'][0]['observations']),3)
            stage_masks(policy,{p.name:(1440,720) for p in selected},folder/'staged');validated['image_policy']=dict(mask_dir=str(folder/'staged'),excluded_images=[n for n,_ in omitted]);Image.new('L',(1440,720),0).save(masks/'frame1.mask.png')
            effective,removed=effective_controls(validated,data['image_sizes']);self.assertEqual(removed,1);self.assertEqual({o['shot_id'] for o in effective['points'][0]['observations']},{'frame1.jpg','frame2.jpg'});self.assertEqual(source.read_bytes(),before)
    def test_refinement_filters_camera_and_masked_tracks_without_changing_inputs(self):
        from spherical_refinement import refine
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);component,cameras,point,xyz=planar_scene(folder,3)
            for i,p in enumerate([[-1,-.3,3],[0,0,4],[1,.2,5],[.3,-.4,6],[.8,.5,7]]):component['points'][str(i)]=dict(coordinates=p,color=[160,175,180],observations=[dict(shot_id=n,pixel=project(p,c)) for n,c in cameras.items()])
            masks=folder/'masks';masks.mkdir();image=np.full((720,1440),255,dtype='uint8');image[360,720]=0;Image.fromarray(image).save(masks/'frame0.mask.png')
            control=dict(schema_version=1,points=[point],image_policy=dict(excluded_images=['frame2.jpg'],mask_dir=str(masks)),image_sizes={n:list(c[2:]) for n,c in cameras.items()})
            source=folder/'reconstruction.json';source.write_text(json.dumps([component]));controls=folder/'controls.json';controls.write_text(json.dumps(control));before=source.read_bytes();saved=controls.read_bytes()
            report=refine(source,folder/'refined.json',controls,max_iterations=5,backend='cpu');output=json.loads((folder/'refined.json').read_text())[0]
            self.assertEqual(source.read_bytes(),before);self.assertEqual(controls.read_bytes(),saved);self.assertNotIn('frame2.jpg',output['shots']);self.assertGreater(report['ignored_automatic_measurements'],0)
            self.assertTrue(all(o['shot_id']!='frame2.jpg' for p in output['points'].values() for o in p.get('observations',[])))


class WorkspaceEvidenceTests(unittest.TestCase):
    setUp=fixtures.WorkspaceTests.setUp;tearDown=fixtures.WorkspaceTests.tearDown;wait=fixtures.WorkspaceTests.wait
    def test_image_rows_follow_image_and_M54_selection_and_focus_shortcut(self):
        e=self.editor;e.new();e.measure(e.panes[0],[125,60]);e.measure(e.panes[1],[120,61]);e.highlight_images();self.assertIn('shown_left',e.measurement_tree.item('a.jpg','tags'));self.assertIn('shown_right',e.measurement_tree.item('b.jpg','tags'))
        with patch.object(e.panes[0]['view'],'focus') as focus:e.set_focus(0);self.assertEqual(e.focus_shortcut(),'break');focus.assert_called_once()
        e.set_pane_mode(e.panes[0],'3D');self.wait(lambda:e.panes[0]['scene'].context and e.panes[0]['scene'].data is not None);e.panes[0]['scene'].select_camera('a.jpg');e.highlight_images();self.assertIn('shown_3d',e.measurement_tree.item('a.jpg','tags'))
        before=copy.deepcopy(e.controls['points']);e.measurement_tree.selection_set('a.jpg');e.toggle_measurement_image();self.assertIn('excluded',e.measurement_tree.item('a.jpg','tags'));self.assertEqual(e.controls['points'],before)
        with self.assertRaises(ValueError):e.refinement_payload()
        e.undo();self.assertNotIn('a.jpg',e.controls.get('image_policy',{}).get('excluded_images',[]))
    def test_mask_panel_apply_save_and_undo_preserves_originals(self):
        e=self.editor;folder=self.folder/'masks';folder.mkdir();Image.new('L',(256,128),255).save(folder/'a.mask.png');before=(self.folder/'a.jpg').read_bytes()
        e.open_image_policy();panel=e.image_policy_panel;panel.directory.set(str(folder));panel.apply();self.assertEqual(e.controls['image_policy']['mask_dir'],str(folder));e.controls_path=self.folder/'controls.json';self.assertTrue(e.save())
        self.assertIn('image_policy',json.loads(e.controls_path.read_text()));self.assertEqual((self.folder/'a.jpg').read_bytes(),before);e.undo();self.assertNotIn('image_policy',e.controls)
    def test_unregistered_visual_proposal_stays_unsaved_until_manual_click(self):
        from types import SimpleNamespace
        e=self.editor;e.new();e.measure(e.panes[0],[125,60]);e.measure(e.panes[1],[120,61]);before=copy.deepcopy(e.controls['points']);e.open_predictions();browser=e.prediction_browser;browser.snapshot=browser.signature();browser.state=SimpleNamespace(process=None,finished=True,cancelled=False)
        candidate=dict(event='candidate',image='c.jpg',pixel=[120,60],score=.8,inliers=20,status='Multiple views agree · verify',supporting_views=2,size=[256,128]);browser.events.put((browser.snapshot,candidate));browser.poll()
        self.assertEqual(e.controls['points'],before);self.assertIn('c.jpg',browser.visual);browser.tree.selection_set('c.jpg');browser.target.set('Right');browser.open_selected();self.wait(lambda:e.panes[1]['loaded_name']=='c.jpg');self.assertEqual(e.panes[1]['view'].prediction,[120,60])
    def test_unregistered_measurements_require_initial_align_and_defaults_restore(self):
        e=self.editor;e.new();e.measure(e.panes[0],[125,60]);e.measure(e.panes[1],[120,61]);Image.new('RGB',(256,128),'yellow').save(self.folder/'new.jpg');e.image_sizes['new.jpg']=[256,128];e.image_names.append('new.jpg');e.selected()['observations'].append(dict(shot_id='new.jpg',pixel=[120,60],sigma_px=1.))
        with self.assertRaises(ValueError):e.refinement_payload()
        self.assertEqual(len(e.refinement_payload(for_initial=True)['points'][0]['observations']),3);self.assertEqual(len(e.effective_point(registered_only=True)['observations']),2)
        e.settings['loss_scale'].set(2.);e.settings['gtol'].set(1e-7);e.restore_refine_defaults();self.assertEqual(e.settings['loss_scale'].get(),1.);self.assertEqual(e.settings['gtol'].get(),1e-8);self.assertEqual(e.selected()['observations'][0]['sigma_px'],1.)


if __name__=='__main__':unittest.main()
