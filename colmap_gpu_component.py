"""Official COLMAP CUDA process adapter; no upstream modifications."""
import os
from pathlib import Path
import subprocess


def installed_executable():
    from abu_engine_packs import executable, platform_key, catalog
    if not any(p['id']=='colmap-cuda' and p['platform']==platform_key() for p in catalog()):
        return None
    return executable('colmap-cuda', 'colmap')


def extract_and_match(executable, database, image_root, names, output, width, features, mask_path=None):
    output=Path(output)
    listing=output/'selected_images.txt'
    if any('\n' in name or '\r' in name for name in names):
        raise ValueError('COLMAP image names cannot contain line breaks')
    listing.write_text('\n'.join(names)+'\n',encoding='utf-8')
    common=['--database_path',str(database),'--log_target','stdout','--log_color','0']
    extract=[executable,'feature_extractor',*common,'--image_path',str(image_root),
             '--image_list_path',str(listing),'--ImageReader.camera_model','EQUIRECTANGULAR',
             '--ImageReader.single_camera','1','--FeatureExtraction.type','SIFT',
             '--FeatureExtraction.use_gpu','1','--FeatureExtraction.gpu_index','0',
             '--FeatureExtraction.num_threads','2','--FeatureExtraction.max_image_size',str(width),
             '--SiftExtraction.max_num_features',str(features)]
    if mask_path:
        extract+=['--ImageReader.mask_path',str(mask_path)]
    match=[executable,'sequential_matcher',*common,'--FeatureMatching.use_gpu','1',
           '--FeatureMatching.gpu_index','0','--FeatureMatching.num_threads','2',
           '--FeatureMatching.guided_matching','0','--SequentialMatching.overlap','10',
           '--SequentialMatching.loop_detection','0']
    env=os.environ.copy()
    # Keep the official archive's dependency search local to its own process.
    env['PATH']=str(Path(executable).parent)+os.pathsep+env.get('PATH','')
    for command in (extract,match):
        print('GPU component:',command[1],flush=True)
        subprocess.run(command,check=True,env=env,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
