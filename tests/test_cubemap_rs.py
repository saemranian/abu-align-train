import os
import tempfile
import shutil
import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import panoramic_splitter as ps

tmp_dir = tempfile.mkdtemp(prefix='test_rs_split_')
try:
    in_dir = os.path.join(tmp_dir, 'in')
    out_dir = os.path.join(tmp_dir, 'out')
    os.makedirs(in_dir)

    # 360 panorama 2048x1024
    img = np.zeros((1024, 2048, 3), dtype=np.uint8)
    cv2.circle(img, (1024, 512), 200, (255, 200, 100), -1)
    cv2.imwrite(os.path.join(in_dir, 'capture_001.jpg'), img)

    res = ps.split_panorama_dataset(
        input_dir=in_dir,
        output_dir=out_dir,
        mode='cubemap',
        fov_deg=114.0,
        crop_size=1024,
        export_xmp=True,
        subfolder_per_perspective=False
    )
    print('Split result:', res['success'], 'Total views:', res['total_views'])

    imgs_dir = os.path.join(out_dir, 'images')
    xmp_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith('.xmp')])
    print('Generated XMP count:', len(xmp_files))
    assert len(xmp_files) == 6

    for xmp_f in xmp_files:
        with open(os.path.join(imgs_dir, xmp_f), encoding='utf-8') as f:
            content = f.read()
        print(f'=== {xmp_f} ===\n{content.strip()}\n')
        assert 'xcr:FocalLength35mm="11.689337"' in content
        assert 'xcr:CalibrationPrior="exact"' in content
        assert 'xcr:DistortionModel="brown3"' in content
        assert 'xcr:Rotation' not in content
        assert 'xcr:Position' not in content
        assert 'xcr:Rig' not in content
        assert 'xcr:LockedPoseGroup' not in content

    print('ALL CUBEMAP REALITYSCAN CHECKS PASSED PERFECTLY!')
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
