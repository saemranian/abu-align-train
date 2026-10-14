"""Local synthetic M54 render/resize benchmark; never opens a client dataset."""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
import tkinter as tk
from types import SimpleNamespace
import numpy as np
import psutil
from PIL import Image


def run(output):
    from abu_m54_host import ViewerHost
    from abu_m54_native import M54NativeViewer
    from abu_m54 import from_records
    from panorama_viewport import PanoramaAsset,PanoramaViewport
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    process=psutil.Process();rss=[];root=tk.Tk();root.geometry('1100x720')
    viewer=M54NativeViewer(ViewerHost(root),root);viewer.pack(fill='both',expand=True);root.update()
    rng=np.random.default_rng(54);points=rng.normal(size=(50000,3))*[3,2,1]
    cameras=[dict(name=f'{i}.jpg',pos=np.array([np.cos(i*.04)*2,np.sin(i*.04)*2,.3]),wxyz=np.array([1.,0,0,0])) for i in range(593)]
    scene=from_records(dict(points=points,colors=np.full((50000,3),180,np.uint8),cameras=cameras,path='synthetic M54 benchmark'))
    viewer.set_data(scene);viewer.ground.set(True);viewer.trajectory.set(True);root.update()
    viewer.gl.glFinish.argtypes=[];frames=[]
    for i in range(60):
        viewer.yaw=i*2;start=time.perf_counter();viewer.draw();viewer.gl.glFinish();frames.append((time.perf_counter()-start)*1000);rss.append(process.memory_info().rss)
    native=dict(renderer=viewer.renderer,opengl=viewer.version,points=50000,cameras=593,frames=60,
                median_ms=statistics.median(frames),p95_ms=float(np.percentile(frames,95)),gpu_completion_synchronized=True)
    viewer.close();viewer.destroy()
    with tempfile.TemporaryDirectory(prefix='m54_synthetic_',dir=output.parent) as directory:
        source=Path(directory)/'synthetic ERP.png';Image.new('RGB',(5760,2880),(110,160,180)).save(source)
        asset=PanoramaAsset(source);pane=PanoramaViewport(root,lambda *a:None,lambda *a:None,lambda *a:None,lambda *a:None);root.update();pane.set_asset(asset)
        times={}
        for resizing in [False,True]:
            samples=[];pane.resizing=resizing
            for i in range(40):
                pane.transform.width=800+i%3*20;pane.transform.height=420;pane.transform.fit()
                start=time.perf_counter();pane.draw();samples.append((time.perf_counter()-start)*1000);rss.append(process.memory_info().rss)
            times['resize_preview' if resizing else 'normal_quality']=dict(median_ms=statistics.median(samples),p95_ms=float(np.percentile(samples,95)))
        times['asset_bytes']=asset.bytes;pane.close()
    report=dict(native=native,panorama=times,observed_rss_max_bytes=max(rss),additional_viewer_vram_peak='not instrumented',
                interpretation='Synthetic single-PC timing; GPU frames include glFinish, panorama timings exclude event dispatch. Not a guaranteed FPS or accuracy benchmark.')
    output.write_text(json.dumps(report,indent=2));root.destroy();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2))
