"""Synthetic worker verification. No user images or external network requests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def viewer_test():
    """Serve the packaged browser assets over loopback without opening a browser."""
    import time
    import urllib.request
    import numpy as np
    from abu_m54 import from_records
    from abu_m54_browser import M54Browser
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    viewer=M54Browser(port=port)
    try:
        viewer.set_scene(from_records({'points':np.array([[0.,0,1],[1,0,2],[0,1,3]])}))
        deadline=time.monotonic()+10
        while True:
            try:
                with urllib.request.urlopen(viewer.url,timeout=2) as response:body=response.read()
                if b'<html' not in body.lower():raise RuntimeError('Viewer did not serve its bundled HTML')
                return {'viewer':'M54','loopback_html':True,'points':3}
            except OSError:
                if time.monotonic()>deadline:raise
                time.sleep(.1)
    finally:viewer.close()


def worker_test():
    import numpy as np
    root=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))
    rng=np.random.default_rng(54)
    shots={f'{i:03}.jpg':{'camera':'pano','rotation':[0,0,0],'translation':[-i*.3,0,0]} for i in range(4)}
    component={'cameras':{'pano':{'projection_type':'spherical','width':1024,'height':512}},'shots':shots,'points':{}}
    for i,xyz in enumerate(rng.uniform([-1,-.7,3],[1,.7,5],(18,3))):
        observations=[]
        for name,shot in shots.items():
            ray=xyz+np.asarray(shot['translation']);ray/=np.linalg.norm(ray)
            observations.append({'shot_id':name,'pixel':[(np.arctan2(ray[0],ray[2])/(2*np.pi)+.5)*1024,(.5+np.arcsin(ray[1])/np.pi)*512]})
        component['points'][str(i)]={'coordinates':xyz.tolist(),'color':[180,220,200],'observations':observations}
    component['shots']['001.jpg']['rotation']=[0,.025,.01]
    with tempfile.TemporaryDirectory(prefix='abu-synthetic-') as temp:
        source=Path(temp)/'source.json';source.write_text(json.dumps([component]),encoding='utf-8')
        output=Path(temp)/'refined.json'
        command=[sys.executable,'-B',str(root/'spherical_refinement.py'),'--input',str(source),'--output',str(output),'--backend','cpu','--iterations','60']
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=90)
        if result.returncode:raise RuntimeError('Bundled refinement worker failed: '+result.stderr[-2000:])
        report=json.loads(output.with_suffix('.report.json').read_text(encoding='utf-8'))
        if not report['after_angular_rms_deg']<report['before_angular_rms_deg']/10:
            raise RuntimeError('CPU refinement did not reduce synthetic angular error')
        if json.loads(source.read_text()) != [component]:raise RuntimeError('Worker changed source geometry')
        return {'worker':'spherical_refinement','before_deg':report['before_angular_rms_deg'],
                'after_deg':report['after_angular_rms_deg'],'backend':report['backend'],'exports':len(report['exports'])}
