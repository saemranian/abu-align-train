"""Build the qualified edition; never silently downgrade a broken GPU catalog."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from abu_engine_packs import catalog,platform_key


def edition():
    packs=[p for p in catalog() if p['platform']==platform_key() and p.get('default_full')]
    training=[p for p in packs if p.get('capability')=='training']
    if not training:return {'edition':'cpu-core-candidate','gpu_setup':False}
    if not any(p.get('capability')=='alignment' for p in packs):raise ValueError('Missing default GPU alignment pack')
    for p in training:
        q=p.get('qualification',{})
        if q.get('archive_sha256')!=p['sha256']:
            raise ValueError('GPU acceptance does not match the trainer archive')
        if any(q.get(k) is not True for k in ('native_equirectangular','gpu_training','finite_loss','gaussian_export','resume')):
            raise ValueError('Trainer has not passed native GPU qualification: '+p['id'])
        if not p.get('compliance_assets'):raise ValueError('Trainer source/license assets missing')
    return {'edition':'windows-nvidia-gpu-candidate','gpu_setup':True,
            'setup_internet_required':True,'packs':[{'id':p['id'],'version':p['version'],'sha256':p['sha256']} for p in packs]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iscc',required=True);parser.add_argument('--app-id')
    a=parser.parse_args();info=edition()
    args=[a.iscc,'/DMyAppVersion='+ (ROOT/'VERSION.txt').read_text().strip()]
    if info['gpu_setup']:args+=['/DFullGpuOnline']
    if a.app_id:args+=['/DMyAppId='+a.app_id]
    args+=[str(ROOT/'installer/windows/abu_align_setup.iss')]
    subprocess.run(args,check=True,cwd=ROOT)
    target=ROOT/'installer/build/windows-edition.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))


if __name__=='__main__':main()
