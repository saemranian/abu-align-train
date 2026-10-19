"""Assemble only qualified GPU archives for a single offline installer.

This gate deliberately refuses to relabel a CPU build as the full application.
It never recursively copies a build directory, user environment or dataset.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abu_engine_packs import catalog, platform_key, digest


def prepare(archives, destination, evidence):
    specs=[p for p in catalog() if p['platform']==platform_key() and p.get('default_full')]
    if {p.get('capability') for p in specs}!={'alignment','training'}:
        raise ValueError('Full installer requires both qualified alignment and training packs')
    evidence=json.loads(Path(evidence).read_text(encoding='utf-8'))
    verified=[]
    for spec in specs:
        if spec.get('bundle_permitted') is not True:
            raise ValueError('Offline redistribution is not cleared for '+spec['id']+'; use managed upstream download')
        record=evidence.get(spec['id'],{})
        if record.get('archive_sha256')!=spec['sha256'] or record.get('gpu_test_passed') is not True:
            raise ValueError('Missing exact-archive GPU qualification: '+spec['id'])
        if record.get('projection')!='equirectangular' or record.get('cubemaps_used') is not False:
            raise ValueError('Native panorama acceptance required: '+spec['id'])
        if spec['capability']=='training' and any(record.get(k) is not True for k in ('finite_loss','export_valid','resume_passed')):
            raise ValueError('Training/export/resume acceptance missing: '+spec['id'])
        files=[(Path(archives)/(spec['sha256']+'.zip'),spec['sha256'],spec['download_bytes'])]
        for item in spec.get('compliance_assets',[]):
            files.append((Path(archives)/item['filename'],item['sha256'],item['bytes']))
        if spec['capability']=='training' and not spec.get('compliance_assets'):
            raise ValueError('Trainer corresponding-source and license assets are required')
        for path,sha,size in files:
            if not path.is_file() or path.stat().st_size!=size or digest(path)!=sha:
                raise ValueError('Missing/corrupt payload: '+path.name)
            verified.append(path)
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Payload staging must be empty; never mix release versions')
    destination.mkdir(parents=True,exist_ok=True)
    for path in verified:shutil.copy2(path,destination/path.name)
    # The public summary intentionally contains no private image/log paths.
    summary={'schema_version':1,'platform':platform_key(),'packs':[
        {'id':p['id'],'version':p['version'],'sha256':p['sha256']} for p in specs]}
    (destination/'full-payload.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',required=True);parser.add_argument('--destination',required=True)
    parser.add_argument('--evidence',required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.archives,args.destination,args.evidence),indent=2))
