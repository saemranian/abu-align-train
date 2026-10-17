"""Validate candidate payloads and record checksums; packaging is not legal clearance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def verify(folder):
    folder=Path(folder)
    assets=[p for p in folder.iterdir() if p.suffix in {'.exe','.dmg','.AppImage'}]
    if not assets:raise ValueError('No installer assets')
    records=[]
    for asset in assets:
        if not 10*1024*1024 < asset.stat().st_size < 2*1024**3:
            raise ValueError(f'{asset.name}: unexpected core size; refuse a source-only shell or oversized asset')
        digest=hashlib.sha256(asset.read_bytes()).hexdigest()
        records.append({'name':asset.name,'bytes':asset.stat().st_size,'sha256':digest})
    (folder/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in records),encoding='utf-8')
    (folder/'release-manifest.json').write_text(json.dumps({'schema_version':1,'channel':'candidate','stable_qualified':False,'assets':records},indent=2),encoding='utf-8')
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    print(json.dumps(verify(args.folder),indent=2))
