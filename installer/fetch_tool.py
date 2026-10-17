"""Fetch only a pinned build tool and verify its hash before use."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(name):
    root=Path(__file__).resolve().parents[1]
    item=json.loads((root/'installer/build-tools.json').read_text())[name]
    directory=root/'installer/build/tools';directory.mkdir(parents=True,exist_ok=True)
    path=directory/item['filename']
    if not path.resolve().is_relative_to(directory.resolve()):raise ValueError('Tool path escapes build folder')
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']:return path
    temporary=path.with_suffix('.download')
    with urllib.request.urlopen(item['url'],timeout=90) as response,temporary.open('wb') as stream:
        while chunk:=response.read(1024*1024):stream.write(chunk)
    if hashlib.sha256(temporary.read_bytes()).hexdigest()!=item['sha256']:
        temporary.unlink();raise ValueError('Build tool checksum mismatch; no execution permitted')
    temporary.replace(path);path.chmod(path.stat().st_mode|0o111)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=['inno','appimagetool']);args=parser.parse_args();print(fetch(args.name))
