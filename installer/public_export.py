"""Explicit source export; never publish the developer workspace or backup tree wholesale."""
import hashlib
import json
from pathlib import Path
import re
import shutil

TOKEN=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,}|hf_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
PRIVATE_PATH=re.compile(rb'documents[/\\]+works[/\\]+clients[/\\]',re.IGNORECASE)


def export(root, destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    spec=json.loads((root/'installer/public-files.json').read_text(encoding='utf-8'))
    written={}
    for relative in spec['files']:
        rel=Path(relative);source=root/rel;target=destination/rel
        if rel.is_absolute() or '..' in rel.parts or not source.resolve().is_relative_to(root):raise ValueError('Invalid export source')
        if source.is_symlink() or not source.is_file():raise ValueError(f'Missing or linked source: {relative}')
        if not target.resolve().is_relative_to(destination):raise ValueError('Invalid export destination')
        data=source.read_bytes()
        if TOKEN.search(data):raise ValueError(f'Credential pattern in {relative}; value suppressed')
        if source.suffix.lower() not in {'.ico','.png'} and PRIVATE_PATH.search(data):
            raise ValueError(f'Private client path in {relative}; redact it before publication')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        written[relative]=hashlib.sha256(data).hexdigest()
    (destination/'PUBLIC_SOURCE_MANIFEST.json').write_text(json.dumps({'schema_version':1,'files':written},indent=2)+'\n',encoding='utf-8')
    return written
