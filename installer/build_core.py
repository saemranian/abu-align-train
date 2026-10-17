"""Build a CPU core containing its interpreter, native libraries and approved files."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'installer/build'
DIST=ROOT/'installer/dist'


def approved_files(root=ROOT, manifest='installer/core-files.json'):
    files=json.loads((root/manifest).read_text(encoding='utf-8'))['files']
    if len(set(files))!=len(files):raise ValueError('Duplicate payload entry')
    result=[]
    for value in files:
        relative=Path(value)
        original=root/relative
        path=original.resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root.resolve()):
            raise ValueError('Payload path escapes source directory')
        if not path.is_file() or original.is_symlink():raise ValueError(f'Missing or linked payload: {value}')
        if any(part in {'.git','.agents','.codex','versions','datasets','checkpoints','presets','logs','examples'} for part in relative.parts):
            raise ValueError(f'Private or historical path in payload: {value}')
        result.append((path,relative))
    return result


def compliance(destination):
    destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for dist in sorted(metadata.distributions(),key=lambda d:d.metadata.get('Name','').lower()):
        name=dist.metadata.get('Name','unknown')
        if name.lower() in {'pip','pip-tools','build','wheel','pyproject-hooks'}:continue
        record={'type':'library','name':name,'version':dist.version,
                'purl':f'pkg:pypi/{name.lower().replace("_","-")}@{dist.version}',
                'licenses':[{'license':{'name':dist.metadata.get('License-Expression') or dist.metadata.get('License') or 'Review required'}}]}
        records.append(record)
        for item in dist.files or []:
            if not any(word in Path(item).name.lower() for word in ['license','copying','copyright','notice']):continue
            source=Path(dist.locate_file(item))
            if not source.is_file() or source.suffix.lower() in {'.py','.pyc','.pyd','.so','.dll'}:continue
            target=destination/name/str(item).replace('..','_').replace(':','_')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for filename in ['LICENSE.txt','LICENSE_PYTHON.txt','LICENSE_PYTHON']:
        source=Path(sys.base_prefix)/filename
        if source.is_file():shutil.copy2(source,destination/('Python-'+filename))
    (destination/'sbom.cdx.json').write_text(json.dumps({'bomFormat':'CycloneDX','specVersion':'1.6','version':1,'components':records},indent=2),encoding='utf-8')


def build():
    # Conda-based developer interpreters keep Python's dependent DLLs here.
    # Add the search directory, not every library; PyInstaller resolves only dependencies.
    if sys.platform=='win32':
        native=Path(sys.base_prefix)/'Library/bin'
        if native.is_dir():os.environ['PATH']=str(native)+os.pathsep+os.environ.get('PATH','')
    import PyInstaller.__main__
    files=approved_files();BUILD.mkdir(parents=True,exist_ok=True);DIST.mkdir(exist_ok=True)
    licenses=BUILD/'compliance';compliance(licenses)
    args=[str(ROOT/'abu_runtime.py'),'--name','AbuRuntime','--onedir','--console','--noconfirm',
          '--distpath',str(DIST),'--workpath',str(BUILD/'pyinstaller'),'--specpath',str(BUILD),
          '--paths',str(ROOT),'--noupx','--collect-data','viser','--collect-binaries','pycolmap',
          '--add-data',f'{licenses}:THIRD_PARTY_LICENSES']
    if sys.platform=='win32':args+=['--icon',str(ROOT/'abu_icon.ico')]
    # Dependencies such as imageio query their distribution metadata at runtime.
    # Preserve installed metadata as well as importable modules in the frozen app.
    for dist in metadata.distributions():
        name=dist.metadata.get('Name')
        if name and name.lower() not in {'pip','pip-tools','build','wheel','pyproject-hooks'}:
            args+=['--copy-metadata',name]
    for source,relative in files:
        args+=['--add-data',f'{source}:{str(relative.parent)}']
        if relative.suffix=='.py':args+=['--hidden-import','.'.join(relative.with_suffix('').parts)]
    for name in ['torch','torchvision','torchaudio','sam2','vggt','vggt_omega','vggttt','lightglue','transformers','nerfstudio','pyceres','IPython','pytest','viser.extras']:
        args+=['--exclude-module',name]
    PyInstaller.__main__.run(args)
    bundle=DIST/'AbuRuntime'
    # Ship the wheel's prebuilt browser assets only. Leaving client/src present
    # makes Viser consider rebuilding with Node on an end user's first launch.
    client=bundle/'_internal/viser/client'
    if not (client/'build/index.html').is_file():raise RuntimeError('Prebuilt Viser browser assets are missing')
    source=client/'src'
    if source.is_dir():
        if not source.resolve().is_relative_to(bundle.resolve()):raise RuntimeError('Invalid generated browser path')
        shutil.rmtree(source)
    record={'schema_version':1,'version':(ROOT/'VERSION.txt').read_text().strip(),
            'edition':'CPU core candidate','system':sys.platform,'architecture':platform.machine(),
            'source_files':{str(r).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p,r in files},
            'restrictions':['No neural checkpoints or training runtimes bundled','Mac/Linux native viewport uses browser fallback','Signing and clean-machine qualification required before stable release']}
    (bundle/'build-manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    binary=bundle/('AbuRuntime.exe' if sys.platform=='win32' else 'AbuRuntime')
    subprocess.run([str(binary),'--self-test'],check=True,cwd=BUILD)
    print(f'BUNDLED_CORE={bundle}')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');options=parser.parse_args()
    if options.validate_only:print(json.dumps({'approved_files':len(approved_files())}))
    else:build()
