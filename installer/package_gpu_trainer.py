"""Package a built trainer and its corresponding source, never a personal env."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import zipfile

LOCK=json.loads((Path(__file__).parent/'gpu-build-lock.json').read_text())


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def files(root,prune=()):
    skip={'.git','.svn','.hg','__pycache__'}
    for directory,dirs,names in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in skip and not (Path(directory)==Path(root) and d in prune))
        for name in sorted(names):
            if name=='.git':continue
            path=Path(directory)/name
            if path.is_symlink():raise ValueError('Linked file in source/payload: '+str(path))
            yield path


def archive(output,entries):
    records={}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for path,name in entries:
            if name in records:raise ValueError('Duplicate source/payload member: '+name)
            records[name]=sha(path);z.write(path,name)
            if len(records)%1000==0:
                print(f'{output.name}: {len(records)} files archived',flush=True)
        z.writestr('ABU_FILE_MANIFEST.json',json.dumps({'schema_version':1,'files':records},indent=2))
    return dict(filename=output.name,bytes=output.stat().st_size,sha256=sha(output),files=len(records))


def package(portable,source,vcpkg_buildtrees,installed,sdk_licenses,manifest,output,source_url):
    if not source_url.startswith('https://github.com/saemranian/abu-align-train/releases/download/') or not source_url.endswith('/LichtFeld-corresponding-source-Windows-x64.zip'):
        raise ValueError('Provide the immutable matching corresponding-source release asset URL')
    portable=Path(portable).resolve();source=Path(source).resolve();output=Path(output).resolve()
    if any(output.is_relative_to(p) for p in (portable,source,Path(vcpkg_buildtrees).resolve())):
        raise ValueError('Archive output must be separate from build/source inputs')
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    if actual!=LOCK['trainer_commit'] or changes:raise ValueError('Trainer must be the clean pinned upstream revision')
    for required in ('LICENSE','LICENSE-EXCEPTION.txt','THIRD-PARTY-LICENSES.md','bin/LichtFeld-Studio.exe'):
        if not (portable/required).is_file():raise ValueError('Incomplete portable install: '+required)
    for path in portable.rglob('*'):
        if path.is_dir() and path.name.lower() in {'datasets','checkpoints','presets','logs','.git'}:
            raise ValueError('State/data directories cannot be packaged: '+str(path))
        if path.is_file() and path.suffix.lower() in {'.json','.ini','.toml','.yaml','.yml','.txt','.log'}:
            text=path.read_bytes()
            if re.search(rb'documents[/\\]+works[/\\]+clients[/\\]|private-training-acceptance',text,re.IGNORECASE):
                raise ValueError('Private run state in portable payload; value suppressed')
    copyright_file=Path(installed)/'share/ffmpeg/copyright'
    if not copyright_file.is_file() or 'LESSER GENERAL PUBLIC LICENSE' not in copyright_file.read_text(errors='replace'):
        raise ValueError('Distribution requires the selected LGPL FFmpeg build')
    if any((Path(installed)/'bin').glob('*264*.dll')):
        raise ValueError('Unexpected GPL x264 binary in the selected triplet')
    output.mkdir(parents=True,exist_ok=True)
    runtime_entries=[(p,p.relative_to(portable).as_posix()) for p in files(portable) if p.suffix.lower() not in {'.pdb','.ilk','.log'}]
    runtime_entries.extend((p,'NVIDIA_LICENSES/'+p.relative_to(sdk_licenses).as_posix()) for p in files(Path(sdk_licenses)))
    source_entries=[]
    # The upstream checkout includes its fetched CMake dependencies under external/.
    # Build directories, Git objects, client data and acceptance logs are excluded.
    for path in files(source,{'build','build-debug','build-windows-release','build-windows-relwithdebinfo','vcpkg_installed','_deps'}):
        if path.suffix.lower() not in {'.obj','.o','.pdb','.ilk','.pyc','.log'}:
            source_entries.append((path,'lichtfeld/'+path.relative_to(source).as_posix()))
    shared=Path(installed)/'share';buildtrees=Path(vcpkg_buildtrees)
    for pkg in sorted(p for p in shared.iterdir() if p.is_dir()):
        # Only dependencies in this exact selected triplet, never unrelated caches.
        folder=buildtrees/pkg.name/'src'
        if folder.is_dir():
            source_entries.extend((p,'dependencies/'+pkg.name+'/'+p.relative_to(folder).as_posix()) for p in files(folder))
        recipes=buildtrees/'versioning_'/'versions'/pkg.name
        if recipes.is_dir():
            source_entries.extend((p,'dependency-recipes/'+pkg.name+'/'+p.relative_to(recipes).as_posix()) for p in files(recipes))
        for p in files(pkg):source_entries.append((p,'dependency-notices/'+pkg.name+'/'+p.relative_to(pkg).as_posix()))
    source_entries.extend((p,'cuda-licenses/'+p.relative_to(sdk_licenses).as_posix()) for p in files(Path(sdk_licenses)))
    source_entries.extend((p,'distribution-build/'+p.name) for p in Path(manifest).iterdir() if p.is_file())
    for name in ('build_gpu_trainer.py','gpu-build-lock.json'):
        source_entries.append((Path(__file__).parent/name,'distribution-build/'+name))
    source_record=archive(output/'LichtFeld-corresponding-source-Windows-x64.zip',source_entries)
    offer=output/'SOURCE_OFFER.txt'
    offer.write_text('LichtFeld Studio is GPL-3.0-or-later with its included NVIDIA library exception.\n'
        'The matching corresponding source, selected dependency sources, build recipes and notices are available alongside this binary:\n'
        +source_url+'\nSHA256: '+source_record['sha256']+'\n'
        'Upstream revision: '+actual+'\n'
        'You may modify and rebuild the GPL/LGPL components under their respective licenses.\n'
        'LGPL shared libraries can be replaced in bin with ABI-compatible rebuilt versions.\n'
        'NVIDIA libraries retain their separate included terms; the source offer does not relicense them.\n'
        'See LICENSE, LICENSE-EXCEPTION.txt, THIRD-PARTY-LICENSES.md and NVIDIA_LICENSES.\n',encoding='utf-8')
    runtime_entries.append((offer,'SOURCE_OFFER.txt'))
    runtime_record=archive(output/'LichtFeld-GPU-Windows-x64.zip',runtime_entries)
    record={'schema_version':1,'upstream_commit':actual,'upstream_source_modified':False,
            'runtime':runtime_record,'corresponding_source':source_record,
            'qualification':'pending real GPU training/export/resume acceptance'}
    (output/'trainer-package-manifest.json').write_text(json.dumps(record,indent=2))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('portable','source','vcpkg-buildtrees','installed','sdk-licenses','manifest','output','source-url'):parser.add_argument('--'+flag,required=True)
    a=parser.parse_args();print(json.dumps(package(a.portable,a.source,a.vcpkg_buildtrees,a.installed,a.sdk_licenses,a.manifest,a.output,a.source_url),indent=2))
