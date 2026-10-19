"""Maintainer-only reproducible Windows GPU trainer builder.

End users install prebuilt components; this program never runs on their first
launch. Builds live outside paths with spaces because upstream pkg-config does
not support those paths. No user environment or image directory is packaged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import sys
import urllib.request
import zipfile

LOCK=json.loads((Path(__file__).parent/'gpu-build-lock.json').read_text(encoding='utf-8'))


def checked_download(url,path,sha,size):
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url,timeout=90) as src,tmp.open('wb') as out:
            shutil.copyfileobj(src,out,length=1024*1024)
        os.replace(tmp,path)
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=sha or path.stat().st_size!=int(size):raise ValueError('Builder download verification failed: '+path.name)


def source_checkout(work,name,url,commit):
    source=work/name
    if not source.exists():
        subprocess.run(['git','clone','--no-checkout',url,str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',commit],check=True)
    actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
    if actual!=commit or changes:raise ValueError('Builder source is not the clean pinned revision: '+name)
    subprocess.run(['git','-C',str(source),'submodule','update','--init','--recursive'],check=True)
    return source


def prepare_sdk(work):
    sdk=work/'cuda';cache=work/'cuda-downloads';sdk.mkdir(exist_ok=True);cache.mkdir(exist_ok=True)
    for name,item in LOCK['cuda_components'].items():
        archive=cache/Path(item['relative_path']).name
        checked_download(LOCK['cuda_origin']+item['relative_path'],archive,item['sha256'],item['size'])
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                rel=PurePosixPath(entry.filename)
                if rel.is_absolute() or '..' in rel.parts or '\\' in entry.filename or ':' in entry.filename:
                    raise ValueError('Unsafe SDK archive member')
                parts=rel.parts[1:]
                if not parts:continue
                target=sdk.joinpath(*parts)
                if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                if target.name.upper().startswith(('LICENSE','EULA')):target=sdk/'licenses'/name/target.name
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(entry) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
    # NVIDIA redist DLLs use bin/x64; upstream portable dependency discovery
    # searches bin. Normalize this private SDK without changing engine sources.
    for library in (sdk/'bin'/'x64').glob('*.dll'):
        shutil.copy2(library,sdk/'bin'/library.name)
    for required in ('bin/nvcc.exe','include/cuda_runtime.h','include/cuda_profiler_api.h','include/nvml.h','include/nvtx3/nvToolsExt.h'):
        if not (sdk/required).is_file():
            raise ValueError('Incomplete isolated CUDA SDK: '+required)
    (sdk/'version.json').write_text(json.dumps({'cuda':{'version':LOCK['cuda_release']}}))
    return sdk


def distribution_manifest(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    spec=json.loads((source/'vcpkg.json').read_text())
    for item in spec['dependencies']:
        if isinstance(item,dict) and item.get('name')=='ffmpeg':
            item['features']=[v for v in item['features'] if v!='x264'];item['default-features']=False
    (destination/'vcpkg.json').write_text(json.dumps(spec,indent=2))
    (destination/'vcpkg-configuration.json').write_text(json.dumps({'overlay-ports':[(source/'.github/overlays').as_posix()]}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',required=True);parser.add_argument('--vs-devcmd',required=True)
    parser.add_argument('--cmake',required=True);parser.add_argument('--ninja-dir',required=True)
    parser.add_argument('--jobs',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.jobs<=16:raise ValueError('Choose 1–16 build workers')
    for value in (args.vs_devcmd,args.cmake,args.ninja_dir):
        if any(c in value for c in '&|<>^%!\r\n"'):raise ValueError('Unsafe build tool path')
    if sys.platform!='win32':raise ValueError('This qualified builder targets Windows x64; Linux needs a separate qualification')
    work=Path(args.work_dir).absolute()
    if any(c in str(work) for c in ' &|<>^%!\r\n"'):raise ValueError('Choose a simple short build path without spaces or shell metacharacters')
    marker=work/'abu-gpu-builder.json'
    if work.exists() and any(work.iterdir()) and not marker.is_file():raise ValueError('Use a new build folder or an existing Abu builder folder')
    work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(work).free<40*1024**3:raise ValueError('GPU builder requires at least 40 GiB free')
    marker.write_text(json.dumps(LOCK,indent=2))
    source=source_checkout(work,'source',LOCK['trainer_repository'],LOCK['trainer_commit'])
    vcpkg=source_checkout(work,'vcpkg',LOCK['vcpkg_repository'],LOCK['vcpkg_commit'])
    sdk=prepare_sdk(work);manifest=work/'distribution-manifest';distribution_manifest(source,manifest)
    for folder in ('downloads','binary-cache','temp'):(work/folder).mkdir(exist_ok=True)
    subprocess.run([str(vcpkg/'bootstrap-vcpkg.bat'),'-disableMetrics'],check=True)
    # Environment changes are local to this build process and its children.
    env=os.environ.copy();env.update(VCPKG_ROOT=str(vcpkg),VCPKG_DOWNLOADS=str(work/'downloads'),
        VCPKG_DEFAULT_BINARY_CACHE=str(work/'binary-cache'),VCPKG_MAX_CONCURRENCY=str(args.jobs),
        CUDA_PATH=str(sdk),TEMP=str(work/'temp'),TMP=str(work/'temp'))
    env['PATH']=str(sdk/'bin')+os.pathsep+args.ninja_dir+os.pathsep+str(Path(args.cmake).parent)+os.pathsep+env['PATH']
    batch=work/'compile.bat'
    batch.write_text('@echo off\n'+f'call "{args.vs_devcmd}" -arch=x64 -host_arch=x64 -vcvars_ver={LOCK["msvc_toolset"]}\n'+
        'if errorlevel 1 exit /b 1\n'+f'cd /d "{source}"\n'+
        f'"{args.cmake}" --preset windows-release -DBUILD_PORTABLE=ON -DBUILD_TESTS=OFF -DLFS_VCPKG_MAX_CONCURRENCY={args.jobs} '+
        f'"-DVCPKG_MANIFEST_DIR={manifest.as_posix()}" "-DCMAKE_CXX_COMPILER=%VCToolsInstallDir%bin/Hostx64/x64/cl.exe" "-DCMAKE_C_COMPILER=%VCToolsInstallDir%bin/Hostx64/x64/cl.exe" "-DCMAKE_CUDA_COMPILER={sdk.as_posix()}/bin/nvcc.exe" "-DCUDAToolkit_ROOT={sdk.as_posix()}" "-DCMAKE_CUDA_ARCHITECTURES={LOCK["cuda_min_sm"]}-virtual"\n'+
        'if errorlevel 1 exit /b 1\n'+f'"{args.cmake}" --build --preset windows-release --parallel {args.jobs}\n'+
        'if errorlevel 1 exit /b 1\n'+f'"{args.cmake}" --install build-windows-release --prefix "{work / "portable"}"\n',encoding='utf-8')
    subprocess.run([str(batch)],check=True,env=env)


if __name__=='__main__':main()
