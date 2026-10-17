"""Validate GitHub workflow syntax and contexts before any public Git mutation."""
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import zipfile

if __package__:
    from .fetch_tool import fetch
else:
    from fetch_tool import fetch


def validate():
    root=Path(__file__).resolve().parents[1]
    machine=platform.machine().lower()
    arch='arm64' if machine in {'arm64','aarch64'} else 'amd64' if machine in {'amd64','x86_64'} else None
    if arch is None:raise RuntimeError('No qualified actionlint binary for this publisher architecture')
    osname={'win32':'windows','linux':'linux','darwin':'darwin'}.get(sys.platform)
    archive=fetch('actionlint-'+str(osname)+'-'+arch)
    filename='actionlint.exe' if sys.platform=='win32' else 'actionlint'
    binary=archive.parent/(osname+'-'+arch+'-'+filename)
    if archive.suffix=='.zip':
        with zipfile.ZipFile(archive) as package:data=package.read(filename)
    else:
        with tarfile.open(archive) as package:
            member=package.getmember(filename)
            if not member.isfile():raise RuntimeError('Invalid validator archive member')
            data=package.extractfile(member).read()
    # Read one known archive member; never extract arbitrary archive paths.
    binary.write_bytes(data);binary.chmod(binary.stat().st_mode|0o111)
    subprocess.run([str(binary),'-shellcheck=','-pyflakes=',str(root/'.github/workflows/release.yml')],cwd=root,check=True)
    print('GitHub workflow syntax and contexts validated by pinned actionlint 1.7.12',flush=True)


if __name__=='__main__':validate()
