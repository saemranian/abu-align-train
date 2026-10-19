"""Flat, architecture-specific evidence and artifacts for upload without private build state."""
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import zipfile
from verify_assets import verify
from public_export import export


def main():
    root=Path(__file__).resolve().parents[1];dist=root/'installer/dist';records=verify(dist)
    output=dist/'release-assets';output.mkdir(exist_ok=True)
    label=sys.platform+'-'+platform.machine()
    for record in records:shutil.copy2(dist/record['name'],output/record['name'])
    shutil.copy2(dist/'release-manifest.json',output/('manifest-'+label+'.json'))
    shutil.copy2(root/'installer/build/compliance/sbom.cdx.json',output/('sbom-'+label+'.cdx.json'))
    edition=root/'installer/build/windows-edition.json'
    if sys.platform=='win32' and edition.is_file():shutil.copy2(edition,output/'capabilities-windows-x64.json')
    with tempfile.TemporaryDirectory(prefix='candidate-source-',dir=root/'installer/build') as temporary:
        source=Path(temporary);export(root,source)
        with zipfile.ZipFile(output/('licenses-and-core-source-'+label+'.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
            for path in source.rglob('*'):
                if path.is_file():archive.write(path,'core-source/'+str(path.relative_to(source)))
            for path in (root/'installer/build/compliance').rglob('*'):
                if path.is_file():archive.write(path,'licenses/'+str(path.relative_to(root/'installer/build/compliance')))


if __name__=='__main__':main()
