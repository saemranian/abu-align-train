"""Create a new DRAFT candidate with checked assets; never overwrite a release."""
import json
import os
from pathlib import Path
import re
import urllib.request
from verify_assets import verify


def main():
    repo=os.environ['GITHUB_REPOSITORY']
    if repo!='saemranian/abu-align-train':raise ValueError('Unexpected publication destination')
    tag=os.environ['GITHUB_REF_NAME']
    version=Path('VERSION.txt').read_text().strip().lstrip('vV')
    if not re.fullmatch(r'v[0-9]\.[0-9]\.[0-9]\.[0-9]',tag) or tag!='v'+version:raise ValueError('Tag and source version differ')
    token=os.environ['GITHUB_TOKEN']
    def request(url,data=None,content_type='application/json'):
        body=json.dumps(data).encode() if isinstance(data,dict) else data
        req=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':content_type,'User-Agent':'Abu-candidate-release'})
        with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
    api='https://api.github.com/repos/'+repo+'/releases'
    if any(r['tag_name']==tag for r in request(api+'?per_page=100')):raise ValueError('Release already exists; refuse replacement')
    records=verify('release-assets')
    expected={'Abu_Align_and_Train_Setup_Windows_x64.exe',
              'Abu_Align_and_Train_Linux_x86_64.AppImage',
              'Abu_Align_and_Train_macOS_arm64.dmg'}
    if {record['name'] for record in records} != expected:
        raise ValueError('The complete supported platform set is required before creating a draft')
    release=request(api,{'tag_name':tag,'target_commitish':os.environ['GITHUB_SHA'],
        'name':'Abu CPU core candidate '+tag,'draft':True,'prerelease':True,
        'body':'Bundled CPU core candidate. Includes interpreter, native dependencies, Manual Control Points and CPU refinement, M54 and spherical COLMAP. No Gaussian training runtimes or neural weights are bundled. Signing, notarization and independent clean-machine qualification are still required before a stable release. See docs/DISTRIBUTION_AUDIT.md and THIRD_PARTY_NOTICES.md. Do not describe this suite as uniformly Apache licensed.'})
    for path in sorted(Path('release-assets').iterdir()):
        if path.is_file():request(release['upload_url'].split('{')[0]+'?name='+path.name,path.read_bytes(),'application/octet-stream')
    print(json.dumps({'draft_url':release['html_url'],'assets':len(records),'stable':False}))


if __name__=='__main__':main()
