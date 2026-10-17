"""Publish an allowlisted source commit and a new candidate tag without rewriting history."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parent
REPO_URL="https://github.com/saemranian/abu-align-train.git"


def git(*args,cwd=ROOT):
    return subprocess.run(['git',*args],cwd=cwd,check=True,capture_output=True,text=True).stdout.strip()


def validate_remote(url):
    if url not in {REPO_URL,REPO_URL[:-4],'git@github.com:saemranian/abu-align-train.git'}:
        raise ValueError('origin does not match the configured Abu repository. It will not be rewritten.')


def validate_version(version):
    if not re.fullmatch(r'[0-9]\.[0-9]\.[0-9]\.[0-9]',version):raise ValueError('Expected the four-field base-10 release version')
    return version


def prepare_export(destination):
    from installer.public_export import export
    return export(ROOT,destination)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',nargs='?',default='Update qualified Abu CPU core candidate')
    parser.add_argument('--details',default='Candidate source release; see docs/DISTRIBUTION_AUDIT.md')
    parser.add_argument('--no-bump',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Validate and export locally without Git or network mutations')
    args=parser.parse_args(argv)
    validate_remote(git('remote','get-url','origin'))
    version=validate_version((ROOT/'VERSION.txt').read_text().strip().lstrip('vV'))
    if not args.dry_run and git('status','--porcelain'):
        raise RuntimeError('Commit the reviewed local release first. This publisher never stages unrelated work.')
    if not args.dry_run and not args.no_bump:
        raise RuntimeError('Prepare the mandatory local version/snapshot/commit first, then use --no-bump. No implicit workspace commit is performed.')
    if not args.dry_run:
        from installer.workflow_lint import validate
        validate()
    staging=ROOT/'installer/build'
    staging.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='public-export-',dir=staging) as temporary:
        target=Path(temporary)
        if args.dry_run:
            records=prepare_export(target)
            print(json.dumps({'dry_run':True,'version':version,'files':len(records),'destination':REPO_URL}));return 0
        # Clone the PUBLIC ancestry rather than pushing private local commits or snapshot history.
        subprocess.run(['git','clone','--depth=1','--branch','main',REPO_URL,str(target)],check=True)
        if git('ls-remote','--tags','origin',f'refs/tags/v{version}',cwd=target):
            raise RuntimeError('That version already exists. Published versions and tags are immutable.')
        git('rm','-r','--ignore-unmatch','.',cwd=target)
        records=prepare_export(target)
        git('add','--all',cwd=target)
        author=git('config','user.name');email=git('config','user.email')
        git('config','user.name',author,cwd=target);git('config','user.email',email,cwd=target)
        git('commit','-m',f'v{version}: {args.summary}\n\n{args.details}',cwd=target)
        git('tag','-a',f'v{version}','-m',f'Abu CPU core candidate v{version}',cwd=target)
        # Atomic, ordinary fast-forward push; a concurrent remote update fails safely.
        git('push','--atomic','origin','HEAD:main',f'refs/tags/v{version}',cwd=target)
        print(json.dumps({'version':version,'public_commit':git('rev-parse','HEAD',cwd=target),
                          'files':len(records),'channel':'draft candidate',
                          'actions':REPO_URL[:-4]+'/actions'}))
    return 0


if __name__=='__main__':raise SystemExit(main())
