"""Verified, per-user binary components. No pip, compiler or dataset uploads.

Only the catalog shipped with the application supplies download URLs and hashes.
Installation publishes an immutable version directory, then atomically switches
the active receipt. A failed/cancelled upgrade leaves the prior receipt intact.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile

from abu_paths import user_data_dir


class Cancelled(RuntimeError):
    pass


def platform_key():
    machine = platform.machine().lower()
    arch = 'arm64' if machine in {'aarch64', 'arm64'} else 'x64' if machine in {'amd64', 'x86_64'} else machine
    return {'win32': 'windows', 'linux': 'linux', 'darwin': 'macos'}.get(sys.platform, sys.platform) + '-' + arch


def catalog():
    return json.loads((Path(__file__).parent / 'installer/engine-packs.json').read_text(encoding='utf-8'))['packs']


def component_root():
    if os.environ.get('ABU_COMPONENT_HOME'):
        return Path(os.environ['ABU_COMPONENT_HOME']).expanduser().absolute()
    config = user_data_dir() / 'components.json'
    if config.is_file():
        return Path(json.loads(config.read_text(encoding='utf-8'))['root']).expanduser().absolute()
    return user_data_dir() / 'components'


def set_component_root(path):
    root = Path(path).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    config = user_data_dir() / 'components.json'
    config.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(config, {'root': str(root)})


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _relative(name):
    # Reject Windows path aliases even when qualifying a Windows pack on Linux.
    if not isinstance(name, str) or '\\' in name or ':' in name or '\x00' in name:
        raise ValueError('Invalid component path')
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(v in {'.', '..'} or v.endswith(('.', ' ')) for v in p.parts):
        raise ValueError('Unsafe component path: ' + name)
    for part in p.parts:
        if part.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
            raise ValueError('Reserved component path')
    return p


def _check_spec(spec):
    for field in ('id', 'version'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', spec[field]):
            raise ValueError('Invalid component identity')
    if not re.fullmatch('[0-9a-f]{64}', spec['sha256']):
        raise ValueError('A pinned SHA256 is required')
    if spec['platform'] != platform_key():
        raise ValueError('Component is incompatible with this platform')
    if spec.get('distribution') not in {'upstream-download','abu-qualified'}:
        raise ValueError('Component is not cleared for this installation method')
    if spec.get('distribution')=='abu-qualified':
        qualification=spec.get('qualification',{})
        if qualification.get('archive_sha256')!=spec['sha256'] or not spec.get('compliance_assets'):
            raise ValueError('Qualified component requires matching acceptance and source/license assets')
    if not spec['url'].startswith('https://github.com/'):
        raise ValueError('Component download must use the pinned HTTPS origin')
    for value in spec['executables'].values():
        _relative(value)


def _download(spec, target, progress, cancel):
    # A partial download is retained for retry. A server ignoring Range causes a
    # clean restart. Size and full digest are verified before any extraction.
    offset = target.stat().st_size if target.exists() else 0
    if offset > spec['download_bytes']:
        target.unlink(); offset = 0
    headers = {'User-Agent': 'AbuAlign-ComponentManager/1'}
    if offset:
        headers['Range'] = f'bytes={offset}-'
    if offset != spec['download_bytes']:
        with urllib.request.urlopen(urllib.request.Request(spec['url'], headers=headers), timeout=30) as response:
            if not response.geturl().startswith('https://'):
                raise ValueError('Insecure component redirect')
            if response.status != 206:
                offset = 0
            elif not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                raise ValueError('Invalid resume response')
            with open(target, 'ab' if offset else 'wb') as out:
                while True:
                    if cancel():
                        raise Cancelled('Download paused; retry resumes the verified archive.')
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    offset += len(block)
                    if offset > spec['download_bytes']:
                        raise ValueError('Component exceeds its pinned size')
                    out.write(block)
                    progress('Downloading', offset, spec['download_bytes'])
    if target.stat().st_size != spec['download_bytes'] or digest(target) != spec['sha256']:
        target.unlink(missing_ok=True)
        raise ValueError('Component archive verification failed; retry the download')


def _extract(archive, destination, limit, progress, cancel):
    with zipfile.ZipFile(archive) as z:
        items = z.infolist()
        if len(items) > 100000 or sum(i.file_size for i in items) > limit:
            raise ValueError('Component exceeds its expanded-size limit')
        names = set()
        for item in items:
            rel = _relative(item.filename.rstrip('/'))
            key = str(rel).casefold()
            if key in names or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Duplicate path or symlink in component')
            names.add(key)
        for index, item in enumerate(items):
            if cancel():
                raise Cancelled('Component installation cancelled')
            target = destination / str(_relative(item.filename.rstrip('/')))
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, open(target, 'xb') as out:
                    shutil.copyfileobj(source, out, length=1024 * 1024)
                if os.name != 'nt' and item.external_attr >> 16 & 0o111:
                    target.chmod(0o755)
            progress('Extracting', index + 1, len(items))


def install(pack_id, root=None, progress=lambda *args: None, cancel=lambda: False, archive=None):
    spec = next(p for p in catalog() if p['id'] == pack_id and p['platform'] == platform_key())
    _check_spec(spec)
    root = Path(root or component_root()); root.mkdir(parents=True, exist_ok=True)
    lock = root / (pack_id + '.lock')
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('This component is already being installed. If its process exited, remove ' + str(lock)) from None
    os.close(lock_fd)
    staging = None
    try:
        # Space for archive + expanded tree + reserve. Existing versions remain
        # available; deleting user-selected component roots is never automatic.
        cached = root / (spec['sha256'] + '.zip.part')
        existing = cached.stat().st_size if cached.exists() else 0
        required = spec['download_bytes'] + spec['expanded_bytes_limit'] + 256 * 1024**2 - existing
        if shutil.disk_usage(root).free < required:
            raise RuntimeError(f'Insufficient space: choose a component folder with {required / 1024**3:.1f} GB free')
        progress('Verifying archive', 0, 1)
        if archive is None:
            _download(spec, cached, progress, cancel)
            archive = cached
        elif Path(archive).stat().st_size != spec['download_bytes'] or digest(archive) != spec['sha256']:
            raise ValueError('Local archive does not match the pinned component')
        staging = Path(tempfile.mkdtemp(prefix='.install-', dir=root))
        _extract(archive, staging, spec['expanded_bytes_limit'], progress, cancel)
        hashes = {}
        for name in spec['executables'].values():
            target = staging / name
            if not target.is_file():
                raise ValueError('Component is missing its required executable: ' + name)
            hashes[name] = digest(target)
        # Extraction and integrity alone are not a GPU benchmark. Runtime GPU
        # qualification is separately recorded by the acceptance command.
        receipt = {'schema_version': 1, 'id': spec['id'], 'version': spec['version'],
                   'archive_sha256': spec['sha256'], 'executables': hashes,
                   'platform': spec['platform'], 'gpu_qualified': False}
        _atomic_json(staging / 'abu-pack-receipt.json', receipt)
        destination = root / (spec['id'] + '-' + spec['version'] + '-' + spec['sha256'][:12])
        if destination.exists():
            # Do not silently overwrite a possibly running binary or trust a
            # stale tree. Receipt and executable integrity must still match.
            if json.loads((destination / 'abu-pack-receipt.json').read_text()) != receipt or any(digest(destination / n) != h for n,h in hashes.items()):
                raise ValueError('Existing component differs; select a clean component folder')
        else:
            os.replace(staging, destination); staging = None
        _atomic_json(root / (pack_id + '.active.json'), {'directory': destination.name})
        if cached.exists():
            cached.unlink()
        progress('Installed; GPU benchmark pending', 1, 1)
        return destination
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def executable(pack_id, role, root=None):
    root = Path(root or component_root())
    active = root / (pack_id + '.active.json')
    if not active.is_file():
        return None
    spec = next(p for p in catalog() if p['id'] == pack_id and p['platform'] == platform_key())
    expected = spec['id'] + '-' + spec['version'] + '-' + spec['sha256'][:12]
    if json.loads(active.read_text())['directory'] != expected:
        raise ValueError('Installed component requires an update')
    folder = root / expected
    receipt = json.loads((folder / 'abu-pack-receipt.json').read_text())
    relative = spec['executables'][role]
    result = folder / relative
    if receipt['archive_sha256'] != spec['sha256'] or digest(result) != receipt['executables'][relative]:
        raise ValueError('Installed executable integrity check failed; reinstall component')
    return str(result)


def optional_executable(pack_id, role, root=None):
    """Unsupported platforms/absent packs return None; corrupt installs fail."""
    if not any(p['id']==pack_id and p['platform']==platform_key() for p in catalog()):
        return None
    return executable(pack_id,role,root)


def install_bundled(directory, root=None, progress=lambda *args:None, cancel=lambda:False):
    """Offline setup uses the app's pinned catalog, never an archive-supplied URL.

    A complete edition requires both alignment and training payloads. Validate
    every archive before activating any pack, so missing/corrupt media fails
    before a partially successful first installation is presented as complete.
    """
    directory=Path(directory)
    specs=[p for p in catalog() if p['platform']==platform_key() and p.get('default_full')]
    if {p.get('capability') for p in specs} != {'alignment','training'}:
        raise RuntimeError('No qualified full GPU edition exists for this platform yet')
    for spec in specs:
        archive=directory/(spec['sha256']+'.zip')
        if not archive.is_file() or archive.stat().st_size!=spec['download_bytes'] or digest(archive)!=spec['sha256']:
            raise ValueError('Missing or corrupt offline component: '+spec['id'])
    return [install(p['id'],root,progress,cancel,directory/(p['sha256']+'.zip')) for p in specs]


def install_default_components(root=None, progress=lambda *args:None, cancel=lambda:False):
    """One online setup downloads all qualified defaults directly to this PC."""
    specs=[p for p in catalog() if p['platform']==platform_key() and p.get('default_full')]
    if {p.get('capability') for p in specs}!={'alignment','training'}:
        raise RuntimeError('Both GPU alignment and training must be qualified before full setup')
    return [install(p['id'],root,progress,cancel) for p in specs]
