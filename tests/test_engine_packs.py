import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import abu_engine_packs as packs


class PackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.archive=self.base/'pack.zip'
        with zipfile.ZipFile(self.archive,'w') as z:z.writestr('bin/engine.exe',b'fixture executable, never run')
        self.spec=dict(id='fixture',version='1.0',platform=packs.platform_key(),distribution='upstream-download',
                       url='https://github.com/example/project/releases/download/v1/fixture.zip',
                       sha256=packs.digest(self.archive),download_bytes=self.archive.stat().st_size,
                       expanded_bytes_limit=1024,executables={'worker':'bin/engine.exe'})
        self.root=self.base/'components'
        self.catalog=patch.object(packs,'catalog',return_value=[self.spec]);self.catalog.start();self.addCleanup(self.catalog.stop)

    def install(self,**kwargs):
        return packs.install('fixture',root=self.root,archive=self.archive,**kwargs)

    def test_verified_install_and_resolve(self):
        directory=self.install()
        self.assertEqual(Path(packs.executable('fixture','worker',self.root)),directory/'bin/engine.exe')
        self.assertFalse(json.loads((directory/'abu-pack-receipt.json').read_text())['gpu_qualified'])

    def test_corrupted_archive_never_activates(self):
        self.archive.write_bytes(b'corrupted')
        with self.assertRaises(ValueError):self.install()
        self.assertFalse((self.root/'fixture.active.json').exists())

    def test_failed_upgrade_preserves_active(self):
        self.install();old=(self.root/'fixture.active.json').read_bytes()
        self.spec['version']='2.0'
        with self.assertRaises(packs.Cancelled):self.install(cancel=lambda:True)
        self.assertEqual(old,(self.root/'fixture.active.json').read_bytes())
        self.assertFalse((self.root/'fixture.lock').exists())

    def test_tampered_executable_rejected(self):
        target=self.install();(target/'bin/engine.exe').write_bytes(b'changed')
        with self.assertRaises(ValueError):packs.executable('fixture','worker',self.root)

    def test_archive_path_aliases_rejected(self):
        for name in ['../escape','/absolute','C:/outside','a\\b','NUL.txt','a/COM1.exe','a./b']:
            with self.subTest(name=name),self.assertRaises(ValueError):packs._relative(name)

    def test_zip_symlink_rejected(self):
        with zipfile.ZipFile(self.archive,'w') as z:
            info=zipfile.ZipInfo('link');info.external_attr=0o120777<<16;z.writestr(info,'target')
        with self.assertRaises(ValueError):packs._extract(self.archive,self.base/'extract',1024,lambda *a:None,lambda:False)

    def test_expansion_limit_rejected(self):
        with self.assertRaises(ValueError):packs._extract(self.archive,self.base/'extract',1,lambda *a:None,lambda:False)

    def test_wrong_platform_rejected(self):
        self.spec['platform']='unsupported'
        with self.assertRaises(ValueError):packs._check_spec(self.spec)

    def test_low_disk_rejected(self):
        from collections import namedtuple
        usage=namedtuple('Usage','total used free')(100,99,1)
        with patch.object(packs.shutil,'disk_usage',return_value=usage),self.assertRaises(RuntimeError):self.install()

    def test_reinstall_is_idempotent(self):
        self.assertEqual(self.install(),self.install())

    def test_concurrent_install_rejected(self):
        self.root.mkdir();(self.root/'fixture.lock').touch()
        with self.assertRaises(RuntimeError):self.install()


if __name__=='__main__':unittest.main()
