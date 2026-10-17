"""Release safety tests use synthetic files and never contact GitHub."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from installer.build_core import approved_files
from installer.public_export import export
from installer.verify_assets import verify
import push_to_github as publisher
import abu_runtime


class DistributionSafety(unittest.TestCase):
    def test_core_manifest_is_explicit_and_complete(self):
        files=approved_files()
        self.assertGreater(len(files),40)
        names={str(r).replace('\\','/') for _,r in files}
        self.assertIn('installer/engines.json',names)
        self.assertFalse(any('checkpoints/' in p or 'versions/' in p for p in names))

    def test_payload_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'installer').mkdir()
            (root/'installer/core-files.json').write_text(json.dumps({'files':['../secret.txt']}))
            with self.assertRaises(ValueError):approved_files(root)

    def test_existing_or_unexpected_remote_is_not_rewritten(self):
        publisher.validate_remote(publisher.REPO_URL)
        with self.assertRaises(ValueError):publisher.validate_remote('https://github.com/someone/else.git')

    def test_odometer_version_validation(self):
        self.assertEqual(publisher.validate_version('0.1.5.9'),'0.1.5.9')
        for invalid in ['0.1.5','v0.1.5.9','0.1.5.10','main;anything','../../else']:
            with self.assertRaises(ValueError):publisher.validate_version(invalid)

    def test_public_export_does_not_copy_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';output=Path(temp)/'public';(root/'installer').mkdir(parents=True)
            (root/'installer/public-files.json').write_text(json.dumps({'files':['safe.py']}))
            (root/'safe.py').write_text('print(54)');(root/'private.jpg').write_bytes(b'not-public')
            export(root,output)
            self.assertTrue((output/'safe.py').is_file());self.assertFalse((output/'private.jpg').exists())

    def test_credentials_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';(root/'installer').mkdir(parents=True)
            (root/'installer/public-files.json').write_text(json.dumps({'files':['safe.py']}))
            (root/'safe.py').write_bytes(b'ghp_'+b'x'*40)
            with self.assertRaisesRegex(ValueError,'Credential pattern'):export(root,Path(temp)/'out')

    def test_source_only_installer_cannot_pass_asset_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'installer.exe').write_bytes(b'MZ'+b'x'*100)
            with self.assertRaises(ValueError):verify(temp)

    def test_client_paths_are_rejected_with_both_slash_styles(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';(root/'installer').mkdir(parents=True)
            (root/'installer/public-files.json').write_text(json.dumps({'files':['safe.py']}))
            for separator in [b'/',b'\\']:
                (root/'safe.py').write_bytes(separator.join([b'documents',b'WORKS',b'Clients',b'private.jpg']))
                with self.assertRaisesRegex(ValueError,'Private client path'):export(root,Path(temp)/'out')

    def test_frozen_worker_cannot_execute_arbitrary_python(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):abu_runtime.dispatch([str(Path(temp)/'align_cli.py')])
        with self.assertRaises(ValueError):abu_runtime.dispatch(['-c','print(1)'])

    def test_bundled_policy_does_not_affect_developer_engines(self):
        from abu_components import check_bundled_engine
        with patch.object(sys,'frozen',False,create=True):check_bundled_engine('opensfm')
        with patch.object(sys,'frozen',True,create=True):
            check_bundled_engine('native_colmap')
            with self.assertRaises(RuntimeError):check_bundled_engine('ace0')

    def test_no_torch_cpu_cli_import(self):
        # Import a second CLI module in an explicitly Torch-free interpreter state.
        with patch.dict(sys.modules,{'torch':None,'torch.nn':None,'torch.nn.functional':None}):
            spec=importlib.util.spec_from_file_location('abu_cpu_cli_test',ROOT/'align_cli.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertIsNone(module.torch);self.assertEqual(module.get_vram_usage(),(0.0,0.0))


if __name__=='__main__':unittest.main()
