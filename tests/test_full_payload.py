import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import abu_engine_packs as packs
from installer import full_payload


class FullPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.archives=self.root/'archives';self.archives.mkdir()
        self.specs=[];self.evidence={}
        for kind in ('alignment','training'):
            payload=self.archives/(kind+'.zip');payload.write_bytes(kind.encode())
            sha=packs.digest(payload);payload.rename(self.archives/(sha+'.zip'))
            spec=dict(id=kind,version='1',platform=packs.platform_key(),default_full=True,
                      capability=kind,sha256=sha,download_bytes=len(kind),bundle_permitted=True)
            self.specs.append(spec)
            self.evidence[kind]=dict(archive_sha256=sha,gpu_test_passed=True,
                                    projection='equirectangular',cubemaps_used=False,
                                    finite_loss=True,export_valid=True,resume_passed=True)
        source=self.archives/'source-and-licenses.zip';source.write_bytes(b'fixture source and license archive')
        self.specs[1]['compliance_assets']=[dict(filename=source.name,sha256=packs.digest(source),bytes=source.stat().st_size)]
        self.record=self.root/'evidence.json'
        self.mock=patch.object(full_payload,'catalog',return_value=self.specs);self.mock.start();self.addCleanup(self.mock.stop)

    def prepare(self):
        self.record.write_text(json.dumps(self.evidence))
        return full_payload.prepare(self.archives,self.root/'staging',self.record)

    def test_validated_explicit_payload_only(self):
        (self.archives/'private-image.jpg').write_bytes(b'not public')
        self.prepare()
        self.assertEqual(len(list((self.root/'staging').iterdir())),4)
        self.assertFalse((self.root/'staging/private-image.jpg').exists())

    def test_cpu_or_alignment_only_cannot_claim_full(self):
        self.specs.pop()
        with self.assertRaises(ValueError):self.prepare()

    def test_no_offline_bundle_without_redistribution_clearance(self):
        self.specs[0]['bundle_permitted']=False
        with self.assertRaisesRegex(ValueError,'redistribution'):self.prepare()

    def test_windows_builder_does_not_silently_downgrade_an_unqualified_trainer(self):
        from installer.windows import build_setup
        with patch.object(build_setup,'catalog',return_value=self.specs),self.assertRaises(ValueError):build_setup.edition()

    def test_wrong_binary_evidence_rejected(self):
        self.evidence['training']['archive_sha256']='0'*64
        with self.assertRaises(ValueError):self.prepare()

    def test_panorama_rendering_is_not_training_acceptance(self):
        self.evidence['training']['projection']='perspective'
        with self.assertRaises(ValueError):self.prepare()

    def test_missing_training_export_or_resume_rejected(self):
        self.evidence['training']['resume_passed']=False
        with self.assertRaises(ValueError):self.prepare()

    def test_corrupt_compliance_asset_rejected(self):
        (self.archives/'source-and-licenses.zip').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.prepare()

    def test_full_runtime_refuses_incomplete_catalog(self):
        with patch.object(packs,'catalog',return_value=self.specs[:1]),self.assertRaises(RuntimeError):
            packs.install_bundled(self.archives,root=self.root/'installed')
        with patch.object(packs,'catalog',return_value=self.specs[:1]),self.assertRaises(RuntimeError):
            packs.install_default_components(root=self.root/'installed')

    def test_native_panorama_metadata_selects_gut(self):
        import train_engine
        sparse=self.root/'sparse/0';sparse.mkdir(parents=True)
        (sparse/'cameras.txt').write_text('# test calibration\n1 EQUIRECTANGULAR 2048 1024 2048 1024\n')
        self.assertTrue(train_engine.lichtfeld_requires_gut(self.root))
        (sparse/'cameras.txt').write_text('1 PINHOLE 2048 1024 1000 1000 1024 512\n')
        self.assertFalse(train_engine.lichtfeld_requires_gut(self.root))

    def test_frozen_training_does_not_search_personal_installations(self):
        import sys
        import train_engine
        with patch.object(sys,'frozen',True,create=True),patch.object(packs,'optional_executable',return_value=None),patch.object(train_engine.os.path,'isfile',side_effect=AssertionError('personal fallback')):
            self.assertIsNone(train_engine.get_lichtfeld_executable())

    def test_failed_trainer_cannot_report_old_ply_as_success(self):
        import train_engine
        from unittest.mock import MagicMock
        output=self.root/'old-training-output';output.mkdir()
        (output/'old.ply').write_bytes(b'pre-existing output')
        process=MagicMock();process.stdout=[];process.returncode=17
        with patch.object(train_engine,'get_lichtfeld_executable',return_value='fixture-trainer.exe'),patch.object(train_engine.subprocess,'Popen',return_value=process),patch.object(train_engine,'check_save_and_finish_sentinel',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'code 17'):
                train_engine.train_lichtfeld(str(self.root),str(output),headless=True,progress_callback=lambda line:None)

    def test_training_resolution_reaches_the_native_command_exactly(self):
        import train_engine
        from unittest.mock import MagicMock
        for selection,width in [('4k','3840'),('3840','3840'),('4K Downscale (3840px Max)','3840'),('1024','1024'),('1920','1920'),('original','0')]:
            with self.subTest(selection=selection):
                process=MagicMock();process.stdout=[];process.returncode=17
                with patch.object(train_engine,'get_lichtfeld_executable',return_value='fixture-trainer.exe'),patch.object(train_engine.subprocess,'Popen',return_value=process) as launch,patch.object(train_engine,'check_save_and_finish_sentinel',return_value=False):
                    with self.assertRaises(RuntimeError):
                        train_engine.train_lichtfeld(str(self.root),str(self.root/'resolution-output'),headless=True,train_res=selection,progress_callback=lambda line:None)
                    command=launch.call_args.args[0]
                    self.assertEqual(command[command.index('--max-width')+1],width)

    def test_installed_cli_defaults_to_available_alignment_engine(self):
        import sys,abu_runtime
        with patch.object(sys,'frozen',True,create=True),patch.object(abu_runtime.runpy,'run_path') as run:
            saved=sys.argv[:]
            try:
                self.assertEqual(abu_runtime.dispatch(['--align','--input','fixture']),0)
                self.assertEqual(sys.argv[-2:],['--engine','native_colmap'])
                run.assert_called_once()
            finally:sys.argv=saved

    def test_installed_cli_cannot_bypass_engine_distribution_gate(self):
        import sys,abu_runtime
        with patch.object(sys,'frozen',True,create=True),self.assertRaises(RuntimeError):
            abu_runtime.dispatch(['--align','--engine','ace0','--input','fixture'])


if __name__=='__main__':unittest.main()
