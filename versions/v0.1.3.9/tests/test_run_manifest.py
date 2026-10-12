import tempfile
import unittest
from pathlib import Path
from PIL import Image
from run_manifest import write_manifest, load_manifest


class TestRunManifest(unittest.TestCase):
    def test_selection_and_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in ['10.JPG', '2.png', '1.jpg']:
                path = root / name
                Image.new('RGB', (64, 32), 'red').save(path)
                paths.append(path)
            a = write_manifest(paths, root / 'manifest.json', '360dvo', {'threshold': 12})
            self.assertEqual([x['name'] for x in load_manifest(root / 'manifest.json')['images']], ['10.JPG', '2.png', '1.jpg'])
            b = write_manifest(paths, root / 'other.json', '360dvo', {'threshold': 15})
            self.assertNotEqual(a['fingerprint'], b['fingerprint'])
            Image.new('RGB', (64, 32), 'blue').save(paths[1])
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_manifest(root / 'manifest.json')
