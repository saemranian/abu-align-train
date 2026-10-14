import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from training_input import prepare_training_input


class TestTrainingInput(unittest.TestCase):
    def test_actual_resolution_and_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            for i in range(12): Image.new('RGB', (128, 64), 'red').save(source/f'{i:03}.jpg')
            prepared = Path(prepare_training_input(source, root/'output', '2'))
            with Image.open(prepared/'images'/'000.jpg') as image: self.assertEqual(image.size, (64, 32))
            with Image.open(source/'000.jpg') as image: self.assertEqual(image.size, (128, 64))
            split = json.loads((prepared/'metadata.json').read_text())
            self.assertFalse(set(split['train']) & set(split['test']))
            self.assertEqual(len(split['train']) + len(split['test']), 12)

    def test_resized_landmark_pixels_and_immutable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            Image.new('RGB', (128, 64), 'red').save(source/'a.jpg')
            component = {'cameras': {'s': {'projection_type': 'spherical', 'width': 128, 'height': 64}},
                         'shots': {'a.jpg': {'camera': 's'}},
                         'points': {'p': {'observations': [{'shot_id': 'a.jpg', 'pixel': [100, 20]}]}}}
            text = json.dumps([component]); (source/'reconstruction.json').write_text(text)
            prepared = Path(prepare_training_input(source, root/'output', '2'))
            result = json.loads((prepared/'opensfm'/'reconstruction.json').read_text())[0]
            self.assertEqual(result['points']['p']['observations'][0]['pixel'], [50, 10])
            self.assertEqual((source/'reconstruction.json').read_text(), text)

    def test_mixed_native_dimensions_are_rejected_without_resizing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            Image.new('RGB', (128, 64)).save(source/'a.jpg')
            Image.new('RGB', (256, 128)).save(source/'b.jpg')
            with self.assertRaisesRegex(ValueError, 'Mixed panorama'):
                prepare_training_input(source, root/'output', '1')
