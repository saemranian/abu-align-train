import copy
import unittest
import torch
from pfgs_memory import BoundedRecords, ViewSequence


class Dataset:
    def __len__(self): return 10000
    def get_data(self, index, image_type): return {'image': torch.zeros((8, 16, 3), dtype=torch.uint8)}


class TestPFGSMemory(unittest.TestCase):
    def test_cache_bytes_are_bounded(self):
        records = BoundedRecords(Dataset(), budget_bytes=1000)
        for index in range(100): records[index]
        self.assertLessEqual(records.bytes, 1000)
        self.assertEqual(len(records.cache), 2)

    def test_whole_dataset_copy_and_transfer_do_not_load_views(self):
        calls = []
        sequence = ViewSequence(10000, lambda i: calls.append(i) or torch.full((2, 4, 1), float(i)), (2, 4, 1), token='test')
        moved = copy.deepcopy(sequence).to('cpu')
        self.assertEqual(calls, [])
        self.assertEqual(moved[3][0, 0, 0], 3)
        self.assertTrue(torch.equal(sequence, moved))
        self.assertEqual((sequence.squeeze(-1) > 2)[3].shape, (2, 4))
        self.assertTrue((sequence.squeeze(-1) > 2)[3].all())
        with self.assertRaises(MemoryError): sequence[:100]
