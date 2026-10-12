"""Byte-bounded records and tensor-like per-view storage for panoramic training."""
from collections import OrderedDict
import copy
import torch


class BoundedRecords:
    def __init__(self, dataset, budget_bytes=256 * 1024**2):
        self.dataset = dataset
        self.budget = budget_bytes
        self.cache = OrderedDict()
        self.bytes = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not 0 <= index < len(self): raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        item = self.dataset.get_data(index, image_type='uint8')
        size = sum(v.numel() * v.element_size() for v in item.values() if torch.is_tensor(v))
        while self.cache and self.bytes + size > self.budget:
            _, (_, old_size) = self.cache.popitem(last=False)
            self.bytes -= old_size
        if size <= self.budget:
            self.cache[index] = (item, size); self.bytes += size
        return item


class ViewSequence:
    """A dataset stays on disk/CPU; indexing alone transfers the requested view.

    Whole-sequence clone/deepcopy/to are metadata operations. Slices are capped
    because pose matching may otherwise request every visited image at once.
    """
    def __init__(self, count, getter, shape, token=None, device='cpu', dtype=torch.float32, max_slice=32):
        self.count, self.getter, self.shape = count, getter, (count, *shape)
        self.token, self.device, self.dtype, self.max_slice = token, torch.device(device), dtype, max_slice

    def __len__(self): return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = list(range(*index.indices(self.count)))
        elif isinstance(index, (list, tuple)) or (torch.is_tensor(index) and index.ndim):
            indices = [int(i) for i in index]
        else:
            index = int(index)
            if index < 0: index += self.count
            if not 0 <= index < self.count: raise IndexError(index)
            return self.getter(index).to(device=self.device, dtype=self.dtype)
        if len(indices) > self.max_slice:
            raise MemoryError(f'View request exceeds bounded slice {self.max_slice}; set match_views/densify_views to a finite window')
        if not indices:
            return torch.empty((0, *self.shape[1:]), device=self.device, dtype=self.dtype)
        return torch.stack([self[i] for i in indices])

    def __deepcopy__(self, memo): return self.clone()

    def clone(self): return copy.copy(self)

    def to(self, device=None, dtype=None):
        result = self.clone()
        if device is not None: result.device = torch.device(device)
        if dtype is not None: result.dtype = dtype
        return result

    def squeeze(self, dim):
        if self.shape[dim] != 1: return self
        shape = list(self.shape[1:]); del shape[dim if dim < 0 else dim-1]
        return ViewSequence(self.count, lambda i: self.getter(i).squeeze(dim), shape,
                            self.token, self.device, self.dtype, self.max_slice)

    def __gt__(self, threshold):
        return ViewSequence(self.count, lambda i: self.getter(i) > threshold, self.shape[1:],
                            self.token, self.device, torch.bool, self.max_slice)

    @classmethod
    def __torch_function__(cls, function, types, args=(), kwargs=None):
        if function is torch.equal:
            a, b = args
            return isinstance(a, cls) and isinstance(b, cls) and a.token is not None and a.token == b.token
        return NotImplemented


def image_sequence(records):
    first = records[0]['image']
    def get(index):
        raw = records[index]['image']
        image = raw.float() / 255 if raw.dtype == torch.uint8 else raw.float()
        if image.shape[-1] == 4: image = image[..., :3] * image[..., 3:4]
        return image
    paths = tuple(str(p) for p in records.dataset.image_filenames)
    return ViewSequence(len(records), get, (*first.shape[:2], 3), token=paths)
