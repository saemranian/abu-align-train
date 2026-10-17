"""Version-gated in-process PFGS360 memory adapter; upstream files stay untouched."""
import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path
import numpy as np
import torch
RUNTIME_METRICS = {}
from pfgs_memory import BoundedRecords, ViewSequence, image_sequence
from run_manifest import file_hash


def install():
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
    from nerfstudio360.pipelines.colmapfree_pipeline import ColmapFreePipeline
    import nerfstudio360.models.posefree_gaussian_splatting_360_base as base
    from nerfstudio360.models.posefree_gaussian_splatting_360 import PoseFreeGSplat360Model

    budget = int(os.environ.get('ABU_IMAGE_CACHE_MB', '256')) * 1024**2
    def load_images(self, split, cache_images_device):
        dataset = self.train_dataset if split == 'train' else self.eval_dataset
        self.train_cameras = self.train_dataset.cameras
        cameras = dataset.cameras
        from nerfstudio.cameras.cameras import CameraType
        if not (cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all():
            raise ValueError('Bounded PFGS360 adapter requires equirectangular cameras')
        return BoundedRecords(dataset, budget)
    FullImageDatamanager._load_images = load_images
    def eval_loader(self):
        for index in range(len(self.eval_dataset)):
            data = self.cached_eval[index].copy()
            data['image'] = data['image'].to(self.device)
            yield self.eval_dataset.cameras[index:index+1].to(self.device), data
    FullImageDatamanager.fixed_indices_eval_dataloader = property(eval_loader)

    constructor = textwrap.dedent(inspect.getsource(ColmapFreePipeline.__init__))
    replacements = {
        'super().__init__()': 'Pipeline.__init__(self)',
        'train_images = _composite_images_with_background(self.datamanager.cached_train)':
            'train_images = image_sequence(self.datamanager.cached_train)',
        'eval_images = _composite_images_with_background(self.datamanager.cached_eval)':
            'eval_images = image_sequence(self.datamanager.cached_eval)',
        'eval_images = train_images[eval_indices].clone()':
            'eval_images = ViewSequence(len(eval_indices), lambda i: train_images.getter(eval_indices[i]), train_images.shape[1:])'}
    for old, new in replacements.items():
        if constructor.count(old) != 1:
            raise RuntimeError(f'Unsupported PFGS360 pipeline revision: {old}')
        constructor = constructor.replace(old, new)
    namespace = dict(ColmapFreePipeline.__init__.__globals__)
    namespace.update(image_sequence=image_sequence, ViewSequence=ViewSequence)
    exec(compile(constructor, '<Abu bounded PFGS360 pipeline>', 'exec'), namespace)
    ColmapFreePipeline.__init__ = namespace['__init__']

    models = {}
    cache_root = Path(os.environ.get('ABU_DEPTH_CACHE', '.abu_depth_cache'))
    cache_root.mkdir(parents=True, exist_ok=True)
    def depths(images, model='unik3d'):
        if model != 'unik3d':
            raise ValueError('Streaming spherical depth currently supports UniK3D only')
        from nerfstudio360.thirdparty.monocular_depth_unik3d import UniK3D
        keys = {}
        def get(index):
            if index not in keys:
                path = images.token[index] if images.token else None
                identity = file_hash(path) if path else str(index)
                key = hashlib.sha256((identity + str(images.shape[1:]) + model + file_hash(__file__)).encode()).hexdigest()
                keys[index] = cache_root / (key + '.npy')
            target = keys[index]
            if not target.exists():
                if model not in models: models[model] = UniK3D(device='cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad(): depth = models[model].infer_gsplat(images.getter(index))
                array = depth.detach().cpu().numpy().astype(np.float32)
                temporary = target.with_suffix('.tmp.npy')
                np.save(temporary, array); temporary.replace(target)
            return torch.from_numpy(np.load(target, allow_pickle=False))
        return ViewSequence(len(images), get, (*images.shape[1:3], 1), token=images.token)
    base.generate_equir_depth_sequence = depths

    # Posed training explicitly freezes trusted geometry unless refinement is requested.
    mode = os.environ.get('ABU_POSE_MODE', 'fixed')
    if mode not in {'fixed', 'refine', 'posefree'}:
        raise ValueError('Unsupported explicit pose mode')
    initialization = textwrap.dedent(inspect.getsource(base.PoseFreeGSplat360BaseModel.__init__))
    start = initialization.find('    # Detect whether train_cameras')
    end = initialization.find('    self.use_prior_poses =', start)
    if start < 0 or end < 0 or initialization.count('super().__init__(*args, **kwargs)') != 1:
        raise RuntimeError('Unsupported PFGS360 prior-pose initialization revision')
    initialization = initialization[:start] + '''    self.has_prior_poses = abu_mode != 'posefree'
    if self.has_prior_poses:
        if train_cameras is None or not torch.isfinite(train_cameras.camera_to_worlds).all():
            raise ValueError('Explicit posed training requires finite aligned cameras')
''' + initialization[end:]
    initialization = initialization.replace('super().__init__(*args, **kwargs)',
                                           'Model.__init__(self, *args, **kwargs)')
    context = dict(base.PoseFreeGSplat360BaseModel.__init__.__globals__)
    context['abu_mode'] = mode
    exec(compile(initialization, '<Abu explicit PFGS360 poses>', 'exec'), context)
    base.PoseFreeGSplat360BaseModel.__init__ = context['__init__']
    original_init = PoseFreeGSplat360Model.__init__
    original_loss = PoseFreeGSplat360Model.get_loss_dict
    def initialize(self, *args, **kwargs):
        config = args[0]
        if hasattr(config, 'freeze_prior_poses'): config.freeze_prior_poses = mode == 'fixed'
        config.use_prior_poses = mode != 'posefree'
        original_init(self, *args, **kwargs)
        self.register_buffer('_abu_pose_anchor', self.camera_optimizer.pose_adjustment.detach().clone())
    PoseFreeGSplat360Model.__init__ = initialize
    def loss(self, outputs, batch=None, metrics_dict=None):
        result = original_loss(self, outputs, batch, metrics_dict)
        if mode == 'refine' and self.use_prior_poses:
            # Absolute SO3xR3 priors prevent unconstrained photometric pose drift.
            anchor = self._abu_pose_anchor.to(self.camera_optimizer.pose_adjustment.device)
            result['abu_pose_prior'] = 10 * (self.camera_optimizer.pose_adjustment - anchor).square().mean()
        return result
    PoseFreeGSplat360Model.get_loss_dict = loss
    original_parameters = base.PoseFreeGSplat360BaseModel.set_parameter
    def parameters(self, key):
        result = original_parameters(self, key)
        if mode == 'fixed' and self.use_prior_poses:
            for parameter in self.camera_optimizer.parameters(): parameter.requires_grad_(False)
            for parameter in self.eval_camera_optimizer.parameters(): parameter.requires_grad_(False)
        return result
    base.PoseFreeGSplat360BaseModel.set_parameter = parameters

    original_after = PoseFreeGSplat360Model.after_train
    def verify_after(self, attributes, step):
        original_after(self, attributes, step)
        drift = float((self.camera_optimizer.pose_adjustment.detach() - self._abu_pose_anchor).abs().max())
        RUNTIME_METRICS.update(gaussians=self.num_points, pose_max_parameter_drift=drift)
        if mode == 'fixed' and self.use_prior_poses and drift > 1e-6:
            raise RuntimeError('Fixed camera poses changed during training')
    PoseFreeGSplat360Model.after_train = verify_after
    cap = int(os.environ.get('ABU_MAX_GAUSSIANS', '0'))
    if cap:
        original_merge = PoseFreeGSplat360Model.merge_new_points
        def merge(self, optimizers, new_points, new_colors, new_alpha=None):
            remaining = cap - self.num_points
            if remaining <= 0: return None
            if len(new_points) > remaining:
                indices = torch.linspace(0, len(new_points)-1, remaining, device=new_points.device).long()
                new_points, new_colors = new_points[indices], new_colors[indices]
                if isinstance(new_alpha, torch.Tensor) and new_alpha.ndim and len(new_alpha) > remaining:
                    new_alpha = new_alpha[indices]
            return original_merge(self, optimizers, new_points, new_colors, new_alpha)
        PoseFreeGSplat360Model.merge_new_points = merge
        original_cap_after = PoseFreeGSplat360Model.after_train
        def after(self, attributes, step):
            original_cap_after(self, attributes, step)
            if self.num_points > cap:
                keep = torch.topk(self.gauss_params['opacities'].detach().flatten(), cap).indices
                remove = torch.ones(self.num_points, dtype=torch.bool, device=keep.device); remove[keep] = False
                deleted = self.cull_gaussians(extra_cull_mask=remove)
                self.remove_from_all_optim(attributes.optimizers, deleted)
        PoseFreeGSplat360Model.after_train = after
        original_adjust = PoseFreeGSplat360Model.adjust_gaussians
        def adjust(self, attributes, do_densify, do_reset=False):
            return original_adjust(self, attributes, do_densify and self.num_points < cap, do_reset)
        PoseFreeGSplat360Model.adjust_gaussians = adjust
    print(f'[Abu memory] image cache <= {budget/1024**2:.0f} MiB; per-view GPU transfer; depth on disk; pose mode {mode}; Gaussian budget {cap or "unlimited"}', flush=True)
