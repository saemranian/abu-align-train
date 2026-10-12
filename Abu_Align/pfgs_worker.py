"""Launch installed Nerfstudio through Abu's versioned streaming adapter."""
import os
import sys
import json
import time
import atexit
from pathlib import Path

if __name__ == '__main__':
    operation = sys.argv.pop(1)
    started = time.perf_counter()
    success = [False]
    def metrics():
        import torch
        output = os.environ.get('ABU_RUN_REPORT')
        if not output and '--output-dir' in sys.argv:
            output = str(Path(sys.argv[sys.argv.index('--output-dir')+1])/'runtime_metrics.json')
        if output:
            try:
                import resource
                peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform != 'darwin' else 1)
            except ImportError:
                import psutil
                memory = psutil.Process().memory_info()
                peak_ram = getattr(memory, 'peak_wset', memory.rss)
            path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
            from pfgs_streaming import RUNTIME_METRICS
            path.write_text(json.dumps({'operation': operation, 'success': success[0], **RUNTIME_METRICS,
                'seconds': time.perf_counter()-started, 'peak_ram_bytes': peak_ram,
                'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
                'pose_mode': os.environ.get('ABU_POSE_MODE', 'fixed'), 'torch': torch.__version__}, indent=2), encoding='utf-8')
    atexit.register(metrics)
    if operation == 'export-checkpoint':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--max-gaussians', type=int, default=0)
        args = parser.parse_args()
        from train_engine import export_pfgs360_checkpoint_to_ply
        from spherical_geometry import validate_gaussian_ply
        export_pfgs360_checkpoint_to_ply(args.checkpoint, args.output, args.max_gaussians)
        validate_gaussian_ply(args.output)
        success[0] = True
        print(f'Verified Gaussian export: {args.output}', flush=True)
        sys.exit(0)
    from pfgs_streaming import install
    install()
    if operation == 'train':
        from nerfstudio.scripts.train import entrypoint
    elif operation == 'export':
        from nerfstudio.scripts.exporter import entrypoint
    else:
        raise ValueError('Worker operation must be train or export')
    try:
        entrypoint()
        success[0] = True
    except SystemExit as error:
        success[0] = error.code in [0, None]
        raise
