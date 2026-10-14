import os
import sys
import argparse
import glob
import math
import json
import subprocess
import time
import numpy as np
import cv2
import torch
from multiprocessing import Process, Queue
from pathlib import Path
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo import projective_ops as pops
from dpvo.lietorch import SE3


def compute_intrinsics(H, W):
    fx = W / (2 * math.pi)
    fy = - H / math.pi
    cx = W / 2
    cy = H / 2
    intrinsics = np.array([fx, fy, cx, cy])
    intrinsics = torch.from_numpy(intrinsics).cuda()
    return intrinsics


from concurrent.futures import ThreadPoolExecutor


def _read_and_process_frame(imfile, max_limit):
    try:
        image = cv2.imread(imfile)
        if image is None:
            return None
        h, w, _ = image.shape
        if w > max_limit or h > (max_limit // 2):
            scale = min(max_limit / float(w), (max_limit / 2.0) / float(h) if w == 2*h else max_limit / float(h))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            new_w = max(16, new_w - new_w % 16)
            new_h = max(16, new_h - new_h % 16)
            # Fast bilinear interpolation is 4x faster on CPU while maintaining sub-pixel tracking accuracy
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            image = image[:h - h % 16, :w - w % 16]
        return image
    except Exception:
        return None


def dvo_image_stream(queue, image_files, stride=1, skip=0, load_res=1024):
    """
    High-throughput equirectangular stream generator with multi-threaded parallel prefetching
    and automatic 8K/4K downsampling to prevent DPVO CUDA correlation integer overflow.
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    if isinstance(image_files, (list, tuple)):
        file_list = image_files[skip::stride]
    else:
        file_list = [
            f for f in sorted(os.listdir(image_files))
            if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(image_files, f))
        ][skip::stride]

    last_image = None
    max_limit = int(load_res) if (load_res and load_res > 0) else 2048
    full_paths = [imfile if os.path.isabs(imfile) else os.path.join(image_files, imfile) for imfile in file_list]
    num_threads = min(8, max(4, os.cpu_count() or 4))

    try:
        # Multi-threaded parallel prefetch over the WSL2 9P filesystem bridge
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for t, image in enumerate(pool.map(lambda p: _read_and_process_frame(p, max_limit), full_paths)):
                if image is None:
                    raise RuntimeError(f"Cannot decode selected image: {full_paths[t]}")
                last_image = image
                queue.put((t, image))
    finally:
        queue.put((-1, last_image))


@torch.no_grad()
def run_dvo(
    imagedir,
    output_dir,
    network='360dvo.pth',
    config_path='config/360.yaml',
    stride=1,
    skip=0,
    patches=96,
    ba_window=8,
    scale=1.0,
    viz=False,
    port=8080,
    dense_cloud=False,
    patch_lifetime=13,
    final_ba_iters=12,
    gradient_bias=True,
    keyframe_thresh=15.0,
    load_res=1024,
    manifest_path=None,
    polar_mask=0.08,
    mask_policy_path=None
):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(config_path):
        config_path = os.path.join(repo_dir, config_path)
    if not os.path.isabs(network):
        network = os.path.join(repo_dir, network)

    os.makedirs(output_dir, exist_ok=True)
    cfg.merge_from_file(config_path)

    if patches and patches > 0:
        cfg.PATCHES_PER_FRAME = int(patches)
        if patches > 128:
            cfg.BUFFER_SIZE = max(cfg.BUFFER_SIZE, 8192)
    if ba_window and ba_window > 0:
        cfg.OPTIMIZATION_WINDOW = int(ba_window)
    if patch_lifetime and patch_lifetime > 0:
        cfg.PATCH_LIFETIME = int(patch_lifetime)
    if keyframe_thresh and keyframe_thresh > 0:
        cfg.KEYFRAME_THRESH = float(keyframe_thresh)
    cfg.GRADIENT_BIAS = bool(gradient_bias)

    # Count total images
    image_files = sorted(glob.glob(os.path.join(imagedir, '*.jpg')) +
                        glob.glob(os.path.join(imagedir, '*.png')) +
                        glob.glob(os.path.join(imagedir, '*.jpeg')))[skip::stride]
    if manifest_path:
        from run_manifest import load_manifest
        manifest = load_manifest(manifest_path)
        image_files = [item['runtime_path'] for item in manifest['images']]
    if not 0 <= polar_mask < 0.5:
        raise ValueError('Polar mask fraction must lie in [0, 0.5)')
    if not 0 < load_res <= 1024:
        raise ValueError('This CUDA build is validated only through width 1024; larger presets require indexing benchmarks')
    if not image_files:
        raise RuntimeError(f'No images found in {imagedir}')

    total_imgs = len(image_files)
    from manual_image_policy import load_policy_file
    mask_policy=load_policy_file(mask_policy_path);mask_atlas=None;mask_rejected=None
    print(f'[*] Initializing 360DVO on GPU: {torch.cuda.get_device_name(0)}')
    print(f'[*] Active Patches: {cfg.PATCHES_PER_FRAME} per frame | BA Window: {cfg.OPTIMIZATION_WINDOW}')
    print(f'[*] Gradient Bias: {cfg.GRADIENT_BIAS} | Patch Lifetime: {cfg.PATCH_LIFETIME} | Final Refine: {final_ba_iters} iters')
    print(f'[*] Dense Sub-Patch Cloud (9x): {dense_cloud}')
    if load_res and load_res > 0:
        print(f'[*] Resolution Limiter: {load_res}px max (safe int32 DPVO correlation)')
    if viz:
        print(f'[VISER:360DVO] Real-time visualizer online at: http://localhost:{port}', flush=True)
    print(f'[*] Processing {total_imgs} equirectangular frames (stride={stride})...', flush=True)

    slam = None
    from dvo_tracks import PersistentTracks
    persistent_tracks = PersistentTracks()
    queue = Queue(maxsize=16)
    reader = Process(target=dvo_image_stream, args=(queue, image_files, 1, 0, load_res), daemon=True)
    reader.start()

    processed = 0
    try:
        while True:
            try:
                (t, image) = queue.get(timeout=5.0)
                if t < 0 or image is None:
                    break
            except Exception:
                if not reader.is_alive() and queue.empty():
                    print('[!] Reader worker exited. Finalizing image stream.', flush=True)
                    break
                continue

            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            C, H, W = image.shape
            intrinsics = compute_intrinsics(H, W)

            if slam is None:
                if viz:
                    import viser
                    _orig_server_cls = viser.ViserServer
                    viser.ViserServer = lambda *a, **kw: _orig_server_cls(*a, host="0.0.0.0", port=port, **{k: v for k, v in kw.items() if k not in ("host", "port")})
                    slam = DPVO(cfg, network, ht=H, wd=W, viz=True)
                    viser.ViserServer = _orig_server_cls
                else:
                    slam = DPVO(cfg, network, ht=H, wd=W, viz=False)
                original_update = slam.network.update.forward
                if mask_policy['mask_dir']:
                    from manual_dvo_masks import MaskAtlas
                    mask_atlas=MaskAtlas(image_files,mask_policy,H,W,slam.RES,image.device)
                    mask_rejected=torch.zeros((),dtype=torch.int64,device=image.device)
                capture_state = {'active': False}
                original_window_update = slam.update
                def window_update():
                    capture_state['active'] = True
                    try: return original_window_update()
                    finally: capture_state['active'] = False
                slam.update = window_update
                def masked_update(net, ctx, corr, flow, ii, jj, kk):
                    updated_net, (delta, weight, extra) = original_update(net, ctx, corr, flow, ii, jj, kk)
                    # Patch coordinates are on the feature grid; exclude masked
                    # source bearings from BA even when gradient sampling finds a boundary.
                    y = slam.patches[0, kk, 1, slam.P // 2, slam.P // 2]
                    valid = (y >= polar_mask * H / slam.RES) & (y < (1 - polar_mask) * H / slam.RES)
                    weight = weight * valid.reshape(1, -1, 1).to(weight.dtype)
                    if mask_atlas is not None and len(kk):
                        center=slam.P//2;source_xy=slam.patches[0,kk,:2,center,center];target_xy=(slam.reproject((ii,jj,kk))[...,center,center]+delta.float())[0]
                        allowed=mask_atlas.allowed(slam.tstamps_[ii],source_xy)&mask_atlas.allowed(slam.tstamps_[jj],target_xy)
                        mask_rejected.add_((~allowed).sum())
                        weight=weight*allowed.reshape(1,-1,1).to(weight.dtype)
                    if capture_state['active'] and len(kk):
                        with torch.no_grad():
                            center = slam.P // 2
                            projected = slam.reproject((ii, jj, kk))[..., center, center] + delta.float()
                            source = slam.patches[0, kk, :2, center, center]
                            normalization = torch.tensor([W, H], device=source.device, dtype=source.dtype) / slam.RES
                            # Preserve measured targets before BA changes the current projection.
                            persistent_tracks.record(
                                slam.tstamps_[ii].cpu().numpy(), slam.tstamps_[jj].cpu().numpy(),
                                (kk % slam.M).cpu().numpy(), (source/normalization).cpu().numpy(),
                                (projected[0]/normalization).cpu().numpy(), weight[0].amin(-1).float().cpu().numpy(),
                                slam.colors_.view(-1, 3)[kk].cpu().numpy())
                    return updated_net, (delta, weight, extra)
                slam.network.update.forward = masked_update
                # Decouple Viser visualizer to asynchronous background thread to prevent GPU stalling
                if viz and hasattr(slam, "viser") and slam.viser is not None:
                    orig_viser_update = slam.viser.update
                    import threading
                    import queue as pyqueue
                    viser_q = pyqueue.Queue(maxsize=2)

                    def _viser_worker():
                        while True:
                            item = viser_q.get()
                            if item is None:
                                break
                            im_c, pos_c, pts_c, clr_c = item
                            try:
                                orig_viser_update(im_c, pos_c, pts_c, clr_c)
                            except Exception:
                                pass
                            viser_q.task_done()

                    viser_worker_thread = threading.Thread(target=_viser_worker, daemon=True)
                    viser_worker_thread.start()

                    viser_counter = 0
                    def _async_viser_update(im_arg, pos_arg, pts_arg, clr_arg):
                        nonlocal viser_counter
                        viser_counter += 1
                        # Throttle to every 2nd keyframe and skip if worker is busy compressing PNG
                        if viser_counter % 2 == 0 and not viser_q.full():
                            try:
                                viser_q.put_nowait((im_arg.detach().clone(), pos_arg.detach().clone(), pts_arg.detach().clone(), clr_arg.detach().clone()))
                            except Exception:
                                pass

                    slam.viser.update = _async_viser_update

            slam(t, image, intrinsics)
            processed += 1
            pct = min(100.0, (processed / total_imgs) * 100.0)
            print(f'[TELEMETRY:STAGE3] batch={processed} total={total_imgs} pct={pct:.1f}', flush=True)
    finally:
        if reader.is_alive():
            reader.terminate()
            reader.join(timeout=1.0)

    if processed != total_imgs or slam is None or not slam.is_initialized:
        raise RuntimeError(f'Incomplete or uninitialized trajectory: {processed}/{total_imgs} selected frames')
    print(f'[*] Running final sliding-window refinement ({final_ba_iters} steps; not global BA)...', flush=True)
    for _ in range(max(1, final_ba_iters)):
        slam.update()

    print('[*] Extracting 6-DoF trajectory and 3D tie points...', flush=True)
    cloud_keep=None
    if mask_atlas is not None:
        center=slam.P//2;xy=slam.patches[0,:slam.m,:2,center,center]
        cloud_keep=mask_atlas.allowed(slam.tstamps_[slam.ix[:slam.m]],xy)

    if dense_cloud and slam.m > 0:
        # Extract all 9 projective depth vertices per patch for 9x higher point cloud density
        try:
            with torch.no_grad():
                pts_homo = pops.point_cloud(
                    SE3(slam.poses),
                    slam.patches[:, :slam.m],
                    slam.intrinsics,
                    slam.ix[:slam.m]
                )
                pts_3d = (pts_homo[..., :3] / pts_homo[..., 3:].clamp(min=1e-6)).reshape(-1, 3)
                clr_base = slam.colors_.view(-1, 3)[:slam.m]
                clr_9x = clr_base.unsqueeze(1).repeat(1, 9, 1).reshape(-1, 3)

                valid = torch.isfinite(pts_3d).all(dim=-1) & (pts_3d.norm(dim=-1) < 200.0)
                if cloud_keep is not None:valid &= cloud_keep.repeat_interleave(9)
                points = pts_3d[valid].cpu().numpy()
                colors = clr_9x[valid].cpu().numpy()
                print(f'[*] Dense 3x3 Micro-Patch extraction: generated {len(points)} high-density tie points.')
        except Exception as e:
            print(f'[WARNING] Dense 9x unprojection fallback to center points: {e}', flush=True)
            points = slam.points_.cpu().numpy()[:slam.m]
            colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
            if cloud_keep is not None:
                keep=cloud_keep.cpu().numpy();points=points[keep];colors=colors[keep]
    else:
        points = slam.points_.cpu().numpy()[:slam.m]
        colors = slam.colors_.view(-1, 3).cpu().numpy()[:slam.m]
        if cloud_keep is not None:
            keep=cloud_keep.cpu().numpy();points=points[keep];colors=colors[keep]

    poses, tstamps = slam.terminate()
    persistent_tracks.save(os.path.join(output_dir, 'dvo_tracks.json'))

    if scale and scale > 0.0 and abs(scale - 1.0) > 1e-4:
        poses[:, :3] *= scale
        points *= scale

    traj = PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=poses[:, [6, 3, 4, 5]],
        timestamps=tstamps
    )

    # Save TUM trajectory
    traj_path = os.path.join(output_dir, '360dvo_trajectory.txt')
    file_interface.write_tum_trajectory_file(traj_path, traj)
    print(f'[SUCCESS] Wrote TUM trajectory to {traj_path}')

    # Save points PLY directly
    ply_path = os.path.join(output_dir, '360dvo_points.ply')
    try:
        with open(ply_path, 'w', encoding='utf-8') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(points, colors):
                r = int(np.clip(c[0] * 255.0 if c.max() <= 1.0 else c[0], 0, 255))
                g = int(np.clip(c[1] * 255.0 if c.max() <= 1.0 else c[1], 0, 255))
                b = int(np.clip(c[2] * 255.0 if c.max() <= 1.0 else c[2], 0, 255))
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")
        print(f'[SUCCESS] Wrote point cloud ({len(points)} points) to {ply_path}')
    except Exception as e:
        print(f'[WARNING] Could not write ply: {e}')

    print('[SUCCESS] 360DVO execution finished successfully!', flush=True)
    provenance = {'backend': 'neural_360dvo_cuda', 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'frames': total_imgs, 'processed': processed,
                  'seconds': time.perf_counter() - started,
                  'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                  'resolution_width_limit': load_res, 'polar_mask': polar_mask,
                  'final_refinement': 'sliding_window',
                  'persistent_observations': persistent_tracks.count,
                  'persistent_tracks_truncated': persistent_tracks.truncated,
                  'external_masks':{'enabled':mask_atlas is not None,'atlas_bytes':mask_atlas.planes.numel()*mask_atlas.planes.element_size() if mask_atlas is not None else 0,
                                    'masked_weight_evaluations':int(mask_rejected.item()) if mask_rejected is not None else 0,
                                    'cloud_patches_excluded':int((~cloud_keep).sum().item()) if cloud_keep is not None else 0},
                  'upstream_commit': subprocess.run(['git', '-C', '/home/ahmad/360DVO', 'rev-parse', 'HEAD'],
                                                   capture_output=True, text=True).stdout.strip()}
    with open(os.path.join(output_dir, 'engine_provenance.json'), 'w', encoding='utf-8') as stream:
        json.dump(provenance, stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--network', type=str, default='360dvo.pth')
    parser.add_argument('--config', type=str, default='config/360.yaml')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--patches', type=int, default=96)
    parser.add_argument('--ba_window', type=int, default=8)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--dense_cloud', action='store_true', default=False)
    parser.add_argument('--patch_lifetime', type=int, default=13)
    parser.add_argument('--final_ba_iters', type=int, default=12)
    parser.add_argument('--no_gradient_bias', dest='gradient_bias', action='store_false', default=True)
    parser.add_argument('--keyframe_thresh', type=float, default=15.0)
    parser.add_argument('--load_res', type=int, default=1024, help='Max resolution width to prevent DPVO cuda_corr int32 overflow')
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--polar_mask', type=float, default=0.08)
    parser.add_argument('--mask-policy',type=str,default=None)

    args = parser.parse_args()

    run_dvo(
        imagedir=args.imagedir,
        output_dir=args.output_dir,
        network=args.network,
        config_path=args.config,
        stride=args.stride,
        skip=args.skip,
        patches=args.patches,
        ba_window=args.ba_window,
        scale=args.scale,
        viz=args.viz,
        port=args.port,
        dense_cloud=args.dense_cloud,
        patch_lifetime=args.patch_lifetime,
        final_ba_iters=args.final_ba_iters,
        gradient_bias=args.gradient_bias,
        keyframe_thresh=args.keyframe_thresh,
        load_res=args.load_res,
        manifest_path=args.manifest,
        polar_mask=args.polar_mask,
        mask_policy_path=args.mask_policy
    )
