"""
Abu Align & Train - Multi-Tier Intelligent Caching Manager
===========================================================
Provides fast persistent caching for neural photogrammetry and multi-view tracking:

Tier 1: Neural Inference Cache (extrinsic poses, intrinsics, 3D point maps, confidence)
        -> Skips Stage 3 & 4 (30-60 neural batch passes) when iterating on point cloud density,
           lens distortion (Pinhole/Brown), or bundle adjustment precision.

Tier 2: Multi-View Feature Tracking Cache (2D feature tracks, visibility scores)
        -> Skips CoTracker deep transformer tracking when re-solving bundle adjustment.

Cache Location: <dataset_dir>/results/.abu_cache/<cache_key>/
"""

import os
import sys
import time
import json
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch


def compute_dataset_hash(image_paths: List[str], engine: str, load_res: int, stride: int) -> str:
    """
    Computes a deterministic SHA-256 fingerprint representing the exact input dataset
    and neural configuration.
    """
    hasher = hashlib.sha256()
    hasher.update(str(engine).lower().encode("utf-8"))
    hasher.update(str(load_res).encode("utf-8"))
    hasher.update(str(stride).encode("utf-8"))
    hasher.update(str(len(image_paths)).encode("utf-8"))

    # Sample up to 100 paths with sizes and timestamps for ultra-fast validation
    step = max(1, len(image_paths) // 100)
    for p in image_paths[::step]:
        hasher.update(os.path.basename(p).encode("utf-8"))
        try:
            st = os.stat(p)
            hasher.update(str(st.st_size).encode("utf-8"))
            hasher.update(str(int(st.st_mtime)).encode("utf-8"))
        except OSError:
            pass

    return hasher.hexdigest()[:16]


def get_cache_root(dataset_dir: str, custom_cache_dir: Optional[str] = None) -> str:
    """
    Determines the root cache directory.
    Default: <dataset_dir>/results/.abu_cache
    Falls back to <dataset_dir>/.abu_cache if it already exists.
    """
    if custom_cache_dir:
        return os.path.abspath(custom_cache_dir)

    # Prefer results/.abu_cache to keep images folder clean
    local_cache = os.path.join(dataset_dir, ".abu_cache")
    if os.path.isdir(local_cache):
        return local_cache

    results_cache = os.path.join(dataset_dir, "results", ".abu_cache")
    return results_cache


def save_neural_cache(
    cache_dir: str,
    cache_key: str,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    points_3d: torch.Tensor,
    depth_conf: torch.Tensor,
    original_coords: torch.Tensor,
    depth_map: Optional[torch.Tensor] = None,
    points_rgb: Optional[np.ndarray] = None,
    images: Optional[torch.Tensor] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves compressed neural predictions to disk.
    Tensors are cast to float16 to minimize disk footprint while preserving millimeter accuracy.
    """
    target_dir = os.path.join(cache_dir, cache_key)
    os.makedirs(target_dir, exist_ok=True)

    t0 = time.time()
    # Save poses & intrinsics (CPU tensors)
    torch.save(extrinsic.detach().cpu().float(), os.path.join(target_dir, "extrinsic.pt"))
    torch.save(intrinsic.detach().cpu().float(), os.path.join(target_dir, "intrinsic.pt"))
    torch.save(original_coords.detach().cpu(), os.path.join(target_dir, "original_coords.pt"))

    # Save 3D points, depth map, and confidence in half-precision to save disk bandwidth
    torch.save(points_3d.detach().cpu().half(), os.path.join(target_dir, "points_3d_fp16.pt"))
    torch.save(depth_conf.detach().cpu().half(), os.path.join(target_dir, "depth_conf_fp16.pt"))
    if depth_map is not None:
        torch.save(depth_map.detach().cpu().half(), os.path.join(target_dir, "depth_map_fp16.pt"))

    if points_rgb is not None:
        np.save(os.path.join(target_dir, "points_rgb.npy"), points_rgb)
    if images is not None:
        torch.save((images.detach().cpu() * 255).byte(), os.path.join(target_dir, "images_u8.pt"))

    # Write metadata
    meta = metadata or {}
    meta.update({
        "cache_key": cache_key,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_frames": int(extrinsic.shape[0]),
        "pts_shape": list(points_3d.shape),
    })
    with open(os.path.join(target_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    return target_dir


def load_neural_cache(
    cache_dir: str,
    cache_key: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> Optional[Dict[str, Any]]:
    """
    Loads cached neural predictions from disk.
    Returns dict of tensors placed on target device/dtype, or None if cache miss.
    """
    target_dir = os.path.join(cache_dir, cache_key)
    required_files = ["extrinsic.pt", "intrinsic.pt", "points_3d_fp16.pt", "depth_conf_fp16.pt", "original_coords.pt"]

    for rf in required_files:
        if not os.path.isfile(os.path.join(target_dir, rf)):
            return None

    try:
        t0 = time.time()
        extrinsic = torch.load(os.path.join(target_dir, "extrinsic.pt"), map_location="cpu").to(device, dtype)
        intrinsic = torch.load(os.path.join(target_dir, "intrinsic.pt"), map_location="cpu").to(device, dtype)
        original_coords = torch.load(os.path.join(target_dir, "original_coords.pt"), map_location="cpu").to(device)

        points_3d = torch.load(os.path.join(target_dir, "points_3d_fp16.pt"), map_location="cpu").to(device, dtype)
        depth_conf = torch.load(os.path.join(target_dir, "depth_conf_fp16.pt"), map_location="cpu").to(device, dtype)

        depth_map = None
        dm_path = os.path.join(target_dir, "depth_map_fp16.pt")
        if os.path.isfile(dm_path):
            depth_map = torch.load(dm_path, map_location="cpu").to(device, dtype)

        points_rgb = None
        rgb_path = os.path.join(target_dir, "points_rgb.npy")
        if os.path.isfile(rgb_path):
            points_rgb = np.load(rgb_path)

        images = None
        images_path = os.path.join(target_dir, "images_u8.pt")
        if os.path.isfile(images_path):
            images = (torch.load(images_path, map_location="cpu").float() / 255.0).to(device)

        meta_file = os.path.join(target_dir, "meta.json")
        meta = {}
        if os.path.isfile(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        elapsed = time.time() - t0
        return {
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "points_3d": points_3d,
            "depth_map": depth_map,
            "depth_conf": depth_conf,
            "original_coords": original_coords,
            "meta": meta,
            "load_time_sec": elapsed,
        }
    except Exception as e:
        print(f"[Abu Cache Warning] Failed to load neural cache from {target_dir}: {e}")
        return None


def save_tracking_cache(
    cache_dir: str,
    cache_key: str,
    query_indices: List[int],
    pred_tracks: np.ndarray,
    pred_vis_scores: np.ndarray,
    pred_confs: Optional[np.ndarray],
    points_3d_ba: Optional[np.ndarray],
    points_rgb_ba: Optional[np.ndarray]
) -> str:
    """
    Saves multi-view 2D feature trajectories and visibility masks.
    """
    target_dir = os.path.join(cache_dir, cache_key, "tracking")
    os.makedirs(target_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(target_dir, "tracks.npz"),
        query_indices=np.array(query_indices, dtype=np.int32),
        pred_tracks=pred_tracks.astype(np.float32),
        pred_vis_scores=pred_vis_scores.astype(np.float32),
        pred_confs=pred_confs.astype(np.float32) if pred_confs is not None else np.array([]),
        points_3d_ba=points_3d_ba.astype(np.float32) if points_3d_ba is not None else np.array([]),
        points_rgb_ba=points_rgb_ba.astype(np.uint8) if points_rgb_ba is not None else np.array([])
    )
    return target_dir


def load_tracking_cache(
    cache_dir: str,
    cache_key: str,
    query_indices: List[int]
) -> Optional[Dict[str, Any]]:
    """
    Loads cached multi-view feature trajectories if query indices match.
    """
    npz_path = os.path.join(cache_dir, cache_key, "tracking", "tracks.npz")
    if not os.path.isfile(npz_path):
        return None

    try:
        data = np.load(npz_path)
        cached_queries = list(data["query_indices"])
        if cached_queries != query_indices:
            # Query configuration differs, invalidating track cache
            return None

        return {
            "pred_tracks": data["pred_tracks"],
            "pred_vis_scores": data["pred_vis_scores"],
            "pred_confs": data["pred_confs"] if len(data["pred_confs"]) > 0 else None,
            "points_3d_ba": data["points_3d_ba"] if len(data["points_3d_ba"]) > 0 else None,
            "points_rgb_ba": data["points_rgb_ba"] if len(data["points_rgb_ba"]) > 0 else None,
        }
    except Exception as e:
        print(f"[Abu Cache Warning] Failed to load tracking cache: {e}")
        return None


def clear_cache(cache_dir: str) -> Tuple[int, int]:
    """
    Purges all cached data within cache_dir.
    Returns (num_entries_removed, total_bytes_freed).
    """
    if not os.path.isdir(cache_dir):
        return 0, 0

    total_bytes = 0
    count = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))

    try:
        shutil.rmtree(cache_dir)
        count = 1
    except Exception as e:
        print(f"[Abu Cache Error] Failed to delete cache dir {cache_dir}: {e}")

    return count, total_bytes
