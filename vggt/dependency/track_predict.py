# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import numpy as np
from .vggsfm_utils import *


def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
    max_augment_queries=0,
    *,
    window_size=32,
    window_stride=24,
    image_paths=None,
    engine="vggt_omega",
    preprocess_mode="max_size",
    vggt_res=512,
    img_load_res=1024,
):
    """
    Predict tracks for the given images and masks.

    Supports both monolithic tracking (for <= 48 frames) and sliding-window zero-paging
    tracking (for > 48 frames or large sequences up to thousands of frames).
    """
    # Support list of image paths passed as images or image_paths
    if image_paths is None and isinstance(images, (list, tuple)) and len(images) > 0 and isinstance(images[0], str):
        image_paths = list(images)
        images = None

    if image_paths is not None:
        num_frames = len(image_paths)
    elif isinstance(images, torch.Tensor):
        num_frames = images.shape[0]
    else:
        num_frames = 0

    # Auto-dispatch to sliding-window mode for large scans or when images is not a monolithic GPU tensor
    if num_frames > 48 or images is None:
        return predict_tracks_windowed(
            images=images,
            conf=conf,
            points_3d=points_3d,
            masks=masks,
            max_query_pts=max_query_pts,
            query_frame_num=query_frame_num,
            keypoint_extractor=keypoint_extractor,
            max_points_num=max_points_num,
            fine_tracking=fine_tracking,
            complete_non_vis=complete_non_vis,
            max_augment_queries=max_augment_queries,
            window_size=window_size,
            window_stride=window_stride,
            image_paths=image_paths,
            engine=engine,
            preprocess_mode=preprocess_mode,
            vggt_res=vggt_res,
            img_load_res=img_load_res,
        )

    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    # Find query frames
    if isinstance(query_frame_num, (list, tuple, np.ndarray)):
        query_frame_indexes = list(query_frame_num)
    else:
        query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

    # Add the first image to the front if not already present
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    # TODO: add the functionality to handle the masks
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    t_track_start = time.time()
    total_queries = len(query_frame_indexes)

    for q_i, query_index in enumerate(query_frame_indexes):
        q_elapsed = time.time() - t_track_start
        q_avg = q_elapsed / max(q_i, 1)
        q_eta = q_avg * (total_queries - q_i) if q_i > 0 else 0
        eta_str = f" | ETA: {int(q_eta//60)}m {int(q_eta%60):02d}s" if q_i > 0 else ""
        pct = (q_i / total_queries) * 100

        print(f"[TELEMETRY:STAGE5] query={q_i+1} total={total_queries} frame={query_index} pct={pct:.1f} eta_sec={int(q_eta)}", flush=True)
        print(f"      * [Deep Tracking] Query [{q_i+1}/{total_queries}] (Frame {query_index}) - {pct:.1f}%{eta_str}...", flush=True)
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)

    if complete_non_vis and max_augment_queries > 0:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            min_vis=500,
            non_vis_thresh=0.1,
            device=device,
            max_augment_queries=max_augment_queries,
        )

    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    # from vggt.utils.visual_track import visualize_tracks_on_images
    # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    """
    Process a single query frame for track prediction.

    Args:
        query_index: Index of the query frame
        images: Tensor of shape [S, 3, H, W] containing the input images
        conf: Confidence tensor
        points_3d: 3D points tensor
        fmaps_for_tracker: Feature maps for the tracker
        keypoint_extractors: Initialized feature extractors
        tracker: VGG-SFM tracker
        max_points_num: Maximum number of points to process at once
        fine_tracking: Whether to use fine tracking
        device: Device to use for computation

    Returns:
        pred_track: Predicted tracks
        pred_vis: Visibility scores for the tracks
        pred_conf: Confidence scores for the tracks
        pred_point_3d: 3D points for the tracks
        pred_color: Point colors for the tracks (0, 255)
    """
    frame_num, _, height, width = images.shape

    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    # Extract the color at the keypoint locations
    query_points_long = query_points.squeeze(0).round().long()
    qy = query_points_long[:, 1].clamp(0, height - 1)
    qx = query_points_long[:, 0].clamp(0, width - 1)
    pred_color = images[query_index][:, qy, qx]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    # Query the confidence and points_3d at the keypoint locations
    if (conf is not None) and (points_3d is not None):
        conf_h, conf_w = conf.shape[-2], conf.shape[-1]
        scale_x = float(conf_w) / float(width)
        scale_y = float(conf_h) / float(height)

        pts = query_points.squeeze(0)
        qx_scaled = (pts[:, 0] * scale_x).round().long().clamp(0, conf_w - 1).cpu().numpy()
        qy_scaled = (pts[:, 1] * scale_y).round().long().clamp(0, conf_h - 1).cpu().numpy()

        pred_conf = conf[query_index][qy_scaled, qx_scaled]
        pred_point_3d = points_3d[query_index][qy_scaled, qx_scaled]

        # heuristic to remove low confidence points
        # should I export this as an input parameter?
        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]  # Make sure shape is compatible
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]  # add batch dimension
    fmaps_feed = fmaps_feed[None]  # add batch dimension

    all_points_num = images_feed.shape[1] * query_points.shape[1]

    # Don't need to be scared, this is just chunking to make GPU happy
    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    reorder_index_dev = reorder_index.to(pred_track.device) if isinstance(reorder_index, torch.Tensor) else reorder_index
    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index_dev, dim=1)

    pred_track = pred_track.detach().squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.detach().squeeze(0).float().cpu().numpy()

    del images_feed, fmaps_feed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _augment_non_visible_frames(
    pred_tracks: list,  # ← running list of np.ndarrays
    pred_vis_scores: list,  # ← running list of np.ndarrays
    pred_confs: list,  # ← running list of np.ndarrays for confidence scores
    pred_points_3d: list,  # ← running list of np.ndarrays for 3D points
    pred_colors: list,  # ← running list of np.ndarrays for colors
    images: torch.Tensor,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    *,
    min_vis: int = 500,
    non_vis_thresh: float = 0.1,
    device: torch.device = None,
    max_augment_queries: int = 2,
):
    """
    Augment tracking for frames with insufficient visibility.
    Bounded by max_augment_queries to prevent infinite / marathon loops on large scans.
    """
    if max_augment_queries <= 0:
        return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors

    last_query = -1
    final_trial = False
    cur_extractors = keypoint_extractors  # may be replaced on the final trial
    augment_count = 0

    while True:
        # Visibility per frame
        vis_array = np.concatenate(pred_vis_scores, axis=1)

        # Count frames with sufficient visibility using numpy
        sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
        non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()

        if len(non_vis_frames) == 0:
            break

        # In long camera trajectories (>20 non-co-visible frames), visibility is local, not global.
        # Avoid looping through hundreds of unlinked frames sequentially.
        if len(non_vis_frames) > 20:
            print(f"      * [Deep Tracking] Scene exhibits open trajectory co-visibility ({len(non_vis_frames)} frames out of view from initial queries); proceeding with anchor tracks to avoid redundant passes.")
            break

        if augment_count >= max_augment_queries:
            print(f"      * [Deep Tracking] Reached max query augmentation limit ({max_augment_queries}); proceeding to bundle adjustment.")
            break
        augment_count += 1

        print(f"Processing non visible frames ({len(non_vis_frames)} remaining, pass {augment_count}/{max_augment_queries}):", non_vis_frames)

        # Decide the frames & extractor for this round
        if non_vis_frames[0] == last_query:
            # Same frame failed twice - final attempt (strictly bounded)
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_vis_frames[:max_augment_queries]
        else:
            query_frame_list = [non_vis_frames[0]]  # Process one at a time

        last_query = non_vis_frames[0]

        # Run the tracker for every selected frame
        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
            )
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_confs.append(new_conf)
            pred_points_3d.append(new_point_3d)
            pred_colors.append(new_color)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if final_trial:
            break  # Stop after final attempt

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def predict_tracks_windowed(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=1024,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=False,
    complete_non_vis=False,
    max_augment_queries=0,
    *,
    window_size=32,
    window_stride=24,
    image_paths=None,
    engine="vggt_omega",
    preprocess_mode="max_size",
    vggt_res=512,
    img_load_res=1024,
    device=None,
    dtype=torch.float32,
):
    """
    Sliding-Window Zero-Paging Multi-View Point Tracker.
    Divides massive sequences (up to thousands of frames) into overlapping temporal windows.
    Runs CoTracker within bounded windows (default 32 frames), preventing PCIe paging and VRAM exhaustion.
    """
    if device is None:
        if isinstance(images, torch.Tensor):
            device = images.device
            dtype = images.dtype
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_paths is not None:
        num_frames = len(image_paths)
    elif isinstance(images, torch.Tensor):
        num_frames = images.shape[0]
    elif isinstance(images, (list, tuple)):
        image_paths = list(images)
        num_frames = len(image_paths)
    else:
        raise ValueError("Either images tensor or image_paths must be provided to predict_tracks_windowed")

    # Generate overlapping windows
    windows = []
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        windows.append((start, end))
        if end == num_frames:
            break
        start += window_stride
    num_windows = len(windows)

    print(f"      * [Sliding Window Tracking] Initialized {num_windows} temporal windows (size={window_size}, stride={window_stride}) for {num_frames} views.")

    # Initialize tracker and feature extractors once
    tracker = build_vggsfm_tracker().to(device, dtype)
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    all_pred_tracks = []
    all_pred_vis = []
    all_pred_confs = []
    all_pred_pts3d = []
    all_pred_colors = []

    t_track_start = time.time()

    # Pre-import image loader if loading from disk
    image_loader = None
    if image_paths is not None:
        from align_cli import load_and_preprocess_vggt_images
        image_loader = load_and_preprocess_vggt_images

    # Normalize conf shape if needed: (N, H, W)
    if conf is not None:
        if isinstance(conf, torch.Tensor):
            conf_np = conf.float().cpu().numpy()
        else:
            conf_np = conf
        if conf_np.ndim == 4 and conf_np.shape[-1] == 1:
            conf_np = conf_np.squeeze(-1)
        elif conf_np.ndim == 4 and conf_np.shape[1] == 1:
            conf_np = conf_np.squeeze(1)
    else:
        conf_np = None

    if points_3d is not None:
        if isinstance(points_3d, torch.Tensor):
            pts3d_np = points_3d.float().cpu().numpy()
        else:
            pts3d_np = points_3d
    else:
        pts3d_np = None

    for w_idx, (s, e) in enumerate(windows):
        w_len = e - s
        q_elapsed = time.time() - t_track_start
        q_avg = q_elapsed / max(w_idx, 1)
        q_eta = q_avg * (num_windows - w_idx) if w_idx > 0 else 0
        eta_str = f" | ETA: {int(q_eta//60)}m {int(q_eta%60):02d}s" if w_idx > 0 else ""
        pct = (w_idx / num_windows) * 100

        print(f"[TELEMETRY:STAGE5] query={w_idx+1} total={num_windows} frame={s} pct={pct:.1f} eta_sec={int(q_eta)}", flush=True)
        print(f"      * [Sliding Window Tracking] Window [{w_idx+1}/{num_windows}] (Frames {s}..{e-1}) - {pct:.1f}%{eta_str}...", flush=True)

        # 1. Load window images on demand (Zero VRAM footprint outside window)
        if image_loader is not None:
            w_paths = image_paths[s:e]
            win_imgs, _, win_rgb, _ = image_loader(
                w_paths,
                engine=engine,
                preprocess_mode=preprocess_mode,
                vggt_res=vggt_res,
                img_load_res=img_load_res,
                device=device,
            )
        else:
            win_imgs = images[s:e].to(device)

        with torch.no_grad():
            fmaps_win = tracker.process_images_to_fmaps(win_imgs)

        # 2. Determine query frame(s) within window
        # Window start frame is always queried (local index 0 -> global frame s)
        # For multi-pass or standard/ultra modes, also query middle frame to maximize track density
        q_indices_local = [0]
        if complete_non_vis and w_len > 12:
            q_indices_local.append(w_len // 2)

        conf_win = conf_np[s:e] if conf_np is not None else None
        pts3d_win = pts3d_np[s:e] if pts3d_np is not None else None

        for q_loc in q_indices_local:
            with torch.no_grad():
                pred_track, pred_vis, pred_conf, pred_pt3d, pred_color = _forward_on_query(
                    q_loc,
                    win_imgs,
                    conf_win,
                    pts3d_win,
                    fmaps_win,
                    keypoint_extractors,
                    tracker,
                    max_points_num,
                    fine_tracking,
                    device,
                )

            K = pred_track.shape[1]
            full_track = np.zeros((num_frames, K, 2), dtype=np.float32)
            full_vis = np.zeros((num_frames, K), dtype=np.float32)
            full_track[s:e] = pred_track
            full_vis[s:e] = pred_vis

            all_pred_tracks.append(full_track)
            all_pred_vis.append(full_vis)
            if pred_conf is not None:
                all_pred_confs.append(pred_conf)
            if pred_pt3d is not None:
                all_pred_pts3d.append(pred_pt3d)
            if pred_color is not None:
                all_pred_colors.append(pred_color)

        # 3. Purge window tensors from GPU immediately
        del win_imgs, fmaps_win
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_tracks = np.concatenate(all_pred_tracks, axis=1) if all_pred_tracks else np.zeros((num_frames, 0, 2), dtype=np.float32)
    pred_vis_scores = np.concatenate(all_pred_vis, axis=1) if all_pred_vis else np.zeros((num_frames, 0), dtype=np.float32)
    pred_confs = np.concatenate(all_pred_confs, axis=0) if all_pred_confs else None
    pred_points_3d = np.concatenate(all_pred_pts3d, axis=0) if all_pred_pts3d else None
    pred_colors = np.concatenate(all_pred_colors, axis=0) if all_pred_colors else None

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors
