# Abu Align & Train v0.0.6.8 — Cross-Device Tensor Synchronization & Zero-Paging Deep Tracking Stability

Version **v0.0.6.8** resolves the cross-device tensor ordering crash `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!` reported in `abu_align_2026-09-25_11-33-15.log` and `abu_align_2026-09-25_11-35-00.log`.

---

## 1. Problem Description & Root Cause Analysis

In `logs/abu_align_2026-09-25_11-33-15.log` & `logs/abu_align_2026-09-25_11-35-00.log`:
- **Stage**: Stage 5 Deep Multi-View Tracking & Ceres BA (`Running Deep Multi-View Tracking [FAST mode: 3 fast queries...]`).
- **Crash Traceback**:
  ```text
  File "align_cli.py", line 1082, in run_vggt_reconstruction
    pred_tracks, pred_vis_scores, pred_confs, points_3d_ba, points_rgb_ba = predict_tracks(...)
  File "vggt\dependency\track_predict.py", line 99, in predict_tracks
    pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(...)
  File "vggt\dependency\track_predict.py", line 242, in _forward_on_query
    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)
  File "vggt\dependency\vggsfm_utils.py", line 186, in switch_tensor_order
    return [torch.index_select(tensor, dim, order) if tensor is not None else None for tensor in tensors]
  RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument index in method wrapper_CUDA__index_select)
  ```

### Root Cause
1. In `vggt/dependency/vggsfm_utils.py`, `predict_tracks_in_chunks` offloads keypoint track prediction chunks to CPU memory (`fine_pred_track_list.append(fine_pred_track.cpu())`) to avoid exhausting GPU VRAM.
2. The outputs `pred_track` and `pred_vis` are therefore tensors stored on **CPU**.
3. However, `reorder_index` was computed on CUDA via `calculate_index_mappings(..., device=device)`.
4. Calling `torch.index_select(pred_track, dim=1, index=reorder_index)` attempted to index a CPU tensor using a CUDA index tensor.
5. PyTorch requires the index tensor and the indexed tensor to reside on the identical device, causing an immediate CUDA kernel exception.

---

## 2. Technical Solution

1. **Dynamic Device Matching in `switch_tensor_order`** (`vggsfm_utils.py`):
   Updated `switch_tensor_order` so that `order` is dynamically transferred to `tensor.device` for every indexed tensor:
   ```python
   def switch_tensor_order(tensors, order, dim=1):
       res = []
       for tensor in tensors:
           if tensor is None:
               res.append(None)
           else:
               order_dev = order.to(tensor.device) if isinstance(order, torch.Tensor) else order
               res.append(torch.index_select(tensor, dim, order_dev))
       return res
   ```
2. **Explicit Device Guard in `_forward_on_query`** (`track_predict.py`):
   Ensured `reorder_index` is matched directly to `pred_track.device` before calling `switch_tensor_order`:
   ```python
   reorder_index_dev = reorder_index.to(pred_track.device) if isinstance(reorder_index, torch.Tensor) else reorder_index
   pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index_dev, dim=1)
   ```

---

## 3. Verification & Testing

- Added `test_switch_tensor_order_cross_device` to `tests/test_tracking_optimization.py` validating that indexing operations succeed regardless of whether tensors and index order originate on CPU, CUDA, or mixed devices.
- Full regression suite execution:
  ```bash
  python -m unittest discover tests
  ..............................................
  Ran 46 tests in 28.929s
  OK
  ```

---

## 4. Versioning & Executables
- Bumped version to `v0.0.6.8` via `version_manager.py`.
- Recompiled `Abu_Align.exe` and `Abu_Align_and_Train.exe`.
- Created version archive snapshot `versions/v0.0.6.8/`.
