"""
Abu Align - SAM 2 AI Masking Engine
Segment Anything Model 2 (SAM 2.1) integration for interactive single-frame and
temporal video/sequence object masking, refinement, and photogrammetry mask export.
"""

import os
import re
import sys
import time
import glob
import math
import shutil
import urllib.request
import warnings
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch

# Suppress the optional C++ post-processing warning on Windows
warnings.filterwarnings("ignore", message=".*cannot import name '_C' from 'sam2'.*")


def compute_scaled_size(orig_w: int, orig_h: int, scale_mode: str) -> Tuple[int, int]:
    """
    Computes output (width, height) according to target scale mode:
      - 'original' / 'native': 100% full resolution
      - '50%': Half resolution (4x pixel reduction)
      - '4k': Max dimension capped to 3840px
      - '2k': Max dimension capped to 2048px (standard for 3DGS & Web)
      - '1k': Max dimension capped to 1024px
    """
    scale_mode = str(scale_mode).lower().strip()
    if scale_mode in ("original", "native", "100%", "1.0"):
        return orig_w, orig_h
    elif scale_mode in ("50%", "half", "0.5"):
        return max(16, (orig_w + 1) // 2), max(16, (orig_h + 1) // 2)
    elif scale_mode in ("4k", "3840"):
        max_dim = 3840
    elif scale_mode in ("2k", "2048"):
        max_dim = 2048
    elif scale_mode in ("1k", "1024"):
        max_dim = 1024
    else:
        return orig_w, orig_h

    cur_max = max(orig_w, orig_h)
    if cur_max <= max_dim:
        return orig_w, orig_h
    scale = max_dim / float(cur_max)
    return max(16, int(round(orig_w * scale))), max(16, int(round(orig_h * scale)))


# Model Catalog
SAM2_MODELS = {
    "tiny": {
        "name": "SAM 2.1 Hiera Tiny",
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "size_mb": 156,
        "description": "Ultra-fast, lowest VRAM (~15ms prompt latency). Ideal for real-time scrubbing."
    },
    "small": {
        "name": "SAM 2.1 Hiera Small",
        "checkpoint": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "size_mb": 185,
        "description": "Fast and balanced accuracy for typical photogrammetry sequences."
    },
    "base_plus": {
        "name": "SAM 2.1 Hiera Base+",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "size_mb": 320,
        "description": "High accuracy, excellent detail retention on complex subjects."
    },
    "large": {
        "name": "SAM 2.1 Hiera Large",
        "checkpoint": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "size_mb": 900,
        "description": "Maximum accuracy and boundary fidelity. Best for intricate thin structures."
    }
}


def natural_sort_key(s: str):
    """Natural alphanumeric sort key (e.g., frame_2 comes before frame_10)."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]


class SAM2MaskingEngine:
    """
    Core engine managing SAM 2 models, frame ingestion, prompt accumulation,
    bidirectional temporal tracking, mask post-processing, and multi-format export.
    """

    def __init__(self, checkpoints_dir: Optional[str] = None, device: Optional[str] = None):
        if checkpoints_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoints_dir = os.path.join(base_dir, "checkpoints", "sam2")
        self.checkpoints_dir = os.path.abspath(checkpoints_dir)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.current_model_key: Optional[str] = None
        self.video_predictor = None
        self.image_predictor = None

        # Sequence Data
        self.source_type: Optional[str] = None  # 'directory', 'video', 'file_list'
        self.source_path: Optional[str] = None
        self.frame_paths: List[str] = []
        self.frame_names: List[str] = []
        self.video_cap = None
        self.num_frames: int = 0
        self.orig_width: int = 0
        self.orig_height: int = 0

        # SAM2 Inference State
        self.inference_state: Optional[Dict[str, Any]] = None
        self.prompts: Dict[int, Dict[str, Any]] = {}  # frame_idx -> {'points': [...], 'labels': [...], 'box': [...]}
        self.tracked_masks: Dict[int, np.ndarray] = {}  # frame_idx -> binary mask (H, W) uint8 or float16

        # Normalization constants (ImageNet standard)
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

        self.log_callback: Optional[Callable[[str], None]] = None

    def set_log_callback(self, cb: Optional[Callable[[str], None]]):
        """Sets external log sink (e.g. Abu Align main console)."""
        self.log_callback = cb

    def log(self, msg: str):
        """Logs message to external console and stdout."""
        if self.log_callback:
            t_str = time.strftime("%H:%M:%S")
            self.log_callback(f"[{t_str}] [SAM 2] {msg}\n")
        else:
            print(f"[SAM 2] {msg}")

    # -------------------------------------------------------------------------
    # Model Management & Downloading
    # -------------------------------------------------------------------------

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Returns availability and disk status for all supported SAM 2 models."""
        status = {}
        for key, info in SAM2_MODELS.items():
            ckpt_path = os.path.join(self.checkpoints_dir, info["checkpoint"])
            exists = os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 1024 * 1024
            status[key] = {
                "name": info["name"],
                "downloaded": exists,
                "path": ckpt_path if exists else None,
                "size_mb": info["size_mb"],
                "description": info["description"]
            }
        return status

    def download_model(self, model_key: str, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> str:
        """Downloads the model checkpoint from Meta AI with progress tracking."""
        if model_key not in SAM2_MODELS:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(SAM2_MODELS.keys())}")

        info = SAM2_MODELS[model_key]
        ckpt_path = os.path.join(self.checkpoints_dir, info["checkpoint"])

        if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 1024 * 1024:
            return ckpt_path

        temp_path = ckpt_path + ".tmp"
        url = info["url"]

        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if progress_callback and total_size > 0:
                pct = min(100, int(downloaded * 100 / total_size))
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                msg = f"Downloading {info['name']}: {mb_down:.1f} / {mb_total:.1f} MB ({pct}%)"
                progress_callback(downloaded, total_size, msg)

        try:
            urllib.request.urlretrieve(url, temp_path, reporthook=_reporthook if progress_callback else None)
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            os.rename(temp_path, ckpt_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"Failed to download {info['name']} from {url}: {e}")

        return ckpt_path

    def load_model(self, model_key: str = "tiny", progress_callback: Optional[Callable[[str], None]] = None):
        """Loads or switches the active SAM 2 model."""
        if self.current_model_key == model_key and self.video_predictor is not None:
            return

        if progress_callback:
            progress_callback(f"Ensuring checkpoint for {SAM2_MODELS[model_key]['name']}...")

        ckpt_path = self.download_model(model_key, progress_callback=lambda d, t, m: progress_callback(m) if progress_callback else None)

        if progress_callback:
            progress_callback(f"Building SAM 2 video predictor ({model_key}) on {self.device}...")

        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        cfg = SAM2_MODELS[model_key]["config"]

        # Clean previous model from GPU memory if active
        if self.video_predictor is not None:
            del self.video_predictor
            del self.image_predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.video_predictor = build_sam2_video_predictor(cfg, ckpt_path, device=self.device)
        sam_base = build_sam2(cfg, ckpt_path, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam_base)
        self.current_model_key = model_key

        if progress_callback:
            progress_callback(f"Loaded {SAM2_MODELS[model_key]['name']} successfully.")

    # -------------------------------------------------------------------------
    # Frame Sequence & Video Ingestion
    # -------------------------------------------------------------------------

    def load_source(
        self,
        source: Any,
        target_size: Optional[int] = None,
        offload_to_cpu: Optional[bool] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> int:
        """
        Loads an input source:
          - source can be a directory path containing image frames
          - source can be a video file path (.mp4, .mov, .avi, etc.)
          - source can be a list of image file paths
        Prepares SAM 2 frames tensor and initializes inference state.
        """
        if self.video_predictor is None:
            self.load_model("tiny")

        model_image_size = self.video_predictor.image_size if target_size is None else target_size

        # Reset any previous session data
        self.clear_all()

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

        frame_data_list: List[Tuple[str, np.ndarray]] = []  # (name, bgr_img) or (name, path)

        if isinstance(source, str):
            source = os.path.abspath(source)
            self.source_path = source

            if os.path.isdir(source):
                self.source_type = "directory"
                all_files = [
                    os.path.join(source, f) for f in os.listdir(source)
                    if os.path.splitext(f)[-1].lower() in image_extensions
                ]
                all_files.sort(key=natural_sort_key)
                if not all_files:
                    raise ValueError(f"No valid image files found in directory: {source}")
                self.frame_paths = all_files
                self.frame_names = [os.path.basename(f) for f in all_files]

            elif os.path.isfile(source):
                ext = os.path.splitext(source)[-1].lower()
                if ext in video_extensions:
                    self.source_type = "video"
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        raise ValueError(f"Cannot open video file: {source}")
                    self.video_cap = cap
                    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame_names = [f"frame_{i:06d}.png" for i in range(total_video_frames)]
                    self.frame_paths = []
                elif ext in image_extensions:
                    # Single image treated as 1-frame sequence
                    self.source_type = "directory"
                    self.frame_paths = [source]
                    self.frame_names = [os.path.basename(source)]
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
            else:
                raise FileNotFoundError(f"Source path does not exist: {source}")

        elif isinstance(source, (list, tuple)):
            self.source_type = "file_list"
            self.frame_paths = [os.path.abspath(f) for f in source]
            self.frame_names = [os.path.basename(f) for f in self.frame_paths]

        self.num_frames = len(self.frame_names)
        if self.num_frames == 0:
            raise ValueError("No frames loaded from source.")

        # Determine dimensions from first frame
        first_frame = self.get_frame_rgb(0)
        self.orig_height, self.orig_width = first_frame.shape[:2]

        # Automatic offloading policy: if num_frames > 80, offload to CPU to preserve GPU memory
        if offload_to_cpu is None:
            offload_to_cpu = (self.num_frames > 80)

        # Build tensor for SAM 2 video predictor
        device_for_tensor = torch.device("cpu") if offload_to_cpu else torch.device(self.device)
        img_mean = torch.tensor(self.img_mean, dtype=torch.float32, device=device_for_tensor)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=torch.float32, device=device_for_tensor)[:, None, None]

        images_tensor = torch.zeros(self.num_frames, 3, model_image_size, model_image_size,
                                    dtype=torch.float32, device=device_for_tensor)

        for i in range(self.num_frames):
            rgb = self.get_frame_rgb(i)
            # Resize with aspect ratio preserved or stretched to model_image_size
            resized = cv2.resize(rgb, (model_image_size, model_image_size), interpolation=cv2.INTER_AREA)
            # Normalization: (RGB / 255.0 - mean) / std
            t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            if offload_to_cpu:
                t = t.to("cpu")
            else:
                t = t.to(device_for_tensor)
            t -= img_mean
            t /= img_std
            images_tensor[i] = t

            if progress_callback and (i % 10 == 0 or i == self.num_frames - 1):
                pct = int((i + 1) * 100 / self.num_frames)
                progress_callback(i + 1, self.num_frames, f"Loaded frame {i + 1}/{self.num_frames} ({pct}%)")

        # Initialize SAM2 Inference State
        compute_device = torch.device(self.device)
        self.inference_state = {
            "images": images_tensor,
            "num_frames": self.num_frames,
            "offload_video_to_cpu": offload_to_cpu,
            "offload_state_to_cpu": False,
            "video_height": self.orig_height,
            "video_width": self.orig_width,
            "device": compute_device,
            "storage_device": compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {}
        }

        # Warm up feature extractor on frame 0
        self.video_predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)
        return self.num_frames

    def get_frame_rgb(self, frame_idx: int) -> np.ndarray:
        """Returns RGB uint8 image of the requested frame in original resolution."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames - 1}]")

        if self.source_type in ("directory", "file_list"):
            path = self.frame_paths[frame_idx]
            # Use cv2.imdecode to support non-ASCII paths on Windows
            with open(path, "rb") as f:
                bytes_data = np.frombuffer(f.read(), np.uint8)
            bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"Failed to read image: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        elif self.source_type == "video":
            if self.video_cap is None:
                self.video_cap = cv2.VideoCapture(self.source_path)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = self.video_cap.read()
            if not ret or bgr is None:
                raise IOError(f"Failed to read video frame {frame_idx} from {self.source_path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raise RuntimeError("Source is not loaded.")

    # -------------------------------------------------------------------------
    # Interactive Prompt System (Points & Bounding Boxes)
    # -------------------------------------------------------------------------

    def add_point(self, frame_idx: int, x: float, y: float, label: int = 1, obj_id: int = 1) -> np.ndarray:
        """
        Adds a positive (label=1) or negative (label=0) click on a frame.
        Coordinates (x, y) are in original image pixel space.
        Returns the updated mask for this frame as (H, W) uint8.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")

        if frame_idx not in self.prompts:
            self.prompts[frame_idx] = {"points": [], "labels": [], "box": None}

        self.prompts[frame_idx]["points"].append([float(x), float(y)])
        self.prompts[frame_idx]["labels"].append(int(label))

        return self._evaluate_frame_prompts(frame_idx, obj_id)

    def add_box(self, frame_idx: int, box: List[float], obj_id: int = 1) -> np.ndarray:
        """
        Sets a bounding box prompt [x0, y0, x1, y1] on the specified frame.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")

        if frame_idx not in self.prompts:
            self.prompts[frame_idx] = {"points": [], "labels": [], "box": None}

        self.prompts[frame_idx]["box"] = [float(v) for v in box]
        return self._evaluate_frame_prompts(frame_idx, obj_id)

    def remove_last_prompt(self, frame_idx: int, obj_id: int = 1) -> Optional[np.ndarray]:
        """Undoes the most recent click on the frame."""
        if frame_idx not in self.prompts:
            return None

        prompt_data = self.prompts[frame_idx]
        if prompt_data["points"]:
            prompt_data["points"].pop()
            prompt_data["labels"].pop()

        if not prompt_data["points"] and prompt_data["box"] is None:
            self.clear_frame_prompts(frame_idx, obj_id)
            return None

        return self._evaluate_frame_prompts(frame_idx, obj_id)

    def clear_frame_prompts(self, frame_idx: int, obj_id: int = 1):
        """Clears all manual prompts on the specified frame."""
        if frame_idx in self.prompts:
            del self.prompts[frame_idx]

        if self.inference_state is not None:
            try:
                self.video_predictor.clear_all_prompts_in_frame(self.inference_state, frame_idx, obj_id)
            except Exception:
                pass

        if frame_idx in self.tracked_masks:
            del self.tracked_masks[frame_idx]

    def clear_all(self):
        """Resets all prompts, tracking caches, and loaded frames."""
        self.prompts.clear()
        self.tracked_masks.clear()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.frame_paths = []
        self.frame_names = []
        self.num_frames = 0
        self.inference_state = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_keyframes(self) -> List[int]:
        """Returns sorted list of frame indices that contain manual user prompts."""
        return sorted(list(self.prompts.keys()))

    def has_prompts(self, frame_idx: int) -> bool:
        """Returns True if user added clicks or box on this frame."""
        return frame_idx in self.prompts and (len(self.prompts[frame_idx]["points"]) > 0 or self.prompts[frame_idx]["box"] is not None)

    def _autocast_context(self):
        """Returns PyTorch autocast context for bfloat16/float16 acceleration on CUDA."""
        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.autocast("cuda", dtype=torch.bfloat16)
            return torch.autocast("cuda", dtype=torch.float16)
        import contextlib
        return contextlib.nullcontext()

    def _evaluate_frame_prompts(self, frame_idx: int, obj_id: int = 1) -> np.ndarray:
        """Evaluates accumulated prompts on a frame and updates SAM 2 state."""
        prompt_data = self.prompts[frame_idx]
        points = np.array(prompt_data["points"], dtype=np.float32) if prompt_data["points"] else None
        labels = np.array(prompt_data["labels"], dtype=np.int32) if prompt_data["labels"] else None
        box = np.array(prompt_data["box"], dtype=np.float32) if prompt_data["box"] is not None else None

        with torch.inference_mode(), self._autocast_context():
            _, _, mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                box=box,
                clear_old_points=True
            )

        # Convert logits tensor to binary mask uint8
        mask_np = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
        self.tracked_masks[frame_idx] = mask_np
        return mask_np

    # -------------------------------------------------------------------------
    # Timeline Propagation & Tracking
    # -------------------------------------------------------------------------

    def propagate(
        self,
        direction: str = "both",
        start_frame: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_token: Optional[Callable[[], bool]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Propagates masks across the timeline using SAM 2's memory-attention mechanism.
        Directions:
          - 'forward': Tracks from start_frame (or first keyframe) forward to end of video.
          - 'backward': Tracks from start_frame backward to frame 0.
          - 'both': Tracks forward and backward to cover the entire sequence.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")
        if not self.prompts:
            raise RuntimeError("Please place at least one prompt click before propagating.")

        keyframes = self.get_keyframes()
        if not keyframes:
            raise RuntimeError("Please place at least one prompt click before propagating.")

        earliest_kf = min(keyframes)
        latest_kf = max(keyframes)

        total_steps = self.num_frames
        current_step = 0

        with torch.inference_mode(), self._autocast_context():
            # Step 1: Forward propagation
            if direction in ("forward", "both"):
                if direction == "both":
                    start_fwd = earliest_kf
                else:
                    start_fwd = start_frame if start_frame is not None else earliest_kf
                    if start_fwd not in keyframes:
                        valid_kfs = [k for k in keyframes if k <= start_fwd]
                        start_fwd = max(valid_kfs) if valid_kfs else earliest_kf

                self.log(f"Starting forward tracking from keyframe {start_fwd} to {self.num_frames - 1}...")
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=start_fwd,
                    reverse=False
                ):
                    if cancel_token and cancel_token():
                        break
                    mask_np = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    self.tracked_masks[out_frame_idx] = mask_np
                    current_step += 1
                    if progress_callback:
                        pct = min(100, int(current_step * 100 / total_steps))
                        progress_callback(current_step, total_steps, f"Forward tracking: frame {out_frame_idx}/{self.num_frames - 1} ({pct}%)")

            # Step 2: Backward propagation
            if direction in ("backward", "both"):
                if direction == "both":
                    start_bwd = earliest_kf
                else:
                    start_bwd = start_frame if start_frame is not None else latest_kf
                    if start_bwd not in keyframes:
                        valid_kfs = [k for k in keyframes if k >= start_bwd]
                        start_bwd = min(valid_kfs) if valid_kfs else latest_kf

                if start_bwd > 0:
                    self.log(f"Starting backward tracking from keyframe {start_bwd} down to frame 0...")
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                        self.inference_state,
                        start_frame_idx=start_bwd,
                        reverse=True
                    ):
                        if cancel_token and cancel_token():
                            break
                        mask_np = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                        self.tracked_masks[out_frame_idx] = mask_np
                        current_step += 1
                        if progress_callback:
                            pct = min(100, int(current_step * 100 / total_steps))
                            progress_callback(current_step, total_steps, f"Backward tracking: frame {out_frame_idx}/0 ({pct}%)")
                else:
                    self.log(f"Backward tracking skipped: sequence start frame {start_bwd} already reached.")
                    if direction == "both" and current_step < total_steps:
                        current_step = total_steps
                        if progress_callback:
                            progress_callback(current_step, total_steps, f"Tracking complete: all {self.num_frames} frames covered.")

        return self.tracked_masks

    # -------------------------------------------------------------------------
    # Mask Post-Processing Pipeline
    # -------------------------------------------------------------------------

    @staticmethod
    def postprocess_mask(
        binary_mask: np.ndarray,
        dilation: int = 0,
        feather: int = 0,
        threshold: int = 128,
        fill_holes: bool = True,
        min_speckle: int = 30,
        invert: bool = False
    ) -> np.ndarray:
        """
        Enhances and filters a binary mask with morphological and edge smoothing ops:
          - fill_holes: Closes internal holes using flood fill and contour hierarchy.
          - min_speckle: Eliminates disconnected noise fragments smaller than min_speckle area in pixels.
          - dilation: Expands (+px) or shrinks (-px) the mask boundary.
          - feather: Gaussian edge softening with radius in pixels.
          - threshold: Binarizes after feathering (if threshold > 0, else leaves soft anti-aliased edge).
          - invert: Inverts foreground and background.
        """
        if binary_mask is None or binary_mask.size == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        mask = (binary_mask > 128).astype(np.uint8) * 255

        # 1. Hole Filling
        if fill_holes and np.any(mask > 0):
            h, w = mask.shape
            flood = mask.copy()
            fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, fill_mask, (0, 0), 255)
            flood_inv = cv2.bitwise_not(flood)
            mask = cv2.bitwise_or(mask, flood_inv)

        # 2. Speckle / Island Removal
        if min_speckle > 0 and np.any(mask > 0):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            cleaned = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_speckle:
                    cleaned[labels == i] = 255
            mask = cleaned

        # 3. Dilation / Erosion
        if dilation != 0 and np.any(mask > 0):
            k_size = abs(dilation) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            if dilation > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        # 4. Invert Mask
        if invert:
            mask = cv2.bitwise_not(mask)

        # 5. Feathering / Edge Softening
        if feather > 0:
            k_size = feather * 2 + 1
            sigma = feather / 2.0
            soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (k_size, k_size), sigma)
            if threshold > 0:
                # Binarize with threshold
                mask = (soft_mask >= threshold).astype(np.uint8) * 255
            else:
                # Return soft anti-aliased mask
                mask = np.clip(soft_mask, 0, 255).astype(np.uint8)

        return mask

    def get_processed_mask(
        self,
        frame_idx: int,
        dilation: int = 0,
        feather: int = 0,
        threshold: int = 128,
        fill_holes: bool = True,
        min_speckle: int = 30,
        invert: bool = False
    ) -> Optional[np.ndarray]:
        """Returns the post-processed mask for a specific frame."""
        raw_mask = self.tracked_masks.get(frame_idx, None)
        if raw_mask is None:
            return None
        return self.postprocess_mask(
            raw_mask,
            dilation=dilation,
            feather=feather,
            threshold=threshold,
            fill_holes=fill_holes,
            min_speckle=min_speckle,
            invert=invert
        )

    # -------------------------------------------------------------------------
    # Exporters (RealityScan Mask & Alpha RGBA Cutout - Hardware Accelerated)
    # -------------------------------------------------------------------------

    def export_masks(
        self,
        output_dir: str,
        export_binary: bool = True,
        export_cutout: bool = True,
        rs_naming: bool = True,          # <name>.mask.<fmt> (RealityScan standard)
        mask_format: str = "png",        # "png", "jpg", "webp"
        cutout_format: str = "webp",      # "webp" (recommended: ~1.5MB), "png"
        scale_mode: str = "original",    # "original", "50%", "4k", "2k", "1k"
        quality: int = 92,               # 1-100 for webp/jpg
        png_compression: int = 4,        # 1-9 for png
        use_hardware_accel: bool = True,
        max_workers: Optional[int] = None,
        cutout_dir: Optional[str] = None,
        postprocess_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_token: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Hardware-accelerated and multi-threaded batch export for photogrammetry:
          - export_binary: Exports black & white matte masks (<stem>.mask.<fmt> for RealityScan).
          - export_cutout: Exports transparent RGBA cutouts (<stem>_cutout.<fmt>).
          - cutout_format: 'webp' (fastest, ~1.5MB vs 70MB PNG, transparent alpha) or 'png'.
          - scale_mode: 'original', '50%', '4k', '2k', '1k'.
          - quality: Compression quality for WebP and JPEG.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        t_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        if export_cutout and cutout_dir:
            os.makedirs(cutout_dir, exist_ok=True)

        params = postprocess_params or {}
        dilation = params.get("dilation", 0)
        feather = params.get("feather", 0)
        threshold = params.get("threshold", 128)
        fill_holes = params.get("fill_holes", True)
        min_speckle = params.get("min_speckle", 30)
        invert = params.get("invert", False)

        total_frames = self.num_frames
        if total_frames == 0:
            return {"binary": 0, "cutout": 0, "total_mb": 0.0, "elapsed_sec": 0.0, "fps": 0.0}

        # Target dimensions
        target_w, target_h = compute_scaled_size(self.orig_width, self.orig_height, scale_mode)

        # Worker threads for concurrent NVMe / disk encoding
        if max_workers is None:
            max_workers = min(16, max(4, os.cpu_count() or 4))

        # Setup format codecs & parameters
        # 1. Mask params
        m_fmt = mask_format.lower().lstrip(".")
        if m_fmt == "jpg" or m_fmt == "jpeg":
            mask_ext = ".jpg"
            mask_params = [cv2.IMWRITE_JPEG_QUALITY, max(int(quality), 95)]
        elif m_fmt == "webp":
            mask_ext = ".webp"
            mask_params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
        else:
            mask_ext = ".png"
            mask_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]

        # 2. Cutout params
        c_fmt = cutout_format.lower().lstrip(".")
        if c_fmt == "webp":
            cutout_ext = ".webp"
            cutout_params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
        else:
            cutout_ext = ".png"
            cutout_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]

        self.log(f"Starting Hardware-Accelerated Export: {total_frames} frames -> "
                 f"Res: {target_w}x{target_h}px ({scale_mode.upper()}) | "
                 f"Cutouts: {cutout_ext.upper()} (Q{quality}) | "
                 f"Masks: {mask_ext.upper()} | "
                 f"Parallel Workers: {max_workers} threads")

        target_cutout_folder = cutout_dir if cutout_dir else os.path.join(output_dir, "cutouts")
        if export_cutout:
            os.makedirs(target_cutout_folder, exist_ok=True)

        exported_counts = {"binary": 0, "cutout": 0}
        total_bytes_written = 0

        def _write_file(path: str, data: np.ndarray, enc_params: list) -> int:
            if path is None or data is None:
                return 0
            ext = os.path.splitext(path)[1]
            success, enc = cv2.imencode(ext, data, enc_params)
            if success:
                with open(path, "wb") as f:
                    f.write(enc)
                return len(enc)
            return 0

        # Run multi-threaded pipeline
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for idx in range(total_frames):
                if cancel_token and cancel_token():
                    break

                # 1. Fetch refined mask
                mask = self.get_processed_mask(
                    idx,
                    dilation=dilation,
                    feather=feather,
                    threshold=threshold,
                    fill_holes=fill_holes,
                    min_speckle=min_speckle,
                    invert=invert
                )
                if mask is None:
                    mask = np.zeros((self.orig_height, self.orig_width), dtype=np.uint8)

                frame_name = self.frame_names[idx]
                stem, _ = os.path.splitext(frame_name)

                # 2. Fetch frame image if cutout needed
                rgb = self.get_frame_rgb(idx) if export_cutout else None

                # 3. Scale if target dimensions differ from original
                if (target_w, target_h) != (self.orig_width, self.orig_height):
                    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    if rgb is not None:
                        interp = cv2.INTER_AREA if target_w < self.orig_width else cv2.INTER_LINEAR
                        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=interp)

                # 4. Prepare Mask Task
                mask_path = None
                if export_binary:
                    if rs_naming:
                        mask_filename = f"{stem}.mask{mask_ext}"
                    else:
                        mask_filename = f"{stem}_mask{mask_ext}"
                    mask_path = os.path.join(output_dir, mask_filename)
                    fut_m = executor.submit(_write_file, mask_path, mask, mask_params)
                    futures.append(("binary", stem, fut_m))

                # 5. Prepare Cutout Task
                if export_cutout and rgb is not None:
                    cutout_filename = f"{stem}_cutout{cutout_ext}"
                    out_cutout_path = os.path.join(target_cutout_folder, cutout_filename)

                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                    bgra[:, :, 3] = mask  # 8-bit alpha transparency

                    fut_c = executor.submit(_write_file, out_cutout_path, bgra, cutout_params)
                    futures.append(("cutout", stem, fut_c))

                if progress_callback and (idx % 5 == 0 or idx == total_frames - 1):
                    pct = int((idx + 1) * 50 / total_frames)  # first 50% for dispatch
                    progress_callback(idx + 1, total_frames, f"Preparing {stem} ({idx + 1}/{total_frames} - {pct}%)")

            # Collect completed writing tasks
            total_tasks = len(futures)
            completed_tasks = 0
            for kind, stem, fut in futures:
                if cancel_token and cancel_token():
                    break
                try:
                    bytes_n = fut.result()
                    total_bytes_written += bytes_n
                    if bytes_n > 0:
                        exported_counts[kind] += 1
                except Exception as e:
                    self.log(f"Error writing {stem}: {e}")

                completed_tasks += 1
                if progress_callback and (completed_tasks % 10 == 0 or completed_tasks == total_tasks):
                    pct = 50 + int(completed_tasks * 50 / max(1, total_tasks))
                    elapsed = max(0.001, time.time() - t_start)
                    cur_fps = (completed_tasks / 2.0) / elapsed
                    progress_callback(
                        completed_tasks,
                        total_tasks,
                        f"Writing {stem} ({completed_tasks}/{total_tasks} - {pct}%, {cur_fps:.1f} fps)"
                    )

        dt = max(0.001, time.time() - t_start)
        fps = total_frames / dt
        total_mb = total_bytes_written / (1024 * 1024)
        avg_cutout_mb = (total_mb / max(1, exported_counts["cutout"])) if exported_counts["cutout"] > 0 else 0.0

        # Calculate space savings vs uncompressed 70MB PNG
        est_old_png_mb = exported_counts["cutout"] * 70.0
        saved_mb = max(0.0, est_old_png_mb - total_mb)

        summary_msg = (
            f"Export Complete: {exported_counts['binary']} masks + {exported_counts['cutout']} cutouts in {dt:.2f}s "
            f"({fps:.1f} fps). Total: {total_mb:.1f} MB (Avg cutout: {avg_cutout_mb:.2f} MB, saved ~{saved_mb:.1f} MB vs 70MB PNG)"
        )
        self.log(summary_msg)

        return {
            "binary": exported_counts["binary"],
            "cutout": exported_counts["cutout"],
            "total_mb": total_mb,
            "elapsed_sec": dt,
            "fps": fps,
            "avg_cutout_mb": avg_cutout_mb,
            "saved_mb": saved_mb,
            "scale_res": f"{target_w}x{target_h}"
        }
