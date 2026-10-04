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
import threading

import numpy as np
import cv2
import torch

# Suppress the optional C++ post-processing warning on Windows
warnings.filterwarnings("ignore", message=".*cannot import name '_C' from 'sam2'.*")


def compute_scaled_size(orig_w: int, orig_h: int, scale_mode: str) -> Tuple[int, int]:
    """
    Computes output (width, height) preserving each frame's native aspect ratio and orientation:
      - 'original' / 'native': 100% full resolution
      - '50%': Half resolution (w // 2, h // 2)
      - '25%': Quarter resolution (w // 4, h // 4)
      - '4k': Longest edge capped to 3840px
      - '2k': Longest edge capped to 2048px (standard for 3DGS & Web)
      - '1080p': Longest edge capped to 1920px
      - '1k': Longest edge capped to 1024px
    """
    sm = str(scale_mode).lower().strip()
    if any(k in sm for k in ("original", "native", "100%", "1.0")):
        return orig_w, orig_h
    elif "25%" in sm or "quarter" in sm or "0.25" in sm:
        return max(16, (orig_w + 3) // 4), max(16, (orig_h + 3) // 4)
    elif "50%" in sm or "half" in sm or "0.5" in sm:
        return max(16, (orig_w + 1) // 2), max(16, (orig_h + 1) // 2)
    elif "4k" in sm or "3840" in sm:
        max_dim = 3840
    elif "2k" in sm or "2048" in sm:
        max_dim = 2048
    elif "1080" in sm or "1920" in sm:
        max_dim = 1920
    elif "1k" in sm or "1024" in sm:
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
        self.frame_dims: List[Tuple[int, int]] = []

        # SAM2 Inference State
        self.inference_state: Optional[Dict[str, Any]] = None
        self.prompts: Dict[int, Dict[str, Any]] = {}  # frame_idx -> {'points': [...], 'labels': [...], 'box': [...]}
        self.tracked_masks: Dict[int, np.ndarray] = {}  # frame_idx -> binary mask (H, W) uint8 or float16

        # Normalization constants (ImageNet standard)
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

        # Base cleaned mask cache for zero-latency parameter scrubbing
        self._base_mask_cache: Dict[Tuple[int, bool, int], np.ndarray] = {}

        # Decoded RGB frame cache (LRU) for instant playback and scrub
        self._frame_rgb_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._frame_cache_max_size: int = 32

        # Multi-level undo history stack: frame_idx -> list of {mask, prompt} snapshots
        self.undo_stacks: Dict[int, List[Dict[str, Any]]] = {}

        # Mask revision tracking: frame_idx -> revision counter integer
        self.mask_revisions: Dict[int, int] = {}

        # Thread safety lock for model loading & switching
        self._model_lock = threading.Lock()

        # Text-grounding open-vocabulary model (OwLV2)
        self.owl_processor = None
        self.owl_model = None

        self.log_callback: Optional[Callable[[str], None]] = None

    @property
    def model_name(self) -> Optional[str]:
        """Returns the active model key name (e.g. 'tiny', 'small', 'base_plus', 'large')."""
        return self.current_model_key

    def get_mask_revision(self, frame_idx: int) -> int:
        """Returns monotonically increasing revision counter for cache invalidation."""
        return self.mask_revisions.get(frame_idx, 0)

    def set_log_callback(self, cb: Optional[Callable[[str], None]]):
        """Sets external log sink (e.g. Abu Align main console)."""
        self.log_callback = cb

    def log(self, msg: str):
        """Logs message to external console and stdout."""
        if self.log_callback:
            t_str = time.strftime("%H:%M:%S")
            self.log_callback(f"[{t_str}] [SAM 2] {msg}\n")
        else:
            try:
                print(f"[SAM 2] {msg}")
            except UnicodeEncodeError:
                safe_msg = msg.encode("ascii", "replace").decode("ascii")
                print(f"[SAM 2] {safe_msg}")

    def get_frame_dims(self, frame_idx: int) -> Tuple[int, int]:
        """Returns (width, height) for the requested frame, respecting native orientation."""
        if 0 <= frame_idx < len(self.frame_dims):
            return self.frame_dims[frame_idx]
        if self.num_frames > 0 and 0 <= frame_idx < self.num_frames:
            try:
                rgb = self.get_frame_rgb(frame_idx)
                h, w = rgb.shape[:2]
                while len(self.frame_dims) <= frame_idx:
                    self.frame_dims.append((self.orig_width, self.orig_height))
                self.frame_dims[frame_idx] = (w, h)
                return w, h
            except Exception:
                pass
        return max(1, self.orig_width), max(1, self.orig_height)

    def _attach_dynamic_predictor_hooks(self, predictor):
        """
        Enables SAM 2 to automatically handle mixed aspect ratios and mixed orientations
        (e.g., sequences containing both horizontal landscape and vertical portrait images).
        Dynamically supplies each frame's native (W_i, H_i) resolution when generating
        output masks, eliminating horizontal stretching or aspect ratio squashing.
        """
        import types
        predictor._engine_ref = self

        def dynamic_get_orig_video_res_output(pred_self, inference_state, any_res_masks):
            frame_idx = inference_state.get("_active_frame_idx", None)
            if frame_idx is None:
                ftpo = inference_state.get("frames_tracked_per_obj", {})
                if ftpo and 0 in ftpo and len(ftpo[0]) > 0:
                    frame_idx = list(ftpo[0].keys())[-1]

            engine = getattr(pred_self, "_engine_ref", None)
            if frame_idx is not None and engine is not None:
                video_W, video_H = engine.get_frame_dims(frame_idx)
            else:
                video_H = inference_state.get("video_height", 1024)
                video_W = inference_state.get("video_width", 1024)

            device = inference_state["device"]
            any_res_masks = any_res_masks.to(device, non_blocking=True)
            if any_res_masks.shape[-2:] == (video_H, video_W):
                video_res_masks = any_res_masks
            else:
                video_res_masks = torch.nn.functional.interpolate(
                    any_res_masks,
                    size=(video_H, video_W),
                    mode="bilinear",
                    align_corners=False,
                )
            if pred_self.non_overlap_masks:
                video_res_masks = pred_self._apply_non_overlapping_constraints(video_res_masks)
            return any_res_masks, video_res_masks

        predictor._get_orig_video_res_output = types.MethodType(dynamic_get_orig_video_res_output, predictor)

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
        with self._model_lock:
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
            self._attach_dynamic_predictor_hooks(self.video_predictor)
            sam_base = build_sam2(cfg, ckpt_path, device=self.device)
            self.image_predictor = SAM2ImagePredictor(sam_base)
            self.current_model_key = model_key

            # If an inference sequence is already active, invalidate old backbone features & re-evaluate prompts
            if self.inference_state is not None:
                if progress_callback:
                    progress_callback(f"Refreshing active sequence features for {SAM2_MODELS[model_key]['name']}...")
                self.inference_state["cached_features"].clear()
                self.video_predictor.reset_state(self.inference_state)
                self._base_mask_cache.clear()

                # Re-warmup feature extractor
                active_frame = self.inference_state.get("_active_frame_idx", 0) or 0
                if active_frame < self.num_frames:
                    self.video_predictor._get_image_feature(self.inference_state, frame_idx=active_frame, batch_size=1)

                # Re-evaluate all existing manual keyframe prompts with the new model weights!
                for kf in self.get_keyframes():
                    try:
                        self._evaluate_frame_prompts(kf, obj_id=1)
                    except Exception as e:
                        self.log(f"Notice: re-evaluating prompt on frame {kf}: {e}")

            if progress_callback:
                progress_callback(f"Loaded {SAM2_MODELS[model_key]['name']} successfully.")

    # -------------------------------------------------------------------------
    # Text-Prompt Open-Vocabulary Grounding (Grounded-SAM 2)
    # -------------------------------------------------------------------------

    def load_text_grounding_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        """Lazy-loads OwLV2 open-vocabulary zero-shot detector for text-prompted segmentation."""
        if self.owl_processor is not None and self.owl_model is not None:
            return

        if progress_callback:
            progress_callback("Loading OwLV2 zero-shot text-grounding vision model...")

        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        model_id = "google/owlv2-base-patch16-ensemble"
        self.owl_processor = Owlv2Processor.from_pretrained(model_id)
        self.owl_model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.owl_model.eval()

        if progress_callback:
            progress_callback("OwLV2 text-grounding model ready.")

    def detect_box_from_text(
        self,
        frame_idx: int,
        text_prompt: str,
        threshold: float = 0.10,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Uses OwLV2 zero-shot detector to detect candidate bounding boxes matching the text description.
        Returns list of detections sorted by confidence score descending:
        [{'box': [x0, y0, x1, y1], 'score': float, 'label': str}, ...]
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")
        if not text_prompt or not text_prompt.strip():
            raise ValueError("Text prompt cannot be empty.")

        self.load_text_grounding_model(progress_callback=progress_callback)

        from PIL import Image
        rgb = self.get_frame_rgb(frame_idx)
        pil_img = Image.fromarray(rgb)
        w_orig, h_orig = pil_img.size

        clean_prompt = text_prompt.strip().rstrip(".").lower()
        # Add background / negative contrast tokens so OwLV2 can properly separate foreground target
        texts = [[clean_prompt, "background", "clutter"]]

        inputs = self.owl_processor(text=texts, images=pil_img, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.owl_model(**inputs)

        target_sizes = torch.Tensor([[h_orig, w_orig]]).to(self.device)
        results = self.owl_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            lbl_idx = label.item()
            # Target is index 0
            if lbl_idx == 0:
                b = box.tolist()
                x0 = max(0.0, min(float(w_orig - 1), b[0]))
                y0 = max(0.0, min(float(h_orig - 1), b[1]))
                x1 = max(0.0, min(float(w_orig - 1), b[2]))
                y1 = max(0.0, min(float(h_orig - 1), b[3]))
                detections.append({
                    "box": [x0, y0, x1, y1],
                    "score": float(score.item()),
                    "label": clean_prompt
                })

        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections

    def generate_mask_from_text(
        self,
        frame_idx: int,
        text_prompt: str,
        threshold: float = 0.10,
        obj_id: int = 1,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[List[float]], float]:
        """
        Grounded-SAM 2 pipeline:
        1. Predicts zero-shot bounding box from text description using OwLV2.
        2. Injects bounding box as SAM 2 prompt on frame_idx.
        3. Evaluates and returns (mask_np, best_box, confidence_score).
        """
        detections = self.detect_box_from_text(frame_idx, text_prompt, threshold=threshold, progress_callback=progress_callback)
        if not detections:
            return None, None, 0.0

        best = detections[0]
        box = best["box"]
        score = best["score"]

        # Apply bounding box to SAM 2
        mask_np = self.add_box(frame_idx, box, obj_id=obj_id, clear_points=True)
        return mask_np, box, score

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

        # Determine dimensions and prepare tensor
        if offload_to_cpu is None:
            offload_to_cpu = (self.num_frames > 80)

        device_for_tensor = torch.device("cpu") if offload_to_cpu else torch.device(self.device)
        gpu_device = torch.device(self.device) if torch.cuda.is_available() and self.device == "cuda" else None

        img_mean_gpu = torch.tensor(self.img_mean, dtype=torch.float32, device=gpu_device)[:, None, None] if gpu_device else None
        img_std_gpu = torch.tensor(self.img_std, dtype=torch.float32, device=gpu_device)[:, None, None] if gpu_device else None

        img_mean_cpu = torch.tensor(self.img_mean, dtype=torch.float32)[:, None, None]
        img_std_cpu = torch.tensor(self.img_std, dtype=torch.float32)[:, None, None]

        images_tensor = torch.zeros(self.num_frames, 3, model_image_size, model_image_size,
                                    dtype=torch.float32, device=device_for_tensor)

        self.frame_dims = []
        load_start = time.perf_counter()

        if self.source_type in ("directory", "file_list"):
            # Multi-threaded parallel file read & decode pool
            def _decode_file(idx_and_path):
                idx, p = idx_and_path
                with open(p, "rb") as f:
                    data = np.frombuffer(f.read(), np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise IOError(f"Failed to read image: {p}")
                h, w = bgr.shape[:2]
                return idx, bgr, (w, h)

            self.frame_dims = [None] * self.num_frames
            workers = min(16, os.cpu_count() or 4)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for idx, bgr, dims in executor.map(_decode_file, enumerate(self.frame_paths)):
                    self.frame_dims[idx] = dims
                    if idx == 0:
                        self.orig_width, self.orig_height = dims
                        self._frame_rgb_cache[0] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    if gpu_device is not None:
                        # Hardware-accelerated CUDA pipeline:
                        # 1. Direct GPU memory transfer
                        # 2. In-VRAM BGR -> RGB channel reordering
                        # 3. GPU bilinear resize
                        # 4. Vectorized SIMD normalization
                        t_gpu = torch.from_numpy(bgr).to(gpu_device, non_blocking=True).permute(2, 0, 1).float()
                        t_rgb = t_gpu[[2, 1, 0], :, :]
                        t_resized = torch.nn.functional.interpolate(
                            t_rgb.unsqueeze(0),
                            size=(model_image_size, model_image_size),
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0)
                        t_norm = (t_resized / 255.0 - img_mean_gpu) / img_std_gpu
                        if offload_to_cpu:
                            images_tensor[idx] = t_norm.cpu()
                        else:
                            images_tensor[idx] = t_norm
                    else:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(rgb, (model_image_size, model_image_size), interpolation=cv2.INTER_LINEAR)
                        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                        t -= img_mean_cpu
                        t /= img_std_cpu
                        images_tensor[idx] = t

                    if progress_callback and (idx % 5 == 0 or idx == self.num_frames - 1):
                        elapsed = max(0.001, time.perf_counter() - load_start)
                        fps = (idx + 1) / elapsed
                        pct = int((idx + 1) * 100 / self.num_frames)
                        hw_label = "CUDA GPU" if gpu_device is not None else "CPU Multi-Thread"
                        progress_callback(idx + 1, self.num_frames, f"Loaded {idx + 1}/{self.num_frames} frames ({pct}%, {fps:.1f} fps) [{hw_label}]")

            if gpu_device is not None:
                torch.cuda.synchronize()

        elif self.source_type == "video":
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.source_path}")
            self.video_cap = cap
            self.frame_dims = []

            for idx in range(self.num_frames):
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                h, w = bgr.shape[:2]
                self.frame_dims.append((w, h))
                if idx == 0:
                    self.orig_width, self.orig_height = w, h
                    self._frame_rgb_cache[0] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if gpu_device is not None:
                    t_gpu = torch.from_numpy(bgr).to(gpu_device, non_blocking=True).permute(2, 0, 1).float()
                    t_rgb = t_gpu[[2, 1, 0], :, :]
                    t_resized = torch.nn.functional.interpolate(
                        t_rgb.unsqueeze(0),
                        size=(model_image_size, model_image_size),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                    t_norm = (t_resized / 255.0 - img_mean_gpu) / img_std_gpu
                    if offload_to_cpu:
                        images_tensor[idx] = t_norm.cpu()
                    else:
                        images_tensor[idx] = t_norm
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (model_image_size, model_image_size), interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                    t -= img_mean_cpu
                    t /= img_std_cpu
                    images_tensor[idx] = t

                if progress_callback and (idx % 10 == 0 or idx == self.num_frames - 1):
                    elapsed = max(0.001, time.perf_counter() - load_start)
                    fps = (idx + 1) / elapsed
                    pct = int((idx + 1) * 100 / self.num_frames)
                    hw_label = "CUDA GPU" if gpu_device is not None else "CPU"
                    progress_callback(idx + 1, self.num_frames, f"Decoded video {idx + 1}/{self.num_frames} frames ({pct}%, {fps:.1f} fps) [{hw_label}]")

            if gpu_device is not None:
                torch.cuda.synchronize()

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
        total_time = max(0.001, time.perf_counter() - load_start)
        hw_desc = "CUDA GPU + Multi-Thread I/O" if gpu_device else "Multi-Thread CPU"
        self.log(f"Ingested {self.num_frames} frames in {total_time:.2f}s ({self.num_frames / total_time:.1f} fps) via {hw_desc}")
        return self.num_frames

    def get_frame_rgb(self, frame_idx: int) -> np.ndarray:
        """Returns RGB uint8 image of the requested frame in original resolution (with LRU caching)."""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames - 1}]")

        if frame_idx in self._frame_rgb_cache:
            self._frame_rgb_cache.move_to_end(frame_idx)
            return self._frame_rgb_cache[frame_idx]

        if self.source_type in ("directory", "file_list"):
            path = self.frame_paths[frame_idx]
            # Use cv2.imdecode to support non-ASCII paths on Windows
            with open(path, "rb") as f:
                bytes_data = np.frombuffer(f.read(), np.uint8)
            bgr = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError(f"Failed to read image: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        elif self.source_type == "video":
            if self.video_cap is None:
                self.video_cap = cv2.VideoCapture(self.source_path)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = self.video_cap.read()
            if not ret or bgr is None:
                raise IOError(f"Failed to read video frame {frame_idx} from {self.source_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        else:
            raise RuntimeError("Source is not loaded.")

        self._frame_rgb_cache[frame_idx] = rgb
        if len(self._frame_rgb_cache) > self._frame_cache_max_size:
            self._frame_rgb_cache.popitem(last=False)
        return rgb

    # -------------------------------------------------------------------------
    # Interactive Prompt System (Points & Bounding Boxes)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Interactive Prompt System & Precision Mask Modification Tools
    # -------------------------------------------------------------------------

    def push_undo_state(self, frame_idx: int):
        """Saves current mask and prompt state snapshot for frame undo."""
        if frame_idx not in self.undo_stacks:
            self.undo_stacks[frame_idx] = []
        cur_mask = self.tracked_masks.get(frame_idx, None)
        cur_prompt = None
        if frame_idx in self.prompts:
            import copy
            cur_prompt = copy.deepcopy(self.prompts[frame_idx])
        self.undo_stacks[frame_idx].append({
            "mask": cur_mask.copy() if cur_mask is not None else None,
            "prompt": cur_prompt
        })
        if len(self.undo_stacks[frame_idx]) > 25:
            self.undo_stacks[frame_idx].pop(0)

    def undo(self, frame_idx: int, obj_id: int = 1) -> Optional[np.ndarray]:
        """Reverts the last action (AI prompt, brush stroke, box cut) on this frame."""
        if frame_idx in self.undo_stacks and self.undo_stacks[frame_idx]:
            prev_state = self.undo_stacks[frame_idx].pop()
            if prev_state["mask"] is not None:
                self.tracked_masks[frame_idx] = prev_state["mask"]
                self._sync_mask_to_sam2(frame_idx, self.tracked_masks[frame_idx], obj_id)
            elif frame_idx in self.tracked_masks:
                del self.tracked_masks[frame_idx]

            if prev_state["prompt"] is not None:
                self.prompts[frame_idx] = prev_state["prompt"]
            elif frame_idx in self.prompts:
                del self.prompts[frame_idx]

            self._base_mask_cache = {k: v for k, v in self._base_mask_cache.items() if k[0] != frame_idx}
            self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1
            return self.tracked_masks.get(frame_idx, None)
        else:
            res = self.remove_last_prompt(frame_idx, obj_id)
            self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1
            return res

    def _sync_mask_to_sam2(self, frame_idx: int, mask_np: np.ndarray, obj_id: int = 1):
        """Feeds the edited binary mask into SAM 2's video memory predictor."""
        if self.inference_state is None or self.video_predictor is None:
            return
        try:
            w_curr, h_curr = self.get_frame_dims(frame_idx)
            self.inference_state["_active_frame_idx"] = frame_idx
            self.inference_state["video_width"] = w_curr
            self.inference_state["video_height"] = h_curr
            bool_mask = (mask_np > 128)
            with torch.inference_mode(), self._autocast_context():
                self.video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=bool_mask
                )
        except Exception as e:
            self.log(f"Notice: mask memory sync: {e}")
        finally:
            if self.inference_state is not None:
                self.inference_state["_active_frame_idx"] = None

    def add_point(self, frame_idx: int, x: float, y: float, label: int = 1, obj_id: int = 1) -> np.ndarray:
        """
        Adds a positive (label=1) or negative (label=0) click on a frame.
        Features intelligent foreground anchor point injection when erasing on tracked frames,
        multi-point negative reinforcement, and a guaranteed fallback eraser.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")

        self.push_undo_state(frame_idx)

        if frame_idx not in self.prompts:
            self.prompts[frame_idx] = {"points": [], "labels": [], "box": None}

        w_curr, h_curr = self.get_frame_dims(frame_idx)
        cx = max(0.0, min(float(w_curr - 1), float(x)))
        cy = max(0.0, min(float(h_curr - 1), float(y)))

        # If negative prompt (label=0) and no positive points currently exist on this frame:
        if label == 0:
            has_pos = any(l == 1 for l in self.prompts[frame_idx]["labels"])
            if not has_pos:
                # If a mask already exists on this frame, extract the highest-confidence foreground anchor point
                existing_mask = self.tracked_masks.get(frame_idx, None)
                if existing_mask is not None and np.any(existing_mask > 0):
                    dist_map = cv2.distanceTransform((existing_mask > 128).astype(np.uint8), cv2.DIST_L2, 5)
                    _, _, _, max_loc = cv2.minMaxLoc(dist_map)
                    anchor_x, anchor_y = float(max_loc[0]), float(max_loc[1])
                    self.prompts[frame_idx]["points"].append([anchor_x, anchor_y])
                    self.prompts[frame_idx]["labels"].append(1)

            # Add negative point with local multi-point reinforcement so SAM 2's negative attention token flips logits
            self.prompts[frame_idx]["points"].append([cx, cy])
            self.prompts[frame_idx]["labels"].append(0)
            d = 3.5
            for dx, dy in [(-d, 0.0), (d, 0.0), (0.0, -d), (0.0, d)]:
                nx = max(0.0, min(float(w_curr - 1), cx + dx))
                ny = max(0.0, min(float(h_curr - 1), cy + dy))
                self.prompts[frame_idx]["points"].append([nx, ny])
                self.prompts[frame_idx]["labels"].append(0)
        else:
            self.prompts[frame_idx]["points"].append([cx, cy])
            self.prompts[frame_idx]["labels"].append(1)

        mask_np = self._evaluate_frame_prompts(frame_idx, obj_id)

        # Fallback guarantee: If it was an erase click and the clicked point is still foreground, apply local erase directly
        if label == 0 and mask_np[int(cy), int(cx)] > 0:
            cv2.circle(mask_np, (int(cx), int(cy)), 16, 0, -1)
            self.tracked_masks[frame_idx] = mask_np
            self._sync_mask_to_sam2(frame_idx, mask_np, obj_id)

        self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1
        return mask_np

    def add_box(self, frame_idx: int, box: List[float], obj_id: int = 1, clear_points: bool = True) -> np.ndarray:
        """
        Sets an AI bounding box prompt [x0, y0, x1, y1] on the specified frame.
        Guarantees coordinates are clamped to frame dimensions, purges contradictory points,
        and clears stale prior frame conditioning so the box prompt segments cleanly.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")

        self.push_undo_state(frame_idx)

        w_curr, h_curr = self.get_frame_dims(frame_idx)
        bx0 = max(0.0, min(float(w_curr - 1), min(box[0], box[2])))
        by0 = max(0.0, min(float(h_curr - 1), min(box[1], box[3])))
        bx1 = max(0.0, min(float(w_curr - 1), max(box[0], box[2])))
        by1 = max(0.0, min(float(h_curr - 1), max(box[1], box[3])))

        if (bx1 - bx0) < 5 or (by1 - by0) < 5:
            return self.tracked_masks.get(frame_idx, np.zeros((h_curr, w_curr), dtype=np.uint8))

        if clear_points or frame_idx not in self.prompts:
            self.prompts[frame_idx] = {"points": [], "labels": [], "box": None}
            self.reset_frame_conditioning(frame_idx, obj_id)
        else:
            # Filter out points that are outside the newly drawn box
            keep_pts = []
            keep_lbls = []
            for (px, py), pl in zip(self.prompts[frame_idx]["points"], self.prompts[frame_idx]["labels"]):
                if (bx0 - 10 <= px <= bx1 + 10) and (by0 - 10 <= py <= by1 + 10):
                    keep_pts.append([px, py])
                    keep_lbls.append(pl)
            self.prompts[frame_idx]["points"] = keep_pts
            self.prompts[frame_idx]["labels"] = keep_lbls

        self.prompts[frame_idx]["box"] = [bx0, by0, bx1, by1]

        mask = self._evaluate_frame_prompts(frame_idx, obj_id)
        self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1
        return mask

    def box_cut_mask(self, frame_idx: int, box: List[float], obj_id: int = 1) -> np.ndarray:
        """
        Instantly zeroes out all mask pixels inside the bounding box rectangle [x0, y0, x1, y1].
        Synchronizes the cut directly with SAM 2 memory so subsequent video tracking respects the erasure.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")

        self.push_undo_state(frame_idx)

        w_curr, h_curr = self.get_frame_dims(frame_idx)
        bx0 = int(round(max(0.0, min(float(w_curr - 1), min(box[0], box[2])))))
        by0 = int(round(max(0.0, min(float(h_curr - 1), min(box[1], box[3])))))
        bx1 = int(round(max(0.0, min(float(w_curr - 1), max(box[0], box[2])))))
        by1 = int(round(max(0.0, min(float(h_curr - 1), max(box[1], box[3])))))

        mask = self.tracked_masks.get(frame_idx, None)
        if mask is None:
            mask = np.zeros((h_curr, w_curr), dtype=np.uint8)
        else:
            mask = mask.copy()

        # Zero out the bounding box
        mask[by0:by1 + 1, bx0:bx1 + 1] = 0
        self.tracked_masks[frame_idx] = mask
        self._base_mask_cache = {k: v for k, v in self._base_mask_cache.items() if k[0] != frame_idx}
        self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1

        # Sync modified mask to SAM 2
        self._sync_mask_to_sam2(frame_idx, mask, obj_id)
        return mask

    def brush_stroke_mask(
        self,
        frame_idx: int,
        stroke_points: List[Tuple[float, float]],
        radius: int = 15,
        mode: str = "erase",
        obj_id: int = 1,
        sync_to_sam2: bool = False
    ) -> np.ndarray:
        """
        Directly paints or erases pixels on the current frame mask using interactive brush strokes.
        mode='erase' zeroes pixels; mode='paint' sets pixels to 255.
        sync_to_sam2: when False (during mouse dragging), only updates the in-memory mask for zero-latency 120 FPS;
        set to True on mouse button release to commit to SAM 2 memory.
        """
        if self.inference_state is None:
            raise RuntimeError("No sequence loaded.")

        if not stroke_points:
            return self.tracked_masks.get(frame_idx, np.zeros(self.get_frame_dims(frame_idx)[::-1], dtype=np.uint8))

        w_curr, h_curr = self.get_frame_dims(frame_idx)
        mask = self.tracked_masks.get(frame_idx, None)
        if mask is None:
            mask = np.zeros((h_curr, w_curr), dtype=np.uint8)
            self.tracked_masks[frame_idx] = mask

        color = 0 if mode == "erase" else 255
        thickness = max(1, radius * 2)

        for pt in stroke_points:
            ix = int(round(max(0, min(w_curr - 1, pt[0]))))
            iy = int(round(max(0, min(h_curr - 1, pt[1]))))
            cv2.circle(mask, (ix, iy), radius, color, -1)

        for i in range(len(stroke_points) - 1):
            p0 = stroke_points[i]
            p1 = stroke_points[i + 1]
            ix0 = int(round(max(0, min(w_curr - 1, p0[0]))))
            iy0 = int(round(max(0, min(h_curr - 1, p0[1]))))
            ix1 = int(round(max(0, min(w_curr - 1, p1[0]))))
            iy1 = int(round(max(0, min(h_curr - 1, p1[1]))))
            cv2.line(mask, (ix0, iy0), (ix1, iy1), color, thickness=thickness)

        self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1

        if sync_to_sam2:
            self._base_mask_cache = {k: v for k, v in self._base_mask_cache.items() if k[0] != frame_idx}
            self._sync_mask_to_sam2(frame_idx, mask, obj_id)
        return mask

    def sync_frame_mask_to_sam2(self, frame_idx: int, obj_id: int = 1, async_exec: bool = True):
        """Commits the current tracked_masks[frame_idx] to SAM 2's memory predictor."""
        mask = self.tracked_masks.get(frame_idx, None)
        if mask is None:
            return
        self._base_mask_cache = {k: v for k, v in self._base_mask_cache.items() if k[0] != frame_idx}
        if async_exec:
            threading.Thread(target=self._sync_mask_to_sam2, args=(frame_idx, mask, obj_id), daemon=True).start()
        else:
            self._sync_mask_to_sam2(frame_idx, mask, obj_id)

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
        """Clears all manual prompts and resets memory conditioning on the specified frame."""
        self.push_undo_state(frame_idx)
        if frame_idx in self.prompts:
            del self.prompts[frame_idx]

        if frame_idx in self.tracked_masks:
            del self.tracked_masks[frame_idx]

        self.reset_frame_conditioning(frame_idx, obj_id)
        self._base_mask_cache = {k: v for k, v in self._base_mask_cache.items() if k[0] != frame_idx}
        self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1

    def reset_frame_conditioning(self, frame_idx: int, obj_id: int = 1):
        """
        Completely clears prior memory outputs and conditioning logits for a specific frame.
        Allows subsequent prompt clicks or box to segment the frame without stale memory interference.
        """
        if self.inference_state is None:
            return
        try:
            obj_idx = self.video_predictor._obj_id_to_idx(self.inference_state, obj_id) if self.video_predictor else 0
            if "point_inputs_per_obj" in self.inference_state and obj_idx in self.inference_state["point_inputs_per_obj"]:
                self.inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
            if "mask_inputs_per_obj" in self.inference_state and obj_idx in self.inference_state["mask_inputs_per_obj"]:
                self.inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
            if "temp_output_dict_per_obj" in self.inference_state and obj_idx in self.inference_state["temp_output_dict_per_obj"]:
                self.inference_state["temp_output_dict_per_obj"][obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
                self.inference_state["temp_output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
            if "output_dict_per_obj" in self.inference_state and obj_idx in self.inference_state["output_dict_per_obj"]:
                self.inference_state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
                self.inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)
            if "frames_tracked_per_obj" in self.inference_state and obj_idx in self.inference_state["frames_tracked_per_obj"]:
                self.inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)
        except Exception as e:
            self.log(f"Notice: reset frame conditioning: {e}")

    def clear_all(self):
        """Resets all prompts, tracking caches, undo stacks, and loaded frames."""
        self.prompts.clear()
        self.tracked_masks.clear()
        self.mask_revisions.clear()
        self._base_mask_cache.clear()
        self._frame_rgb_cache.clear()
        self.undo_stacks.clear()
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.frame_paths = []
        self.frame_names = []
        self.frame_dims.clear()
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

    def get_frame_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Returns the binary mask for the specified frame, or None if unannotated."""
        return self.tracked_masks.get(frame_idx, None)

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

        # Dynamically set current frame native dimensions in inference state
        w_curr, h_curr = self.get_frame_dims(frame_idx)
        self.inference_state["_active_frame_idx"] = frame_idx
        self.inference_state["video_width"] = w_curr
        self.inference_state["video_height"] = h_curr

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

        self.inference_state["_active_frame_idx"] = None
        # Convert logits tensor to binary mask uint8
        mask_np = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
        self.tracked_masks[frame_idx] = mask_np
        self._base_mask_cache = {k: v for k, v in self._base_mask_cache.items() if k[0] != frame_idx}
        self.mask_revisions[frame_idx] = self.mask_revisions.get(frame_idx, 0) + 1
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

        # Clear any active prompt frame override
        self.inference_state["_active_frame_idx"] = None
        self._base_mask_cache.clear()

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
                    self.mask_revisions[out_frame_idx] = self.mask_revisions.get(out_frame_idx, 0) + 1
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
                        self.mask_revisions[out_frame_idx] = self.mask_revisions.get(out_frame_idx, 0) + 1
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

        # Fast binarization to uint8 (0 or 255)
        if binary_mask.dtype == np.uint8:
            _, mask = cv2.threshold(binary_mask, 128, 255, cv2.THRESH_BINARY)
        else:
            mask = (binary_mask > 128).astype(np.uint8) * 255

        # 1. Hole Filling
        if fill_holes and np.any(mask > 0):
            h, w = mask.shape
            flood = mask.copy()
            fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, fill_mask, (0, 0), 255)
            flood_inv = cv2.bitwise_not(flood)
            mask = cv2.bitwise_or(mask, flood_inv)

        # 2. Speckle / Island Removal (High-Performance Vectorized LUT)
        if min_speckle > 0 and np.any(mask > 0):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            if num_labels > 1:
                areas = stats[:, cv2.CC_STAT_AREA]
                lut = np.zeros(num_labels, dtype=np.uint8)
                lut[areas >= min_speckle] = 255
                lut[0] = 0  # Background remains 0
                mask = lut[labels]

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

        # 5. Feathering / Edge Softening (Fast uint8 GaussianBlur)
        if feather > 0:
            k_size = feather * 2 + 1
            sigma = feather / 2.0
            soft_mask = cv2.GaussianBlur(mask, (k_size, k_size), sigma)
            if threshold > 0:
                _, mask = cv2.threshold(soft_mask, threshold, 255, cv2.THRESH_BINARY)
            else:
                mask = soft_mask

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
        """
        Returns the post-processed mask for a specific frame.
        Caches the base cleaned mask (fill_holes + min_speckle) to achieve
        sub-100ms parameter scrubbing on 60+ Megapixel datasets.
        """
        raw_mask = self.tracked_masks.get(frame_idx, None)
        if raw_mask is None:
            return None

        # Check if base cleaned mask is cached for this frame
        cache_key = (frame_idx, fill_holes, min_speckle)
        base_mask = self._base_mask_cache.get(cache_key, None)

        if base_mask is None:
            if raw_mask.dtype == np.uint8:
                _, mask = cv2.threshold(raw_mask, 128, 255, cv2.THRESH_BINARY)
            else:
                mask = (raw_mask > 128).astype(np.uint8) * 255

            if fill_holes and np.any(mask > 0):
                h, w = mask.shape
                flood = mask.copy()
                fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(flood, fill_mask, (0, 0), 255)
                flood_inv = cv2.bitwise_not(flood)
                mask = cv2.bitwise_or(mask, flood_inv)

            if min_speckle > 0 and np.any(mask > 0):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                if num_labels > 1:
                    areas = stats[:, cv2.CC_STAT_AREA]
                    lut = np.zeros(num_labels, dtype=np.uint8)
                    lut[areas >= min_speckle] = 255
                    lut[0] = 0
                    mask = lut[labels]

            base_mask = mask
            # Keep cache size bounded
            if len(self._base_mask_cache) > 8:
                self._base_mask_cache.clear()
            self._base_mask_cache[cache_key] = base_mask

        # Apply dilation, inversion, and feathering on a copy of the base mask
        mask = base_mask.copy()

        if dilation != 0 and np.any(mask > 0):
            k_size = abs(dilation) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            if dilation > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)

        if invert:
            mask = cv2.bitwise_not(mask)

        if feather > 0:
            k_size = feather * 2 + 1
            sigma = feather / 2.0
            soft_mask = cv2.GaussianBlur(mask, (k_size, k_size), sigma)
            if threshold > 0:
                _, mask = cv2.threshold(soft_mask, threshold, 255, cv2.THRESH_BINARY)
            else:
                mask = soft_mask

        return mask

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
        png_compression: int = 1,        # 1-9 for png (level 1 is fastest lossless)
        use_hardware_accel: bool = True,
        max_workers: Optional[int] = None,
        cutout_dir: Optional[str] = None,
        postprocess_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_token: Optional[Callable[[], bool]] = None,
        pause_token: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Ultra-high-performance GPU-accelerated (CUDA PyTorch) and multi-threaded batch export
        for photogrammetry sequences (RealityScan, 3DGS, Gaussian Splatting, NeRF):
          - Hardware Acceleration: Executes dilation/erosion (max_pool2d), Gaussian feathering (1D separable conv2d),
            thresholding, and bilinear downsampling directly in VRAM on the NVIDIA CUDA GPU.
          - Asynchronous I/O Pipeline: Overlaps GPU tensor operations with multi-core disk encoding
            to maximize NVMe / SSD bandwidth while maintaining bounded memory consumption (<500MB).
          - Encoders: Fast lossless PNG (compression level 1), optimized WebP (method 0), or high-speed JPEG.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image

        t_start = time.time()
        os.makedirs(output_dir, exist_ok=True)
        target_cutout_folder = cutout_dir if cutout_dir else os.path.join(output_dir, "cutouts")
        if export_cutout:
            os.makedirs(target_cutout_folder, exist_ok=True)

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
        needs_rescale = (target_w, target_h) != (self.orig_width, self.orig_height)

        # Worker threads for concurrent NVMe / disk encoding
        if max_workers is None:
            max_workers = min(16, max(4, os.cpu_count() or 4))

        # Check CUDA GPU Availability
        use_gpu = use_hardware_accel and torch.cuda.is_available()
        gpu_device = torch.device("cuda") if use_gpu else None
        gpu_name = torch.cuda.get_device_name(0) if use_gpu else "Multi-Core CPU"

        # Codec setup
        m_fmt = mask_format.lower().lstrip(".")
        if m_fmt in ("jpg", "jpeg"):
            mask_ext = ".jpg"
        elif m_fmt == "webp":
            mask_ext = ".webp"
        else:
            mask_ext = ".png"

        c_fmt = cutout_format.lower().lstrip(".")
        cutout_ext = ".webp" if c_fmt == "webp" else ".png"

        accel_tag = f"CUDA GPU ({gpu_name})" if use_gpu else "CPU Multi-Core"
        has_mixed_ratios = len(set(self.frame_dims)) > 1
        first_w, first_h = self.get_frame_dims(0)
        sample_w, sample_h = compute_scaled_size(first_w, first_h, scale_mode)
        res_info = f"Adaptive Per-Frame [{scale_mode.upper()}] (Preserves Horizontal & Vertical Ratios)" if has_mixed_ratios else f"{sample_w}x{sample_h}px ({scale_mode.upper()})"
        self.log(f"Starting [{accel_tag}] Batch Export: {total_frames} frames -> "
                 f"Res: {res_info} | "
                 f"Cutouts: {cutout_ext.upper()} (Q{quality}) | "
                 f"Masks: {mask_ext.upper()} | "
                 f"Disk Workers: {max_workers} threads")

        # Precompute GPU Gaussian kernel if feathering on CUDA
        k_h, k_w, k_size = None, None, 0
        if use_gpu and feather > 0:
            k_size = feather * 2 + 1
            sigma = feather / 2.0
            x = torch.arange(k_size, dtype=torch.float32, device=gpu_device) - (k_size - 1) / 2
            k1d = torch.exp(-0.5 * (x / sigma) ** 2)
            k1d = k1d / k1d.sum()
            k_h = k1d.view(1, 1, k_size, 1)
            k_w = k1d.view(1, 1, 1, k_size)

        exported_counts = {"binary": 0, "cutout": 0}
        total_bytes_written = 0
        completed_tasks = 0

        # Writer task function
        def _write_single_frame(m_arr: Optional[np.ndarray], c_rgba: Optional[np.ndarray], m_path: Optional[str], c_path: Optional[str]):
            bw = 0
            if m_path and m_arr is not None:
                if mask_ext == ".jpg":
                    cv2.imwrite(m_path, m_arr, [cv2.IMWRITE_JPEG_QUALITY, max(int(quality), 95)])
                elif mask_ext == ".webp":
                    cv2.imwrite(m_path, m_arr, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
                else:
                    cv2.imwrite(m_path, m_arr, [cv2.IMWRITE_PNG_COMPRESSION, max(1, min(9, int(png_compression)))])
                if os.path.exists(m_path):
                    bw += os.path.getsize(m_path)

            if c_path and c_rgba is not None:
                if cutout_ext == ".webp":
                    # Fast Pillow WebP with method=0 (2.6x speedup over standard libwebp)
                    im = Image.fromarray(c_rgba)
                    im.save(c_path, format="WEBP", quality=int(quality), method=0)
                else:
                    cv2.imwrite(c_path, cv2.cvtColor(c_rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, max(1, min(9, int(png_compression)))])
                if os.path.exists(c_path):
                    bw += os.path.getsize(c_path)

            return bw, (1 if m_path and m_arr is not None else 0), (1 if c_path and c_rgba is not None else 0)

        # Bounded asynchronous producer-consumer pipeline
        max_in_flight = max(4, max_workers * 2)
        pending_futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as writer_pool:
            for idx in range(total_frames):
                # Handle user pause
                while pause_token and pause_token():
                    if cancel_token and cancel_token():
                        break
                    time.sleep(0.08)

                if cancel_token and cancel_token():
                    self.log("Batch export cancelled by user.")
                    break

                frame_name = self.frame_names[idx]
                stem, _ = os.path.splitext(frame_name)
                cur_w, cur_h = self.get_frame_dims(idx)
                target_w, target_h = compute_scaled_size(cur_w, cur_h, scale_mode)
                needs_rescale = (target_w, target_h) != (cur_w, cur_h)

                # 1. Obtain Base Clean Mask (Hole-Filling & Speckle Removal)
                raw_mask = self.tracked_masks.get(idx, None)
                if raw_mask is None:
                    raw_mask = np.zeros((cur_h, cur_w), dtype=np.uint8)

                cache_key = (idx, fill_holes, min_speckle)
                base_mask = self._base_mask_cache.get(cache_key, None)

                if base_mask is None:
                    if raw_mask.dtype == np.uint8:
                        _, base_mask = cv2.threshold(raw_mask, 128, 255, cv2.THRESH_BINARY)
                    else:
                        base_mask = (raw_mask > 128).astype(np.uint8) * 255

                    if fill_holes and np.any(base_mask > 0):
                        h, w = base_mask.shape
                        flood = base_mask.copy()
                        fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                        cv2.floodFill(flood, fill_mask, (0, 0), 255)
                        base_mask = cv2.bitwise_or(base_mask, cv2.bitwise_not(flood))

                    if min_speckle > 0 and np.any(base_mask > 0):
                        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(base_mask)
                        if num_labels > 1:
                            areas = stats[:, cv2.CC_STAT_AREA]
                            lut = np.zeros(num_labels, dtype=np.uint8)
                            lut[areas >= min_speckle] = 255
                            lut[0] = 0
                            base_mask = lut[labels]

                    self._base_mask_cache[cache_key] = base_mask

                # Guarantee base_mask matches native frame dimensions
                if base_mask.shape[:2] != (cur_h, cur_w):
                    base_mask = cv2.resize(base_mask, (cur_w, cur_h), interpolation=cv2.INTER_NEAREST)

                # 2. Frame Image Fetch (Only when Cutout is requested)
                rgb = self.get_frame_rgb(idx) if export_cutout else None
                if rgb is not None and rgb.shape[:2] != (cur_h, cur_w):
                    cur_h, cur_w = rgb.shape[:2]
                    target_w, target_h = compute_scaled_size(cur_w, cur_h, scale_mode)
                    needs_rescale = (target_w, target_h) != (cur_w, cur_h)

                # 3. Morphology, Smoothing, Rescaling & Channel Packing
                if use_gpu:
                    # ⚡ GPU Hardware Accelerated CUDA Pipeline
                    with torch.inference_mode():
                        m_t = torch.from_numpy(base_mask).to(gpu_device, non_blocking=True).unsqueeze(0).unsqueeze(0).float()

                        # GPU Dilation / Erosion via max_pool2d
                        if dilation > 0:
                            k = dilation * 2 + 1
                            m_t = torch.nn.functional.max_pool2d(m_t, kernel_size=k, stride=1, padding=dilation)
                        elif dilation < 0:
                            k = abs(dilation) * 2 + 1
                            m_t = -torch.nn.functional.max_pool2d(-m_t, kernel_size=k, stride=1, padding=abs(dilation))

                        # GPU Inversion
                        if invert:
                            m_t = 255.0 - m_t

                        # GPU Gaussian Edge Feathering
                        if feather > 0 and k_h is not None:
                            m_t = torch.nn.functional.conv2d(m_t, k_h, padding=(k_size // 2, 0))
                            m_t = torch.nn.functional.conv2d(m_t, k_w, padding=(0, k_size // 2))
                            if threshold > 0:
                                m_t = (m_t >= float(threshold)).float() * 255.0

                        # GPU Rescaling preserving individual frame aspect ratio
                        if needs_rescale:
                            m_t = torch.nn.functional.interpolate(m_t, size=(target_h, target_w), mode="nearest")
                            if rgb is not None:
                                rgb_t = torch.from_numpy(rgb).to(gpu_device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
                                rgb_t = torch.nn.functional.interpolate(rgb_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
                            else:
                                rgb_t = None
                        else:
                            if rgb is not None:
                                rgb_t = torch.from_numpy(rgb).to(gpu_device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).float()
                            else:
                                rgb_t = None

                        mask_host = m_t.squeeze().clamp(0, 255).to(torch.uint8).cpu().numpy()

                        if rgb_t is not None:
                            rgba_t = torch.cat([rgb_t[:, 0:1], rgb_t[:, 1:2], rgb_t[:, 2:3], m_t], dim=1)
                            rgba_host = rgba_t.squeeze().permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        else:
                            rgba_host = None
                else:
                    # Multi-Core CPU Fallback Pipeline
                    mask_host = base_mask.copy()
                    if dilation != 0 and np.any(mask_host > 0):
                        k_s = abs(dilation) * 2 + 1
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_s, k_s))
                        mask_host = cv2.dilate(mask_host, kernel) if dilation > 0 else cv2.erode(mask_host, kernel)

                    if invert:
                        mask_host = cv2.bitwise_not(mask_host)

                    if feather > 0:
                        soft = cv2.GaussianBlur(mask_host, (feather * 2 + 1, feather * 2 + 1), feather / 2.0)
                        if threshold > 0:
                            _, mask_host = cv2.threshold(soft, threshold, 255, cv2.THRESH_BINARY)
                        else:
                            mask_host = soft

                    if needs_rescale:
                        mask_host = cv2.resize(mask_host, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        if rgb is not None:
                            interp = cv2.INTER_AREA if (target_w < cur_w or target_h < cur_h) else cv2.INTER_LINEAR
                            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=interp)

                    rgba_host = np.dstack([rgb, mask_host]) if rgb is not None else None

                # 4. Target file paths
                mask_path = None
                if export_binary:
                    mask_filename = f"{stem}.mask{mask_ext}" if rs_naming else f"{stem}_mask{mask_ext}"
                    mask_path = os.path.join(output_dir, mask_filename)

                out_cutout_path = None
                if export_cutout and rgba_host is not None:
                    out_cutout_path = os.path.join(target_cutout_folder, f"{stem}_cutout{cutout_ext}")

                # 5. Dispatch async write task
                fut = writer_pool.submit(_write_single_frame, mask_host, rgba_host, mask_path, out_cutout_path)
                pending_futures.append((stem, fut))

                # 6. Throttle producer queue to bound host memory
                while len(pending_futures) >= max_in_flight:
                    st, done_fut = pending_futures.pop(0)
                    try:
                        bw, m_inc, c_inc = done_fut.result()
                        total_bytes_written += bw
                        exported_counts["binary"] += m_inc
                        exported_counts["cutout"] += c_inc
                        completed_tasks += 1
                    except Exception as e:
                        self.log(f"Error writing frame {st}: {e}")

                    if progress_callback:
                        pct = int(completed_tasks * 100 / total_frames)
                        elapsed = max(0.001, time.time() - t_start)
                        cur_fps = completed_tasks / elapsed
                        progress_callback(
                            completed_tasks,
                            total_frames,
                            f"Exporting {st} ({completed_tasks}/{total_frames} - {pct}%, {cur_fps:.1f} fps) [{accel_tag}]"
                        )

            # Drain remaining queued writer tasks
            for st, done_fut in pending_futures:
                if cancel_token and cancel_token():
                    break
                try:
                    bw, m_inc, c_inc = done_fut.result()
                    total_bytes_written += bw
                    exported_counts["binary"] += m_inc
                    exported_counts["cutout"] += c_inc
                    completed_tasks += 1
                except Exception as e:
                    self.log(f"Error writing frame {st}: {e}")

                if progress_callback:
                    pct = int(completed_tasks * 100 / total_frames)
                    elapsed = max(0.001, time.time() - t_start)
                    cur_fps = completed_tasks / elapsed
                    progress_callback(
                        completed_tasks,
                        total_frames,
                        f"Finalizing {st} ({completed_tasks}/{total_frames} - {pct}%, {cur_fps:.1f} fps) [{accel_tag}]"
                    )

        is_cancelled = bool(cancel_token and cancel_token())
        dt = max(0.001, time.time() - t_start)
        fps = completed_tasks / dt if completed_tasks > 0 else 0.0
        total_mb = total_bytes_written / (1024 * 1024)
        avg_cutout_mb = (total_mb / max(1, exported_counts["cutout"])) if exported_counts["cutout"] > 0 else 0.0

        # Space savings comparison against uncompressed 70MB PNGs
        est_old_png_mb = exported_counts["cutout"] * 70.0
        saved_mb = max(0.0, est_old_png_mb - total_mb)

        if is_cancelled:
            summary_msg = f"Export Cancelled: {completed_tasks}/{total_frames} frames processed before stop."
        else:
            summary_msg = (
                f"Export Complete: {exported_counts['binary']} masks + {exported_counts['cutout']} cutouts in {dt:.2f}s "
                f"({fps:.1f} fps) via {accel_tag}. Total: {total_mb:.1f} MB (Avg cutout: {avg_cutout_mb:.2f} MB, saved ~{saved_mb:.1f} MB vs 70MB PNG)"
            )
        self.log(summary_msg)

        return {
            "cancelled": is_cancelled,
            "binary": exported_counts["binary"],
            "cutout": exported_counts["cutout"],
            "total_mb": total_mb,
            "elapsed_sec": dt,
            "fps": fps,
            "avg_cutout_mb": avg_cutout_mb,
            "saved_mb": saved_mb,
            "scale_res": "Adaptive (Preserves H/V Ratios)" if has_mixed_ratios else f"{sample_w}x{sample_h}"
        }
