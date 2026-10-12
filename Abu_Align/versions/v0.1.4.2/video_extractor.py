"""
Abu Align & Train - High-Performance Video to Frame Sequence Extractor
======================================================================
Professional video frame extractor engineered for 360° Equirectangular
and planar video sequences for photogrammetry and 3D Gaussian Splatting.

Features:
- Primary Engine: Multi-threaded FFmpeg with NVIDIA NVDEC/CUDA hardware acceleration
- Fallback Engine: OpenCV VideoCapture with multithreaded disk writer pool
- Flexible Extraction: Target FPS (native/custom), Frame Stride (1/N), Keyframe/I-frame only
- Professional Scaling: Native 8K/5.7K/4K/2K ERP scaling with 2:1 aspect ratio enforcement
- Lossless PNG & High-Quality JPEG (custom compression/quality 1-100)
- Motion Blur Filtering: Laplacian variance thresholding to discard blurred frames
- Automated Manifest: Writes extraction_manifest.json with frame timestamps & metadata
- Standalone CLI & GUI execution + Automated Pipeline Ingestion for Abu Align & Train
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# OpenCV is imported conditionally for fallback and blur detection
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Paths and Executables
# ----------------------------------------------------------------------
FFMPEG_CANDIDATES = [
    "ffmpeg",
    r"C:\Users\USer\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]

FFPROBE_CANDIDATES = [
    "ffprobe",
    r"C:\Users\USer\ffmpeg\bin\ffprobe.exe",
    r"C:\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
]

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".insv", ".avi", ".mkv", ".webm", ".m4v", ".flv"}


def get_ffmpeg_executable() -> Optional[str]:
    """Finds the ffmpeg executable on PATH or candidate paths."""
    for cand in FFMPEG_CANDIDATES:
        found = shutil.which(cand)
        if found:
            return found
        if os.path.isfile(cand):
            return cand
    return None


def get_ffprobe_executable() -> Optional[str]:
    """Finds the ffprobe executable on PATH or candidate paths."""
    for cand in FFPROBE_CANDIDATES:
        found = shutil.which(cand)
        if found:
            return found
        if os.path.isfile(cand):
            return cand
    return None


def is_video_file(path: str) -> bool:
    """Checks whether the file path corresponds to a supported video format."""
    if not os.path.isfile(path):
        return False
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTS


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extracts technical metadata (duration, total frames, FPS, resolution, codec, rotation)
    from a video file using ffprobe, falling back to OpenCV.
    Accurately handles stream rotation metadata (tags & display matrix) so upright
    display dimensions and orientation (portrait vs landscape) are correctly detected.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffprobe_exe = get_ffprobe_executable()
    info = {
        "file_path": str(Path(video_path).resolve()),
        "file_name": Path(video_path).name,
        "file_size_bytes": os.path.getsize(video_path),
        "duration_sec": 0.0,
        "total_frames": 0,
        "fps": 30.0,
        "raw_width": 0,
        "raw_height": 0,
        "rotation": 0,
        "width": 0,
        "height": 0,
        "is_vertical": False,
        "codec": "unknown",
        "aspect_ratio": "unknown",
        "is_erp_360": False,
    }

    if ffprobe_exe:
        try:
            cmd = [
                ffprobe_exe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames,codec_name,side_data_list:stream_tags=rotate:format=duration:format_tags=rotate",
                "-of", "json",
                video_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            if streams:
                st = streams[0]
                raw_w = int(st.get("width", 0))
                raw_h = int(st.get("height", 0))
                info["raw_width"] = raw_w
                info["raw_height"] = raw_h
                info["codec"] = st.get("codec_name", "unknown")

                # Frame rate parsing
                r_fps = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
                if "/" in r_fps:
                    num, den = r_fps.split("/")
                    info["fps"] = float(num) / float(den) if float(den) > 0 else 30.0
                else:
                    info["fps"] = float(r_fps)

                # Duration
                dur = st.get("duration") or data.get("format", {}).get("duration")
                if dur:
                    info["duration_sec"] = float(dur)

                # Total frames
                nb_f = st.get("nb_frames")
                if nb_f and nb_f.isdigit():
                    info["total_frames"] = int(nb_f)
                elif info["duration_sec"] > 0 and info["fps"] > 0:
                    info["total_frames"] = int(round(info["duration_sec"] * info["fps"]))

                # Detect rotation from Display Matrix or Tags
                rot = 0
                side_data = st.get("side_data_list", [])
                if isinstance(side_data, list):
                    for sd in side_data:
                        if isinstance(sd, dict) and "rotation" in sd:
                            try:
                                rot = int(round(float(sd["rotation"])))
                                break
                            except Exception:
                                pass
                if rot == 0:
                    st_tags = st.get("tags", {})
                    if isinstance(st_tags, dict) and "rotate" in st_tags:
                        try:
                            rot = int(round(float(st_tags["rotate"])))
                        except Exception:
                            pass
                if rot == 0:
                    fmt_tags = data.get("format", {}).get("tags", {})
                    if isinstance(fmt_tags, dict) and "rotate" in fmt_tags:
                        try:
                            rot = int(round(float(fmt_tags["rotate"])))
                        except Exception:
                            pass

                info["rotation"] = (rot % 360 + 360) % 360

                # Compute upright display dimensions
                if info["rotation"] in (90, 270):
                    info["width"] = raw_h
                    info["height"] = raw_w
                else:
                    info["width"] = raw_w
                    info["height"] = raw_h
        except Exception:
            pass

    # OpenCV fallback if ffprobe didn't resolve full metadata
    if (info["width"] == 0 or info["total_frames"] == 0) and HAS_CV2:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if info["raw_width"] == 0:
                    info["raw_width"] = raw_w
                if info["raw_height"] == 0:
                    info["raw_height"] = raw_h

                if info["rotation"] == 0 and hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
                    try:
                        c_rot = int(round(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
                        info["rotation"] = (c_rot % 360 + 360) % 360
                    except Exception:
                        pass

                if info["width"] == 0:
                    if info["rotation"] in (90, 270):
                        info["width"] = raw_h
                        info["height"] = raw_w
                    else:
                        info["width"] = raw_w
                        info["height"] = raw_h

                if info["fps"] <= 0 or info["fps"] == 30.0:
                    cv_fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if cv_fps > 0:
                        info["fps"] = cv_fps
                if info["total_frames"] == 0:
                    info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if info["duration_sec"] <= 0 and info["total_frames"] > 0 and info["fps"] > 0:
                    info["duration_sec"] = info["total_frames"] / info["fps"]
                cap.release()
        except Exception:
            pass

    if info["width"] > 0 and info["height"] > 0:
        info["is_vertical"] = info["height"] > info["width"]
        ratio = info["width"] / info["height"]
        info["aspect_ratio"] = f"{ratio:.2f}:1"
        # Equirectangular standard is 2:1 aspect ratio on horizontal video
        info["is_erp_360"] = abs(ratio - 2.0) <= 0.15 and not info["is_vertical"]

    return info


def calculate_laplacian_variance(image_path: str) -> float:
    """Computes the Laplacian variance of an image as a measure of sharpness / blur."""
    if not HAS_CV2 or not os.path.isfile(image_path):
        return 0.0
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Core Video Frame Extractor
# ----------------------------------------------------------------------
def extract_video_frames(
    video_path: str,
    output_dir: str,
    mode: str = "fps",            # "fps", "stride", "keyframes", "all"
    fps: float = 5.0,             # Target FPS when mode="fps"
    stride: int = 1,              # Frame stride when mode="stride"
    image_format: str = "jpg",    # "jpg" or "png"
    quality: int = 95,            # JPEG quality (1-100) or PNG compression
    resolution: Optional[str] = None, # None (native), "8k", "5.7k", "4k", "2k", "1k", "custom", or "WxH"
    custom_resolution: Optional[str] = None, # "WxH" e.g. "1080x1920"
    rotation: str = "original",   # "original", "horizontal", "cw90", "ccw90", "180"
    blur_threshold: float = 0.0,  # 0.0 = disabled, >0 = Laplacian variance threshold
    use_hwaccel: bool = True,     # Enable NVDEC / CUDA hardware decoding
    name_prefix: str = "frame",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Extracts a frame sequence from a video file into a target directory.
    Supports intelligent aspect-ratio preservation for vertical (portrait) and
    horizontal (landscape) planar video, standard 2:1 ERP scaling for 360 videos,
    arbitrary custom resolutions (WxH), and orientation transforms (keeping native
    ratio or rotating to horizontal landscape).

    Returns an extraction summary dict including total frames, elapsed time, and paths.
    """
    start_time = time.time()
    video_abs = str(Path(video_path).resolve())
    out_abs = str(Path(output_dir).resolve())
    os.makedirs(out_abs, exist_ok=True)

    info = get_video_info(video_abs)
    ffmpeg_exe = get_ffmpeg_executable()

    # Determine orientation transform
    rot_filter: Optional[str] = None
    cv_rot: Optional[int] = None
    post_w = info.get("width", 0)
    post_h = info.get("height", 0)
    is_vert = info.get("is_vertical", False)

    rot_mode = (rotation or "original").lower().strip()
    if rot_mode == "horizontal":
        # If input is portrait/vertical, rotate 90° CW to horizontal landscape
        if is_vert:
            rot_filter = "transpose=1"
            if HAS_CV2:
                cv_rot = cv2.ROTATE_90_CLOCKWISE
            post_w, post_h = post_h, post_w
    elif rot_mode == "cw90":
        rot_filter = "transpose=1"
        if HAS_CV2:
            cv_rot = cv2.ROTATE_90_CLOCKWISE
        post_w, post_h = post_h, post_w
    elif rot_mode == "ccw90":
        rot_filter = "transpose=2"
        if HAS_CV2:
            cv_rot = cv2.ROTATE_90_COUNTERCLOCKWISE
        post_w, post_h = post_h, post_w
    elif rot_mode == "180":
        rot_filter = "hflip,vflip"
        if HAS_CV2:
            cv_rot = cv2.ROTATE_180

    # Determine target resolution
    target_w, target_h = None, None
    res_input = (custom_resolution if (resolution == "custom" and custom_resolution) else (custom_resolution or resolution or ""))
    res_str = str(res_input).lower().strip()

    if res_str and res_str not in ("native", "original", "none"):
        if "x" in res_str:
            try:
                parts = res_str.split("x")
                tw = int(parts[0].strip())
                th = int(parts[1].strip())
                if tw > 0 and th > 0:
                    target_w, target_h = tw, th
            except Exception:
                target_w, target_h = None, None
        else:
            is_erp = info.get("is_erp_360", False) and rot_mode in ("original", "auto", "native")
            if is_erp:
                erp_map = {
                    "8k": (7680, 3840),
                    "5.7k": (5760, 2880),
                    "4k": (3840, 1920),
                    "2k": (1920, 960),
                    "1k": (1024, 512),
                }
                if res_str in erp_map:
                    target_w, target_h = erp_map[res_str]
            else:
                # Planar / Portrait / Landscape: scale longest edge while maintaining aspect ratio
                preset_max = {
                    "8k": 7680,
                    "5.7k": 5760,
                    "4k": 3840,
                    "2k": 1920,
                    "1k": 1024,
                }
                if res_str in preset_max:
                    max_dim = preset_max[res_str]
                    if post_h > post_w and post_h > 0:
                        # Portrait / vertical frame
                        target_h = max_dim
                        target_w = int(round(max_dim * (post_w / post_h)))
                    elif post_w > 0:
                        # Landscape / horizontal frame
                        target_w = max_dim
                        target_h = int(round(max_dim * (post_h / post_w))) if post_w > 0 else post_h

    # Ensure even dimensions (required for lanczos filter and video decoders)
    if target_w and target_h:
        target_w = max(2, target_w - (target_w % 2))
        target_h = max(2, target_h - (target_h % 2))

    fmt = image_format.lower().replace(".", "")
    if fmt not in ("jpg", "jpeg", "png"):
        fmt = "jpg"
    ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"

    def emit_progress(current: int, total: int, status: str = "extracting"):
        if progress_callback:
            pct = (current / total * 100.0) if total > 0 else 0.0
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0.0
            progress_callback({
                "status": status,
                "current_frame": current,
                "total_frames": total,
                "percent": min(pct, 100.0),
                "elapsed_sec": elapsed,
                "fps_rate": rate,
                "output_dir": out_abs,
            })

    emit_progress(0, info.get("total_frames", 100), "starting")

    extracted_frames: List[str] = []

    # Execution Pathway 1: Multi-Threaded FFmpeg
    if ffmpeg_exe:
        success = _extract_with_ffmpeg(
            ffmpeg_exe=ffmpeg_exe,
            video_path=video_abs,
            output_dir=out_abs,
            mode=mode,
            fps=fps,
            stride=stride,
            fmt=fmt,
            quality=quality,
            target_w=target_w,
            target_h=target_h,
            rot_filter=rot_filter,
            use_hwaccel=use_hwaccel,
            name_prefix=name_prefix,
            info=info,
            emit_progress=emit_progress,
        )
        if success:
            extracted_frames = sorted([
                f for f in os.listdir(out_abs)
                if f.lower().endswith(ext) and f.startswith(name_prefix)
            ])

    # Execution Pathway 2: OpenCV Fallback (if FFmpeg failed or was unavailable)
    if not extracted_frames and HAS_CV2:
        emit_progress(0, info.get("total_frames", 100), "fallback_opencv")
        extracted_frames = _extract_with_opencv(
            video_path=video_abs,
            output_dir=out_abs,
            mode=mode,
            fps=fps,
            stride=stride,
            fmt=fmt,
            quality=quality,
            target_w=target_w,
            target_h=target_h,
            cv_rot=cv_rot,
            name_prefix=name_prefix,
            info=info,
            emit_progress=emit_progress,
        )

    # Optional Post-Processing: Motion Blur Filtering
    discarded_blur_count = 0
    if blur_threshold > 0.0 and HAS_CV2 and extracted_frames:
        emit_progress(len(extracted_frames), len(extracted_frames), "filtering_blur")
        surviving_frames = []
        for f in extracted_frames:
            f_path = os.path.join(out_abs, f)
            var = calculate_laplacian_variance(f_path)
            if var < blur_threshold:
                try:
                    os.remove(f_path)
                    discarded_blur_count += 1
                except Exception:
                    surviving_frames.append(f)
            else:
                surviving_frames.append(f)
        extracted_frames = surviving_frames

    total_time = time.time() - start_time
    emit_progress(len(extracted_frames), len(extracted_frames), "complete")

    # Generate extraction manifest
    manifest = {
        "version": "1.1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_video": info,
        "extraction_config": {
            "mode": mode,
            "target_fps": fps if mode == "fps" else None,
            "stride": stride if mode == "stride" else None,
            "image_format": fmt,
            "quality": quality,
            "rotation": rot_mode,
            "resolution": f"{target_w}x{target_h}" if target_w and target_h else "native",
            "effective_dimensions": f"{post_w}x{post_h}",
            "blur_threshold": blur_threshold,
            "discarded_blurred_frames": discarded_blur_count,
            "use_hwaccel": use_hwaccel,
        },
        "total_extracted_frames": len(extracted_frames),
        "elapsed_seconds": round(total_time, 2),
        "frame_files": extracted_frames[:50] + (["..."] if len(extracted_frames) > 50 else []),
    }

    manifest_path = os.path.join(out_abs, "extraction_manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
    except Exception:
        pass

    return {
        "success": len(extracted_frames) > 0,
        "total_frames": len(extracted_frames),
        "output_dir": out_abs,
        "manifest_path": manifest_path,
        "elapsed_seconds": round(total_time, 2),
        "fps_rate": round(len(extracted_frames) / total_time, 1) if total_time > 0 else 0.0,
        "discarded_blurred_frames": discarded_blur_count,
    }


def _extract_with_ffmpeg(
    ffmpeg_exe: str,
    video_path: str,
    output_dir: str,
    mode: str,
    fps: float,
    stride: int,
    fmt: str,
    quality: int,
    target_w: Optional[int],
    target_h: Optional[int],
    rot_filter: Optional[str],
    use_hwaccel: bool,
    name_prefix: str,
    info: Dict[str, Any],
    emit_progress: Callable[[int, int, str], None],
) -> bool:
    """Executes FFmpeg with hardware acceleration, orientation handling, and live progress tracking."""
    # Build filter chain: sampling -> orientation transform -> lanczos scaling
    filters = []
    if mode == "fps" and fps > 0:
        filters.append(f"fps={fps}")
    elif mode == "stride" and stride > 1:
        filters.append(f"select='not(mod(n\\,{stride}))'")

    if rot_filter:
        filters.append(rot_filter)

    if target_w and target_h:
        filters.append(f"scale={target_w}:{target_h}:flags=lanczos")

    filter_str = ",".join(filters)

    ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"
    out_pattern = os.path.join(output_dir, f"{name_prefix}_%05d{ext}")

    def build_cmd(hw: bool) -> List[str]:
        cmd = [ffmpeg_exe, "-y"]
        if hw:
            cmd.extend(["-hwaccel", "cuda"])
        if mode == "keyframes":
            cmd.extend(["-skip_frame", "nokey"])
        cmd.extend(["-i", video_path])
        if filter_str:
            cmd.extend(["-vf", filter_str])
        if mode == "keyframes" or (mode == "stride" and stride > 1):
            cmd.extend(["-vsync", "vfr"])

        # Format and quality controls
        if fmt in ("jpg", "jpeg"):
            # Map quality (1-100) to FFmpeg q:v (31 to 1)
            q_val = max(1, min(31, int(round((100 - quality) / 100.0 * 30 + 1))))
            cmd.extend(["-q:v", str(q_val)])
        else:
            cmd.extend(["-c:v", "png", "-compression_level", "3"])

        cmd.extend(["-stats", out_pattern])
        return cmd

    # Attempt with hardware acceleration first (if requested)
    cmd = build_cmd(hw=use_hwaccel)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        return False

    # Estimate expected frames for progress tracking
    total_native_frames = info.get("total_frames", 0)
    dur = info.get("duration_sec", 0.0)
    if mode == "fps" and fps > 0 and dur > 0:
        est_total = int(round(dur * fps))
    elif mode == "stride" and stride > 1 and total_native_frames > 0:
        est_total = total_native_frames // stride
    else:
        est_total = total_native_frames if total_native_frames > 0 else 100

    hw_failed = False
    for line in proc.stderr:
        line_s = line.strip()
        if not line_s:
            continue
        if "device creation failed" in line_s.lower() or "cannot create hwaccel" in line_s.lower():
            hw_failed = True
        if "frame=" in line_s:
            parts = line_s.split()
            for i, p in enumerate(parts):
                if p.startswith("frame=") or p == "frame=":
                    val = p.split("=")[-1] if "=" in p else (parts[i+1] if i+1 < len(parts) else "")
                    if val.isdigit():
                        emit_progress(int(val), est_total, "extracting")
                        break

    proc.wait()

    # If CUDA hwaccel failed, cleanly retry using CPU decoding
    if (proc.returncode != 0 or hw_failed) and use_hwaccel:
        cmd_cpu = build_cmd(hw=False)
        try:
            proc_cpu = subprocess.Popen(
                cmd_cpu,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace"
            )
            for line in proc_cpu.stderr:
                if "frame=" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.startswith("frame=") or p == "frame=":
                            val = p.split("=")[-1] if "=" in p else (parts[i+1] if i+1 < len(parts) else "")
                            if val.isdigit():
                                emit_progress(int(val), est_total, "extracting_cpu")
                                break
            proc_cpu.wait()
            return proc_cpu.returncode == 0
        except Exception:
            return False

    return proc.returncode == 0


def _extract_with_opencv(
    video_path: str,
    output_dir: str,
    mode: str,
    fps: float,
    stride: int,
    fmt: str,
    quality: int,
    target_w: Optional[int],
    target_h: Optional[int],
    cv_rot: Optional[int],
    name_prefix: str,
    info: Dict[str, Any],
    emit_progress: Callable[[int, int, str], None],
) -> List[str]:
    """Pure-Python OpenCV fallback with ThreadPoolExecutor disk writing and orientation support."""
    if not HAS_CV2:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = 1
    if mode == "stride" and stride > 1:
        step = stride
    elif mode == "fps" and fps > 0 and native_fps > 0:
        step = max(1, int(round(native_fps / fps)))

    ext = ".jpg" if fmt in ("jpg", "jpeg") else ".png"
    encode_params = []
    if fmt in ("jpg", "jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, quality))]
    else:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    extracted_files = []
    frame_idx = 0
    saved_idx = 0

    pool = ThreadPoolExecutor(max_workers=4)
    meta_rot = info.get("rotation", 0)
    raw_w = info.get("raw_width", 0)
    raw_h = info.get("raw_height", 0)

    def write_task(fpath: str, frame_data: Any):
        # Normalize upright if OpenCV didn't auto-rotate stream metadata
        fh, fw = frame_data.shape[:2]
        if meta_rot in (90, 270) and fw == raw_w and fh == raw_h:
            if meta_rot == 90:
                frame_data = cv2.rotate(frame_data, cv2.ROTATE_90_CLOCKWISE)
            elif meta_rot == 270:
                frame_data = cv2.rotate(frame_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif meta_rot == 180 and fw == raw_w and fh == raw_h:
            frame_data = cv2.rotate(frame_data, cv2.ROTATE_180)

        # Apply user requested rotation transform
        if cv_rot is not None:
            frame_data = cv2.rotate(frame_data, cv_rot)

        # Apply scaling
        if target_w and target_h:
            frame_data = cv2.resize(frame_data, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(fpath, frame_data, encode_params)

    futures = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            saved_idx += 1
            filename = f"{name_prefix}_{saved_idx:05d}{ext}"
            filepath = os.path.join(output_dir, filename)
            extracted_files.append(filename)
            futures.append(pool.submit(write_task, filepath, frame.copy()))
            if saved_idx % 10 == 0:
                emit_progress(frame_idx, total_frames, "extracting_cv2")

        frame_idx += 1

    cap.release()
    for fut in futures:
        fut.result()
    pool.shutdown(wait=True)

    return extracted_files


# ----------------------------------------------------------------------
# Automated Dataset Intake Bridge
# ----------------------------------------------------------------------
def prepare_dataset_from_video_or_images(
    input_path: str,
    target_dataset_dir: str,
    target_fps: float = 5.0,
    stride: int = 1,
    mode: str = "fps",
    resolution: Optional[str] = None,
    custom_resolution: Optional[str] = None,
    rotation: str = "original",
    image_format: str = "jpg",
    quality: int = 95,
    blur_threshold: float = 0.0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, bool]:
    """
    Intelligent intake handler:
    1. If input_path is a directory of images: returns (input_path, False) immediately.
    2. If input_path is a video file: extracts frames into <target_dataset_dir>/images,
       generates manifest, and returns (target_dataset_dir, True).
    """
    if os.path.isdir(input_path):
        # Already an image folder
        return input_path, False

    if not is_video_file(input_path):
        raise ValueError(f"Input is neither an existing directory nor a supported video format: {input_path}")

    images_dir = os.path.join(target_dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Check if frames already exist to avoid re-extracting unnecessarily
    existing_frames = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    manifest_file = os.path.join(images_dir, "extraction_manifest.json")
    if existing_frames and os.path.isfile(manifest_file):
        return target_dataset_dir, False

    # Extract frames
    res = extract_video_frames(
        video_path=input_path,
        output_dir=images_dir,
        mode=mode,
        fps=target_fps,
        stride=stride,
        image_format=image_format,
        quality=quality,
        resolution=resolution,
        custom_resolution=custom_resolution,
        rotation=rotation,
        blur_threshold=blur_threshold,
        use_hwaccel=True,
        progress_callback=progress_callback,
    )
    if not res.get("success", False):
        raise RuntimeError(f"Video frame extraction failed for {input_path}")

    return target_dataset_dir, True


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Abu Align & Train - High-Performance Video to Frame Sequence Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file path or directory of videos")
    parser.add_argument("-o", "--output", help="Output directory for extracted frames (defaults to <input>_frames)")
    parser.add_argument("-m", "--mode", choices=["fps", "stride", "keyframes", "all"], default="fps", help="Frame sampling mode")
    parser.add_argument("--fps", type=float, default=5.0, help="Target frames per second when mode=fps")
    parser.add_argument("--stride", type=int, default=1, help="Extract every N-th frame when mode=stride")
    parser.add_argument("-f", "--format", choices=["jpg", "png"], default="jpg", help="Output image format")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("-r", "--resolution", default="native", help="Target resolution: 'native', '4k', '2k', '1k', '8k', '5.7k', 'custom', or 'WIDTHxHEIGHT' (e.g. 1080x1920)")
    parser.add_argument("--custom-res", default=None, help="Custom resolution in 'WIDTHxHEIGHT' format (e.g. 1080x1920)")
    parser.add_argument("--rotation", choices=["original", "horizontal", "cw90", "ccw90", "180"], default="original", help="Frame orientation: 'original' (follow input ratio), 'horizontal' (rotate portrait to landscape), 'cw90', 'ccw90', or '180'")
    parser.add_argument("--blur-threshold", type=float, default=0.0, help="Laplacian variance threshold to discard blurred frames (0 = disabled)")
    parser.add_argument("--no-hwaccel", action="store_true", help="Disable NVIDIA NVDEC/CUDA hardware acceleration")
    parser.add_argument("--prefix", default="frame", help="Filename prefix for extracted frames")
    return parser


def main():
    parser = build_cli_parser()
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    # Batch directory mode
    if os.path.isdir(input_path):
        video_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if is_video_file(os.path.join(input_path, f))
        ]
        if not video_files:
            print(f"[ERROR] No supported video files found in directory: {input_path}")
            sys.exit(1)

        print(f"[*] Found {len(video_files)} video files for batch extraction.")
        out_base = args.output or os.path.join(input_path, "extracted_sequences")
        for idx, vid in enumerate(video_files, start=1):
            vid_stem = Path(vid).stem
            vid_out = os.path.join(out_base, vid_stem)
            print(f"\n[{idx}/{len(video_files)}] Extracting: {Path(vid).name} -> {vid_out}")
            res = extract_video_frames(
                video_path=vid,
                output_dir=vid_out,
                mode=args.mode,
                fps=args.fps,
                stride=args.stride,
                image_format=args.format,
                quality=args.quality,
                resolution=args.resolution,
                custom_resolution=args.custom_res,
                rotation=args.rotation,
                blur_threshold=args.blur_threshold,
                use_hwaccel=not args.no_hwaccel,
                name_prefix=args.prefix,
                progress_callback=lambda p: print(f"    Frame {p['current_frame']} ({p['percent']:.1f}%)", end="\r"),
            )
            print(f"\n    Extracted {res['total_frames']} frames in {res['elapsed_seconds']}s ({res['fps_rate']} FPS)")
        sys.exit(0)

    # Single video file mode
    out_dir = args.output or os.path.join(os.path.dirname(input_path), f"{Path(input_path).stem}_frames")
    print(f"[*] Extracting frames from: {Path(input_path).name}")
    print(f"    Output directory:      {out_dir}")
    print(f"    Sampling mode:         {args.mode.upper()} ({args.fps} FPS / stride {args.stride})")
    print(f"    Format & Quality:      {args.format.upper()} (quality {args.quality})")
    print(f"    Resolution:            {args.resolution.upper() if args.resolution else 'NATIVE'}{f' ({args.custom_res})' if args.custom_res else ''}")
    print(f"    Orientation/Rotation:  {args.rotation.upper()}")
    print(f"    Hardware Acceleration: {'Enabled (CUDA)' if not args.no_hwaccel else 'Disabled'}")

    res = extract_video_frames(
        video_path=input_path,
        output_dir=out_dir,
        mode=args.mode,
        fps=args.fps,
        stride=args.stride,
        image_format=args.format,
        quality=args.quality,
        resolution=args.resolution,
        custom_resolution=args.custom_res,
        rotation=args.rotation,
        blur_threshold=args.blur_threshold,
        use_hwaccel=not args.no_hwaccel,
        name_prefix=args.prefix,
        progress_callback=lambda p: print(f"    Frame {p['current_frame']}/{p['total_frames']} ({p['percent']:.1f}%) - {p['fps_rate']:.1f} FPS", end="\r"),
    )
    print(f"\n[*] Extraction complete!")
    print(f"    Total Frames: {res['total_frames']}")
    print(f"    Time Elapsed: {res['elapsed_seconds']}s ({res['fps_rate']} FPS)")
    print(f"    Manifest:     {res['manifest_path']}")


if __name__ == "__main__":
    main()
