"""
Abu Align & Train - Mapillary OpenSfM Engine
Native 360° Spherical (Equirectangular) & Street-Scale Structure-from-Motion
Integrates OpenSfM S^2 bearing vector bundle adjustment, multi-camera rigs, GPS trajectories,
and automated 3DGS dataset handoff (sparse/0/, transforms.json, points.ply).
"""

import os
import sys
import json
import yaml
import time
import shutil
import subprocess
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

class OpenSfMEngine:
    """
    Manages OpenSfM configuration, execution, telemetry, and COLMAP export.
    """
    def __init__(
        self,
        dataset_path: str,
        output_dir: Optional[str] = None,
        projection_type: str = "equirectangular",
        feature_type: str = "SIFT",
        matcher_type: str = "WORDS",
        bundle_interval: int = 1,
        optimize_camera_parameters: bool = True,
        use_gpu: bool = True
    ):
        self.dataset_path = os.path.abspath(dataset_path)
        self.output_dir = os.path.abspath(output_dir or os.path.join(self.dataset_path, "opensfm_workspace"))
        self.projection_type = projection_type.lower()
        self.feature_type = feature_type.upper()
        self.matcher_type = matcher_type.upper()
        self.bundle_interval = bundle_interval
        self.optimize_camera_parameters = optimize_camera_parameters
        self.use_gpu = use_gpu

        self.opensfm_bin = self._locate_opensfm_executable()

    def _locate_opensfm_executable(self) -> Optional[str]:
        """Finds OpenSfM CLI binary in PATH, Conda, or standard locations."""
        # 1. Direct command in PATH
        opensfm_path = shutil.which("opensfm")
        if opensfm_path:
            return opensfm_path

        # 2. Check conda environment Scripts/bin
        env_dir = os.path.dirname(sys.executable)
        candidates = [
            os.path.join(env_dir, "opensfm"),
            os.path.join(env_dir, "opensfm.exe"),
            os.path.join(env_dir, "bin", "opensfm"),
            os.path.join(env_dir, "Scripts", "opensfm.exe"),
            os.path.join(os.path.expanduser("~"), "OpenSfM", "bin", "opensfm"),
            "/usr/local/bin/opensfm",
            "/opt/OpenSfM/bin/opensfm"
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

        return None

    def setup_workspace(self, image_paths: List[str]):
        """Sets up the OpenSfM workspace layout, config.yaml, and overrides."""
        os.makedirs(self.output_dir, exist_ok=True)
        img_dir = os.path.join(self.output_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        # Link or copy images into workspace/images/
        for p in image_paths:
            fname = os.path.basename(p)
            dest = os.path.join(img_dir, fname)
            if not os.path.exists(dest):
                try:
                    os.link(p, dest)
                except Exception:
                    try:
                        os.symlink(p, dest)
                    except Exception:
                        shutil.copy2(p, dest)

        # Generate config.yaml
        config_data = {
            "use_exif_size": True,
            "feature_type": self.feature_type,
            "feature_min_frames": 3000,
            "feature_process_size": 2048,
            "matching_gps_distance": 150,
            "matching_gps_neighbors": 8,
            "matcher_type": self.matcher_type,
            "bundle_interval": self.bundle_interval,
            "optimize_camera_parameters": self.optimize_camera_parameters,
            "processes": max(1, os.cpu_count() - 2) if os.cpu_count() else 4
        }
        with open(os.path.join(self.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Camera overrides for 360 spherical projection
        if self.projection_type in ["equirectangular", "spherical"]:
            overrides = {
                "all": {
                    "projection_type": "spherical"
                }
            }
            with open(os.path.join(self.output_dir, "camera_models_overrides.json"), "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)

    def run_pipeline(self, log_callback=None) -> bool:
        """
        Executes OpenSfM sequential actions:
        extract_metadata -> detect_features -> match_features -> create_tracks -> reconstruct -> export_colmap
        """
        def log(msg: str):
            if log_callback:
                log_callback(msg)
            else:
                print(msg, flush=True)

        log("=" * 70)
        log("   Mapillary OpenSfM - Neural & Spherical Photogrammetry Engine")
        log(f"   * Projection: {self.projection_type.upper()}")
        log(f"   * Features:   {self.feature_type}")
        log(f"   * Workspace:  {self.output_dir}")
        log("=" * 70)

        if not self.opensfm_bin:
            log("[ERROR] OpenSfM binary ('opensfm') not found on system PATH.")
            log("[HINT] On Linux/RunPod: run install_abu_align.sh or 'pip install opensfm'.")
            log("[HINT] On Windows: OpenSfM can run via WSL2/Docker, or use Abu Align's Auto-Cubemap Transcoder.")
            return False

        stages = [
            ("extract_metadata", "Reading EXIF, GPS & Spherical GPano Tags"),
            ("detect_features",  f"Extracting {self.feature_type} Keypoints on Spherical S² Manifold"),
            ("match_features",   "Pairwise Keypoint Matching with GPS/Bearing Priors"),
            ("create_tracks",    "Synthesizing Multi-View 3D Feature Tracks"),
            ("reconstruct",      "Incremental SfM with Ceres Spherical Bundle Adjustment"),
            ("export_colmap",    "Exporting to Standard COLMAP (sparse/0/) format")
        ]

        for idx, (cmd_name, desc) in enumerate(stages):
            log(f"\n[OpenSfM Stage {idx+1}/{len(stages)}] {desc}...")
            print(f"[TELEMETRY:STAGE3] batch={idx+1} total={len(stages)} pct={(idx+1)/len(stages)*100:.1f}", flush=True)

            cmd = [self.opensfm_bin, cmd_name, self.output_dir]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            for line in proc.stdout:
                line_str = line.strip()
                if line_str:
                    log(f"   [OpenSfM] {line_str}")

            proc.wait()
            if proc.returncode != 0:
                log(f"[ERROR] OpenSfM stage '{cmd_name}' failed with return code {proc.returncode}.")
                return False

        log("\n[SUCCESS] OpenSfM 360° Reconstruction & COLMAP Export Completed Successfully.")
        return True

    def export_to_3dgs_dataset(self, target_3dgs_dir: str) -> bool:
        """
        Transfers OpenSfM exported COLMAP files into Abu Align's universal 3dgs_dataset/ directory:
        - 3dgs_dataset/sparse/0/cameras.bin (or .txt)
        - 3dgs_dataset/sparse/0/images.bin (or .txt)
        - 3dgs_dataset/sparse/0/points3D.bin (or .txt)
        - 3dgs_dataset/images/
        - 3dgs_dataset/points.ply
        """
        colmap_source = os.path.join(self.output_dir, "colmap")
        if not os.path.isdir(colmap_source):
            # Some versions export directly to undistorted/ or output_dir
            for alt in [os.path.join(self.output_dir, "undistorted"), os.path.join(self.output_dir, "sparse")]:
                if os.path.isdir(alt):
                    colmap_source = alt
                    break

        if not os.path.isdir(colmap_source):
            return False

        sparse_target = os.path.join(target_3dgs_dir, "sparse", "0")
        os.makedirs(sparse_target, exist_ok=True)

        # Copy COLMAP model files
        for fname in os.listdir(colmap_source):
            if fname.endswith((".bin", ".txt")):
                shutil.copy2(os.path.join(colmap_source, fname), os.path.join(sparse_target, fname))

        # Copy or link images
        img_src = os.path.join(self.output_dir, "images")
        img_dst = os.path.join(target_3dgs_dir, "images")
        if not os.path.exists(img_dst):
            try:
                os.symlink(img_src, img_dst)
            except Exception:
                shutil.copytree(img_src, img_dst)

        # Copy reconstruction point cloud ply if available
        ply_candidates = [
            os.path.join(self.output_dir, "reconstruction.ply"),
            os.path.join(self.output_dir, "depthmaps", "merged.ply"),
            os.path.join(colmap_source, "points3D.ply")
        ]
        for ply_file in ply_candidates:
            if os.path.isfile(ply_file):
                shutil.copy2(ply_file, os.path.join(target_3dgs_dir, "points.ply"))
                shutil.copy2(ply_file, os.path.join(target_3dgs_dir, "points3D.ply"))
                break

        # Also preserve OpenSfM reconstruction.json and metadata.json for direct PFGS360 training
        rec_json = os.path.join(self.output_dir, "reconstruction.json")
        if os.path.isfile(rec_json):
            osfm_target = os.path.join(target_3dgs_dir, "opensfm")
            os.makedirs(osfm_target, exist_ok=True)
            shutil.copy2(rec_json, os.path.join(osfm_target, "reconstruction.json"))
            ensure_opensfm_metadata(target_3dgs_dir)

            # Condition 2 Bridge: Generate transforms.json with EQUIRECTANGULAR camera model
            tf_path = os.path.join(target_3dgs_dir, "transforms.json")
            pts_path = os.path.join(target_3dgs_dir, "points3D.ply")
            opensfm_to_transforms_json(
                reconstruction_json_path=rec_json,
                output_transforms_file=tf_path,
                ply_source=os.path.join(target_3dgs_dir, "points.ply") if os.path.isfile(os.path.join(target_3dgs_dir, "points.ply")) else None,
                output_ply_dest=pts_path
            )

        return True


def opensfm_to_transforms_json(
    reconstruction_json_path: str,
    output_transforms_file: str,
    ply_source: Optional[str] = None,
    output_ply_dest: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Converts OpenSfM reconstruction.json into Nerfstudio-compliant transforms.json
    with "camera_model": "EQUIRECTANGULAR" according to Condition 2 of 360° Data Intake specification.
    Copies reconstruction.ply to points3D.ply.
    """
    if not os.path.isfile(reconstruction_json_path):
        return False, f"Reconstruction file not found: {reconstruction_json_path}"

    try:
        with open(reconstruction_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse reconstruction.json: {e}"

    if not isinstance(data, list) or len(data) == 0:
        return False, "Invalid reconstruction.json format (expected non-empty list)."

    rec = data[0]
    shots = rec.get("shots", {})
    if not shots:
        return False, "No shots found in reconstruction.json."

    # Determine resolution from cameras
    cameras = rec.get("cameras", {})
    width, height = 1024, 512
    for cam_id, cam in cameras.items():
        width = cam.get("width", width)
        height = cam.get("height", height)
        break

    frames = []
    for shot_id, shot_data in sorted(shots.items()):
        r_vec = np.array(shot_data.get("rotation", [0, 0, 0]), dtype=np.float64)
        t_vec = np.array(shot_data.get("translation", [0, 0, 0]), dtype=np.float64)

        # OpenSfM convention: rotation is Rodrigues axis-angle vector, translation is in camera coords
        R_w2c, _ = cv2.Rodrigues(r_vec)
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_vec

        # Convert OpenCV camera frame (+X right, +Y down, +Z forward)
        # to OpenGL/Nerfstudio camera frame (+X right, +Y up, +Z back)
        R_c2w_ns = R_c2w @ np.diag([1.0, -1.0, -1.0])

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_c2w_ns
        c2w[:3, 3] = t_c2w

        frames.append({
            "file_path": f"images/{shot_id}",
            "transform_matrix": c2w.tolist()
        })

    transforms_payload = {
        "camera_model": "EQUIRECTANGULAR",
        "w": int(width),
        "h": int(height),
        "frames": frames
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_transforms_file)), exist_ok=True)
    with open(output_transforms_file, "w", encoding="utf-8") as tf:
        json.dump(transforms_payload, tf, indent=2)

    if ply_source and os.path.isfile(ply_source) and output_ply_dest:
        try:
            shutil.copy2(ply_source, output_ply_dest)
        except Exception:
            pass

    return True, f"Generated transforms.json (EQUIRECTANGULAR, {len(frames)} frames) at {output_transforms_file}"


def ensure_opensfm_metadata(dataset_dir: str) -> bool:
    """
    Ensures <dataset_dir>/metadata.json exists with valid train and test splits,
    which is required by PFGS360's OpenSfMDataParser for spherical 3DGS training.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    meta_path = os.path.join(dataset_dir, "metadata.json")
    if os.path.isfile(meta_path):
        return True

    rec_cands = [
        os.path.join(dataset_dir, "opensfm", "reconstruction.json"),
        os.path.join(dataset_dir, "reconstruction.json")
    ]
    valid_names = []
    for rec_path in rec_cands:
        if os.path.isfile(rec_path):
            try:
                with open(rec_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    shots = data[0].get("shots", {})
                    valid_names = sorted(list(shots.keys()))
                    if valid_names:
                        break
            except Exception:
                pass

    if not valid_names:
        img_dir = os.path.join(dataset_dir, "images")
        if os.path.isdir(img_dir):
            valid_names = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))
            ])

    if not valid_names:
        return False

    test_step = max(1, len(valid_names) // 10) if len(valid_names) >= 5 else 1
    metadata_payload = {
        "train": valid_names,
        "test": valid_names[::test_step]
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=4)

    return True
