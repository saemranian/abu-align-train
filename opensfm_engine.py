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
                    "projection_type": "equirectangular"
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
                break

        return True
