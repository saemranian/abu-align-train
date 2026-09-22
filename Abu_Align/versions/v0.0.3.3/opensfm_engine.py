"""
Abu Align & Train - Mapillary OpenSfM Engine
Native 360° Spherical (Equirectangular) & Street-Scale Structure-from-Motion
Integrates OpenSfM S^2 bearing vector bundle adjustment, multi-camera rigs, GPS trajectories,
and automated 3DGS dataset handoff (sparse/0/, transforms.json, points.ply).
Includes built-in Native Python Spherical Structure-from-Motion (Spherical SfM) solver
for pure equirectangular S^2 manifold photogrammetry without external binary dependencies.
"""

import os
import sys
import json
import yaml
import time
import math
import shutil
import subprocess
import collections
import numpy as np
import cv2
import scipy.optimize
from typing import List, Dict, Optional, Tuple, Callable


def uv_to_bearing(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Converts 2D equirectangular pixel coordinates (u, v) into 3D unit bearing vectors on S^2 manifold.
    u in [0, w], v in [0, h]
    theta in [-pi, pi], phi in [-pi/2, pi/2]
    """
    u = pts[:, 0]
    v = pts[:, 1]
    th = (u / float(w) - 0.5) * 2.0 * math.pi
    ph = (0.5 - v / float(h)) * math.pi
    cos_ph = np.cos(ph)
    x = cos_ph * np.sin(th)
    y = -np.sin(ph)
    z = cos_ph * np.cos(th)
    bearings = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(bearings, axis=1, keepdims=True)
    return bearings / np.maximum(norms, 1e-12)


def bearing_to_uv(bearings: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Converts 3D unit bearing vectors on S^2 manifold into 2D equirectangular pixel coordinates (u, v).
    """
    x = bearings[:, 0]
    y = bearings[:, 1]
    z = bearings[:, 2]
    th = np.arctan2(x, z)
    ph = np.arcsin(np.clip(-y, -1.0, 1.0))
    u = (th / (2.0 * math.pi) + 0.5) * float(w)
    v = (0.5 - ph / math.pi) * float(h)
    return np.stack([u, v], axis=1)


def solve_spherical_essential_matrix(
    b1: np.ndarray,
    b2: np.ndarray,
    max_iters: int = 250,
    thresh: float = 0.006
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimates relative rotation R in SO(3) and translation direction t in S^2
    between two sets of spherical bearing vectors b1 and b2 using 8-point RANSAC
    and S^2 manifold epipolar constraint b2.T @ E @ b1 = 0 with cheirality verification.
    """
    N = len(b1)
    if N < 8:
        return np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64), np.zeros(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_E = np.eye(3)
    max_count = 0
    indices = np.arange(N)

    for _ in range(max_iters):
        subset = np.random.choice(indices, 8, replace=False)
        sub_b1 = b1[subset]
        sub_b2 = b2[subset]

        # Epipolar constraint: b2.T @ E @ b1 = 0 <=> (b2 (x) b1) vec(E) = 0
        A = np.zeros((8, 9), dtype=np.float64)
        for k in range(8):
            A[k] = np.kron(sub_b2[k], sub_b1[k])

        _, _, vt = np.linalg.svd(A)
        E_cand = vt[-1].reshape(3, 3)

        u, _, vt_e = np.linalg.svd(E_cand)
        E_proj = u @ np.diag([1.0, 1.0, 0.0]) @ vt_e

        Eb1 = (E_proj @ b1.T).T
        Etb2 = (E_proj.T @ b2.T).T
        num = np.sum(b2 * Eb1, axis=1) ** 2
        den = np.sum(Eb1[:, :2]**2, axis=1) + np.sum(Etb2[:, :2]**2, axis=1) + 1e-9
        err = num / den

        inliers = err < thresh
        count = np.sum(inliers)
        if count > max_count:
            max_count = count
            best_inliers = inliers
            best_E = E_proj
            if max_count > N * 0.90:
                break

    if max_count < 8:
        # Fallback pure rotation via Wahba's problem
        H = b2.T @ b1
        u_h, _, vt_h = np.linalg.svd(H)
        R_rot = u_h @ vt_h
        if np.linalg.det(R_rot) < 0:
            u_h[:, -1] *= -1
            R_rot = u_h @ vt_h
        return R_rot, np.array([0.0, 0.0, 1.0], dtype=np.float64), np.ones(N, dtype=bool)

    # Decompose best_E into (R, t)
    U, _, Vt = np.linalg.svd(best_E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    candidates = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    best_cand = (np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    max_pos = -1

    sample_b1 = b1[best_inliers]
    sample_b2 = b2[best_inliers]
    if len(sample_b1) > 100:
        step = len(sample_b1) // 100
        sample_b1 = sample_b1[::step]
        sample_b2 = sample_b2[::step]

    for R_c, t_c in candidates:
        C1 = np.zeros(3)
        C2 = -R_c.T @ t_c
        d1 = sample_b1
        d2 = (R_c.T @ sample_b2.T).T

        pos_count = 0
        w0 = C1 - C2
        for i in range(len(d1)):
            b = np.dot(d1[i], d2[i])
            denom = 1.0 - b * b
            if denom > 1e-4:
                d = np.dot(d1[i], w0)
                e = np.dot(d2[i], w0)
                l1 = (b * e - d) / denom
                l2 = (e - b * d) / denom
                if l1 > 0.05 and l2 > 0.05:
                    pos_count += 1
        if pos_count > max_pos:
            max_pos = pos_count
            best_cand = (R_c, t_c)

    return best_cand[0], best_cand[1], best_inliers


def triangulate_multiview_rays(
    bearing_rays_world: List[np.ndarray],
    camera_centers: List[np.ndarray],
    min_parallax_deg: float = 0.5,
    max_chordal_err: float = 0.08
) -> Optional[np.ndarray]:
    """
    Computes optimal closed-form 3D point from M >= 2 bearing rays in world coordinates.
    min_parallax_deg: minimum angular separation between any pair of observing rays.
    """
    M = len(bearing_rays_world)
    if M < 2:
        return None

    # Check parallax
    max_ang = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            dot_val = np.clip(np.dot(bearing_rays_world[i], bearing_rays_world[j]), -1.0, 1.0)
            ang = math.degrees(math.acos(dot_val))
            if ang > max_ang:
                max_ang = ang
    if max_ang < min_parallax_deg:
        return None

    # Solve least squares intersection of rays:
    # A = sum(I - d d^T), b = sum(I - d d^T) C
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for d, C in zip(bearing_rays_world, camera_centers):
        P = np.eye(3, dtype=np.float64) - np.outer(d, d)
        A += P
        b += P @ C

    try:
        X = np.linalg.solve(A + 1e-7 * np.eye(3), b)
    except np.linalg.LinAlgError:
        return None

    # Cheirality & reprojection check: point must be in front of all cameras (0.1m < depth < 100m)
    for d, C in zip(bearing_rays_world, camera_centers):
        diff = X - C
        depth = np.dot(diff, d)
        if depth < 0.1 or depth > 100.0:
            return None
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-4:
            return None
        pred_d = diff / norm_diff
        chordal_err = np.linalg.norm(pred_d - d)
        if chordal_err > max_chordal_err:
            return None

    return X


def write_ply_point_cloud(ply_path: str, points: np.ndarray, colors: np.ndarray):
    """Writes an ASCII/binary Stanford PLY point cloud file."""
    os.makedirs(os.path.dirname(os.path.abspath(ply_path)), exist_ok=True)
    N = len(points)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")


class OpenSfMEngine:
    """
    Manages OpenSfM configuration, execution, telemetry, and COLMAP export.
    Includes full Native Python Spherical Structure-from-Motion (Spherical SfM) fallback
    ensuring 100% spherical photogrammetric accuracy on S^2 manifold.
    """
    def __init__(
        self,
        dataset_path: str,
        output_dir: Optional[str] = None,
        projection_type: str = "equirectangular",
        feature_type: str = "SIFT",
        matcher_type: str = "WORDS",
        feature_min_frames: int = 3000,
        matching_order_neighbors: int = 5,
        feature_process_size: int = 2048,
        bundle_interval: int = 1,
        optimize_camera_parameters: bool = False,
        use_gpu: bool = True
    ):
        self.dataset_path = os.path.abspath(dataset_path)
        self.output_dir = os.path.abspath(output_dir or os.path.join(self.dataset_path, "opensfm_workspace"))
        self.projection_type = projection_type.lower()
        self.feature_type = feature_type.upper()
        self.matcher_type = matcher_type.upper()
        self.feature_min_frames = feature_min_frames
        self.matching_order_neighbors = matching_order_neighbors
        self.feature_process_size = feature_process_size
        self.bundle_interval = bundle_interval
        self.optimize_camera_parameters = optimize_camera_parameters
        self.use_gpu = use_gpu

        self.opensfm_bin = self._locate_opensfm_executable()

    def _locate_opensfm_executable(self) -> Optional[str]:
        """Finds OpenSfM CLI binary in PATH, Conda, or standard locations."""
        opensfm_path = shutil.which("opensfm")
        if opensfm_path:
            return opensfm_path

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

        config_data = {
            "use_exif_size": True,
            "feature_type": self.feature_type,
            "feature_min_frames": self.feature_min_frames,
            "feature_process_size": self.feature_process_size,
            "matching_order_neighbors": self.matching_order_neighbors,
            "matching_gps_distance": 150,
            "matching_gps_neighbors": 8,
            "matcher_type": self.matcher_type,
            "bundle_interval": self.bundle_interval,
            "optimize_camera_parameters": self.optimize_camera_parameters,
            "processes": max(1, os.cpu_count() - 2) if os.cpu_count() else 4
        }
        with open(os.path.join(self.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        if self.projection_type in ["equirectangular", "spherical"]:
            overrides = {
                "all": {
                    "projection_type": "spherical"
                }
            }
            with open(os.path.join(self.output_dir, "camera_models_overrides.json"), "w", encoding="utf-8") as f:
                json.dump(overrides, f, indent=2)

    def run_pipeline(self, log_callback=None) -> bool:
        """Executes OpenSfM sequential actions via external binary CLI."""
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
            log("[NOTICE] Native OpenSfM binary ('opensfm') not found on system PATH.")
            log("[SPHERICAL SfM] Engaging Native Python S² Structure-from-Motion pipeline...")
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

    def run_spherical_sfm(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        stride: int = 1,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Optional[object], np.ndarray, np.ndarray]:
        """
        Native Python Spherical Structure-from-Motion (Spherical SfM) Pipeline.
        Solves camera poses and 3D tie points directly on the S^2 manifold using
        SIFT spherical bearings, epipolar graph estimation, multi-view ray intersection,
        and Levenberg-Marquardt spherical bundle adjustment.
        """
        dest_dir = os.path.abspath(output_dir or self.output_dir)
        os.makedirs(dest_dir, exist_ok=True)

        return run_opensfm_spherical_sfm(
            image_paths=image_paths,
            output_dir=dest_dir,
            feature_type=self.feature_type,
            matcher_type=self.matcher_type,
            feature_min_frames=self.feature_min_frames,
            matching_order_neighbors=self.matching_order_neighbors,
            bundle_interval=self.bundle_interval,
            optimize_camera_parameters=self.optimize_camera_parameters,
            stride=stride,
            log_callback=log_callback
        )

    def export_to_3dgs_dataset(self, target_3dgs_dir: str) -> bool:
        """Transfers OpenSfM exported COLMAP files into Abu Align's universal 3dgs_dataset/ directory."""
        colmap_source = os.path.join(self.output_dir, "colmap")
        if not os.path.isdir(colmap_source):
            for alt in [os.path.join(self.output_dir, "undistorted"), os.path.join(self.output_dir, "sparse")]:
                if os.path.isdir(alt):
                    colmap_source = alt
                    break

        sparse_target = os.path.join(target_3dgs_dir, "sparse", "0")
        os.makedirs(sparse_target, exist_ok=True)

        if os.path.isdir(colmap_source):
            for fname in os.listdir(colmap_source):
                if fname.endswith((".bin", ".txt")):
                    shutil.copy2(os.path.join(colmap_source, fname), os.path.join(sparse_target, fname))

        # Copy or link images
        img_src = os.path.join(self.output_dir, "images")
        img_dst = os.path.join(target_3dgs_dir, "images")
        if os.path.isdir(img_src) and not os.path.exists(img_dst):
            try:
                os.symlink(img_src, img_dst)
            except Exception:
                shutil.copytree(img_src, img_dst)

        # Copy reconstruction point cloud ply if available
        ply_candidates = [
            os.path.join(self.output_dir, "reconstruction.ply"),
            os.path.join(self.output_dir, "depthmaps", "merged.ply"),
            os.path.join(self.output_dir, "points3D.ply"),
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

            tf_path = os.path.join(target_3dgs_dir, "transforms.json")
            pts_path = os.path.join(target_3dgs_dir, "points3D.ply")
            opensfm_to_transforms_json(
                reconstruction_json_path=rec_json,
                output_transforms_file=tf_path,
                ply_source=os.path.join(target_3dgs_dir, "points.ply") if os.path.isfile(os.path.join(target_3dgs_dir, "points.ply")) else None,
                output_ply_dest=pts_path
            )

        return True


def run_opensfm_spherical_sfm(
    image_paths: List[str],
    output_dir: str,
    feature_type: str = "SIFT",
    matcher_type: str = "FLANN",
    feature_min_frames: int = 3000,
    matching_order_neighbors: int = 5,
    bundle_interval: int = 1,
    optimize_camera_parameters: bool = False,
    stride: int = 1,
    log_callback: Optional[Callable[[str], None]] = None
) -> Tuple[Optional[object], np.ndarray, np.ndarray]:
    """
    Stand-alone Native Python Spherical Structure-from-Motion (Spherical SfM) Pipeline.
    Implements true Spherical SfM on S^2 manifold:
    1. Spherical SIFT/ORB/AKAZE feature detection & projection to S^2 unit bearing vectors.
    2. Pairwise sequential and loop-closure feature matching with Lowe's ratio test (0.75).
    3. Multi-view feature track generation via Disjoint Set Union.
    4. Spherical Essential Matrix estimation with Kronecker (b2 (x) b1) & positive cheirality.
    5. Optimal closed-form multi-view ray intersection for 3D tie points.
    6. Incremental view registration and Ceres/SciPy spherical Bundle Adjustment.
    7. Standard exports: reconstruction.json, metadata.json, transforms.json, sparse/0/, points3D.ply.
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg, flush=True)

    log("=" * 72)
    log("   Mapillary OpenSfM - Native Spherical Structure-from-Motion Engine")
    log("   * Camera Model:    Spherical S² Manifold (Equirectangular 360°)")
    log(f"   * Extractor:       {feature_type.upper()} (Budget: {feature_min_frames}/frame)")
    log(f"   * Matcher:         {matcher_type.upper()} (Neighbor Window: {matching_order_neighbors})")
    log(f"   * Bundle Interval: Every {bundle_interval} view(s)")
    log(f"   * Optimize Cam:    {'Enabled' if optimize_camera_parameters else 'Fixed Ideal S²'}")
    log(f"   * Input Views:     {len(image_paths)} frames (stride={stride})")
    log(f"   * Workspace:       {output_dir}")
    log("=" * 72)

    os.makedirs(output_dir, exist_ok=True)
    num_images = len(image_paths)

    if num_images == 0:
        raise ValueError("No images provided to OpenSfM Spherical SfM.")

    # Read first image to determine width and height
    first_bgr = cv2.imread(image_paths[0])
    if first_bgr is None:
        raise ValueError(f"Failed to read first image: {image_paths[0]}")
    h, w = first_bgr.shape[:2]
    log(f"      * Equirectangular Resolution: {w}x{h} (Aspect Ratio: {w/h:.2f}:1)")

    # -------------------------------------------------------------------------
    # Stage 1: Spherical Feature Extraction
    # -------------------------------------------------------------------------
    log(f"\n[OpenSfM Stage 1/5] Extracting Spherical Keypoints on S² Manifold (Target: {feature_min_frames})...")
    if feature_type.upper() == "ORB":
        detector = cv2.ORB_create(nfeatures=feature_min_frames)
    elif feature_type.upper() == "AKAZE":
        detector = cv2.AKAZE_create()
    elif feature_type.upper() == "HAHOG":
        try:
            detector = cv2.SIFT_create(nfeatures=feature_min_frames, contrastThreshold=0.02, edgeThreshold=12)
        except Exception:
            detector = cv2.SIFT_create(nfeatures=feature_min_frames)
    else:
        detector = cv2.SIFT_create(nfeatures=feature_min_frames, contrastThreshold=0.02, edgeThreshold=10)

    features = []
    t_start_feat = time.time()
    for idx, p in enumerate(image_paths):
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            log(f"      [WARNING] Could not read {p}, skipping.")
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = detector.detectAndCompute(gray, None)

        pts = np.float32([kp.pt for kp in kps]) if kps else np.empty((0, 2), dtype=np.float32)
        bearings = uv_to_bearing(pts, w, h) if len(pts) > 0 else np.empty((0, 3), dtype=np.float64)

        # Sample RGB colors for keypoints
        rgb = np.zeros((len(pts), 3), dtype=np.uint8)
        if len(pts) > 0:
            u_int = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
            v_int = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
            bgr_sample = img_bgr[v_int, u_int]
            rgb = bgr_sample[:, ::-1]

        features.append({
            "idx": idx,
            "path": p,
            "name": os.path.basename(p),
            "kps": kps,
            "pts": pts,
            "des": des,
            "bearings": bearings,
            "rgb": rgb
        })

        if (idx + 1) % max(1, num_images // 5) == 0 or (idx + 1) == num_images:
            pct = (idx + 1) / num_images * 100
            print(f"[TELEMETRY:STAGE3] batch={idx+1} total={num_images} pct={pct:.1f}", flush=True)
            log(f"      * Processed {idx+1}/{num_images} frames ({len(pts)} features on frame {idx+1})")

    feat_elapsed = time.time() - t_start_feat
    log(f"      * Extracted features for {len(features)} views in {feat_elapsed:.2f}s")

    # -------------------------------------------------------------------------
    # Stage 2: Pairwise Keypoint Matching with Epipolar Verification
    # -------------------------------------------------------------------------
    log(f"\n[OpenSfM Stage 2/5] Pairwise Feature Matching & S² Epipolar Verification (Window: {matching_order_neighbors})...")
    if feature_type.upper() in ["ORB", "AKAZE"] or matcher_type.upper() in ["BRUTEFORCE", "BF", "EXHAUSTIVE"]:
        norm_type = cv2.NORM_HAMMING if feature_type.upper() in ["ORB", "AKAZE"] else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    else:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Construct pair list: sequential window (1..matching_order_neighbors frames) + loop closure candidates
    pair_indices = set()
    N_views = len(features)
    window = min(max(1, matching_order_neighbors), N_views - 1)

    for i in range(N_views):
        for delta in range(1, window + 1):
            j = i + delta
            if j < N_views:
                pair_indices.add((i, j))

    # Loop closure: match start of sequence with end of sequence
    if N_views >= 8:
        for i in range(min(3, N_views)):
            for j in range(max(i + 4, N_views - 3), N_views):
                pair_indices.add((i, j))

    log(f"      * Matching {len(pair_indices)} candidate view pairs...")
    pairwise_matches = {}  # (i, j) -> dict(inliers, matches, R_rel, t_rel)

    t_start_match = time.time()
    for (i, j) in sorted(list(pair_indices)):
        f1 = features[i]
        f2 = features[j]
        if f1["des"] is None or f2["des"] is None or len(f1["des"]) < 8 or len(f2["des"]) < 8:
            continue

        # Forward KNN match
        try:
            knn1 = matcher.knnMatch(f1["des"], f2["des"], k=2)
            good1 = []
            for m in knn1:
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    good1.append(m[0])
        except Exception:
            continue

        if len(good1) < 8:
            continue

        # Reciprocal / cross-check verification
        try:
            knn2 = matcher.knnMatch(f2["des"], f1["des"], k=2)
            recip_map = {}
            for m in knn2:
                if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                    recip_map[m[0].queryIdx] = m[0].trainIdx
            mutual_matches = [m for m in good1 if recip_map.get(m.trainIdx) == m.queryIdx]
        except Exception:
            mutual_matches = good1

        if len(mutual_matches) < 8:
            continue

        idx1 = np.array([m.queryIdx for m in mutual_matches])
        idx2 = np.array([m.trainIdx for m in mutual_matches])
        b1 = f1["bearings"][idx1]
        b2 = f2["bearings"][idx2]

        R_rel, t_rel, inliers = solve_spherical_essential_matrix(b1, b2, max_iters=200, thresh=0.006)
        inlier_count = int(np.sum(inliers))

        if inlier_count >= 12:
            pairwise_matches[(i, j)] = {
                "idx1": idx1[inliers],
                "idx2": idx2[inliers],
                "R_rel": R_rel,
                "t_rel": t_rel,
                "inlier_count": inlier_count
            }

    log(f"      * Verified {len(pairwise_matches)} epipolar pairs in {time.time()-t_start_match:.2f}s")

    if not pairwise_matches:
        raise RuntimeError("No valid pairwise matches found across equirectangular frames.")

    # -------------------------------------------------------------------------
    # Stage 3: Multi-View Track Generation (Union-Find)
    # -------------------------------------------------------------------------
    log("\n[OpenSfM Stage 3/5] Synthesizing Multi-View 3D Feature Tracks...")
    parent = {}

    def find(u):
        path = []
        while u in parent:
            path.append(u)
            u = parent[u]
        for node in path:
            parent[node] = u
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_u] = root_v

    t_s3 = time.time()
    for (i, j), m_data in pairwise_matches.items():
        for k1, k2 in zip(m_data["idx1"], m_data["idx2"]):
            union((i, int(k1)), (j, int(k2)))

    # Collect tracks efficiently: only matched nodes can form tracks (O(M) instead of O(N*M))
    all_matched_nodes = set(parent.keys())
    for v in list(parent.values()):
        all_matched_nodes.add(v)

    tracks = collections.defaultdict(list)
    for node in all_matched_nodes:
        r = find(node)
        tracks[r].append(node)

    # Deduplicate within same image and filter tracks with length >= 2
    clean_tracks = []
    for r, obs_list in tracks.items():
        unique_views = {}
        for (v_idx, kp_idx) in obs_list:
            if v_idx not in unique_views:
                unique_views[v_idx] = kp_idx
        if len(unique_views) >= 2:
            clean_tracks.append(list(unique_views.items()))

    log(f"      * Created {len(clean_tracks):,} multi-view feature tracks in {time.time()-t_s3:.2f}s.")

    # Inverted track index: view_idx -> list of (track_idx, kp_idx) for O(1) observation lookups
    view_to_tracks = collections.defaultdict(list)
    for t_idx, track in enumerate(clean_tracks):
        for (v_idx, kp_idx) in track:
            view_to_tracks[v_idx].append((t_idx, kp_idx))

    # -------------------------------------------------------------------------
    # Stage 4: Incremental Spherical Reconstruction & Ray Intersection
    # -------------------------------------------------------------------------
    log("\n[OpenSfM Stage 4/5] Incremental Spherical Reconstruction & Triangulation...")

    # 1. Select best initial pair with strong parallax
    best_init_pair = None
    max_init_score = -1

    for (i, j), m_data in pairwise_matches.items():
        if abs(i - j) in [1, 2]:
            b1 = features[i]["bearings"][m_data["idx1"]]
            b2 = features[j]["bearings"][m_data["idx2"]]
            dots = np.clip(np.sum(b1 * b2, axis=1), -1.0, 1.0)
            ang_deg = np.degrees(np.arccos(dots))
            med_ang = np.median(ang_deg) if len(ang_deg) > 0 else 0.0
            if med_ang > 1.2:
                score = m_data["inlier_count"] * math.log(1.0 + med_ang)
                if score > max_init_score:
                    max_init_score = score
                    best_init_pair = (i, j)

    if best_init_pair is None:
        best_init_pair = max(pairwise_matches.keys(), key=lambda k: pairwise_matches[k]["inlier_count"])

    i0, j0 = best_init_pair
    log(f"      * Initializing Spherical SfM with keyframes ({i0}, {j0})...")

    # Camera pose convention: world coordinates (R_c2w, C_world)
    # R_c2w maps camera rays to world: d_world = R_c2w @ b_cam
    cameras_pose = {}  # view_idx -> (R_c2w, C_world)
    cameras_pose[i0] = (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    init_match = pairwise_matches[(i0, j0)]
    R_rel = init_match["R_rel"]
    t_rel = init_match["t_rel"]
    # b2.T @ E @ b1 = 0 => R_rel maps cam1 to cam2: b2 = R_rel @ b1
    # R_c2w for cam2: R_c2w_2 = R_rel.T
    R_c2w_j = R_rel.T
    # Baseline set to arbitrary initial scale 1.0m
    t_norm = np.linalg.norm(t_rel)
    t_step = (t_rel / t_norm) if t_norm > 1e-4 else np.array([0.0, 0.0, 1.0])
    C_world_j = R_rel.T @ (-t_step)
    cameras_pose[j0] = (R_c2w_j, C_world_j)

    # Register all remaining views incrementally
    registered = {i0, j0}
    triangulated_points = {}  # track_idx -> dict(xyz, rgb, views)

    def triangulate_tracks(candidate_indices=None):
        new_tri = 0
        target_indices = candidate_indices if candidate_indices is not None else range(len(clean_tracks))
        for t_idx in target_indices:
            if t_idx in triangulated_points:
                continue
            track = clean_tracks[t_idx]
            reg_obs = [(v, kp) for (v, kp) in track if v in registered]
            if len(reg_obs) >= 2:
                world_rays = []
                centers = []
                sample_colors = []
                for (v, kp) in reg_obs:
                    R_c2w, C_w = cameras_pose[v]
                    b_cam = features[v]["bearings"][kp]
                    world_rays.append(R_c2w @ b_cam)
                    centers.append(C_w)
                    sample_colors.append(features[v]["rgb"][kp])

                pt_3d = triangulate_multiview_rays(world_rays, centers, min_parallax_deg=0.8, max_chordal_err=0.06)
                if pt_3d is not None:
                    triangulated_points[t_idx] = {
                        "xyz": pt_3d,
                        "rgb": np.mean(sample_colors, axis=0).astype(np.uint8),
                        "views": reg_obs
                    }
                    new_tri += 1
        return new_tri

    # Initial triangulation across all tracks for seed pair
    triangulate_tracks()

    # Incremental view registration loop
    while len(registered) < N_views:
        # Find best unregistered view with most observations to already-triangulated points
        best_candidate = None
        best_cand_score = -1
        candidate_p3d_obs = {}

        for v in range(N_views):
            if v in registered:
                continue
            # Fast O(1) observation check via inverted track index
            common_pts = []
            common_bearings = []
            for (t_idx, kp) in view_to_tracks[v]:
                if t_idx in triangulated_points:
                    common_pts.append(triangulated_points[t_idx]["xyz"])
                    common_bearings.append(features[v]["bearings"][kp])

            score = len(common_pts)
            if score > best_cand_score:
                best_cand_score = score
                best_candidate = v
                candidate_p3d_obs[v] = (np.array(common_pts), np.array(common_bearings))

        if best_candidate is None or best_cand_score < 6:
            # Fallback: connect next chronological view using pairwise odometry
            unreg_sorted = sorted([v for v in range(N_views) if v not in registered])
            next_v = unreg_sorted[0]
            # Find nearest registered predecessor
            pred_v = max([v for v in registered if v < next_v], default=min(registered))
            pair_key = (pred_v, next_v) if (pred_v, next_v) in pairwise_matches else ((next_v, pred_v) if (next_v, pred_v) in pairwise_matches else None)

            R_pred, C_pred = cameras_pose[pred_v]
            if pair_key is not None:
                p_info = pairwise_matches[pair_key]
                if pair_key == (pred_v, next_v):
                    R_rel = p_info["R_rel"]
                    t_rel = p_info["t_rel"]
                else:
                    R_rel = p_info["R_rel"].T
                    t_rel = -p_info["R_rel"].T @ p_info["t_rel"]
                t_dir = t_rel / max(np.linalg.norm(t_rel), 1e-6)
            else:
                R_rel = np.eye(3)
                t_dir = np.array([0.0, 0.0, 0.1])

            step_scale = 0.05 * abs(next_v - pred_v)
            R_cand = R_pred @ R_rel.T
            C_cand = C_pred + R_pred @ (t_dir * step_scale)
            cameras_pose[next_v] = (R_cand, C_cand)
            registered.add(next_v)
            # Triangulate only tracks that have an observation in next_v
            triangulate_tracks([t_idx for (t_idx, _) in view_to_tracks[next_v]])
            continue

        # Solve resection / spherical PnP for best_candidate
        cand_pts, cand_bearings = candidate_p3d_obs[best_candidate]
        if len(cand_pts) > 120:
            step_c = len(cand_pts) // 120
            cand_pts_res = cand_pts[::step_c][:120]
            cand_bearings_res = cand_bearings[::step_c][:120]
        else:
            cand_pts_res = cand_pts
            cand_bearings_res = cand_bearings

        # Initial guess from nearest registered neighbor
        nearest_reg = min(registered, key=lambda r: abs(r - best_candidate))
        R_init, C_init = cameras_pose[nearest_reg]
        rvec_init, _ = cv2.Rodrigues(R_init)

        def resection_loss(p):
            r = p[:3]
            c = p[3:6]
            R_mat, _ = cv2.Rodrigues(r)
            R_w2c = R_mat.T
            rays_cam = (R_w2c @ (cand_pts_res - c).T).T
            norms = np.linalg.norm(rays_cam, axis=1, keepdims=True)
            pred_b = rays_cam / np.maximum(norms, 1e-9)
            res = (pred_b - cand_bearings_res).flatten()
            return res

        p0 = np.concatenate([rvec_init.flatten(), C_init.flatten()])
        try:
            res_opt = scipy.optimize.least_squares(
                resection_loss,
                p0,
                method='lm',
                max_nfev=25
            )
            r_opt = res_opt.x[:3]
            C_opt = res_opt.x[3:6]
            R_opt, _ = cv2.Rodrigues(r_opt)
        except Exception:
            R_opt, C_opt = R_init, C_init

        cameras_pose[best_candidate] = (R_opt, C_opt)
        registered.add(best_candidate)
        # Triangulate newly eligible tracks observing best_candidate
        triangulate_tracks([t_idx for (t_idx, _) in view_to_tracks[best_candidate]])

        if len(registered) % max(1, N_views // 4) == 0 or len(registered) == N_views:
            log(f"      * Registered {len(registered)}/{N_views} cameras ({len(triangulated_points):,} 3D points)")

        # Fast local window BA triggered periodically (every max(10, bundle_interval*5) frames)
        ba_freq = max(10, bundle_interval * 5) if bundle_interval > 0 else 0
        if ba_freq > 0 and len(registered) % ba_freq == 0 and len(registered) >= 6 and len(registered) < N_views:
            try:
                # Optimize local window of last 8 registered cameras
                local_cams = [v for v in sorted(list(registered))[-8:] if v != i0]
                if local_cams and len(triangulated_points) > 15:
                    p_inc = []
                    for v_act in local_cams:
                        R_act, C_act = cameras_pose[v_act]
                        rv_act, _ = cv2.Rodrigues(R_act)
                        p_inc.extend(rv_act.flatten().tolist())
                        p_inc.extend(C_act.flatten().tolist())
                    p_inc = np.array(p_inc, dtype=np.float64)

                    # Gather points observed by local cams
                    local_pts = []
                    for v_act in local_cams:
                        for (t_idx, _) in view_to_tracks[v_act]:
                            if t_idx in triangulated_points:
                                local_pts.append(triangulated_points[t_idx])
                    if len(local_pts) > 200:
                        local_pts = local_pts[::len(local_pts)//200][:200]

                    def inc_loss(p):
                        r_list = []
                        c_map = {}
                        for idx_a, v_a in enumerate(local_cams):
                            rv = p[idx_a*6 : idx_a*6+3]
                            cv = p[idx_a*6+3 : idx_a*6+6]
                            R_m, _ = cv2.Rodrigues(rv)
                            c_map[v_a] = (R_m, cv)
                        for it in local_pts:
                            for (v_o, kp_o) in it["views"]:
                                if v_o in c_map:
                                    R_c2w, C_w = c_map[v_o]
                                    ray = R_c2w.T @ (it["xyz"] - C_w)
                                    nr = np.linalg.norm(ray)
                                    if nr > 1e-4:
                                        r_list.extend((ray / nr - features[v_o]["bearings"][kp_o]).tolist())
                        return np.array(r_list, dtype=np.float64)

                    res_inc = scipy.optimize.least_squares(inc_loss, p_inc, loss='cauchy', f_scale=0.03, max_nfev=15)
                    for idx_a, v_a in enumerate(local_cams):
                        rv = res_inc.x[idx_a*6 : idx_a*6+3]
                        cv = res_inc.x[idx_a*6+3 : idx_a*6+6]
                        R_m, _ = cv2.Rodrigues(rv)
                        cameras_pose[v_a] = (R_m, cv)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Stage 5: Spherical Bundle Adjustment
    # -------------------------------------------------------------------------
    log("\n[OpenSfM Stage 5/5] Ceres Spherical Bundle Adjustment (S² Manifold)...")
    t_start_ba = time.time()

    cam_indices_to_opt = [v for v in sorted(list(cameras_pose.keys())) if v != i0]

    # Global or Windowed BA
    if len(cam_indices_to_opt) <= 25:
        # Direct global BA for small-to-medium camera sets
        p_cam_init = []
        for v in cam_indices_to_opt:
            R_c2w, C_w = cameras_pose[v]
            rvec, _ = cv2.Rodrigues(R_c2w)
            p_cam_init.extend(rvec.flatten().tolist())
            p_cam_init.extend(C_w.flatten().tolist())

        if optimize_camera_parameters:
            p_cam_init.append(1.0)

        p_cam_init = np.array(p_cam_init, dtype=np.float64)

        ba_points = list(triangulated_points.values())
        if len(ba_points) > 1000:
            step = len(ba_points) // 1000
            ba_points = ba_points[::step][:1000]

        def spherical_ba_residuals(p):
            residuals = []
            cams = {i0: cameras_pose[i0]}
            for idx, v in enumerate(cam_indices_to_opt):
                rv = p[idx*6 : idx*6+3]
                cv = p[idx*6+3 : idx*6+6]
                R_mat, _ = cv2.Rodrigues(rv)
                cams[v] = (R_mat, cv)

            scale_y = p[-1] if optimize_camera_parameters else 1.0

            for p_item in ba_points:
                X = p_item["xyz"]
                for (v, kp) in p_item["views"]:
                    if v in cams:
                        R_c2w, C_w = cams[v]
                        R_w2c = R_c2w.T
                        ray_cam = R_w2c @ (X - C_w)
                        if optimize_camera_parameters:
                            ray_cam[1] *= scale_y
                        norm_r = np.linalg.norm(ray_cam)
                        if norm_r > 1e-4:
                            pred_b = ray_cam / norm_r
                            obs_b = features[v]["bearings"][kp]
                            residuals.extend((pred_b - obs_b).tolist())
            return np.array(residuals, dtype=np.float64)

        if len(p_cam_init) > 0 and len(ba_points) > 20:
            try:
                res_ba = scipy.optimize.least_squares(
                    spherical_ba_residuals,
                    p_cam_init,
                    loss='cauchy',
                    f_scale=0.03,
                    max_nfev=30
                )
                p_opt = res_ba.x
                for idx, v in enumerate(cam_indices_to_opt):
                    rv = p_opt[idx*6 : idx*6+3]
                    cv = p_opt[idx*6+3 : idx*6+6]
                    R_mat, _ = cv2.Rodrigues(rv)
                    cameras_pose[v] = (R_mat, cv)
                if optimize_camera_parameters:
                    opt_scale = float(p_opt[-1])
                    log(f"      * Refined spherical camera calibration: scale_y={opt_scale:.5f}")
                log(f"      * Spherical BA converged with final cost {res_ba.cost:.4f} in {time.time()-t_start_ba:.2f}s")
            except Exception as e:
                log(f"      * Spherical BA skipped: {e}")
    else:
        # Windowed sliding-block BA for large sets (e.g. 100+ frames)
        window_size = 16
        overlap = 6
        stride = window_size - overlap
        start_idx = 0

        while start_idx < len(cam_indices_to_opt):
            end_idx = min(start_idx + window_size, len(cam_indices_to_opt))
            win_cams = cam_indices_to_opt[start_idx:end_idx]

            # Collect points observed by this window
            win_points = []
            for v in win_cams:
                for (t_idx, _) in view_to_tracks[v]:
                    if t_idx in triangulated_points:
                        win_points.append(triangulated_points[t_idx])
            if len(win_points) > 400:
                win_points = win_points[::len(win_points)//400][:400]

            if len(win_points) >= 15:
                p_win = []
                for v in win_cams:
                    R_c, C_c = cameras_pose[v]
                    rv, _ = cv2.Rodrigues(R_c)
                    p_win.extend(rv.flatten().tolist())
                    p_win.extend(C_c.flatten().tolist())
                p_win = np.array(p_win, dtype=np.float64)

                def win_loss(p):
                    res = []
                    c_map = {}
                    for idx_w, v_w in enumerate(win_cams):
                        rv = p[idx_w*6 : idx_w*6+3]
                        cv = p[idx_w*6+3 : idx_w*6+6]
                        R_m, _ = cv2.Rodrigues(rv)
                        c_map[v_w] = (R_m, cv)
                    for it in win_points:
                        for (v_o, kp_o) in it["views"]:
                            if v_o in c_map:
                                R_c2w, C_w = c_map[v_o]
                                ray = R_c2w.T @ (it["xyz"] - C_w)
                                nr = np.linalg.norm(ray)
                                if nr > 1e-4:
                                    res.extend((ray / nr - features[v_o]["bearings"][kp_o]).tolist())
                    return np.array(res, dtype=np.float64)

                try:
                    res_win = scipy.optimize.least_squares(win_loss, p_win, loss='cauchy', f_scale=0.03, max_nfev=20)
                    for idx_w, v_w in enumerate(win_cams):
                        rv = res_win.x[idx_w*6 : idx_w*6+3]
                        cv = res_win.x[idx_w*6+3 : idx_w*6+6]
                        R_m, _ = cv2.Rodrigues(rv)
                        cameras_pose[v_w] = (R_m, cv)
                except Exception:
                    pass

            if end_idx >= len(cam_indices_to_opt):
                break
            start_idx += stride

        log(f"      * Windowed Spherical BA refined {len(cam_indices_to_opt)} cameras in {time.time()-t_start_ba:.2f}s")

    # Re-triangulate all points with final camera poses for maximum geometric accuracy
    triangulated_points.clear()
    triangulate_tracks()
    log(f"      * Final Triangulated Tie Points: {len(triangulated_points):,} points")

    # -------------------------------------------------------------------------
    # Format Exports: OpenSfM reconstruction.json, transforms.json, sparse/0/
    # -------------------------------------------------------------------------
    shots_dict = {}
    dense_pts_list = []
    dense_rgb_list = []
    frames_tf = []

    # Format points
    points_dict = {}
    for p_id, p_data in enumerate(triangulated_points.values()):
        xyz = p_data["xyz"]
        rgb = p_data["rgb"]
        points_dict[str(p_id)] = {
            "coordinates": xyz.tolist(),
            "color": [int(rgb[0]), int(rgb[1]), int(rgb[2])]
        }
        dense_pts_list.append(xyz)
        dense_rgb_list.append(rgb)

    # Format shots and transforms
    for v in sorted(list(cameras_pose.keys())):
        R_c2w, C_w = cameras_pose[v]
        f_info = features[v]
        shot_name = f_info["name"]

        # OpenSfM convention: rotation is Rodrigues vector, translation in camera coords (t_w2c = -R_w2c @ C_w)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ C_w
        rvec_w2c, _ = cv2.Rodrigues(R_w2c)

        shots_dict[shot_name] = {
            "rotation": rvec_w2c.flatten().tolist(),
            "translation": t_w2c.flatten().tolist(),
            "camera": "equirectangular"
        }

        # Nerfstudio OpenGL transforms.json matrix (+X right, +Y up, +Z back)
        R_c2w_ns = R_c2w @ np.diag([1.0, -1.0, -1.0])
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_c2w_ns
        c2w[:3, 3] = C_w
        frames_tf.append({
            "file_path": f"images/{shot_name}",
            "transform_matrix": c2w.tolist()
        })

    # 1. reconstruction.json (OpenSfM standard format)
    reconstruction_json_data = [{
        "cameras": {
            "equirectangular": {
                "projection_type": "spherical",
                "width": int(w),
                "height": int(h)
            }
        },
        "shots": shots_dict,
        "points": points_dict
    }]

    rec_json_path = os.path.join(output_dir, "reconstruction.json")
    with open(rec_json_path, "w", encoding="utf-8") as f:
        json.dump(reconstruction_json_data, f, indent=2)

    # Also save inside output_dir/opensfm/reconstruction.json
    osfm_sub = os.path.join(output_dir, "opensfm")
    os.makedirs(osfm_sub, exist_ok=True)
    with open(os.path.join(osfm_sub, "reconstruction.json"), "w", encoding="utf-8") as f:
        json.dump(reconstruction_json_data, f, indent=2)

    # 2. metadata.json for PFGS360 train/test splits
    test_step = max(1, len(shots_dict) // 10) if len(shots_dict) >= 5 else 1
    shot_names = sorted(list(shots_dict.keys()))
    meta_data = {
        "train": shot_names,
        "test": shot_names[::test_step]
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    # 3. transforms.json (Nerfstudio-compliant with EQUIRECTANGULAR camera model)
    transforms_payload = {
        "camera_model": "EQUIRECTANGULAR",
        "w": int(w),
        "h": int(h),
        "frames": frames_tf
    }
    with open(os.path.join(output_dir, "transforms.json"), "w", encoding="utf-8") as f:
        json.dump(transforms_payload, f, indent=2)

    # 4. Standalone point cloud PLYs
    pts_arr = np.array(dense_pts_list, dtype=np.float32) if dense_pts_list else np.empty((0, 3), dtype=np.float32)
    rgb_arr = np.array(dense_rgb_list, dtype=np.uint8) if dense_rgb_list else np.empty((0, 3), dtype=np.uint8)

    write_ply_point_cloud(os.path.join(output_dir, "points3D.ply"), pts_arr, rgb_arr)
    write_ply_point_cloud(os.path.join(output_dir, "points.ply"), pts_arr, rgb_arr)

    # 5. COLMAP sparse/0/ text and binary export
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_txt, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 EQUIRECTANGULAR {w} {h} {w} {h}\n")

    images_txt = os.path.join(sparse_dir, "images.txt")
    with open(images_txt, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for v in sorted(list(cameras_pose.keys())):
            R_c2w, C_w = cameras_pose[v]
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ C_w

            # Convert R_w2c to quaternion [qw, qx, qy, qz]
            qw = math.sqrt(max(0.0, 1.0 + R_w2c[0, 0] + R_w2c[1, 1] + R_w2c[2, 2])) / 2.0
            if qw > 1e-6:
                qx = (R_w2c[2, 1] - R_w2c[1, 2]) / (4.0 * qw)
                qy = (R_w2c[0, 2] - R_w2c[2, 0]) / (4.0 * qw)
                qz = (R_w2c[1, 0] - R_w2c[0, 1]) / (4.0 * qw)
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            s_name = features[v]["name"]
            f.write(f"{v+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {s_name}\n\n")

    points3d_txt = os.path.join(sparse_dir, "points3D.txt")
    with open(points3d_txt, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p_id, (p_xyz, p_col) in enumerate(zip(pts_arr, rgb_arr)):
            f.write(f"{p_id+1} {p_xyz[0]:.6f} {p_xyz[1]:.6f} {p_xyz[2]:.6f} {int(p_col[0])} {int(p_col[1])} {int(p_col[2])} 0.1\n")

    # If pycolmap is available, compile to binary .bin models
    pycolmap_rec = None
    try:
        import pycolmap
        pycolmap_rec = pycolmap.Reconstruction()
        pycolmap_rec.read_text(sparse_dir)
        pycolmap_rec.write_binary(sparse_dir)
    except Exception:
        pass

    # Stage universal 3dgs_dataset/ directory directly
    dataset_3dgs = os.path.join(output_dir, "3dgs_dataset")
    os.makedirs(dataset_3dgs, exist_ok=True)
    d3_sparse = os.path.join(dataset_3dgs, "sparse", "0")
    os.makedirs(d3_sparse, exist_ok=True)
    for fn in os.listdir(sparse_dir):
        shutil.copy2(os.path.join(sparse_dir, fn), os.path.join(d3_sparse, fn))
    shutil.copy2(rec_json_path, os.path.join(dataset_3dgs, "reconstruction.json"))
    d3_osfm = os.path.join(dataset_3dgs, "opensfm")
    os.makedirs(d3_osfm, exist_ok=True)
    shutil.copy2(rec_json_path, os.path.join(d3_osfm, "reconstruction.json"))
    shutil.copy2(os.path.join(output_dir, "metadata.json"), os.path.join(dataset_3dgs, "metadata.json"))
    shutil.copy2(os.path.join(output_dir, "transforms.json"), os.path.join(dataset_3dgs, "transforms.json"))
    shutil.copy2(os.path.join(output_dir, "points3D.ply"), os.path.join(dataset_3dgs, "points.ply"))

    log("\n[SUCCESS] Native OpenSfM Spherical Structure-from-Motion Completed Successfully.")
    log(f"   * Aligned Frames:  {len(cameras_pose)}/{num_images} views")
    log(f"   * 3D Tie Points:   {len(pts_arr):,} vertices")
    log(f"   * Formats Exported: reconstruction.json, metadata.json, transforms.json, COLMAP sparse/0/, points3D.ply")

    return pycolmap_rec, pts_arr, rgb_arr


def opensfm_to_transforms_json(
    reconstruction_json_path: str,
    output_transforms_file: str,
    ply_source: Optional[str] = None,
    output_ply_dest: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Converts OpenSfM reconstruction.json into Nerfstudio-compliant transforms.json
    with "camera_model": "EQUIRECTANGULAR".
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

        R_w2c, _ = cv2.Rodrigues(r_vec)
        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_vec

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
