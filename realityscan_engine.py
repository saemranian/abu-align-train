"""
RealityScan (Epic Games / Capturing Reality) Alignment Engine Adapter
====================================================================
Integrates Epic Games' RealityScan into the Abu Align & Train ecosystem.
Supports:
  - Trajectory prior ingestion (from VGGT-Omega, VGG-TTT, ACE0, PyCuSFM, etc.)
  - Verbatim alignment settings & prior calibrations (Screenshots 2, 3, 4)
  - Headless execution with real-time stdout streaming into Abu Align activity console
  - GUI software window mode for direct user interaction
  - Automated exports: COLMAP for LichtFeld (sparse/0/), Postshot (cameras.csv, points.ply, transforms.json)
"""

import os
import sys
import time
import glob
import json
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable


# =========================================================================
# Executable Discovery
# =========================================================================

def find_realityscan_executable() -> Optional[str]:
    """
    Auto-detects the installed RealityScan / RealityCapture binary.
    Checks RealityScan 2.2, 2.1, 2.0, and legacy Capturing Reality paths.
    """
    candidates = [
        r"C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe",
        r"C:\Program Files\Epic Games\RealityScan_2.1\RealityScan.exe",
        r"C:\Program Files\Epic Games\RealityScan_2.0\RealityScan.exe",
        r"C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe",
    ]
    # Check custom environment variable
    env_path = os.environ.get("REALITYSCAN_PATH", "")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # Check PATH
    which_rs = shutil.which("RealityScan.exe") or shutil.which("RealityCapture.exe")
    if which_rs and os.path.isfile(which_rs):
        return os.path.abspath(which_rs)

    return None


# =========================================================================
# Trajectory Import XML & Settings Generator (Screenshot 4)
# =========================================================================

def generate_trajectory_import_xml(output_xml_path: str, format_id: str = "{35CD8B84-6573-417D-8FEE-BE8BBEEC00D3}") -> str:
    """
    Generates RealityScan Trajectory Import settings XML conforming to Screenshot 4:
    - Format: Image X/Lon Y/Lat Z/Alt Yaw Pitch Roll ({35CD8B84-6573-417D-8FEE-BE8BBEEC00D3})
    - Euler order: ZYX (photogrammetric YPR convention)
    - Camera mount: Nadir-facing, X-axis to right wing (RealityScan convention)
    - Values separator: Comma ","
    - Ignore first line: true
    - Coordinate system: local:1 - Euclidean
    - Group calibration: By focal length
    - Accuracy settings source: Global camera prior settings
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_xml_path)), exist_ok=True)
    xml_content = f"""<ImporterSettings formatId="{format_id}">
    <format id="{format_id}" desc="Image X/Lon Y/Lat Z/Alt Yaw Pitch Roll" reader="RealityScan.Import.CSVFlightLog">
        <parser allowedSeparators=",; &tab;" comment="#" showIgnoreFirstline="true" qualifiers="&quot;optional">
            <Image index="0" format="name.ext"/>
            <X index="1" format="value"/>
            <Y index="2" format="value"/>
            <Altitude index="3" format="value"/>
            <Yaw index="4" format="value"/>
            <Pitch index="5" format="value"/>
            <Roll index="6" format="value"/>
        </parser>
    </format>
    <coordinateSystem>local:1 - Euclidean</coordinateSystem>
    <cameraMount>Nadir-facing, X-axis to right wing (RealityScan convention)</cameraMount>
    <eulerOrder>ZYX</eulerOrder>
    <separator>,</separator>
    <ignoreFirstLine>true</ignoreFirstLine>
    <groupCalibration>Use the focal length to group cameras intrinsic parameters</groupCalibration>
    <accuracySource>Global camera prior settings</accuracySource>
</ImporterSettings>
"""
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
    return output_xml_path


# =========================================================================
# Prior Trajectory Resolution & Adapter
# =========================================================================

def find_prior_trajectory(
    scan_dir: str,
    explicit_prior_run: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Discovers prior alignment trajectory (cameras.csv) from previous Abu Align runs
    (VGGT-Omega, VGG-TTT, ACE0, PyCuSFM, OpenSfM, 360DVO).
    Returns (cameras_csv_path, source_run_dir).
    """
    # 1. If user explicitly provided a path (file or run dir)
    if explicit_prior_run:
        p = os.path.abspath(explicit_prior_run)
        if os.path.isfile(p):
            return p, os.path.dirname(p)
        if os.path.isdir(p):
            csv_cand = os.path.join(p, "Postshot_RealityCapture", "cameras.csv")
            if os.path.isfile(csv_cand): return csv_cand, p
            csv_cand2 = os.path.join(p, "cameras.csv")
            if os.path.isfile(csv_cand2): return csv_cand2, p

    # 2. Check results/ subfolder
    results_dir = os.path.join(scan_dir, "results")
    if not os.path.isdir(results_dir):
        parent = os.path.dirname(scan_dir)
        cand_res = os.path.join(parent, "results")
        if os.path.isdir(cand_res):
            results_dir = cand_res

    if not os.path.isdir(results_dir):
        return None, None

    # Discover candidate runs sorted by modification time descending
    runs = []
    for entry in os.listdir(results_dir):
        full_p = os.path.join(results_dir, entry)
        if os.path.isdir(full_p) and not entry.startswith(".") and entry.lower() != "3dgs_models":
            # Exclude existing realityscan runs to avoid self-referencing
            if "realityscan" not in entry.lower() and "realitycapture" not in entry.lower():
                runs.append(full_p)

    runs.sort(key=lambda d: os.path.getmtime(d), reverse=True)

    for run_dir in runs:
        # Check Postshot_RealityCapture/cameras.csv
        cand1 = os.path.join(run_dir, "Postshot_RealityCapture", "cameras.csv")
        if os.path.isfile(cand1) and os.path.getsize(cand1) > 200:
            return cand1, run_dir

        # Check cameras.csv in root of run
        cand2 = os.path.join(run_dir, "cameras.csv")
        if os.path.isfile(cand2) and os.path.getsize(cand2) > 200:
            return cand2, run_dir

    return None, None


# =========================================================================
# Trajectory CSV Converter (from transforms.json or COLMAP)
# =========================================================================

def convert_transforms_to_realityscan_csv(transforms_path: str, output_csv_path: str) -> bool:
    """
    Converts NeRF/3DGS transforms.json into RealityScan cameras.csv
    with Image X/Lon Y/Lat Z/Alt Yaw Pitch Roll format.
    """
    if not os.path.isfile(transforms_path):
        return False
    try:
        with open(transforms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        if not frames:
            return False

        # Transform from NeRF/OpenGL (X-right, Y-up, Z-back) to RealityCapture (Z-up, Y-north)
        rows = ["#name,x,y,alt,yaw,pitch,roll,f_35mm,px_norm,py_norm,k1,k2,k3,k4,t1,t2\n"]
        for frame in frames:
            name = os.path.basename(frame.get("file_path", ""))
            matrix = np.array(frame["transform_matrix"], dtype=np.float64)
            # NeRF c2w: matrix[:3, :3] is camera rotation, matrix[:3, 3] is position
            pos = matrix[:3, 3]
            # Convert NeRF (OpenGL) to COLMAP (OpenCV): flip Y and Z
            R_gl = matrix[:3, :3]
            R_cv = R_gl @ np.diag([1.0, -1.0, -1.0])

            # RealityCapture world: X_rc = X_cv, Y_rc = Z_cv, Z_rc = -Y_cv
            M_world = np.array([
                [1.0,  0.0,  0.0],
                [0.0,  0.0,  1.0],
                [0.0, -1.0,  0.0]
            ])
            pos_rc = M_world @ pos

            # Optical axes in RC world space:
            fwd_rc = M_world @ (R_cv @ np.array([0.0, 0.0, 1.0]))
            up_rc  = M_world @ (R_cv @ np.array([0.0, -1.0, 0.0]))
            fwd_rc /= max(1e-9, np.linalg.norm(fwd_rc))
            up_rc  /= max(1e-9, np.linalg.norm(up_rc))

            yaw = np.degrees(np.arctan2(fwd_rc[0], fwd_rc[1])) % 360.0
            pitch = np.degrees(np.arccos(np.clip(-fwd_rc[2], -1.0, 1.0)))
            yaw_rad = np.radians(yaw)
            w0 = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0.0])
            w0_norm = np.linalg.norm(w0)
            if w0_norm > 1e-6:
                w0 /= w0_norm
                u0 = np.cross(w0, fwd_rc)
                u0 /= max(1e-9, np.linalg.norm(u0))
                roll = np.degrees(np.arctan2(np.dot(up_rc, w0), np.dot(up_rc, u0)))
            else:
                roll = 0.0

            rows.append(f"{name},{pos_rc[0]:.6f},{pos_rc[1]:.6f},{pos_rc[2]:.6f},{yaw:.6f},{pitch:.6f},{roll:.6f},35.0,0.0,0.0,0,0,0,0,0,0\n")

        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, "w", encoding="utf-8") as f:
            f.writelines(rows)
        return True
    except Exception as e:
        print(f"[Warning] Failed to convert transforms.json to cameras.csv: {e}")
        return False


# =========================================================================
# Main RealityScan Execution Pipeline
# =========================================================================

def run_realityscan_alignment(
    images_dir: str,
    output_dir: str,
    prior_trajectory_csv: Optional[str] = None,
    # GUI vs Headless Mode
    headless: bool = True,
    executable_path: Optional[str] = None,
    # Screenshot 3: Alignment Settings
    feature_detection_quality: str = "High",
    max_features_per_mpx: int = 80000,
    max_features_per_image: int = 320000,
    image_overlap: str = "Low",
    image_downscale_factor: int = 1,
    max_feature_reprojection_error: float = 2.0,
    # Advanced Options (Screenshot 3)
    gpu_acceleration: bool = True,
    add_reconstruction_region: bool = True,
    force_component_rematch: bool = True,
    preselector_features: int = 40000,
    detector_sensitivity: str = "High",
    distortion_model: str = "Brown3",
    use_mosaic_images: bool = True,
    # Screenshot 2: Prior Pose & Calibration
    absolute_pose: str = "Position and orientation",
    absolute_coords: str = "local:1 - Euclidean",
    prior_calibration: str = "Approximate",
    focal_length_35mm: float = 25.0,
    export_colmap: bool = True,
    export_postshot: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Any], np.ndarray, np.ndarray]:
    """
    Executes RealityScan alignment via CLI, streaming logs in real time.
    Returns (reconstruction, dense_3d, dense_rgb).
    """
    log = progress_callback or print

    exe = executable_path or find_realityscan_executable()
    if not exe or not os.path.isfile(exe):
        raise FileNotFoundError(
            f"RealityScan executable not found! Please install Epic Games RealityScan or set REALITYSCAN_PATH.\n"
            f"Checked default paths: C:\\Program Files\\Epic Games\\RealityScan_2.2\\RealityScan.exe"
        )

    log(f"[*] RealityScan Executable: {exe}")
    log(f"[*] Images Directory:      {images_dir}")
    log(f"[*] Output Directory:      {output_dir}")
    log(f"[*] Execution Mode:        {'Headless (Background CLI)' if headless else 'Interactive (Open RealityScan Window)'}")

    os.makedirs(output_dir, exist_ok=True)
    postshot_dir = os.path.join(output_dir, "Postshot_RealityCapture")
    colmap_dir = os.path.join(output_dir, "COLMAP_Lichtfield")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    dataset_3dgs_dir = os.path.join(output_dir, "3dgs_dataset")
    os.makedirs(postshot_dir, exist_ok=True)
    os.makedirs(colmap_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dataset_3dgs_dir, exist_ok=True)

    # 1. Trajectory Prior Resolution
    traj_csv = prior_trajectory_csv
    if not traj_csv or not os.path.isfile(traj_csv):
        cand_csv, cand_run = find_prior_trajectory(images_dir)
        if cand_csv:
            traj_csv = cand_csv
            log(f"[*] Auto-detected Prior Trajectory from Run: {os.path.basename(cand_run)} ({traj_csv})")

    # If transforms.json exists but no cameras.csv, convert it
    if not traj_csv:
        for tf_cand in [
            os.path.join(images_dir, "transforms.json"),
            os.path.join(images_dir, "results", "transforms.json"),
            os.path.join(output_dir, "transforms.json")
        ]:
            if os.path.isfile(tf_cand):
                gen_csv = os.path.join(output_dir, "imported_prior_cameras.csv")
                if convert_transforms_to_realityscan_csv(tf_cand, gen_csv):
                    traj_csv = gen_csv
                    log(f"[*] Converted NeRF transforms.json -> Prior Trajectory: {traj_csv}")
                    break

    # 2. Build CLI Arguments
    cmd = [exe, "-stdConsole", "-printProgress"]

    # Add images folder
    cmd.extend(["-addFolder", os.path.abspath(images_dir)])

    # Trajectory Import (Screenshot 4)
    if traj_csv and os.path.isfile(traj_csv):
        # Generate trajectory import settings XML
        traj_xml = os.path.join(output_dir, "realityscan_trajectory_import.xml")
        generate_trajectory_import_xml(traj_xml)
        cmd.extend(["-importTrajectory", os.path.abspath(traj_csv), os.path.abspath(traj_xml)])
        log(f"[*] Ingesting Trajectory Prior: {os.path.basename(traj_csv)}")

        # Screenshot 2: Prior Pose & Accuracy
        cmd.extend([
            "-set", "sfmEnableCameraPrior=true",
            "-set", "sfmCameraPriorWeight=1.0",
            "-set", "sfmCameraPriorWeightOrientation=1.0",
        ])

    # Screenshot 3: Alignment Settings
    cmd.extend([
        "-set", f"sfmFeatureDetectionQuality={feature_detection_quality}",
        "-set", f"sfmMaxFeaturesPerMpx={max_features_per_mpx}",
        "-set", f"sfmMaxFeaturesPerImage={max_features_per_image}",
        "-set", f"sfmImagesOverlap={image_overlap}",
        "-set", f"sfmImageDownscaleFactor={image_downscale_factor}",
        "-set", f"sfmMaxFeatureReprojectionError={max_feature_reprojection_error:.6f}",
        "-set", f"sfmAutoReconRegionAfterAlignment={'true' if add_reconstruction_region else 'false'}",
        "-set", f"sfmForceComponentRematch={'true' if force_component_rematch else 'false'}",
        "-set", f"sfmPreselectorFeatures={preselector_features}",
        "-set", f"sfmDetectorSensitivity={detector_sensitivity}",
        "-set", f"sfmDistortionModel={distortion_model}",
    ])

    # Trigger Alignment
    cmd.append("-align")

    # Exports
    ply_export_path = os.path.join(postshot_dir, "points.ply")
    csv_export_path = os.path.join(postshot_dir, "cameras.csv")
    colmap_export_path = os.path.join(sparse_dir, "cameras.txt")

    if export_postshot:
        cmd.extend(["-exportSparsePointCloud", os.path.abspath(ply_export_path)])
        cmd.extend(["-exportRegistration", os.path.abspath(csv_export_path)])

    if export_colmap:
        cmd.extend(["-exportRegistration", os.path.abspath(colmap_export_path)])

    # Project Save
    project_path = os.path.join(output_dir, "realityscan_project.rsproj")
    cmd.extend(["-save", os.path.abspath(project_path)])

    # Headless vs GUI
    if headless:
        cmd.append("-quit")

    log(f"[*] Launching RealityScan...")
    start_time = time.time()

    # 3. Process Execution & Live Telemetry
    if headless:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )
        for line in proc.stdout:
            clean = line.strip()
            if clean:
                log(f"    [RealityScan] {clean}")

        proc.wait()
        elapsed = time.time() - start_time
        retcode = proc.returncode
        log(f"[*] RealityScan completed with return code {retcode} in {elapsed:.1f}s")
    else:
        # Launch detached interactive window
        subprocess.Popen(cmd)
        elapsed = time.time() - start_time
        log(f"[*] RealityScan window launched in separate interactive process.")
        log(f"[*] Project will save to: {project_path}")

    # 4. Standardize Exports for Abu Train & 3DGS
    dense_3d = np.empty((0, 3), dtype=np.float32)
    dense_rgb = np.empty((0, 3), dtype=np.uint8)

    # Read exported point cloud if available
    for ply_cand in [
        ply_export_path,
        os.path.join(output_dir, "points.ply"),
        os.path.join(output_dir, "points3D.ply"),
        os.path.join(sparse_dir, "points3D.ply")
    ]:
        if os.path.isfile(ply_cand) and os.path.getsize(ply_cand) > 100:
            try:
                import trimesh
                mesh = trimesh.load(ply_cand)
                dense_3d = np.asarray(mesh.vertices, dtype=np.float32)
                if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                    dense_rgb = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
                else:
                    dense_rgb = np.full((len(dense_3d), 3), 180, dtype=np.uint8)
                log(f"[*] Loaded {len(dense_3d):,} point cloud vertices from: {os.path.basename(ply_cand)}")

                # Distribute points.ply to standard destinations
                for dst_dir in [output_dir, dataset_3dgs_dir, colmap_dir]:
                    shutil.copy2(ply_cand, os.path.join(dst_dir, "points.ply"))
                    shutil.copy2(ply_cand, os.path.join(dst_dir, "points3D.ply"))
                break
            except Exception as e:
                log(f"[Warning] Failed to parse PLY {ply_cand}: {e}")

    # If prior cameras.csv was used and no new cameras.csv was exported, copy it
    if traj_csv and os.path.isfile(traj_csv) and not os.path.isfile(csv_export_path):
        try:
            shutil.copy2(traj_csv, csv_export_path)
            shutil.copy2(traj_csv, os.path.join(output_dir, "cameras.csv"))
        except Exception:
            pass

    # Link/stage images folder in 3dgs_dataset/ (File-based hard-link/copy, never directory junctions to prevent recursive loops)
    images_dst = os.path.join(dataset_3dgs_dir, "images")
    if not os.path.exists(images_dst):
        os.makedirs(images_dst, exist_ok=True)
        for f in os.listdir(images_dir):
            sp = os.path.join(images_dir, f)
            if os.path.isfile(sp) and any(f.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                try:
                    os.link(sp, os.path.join(images_dst, f))
                except Exception:
                    try:
                        shutil.copy2(sp, os.path.join(images_dst, f))
                    except Exception:
                        pass

    # Convert cameras.csv or sparse/0/ to transforms.json if not present
    tf_out = os.path.join(dataset_3dgs_dir, "transforms.json")
    if not os.path.isfile(tf_out) and os.path.isfile(csv_export_path):
        from fast_viewer import load_cameras_from_cameras_csv
        cams = load_cameras_from_cameras_csv(csv_export_path)
        if cams:
            tf_data = {
                "camera_model": "OPENCV",
                "fl_x": cams[0].get("fx", 1000.0),
                "fl_y": cams[0].get("fy", 1000.0),
                "cx": cams[0].get("cx", 960.0),
                "cy": cams[0].get("cy", 540.0),
                "w": 1920,
                "h": 1080,
                "frames": []
            }
            from scipy.spatial.transform import Rotation
            for c in cams:
                pos = np.asarray(c["pos"], dtype=float)
                wxyz = np.asarray(c["wxyz"], dtype=float)
                rot_mat = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
                # RealityCapture to NeRF OpenGL transformation
                M_rc2nerf = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]
                ])
                R_nerf = M_rc2nerf @ rot_mat @ np.diag([1.0, -1.0, -1.0])
                p_nerf = M_rc2nerf @ pos
                c2w = np.eye(4)
                c2w[:3, :3] = R_nerf
                c2w[:3, 3] = p_nerf
                tf_data["frames"].append({
                    "file_path": f"images/{c['name']}",
                    "transform_matrix": c2w.tolist()
                })
            with open(tf_out, "w", encoding="utf-8") as f:
                json.dump(tf_data, f, indent=2)
            shutil.copy2(tf_out, os.path.join(output_dir, "transforms.json"))
            shutil.copy2(tf_out, os.path.join(postshot_dir, "transforms.json"))
            log(f"[*] Generated 3DGS/NeRF transforms.json ({len(cams)} views)")

    # Attempt to load pycolmap reconstruction if COLMAP files exist
    reconstruction = None
    try:
        import pycolmap
        for cand_colmap in [sparse_dir, colmap_dir]:
            if any(os.path.isfile(os.path.join(cand_colmap, f)) for f in ["cameras.txt", "cameras.bin"]):
                try:
                    reconstruction = pycolmap.Reconstruction(cand_colmap)
                    break
                except Exception:
                    pass
    except Exception:
        pass

    return reconstruction, dense_3d, dense_rgb
