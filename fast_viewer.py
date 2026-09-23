"""
Abu Align & Train - Interactive 3D Browser Viewer (Viser)
Inspection suite for 3D point clouds, camera trajectories, and 3D Gaussian Splats.
Features multi-run discovery, automatic latest-run tracking, dataset provenance cards,
trajectory path rendering, and adaptive outlier clipping.
"""
import os
import sys
import time
import glob
import json
import math
import argparse
import webbrowser
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Optional

# =========================================================================
# Robust Point Cloud & Camera Parsers
# =========================================================================

def read_ply_data(ply_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    High-performance PLY reader supporting ASCII and Binary Little-Endian,
    with automatic fallbacks (numpy buffer -> plyfile -> trimesh -> ASCII).
    """
    if not os.path.isfile(ply_path):
        return None, None

    # Fast binary numpy reader
    try:
        with open(ply_path, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header.append(line)
                if line == 'end_header':
                    break

            format_line = [h for h in header if h.startswith('format')][0]
            element_line = [h for h in header if h.startswith('element vertex')][0]
            num_verts = int(element_line.split()[-1])

            if 'binary_little_endian' in format_line:
                props = [h.split() for h in header if h.startswith('property')]
                dt = []
                for pr in props:
                    t = pr[1]
                    name = pr[2]
                    if t in ['float', 'float32']: dt.append((name, '<f4'))
                    elif t in ['uchar', 'uint8']: dt.append((name, 'u1'))
                    elif t in ['int', 'int32']: dt.append((name, '<i4'))
                    elif t in ['double', 'float64']: dt.append((name, '<f8'))
                    else: dt.append((name, '<f4'))

                buf = f.read()
                data = np.frombuffer(buf, dtype=np.dtype(dt), count=num_verts)
                pts = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float32)

                if 'red' in data.dtype.names and 'green' in data.dtype.names and 'blue' in data.dtype.names:
                    cols = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.uint8)
                else:
                    cols = np.full((len(pts), 3), 180, dtype=np.uint8)
                return pts, cols

            elif 'ascii' in format_line:
                pts = []
                cols = []
                for _ in range(num_verts):
                    parts = f.readline().decode('utf-8', errors='ignore').strip().split()
                    if len(parts) >= 3:
                        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
                        else:
                            cols.append([180, 180, 180])
                return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)
    except Exception:
        pass

    # Fallback to plyfile or trimesh
    try:
        from plyfile import PlyData
        pdata = PlyData.read(ply_path)
        v = pdata['vertex']
        pts = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
        if 'red' in v and 'green' in v and 'blue' in v:
            cols = np.stack([v['red'], v['green'], v['blue']], axis=1).astype(np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)
        return pts, cols
    except Exception:
        pass

    try:
        import trimesh
        mesh = trimesh.load(ply_path)
        pts = np.array(mesh.vertices, dtype=np.float32)
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            cols = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
        else:
            cols = np.full((len(pts), 3), 180, dtype=np.uint8)
        return pts, cols
    except Exception as e:
        print(f"[Warning] Failed to read PLY {ply_path}: {e}")
        return None, None


def load_cameras_from_tum(traj_path: str) -> List[Dict]:
    """Loads camera poses from robotics TUM trajectory file: timestamp tx ty tz qx qy qz qw."""
    poses = []
    if not os.path.isfile(traj_path):
        return poses
    try:
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 8:
                    ts = parts[0]
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    poses.append({
                        "name": f"cam_{ts}",
                        "pos": np.array([tx, ty, tz], dtype=np.float32),
                        "wxyz": np.array([qw, qx, qy, qz], dtype=np.float32)
                    })
    except Exception as e:
        print(f"[Warning] Failed to parse TUM trajectory {traj_path}: {e}")
    return poses


def load_cameras_from_colmap_images(images_txt_path: str) -> List[Dict]:
    """Loads camera poses from standard COLMAP images.txt format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME."""
    poses = []
    if not os.path.isfile(images_txt_path):
        return poses
    try:
        with open(images_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    qw = float(parts[1])
                    qx = float(parts[2])
                    qy = float(parts[3])
                    qz = float(parts[4])
                    tx = float(parts[5])
                    ty = float(parts[6])
                    tz = float(parts[7])
                    name = parts[9]

                    # Convert w2c to c2w
                    R_w2c = np.array([
                        [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),       2.0*(qx*qz + qy*qw)],
                        [2.0*(qx*qy + qz*qw),       1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
                        [2.0*(qx*qz - qy*qw),       2.0*(qy*qz + qx*qw),       1.0 - 2.0*(qx*qx + qy*qy)]
                    ], dtype=np.float64)
                    t_w2c = np.array([tx, ty, tz], dtype=np.float64)

                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c

                    # R_c2w to quaternion wxyz
                    tr = np.trace(R_c2w)
                    if tr > 0:
                        S = math.sqrt(tr + 1.0) * 2.0
                        qw_c = 0.25 * S
                        qx_c = (R_c2w[2, 1] - R_c2w[1, 2]) / S
                        qy_c = (R_c2w[0, 2] - R_c2w[2, 0]) / S
                        qz_c = (R_c2w[1, 0] - R_c2w[0, 1]) / S
                    else:
                        qw_c, qx_c, qy_c, qz_c = 1.0, 0.0, 0.0, 0.0

                    poses.append({
                        "name": os.path.basename(name),
                        "pos": t_c2w.astype(np.float32),
                        "wxyz": np.array([qw_c, qx_c, qy_c, qz_c], dtype=np.float32)
                    })
    except Exception as e:
        print(f"[Warning] Failed to parse COLMAP images.txt {images_txt_path}: {e}")
    return poses


def load_cameras_from_transforms(tf_path: str) -> List[Dict]:
    """Loads camera poses from a transforms.json file."""
    poses = []
    if not os.path.isfile(tf_path):
        return poses
    try:
        with open(tf_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        for idx, frame in enumerate(frames):
            c2w = np.array(frame.get("transform_matrix", np.eye(4)), dtype=np.float32)
            R = c2w[:3, :3]
            t = c2w[:3, 3]

            tr = np.trace(R)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            else:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            name = os.path.basename(frame.get("file_path", f"frame_{idx:04d}"))
            poses.append({
                "name": name,
                "pos": t,
                "wxyz": np.array([qw, qx, qy, qz], dtype=np.float32)
            })
    except Exception as e:
        print(f"[Warning] Failed to parse transforms.json {tf_path}: {e}")
    return poses


def parse_run_report(run_dir: str) -> Dict[str, str]:
    """Extracts metadata from reconstruction_report.txt if available."""
    meta = {
        "engine": "Unknown",
        "date": "Unknown",
        "dataset": "",
        "photos": "0",
        "registered": "0",
        "points": "0"
    }
    report_path = os.path.join(run_dir, "reconstruction_report.txt")
    if not os.path.isfile(report_path):
        # Infer engine from run dir name
        name = os.path.basename(run_dir).lower()
        if "360dvo" in name: meta["engine"] = "360DVO (Deep Visual Odometry)"
        elif "opensfm" in name: meta["engine"] = "OpenSfM (Equirectangular SfM)"
        elif "vggt" in name: meta["engine"] = "VGGT-Omega (Feedforward)"
        elif "ace0" in name: meta["engine"] = "ACE0 (Visual Relocalization)"
        elif "pycusfm" in name: meta["engine"] = "NVIDIA PyCuSFM (GPU SfM)"
        return meta

    try:
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip().lstrip("*").strip()
                if "date & time:" in line.lower():
                    meta["date"] = line.split(":", 1)[1].strip()
                elif "photogrammetry engine:" in line.lower():
                    meta["engine"] = line.split(":", 1)[1].strip()
                elif "dataset source:" in line.lower():
                    meta["dataset"] = line.split(":", 1)[1].strip()
                elif "total photos:" in line.lower():
                    meta["photos"] = line.split(":", 1)[1].strip()
                elif "registered views:" in line.lower():
                    meta["registered"] = line.split(":", 1)[1].strip()
                elif "3d points created:" in line.lower():
                    meta["points"] = line.split(":", 1)[1].strip()
    except Exception:
        pass

    if meta["engine"] == "Unknown":
        name = os.path.basename(run_dir).lower()
        if "360dvo" in name: meta["engine"] = "360DVO (Deep Visual Odometry)"
        elif "opensfm" in name: meta["engine"] = "OpenSfM (Equirectangular SfM)"
        elif "vggt" in name: meta["engine"] = "VGGT-Omega (Feedforward)"
        elif "ace0" in name: meta["engine"] = "ACE0 (Visual Relocalization)"
        elif "pycusfm" in name: meta["engine"] = "NVIDIA PyCuSFM (GPU SfM)"
    return meta


# =========================================================================
# Run Discovery & Workspace Resolution
# =========================================================================

def resolve_project_structure(target_dir: str) -> Tuple[str, str, Dict[str, str]]:
    """
    Resolves project name, dataset root path, and dictionary of all runs in results/.
    Returns (project_title, dataset_dir, runs_dict).
    """
    target_dir = os.path.abspath(target_dir)

    # Detect if target_dir is a run folder inside results/
    parent = os.path.dirname(target_dir)
    grandparent = os.path.dirname(parent)

    if os.path.basename(parent).lower() == "results":
        dataset_dir = grandparent
        results_dir = parent
    elif os.path.isdir(os.path.join(target_dir, "results")):
        dataset_dir = target_dir
        results_dir = os.path.join(target_dir, "results")
    else:
        dataset_dir = target_dir
        results_dir = target_dir

    # Derive human-friendly project name from directory hierarchy
    parts = [p for p in dataset_dir.replace("\\", "/").split("/") if p]
    if len(parts) >= 2:
        project_title = f"{parts[-2]} / {parts[-1]}"
    elif parts:
        project_title = parts[-1]
    else:
        project_title = "Active Project"

    # Discover runs in results_dir
    runs = {}
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            p = os.path.join(results_dir, entry)
            if os.path.isdir(p) and any(tag in entry.lower() for tag in ["sparse", "dvo", "vggt", "opensfm", "ace", "cusfm"]):
                runs[entry] = p

    # If no results subfolder, check if target_dir itself contains reconstruction data
    if not runs:
        has_direct_ply = any(f.endswith(".ply") for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)))
        if has_direct_ply:
            runs[os.path.basename(target_dir)] = target_dir

    # Sort runs by directory modification time descending (latest first)
    sorted_runs = dict(sorted(runs.items(), key=lambda item: os.path.getmtime(item[1]), reverse=True))
    return project_title, dataset_dir, sorted_runs


def find_run_assets(run_dir: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Finds best point cloud (.ply), TUM trajectory, COLMAP images.txt, and transforms.json in run_dir."""
    ply_files = glob.glob(os.path.join(run_dir, "*.ply")) + glob.glob(os.path.join(run_dir, "**", "*.ply"), recursive=True)

    def score_ply(path):
        name = os.path.basename(path).lower()
        score = 0
        if "splat" in name: score = 50
        elif "360dvo_points" in name: score = 40
        elif name == "points.ply": score = 30
        elif name == "points3d.ply": score = 20
        return (score, os.path.getmtime(path))

    best_ply = max(ply_files, key=score_ply) if ply_files else None

    # Camera sources
    tum_traj = os.path.join(run_dir, "360dvo_trajectory.txt")
    tum_traj = tum_traj if os.path.isfile(tum_traj) else None

    colmap_images = None
    colmap_cands = [
        os.path.join(run_dir, "COLMAP_Lichtfield", "images.txt"),
        os.path.join(run_dir, "3dgs_dataset", "sparse", "0", "images.txt"),
        os.path.join(run_dir, "sparse", "0", "images.txt"),
    ]
    for c in colmap_cands:
        if os.path.isfile(c):
            colmap_images = c
            break

    tf_json = None
    tf_cands = [
        os.path.join(run_dir, "transforms.json"),
        os.path.join(run_dir, "3dgs_dataset", "transforms.json"),
    ]
    for tf in tf_cands:
        if os.path.isfile(tf):
            tf_json = tf
            break

    return best_ply, tum_traj, colmap_images, tf_json


# =========================================================================
# Main Interactive Viser Viewer
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Abu Align & Train - Interactive 3D Browser Viewer")
    parser.add_argument("--results_dir", "-r", type=str, default="", help="Directory containing reconstruction results")
    parser.add_argument("--input", "-i", type=str, default="", help="Dataset or results directory")
    parser.add_argument("path", nargs="?", default="", help="Optional positional directory path")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    target_dir = args.results_dir or args.input or args.path or os.getcwd()
    target_dir = os.path.abspath(target_dir)

    project_title, dataset_dir, runs = resolve_project_structure(target_dir)

    print("=" * 72)
    print("   Abu Align & Train - Interactive 3D Inspection Viewer (Viser)")
    print(f"   * Project Name:     {project_title}")
    print(f"   * Dataset Root:     {dataset_dir}")
    print(f"   * Discovered Runs:  {len(runs)}")
    print(f"   * Preferred Port:   {args.port}")
    print("=" * 72)

    import viser

    server = None
    active_port = args.port
    for offset in range(10):
        try:
            port_to_try = active_port + offset
            server = viser.ViserServer(host="0.0.0.0", port=port_to_try)
            active_port = port_to_try
            break
        except Exception:
            pass

    if server is None:
        print("[ERROR] Could not start Viser server on any port 8080-8089.")
        sys.exit(1)

    # Configure sleek fixed dark theme matching Abu Align & Train GUI
    server.gui.configure_theme(
        titlebar_content=None,
        control_layout="fixed",
        control_width="medium",
        dark_mode=True,
        brand_color=(16, 185, 129),  # Emerald green (#10b981)
        show_logo=True,
        show_share_button=False
    )

    if not runs:
        runs[os.path.basename(target_dir)] = target_dir

    run_names = list(runs.keys())
    current_run_name = run_names[0]

    # --- TOP GUI: Project & Provenance Card ---
    server.gui.add_markdown(f"## 🌐 Abu Align 3D Inspector\n**Project**: `{project_title}`")

    # Run selection dropdown
    run_dropdown = server.gui.add_dropdown(
        "Reconstruction Run",
        options=run_names,
        initial_value=current_run_name
    )

    # Dynamic run info card
    info_card = server.gui.add_markdown(content="*Inspecting run...*")

    # Action Toolbar
    refresh_btn = server.gui.add_button("🔄 Refresh Runs")
    focus_btn = server.gui.add_button("🎯 Focus View on Trajectory")
    folder_btn = server.gui.add_button("📂 Open Run Folder in Explorer")

    # --- VIEW CONTROLS FOLDER ---
    with server.gui.add_folder("Visual Aids & Toggles"):
        show_traj_check = server.gui.add_checkbox("Show Camera Trajectory Path", initial_value=True)
        show_cams_check = server.gui.add_checkbox("Show Camera Frustums", initial_value=True)
        filter_outliers_check = server.gui.add_checkbox("Filter Extreme Outliers (<30m)", initial_value=True)
        point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.006)
        frustum_scale_slider = server.gui.add_slider("Camera Frustum Scale", min=0.01, max=0.3, step=0.01, initial_value=0.05)
        color_mode_dropdown = server.gui.add_dropdown(
            "Color Mode",
            options=["Original RGB", "Height Ramp (Z-axis)", "Distance Ramp", "Solid Emerald"],
            initial_value="Original RGB"
        )

    # State tracking
    current_camera_handles = []
    current_traj_handles = []
    current_pcd_handle = None
    latest_camera_center = np.zeros(3, dtype=np.float32)

    def load_and_render_run(run_name: str):
        nonlocal current_camera_handles, current_traj_handles, current_pcd_handle, latest_camera_center

        run_path = runs.get(run_name, target_dir)
        best_ply, tum_traj, colmap_images, tf_json = find_run_assets(run_path)
        report_meta = parse_run_report(run_path)

        # Clear previous cameras & trajectory
        for h in current_camera_handles:
            try: h.remove()
            except Exception: pass
        current_camera_handles.clear()

        for h in current_traj_handles:
            try: h.remove()
            except Exception: pass
        current_traj_handles.clear()

        # 1. Load Camera Poses (Priority: TUM odometry -> COLMAP images.txt -> transforms.json)
        cameras = []
        cam_source_label = "None"
        if tum_traj:
            cameras = load_cameras_from_tum(tum_traj)
            cam_source_label = "360DVO Trajectory (TUM)"
        elif colmap_images:
            cameras = load_cameras_from_colmap_images(colmap_images)
            cam_source_label = "COLMAP images.txt"
        elif tf_json:
            cameras = load_cameras_from_transforms(tf_json)
            cam_source_label = "transforms.json"

        # Calculate camera trajectory center
        if cameras:
            cam_positions = np.array([c["pos"] for c in cameras], dtype=np.float32)
            latest_camera_center = np.mean(cam_positions, axis=0)

            # Render camera trajectory as smooth 3D Catmull-Rom spline with color gradient
            if show_traj_check.value and len(cam_positions) >= 2:
                try:
                    # Draw continuous spline tube connecting camera positions
                    spline_handle = server.scene.add_spline_catmull_rom(
                        name="/camera_trajectory/path",
                        positions=cam_positions,
                        line_width=2.5,
                        color=(16, 185, 129),  # Emerald line
                    )
                    current_traj_handles.append(spline_handle)
                except Exception:
                    pass

            # Render frustums
            if show_cams_check.value:
                # If there are many cameras (>150), sample keyframe frustums to keep viewport super fluid
                step = 1 if len(cameras) <= 100 else 2
                for i in range(0, len(cameras), step):
                    cam = cameras[i]
                    # Color code: first camera green, last camera red
                    pct = float(i) / max(1, len(cameras) - 1)
                    r = int(pct * 255)
                    g = int((1.0 - pct) * 255)
                    b = 100

                    try:
                        handle = server.scene.add_camera_frustum(
                            name=f"/cameras/{cam['name']}",
                            fov=np.radians(60.0),
                            aspect=1.5,
                            scale=frustum_scale_slider.value,
                            wxyz=cam["wxyz"],
                            position=cam["pos"],
                            color=(r, g, b)
                        )
                        current_camera_handles.append(handle)
                    except Exception:
                        pass

        # 2. Load Point Cloud
        pts, cols = (None, None)
        ply_label = "None found"
        total_pts = 0
        inlier_pts = 0

        if best_ply and os.path.isfile(best_ply):
            pts, cols = read_ply_data(best_ply)
            ply_label = os.path.basename(best_ply)

        if pts is not None and len(pts) > 0:
            total_pts = len(pts)

            # Filter distant outliers if enabled
            if filter_outliers_check.value and len(pts) > 100:
                ref_center = latest_camera_center if len(cameras) > 0 else np.median(pts, axis=0)
                dists = np.linalg.norm(pts - ref_center, axis=1)

                # Use adaptive 90th percentile * 2.0 or 35m max
                dist_cutoff = min(35.0, max(8.0, float(np.percentile(dists, 90) * 1.8)))
                valid_mask = dists < dist_cutoff
                pts_render = pts[valid_mask]
                cols_render = cols[valid_mask] if cols is not None else None
                inlier_pts = len(pts_render)
            else:
                pts_render = pts
                cols_render = cols
                inlier_pts = total_pts

            # Apply Color Modes
            mode = color_mode_dropdown.value
            if mode == "Height Ramp (Z-axis)":
                zs = pts_render[:, 2]
                z_min, z_max = zs.min(), zs.max()
                z_norm = (zs - z_min) / max(1e-6, z_max - z_min)
                cols_render = np.zeros((len(pts_render), 3), dtype=np.uint8)
                cols_render[:, 0] = (z_norm * 255).astype(np.uint8)
                cols_render[:, 1] = ((1.0 - z_norm) * 200).astype(np.uint8)
                cols_render[:, 2] = 220
            elif mode == "Distance Ramp":
                ref_center = latest_camera_center if len(cameras) > 0 else np.median(pts_render, axis=0)
                d = np.linalg.norm(pts_render - ref_center, axis=1)
                d_norm = d / max(1e-6, d.max())
                cols_render = np.zeros((len(pts_render), 3), dtype=np.uint8)
                cols_render[:, 0] = (d_norm * 255).astype(np.uint8)
                cols_render[:, 1] = 120
                cols_render[:, 2] = ((1.0 - d_norm) * 255).astype(np.uint8)
            elif mode == "Solid Emerald":
                cols_render = np.full((len(pts_render), 3), (16, 185, 129), dtype=np.uint8)

            current_pcd_handle = server.scene.add_point_cloud(
                name="/reconstruction/points",
                points=pts_render,
                colors=cols_render if cols_render is not None else np.full((len(pts_render), 3), 180, dtype=np.uint8),
                point_size=point_size_slider.value,
            )
            print(f"[Viewer] Rendered {inlier_pts:,} points from {ply_label} ({run_name})")
        else:
            try:
                server.scene.remove("/reconstruction/points")
            except Exception:
                pass

        # 3. Update Provenance & Metadata Info Card
        eng = report_meta.get("engine", "Auto-Detected")
        date_str = report_meta.get("date", "Recent Alignment")
        tag_status = "🟢 Current Latest Run" if run_name == run_names[0] else "📁 Previous Run"

        info_card.content = (
            f"**Status**: {tag_status}\n\n"
            f"| Attribute | Value |\n"
            f"| :--- | :--- |\n"
            f"| **Engine** | `{eng}` |\n"
            f"| **Timestamp** | `{date_str}` |\n"
            f"| **Cameras** | **{len(cameras)} tracked** ({cam_source_label}) |\n"
            f"| **3D Points** | **{inlier_pts:,}** / {total_pts:,} vertices ({ply_label}) |\n"
            f"| **Folder** | `{os.path.basename(run_path)}` |\n\n"
            f"<small style='color:#94a3b8;'>Source: {run_path}</small>"
        )

    # Initial Render
    load_and_render_run(current_run_name)

    # Setup Dynamic Callbacks
    @run_dropdown.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @point_size_slider.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @frustum_scale_slider.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @show_traj_check.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @show_cams_check.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @filter_outliers_check.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @color_mode_dropdown.on_update
    def _(_) -> None:
        load_and_render_run(run_dropdown.value)

    @refresh_btn.on_click
    def _(_) -> None:
        nonlocal runs, run_names
        _, _, new_runs = resolve_project_structure(target_dir)
        if new_runs:
            runs = new_runs
            run_names = list(runs.keys())
            run_dropdown.options = run_names
            run_dropdown.value = run_names[0]
            load_and_render_run(run_names[0])
            print(f"[Viewer] Refreshed runs: {len(runs)} found. Loaded latest: {run_names[0]}")

    @focus_btn.on_click
    def _(_) -> None:
        server.scene.reset()
        print("[Viewer] Camera view reset to center.")

    @folder_btn.on_click
    def _(_) -> None:
        active_path = runs.get(run_dropdown.value, target_dir)
        if os.path.isdir(active_path):
            if sys.platform == "win32":
                os.startfile(active_path)
            else:
                subprocess.Popen(["xdg-open", active_path])

    url = f"http://localhost:{active_port}"
    print(f"\n[READY] Interactive 3D Viewer running at: {url}")
    print(f"Viewing active run: {current_run_name}")
    print("Press Ctrl+C to stop viewer.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
