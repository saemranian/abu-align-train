"""
Fast 3D Browser Viewer for Abu Align & Train
Interactive multi-run 3D visualizer powered by Viser.
Supports 360DVO tie points, COLMAP sparse point clouds, camera trajectories,
and 3D Gaussian Splats with real-time run switching.
"""
import os
import sys
import time
import glob
import json
import argparse
import webbrowser
import numpy as np

def load_ply_points_and_colors(ply_path: str):
    """Loads 3D points and optional RGB colors from a PLY file."""
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
        print(f"[Warning] Failed to load PLY {ply_path}: {e}")
        return None, None


def load_cameras_from_tum(traj_path: str):
    """Loads camera poses from a TUM format trajectory file: timestamp tx ty tz qx qy qz qw."""
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


def load_cameras_from_transforms(tf_path: str):
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

            # Convert R to quaternion wxyz
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


def discover_runs(target_dir: str):
    """
    Finds all reconstruction run directories inside target_dir or results/.
    Returns a dict mapping run_label -> directory_path, sorted by newest first.
    """
    runs = {}
    base_results = os.path.join(target_dir, "results")
    search_dirs = [base_results, target_dir] if os.path.isdir(base_results) else [target_dir]

    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for entry in os.listdir(sdir):
            p = os.path.join(sdir, entry)
            if os.path.isdir(p) and ("sparse" in entry.lower() or "dvo" in entry.lower() or "vggt" in entry.lower() or "opensfm" in entry.lower()):
                runs[entry] = p

    # If target_dir itself has point clouds, add it
    has_direct_ply = any(f.endswith(".ply") for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f)))
    if has_direct_ply:
        runs[os.path.basename(target_dir)] = target_dir

    # Sort runs by directory modification time descending (latest run first)
    sorted_runs = dict(sorted(runs.items(), key=lambda item: os.path.getmtime(item[1]), reverse=True))
    return sorted_runs


def find_best_model_in_run(run_dir: str):
    """Finds best point cloud / splat file and camera trajectory in run_dir."""
    ply_files = glob.glob(os.path.join(run_dir, "*.ply")) + glob.glob(os.path.join(run_dir, "**", "*.ply"), recursive=True)

    def score_ply(path):
        name = os.path.basename(path).lower()
        score = 0
        if "splat" in name:
            score = 50
        elif "360dvo_points" in name:
            score = 40
        elif name == "points.ply":
            score = 30
        elif name == "points3d.ply":
            score = 20
        return (score, os.path.getmtime(path))

    best_ply = max(ply_files, key=score_ply) if ply_files else None

    # Check for trajectory files
    tum_traj = os.path.join(run_dir, "360dvo_trajectory.txt")
    tf_json = os.path.join(run_dir, "transforms.json")
    if not os.path.isfile(tf_json):
        tf_json = os.path.join(run_dir, "3dgs_dataset", "transforms.json")

    return best_ply, tum_traj if os.path.isfile(tum_traj) else None, tf_json if os.path.isfile(tf_json) else None


def main():
    parser = argparse.ArgumentParser(description="Fast 3D Browser Viewer for Photogrammetry & 3DGS")
    parser.add_argument("--results_dir", "-r", type=str, default="", help="Directory containing reconstruction results")
    parser.add_argument("--input", "-i", type=str, default="", help="Dataset or results directory")
    parser.add_argument("path", nargs="?", default="", help="Optional positional directory path")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    target_dir = args.results_dir or args.input or args.path or os.getcwd()
    target_dir = os.path.abspath(target_dir)

    print("=" * 70)
    print("   Abu Align & Train - Interactive 3D Browser Viewer (Viser)")
    print(f"   * Target Directory: {target_dir}")
    print(f"   * Server Port:      {args.port}")
    print("=" * 70)

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

    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Discover all runs
    runs = discover_runs(target_dir)
    if not runs:
        runs[os.path.basename(target_dir)] = target_dir

    run_names = list(runs.keys())
    current_run_name = run_names[0]

    status_markdown = server.gui.add_markdown(content="**Status**: Initializing 3D viewer...")
    run_dropdown = server.gui.add_dropdown("Reconstruction Run", options=run_names, initial_value=current_run_name)
    point_size_slider = server.gui.add_slider("Point Size", min=0.001, max=0.05, step=0.001, initial_value=0.006)
    frustum_scale_slider = server.gui.add_slider("Camera Frustum Size", min=0.01, max=0.5, step=0.01, initial_value=0.08)

    current_camera_handles = []

    def render_run(run_name: str):
        nonlocal current_camera_handles
        run_path = runs.get(run_name, target_dir)
        best_ply, tum_traj, tf_json = find_best_model_in_run(run_path)

        # Clear previous cameras
        for handle in current_camera_handles:
            try:
                handle.remove()
            except Exception:
                pass
        current_camera_handles.clear()

        # Load point cloud
        if best_ply and os.path.isfile(best_ply):
            pts, cols = load_ply_points_and_colors(best_ply)
            if pts is not None and len(pts) > 0:
                server.scene.add_point_cloud(
                    name="/reconstruction/points",
                    points=pts,
                    colors=cols,
                    point_size=point_size_slider.value,
                )
                ply_label = os.path.basename(best_ply)
                print(f"[Viewer] Loaded {len(pts):,} points from {ply_label} in {run_name}")
            else:
                pts = []
        else:
            pts = []
            ply_label = "None found"

        # Load cameras
        cameras = []
        if tum_traj:
            cameras = load_cameras_from_tum(tum_traj)
        elif tf_json:
            cameras = load_cameras_from_transforms(tf_json)

        for cam in cameras:
            handle = server.scene.add_camera_frustum(
                name=f"/cameras/{cam['name']}",
                fov=np.radians(60.0),
                aspect=1.5,
                scale=frustum_scale_slider.value,
                wxyz=cam["wxyz"],
                position=cam["pos"]
            )
            current_camera_handles.append(handle)

        status_markdown.content = (
            f"### Active Reconstruction: `{run_name}`\n"
            f"- **Model**: `{ply_label}` ({len(pts):,} points)\n"
            f"- **Cameras**: {len(cameras)} tracked poses\n"
            f"- **Path**: `{run_path}`"
        )

    # Initial render of newest run
    render_run(current_run_name)

    # Setup callbacks
    @run_dropdown.on_update
    def _(_) -> None:
        render_run(run_dropdown.value)

    @point_size_slider.on_update
    def _(_) -> None:
        render_run(run_dropdown.value)

    @frustum_scale_slider.on_update
    def _(_) -> None:
        render_run(run_dropdown.value)

    server.gui.add_button("Reset View").on_click(lambda _: server.scene.reset())

    url = f"http://localhost:{active_port}"
    print(f"\n[READY] Interactive 3D Viewer running at: {url}")
    print(f"Viewing latest run: {current_run_name}")
    print("Press Ctrl+C to close viewer.\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
