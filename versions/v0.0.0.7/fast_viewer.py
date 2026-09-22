"""
Fast 3D Browser Viewer for Abu Align
Instantly loads pre-computed 3D point cloud (.ply) and camera alignments into Viser.
"""
import os
import sys
import time
import argparse
import webbrowser
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Fast 3D Browser Viewer for Photogrammetry")
    parser.add_argument("--results_dir", "-r", type=str, default="", help="Directory containing reconstruction results")
    parser.add_argument("--input", "-i", type=str, default="", help="Dataset or results directory")
    parser.add_argument("path", nargs="?", default="", help="Optional positional directory path")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    target_dir = args.results_dir or args.input or args.path or os.getcwd()
    target_dir = os.path.abspath(target_dir)

    ply_path = None
    csv_path = None

    # Search for points.ply and cameras.csv in target_dir and subdirectories
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower() == "points.ply" and not ply_path:
                ply_path = os.path.join(root, f)
            elif f.lower() == "cameras.csv" and not csv_path:
                csv_path = os.path.join(root, f)

    if not ply_path or not os.path.isfile(ply_path):
        # Check if there are images to run live demo_viser
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        images = [
            os.path.join(target_dir, f) for f in os.listdir(target_dir)
            if os.path.splitext(f.lower())[1] in valid_exts
        ] if os.path.isdir(target_dir) else []

        if images:
            print(f"[INFO] No precomputed points.ply found in {target_dir}.")
            print(f"Launching live VGGT inference viewer with {len(images)} photos...")
            import subprocess
            demo_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_viser.py")
            cmd = [sys.executable, demo_script, "--image_folder", target_dir, "--port", str(args.port)]
            subprocess.run(cmd)
            return

        print(f"[ERROR] No points.ply or image files found in: {target_dir}")
        print("Please run Abu Align reconstruction first or select a valid dataset folder.")
        time.sleep(3)
        sys.exit(1)

    print(f"Loading 3D Point Cloud from: {ply_path}...")
    import trimesh
    mesh = trimesh.load(ply_path)
    points = np.array(mesh.vertices, dtype=np.float32)
    colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8) if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None else None

    import viser
    print(f"Starting Viser Server on http://localhost:{args.port}...")
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Add point cloud
    server.scene.add_point_cloud(
        name="/reconstruction/points",
        points=points,
        colors=colors,
        point_size=0.006,
    )
    print(f"Loaded {len(points):,} 3D points.")

    # Load cameras if cameras.csv exists
    if csv_path and os.path.isfile(csv_path):
        print(f"Loading cameras from: {csv_path}...")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    name = parts[0]
                    x, y, alt = float(parts[1]), float(parts[2]), float(parts[3])
                    yaw, pitch, roll = float(parts[4]), float(parts[5]), float(parts[6])

                    # Direction vector from yaw, pitch
                    yaw_rad = np.radians(yaw)
                    pitch_rad = np.radians(pitch)
                    fwd = np.array([
                        np.cos(pitch_rad) * np.sin(yaw_rad),
                        np.cos(pitch_rad) * np.cos(yaw_rad),
                        np.sin(pitch_rad)
                    ])
                    # Add camera frustum
                    server.scene.add_camera_frustum(
                        name=f"/cameras/{name}",
                        fov=np.radians(60.0),
                        aspect=1.5,
                        scale=0.08,
                        position=np.array([x, y, alt]),
                    )
        except Exception as e:
            print(f"[WARNING] Could not parse cameras: {e}")

    try:
        webbrowser.open(f"http://localhost:{args.port}")
    except Exception:
        pass

    print(f"\n[READY] Interactive 3D Scene running at: http://localhost:{args.port}")
    print("Press Ctrl+C to stop viewer server.\n")

    while True:
        time.sleep(0.05)

if __name__ == "__main__":
    main()
