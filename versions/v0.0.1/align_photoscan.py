"""
VGGT Photoscanning Alignment Helper
===================================
A user-friendly CLI to align multi-view photoscanning datasets into standard COLMAP format
and visualize them interactively in 3D.
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Align photoscanning image dataset using VGGT")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to folder containing photoscan images")
    parser.add_argument("--use_ba", action="store_true", help="Enable Bundle Adjustment refinement (more accurate, uses tracker + Ceres BA)")
    parser.add_argument("--shared_camera", action="store_true", help="Assume all photos were taken with the same camera (recommended for single-camera scans)")
    parser.add_argument("--stride", type=int, default=1, help="Stride to subsample input images (e.g. 2 or 3 for large datasets)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory for sparse COLMAP files")
    parser.add_argument("--view", "-v", action="store_true", help="Launch interactive 3D browser viewer (Viser) after alignment")
    parser.add_argument("--port", type=int, default=8080, help="Viser browser viewer port (default: 8080)")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Determine image folder
    images_subdir = os.path.join(input_dir, "images")
    images_dir = images_subdir if os.path.isdir(images_subdir) else input_dir

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_files:
        print(f"Error: No image files found in {images_dir}")
        sys.exit(1)

    effective_count = len(image_files[::args.stride])
    if args.max_frames:
        effective_count = min(effective_count, args.max_frames)

    print("=" * 60)
    print(f"VGGT Photoscanning Alignment")
    print(f"Target Directory: {input_dir}")
    print(f"Images Available: {len(image_files)}")
    print(f"Stride:           {args.stride} (Processing {effective_count} keyframes)")
    print(f"Mode:             {'Bundle Adjustment (VGGT+BA)' if args.use_ba else 'Fast Feed-Forward'}")
    print(f"Shared Camera:    {args.shared_camera}")
    print("=" * 60)

    # Build demo_colmap command
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_colmap_path = os.path.join(script_dir, "demo_colmap.py")

    cmd = [
        sys.executable,
        demo_colmap_path,
        "--scene_dir", input_dir,
        "--stride", str(args.stride),
    ]
    if args.use_ba:
        cmd.append("--use_ba")
    if args.shared_camera:
        cmd.append("--shared_camera")
    if args.max_frames:
        cmd.extend(["--max_frames", str(args.max_frames)])
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"\nAlignment failed with return code {ret.returncode}")
        sys.exit(ret.returncode)

    sparse_dir = args.output_dir if args.output_dir else os.path.join(input_dir, "sparse")
    ply_file = os.path.join(sparse_dir, "points.ply")
    print("\n" + "=" * 60)
    print("Alignment completed successfully!")
    print(f"COLMAP Output: {sparse_dir}")
    print(f"Point Cloud:   {ply_file}")
    print("=" * 60)

    if args.view:
        demo_viser_path = os.path.join(script_dir, "demo_viser.py")
        print(f"\nLaunching 3D Interactive Viewer at http://localhost:{args.port} ...")
        subprocess.run([
            sys.executable,
            demo_viser_path,
            "--image_folder", images_dir,
            "--port", str(args.port)
        ])

if __name__ == "__main__":
    main()
