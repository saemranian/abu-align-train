"""
Abu Align & Train - BlurDrop CLI (Command Line Interface)
==========================================================
Headless batch execution tool for filtering and curating blurry images.
"""

import os
import sys
import argparse
import time
from pathlib import Path

from blur_drop_engine import BlurDropEngine


def main():
    parser = argparse.ArgumentParser(description="BlurDrop: Hardware-Accelerated Blur & Sharpness Curation CLI")
    parser.add_argument("--input", "-i", required=True, help="Path to input image folder")
    parser.add_argument("--dest", "-d", default=None, help="Destination folder for blurry images (default: <input>/blurred_images)")
    parser.add_argument("--threshold", "-t", type=float, default=40.0, help="Sharpness cutoff threshold (0-100, default: 40.0)")
    parser.add_argument("--preset", "-p", choices=["balanced", "strict", "permissive", "custom"], default="balanced", help="Blur detection preset")
    parser.add_argument("--motion-sens", type=float, default=1.0, help="Motion blur streak sensitivity (default: 1.0)")
    parser.add_argument("--subject-aware", action="store_true", default=True, help="Enable subject-aware depth-of-field preservation")
    parser.add_argument("--move", action="store_true", help="Execute moving of flagged blurry images to destination folder")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Compute device (default: auto cuda/cpu)")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        print(f"[Error] Input directory not found: {input_dir}")
        sys.exit(1)

    dest_dir = os.path.abspath(args.dest) if args.dest else os.path.join(input_dir, "blurred_images")

    print("=" * 70)
    print("BLURDROP · HARDWARE-ACCELERATED SHARPNESS & BLUR CURATION")
    print(f"Input Directory  : {input_dir}")
    print(f"Cutoff Threshold : {args.threshold}%")
    print(f"Preset Mode      : {args.preset.upper()}")
    print(f"Motion Sens      : {args.motion_sens}x")
    print(f"Move Enabled     : {args.move} (Dest: {dest_dir})")
    print("=" * 70)

    engine = BlurDropEngine(device=args.device)

    def _prog(cur, tot, msg):
        pct = int(cur * 100 / tot)
        print(f"\r[{pct:3d}%] {msg}", end="", flush=True)

    t0 = time.time()
    results = engine.analyze_folder(
        folder_path=input_dir,
        threshold=args.threshold,
        preset=args.preset,
        motion_sensitivity=args.motion_sens,
        subject_aware=args.subject_aware,
        progress_callback=_prog
    )
    print()

    total = len(results)
    blurry = [r for r in results if r.is_blurry]
    sharp = [r for r in results if not r.is_blurry]

    print("\n" + "-" * 70)
    print(f"ANALYSIS SUMMARY ({time.time() - t0:.2f}s total):")
    print(f"  • Total Images Scanned : {total}")
    print(f"  • Sharp Images (Keep)  : {len(sharp)} ({len(sharp) * 100 / max(1, total):.1f}%)")
    print(f"  • Blurry Images (Drop) : {len(blurry)} ({len(blurry) * 100 / max(1, total):.1f}%)")
    print("-" * 70)

    if blurry:
        print("\nFlagged Blurry Images:")
        for r in blurry[:20]:
            print(f"  - {r.filename:28s} | Score: {r.composite_score:4.1f} | Class: {r.classification:12s} | Defocus: {r.defocus_score:4.1f} | Motion: {r.motion_score:4.1f}")
        if len(blurry) > 20:
            print(f"  ... and {len(blurry) - 20} more.")

    if args.move and blurry:
        print(f"\nMoving {len(blurry)} blurry images to: {dest_dir}...")
        move_res = engine.cut_and_move_blurry_images(
            destination_dir=dest_dir,
            selected_paths=[r.path for r in blurry]
        )
        print(f"SUCCESS: Moved {move_res['moved_count']} images. Cleaned main folder has {move_res['remaining_sharp_count']} sharp images.")
    elif not args.move and blurry:
        print("\nNote: Run with '--move' to cut and move these blurry images into destination folder.")


if __name__ == "__main__":
    main()
