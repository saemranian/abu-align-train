"""
Abu Align & Train - Frame Picker Headless Command-Line Interface (CLI)
========================================================================
Light-speed frame cadence decimator and sequence sampling CLI.
Supports natural sequence sorting, dry-run previews, parallel multi-threaded
copy or cut file transfers, and one-click undo restoration.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from frame_picker_engine import (
    FramePickerEngine,
    FramePickerManifest
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def format_bytes(b: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if b < 1024.0:
            return f"{b:.1f} {unit}"
        b /= 1024.0
    return f"{b:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Frame Picker: Light-Speed Cadence Decimator & Sequence Sampler CLI"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to folder containing image sequence"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Destination folder for picked frames (default: <input>_picked_cadence_<N>)"
    )
    parser.add_argument(
        "--cadence", "--every", "--stride", "-c", "-n",
        dest="cadence",
        type=int,
        default=5,
        help="Sampling cadence: pick every N-th frame (e.g. 5 means pick every 5th frame, default: 5)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=1,
        help="1-based start offset frame (default: 1)"
    )
    parser.add_argument(
        "--action", "-a",
        choices=["copy", "move", "dry-run"],
        default="copy",
        help="Transfer action: 'copy', 'move' (cut), or 'dry-run' (default: copy)"
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        default=False,
        help="Invert selection (drop every Nth frame, keep others)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=False,
        help="Recursively scan subfolders"
    )
    parser.add_argument(
        "--sort-by",
        choices=["natural", "name", "date_asc", "date_desc"],
        default="natural",
        help="Sequence sorting strategy (default: natural alphanumeric)"
    )
    parser.add_argument(
        "--range",
        default=None,
        help="Optional 1-based frame range, e.g. '1:500' or '100:200'"
    )
    parser.add_argument(
        "--undo",
        default=None,
        nargs="?",
        const="auto",
        help="Revert the last move operation. Optionally specify the path to .frame_picker_manifest.json"
    )

    args = parser.parse_args()

    engine = FramePickerEngine()

    # 1. Handle Undo Operation
    if args.undo:
        manifest_path = None
        if args.undo != "auto" and os.path.isfile(args.undo):
            manifest_path = args.undo
        elif args.output and os.path.isfile(os.path.join(args.output, ".frame_picker_manifest.json")):
            manifest_path = os.path.join(args.output, ".frame_picker_manifest.json")
        elif args.input and os.path.isfile(os.path.join(args.input, ".frame_picker_manifest.json")):
            manifest_path = os.path.join(args.input, ".frame_picker_manifest.json")

        if not manifest_path or not os.path.isfile(manifest_path):
            print(f"[Error] Could not locate manifest file for Undo. Please provide path to .frame_picker_manifest.json", file=sys.stderr)
            sys.exit(1)

        print(f"[FramePicker] Loading undo manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = FramePickerManifest(
            timestamp=data.get("timestamp", 0),
            action=data.get("action", "move"),
            source_folder=data.get("source_folder", ""),
            destination_folder=data.get("destination_folder", ""),
            cadence=data.get("cadence", 1),
            offset=data.get("offset", 1),
            transferred_count=data.get("transferred_count", 0),
            mappings=data.get("mappings", [])
        )

        print(f"[FramePicker] Restoring {len(manifest.mappings):,} files back to: {manifest.source_folder}...")
        t0 = time.time()
        restored, errors = engine.undo_last_operation(manifest=manifest)
        print(f"[FramePicker] Undo finished: {restored:,} files restored ({errors} errors) in {time.time() - t0:.2f}s")
        sys.exit(0 if errors == 0 else 1)

    # 2. Validate Input
    if not args.input:
        print("[Error] Missing required argument: --input <folder>", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    src_folder = os.path.abspath(args.input)
    if not os.path.isdir(src_folder):
        print(f"[Error] Input directory does not exist: {src_folder}", file=sys.stderr)
        sys.exit(1)

    cadence = max(1, args.cadence)
    offset = max(1, args.offset)

    dest_folder = args.output
    if not dest_folder:
        inv_str = "_inverted" if args.invert else ""
        dest_folder = f"{src_folder}_picked_cadence_{cadence}{inv_str}"
    dest_folder = os.path.abspath(dest_folder)

    # Parse range if specified
    r_start, r_end = None, None
    if args.range:
        parts = args.range.split(":")
        if len(parts) == 2:
            try:
                r_start = int(parts[0]) if parts[0] else None
                r_end = int(parts[1]) if parts[1] else None
            except ValueError:
                pass

    print(f"======================================================================")
    print(f"FRAME PICKER: LIGHT-SPEED CADENCE DECIMATOR CLI")
    print(f"======================================================================")
    print(f"Source Folder     : {src_folder}")
    print(f"Destination Folder: {dest_folder}")
    print(f"Sampling Cadence  : Every {cadence} frames (Start Offset: {offset})")
    print(f"Invert Selection  : {args.invert}")
    print(f"Action Mode       : {args.action.upper()}")
    print(f"Sorting Strategy  : {args.sort_by}")
    print(f"----------------------------------------------------------------------")

    # Step 1: Fast Scan & Sort
    t0 = time.time()
    frames = engine.scan_folder(
        folder_path=src_folder,
        recursive=args.recursive,
        sort_by=args.sort_by
    )
    scan_dur = time.time() - t0

    if not frames:
        print(f"[Warning] No supported image frames found in {src_folder}")
        sys.exit(0)

    # Step 2: Vectorized Decimation Selection
    items, stats = engine.compute_selection(
        cadence=cadence,
        offset=offset,
        invert=args.invert,
        range_start=r_start,
        range_end=r_end
    )

    print(f"[Scan] Scanned {stats.total_frames:,} images in {scan_dur * 1000:.1f}ms")
    print(f"[Cadence] Total Frames : {stats.total_frames:,} ({format_bytes(stats.picked_size_bytes + stats.skipped_size_bytes)})")
    print(f"[Cadence] Picked Frames: {stats.picked_frames:,} ({format_bytes(stats.picked_size_bytes)}, {stats.decimation_ratio_pct:.1f}%)")
    print(f"[Cadence] Skipped      : {stats.skipped_frames:,} ({format_bytes(stats.skipped_size_bytes)})")
    print(f"----------------------------------------------------------------------")

    # Step 3: Action Execution or Dry-Run
    picked_items = [it for it in items if it.selected]

    if args.action == "dry-run":
        print(f"[Dry-Run] Sample of first 10 picked frames:")
        for idx, it in enumerate(picked_items[:10], start=1):
            print(f"  [{idx:02d}] Seq #{it.sequence_index:04d} -> {it.filename} ({format_bytes(it.size_bytes)})")
        if len(picked_items) > 10:
            print(f"  ... and {len(picked_items) - 10:,} more picked frames.")
        print(f"[Dry-Run Complete] No files were modified.")
        sys.exit(0)

    print(f"[Execute] Starting parallel {args.action.upper()} of {len(picked_items):,} frames...")
    t_xfer = time.time()
    success, errors, manifest = engine.execute_transfer(
        items_to_transfer=picked_items,
        destination_folder=dest_folder,
        action=args.action,
        preserve_subdirs=True
    )
    xfer_dur = time.time() - t_xfer
    rate = success / max(0.001, xfer_dur)

    print(f"----------------------------------------------------------------------")
    print(f"[Success] {args.action.capitalize()}ed {success:,} frames to:\n          {dest_folder}")
    if errors > 0:
        print(f"[Warning] Encountered {errors} errors during transfer.")
    print(f"[Performance] Elapsed: {xfer_dur:.2f}s | Throughput: {rate:.1f} frames/sec")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
