"""
Abu Align & Train - FileSift Headless Command-Line Interface (CLI)
==================================================================
Lightning-fast command-line tool for cross-folder file matching,
cross-format pairing (e.g. JPG to RAW), and parallel copy/cut transfers.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from file_sift_engine import FileSiftEngine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="FileSift: High-Speed Cross-Folder File Matching and Sifter CLI"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to source reference folder, list file (.txt, .csv, .json), or comma-separated filenames"
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Path to target search folder (Folder 2)"
    )
    parser.add_argument(
        "--dest", "-d",
        default=None,
        help="Path to destination folder (Folder 3). Required if --action is copy or cut"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["stem", "exact", "prefix", "suffix"],
        default="stem",
        help="Matching strategy (default: stem [ignores extensions to match JPG <-> RAW/DNG])"
    )
    parser.add_argument(
        "--action", "-a",
        choices=["copy", "cut", "dry-run"],
        default="dry-run",
        help="Action to perform on matched files (default: dry-run)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Scan target directory recursively (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Scan top-level of target directory only"
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        default=False,
        help="Enforce exact case-sensitive matching"
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        default=False,
        help="Invert selection (sift files from target that do NOT match source)"
    )
    parser.add_argument(
        "--flatten",
        action="store_false",
        dest="preserve_hierarchy",
        default=True,
        help="Flatten all output files into root destination instead of preserving subfolder hierarchy"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite destination files if collisions occur"
    )

    args = parser.parse_args()

    engine = FileSiftEngine()

    print("=" * 70)
    print("⚡ FILESIFT HIGH-SPEED FILE MATCHER & CROSS-FOLDER SIFTER")
    print("=" * 70)

    # 1. Ingest Source
    src = args.source
    if os.path.isdir(src):
        print(f"[1/3] Ingesting source reference folder: {src}")
        engine.load_source_from_folder(src, recursive=args.recursive)
    elif os.path.isfile(src):
        print(f"[1/3] Ingesting source list file: {src}")
        engine.load_source_from_list_file(src)
    else:
        print("[1/3] Parsing source reference text entries...")
        engine.load_source_from_text(src)

    print(f"      Loaded {len(engine.source_reference_files):,} source items.")

    # 2. Scan Target
    tgt = args.target
    if not os.path.isdir(tgt):
        print(f"Error: Target directory does not exist: {tgt}", file=sys.stderr)
        sys.exit(1)

    print(f"[2/3] Scanning target folder: {tgt} (Recursive={args.recursive})")
    engine.scan_target_folder(tgt, recursive=args.recursive)
    print(f"      Indexed {len(engine.target_files_cache):,} files in target directory.")

    # 3. Match
    print(f"[3/3] Performing O(1) set-based matching (Mode='{args.mode}', CaseSensitive={args.case_sensitive}, Invert={args.invert})...")
    t0 = time.time()
    matches, missing_src, unmatched_tgt = engine.match_files(
        mode=args.mode,
        case_sensitive=args.case_sensitive,
        invert_selection=args.invert
    )
    match_ms = (time.time() - t0) * 1000.0

    total_mb = sum(r.file_size_bytes for r in matches) / (1024 * 1024)
    print(f"      ✓ Matched {len(matches):,} files ({total_mb:.1f} MB) in {match_ms:.2f}ms!")
    print(f"      - Missing from target : {len(missing_src):,} items")
    print(f"      - Unmatched in target : {len(unmatched_tgt):,} files")

    # 4. Action
    if args.action == "dry-run":
        print("\n[DRY RUN COMPLETE] Use --action copy or --action cut with --dest <folder> to execute transfer.")
        if matches:
            print("\nSample Matched Items (First 10):")
            for m in matches[:10]:
                sz_str = f"{m.file_size_bytes / (1024*1024):.1f} MB" if m.file_size_bytes > 1024*1024 else f"{m.file_size_bytes / 1024:.0f} KB"
                print(f"  • {m.target_filename:30} <== matched key '{m.source_key}' ({sz_str})")
            if len(matches) > 10:
                print(f"  ... and {len(matches) - 10:,} more files.")
        sys.exit(0)

    dest = args.dest
    if not dest:
        print("Error: --dest must be specified when --action is copy or cut.", file=sys.stderr)
        sys.exit(1)

    print(f"\nExecuting {args.action.upper()} transfer of {len(matches):,} files to: {dest}")
    t_xfer = time.time()
    succ, fail, mb = engine.execute_transfer(
        destination_folder=dest,
        action=args.action,
        preserve_hierarchy=args.preserve_hierarchy,
        overwrite_existing=args.overwrite
    )
    elapsed = max(0.001, time.time() - t_xfer)
    rate = mb / elapsed

    print(f"\n✓ Completed {args.action.upper()}: {succ:,} succeeded, {fail:,} failed ({mb:.1f} MB in {elapsed:.2f}s, {rate:.1f} MB/s)")


if __name__ == "__main__":
    main()
