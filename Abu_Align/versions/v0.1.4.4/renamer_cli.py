"""
Abu Align & Train - Renamer Headless Command-Line Interface (CLI)
==================================================================
Lightning-fast sequence renamer CLI supporting natural sorting,
live dry-run preview, collision-safe in-place renaming, and destination copying.
"""

import os
import sys
import argparse
from pathlib import Path
from renamer_engine import RenamerEngine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Renamer: Sequence & Batch File Renamer CLI"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input folder containing files to rename"
    )
    parser.add_argument(
        "--name", "-n",
        default="frame",
        help="Desired base name for the sequence (default: frame)"
    )
    parser.add_argument(
        "--dest", "-d",
        default=None,
        help="Destination folder if using copy mode (default: <input>/renamed_sequence)"
    )
    parser.add_argument(
        "--copy", "-c",
        action="store_true",
        default=False,
        help="Copy renamed files to destination instead of renaming in-place"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting sequence number (default: 1)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=4,
        help="Zero padding width for numbers (default: 4, e.g. 0001)"
    )
    parser.add_argument(
        "--delimiter",
        default="_",
        help="Separator between base name and sequence number (default: _)"
    )
    parser.add_argument(
        "--sort",
        choices=["natural", "name", "date_asc", "date_desc"],
        default="natural",
        help="File sorting criteria (default: natural)"
    )
    parser.add_argument(
        "--lowercase-ext",
        action="store_true",
        default=True,
        help="Convert extensions to lowercase (default: True)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview final sequenced names without modifying disk"
    )

    args = parser.parse_args()

    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"Error: Input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    mode = "copy" if args.copy else "inplace"
    dest = args.dest if args.copy else None
    if args.copy and not dest:
        dest = os.path.join(input_dir, "renamed_sequence")

    engine = RenamerEngine()

    print("=" * 70)
    print("[RENAMER] SEQUENCE & BATCH FILE RENAMER")
    print("=" * 70)
    print(f"Folder        : {input_dir}")
    print(f"Desired Name  : '{args.name}'")
    print(f"Sequence      : Start={args.start_index}, Padding={args.padding}, Delim='{args.delimiter}'")
    print(f"Sort Strategy : {args.sort}")
    print(f"Mode          : {'COPY to ' + dest if mode == 'copy' else 'IN-PLACE OVERWRITE'}")
    print("-" * 70)

    # 1. Generate Preview
    items = engine.generate_preview(
        folder_path=input_dir,
        base_name=args.name,
        start_index=args.start_index,
        padding=args.padding,
        delimiter=args.delimiter,
        sort_by=args.sort,
        lowercase_ext=args.lowercase_ext,
        dest_folder=dest
    )

    total = len(items)
    if total == 0:
        print("No files found in folder to rename.")
        sys.exit(0)

    print(f"\n[LIVE PREVIEW] {total:,} files will be sequenced:\n")
    sample_count = min(10, total)
    for idx in range(sample_count):
        it = items[idx]
        print(f"  * {it.original_filename:32} -> {it.new_filename}")

    if total > sample_count:
        print(f"  ... and {total - sample_count:,} more files.")
        last_it = items[-1]
        print(f"  * {last_it.original_filename:32} -> {last_it.new_filename}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] No filesystem changes made.")
        sys.exit(0)

    # 2. Execute
    print(f"\nExecuting {mode.upper()} rename on {total:,} files...")
    succ, err = engine.execute_rename(
        items=items,
        mode=mode,
        dest_folder=dest,
        progress_callback=lambda cur, tot, fname: print(f"[{cur:,}/{tot:,}] {fname}", end="\r")
    )
    print(f"\n\n✓ Successfully processed {succ:,} files ({err} errors).")


if __name__ == "__main__":
    main()
