"""
Abu Align & Train - FileSift High-Speed File Matching & Cross-Folder Curation Engine
===================================================================================
State-of-the-Art, sub-millisecond file comparison, set-based matching, and parallel
copy/cut transfer engine for Photogrammetry, 3DGS, and VFX production pipelines.
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".ppm", ".pgm",
    ".exr", ".hdr", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2", ".pef",
    ".raf", ".mp4", ".mov", ".insv", ".avi", ".mkv"
}


@dataclass
class SiftMatchRecord:
    """Record representing a match (or miss) found between Source Reference and Target Folder."""
    source_key: str              # Filename or stem from source reference
    source_original_path: str    # Path in source folder (or empty if from text list)
    target_path: Optional[str]   # Full path to file found in Target (2nd folder)
    target_filename: str         # Name of file in Target folder
    file_size_bytes: int         # Size in bytes
    matched: bool                # True if found in Target folder
    relative_path: str = ""      # Relative path within target folder (for hierarchy preservation)
    selected: bool = True        # User checkbox selection for transfer action


@dataclass
class SiftTransferManifest:
    """Audit manifest tracking a batch transfer operation for instant one-click Undo."""
    timestamp: float
    action: str                  # "copy" or "cut"
    target_folder: str           # 2nd folder
    destination_folder: str      # 3rd folder
    transferred_files: List[Dict[str, str]] = field(default_factory=list)  # [{"orig": path, "dest": path}]


class FileSiftEngine:
    """
    High-Speed File Matching and Cross-Folder Sifter Engine.
    Achieves sub-millisecond matching across tens of thousands of files using O(1) set
    and dictionary hash tables, paired with a multi-threaded parallel I/O pipeline.
    """

    def __init__(self):
        self.source_reference_files: List[str] = []   # Raw names or paths from source
        self.target_files_cache: List[Tuple[str, str, int]] = []  # [(full_path, rel_path, size_bytes)]
        self.cached_matches: List[SiftMatchRecord] = []
        self.cached_unmatched_source: List[str] = []
        self.cached_unmatched_target: List[Tuple[str, str, int]] = []
        self.last_manifest: Optional[SiftTransferManifest] = None

    # -------------------------------------------------------------------------
    # 1. Source Reference Ingestion (Folder, File List, or Text/Clipboard)
    # -------------------------------------------------------------------------

    def load_source_from_folder(
        self,
        folder_path: str,
        recursive: bool = False,
        extension_filter: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Scans a source reference folder and extracts filenames.
        Uses os.scandir for high-speed file discovery.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Source folder does not exist: {folder_path}")

        files: List[str] = []
        if recursive:
            for root, _, entries in os.walk(folder_path):
                for e in entries:
                    if extension_filter:
                        ext = os.path.splitext(e)[1].lower()
                        if ext not in extension_filter:
                            continue
                    files.append(os.path.join(root, e))
        else:
            with os.scandir(folder_path) as it:
                for entry in it:
                    if entry.is_file():
                        if extension_filter:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext not in extension_filter:
                                continue
                        files.append(entry.path)

        self.source_reference_files = files
        return files

    def load_source_from_list_file(self, file_path: str) -> List[str]:
        """
        Loads filenames or paths from a text file (.txt, .lst, .csv, .json).
        Cleans surrounding whitespace and quotes.
        """
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Source list file not found: {file_path}")

        ext = p.suffix.lower()
        items: List[str] = []

        if ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = [str(x).strip() for x in data if str(x).strip()]
                elif isinstance(data, dict):
                    items = [str(x).strip() for x in data.keys() if str(x).strip()]
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_clean = line.strip().strip('"').strip("'")
                    if not line_clean or line_clean.startswith("#"):
                        continue
                    # If CSV with multiple columns, take the first column
                    if "," in line_clean and not os.path.exists(line_clean):
                        line_clean = line_clean.split(",")[0].strip().strip('"').strip("'")
                    items.append(line_clean)

        self.source_reference_files = items
        return items

    def load_source_from_text(self, text_content: str) -> List[str]:
        """Parses a multi-line or comma-separated string of filenames or paths."""
        items: List[str] = []
        for line in text_content.splitlines():
            line_clean = line.strip().strip('"').strip("'")
            if not line_clean or line_clean.startswith("#"):
                continue
            if "," in line_clean and not os.path.exists(line_clean):
                for part in line_clean.split(","):
                    p_clean = part.strip().strip('"').strip("'")
                    if p_clean:
                        items.append(p_clean)
            else:
                items.append(line_clean)

        self.source_reference_files = items
        return items

    # -------------------------------------------------------------------------
    # 2. Target Folder Fast Scanning (Folder 2)
    # -------------------------------------------------------------------------

    def scan_target_folder(
        self,
        target_folder: str,
        recursive: bool = True,
        extension_filter: Optional[Set[str]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[Tuple[str, str, int]]:
        """
        Rapidly indexes the target folder (Folder 2) using os.scandir.
        Returns a list of tuples: (full_path, relative_path, file_size_bytes).
        """
        target = Path(target_folder)
        if not target.is_dir():
            raise FileNotFoundError(f"Target folder does not exist: {target_folder}")

        target_base = str(target.resolve())
        results: List[Tuple[str, str, int]] = []
        count = 0

        if recursive:
            for root, _, files in os.walk(target_base):
                for f in files:
                    if extension_filter:
                        ext = os.path.splitext(f)[1].lower()
                        if ext not in extension_filter:
                            continue
                    full = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(full)
                    except OSError:
                        sz = 0
                    rel = os.path.relpath(full, target_base)
                    results.append((full, rel, sz))
                    count += 1
                    if progress_callback and (count % 1000 == 0):
                        progress_callback(count, f"Scanned {count:,} target files...")
        else:
            with os.scandir(target_base) as it:
                for entry in it:
                    if entry.is_file():
                        if extension_filter:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext not in extension_filter:
                                continue
                        try:
                            sz = entry.stat().st_size
                        except OSError:
                            sz = 0
                        results.append((entry.path, entry.name, sz))
                        count += 1
                        if progress_callback and (count % 1000 == 0):
                            progress_callback(count, f"Scanned {count:,} target files...")

        self.target_files_cache = results
        return results

    # -------------------------------------------------------------------------
    # 3. High-Speed Set-Based Matching Engine ("Speed of Light")
    # -------------------------------------------------------------------------

    def match_files(
        self,
        mode: str = "stem",
        case_sensitive: bool = False,
        invert_selection: bool = False
    ) -> Tuple[List[SiftMatchRecord], List[str], List[Tuple[str, str, int]]]:
        """
        Executes sub-millisecond matching between Source Reference and Target Folder.
        
        Modes:
        - "stem": Matches by base name without extension (e.g. frame_001.jpg matches frame_001.ARW)
        - "exact": Matches exact filename and extension
        - "prefix": Matches if target filename starts with source stem
        - "suffix": Matches if target filename ends with source stem
        
        Returns:
        - matched_records: List of SiftMatchRecord (files in target that matched source)
        - unmatched_source: List of source names that were not found in target
        - unmatched_target: List of target files that did not match any source item
        """
        t0 = time.time()

        # Step A: Build fast O(1) in-memory indices for Target files
        # We index Target files so that multiple target files matching one source stem
        # (e.g. both .ARW and .DNG) can be cleanly discovered!
        exact_map: Dict[str, List[Tuple[str, str, int]]] = {}
        stem_map: Dict[str, List[Tuple[str, str, int]]] = {}

        for full_p, rel_p, sz in self.target_files_cache:
            fname = os.path.basename(full_p)
            stem, _ = os.path.splitext(fname)

            k_exact = fname if case_sensitive else fname.lower()
            k_stem = stem if case_sensitive else stem.lower()

            exact_map.setdefault(k_exact, []).append((full_p, rel_p, sz))
            stem_map.setdefault(k_stem, []).append((full_p, rel_p, sz))

        matched_records: List[SiftMatchRecord] = []
        unmatched_source: List[str] = []
        matched_target_paths: Set[str] = set()

        # Step B: Query each Source Reference item in O(1) time
        for src in self.source_reference_files:
            src_fname = os.path.basename(src)
            src_stem, _ = os.path.splitext(src_fname)

            found_items: List[Tuple[str, str, int]] = []

            if mode == "exact":
                query = src_fname if case_sensitive else src_fname.lower()
                found_items = exact_map.get(query, [])
            elif mode == "stem":
                query = src_stem if case_sensitive else src_stem.lower()
                found_items = stem_map.get(query, [])
            elif mode == "prefix":
                query = src_stem if case_sensitive else src_stem.lower()
                for k_stem, items in stem_map.items():
                    if k_stem.startswith(query):
                        found_items.extend(items)
            elif mode == "suffix":
                query = src_stem if case_sensitive else src_stem.lower()
                for k_stem, items in stem_map.items():
                    if k_stem.endswith(query):
                        found_items.extend(items)

            if found_items:
                for full_p, rel_p, sz in found_items:
                    matched_records.append(SiftMatchRecord(
                        source_key=src_fname,
                        source_original_path=src if os.path.isabs(src) else "",
                        target_path=full_p,
                        target_filename=os.path.basename(full_p),
                        file_size_bytes=sz,
                        matched=True,
                        relative_path=rel_p,
                        selected=True
                    ))
                    matched_target_paths.add(full_p)
            else:
                unmatched_source.append(src_fname)

        # Step C: Identify target files that were NOT matched by source
        unmatched_target: List[Tuple[str, str, int]] = []
        for full_p, rel_p, sz in self.target_files_cache:
            if full_p not in matched_target_paths:
                unmatched_target.append((full_p, rel_p, sz))

        # Handle Invert Selection: If inverted, the user desires to take files that did NOT match
        final_matches = matched_records
        if invert_selection:
            inverted_records: List[SiftMatchRecord] = []
            for full_p, rel_p, sz in unmatched_target:
                fname = os.path.basename(full_p)
                inverted_records.append(SiftMatchRecord(
                    source_key="[UNMATCHED]",
                    source_original_path="",
                    target_path=full_p,
                    target_filename=fname,
                    file_size_bytes=sz,
                    matched=True,
                    relative_path=rel_p,
                    selected=True
                ))
            final_matches = inverted_records

        elapsed_ms = (time.time() - t0) * 1000.0
        print(f"[FileSift] Matched {len(final_matches):,} files ({len(unmatched_source):,} unmatched) in {elapsed_ms:.2f}ms")

        self.cached_matches = final_matches
        self.cached_unmatched_source = unmatched_source
        self.cached_unmatched_target = unmatched_target
        return final_matches, unmatched_source, unmatched_target

    # -------------------------------------------------------------------------
    # 4. Multi-Threaded Parallel Copy / Cut Transfer Pipeline
    # -------------------------------------------------------------------------

    def execute_transfer(
        self,
        destination_folder: str,
        action: str = "copy",  # "copy" or "cut"
        preserve_hierarchy: bool = True,
        overwrite_existing: bool = False,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        cancel_token: Optional[Callable[[], bool]] = None
    ) -> Tuple[int, int, float]:
        """
        Executes parallel copying or cutting of selected matched files from Target (Folder 2)
        into Destination (Folder 3).
        
        Returns:
        - success_count: Number of successfully transferred files
        - fail_count: Number of failed files
        - total_mb_transferred: Volume transferred in megabytes
        """
        dest_base = Path(destination_folder)
        dest_base.mkdir(parents=True, exist_ok=True)

        selected_records = [r for r in self.cached_matches if r.selected and r.target_path]
        total_count = len(selected_records)
        if total_count == 0:
            return 0, 0, 0.0

        manifest = SiftTransferManifest(
            timestamp=time.time(),
            action=action.lower(),
            target_folder=os.path.commonpath([r.target_path for r in selected_records]) if total_count > 1 else os.path.dirname(selected_records[0].target_path),
            destination_folder=str(dest_base.resolve())
        )

        success_count = 0
        fail_count = 0
        total_bytes = 0

        # Worker count tuned for SSD/NVMe saturation
        num_workers = min(16, os.cpu_count() or 4)

        def _transfer_single(record: SiftMatchRecord) -> Tuple[bool, int, Optional[Dict[str, str]]]:
            src_path = record.target_path
            if not src_path or not os.path.isfile(src_path):
                return False, 0, None

            # Determine target destination path
            if preserve_hierarchy and record.relative_path:
                dest_path = dest_base / record.relative_path
            else:
                dest_path = dest_base / record.target_filename

            # Ensure parent directories exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            str_dest = str(dest_path.resolve())

            # Collision check
            if os.path.exists(str_dest) and not overwrite_existing:
                # Rename with unique counter
                stem = dest_path.stem
                suffix = dest_path.suffix
                idx = 1
                while os.path.exists(str_dest):
                    str_dest = str(dest_path.parent / f"{stem}_{idx}{suffix}")
                    idx += 1

            try:
                sz = os.path.getsize(src_path)
                if action.lower() == "cut":
                    # Instant O(1) replace if on same drive, fallback to copy+delete across drives
                    try:
                        os.replace(src_path, str_dest)
                    except OSError:
                        shutil.move(src_path, str_dest)
                else:
                    # High-speed copy preserving timestamps and metadata
                    shutil.copy2(src_path, str_dest)

                return True, sz, {"orig": src_path, "dest": str_dest}
            except Exception as e:
                print(f"[FileSift Error] Failed to {action} {src_path} -> {str_dest}: {e}")
                return False, 0, None

        t_start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for idx, res in enumerate(pool.map(_transfer_single, selected_records)):
                if cancel_token and cancel_token():
                    print("[FileSift] Transfer operation cancelled by user.")
                    break

                ok, bytes_done, item_audit = res
                if ok:
                    success_count += 1
                    total_bytes += bytes_done
                    if item_audit:
                        manifest.transferred_files.append(item_audit)
                else:
                    fail_count += 1

                if progress_callback:
                    curr_file = selected_records[idx].target_filename
                    mb_done = total_bytes / (1024 * 1024)
                    progress_callback(idx + 1, total_count, curr_file, mb_done)

        self.last_manifest = manifest

        # Write manifest file to destination folder for external recovery audit
        try:
            audit_file = dest_base / f".filesift_manifest_{int(manifest.timestamp)}.json"
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": manifest.timestamp,
                    "action": manifest.action,
                    "target_folder": manifest.target_folder,
                    "destination_folder": manifest.destination_folder,
                    "total_files": len(manifest.transferred_files),
                    "files": manifest.transferred_files
                }, f, indent=2)
        except Exception:
            pass

        total_mb = total_bytes / (1024 * 1024)
        elapsed = max(0.001, time.time() - t_start)
        rate_mbs = total_mb / elapsed
        print(f"[FileSift] {action.capitalize()} completed: {success_count:,} files ({total_mb:.1f} MB) in {elapsed:.2f}s ({rate_mbs:.1f} MB/s)")

        return success_count, fail_count, total_mb

    # -------------------------------------------------------------------------
    # 5. One-Click Safe Undo Restoration
    # -------------------------------------------------------------------------

    def undo_last_transfer(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, int]:
        """
        Reverts the last copy or cut action:
        - If action was 'cut': moves files back from Destination (Folder 3) to Target (Folder 2).
        - If action was 'copy': safely deletes the copied files in Destination (Folder 3).
        
        Returns:
        - (restored_count, error_count)
        """
        if not self.last_manifest or not self.last_manifest.transferred_files:
            raise ValueError("No transfer operations recorded to undo.")

        manifest = self.last_manifest
        action = manifest.action
        total = len(manifest.transferred_files)
        restored_count = 0
        error_count = 0

        for idx, item in enumerate(manifest.transferred_files):
            orig_p = item["orig"]
            dest_p = item["dest"]

            try:
                if action == "cut":
                    # Move back to original target location
                    if os.path.exists(dest_p):
                        os.makedirs(os.path.dirname(orig_p), exist_ok=True)
                        try:
                            os.replace(dest_p, orig_p)
                        except OSError:
                            shutil.move(dest_p, orig_p)
                        restored_count += 1
                else:
                    # Action was copy: remove the destination replica
                    if os.path.isfile(dest_p):
                        os.remove(dest_p)
                        restored_count += 1
            except Exception as e:
                print(f"[FileSift Undo Error] Failed on {dest_p}: {e}")
                error_count += 1

            if progress_callback:
                progress_callback(idx + 1, total, os.path.basename(orig_p))

        self.last_manifest = None
        return restored_count, error_count
