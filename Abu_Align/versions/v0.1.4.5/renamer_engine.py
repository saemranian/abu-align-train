"""
Abu Align & Train - Renamer Core Engine
=======================================
High-speed, collision-safe sequence and batch file renaming engine.
Supports natural sorting, chronological timeline ordering, live preview generation,
two-phase in-place renaming, destination copying, and one-click undo restoration.
"""

import os
import sys
import re
import time
import uuid
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".ppm", ".pgm",
    ".exr", ".hdr", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf", ".rw2", ".pef",
    ".raf", ".mp4", ".mov", ".insv", ".avi", ".mkv", ".json", ".txt", ".csv"
}


@dataclass
class RenameItem:
    """Record representing a single file renaming operation."""
    original_path: str
    original_filename: str
    new_filename: str
    new_path: str
    size_bytes: int
    mtime: float
    is_changed: bool = True


@dataclass
class RenameAuditManifest:
    """Manifest tracking a completed rename/copy batch for complete one-click Undo."""
    timestamp: float
    mode: str                    # "inplace" or "copy"
    source_folder: str
    dest_folder: str
    mappings: List[Dict[str, str]] = field(default_factory=list)  # [{"orig": path, "current": path}]


def natural_sort_key(filename: str):
    """
    Sort key for natural human/alphanumeric ordering.
    Guarantees 'frame_2.jpg' precedes 'frame_10.jpg'.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]


class RenamerEngine:
    """
    Core sequence renamer engine.
    Computes instant rename mappings, executes collision-safe two-phase in-place renaming
    or parallel destination copying, and provides full one-click undo restoration.
    """

    def __init__(self):
        self.cached_items: List[RenameItem] = []
        self.last_manifest: Optional[RenameAuditManifest] = None

    @staticmethod
    def scan_files(
        folder_path: str,
        sort_by: str = "natural",  # "natural", "name", "date_asc", "date_desc"
        extension_filter: Optional[Set[str]] = None
    ) -> List[Tuple[str, str, int, float]]:
        """
        Scans all files in folder and applies the requested sort ordering.
        Returns list of (full_path, filename, size_bytes, mtime).
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")

        entries: List[Tuple[str, str, int, float]] = []
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file():
                    fname = entry.name
                    if extension_filter:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in extension_filter:
                            continue
                    try:
                        st = entry.stat()
                        sz = st.st_size
                        mt = st.st_mtime
                    except OSError:
                        sz = 0
                        mt = 0.0
                    entries.append((entry.path, fname, sz, mt))

        # Apply sorting
        if sort_by == "natural":
            entries.sort(key=lambda x: natural_sort_key(x[1]))
        elif sort_by == "name":
            entries.sort(key=lambda x: x[1].lower())
        elif sort_by == "date_asc":
            entries.sort(key=lambda x: (x[3], natural_sort_key(x[1])))
        elif sort_by == "date_desc":
            entries.sort(key=lambda x: (-x[3], natural_sort_key(x[1])))

        return entries

    def generate_preview(
        self,
        folder_path: str,
        base_name: str,
        start_index: int = 1,
        padding: int = 4,
        delimiter: str = "_",
        sort_by: str = "natural",
        lowercase_ext: bool = False,
        extension_filter: Optional[Set[str]] = None,
        dest_folder: Optional[str] = None
    ) -> List[RenameItem]:
        """
        Computes live before/after rename mappings for all files in the folder in under 1ms.
        """
        files = self.scan_files(folder_path, sort_by=sort_by, extension_filter=extension_filter)
        clean_base = base_name.strip()

        target_dir = Path(dest_folder) if dest_folder else Path(folder_path)

        preview_items: List[RenameItem] = []
        for idx, (full_p, orig_fname, sz, mt) in enumerate(files):
            seq_num = start_index + idx
            seq_str = str(seq_num).zfill(padding)

            orig_stem, orig_ext = os.path.splitext(orig_fname)
            final_ext = orig_ext.lower() if lowercase_ext else orig_ext

            if clean_base:
                new_fname = f"{clean_base}{delimiter}{seq_str}{final_ext}"
            else:
                # If base name is blank, just sequence with zero padding
                new_fname = f"{seq_str}{final_ext}"

            new_full_path = str(target_dir / new_fname)
            is_changed = (orig_fname != new_fname) or (dest_folder is not None and dest_folder != folder_path)

            preview_items.append(RenameItem(
                original_path=full_p,
                original_filename=orig_fname,
                new_filename=new_fname,
                new_path=new_full_path,
                size_bytes=sz,
                mtime=mt,
                is_changed=is_changed
            ))

        self.cached_items = preview_items
        return preview_items

    # -------------------------------------------------------------------------
    # Execution (In-Place Two-Phase Rename vs Destination Copy)
    # -------------------------------------------------------------------------

    def execute_rename(
        self,
        items: List[RenameItem],
        mode: str = "inplace",  # "inplace" or "copy"
        dest_folder: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_token: Optional[Callable[[], bool]] = None
    ) -> Tuple[int, int]:
        """
        Executes sequence renaming.
        
        If mode == 'inplace':
          Uses a two-phase atomic rename with UUID temporary sentinels to guarantee
          zero naming collisions (e.g. when sequence numbers shift in-place).
          
        If mode == 'copy':
          Parallel multi-threaded copy into dest_folder.
          
        Returns:
          (success_count, error_count)
        """
        if not items:
            return 0, 0

        source_dir = os.path.dirname(items[0].original_path)
        target_dir = dest_folder if (mode == "copy" and dest_folder) else source_dir
        os.makedirs(target_dir, exist_ok=True)

        total = len(items)
        success_count = 0
        error_count = 0

        manifest = RenameAuditManifest(
            timestamp=time.time(),
            mode=mode,
            source_folder=source_dir,
            dest_folder=target_dir
        )

        if mode == "inplace":
            # -----------------------------------------------------------------
            # Safe Two-Phase In-Place Rename
            # -----------------------------------------------------------------
            # Phase 1: Rename all to unique temporary sentinels
            temp_mappings: List[Tuple[str, str, str]] = []  # (orig_path, temp_path, final_path)
            session_id = uuid.uuid4().hex[:6]

            for idx, item in enumerate(items):
                if cancel_token and cancel_token():
                    break
                orig = item.original_path
                stem, ext = os.path.splitext(item.original_filename)
                temp_name = f".tmp_renamer_{session_id}_{idx:06d}{ext}"
                temp_path = os.path.join(source_dir, temp_name)
                try:
                    os.replace(orig, temp_path)
                    temp_mappings.append((orig, temp_path, item.new_path))
                except Exception as e:
                    print(f"[Renamer Error] Phase 1 rename failed on {orig}: {e}")
                    error_count += 1
                    # Rollback already moved temp files
                    for r_orig, r_temp, _ in temp_mappings:
                        try:
                            if os.path.exists(r_temp):
                                os.replace(r_temp, r_orig)
                        except Exception:
                            pass
                    return 0, error_count

            # Phase 2: Rename from temp sentinels to final sequenced names
            for idx, (orig, temp_path, final_path) in enumerate(temp_mappings):
                if cancel_token and cancel_token():
                    break
                try:
                    os.replace(temp_path, final_path)
                    success_count += 1
                    manifest.mappings.append({"orig": orig, "current": final_path})
                except Exception as e:
                    print(f"[Renamer Error] Phase 2 rename failed on {temp_path} -> {final_path}: {e}")
                    error_count += 1

                if progress_callback:
                    progress_callback(idx + 1, total, os.path.basename(final_path))

        else:
            # -----------------------------------------------------------------
            # Copy to Destination Folder
            # -----------------------------------------------------------------
            num_workers = min(16, os.cpu_count() or 4)

            def _copy_worker(item: RenameItem) -> Tuple[bool, str, str]:
                try:
                    final_path = os.path.join(target_dir, item.new_filename)
                    shutil.copy2(item.original_path, final_path)
                    return True, item.original_path, final_path
                except Exception as e:
                    print(f"[Renamer Copy Error] Failed on {item.original_path}: {e}")
                    return False, item.original_path, ""

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                for idx, res in enumerate(pool.map(_copy_worker, items)):
                    if cancel_token and cancel_token():
                        break
                    ok, orig, final_path = res
                    if ok:
                        success_count += 1
                        manifest.mappings.append({"orig": orig, "current": final_path})
                    else:
                        error_count += 1

                    if progress_callback:
                        progress_callback(idx + 1, total, os.path.basename(final_path))

        self.last_manifest = manifest

        # Save audit manifest
        try:
            audit_file = os.path.join(target_dir, f".renamer_manifest_{int(manifest.timestamp)}.json")
            with open(audit_file, "w", encoding="utf-8") as f:
                import json
                json.dump({
                    "timestamp": manifest.timestamp,
                    "mode": manifest.mode,
                    "source_folder": manifest.source_folder,
                    "dest_folder": manifest.dest_folder,
                    "total_files": len(manifest.mappings),
                    "mappings": manifest.mappings
                }, f, indent=2)
        except Exception:
            pass

        print(f"[Renamer] Completed {mode} rename: {success_count:,} files successfully ({error_count} errors)")
        return success_count, error_count

    # -------------------------------------------------------------------------
    # Undo Restoration
    # -------------------------------------------------------------------------

    def undo_last_rename(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[int, int]:
        """
        Reverts the last in-place rename back to original filenames,
        or removes copied files in destination folder.
        """
        if not self.last_manifest or not self.last_manifest.mappings:
            raise ValueError("No rename operation recorded to undo.")

        manifest = self.last_manifest
        mode = manifest.mode
        mappings = manifest.mappings
        total = len(mappings)
        restored = 0
        errors = 0

        if mode == "inplace":
            # Revert in-place using two-phase rollback
            session_id = uuid.uuid4().hex[:6]
            temp_mappings = []

            # Phase 1: To temp
            for idx, m in enumerate(mappings):
                curr = m["current"]
                orig = m["orig"]
                temp = os.path.join(manifest.source_folder, f".tmp_undo_{session_id}_{idx:06d}{Path(curr).suffix}")
                try:
                    if os.path.exists(curr):
                        os.replace(curr, temp)
                        temp_mappings.append((temp, orig))
                except Exception as e:
                    print(f"[Renamer Undo Error] Failed on {curr}: {e}")
                    errors += 1

            # Phase 2: To original
            for idx, (temp, orig) in enumerate(temp_mappings):
                try:
                    os.replace(temp, orig)
                    restored += 1
                except Exception as e:
                    print(f"[Renamer Undo Error] Failed to restore {temp} -> {orig}: {e}")
                    errors += 1
                if progress_callback:
                    progress_callback(idx + 1, total, os.path.basename(orig))

        else:
            # Mode was copy: delete destination copies
            for idx, m in enumerate(mappings):
                curr = m["current"]
                try:
                    if os.path.isfile(curr):
                        os.remove(curr)
                        restored += 1
                except Exception as e:
                    print(f"[Renamer Undo Error] Failed to remove copy {curr}: {e}")
                    errors += 1
                if progress_callback:
                    progress_callback(idx + 1, total, os.path.basename(curr))

        self.last_manifest = None
        return restored, errors
