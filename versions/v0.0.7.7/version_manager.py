"""
Abu Align & Train - Automated Base-10 Version Management System
================================================================
Implements strict base-10 odometer versioning:
Format: vMAJOR.MINOR.PATCH.BUILD (e.g., v0.0.1.3)

Rules:
- Each digit is strictly in the range [0..9].
- When incrementing, if the rightmost digit hits 9, the next increment
  resets it to 0 and carries +1 to the digit to its left (cascading):
    v0.0.0.9 -> v0.0.1.0
    v0.0.1.2 -> v0.0.1.3
    v0.0.1.9 -> v0.0.2.0
    v0.0.9.9 -> v0.1.0.0
    v0.9.9.9 -> v1.0.0.0

Provides automated synchronization across:
- GUI (window title, version badge, about dialog, console message)
- CLI (startup banner, report headers)
- 3DGS Training Engine (banner)
- C# Launcher & compiled executables
- Windows Batch scripts (.bat)
"""

import os
import sys
import re
import subprocess
from typing import Tuple, List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(ROOT_DIR, "VERSION.txt")
VERSIONS_DIR = os.path.join(ROOT_DIR, "versions")


def parse_version(ver_str: str) -> Tuple[int, int, int, int]:
    """Parses 'v0.0.1.3' or '0.0.1.3' into (0, 0, 1, 3)."""
    clean = ver_str.strip().lstrip("vV")
    parts = clean.split(".")
    while len(parts) < 4:
        parts.append("0")
    digits = [int(p) for p in parts[:4]]
    return tuple(digits)  # type: ignore


def format_version(digits: Tuple[int, int, int, int], prefix: bool = False) -> str:
    """Formats (0, 0, 1, 3) as 'v0.0.1.3' or '0.0.1.3'."""
    s = ".".join(str(d) for d in digits)
    return f"v{s}" if prefix else s


def increment_base10(digits: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Cascading base-10 increment (odometer style):
    Each digit is 0-9. If a digit reaches 10, it resets to 0 and carries 1 left.
    """
    d = list(digits)
    d[3] += 1
    for i in range(3, 0, -1):
        if d[i] > 9:
            d[i] = 0
            d[i - 1] += 1
    if d[0] > 9:
        d[0] = 9  # Cap at max major
    return tuple(d)  # type: ignore


def get_current_version() -> str:
    """Reads current version from VERSION.txt or defaults to 0.0.3.4."""
    if os.path.isfile(VERSION_FILE):
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content.lstrip("vV")
    return "0.0.3.4"


def save_current_version(ver_str: str) -> None:
    """Persists version string to VERSION.txt."""
    clean = ver_str.strip().lstrip("vV")
    try:
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            f.write(clean + "\n")
    except Exception as e:
        print(f"[VersionManager Warning] Failed to write to {VERSION_FILE}: {e}")


def create_version_snapshot(version_str: str) -> None:
    """Creates a snapshot folder in versions/vX.Y.Z.W archiving core code."""
    import shutil
    target_dir = os.path.join(VERSIONS_DIR, f"v{version_str}")
    os.makedirs(target_dir, exist_ok=True)
    snapshot_files = [
        "abu_align_gui.py",
        "desktop_ui.py",
        "rig_designer_ui.py",
        "rig_viewport.py",
        "fast_viewer.py",
        "align_cli.py",
        "train_engine.py",
        "opensfm_engine.py",
        "360dvo_engine.py",
        "dvo360_engine.py",
        "acezero_engine.py",
        "pycusfm_engine.py",
        "wsl_bridge.py",
        "cache_manager.py",
        "video_extractor.py",
        "panoramic_splitter.py",
        "equirectangular_transcoder.py",
        "engine_tracker.py",
        "version_manager.py",
        "VERSION.txt",
        "Abu_Align_and_Train.bat",
        "Abu_Align_and_Train.exe",
        "Abu_Align.bat",
        "Abu_Align.exe",
        "launcher.cs",
        "walkthrough.md"
    ]
    for sf in snapshot_files:
        src = os.path.join(ROOT_DIR, sf)
        dst = os.path.join(target_dir, sf)
        if os.path.isfile(src):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass


def sync_files_with_version(new_version: str) -> List[str]:
    """
    Replaces all version references across GUI, CLI, training engine, launchers, and docs.
    """
    v_clean = new_version.lstrip("vV")
    v_prefix = f"v{v_clean}"
    v_dotted = v_clean

    updated_files = []

    # Canonical files to update with regex patterns
    targets = [
        os.path.join(ROOT_DIR, "abu_align_gui.py"),
        os.path.join(ROOT_DIR, "desktop_ui.py"),
        os.path.join(ROOT_DIR, "align_cli.py"),
        os.path.join(ROOT_DIR, "train_engine.py"),
        os.path.join(ROOT_DIR, "launcher.cs"),
        os.path.join(ROOT_DIR, "Abu_Align_and_Train.bat"),
        os.path.join(ROOT_DIR, "Abu_Align.bat"),
    ]

    for file_path in set(targets):
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Pattern 1: vX.Y.Z.W (e.g. v0.0.1.2)
            new_content = re.sub(r"v\d+\.\d+\.\d+\.\d+", v_prefix, content)
            # Pattern 2: Version X.Y.Z.W
            new_content = re.sub(r"Version \d+\.\d+\.\d+\.\d+", f"Version {v_dotted}", new_content)

            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                updated_files.append(file_path)
        except Exception as e:
            print(f"[VersionManager Warning] Failed to update {file_path}: {e}")

    # Recompile executable launchers and save VERSION.txt
    recompile_executables(v_clean)
    save_current_version(v_clean)
    create_version_snapshot(v_clean)
    return updated_files


def recompile_executables(version_str: str) -> None:
    """Recompiles Abu_Align_and_Train.exe and archives Abu_Align_and_Train_vX.Y.Z.W.exe using csc.exe."""
    csc_path = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
    if not os.path.isfile(csc_path):
        print("[VersionManager Warning] csc.exe not found. Skipping C# compilation.")
        return

    launcher_cs = os.path.join(ROOT_DIR, "launcher.cs")
    if not os.path.isfile(launcher_cs):
        return

    bin_dir = os.path.join(VERSIONS_DIR, "historical_binaries")
    os.makedirs(bin_dir, exist_ok=True)

    exe_targets = [
        os.path.join(ROOT_DIR, "Abu_Align_and_Train.exe"),
        os.path.join(ROOT_DIR, "Abu_Align.exe"),
        os.path.join(bin_dir, f"Abu_Align_and_Train_v{version_str}.exe"),
        os.path.join(bin_dir, f"Abu_Align_v{version_str}.exe"),
    ]

    for exe_out in exe_targets:
        try:
            cmd = [csc_path, "/nologo", "/target:winexe", f"/out:{exe_out}", launcher_cs]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[VersionManager Warning] Failed to compile {exe_out}: {e}")


def bump_and_sync() -> str:
    """Computes next base-10 version and synchronizes all files."""
    cur = parse_version(get_current_version())
    next_ver = increment_base10(cur)
    next_str = format_version(next_ver, prefix=False)
    sync_files_with_version(next_str)
    print(f"[VersionManager] Bumped version: v{format_version(cur)} -> v{next_str}")
    return next_str


def set_and_sync(version_str: str) -> str:
    """Explicitly sets version and synchronizes all files."""
    parsed = parse_version(version_str)
    clean = format_version(parsed, prefix=False)
    sync_files_with_version(clean)
    print(f"[VersionManager] Set version to: v{clean}")
    return clean


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Abu Align & Train Base-10 Version Manager")
    parser.add_argument("--current", action="store_true", help="Print current version")
    parser.add_argument("--bump", action="store_true", help="Increment version by 1 with base-10 rollover and sync files")
    parser.add_argument("--set", type=str, default=None, help="Set explicit version (e.g. 0.0.1.3) and sync files")
    parser.add_argument("--snapshot", action="store_true", help="Create snapshot in versions/ for current version")

    args = parser.parse_args()

    if args.bump:
        bump_and_sync()
    elif args.set:
        set_and_sync(args.set)
    elif args.snapshot:
        ver = get_current_version()
        create_version_snapshot(ver)
        print(f"[VersionManager] Snapshot created in versions/v{ver}")
    else:
        print(f"v{get_current_version()}")
