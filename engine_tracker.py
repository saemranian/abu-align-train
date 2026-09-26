"""
Abu Align & Train - Engine Repository Version Tracker & Update Monitor
========================================================================
Tracks, compares, and monitors upstream GitHub repositories for all photogrammetry,
Structure-from-Motion, and 3D Gaussian Splatting engines integrated into Abu Align:
- VGGT-Omega (Facebook Research)
- Niantic Labs ACE0 (Scene Coordinate Regression)
- NVIDIA 3DGRUT (Ray Traced Gaussian Splatting)
- PFGS360 (CVPR 2026 Pose-Free Gaussian Splatting for 360)
- NVIDIA Isaac pyCuSFM (GPU Structure-from-Motion)
- Mapillary OpenSfM (360° Spherical Photogrammetry)
- LichtFeld Studio (Real-time Gaussian Splatting)
- Nerfstudio (Splatfacto Academic Pipeline)
- VGGT Original (Facebook Research)

Provides:
- Asynchronous background update checking on application launch.
- Non-blocking remote GitHub query via GitHub API & git ls-remote fallback.
- Local repository commit identification & diff comparison.
- Structured administrative reports and interactive GUI update dialog.
"""

import os
import sys
import json
import socket
import urllib.request
import urllib.error
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent

# Complete registry of all alignment and 3DGS engines integrated into Abu Align & Train
ENGINE_REGISTRY: List[Dict[str, Any]] = [
    {
        "id": "vggt_omega",
        "name": "VGGT-Omega",
        "category": "Alignment Engine",
        "role": "Neural Foundation Transformer (Low VRAM, Arbitrary Aspect)",
        "repo_slug": "facebookresearch/vggt-omega",
        "repo_url": "https://github.com/facebookresearch/vggt-omega.git",
        "local_candidates": [
            PARENT_DIR / "vggt-omega",
            ROOT_DIR / "vggt-omega",
        ],
        "default_branch": "main",
        "doc_url": "https://github.com/facebookresearch/vggt-omega",
    },
    {
        "id": "vggttt",
        "name": "NVIDIA VGG-T3 (VGG-TTT)",
        "category": "Alignment Engine",
        "role": "Linear-Scaling Test-Time Training Transformer (CVPR 2026)",
        "repo_slug": "nv-dvl/vgg-ttt",
        "repo_url": "https://github.com/nv-dvl/vgg-ttt.git",
        "local_candidates": [
            PARENT_DIR / "vgg-ttt",
            ROOT_DIR / "vgg-ttt",
        ],
        "default_branch": "main",
        "doc_url": "https://research.nvidia.com/labs/dvl/projects/vgg-ttt/",
    },
    {
        "id": "ace0",
        "name": "Niantic Labs ACE0",
        "category": "Alignment Engine",
        "role": "Scene Coordinate Regression & 25k Refit SfM",
        "repo_slug": "nianticlabs/acezero",
        "repo_url": "https://github.com/nianticlabs/acezero.git",
        "local_candidates": [
            PARENT_DIR / "acezero",
            ROOT_DIR / "acezero",
        ],
        "default_branch": "main",
        "doc_url": "https://github.com/nianticlabs/acezero",
    },
    {
        "id": "3dgrut",
        "name": "NVIDIA 3DGRUT",
        "category": "3DGS Training Engine",
        "role": "Differentiable Ray Tracing & Accelerated Gaussian Splatting",
        "repo_slug": "nv-tlabs/3dgrut",
        "repo_url": "https://github.com/nv-tlabs/3dgrut.git",
        "local_candidates": [
            PARENT_DIR / "3dgrut",
            ROOT_DIR / "3dgrut",
        ],
        "default_branch": "main",
        "doc_url": "https://github.com/nv-tlabs/3dgrut",
    },
    {
        "id": "pfgs360",
        "name": "PFGS360",
        "category": "3DGS Training Engine",
        "role": "CVPR 2026 Pose-Free Gaussian Splatting for 360° Videos",
        "repo_slug": "zcq15/PFGS360",
        "repo_url": "https://github.com/zcq15/PFGS360.git",
        "local_candidates": [
            ROOT_DIR / "PFGS360",
            PARENT_DIR / "PFGS360",
        ],
        "default_branch": "main",
        "doc_url": "https://github.com/zcq15/PFGS360",
    },
    {
        "id": "pycusfm",
        "name": "NVIDIA Isaac pyCuSFM",
        "category": "Alignment Engine",
        "role": "GPU Structure-from-Motion (ALIKED, LightGlue, cuDSS BA)",
        "repo_slug": "nvidia-isaac/pyCuSFM",
        "repo_url": "https://github.com/nvidia-isaac/pyCuSFM.git",
        "local_candidates": [
            PARENT_DIR / "pyCuSFM",
            ROOT_DIR / "pyCuSFM",
        ],
        "pinned_version": "v0.1.3 (0c97a67)",
        "default_branch": "main",
        "doc_url": "https://github.com/nvidia-isaac/pyCuSFM",
    },
    {
        "id": "opensfm",
        "name": "Mapillary OpenSfM",
        "category": "Alignment Engine",
        "role": "Native 360° Spherical & Street-Scale S^2 Bundle Adjustment",
        "repo_slug": "mapillary/OpenSfM",
        "repo_url": "https://github.com/mapillary/OpenSfM.git",
        "local_candidates": [
            PARENT_DIR / "OpenSfM",
            ROOT_DIR / "OpenSfM",
        ],
        "pinned_version": "v0.5.2 / Native S^2 Solver",
        "default_branch": "main",
        "doc_url": "https://github.com/mapillary/OpenSfM",
    },
    {
        "id": "360dvo",
        "name": "360DVO",
        "category": "Alignment Engine",
        "role": "Deep Omnidirectional Differentiable Visual Odometry (IEEE RA-L 2026)",
        "repo_slug": None,
        "repo_url": None,
        "local_candidates": [
            ROOT_DIR / "360dvo_engine.py",
            PARENT_DIR / "360dvo",
        ],
        "pinned_version": "Native DAS-Feat & ODBA",
        "default_branch": "main",
        "doc_url": "https://ieeexplore.ieee.org/document/10375881",
    },
    {
        "id": "lichtfeld",
        "name": "LichtFeld Studio",
        "category": "3DGS Training Engine",
        "role": "Real-time Gaussian Splatting Visualizer & CUDA Trainer",
        "repo_slug": "MrNeRF/LichtFeld-Studio",
        "repo_url": "https://github.com/MrNeRF/LichtFeld-Studio.git",
        "local_candidates": [
            PARENT_DIR / "LichtFeld-Studio",
            ROOT_DIR / "LichtFeld-Studio",
        ],
        "pinned_version": "Integrated LichtFeld Studio Executable",
        "default_branch": "main",
        "doc_url": "https://github.com/MrNeRF/LichtFeld-Studio",
    },
    {
        "id": "nerfstudio",
        "name": "Nerfstudio",
        "category": "3DGS Training Engine",
        "role": "Academic Splatfacto / Nerfacto Optimization Pipeline",
        "repo_slug": "nerfstudio-project/nerfstudio",
        "repo_url": "https://github.com/nerfstudio-project/nerfstudio.git",
        "local_candidates": [
            PARENT_DIR / "nerfstudio",
            ROOT_DIR / "nerfstudio",
        ],
        "pinned_version": "Conda Environment (splatfacto / nerfacto)",
        "default_branch": "main",
        "doc_url": "https://github.com/nerfstudio-project/nerfstudio",
    },
    {
        "id": "vggt_base",
        "name": "VGGT (Original)",
        "category": "Foundation Model",
        "role": "Original Vision Geometry Transformer Architecture",
        "repo_slug": "facebookresearch/vggt",
        "repo_url": "https://github.com/facebookresearch/vggt.git",
        "local_candidates": [
            PARENT_DIR / "vggt",
            ROOT_DIR / "vggt",
        ],
        "default_branch": "main",
        "doc_url": "https://github.com/facebookresearch/vggt",
    },
]


def check_internet_connectivity(host: str = "github.com", port: int = 443, timeout: float = 2.0) -> bool:
    """Verifies active internet connectivity to GitHub."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
        return True
    except Exception:
        return False


def get_local_git_info(candidate_paths: List[Path]) -> Dict[str, Any]:
    """Inspects local filesystem for git repository HEAD information."""
    for p in candidate_paths:
        if p.is_dir() and (p / ".git").exists():
            try:
                # Query full sha
                sha = subprocess.check_output(
                    ["git", "-C", str(p), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL, text=True, timeout=5
                ).strip()

                # Query short hash, commit date, and subject line
                log_out = subprocess.check_output(
                    ["git", "-C", str(p), "log", "-1", "--format=%h|%cd|%s", "--date=short"],
                    stderr=subprocess.DEVNULL, text=True, timeout=5
                ).strip()

                short_sha, commit_date, subject = log_out.split("|", 2) if "|" in log_out else (sha[:7], "Unknown", "")
                
                # Check for uncommitted working tree changes
                status_out = subprocess.check_output(
                    ["git", "-C", str(p), "status", "--porcelain"],
                    stderr=subprocess.DEVNULL, text=True, timeout=5
                ).strip()
                has_dirty = len(status_out) > 0

                return {
                    "found": True,
                    "is_git": True,
                    "path": str(p.resolve()),
                    "sha": sha,
                    "short_sha": short_sha,
                    "date": commit_date,
                    "message": subject,
                    "dirty": has_dirty,
                }
            except Exception:
                pass

        elif p.is_dir():
            return {
                "found": True,
                "is_git": False,
                "path": str(p.resolve()),
                "sha": None,
                "short_sha": "Local Folder",
                "date": "Installed",
                "message": "Local files present (Non-git directory)",
                "dirty": False,
            }

    return {
        "found": False,
        "is_git": False,
        "path": None,
        "sha": None,
        "short_sha": "Not Found",
        "date": "N/A",
        "message": "Not cloned locally",
        "dirty": False,
    }


def get_remote_git_info(repo_slug: str, repo_url: str, timeout: float = 4.0) -> Optional[Dict[str, Any]]:
    """Queries latest remote GitHub commit via GitHub API, falling back to git ls-remote."""
    # 1. Primary: Fast GitHub API
    api_url = f"https://api.github.com/repos/{repo_slug}/commits/HEAD"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "AbuAlign-EngineTracker"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            sha = data.get("sha", "")
            commit_info = data.get("commit", {})
            committer = commit_info.get("committer", {})
            commit_date = committer.get("date", "")[:10]
            message = commit_info.get("message", "").splitlines()[0] if commit_info.get("message") else ""
            return {
                "sha": sha,
                "short_sha": sha[:7],
                "date": commit_date,
                "message": message,
                "url": f"https://github.com/{repo_slug}/commit/{sha}",
                "method": "api",
            }
    except Exception:
        pass

    # 2. Fallback: git ls-remote (unrestricted by API rate limits)
    try:
        res = subprocess.check_output(
            ["git", "ls-remote", repo_url, "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=timeout
        ).strip()
        if res:
            sha = res.split()[0]
            return {
                "sha": sha,
                "short_sha": sha[:7],
                "date": "Latest",
                "message": "Latest commit on GitHub HEAD",
                "url": f"https://github.com/{repo_slug}/commit/{sha}",
                "method": "ls-remote",
            }
    except Exception:
        pass

    return None


def evaluate_engine(engine_def: Dict[str, Any], has_internet: bool, timeout: float = 4.0) -> Dict[str, Any]:
    """Evaluates a single engine definition for local vs remote state."""
    res = {
        "id": engine_def["id"],
        "name": engine_def["name"],
        "category": engine_def["category"],
        "role": engine_def["role"],
        "repo_slug": engine_def["repo_slug"],
        "repo_url": engine_def["repo_url"],
        "doc_url": engine_def.get("doc_url", ""),
        "pinned_version": engine_def.get("pinned_version", None),
    }

    if not engine_def.get("repo_slug"):
        res["status"] = "integrated"
        res["status_label"] = "[OK] Native Engine"
        return res

    local_info = get_local_git_info(engine_def.get("local_candidates", []))
    res["local"] = local_info

    remote_info = None
    if has_internet:
        remote_info = get_remote_git_info(engine_def["repo_slug"], engine_def["repo_url"], timeout=timeout)
    res["remote"] = remote_info

    # Status Determination
    # Options: 'up_to_date', 'update_available', 'integrated', 'custom_ahead', 'offline', 'not_installed'
    if not has_internet:
        if local_info["found"]:
            res["status"] = "offline"
            res["status_label"] = "[OFFLINE] Local Present"
        elif res.get("pinned_version"):
            res["status"] = "integrated"
            res["status_label"] = "[OK] Integrated"
        else:
            res["status"] = "not_installed"
            res["status_label"] = "[OFFLINE] Not Installed"
    elif remote_info is None:
        if local_info["found"]:
            res["status"] = "offline"
            res["status_label"] = "[OFFLINE] Remote Unreachable"
        elif res.get("pinned_version"):
            res["status"] = "integrated"
            res["status_label"] = "[OK] Integrated"
        else:
            res["status"] = "not_installed"
            res["status_label"] = "[OFFLINE] Remote Unreachable"
    else:
        # Both local and remote are known
        if local_info["is_git"] and local_info["sha"]:
            loc_sha = local_info["sha"].lower()
            rem_sha = remote_info["sha"].lower()

            if loc_sha == rem_sha:
                res["status"] = "up_to_date"
                res["status_label"] = "[OK] Up to Date"
            else:
                # Check if local repo knows the remote commit (i.e. is remote ahead or behind?)
                is_behind = False
                try:
                    chk = subprocess.run(
                        ["git", "-C", local_info["path"], "merge-base", "--is-ancestor", loc_sha, rem_sha],
                        capture_output=True, timeout=3
                    )
                    if chk.returncode == 0:
                        is_behind = True
                except Exception:
                    is_behind = True

                if is_behind:
                    res["status"] = "update_available"
                    res["status_label"] = "[UPDATE] Available"
                else:
                    res["status"] = "custom_ahead"
                    res["status_label"] = "[OK] Custom Commits"
        elif local_info["found"]:
            res["status"] = "integrated"
            res["status_label"] = "[OK] Integrated"
        elif res["pinned_version"]:
            res["status"] = "integrated"
            res["status_label"] = f"[OK] {res['pinned_version'][:15]}"
        else:
            res["status"] = "not_installed"
            res["status_label"] = "Available on GitHub"

    return res


def check_all_engines(timeout: float = 4.0, max_workers: int = 5) -> Dict[str, Any]:
    """
    Checks all registered engines against their upstream GitHub repositories.
    Returns structured results containing status, commit hashes, update counts, and timestamp.
    """
    has_internet = check_internet_connectivity()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(evaluate_engine, eng, has_internet, timeout): eng["id"]
            for eng in ENGINE_REGISTRY
        }
        for future in as_completed(future_map):
            try:
                results.append(future.result())
            except Exception:
                pass

    # Sort results in predictable original order
    order_map = {eng["id"]: idx for idx, eng in enumerate(ENGINE_REGISTRY)}
    results.sort(key=lambda r: order_map.get(r["id"], 99))

    updates_count = sum(1 for r in results if r["status"] == "update_available")

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "has_internet": has_internet,
        "update_count": updates_count,
        "total_engines": len(results),
        "engines": results,
    }


def generate_report_text(data: Dict[str, Any]) -> str:
    """Generates an ASCII diagnostic report of all engines and their status."""
    lines = [
        "=" * 82,
        " ABU ALIGN & TRAIN - ENGINE REPOSITORY VERSION REPORT",
        "=" * 82,
        f" Check Time:          {data.get('timestamp', 'N/A')}",
        f" Internet Status:     {'CONNECTED' if data.get('has_internet') else 'OFFLINE / UNREACHABLE'}",
        f" Updates Available:   {data.get('update_count', 0)} of {data.get('total_engines', len(ENGINE_REGISTRY))} engines",
        "-" * 82,
        f"{'ENGINE':<18} | {'CATEGORY':<20} | {'APPLIED VERSION':<16} | {'STATUS':<20}",
        "-" * 82,
    ]

    for eng in data.get("engines", []):
        loc = eng.get("local", {})
        if loc.get("is_git") and loc.get("short_sha"):
            loc_str = loc.get("short_sha")
        elif eng.get("pinned_version"):
            loc_str = eng.get("pinned_version")
        elif loc.get("found"):
            loc_str = "Local Folder"
        else:
            loc_str = "Not Installed"
        if len(loc_str) > 15:
            loc_str = loc_str[:13] + ".."
        status_str = eng.get("status_label", "Unknown")
        lines.append(f"{eng['name']:<18} | {eng['category']:<20} | {loc_str:<16} | {status_str:<20}")

    lines.append("-" * 82)

    # Detailed section for engines with updates available
    updates = [e for e in data.get("engines", []) if e.get("status") == "update_available"]
    if updates:
        lines.append("\n" + "=" * 82)
        lines.append(" [!] AVAILABLE ENGINE UPDATES ON GITHUB:")
        lines.append("=" * 82)
        for u in updates:
            loc = u.get("local", {})
            rem = u.get("remote", {})
            lines.extend([
                f"\n>>> {u['name']} ({u['category']})",
                f"    * Role:            {u['role']}",
                f"    * Repository:      {u['repo_url']}",
                f"    * Current Local:   {loc.get('short_sha', 'None')} ({loc.get('date', '')}) - {loc.get('message', '')}",
                f"    * Latest Remote:   {rem.get('short_sha', 'HEAD')} ({rem.get('date', '')}) - {rem.get('message', '')}",
                f"    * Commit URL:      {rem.get('url', '')}",
                f"    * Action Required: Request agent or run git pull to update this engine.",
            ])
    else:
        lines.append("\n[OK] All installed engines are currently synchronized with the latest GitHub versions.")

    lines.append("=" * 82 + "\n")
    return "\n".join(lines)


def print_terminal_report():
    """CLI helper to run the check and print the report directly."""
    print("\nChecking upstream GitHub repositories for Abu Align engines...\n")
    res = check_all_engines()
    try:
        print(generate_report_text(res))
    except UnicodeEncodeError:
        print(generate_report_text(res).encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    print_terminal_report()
