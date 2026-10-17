"""
Abu Align & Train - One-Command GitHub Release & Installer Publisher
====================================================================
Usage:
    python push_to_github.py "Release summary" [--details "Detailed release notes"] [--no-bump]

When invoked (e.g. when the user says 'Push to GitHub'), this script:
1. Configures the 'origin' remote to https://github.com/saemranian/abu-align-train.git
2. Runs the mandatory Base-10 version bump, C# launcher compilation, and parity snapshot (unless --no-bump is passed and working tree is clean)
3. Synchronizes installer metadata with the active version
4. Commits all staged changes, creates tag vX.Y.Z.W, and pushes HEAD -> main + tags to GitHub
5. Triggers GitHub Actions to build and attach the single-file installers (.exe, .dmg, .AppImage) to GitHub Releases.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import release_checkpoint
import version_manager

REPO_URL = "https://github.com/saemranian/abu-align-train.git"
ROOT_DIR = Path(__file__).resolve().parent


def run_git(args, check=True):
    return subprocess.run(["git"] + args, cwd=ROOT_DIR, check=check, text=True, capture_output=True)


def ensure_remote():
    res = run_git(["remote"], check=False)
    remotes = res.stdout.split() if res.returncode == 0 else []
    if "origin" in remotes:
        run_git(["remote", "set-url", "origin", REPO_URL])
    else:
        run_git(["remote", "add", "origin", REPO_URL])
    print(f"[GitHub Publisher] Remote 'origin' -> {REPO_URL}")


def sync_installer_version(ver_str: str):
    clean = ver_str.strip().lstrip("vV")
    iss_path = ROOT_DIR / "installer" / "windows" / "abu_align_setup.iss"
    if iss_path.is_file():
        content = iss_path.read_text(encoding="utf-8")
        new_content = re.sub(
            r'#define MyAppVersion "[^"]+"',
            f'#define MyAppVersion "{clean}"',
            content,
        )
        if new_content != content:
            iss_path.write_text(new_content, encoding="utf-8")

    wf_path = ROOT_DIR / ".github" / "workflows" / "release.yml"
    if wf_path.is_file():
        content = wf_path.read_text(encoding="utf-8")
        new_content = re.sub(r"v\d+\.\d+\.\d+\.\d+", f"v{clean}", content)
        if new_content != content:
            wf_path.write_text(new_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Push Abu Align & Train release to GitHub and build installers")
    parser.add_argument("summary", nargs="?", default="Update Abu Align & Train release and cross-platform installers", help="Short summary of release")
    parser.add_argument("--details", default="Automated release publish with synchronized Windows (.exe), macOS (.dmg), and Linux (.AppImage) single-file installers.", help="Detailed release notes")
    parser.add_argument("--no-bump", action="store_true", help="Publish current version without incrementing base-10 odometer")
    args = parser.parse_args()

    ensure_remote()

    if not args.no_bump:
        ver = release_checkpoint.publish_stage(args.summary, args.details)
    else:
        ver = version_manager.get_current_version()

    sync_installer_version(ver)

    # Check if there are changes to commit
    status = run_git(["status", "--porcelain"]).stdout.strip()
    if status:
        run_git(["add", "-A"])
        commit_msg = f"v{ver}: {args.summary}\n\n{args.details}"
        run_git(["commit", "-m", commit_msg])
        print(f"[GitHub Publisher] Committed release v{ver}")
    else:
        print(f"[GitHub Publisher] Working tree already clean at v{ver}")

    tag_name = f"v{ver}"
    run_git(["tag", "-f", "-a", tag_name, "-m", f"Abu Align & Train {tag_name}: {args.summary}"])
    print(f"[GitHub Publisher] Tagged release: {tag_name}")

    print("[GitHub Publisher] Pushing branch 'main' and release tag to GitHub...")
    push_main = subprocess.run(["git", "push", "-u", "origin", "HEAD:main", "--force"], cwd=ROOT_DIR)
    if push_main.returncode != 0:
        sys.exit(push_main.returncode)

    push_tag = subprocess.run(["git", "push", "origin", tag_name, "--force"], cwd=ROOT_DIR)
    if push_tag.returncode != 0:
        sys.exit(push_tag.returncode)

    print("======================================================================")
    print(f"[SUCCESS] Pushed Abu Align & Train {tag_name} to GitHub!")
    print(f"  * Repository: https://github.com/saemranian/abu-align-train")
    print(f"  * Actions CI: https://github.com/saemranian/abu-align-train/actions")
    print(f"  * Releases:   https://github.com/saemranian/abu-align-train/releases/tag/{tag_name}")
    print("======================================================================")


if __name__ == "__main__":
    main()
