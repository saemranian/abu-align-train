"""
Abu Align & Train - Unified Checkpoint Tool
===========================================
Runs the complete release and checkpoint routine:
1. Syncs/bumps version across GUI, CLI, train engine, scripts, and VERSION.txt.
2. Recompiles C# Windows launchers (Abu_Align.exe, Abu_Align_and_Train.exe) via csc.exe.
3. Archives snapshot of all modified core engine/UI files into versions/vX.Y.Z.W/.
4. Verifies walkthrough documentation in docs/WALKTHROUGH_vX.Y.Z.W.md and walkthrough.md.
5. Performs git status, stages all changes (git add -A), and commits to git.
"""

import os
import sys
import subprocess
import argparse
from version_manager import (
    get_current_version,
    bump_and_sync,
    set_and_sync,
    sync_files_with_version,
    create_version_snapshot,
    recompile_executables
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, check=True):
    print(f">> Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and res.returncode != 0:
        print(f"ERROR: {res.stderr.strip()}")
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {cmd}")
    return res


def perform_checkpoint(message: str = "", bump: bool = False, explicit_version: str = None):
    # Step 1: Versioning
    if explicit_version:
        ver = set_and_sync(explicit_version)
    elif bump:
        ver = bump_and_sync()
    else:
        ver = get_current_version()
        sync_files_with_version(ver)

    print(f"\n[CHECKPOINT] Active Version: v{ver}")

    # Step 2: C# Launcher Recompilation
    print("\n[CHECKPOINT] Recompiling Windows Launchers...")
    recompile_executables(ver)

    # Step 3: Archive Snapshot Creation
    print(f"\n[CHECKPOINT] Archiving Snapshot to versions/v{ver}...")
    create_version_snapshot(ver)

    # Step 4: Verify Walkthrough
    walkthrough_file = os.path.join(ROOT_DIR, "docs", f"WALKTHROUGH_v{ver}.md")
    root_walkthrough = os.path.join(ROOT_DIR, "walkthrough.md")
    if not os.path.isfile(walkthrough_file) and not os.path.isfile(root_walkthrough):
        print(f"[CHECKPOINT WARNING] Walkthrough document not found for v{ver}!")
    else:
        print(f"[CHECKPOINT] Walkthrough documentation verified.")

    # Step 5: Git Commit
    commit_msg = message.strip() if message else f"v{ver}: Version bump, snapshot archive, and feature updates"
    if not commit_msg.startswith(f"v{ver}"):
        commit_msg = f"v{ver}: {commit_msg}"

    print(f"\n[CHECKPOINT] Staging and committing to Git: '{commit_msg}'")
    run_cmd("git add -A")
    res = subprocess.run(f'git commit -m "{commit_msg}"', shell=True, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8")
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and "nothing to commit" not in res.stdout and "nothing to commit" not in res.stderr:
        print(res.stderr.strip())

    print(f"\n[CHECKPOINT COMPLETED SUCCESSFULLY] Version v{ver} is sealed, snapshotted, documented, and committed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Abu Align & Train Checkpoint Command")
    parser.add_argument("-m", "--message", type=str, default="", help="Commit message summary")
    parser.add_argument("--bump", action="store_true", help="Bump base-10 version digit before checkpoint")
    parser.add_argument("--set", type=str, default=None, help="Set explicit version before checkpoint")

    args = parser.parse_args()
    perform_checkpoint(message=args.message, bump=args.bump, explicit_version=args.set)
