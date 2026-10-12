"""Complete the mandatory root/mirror/snapshot release protocol without recursion."""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
import version_manager


def publish_stage(summary, details):
    root = Path(__file__).resolve().parent
    version = version_manager.format_version(version_manager.increment_base10(
        version_manager.parse_version(version_manager.get_current_version())))
    walkthrough = root / "walkthrough.md"
    previous = walkthrough.read_text(encoding="utf-8") if walkthrough.exists() else ""
    walkthrough.write_text(f"# Abu Align & Train v{version}: {summary}\n\n{details}\n\n---\n\n{previous}", encoding="utf-8")
    # Version manager compiles with csc.exe and creates the canonical snapshot.
    subprocess.run([sys.executable, "-B", str(root / "version_manager.py"), "--bump"], check=True, cwd=root)
    actual = version_manager.get_current_version()
    if actual != version:
        raise RuntimeError("Version changed during release")
    mirror = root / "Abu_Align"
    snapshot = mirror / "versions" / f"v{version}"
    canonical = root / "versions" / f"v{version}"
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    shutil.copy2(walkthrough, artifacts / "walkthrough.md")
    files = [p for p in root.iterdir() if p.is_file() and
             (p.suffix in {".py", ".exe", ".bat", ".cs", ".ico"} or p.name in {"VERSION.txt", "walkthrough.md"})]
    for folder in (root / "docs", root / "tests"):
        files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".md", ".py", ".json", ".txt"})
    for source in files:
        relative = source.relative_to(root)
        for destination in (mirror / relative, snapshot / relative, canonical / relative):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                raise RuntimeError(f"Parity failed: {destination}")
    binary = root / "versions" / "historical_binaries" / f"Abu_Align_and_Train_v{version}.exe"
    if not binary.is_file():
        raise RuntimeError("Versioned C# launcher was not compiled")
    for target in (root, mirror, snapshot, canonical):
        shutil.copy2(binary, target / binary.name)
    print(f"RELEASE v{version}; {len(files)} active files synchronized; snapshot {snapshot}")
    return version
