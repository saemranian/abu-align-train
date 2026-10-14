"""Ordered, content-addressed inputs shared by alignment workers and benchmarks."""
import hashlib
import json
import platform
from pathlib import Path
from PIL import Image


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(image_paths, output_path, engine, settings=None, runtime_path=None):
    if not image_paths:
        raise ValueError("An alignment manifest must contain images")
    names = set()
    images = []
    for index, value in enumerate(image_paths):
        path = Path(value).absolute()
        if path.name.casefold() in names:
            raise ValueError(f"Duplicate image basename: {path.name}")
        names.add(path.name.casefold())
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        images.append({"id": index, "name": path.name, "path": str(path),
                       "runtime_path": runtime_path(str(path)) if runtime_path else str(path),
                       "width": width, "height": height, "sha256": file_hash(path)})
    payload = {"schema_version": 1, "engine": engine, "settings": settings or {}, "images": images,
               "host": {"system": platform.system(), "python": platform.python_version()}}
    # Runtime paths and host do not change the identity of the selected dataset.
    identity = {"images": [{k: x[k] for k in ("id", "name", "width", "height", "sha256")} for x in images],
                "settings": payload["settings"], "engine": engine}
    payload["fingerprint"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manifest(path, verify=True):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not payload.get("images"):
        raise ValueError("Unsupported or empty run manifest")
    if verify:
        for image in payload["images"]:
            if file_hash(image["runtime_path"]) != image["sha256"]:
                raise ValueError(f"Image changed after selection: {image['name']}")
    return payload
