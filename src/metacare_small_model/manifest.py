"""Build the manifest the app pins a model with, and check it against the phone budget.

The shape is the one metacare-gpt-frontend reads from src/localai/chatManifest.*.json: the app downloads exactly
these files from MetaCare's own model host, checks each size and SHA 256, and refuses anything else. A manifest
built here can be copied into the app as a new chatManifest file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Config

# metacare-gpt-frontend, src/localai/capability.ts: a model at or over this many bytes, counting the ONNX Runtime
# binary, is not offered on an iPhone (MET-1436, MET-1504). 1.0 GB failed on an iPhone, 265 MB installs.
IPHONE_MAX_BYTES = 600_000_000
# Chrome on one test Mac refused to cache a single file between 600 and 800 MB (MET-1503). Treat the low end as the
# limit for any one file until that is measured on more machines.
CACHE_ENTRY_MAX_BYTES = 600_000_000

FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(folder: Path, dtype: str) -> list[Path]:
    wanted = [folder / name for name in FILES] + sorted((folder / "onnx").glob(f"model_{dtype}.onnx*"))
    missing = [str(p) for p in wanted[: len(FILES)] if not p.exists()]
    if missing or not (folder / "onnx" / f"model_{dtype}.onnx").exists():
        raise SystemExit(f"{folder}: missing {missing or [f'onnx/model_{dtype}.onnx']}")
    return wanted


def build(config: Config, folder: Path, revision: str, runtime: dict | None = None) -> dict:
    dtype = config.get("export.dtype")
    files = [
        {"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in model_files(folder, dtype)
    ]
    manifest = {
        "id": f"{config.name}-{dtype}-{revision[:7]}",
        "repo": config.get("manifest.repo"),
        "revision": revision,
        "license": config.get("manifest.license", "Apache 2.0"),
        "dtype": dtype,
        "tasks": ["generate"],
        "webgpuOnly": True,
        "experimental": True,
        "mobile": False,
        "cacheName": config.get("manifest.cache_name"),
        "disclosure": config.get("manifest.disclosure"),
        "files": files,
    }
    if runtime:
        manifest["runtime"] = runtime
    manifest["mobile"] = not budget(manifest)["problems"]
    return manifest


def runtime_from(app_manifest: Path) -> dict:
    """Copy the runtime block (Transformers.js, ONNX Runtime and its binaries) from a manifest the app ships."""
    return json.loads(app_manifest.read_text(encoding="utf-8"))["runtime"]


def budget(manifest: dict) -> dict:
    model = sum(f["bytes"] for f in manifest["files"])
    wasm = ((manifest.get("runtime") or {}).get("wasm") or {}).get("asyncify", {}).get("bytes", 0)
    largest = max(manifest["files"], key=lambda f: f["bytes"])
    problems = []
    if not wasm:
        problems.append("no runtime block, so the ONNX Runtime binary is not counted; pass --runtime-from")
    if model + wasm >= IPHONE_MAX_BYTES:
        problems.append(f"total {model + wasm:,} bytes is at or over the iPhone limit of {IPHONE_MAX_BYTES:,}")
    if largest["bytes"] >= CACHE_ENTRY_MAX_BYTES:
        problems.append(f"{largest['path']} is {largest['bytes']:,} bytes; browsers here refused single files that large")
    return {
        "model_bytes": model,
        "runtime_bytes": wasm,
        "total_bytes": model + wasm,
        "largest_file": largest["path"],
        "problems": problems,
    }
