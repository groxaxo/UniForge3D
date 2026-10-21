#!/usr/bin/env python3
"""Download the exact inference dependencies, with resumable Hub transfers."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ("uniforge3d", "xibushu/UniForge3D", "a084333f2aae0213e85b9a4541752d1221d618db",
     ["pipeline.json", "README.md", "uniforge3d/**/*.pt"]),
    ("trellis2", "microsoft/TRELLIS.2-4B", "af44b45f2e35a493886929c6d786e563ec68364d",
     ["README.md", "LICENSE*", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
      "ckpts/shape_dec_next_dc_f16c32_fp16.*", "ckpts/tex_dec_next_dc_f16c32_fp16.*"]),
    ("trellis1", "microsoft/TRELLIS-image-large", "25e0d31ffbebe4b5a97464dd851910efc3002d96",
     ["README.md", "LICENSE*", "ckpts/ss_dec_conv3d_16l8_fp16.*"]),
    ("rmbg", "briaai/RMBG-2.0", "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
     ["*.json", "*.py", "README.md", "LICENSE*", "model.safetensors"]),
    ("dinov3", "facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51",
     ["*.json", "README.md", "LICENSE*", "model.safetensors"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "weights" / "source")
    parser.add_argument("--only", choices=[s[0] for s in SPECS])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / "download-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    def download(spec):
        name, repo, revision, patterns = spec
        destination = args.root / name
        print(f"START {name}: {repo}@{revision}", flush=True)
        started = time.time()
        try:
            snapshot_download(repo_id=repo, revision=revision, local_dir=destination,
                              allow_patterns=patterns, max_workers=2)
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            files = []
            for remote in info.siblings:
                path = destination / remote.rfilename
                if not path.is_file():
                    continue
                actual = path.stat().st_size
                if actual != remote.size:
                    raise RuntimeError(f"Size mismatch: {remote.rfilename}: {actual} != {remote.size}")
                lfs = remote.lfs
                files.append({"path": remote.rfilename, "bytes": actual,
                              "expected_sha256": getattr(lfs, "sha256", None)})
            entry = {"status": "complete", "repo": repo, "revision": revision,
                     "path": str(destination), "files": files,
                     "bytes": sum(f["bytes"] for f in files), "seconds": time.time() - started}
            print(f"COMPLETE {name}: {entry['bytes'] / 1e9:.3f} GB", flush=True)
        except Exception as error:
            entry = {"status": "blocked" if "GatedRepo" in type(error).__name__ else "error",
                     "repo": repo, "revision": revision, "error_type": type(error).__name__,
                     "error": str(error).splitlines()[0]}
            print(f"FAILED {name}: {entry['error_type']}: {entry['error']}", flush=True)
        return name, entry

    specs = [s for s in SPECS if args.only is None or s[0] == args.only]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for name, entry in pool.map(download, specs):
            status[name] = entry
            temp = status_path.with_suffix(".tmp")
            temp.write_text(json.dumps(status, indent=2) + "\n")
            os.replace(temp, status_path)
    return int(any(s["status"] != "complete" for s in status.values()))


if __name__ == "__main__":
    raise SystemExit(main())
