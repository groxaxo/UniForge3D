#!/usr/bin/env python3
"""Download the explicitly requested public checkpoint at its pinned revision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from uniforge3d_mlx.softeq import SOFTEQ_REPO, SOFTEQ_REVISION, SOFTEQ_SHA256, BACKBONE_CONFIG
    root = ROOT / "weights/source/dinov3-softeq"
    root.mkdir(parents=True,exist_ok=True)
    required = ("README.md", "config.json", BACKBONE_CONFIG, "configuration_softeq.py",
                "filtered_layers_dinov3.py", "modeling_filtered_dinov3_seg.py",
                "softeq/layers/fconv2d.py", "softeq/equi_utils/inv_projector.py", "model.safetensors")
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{SOFTEQ_REPO}/revision/{SOFTEQ_REVISION}?blobs=true",timeout=30) as response:
        info = json.load(response)
    remote = {f["rfilename"]: f for f in info["siblings"]}
    verified = []
    for name in required:
        path = root / name
        entry = remote[name]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            path.parent.mkdir(parents=True,exist_ok=True)
            partial = path.with_suffix(path.suffix + ".download-partial")
            with urllib.request.urlopen(f"https://huggingface.co/{SOFTEQ_REPO}/resolve/{SOFTEQ_REVISION}/{name}",timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(8*1024*1024):
                    out.write(chunk)
            if partial.stat().st_size != entry["size"]:
                raise RuntimeError(f"Incomplete download: {name}")
            partial.replace(path)
        digest = hashlib.file_digest(path.open("rb"),"sha256").hexdigest()
        expected = (entry.get("lfs") or {}).get("sha256")
        if expected and expected != digest:
            raise ValueError(f"Publisher SHA-256 mismatch: {name}")
        if name == "model.safetensors" and digest != SOFTEQ_SHA256:
            raise ValueError("Pinned checkpoint SHA-256 mismatch")
        verified.append({"path":name,"bytes":path.stat().st_size,"sha256":digest,"publisher_sha256_matched":bool(expected)})
        print("VERIFIED",name,flush=True)
    report = {"repo":SOFTEQ_REPO,"revision":SOFTEQ_REVISION,"gated":False,"files":verified}
    (root / "download-verification.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
