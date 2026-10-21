#!/usr/bin/env python3
"""Protect the vision encoder at FP16 when low-bit feature drift is material."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from safetensors.torch import load_file, save_file
    source = ROOT / "weights/source/dinov3-softeq-hf"
    destination = ROOT / "weights/UniForge3D-oQ5-SoftEq-FP16/dinov3"
    if destination.exists():
        raise FileExistsError(destination)
    preparation = json.loads((source / "preparation.json").read_text())
    if hashlib.file_digest((source / "model.safetensors").open("rb"),"sha256").hexdigest() != preparation["normalized_sha256"]:
        raise ValueError("Prepared source checkpoint changed")
    original = load_file(str(source / "model.safetensors"))
    converted = {k: v.half() if v.is_floating_point() else v for k,v in original.items()}
    if not all(not v.is_floating_point() or torch.isfinite(v).all() for v in converted.values()):
        raise ValueError("Non-finite FP16 conversion")
    destination.mkdir(parents=True)
    save_file(converted, str(destination / "model.safetensors"))
    written = load_file(str(destination / "model.safetensors"))
    for name in converted:
        if written[name].dtype != converted[name].dtype or not torch.equal(written[name], converted[name]):
            raise ValueError(f"FP16 tensor audit failed: {name}")
    config = json.loads((source / "config.json").read_text())
    config["dtype"] = "float16"
    config["torch_dtype"] = "float16"
    (destination / "config.json").write_text(json.dumps(config,indent=2)+"\n")
    report = {"source":preparation,"precision":"float16","tensor_count":len(written),
              "all_cast_tensors_verified":True,"sha256":hashlib.file_digest((destination / "model.safetensors").open("rb"),"sha256").hexdigest(),
              "bytes":(destination / "model.safetensors").stat().st_size}
    (destination / "precision-verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
