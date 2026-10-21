#!/usr/bin/env python3
"""Build a separately identified oQ5 DINO replacement using real activation stats."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-policy", choices=("quantize", "preserve"), default="quantize")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/UniForge3D-oQ5-SoftEq")
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    import torch
    from uniforge3d_mlx.source import SafeSource
    from uniforge3d_mlx.quantization import convert_component
    source_root = ROOT / "weights/source/dinov3-softeq-hf"
    calibration_root = ROOT / "weights/work/softeq-validation"
    source_parity = json.loads((calibration_root / "source-parity.json").read_text())
    if any(t["cosine"] < .999 or t["minimum_token_cosine"] < .995 for t in source_parity["tests"]):
        raise RuntimeError("Source-precision parity must pass before conversion")
    prep = json.loads((source_root / "preparation.json").read_text())
    if hashlib.file_digest((source_root / "model.safetensors").open("rb"),"sha256").hexdigest() != prep["normalized_sha256"]:
        raise RuntimeError("Normalized source changed after preparation")
    source = SafeSource(source_root / "model.safetensors")
    archive = np.load(calibration_root / "importance.npz")
    source.importance = {k: archive[k] for k in archive.files}
    source.calibration = json.loads((calibration_root / "calibration.json").read_text())
    if args.attention_policy == "preserve":
        source.quantization_allowlist = {k for k in source.importance if ".mlp." in k}
        source.importance = {k: v for k, v in source.importance.items() if k in source.quantization_allowlist}
        source.quantization_policy = "vision-mlp-only-source-precision-attention-v1"
    source.calibration["quantized_matrix_count"] = len(source.importance)
    source.calibration["attention_policy"] = args.attention_policy
    output = args.output
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    mx.set_cache_limit(256*1024*1024)
    manifest = convert_component(source, "dinov3", output, omlx_root=Path("/Users/op/AI/omlx"),
                                 source_paths=[source_root / "model.safetensors"])
    manifest["replacement"] = prep
    (output / "dinov3/manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for name in ("config.json", "preparation.json"):
        shutil.copy2(source_root / name, output / "dinov3" / name)
    shutil.copy2(ROOT / "weights/source/dinov3-softeq/README.md",output / "dinov3/UPSTREAM_README.md")
    print("Replacement built:",output / "dinov3",flush=True)


if __name__ == "__main__":
    main()
