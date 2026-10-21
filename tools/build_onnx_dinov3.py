#!/usr/bin/env python3
"""Build measured oQ5/MLP-oQ5/FP16 candidates from the verified public export."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FOLDERS = {"oq5": "UniForge3D-oQ5-ONNX",
           "mlp": "UniForge3D-oQ5-ONNX-MLP",
           "fp16": "UniForge3D-oQ5-ONNX-FP16"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=FOLDERS, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from uniforge3d_mlx.onnx_dinov3 import sha256
    from uniforge3d_mlx.quantization import convert_component, eligible
    from uniforge3d_mlx.source import SafeSource

    source_root = ROOT / "weights/source/dinov3-onnx-hf"
    work = ROOT / "weights/work/onnx-validation"
    output = ROOT / "weights" / FOLDERS[args.policy]
    destination = output / "dinov3"
    preparation = json.loads((source_root / "preparation.json").read_text())
    if sha256(source_root / "model.safetensors") != preparation["normalized_sha256"]:
        raise ValueError("Recovered source changed")
    for filename in ("torch-parity.json", "source-parity.json"):
        parity = json.loads((work / filename).read_text())
        if not parity["all_passed"] or parity["source_sha256"] != preparation["normalized_sha256"]:
            raise ValueError("ONNX/source parity must pass first")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite candidate: {destination}")
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    mx.clear_cache()
    mx.set_cache_limit(128 * 1024 * 1024)

    if args.policy == "fp16":
        state = load_file(str(source_root / "model.safetensors"))
        converted = {
            name: value.half() if eligible("dinov3", name, tuple(value.shape), str(value.dtype).removeprefix("torch.")) else value
            for name, value in state.items()
        }
        if not all(torch.isfinite(value).all() for value in converted.values()):
            raise ValueError("FP16 conversion produced non-finite values")
        staging = destination.with_name("dinov3.partial")
        staging.mkdir(exist_ok=False)
        save_file(converted, str(staging / "model.safetensors"), metadata={"format": "pt"})
        written = load_file(str(staging / "model.safetensors"))
        for name, expected in converted.items():
            if written[name].dtype != expected.dtype or not torch.equal(expected, written[name]):
                raise ValueError(f"FP16 readback mismatch: {name}")
        audit = {
            "policy": "internal-projection-fp16-source-precision-boundaries-v1",
            "tensor_count": len(written), "all_cast_tensors_verified": True,
            "dtype_counts": dict(Counter(str(v.dtype) for v in written.values())),
            "source_sha256": preparation["normalized_sha256"],
            "sha256": sha256(staging / "model.safetensors"),
            "bytes": (staging / "model.safetensors").stat().st_size,
        }
        (staging / "precision-verification.json").write_text(json.dumps(audit, indent=2) + "\n")
        os.replace(staging, destination)
        print(json.dumps(audit, indent=2), flush=True)
    else:
        class VisionSource(SafeSource):
            def tensor(self, name):
                value = super().tensor(name)
                if args.policy == "mlp" and ".attention." in name and name.endswith(".weight"):
                    return value.half()
                return value

            def dtype(self, name):
                if args.policy == "mlp" and ".attention." in name and name.endswith(".weight"):
                    return "float16"
                return super().dtype(name)

        source = VisionSource(source_root / "model.safetensors")
        calibration = json.loads((work / "calibration.json").read_text())
        if calibration["source_sha256"] != preparation["normalized_sha256"] or sha256(work / "importance.npz") != calibration["importance_sha256"]:
            raise ValueError("Calibration/source identity mismatch")
        with np.load(work / "importance.npz") as archive:
            source.importance = {name: archive[name] for name in archive.files}
        source.calibration = calibration
        if args.policy == "mlp":
            source.quantization_allowlist = {n for n in source.importance if ".mlp." in n}
            source.importance = {n: v for n, v in source.importance.items()
                                 if n in source.quantization_allowlist}
            source.quantization_policy = "vision-mlp-oq5-fp16-attention-fp32-boundaries-v1"
        else:
            source.quantization_policy = "vision-attention-mlp-oq5-fp32-boundaries-v1"
        source.calibration["quantized_matrix_count"] = len(source.importance)
        source.calibration["attention_policy"] = "fp16" if args.policy == "mlp" else "oq5"
        convert_component(source, "dinov3", output, omlx_root=Path("/Users/op/AI/omlx"),
                          source_paths=[source_root / "model.safetensors"])
    for name in ("config.json", "preprocessor_config.json", "preparation.json"):
        shutil.copy2(source_root / name, destination / name)
    shutil.copy2(source_root / "README.md", destination / "UPSTREAM_README.md")
    print("BUILT", destination, flush=True)


if __name__ == "__main__":
    main()
