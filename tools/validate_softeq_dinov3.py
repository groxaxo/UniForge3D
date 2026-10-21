#!/usr/bin/env python3
"""Compare the replacement's source and packed MLX encoders at real resolutions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def metrics(expected, actual):
    import numpy as np
    expected, actual = expected.astype(np.float64), actual.astype(np.float64)
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("DINO feature shape mismatch or non-finite output")
    cos = (expected * actual).sum(-1) / np.maximum(np.linalg.norm(expected, axis=-1) * np.linalg.norm(actual, axis=-1), 1e-20)
    flat_cos = float(np.dot(expected.ravel(), actual.ravel()) / (np.linalg.norm(expected) * np.linalg.norm(actual)))
    return {"shape": list(actual.shape), "cosine": flat_cos, "minimum_token_cosine": float(cos.min()),
            "mean_token_cosine": float(cos.mean()), "mean_absolute_error": float(np.abs(actual-expected).mean()),
            "max_absolute_error": float(np.abs(actual-expected).max()),
            "relative_rmse": float(np.sqrt(np.square(actual-expected).sum() / np.square(expected).sum()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("source", "quantized", "fp16"), required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "weights/source/dinov3-softeq-hf")
    parser.add_argument("--packed", type=Path, default=ROOT / "weights/UniForge3D-oQ5-SoftEq/dinov3")
    parser.add_argument("--image", type=Path, default=ROOT / "assets/case01/001.png")
    parser.add_argument("--output", type=Path, default=ROOT / "weights/work/softeq-validation")
    parser.add_argument("--reference", type=Path, help="Frozen source reference directory when evaluating another candidate")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from uniforge3d_mlx.dinov3 import DinoV3Encoder
    from uniforge3d_mlx.quantization import eligible
    from uniforge3d_mlx.softeq import torch_backbone, torch_features
    torch.set_num_threads(4)
    mx.set_cache_limit(256 * 1024 * 1024)
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")
    inputs = {size: np.asarray(image.resize((size,size),Image.Resampling.LANCZOS),dtype=np.float32).transpose(2,0,1)[None]/255 for size in args.sizes}
    image_sha = hashlib.file_digest(args.image.open("rb"), "sha256").hexdigest()
    reference_root = args.reference or args.output
    calibration_path = reference_root / "calibration.json"
    reference_report = {"image": str(args.image), "image_sha256": image_sha, "sizes": args.sizes,
                        "normalization": "ImageNet RGB mean/std, UniForge3D final non-affine layer_norm", "tests": []}
    if args.phase == "source":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; run with access to the Mac GPU")
        config = json.loads((args.source / "config.json").read_text())
        model = torch_backbone(config, load_file(str(args.source / "model.safetensors"))).to("mps")
        collected, handles = {}, []

        def hook_for(name):
            def capture(module, values):
                x = values[0].detach().float().reshape(-1, values[0].shape[-1])
                moment = x.square().sum(0)
                previous, count = collected.get(name, (0, 0))
                collected[name] = (previous + moment, count + x.shape[0])
            return capture

        for name, module in model.named_modules():
            canonical = name.removeprefix("model.") + ".weight"
            if isinstance(module, torch.nn.Linear) and eligible("dinov3", canonical, tuple(module.weight.shape), "float32"):
                handles.append(module.register_forward_pre_hook(hook_for(canonical)))
        with torch.inference_mode():
            for size, arr in inputs.items():
                print(f"TORCH SOURCE {size}px", flush=True)
                pixels = torch.from_numpy(arr.copy()).to("mps")
                pixels = (pixels - torch.tensor([.485,.456,.406],device="mps").reshape(1,3,1,1)) / torch.tensor([.229,.224,.225],device="mps").reshape(1,3,1,1)
                started = time.perf_counter()
                features = torch_features(model, pixels).cpu().numpy()
                np.save(args.output / f"reference-{size}.npy", features)
                reference_report["tests"].append({"size": size, "shape": list(features.shape), "seconds": time.perf_counter()-started})
        for handle in handles:
            handle.remove()
        importance = {k: (v / n).cpu().numpy() for k, (v,n) in collected.items()}
        np.savez(args.output / "importance.npz", **importance)
        calibration = {"image": str(args.image), "image_sha256": image_sha, "sizes": args.sizes,
                       "method": "input activation mean square, diagonal only, original FP32 replacement backbone",
                       "matrix_count": len(importance), "token_counts": {k: n for k,(_,n) in collected.items()}}
        calibration_path.write_text(json.dumps(calibration,indent=2)+"\n")
        (args.output / "torch-reference.json").write_text(json.dumps(reference_report,indent=2)+"\n")
        del model, collected, importance, handles, pixels
        gc.collect()
        torch.mps.empty_cache()
    else:
        calibration = json.loads(calibration_path.read_text())
        if calibration["image_sha256"] != image_sha or calibration["sizes"] != args.sizes:
            raise ValueError("Validation inputs differ from the frozen source reference")
    encoder = DinoV3Encoder(args.source if args.phase == "source" else args.packed,
                             config_path=args.source / "config.json")
    result = {"phase": args.phase, "image_sha256": image_sha,
              "original_meta_checkpoint_equivalence": "not claimed", "tests": []}
    for size, arr in inputs.items():
        print(f"MLX {args.phase.upper()} {size}px", flush=True)
        started = time.perf_counter()
        output = encoder(arr)
        mx.eval(output)
        comparison = metrics(np.load(reference_root / f"reference-{size}.npy"), np.asarray(output))
        comparison.update({"size": size, "seconds": time.perf_counter()-started})
        result["tests"].append(comparison)
        print(json.dumps(comparison),flush=True)
        if args.phase == "source" and (comparison["cosine"] < 0.999 or comparison["minimum_token_cosine"] < 0.995):
            raise RuntimeError("Source-precision MLX feature parity failed")
    result["scope"] = "full DINO feature encoder on one frozen image at both project resolutions; not full 3D quality"
    (args.output / f"{args.phase}-parity.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
