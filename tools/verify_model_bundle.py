#!/usr/bin/env python3
"""Verify the complete bundle's hashes and optionally run its DINO encoder."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_bundle(root):
    from uniforge3d_mlx.onnx_dinov3 import sha256
    from uniforge3d_mlx.source import COMPONENTS
    root = Path(root).resolve()
    report = json.loads((root / "bundle.json").read_text())
    if report["format"] != "uniforge3d-complete-model-bundle-v1" or set(report["components"]) != set(COMPONENTS):
        raise ValueError("Expected the complete fifteen-component model bundle")
    total = 0
    for name, component in report["components"].items():
        directory = (root / component["path"]).resolve()
        if directory.parent != root or directory.name != name:
            raise ValueError(f"Component path escapes the self-contained bundle: {name}")
        if set(p.name for p in directory.iterdir() if p.is_file()) != set(component["files"]):
            raise ValueError(f"Unexpected component file inventory: {name}")
        for filename, expected in component["files"].items():
            if Path(filename).name != filename:
                raise ValueError("Unsafe component file path")
            path = directory / filename
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"Bundle integrity failed: {name}/{filename}")
            total += 1
        print("VERIFIED", name, flush=True)
    return report, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "weights/UniForge3D-oQ5-complete")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report, count = verify_bundle(args.root)
    result = {"components_verified": len(report["components"]), "files_verified": count,
              "all_hashes_match": True, "full_3d_inference_tested": False}
    if args.smoke:
        import mlx.core as mx
        import numpy as np
        from uniforge3d_mlx.dinov3 import DinoV3Encoder
        from tools.validate_onnx_dinov3 import frozen_cases, reference_path, metrics, candidate_gate
        mx.clear_cache()
        mx.set_cache_limit(128 * 1024 * 1024)
        encoder = DinoV3Encoder(args.root / "dinov3")
        held_out = [(info, pixels) for info, pixels in frozen_cases([512]) if info["split"] == "held_out"]
        images = np.stack([pixels[0] for _, pixels in held_out])[None]
        output = encoder(images)
        mx.eval(output)
        expected = np.stack([np.load(reference_path(info))[0] for info, _ in held_out])[None]
        result["multiview_encoder_smoke"] = metrics(expected, np.asarray(output))
        if not candidate_gate(result["multiview_encoder_smoke"]):
            raise ValueError("Complete-bundle DINO feature check failed")
        del encoder, output
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    (args.root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
