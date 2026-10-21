#!/usr/bin/env python3
"""Audit every written tensor and execute sampled packed matrix multiplies."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit_component(root):
    import mlx.core as mx
    import numpy as np
    from uniforge3d_mlx.checkpoint import read_manifest
    from uniforge3d_mlx.quantization import array_sha256
    manifest = read_manifest(root)
    by_shard = {}
    for name, entry in manifest["tensors"].items():
        by_shard.setdefault(entry["shard"], []).append((name, entry))
    counts = Counter()
    worst_multiply_error = 0.0
    for shard, entries in by_shard.items():
        arrays = mx.load(str(root / shard))
        expected_keys = set()
        for name, entry in entries:
            if "quantization" in entry:
                q = entry["quantization"]
                qw, scales, biases = [arrays[name + s] for s in (".__qweight", ".__scales", ".__biases")]
                rows, cols = entry["shape"]
                assert tuple(qw.shape) == (rows, cols * q["bits"] // 32), name
                assert tuple(scales.shape) == tuple(biases.shape) == (rows, cols // 64), name
                assert qw.dtype == mx.uint32, name
                assert bool(mx.all(mx.isfinite(scales)).item()) and bool(mx.all(mx.isfinite(biases)).item()), name
                # Exercise native quantized_matmul on first and last rows.
                idx = mx.array([0, rows - 1], dtype=mx.int32)
                wq, ws, wb = qw[idx], scales[idx].astype(mx.float32), biases[idx].astype(mx.float32)
                x = mx.array(np.random.default_rng(42).normal(size=(2, cols)).astype(np.float32))
                actual = mx.quantized_matmul(x, wq, ws, wb, transpose=True, group_size=64, bits=q["bits"])
                dense = mx.dequantize(wq, ws, wb, group_size=64, bits=q["bits"])
                # MLX's ordinary FP32 Metal matmul uses a fast product path;
                # NumPy FP64 is the independent numerical reference here.
                expected = np.asarray(x).astype(np.float64) @ np.asarray(dense).astype(np.float64).T
                error = float(np.max(np.abs(np.asarray(actual) - expected)))
                np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=2e-4)
                worst_multiply_error = max(worst_multiply_error, error)
                counts[str(q["bits"])] += 1
                expected_keys.update(name + s for s in (".__qweight", ".__scales", ".__biases"))
            else:
                value = arrays[name]
                assert tuple(value.shape) == tuple(entry["shape"]), name
                assert array_sha256(value) == entry["passthrough_sha256"], name
                expected_keys.add(name)
                counts["source_precision"] += 1
        assert set(arrays) == expected_keys, f"Unexpected physical tensors in {shard}"
        del arrays
        mx.clear_cache()
    assert dict(counts) == manifest["matrix_counts"]
    return {"status": "passed", "tensor_count": manifest["tensor_count"],
            "matrix_counts": dict(counts), "passthrough_hashes": "all matched",
            "max_sampled_packed_matmul_absolute_error": worst_multiply_error,
            "artifact_bytes": sum(p.stat().st_size for p in root.glob("*.safetensors")),
            "source_logical_bytes": manifest["source_logical_bytes"],
            "merged_lora_pairs": manifest["merged_lora_pairs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "weights/UniForge3D-oQ5")
    args = parser.parse_args()
    result = {"scope": "checkpoint integrity and packed operations; not image-to-3D acceptance", "components": {}}
    for manifest in sorted(args.root.glob("*/manifest.json")):
        print(f"AUDIT {manifest.parent.name}", flush=True)
        result["components"][manifest.parent.name] = audit_component(manifest.parent)
    (args.root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
